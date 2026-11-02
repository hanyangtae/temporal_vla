"""OpenDrawer QAM baseline: 추출 산출물 정렬·terminal/partial·resume 거부·validation-only 선택·CLI."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import ADAPTER_DIR, REPO_ROOT
from baseline_data import (
    AUX_SPLIT_V2_SHA256,
    EXPECTED_FEATURE_SIGNATURE,
    audit_rewards,
    build_episode_arrays,
    commit_episode,
    contract_sha256,
    decision_seed,
    ensure_same_or_create,
    extraction_contract,
    load_extraction,
    sha256_path,
    verify_committed_episode,
)
from composition import SPEC
from contract import ACTION_KEYS, ActionCodec, ContractError, raw_to_model_index
from transitions import EpisodeInput, build_episode_transitions, expert_chunk_targets

SPLIT_V2 = REPO_ROOT / "outputs/analysis/groot_robocasa_lineage_20261021/OpenDrawer_aux_split_v2.json"
RAW_INDEX = raw_to_model_index()


def _codec():
    modes = tuple(m for _, d, m in ACTION_KEYS for _ in range(d))
    lo = np.full(12, -2.0)
    hi = np.full(12, 2.0)
    hi[10] = lo[10]  # degenerate 차원 (실제 metadata idx 10 과 같은 성질)
    return ActionCodec(lo, hi, modes)


def _episode(T, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.5, 1.5, size=(T, 12))
    r = np.zeros(T, np.float32)
    r[-16:] = 1.0
    d = np.zeros(T, bool)
    d[-1] = True
    return raw, r, d


def _arrays(eid, T, codec, *, base_seed=42, seed=0):
    raw, r, d = _episode(T, seed)
    n = len(range(0, T, 5))
    lat = np.random.default_rng(seed + 1).normal(size=(n, 1024)).astype(np.float32)
    seeds = [decision_seed(base_seed, eid, t) for t in range(0, T, 5)]
    return build_episode_arrays(episode_id=eid, latents=lat, decoder_calls=[4] * n, seeds=seeds,
                                raw_actions=raw, rewards=r, done=d, frame_index=np.arange(T),
                                codec=codec, raw_index=RAW_INDEX, base_seed=base_seed)


def _contract(**kw):
    base = dict(base_seed=42, aux_split_sha256=AUX_SPLIT_V2_SHA256, code_sha256={"x.py": "0" * 64},
                profile_rel="configs/checkpoints/x.yaml", profile_sha256="1" * 64, image_id="test-image")
    base.update(kw)
    return extraction_contract(**base)


# ---------------- 정렬 (decision grid / seed / expert target) ----------------

def test_feature_signature_pinned():
    assert SPEC.signature() == EXPECTED_FEATURE_SIGNATURE


def test_extraction_alignment_grid_targets_and_seeds():
    codec = _codec()
    a = _arrays("ep_a", 37, codec)
    assert a["decision_steps"].tolist() == [0, 5, 10, 15, 20, 25, 30, 35]
    raw, _, _ = _episode(37)
    norm = codec.normalize(raw[:, RAW_INDEX])
    # 결정 t 의 target 은 expert action t..t+15 (모델 예측 아님), 끝 넘으면 0 + invalid
    np.testing.assert_allclose(a["expert_chunks"][1], norm[5:21].astype(np.float32))
    assert a["chunk_valid"][-1].tolist() == [True, True] + [False] * 14
    assert np.all(a["expert_chunks"][-1, 2:] == 0)
    assert a["chunk_valid"][:5].all()            # t<=20: 16 개 전부 episode 안 (20+15=35<=36)
    assert a["chunk_valid"][5].sum() == 12       # t=25 → 25..36
    assert np.array_equal(a["raw_actions"], raw)  # verbatim
    assert a["decision_seeds"][0] == decision_seed(42, "ep_a", 0)
    assert decision_seed(42, "ep_a", 5) != decision_seed(42, "ep_b", 5)
    assert decision_seed(42, "ep_a", 5) == decision_seed(42, "ep_a", 5)


@pytest.mark.parametrize("bad", ["latent_count", "seed", "calls", "frame_index"])
def test_extraction_alignment_rejections(bad):
    codec = _codec()
    raw, r, d = _episode(37)
    n = 8
    kw = dict(episode_id="e", latents=np.zeros((n, 1024), np.float32), decoder_calls=[4] * n,
              seeds=[decision_seed(42, "e", t) for t in range(0, 37, 5)], raw_actions=raw, rewards=r,
              done=d, frame_index=np.arange(37), codec=codec, raw_index=RAW_INDEX, base_seed=42)
    if bad == "latent_count":
        kw["latents"] = np.zeros((n + 1, 1024), np.float32)   # 결정 1개 밀림
    elif bad == "seed":
        kw["seeds"] = list(kw["seeds"])[::-1]
    elif bad == "calls":
        kw["decoder_calls"] = [4] * (n - 1) + [3]
    else:
        fi = np.arange(37)
        fi[10] = 11
        kw["frame_index"] = fi
    with pytest.raises(ContractError):
        build_episode_arrays(**kw)


@pytest.mark.parametrize("mut", ["last3", "early_done", "nonzero_head", "short"])
def test_reward_audit_rejects_other_conventions(mut):
    _, r, d = _episode(40)
    if mut == "last3":
        r[:] = 0
        r[-3:] = 1
    elif mut == "early_done":
        d[20] = True
    elif mut == "nonzero_head":
        r[3] = 0.5
    else:
        r, d = r[-16:], d[-16:]
    with pytest.raises(ContractError):
        audit_rewards(r, d)


def test_reward_audit_accepts_dataset_pattern():
    _, r, d = _episode(40)
    out = audit_rewards(r, d)
    assert out["positive_steps"] == 16 and out["done_indices"] == [39] and out["first_positive"] == 24


# ---------------- terminal / partial ----------------

def test_offline_terminal_and_partial_transitions():
    codec = _codec()
    a = _arrays("ep_t", 37, codec)
    ep = EpisodeInput(episode_id="ep_t", run_group="ep_t", split="train", length=37, end_reason="terminal",
                      decision_steps=a["decision_steps"], latents=a["latents"],
                      action_chunks=a["expert_chunks"], rewards=a["rewards"].astype(np.float64),
                      chunk_valid=a["chunk_valid"])
    tr = build_episode_transitions(ep, execute_steps=5, gamma=0.99)
    assert tr["executed_m"].tolist() == [5] * 7 + [2]
    assert tr["terminals"].tolist() == [False] * 7 + [True]
    assert tr["discounts"][-1] == 0.0 and np.allclose(tr["discounts"][:-1], 0.99 ** 5)
    assert not tr["has_next"][-1] and np.all(tr["next_observations"][-1] == 0)
    np.testing.assert_array_equal(tr["next_observations"][:-1], a["latents"][1:])  # 시간순 다음 latent
    # 결정 t=20: reward 행 20..24 중 21..24 가 1 (first positive=21)
    want = sum(0.99 ** k for k in range(1, 5))
    assert np.isclose(tr["rewards"][4], want)
    assert np.isclose(tr["rewards"][-1], 1 + 0.99)       # 마지막 partial chunk (35, 36)
    assert tr["rewards"][:4].tolist() == [0, 0, 0, 0]


# ---------------- commit / resume ----------------

def _meta(eid, split, psha="p" * 64):
    return {"episode_id": eid, "split": split, "end_reason": "terminal",
            "sources": {"parquet": {"rel": f"data/{eid}.parquet", "sha256": psha}}}


def test_commit_then_resume_verify_and_reject(tmp_path):
    codec = _codec()
    c = _contract()
    ensure_same_or_create(tmp_path / "extraction_contract.json", c, what="c")
    ensure_same_or_create(tmp_path / "extraction_contract.json", c, what="c")   # 같은 계약 → 통과
    with pytest.raises(ContractError, match="충돌"):
        ensure_same_or_create(tmp_path / "extraction_contract.json", _contract(base_seed=7), what="c")
    a = _arrays("ep0", 37, codec)
    rec = commit_episode(tmp_path, episode_index=0, arrays=a, meta=_meta("ep0", "train"), contract=c)
    assert rec["npz"]["file"].endswith(f"{rec['npz']['sha256'][:16]}.npz")
    assert verify_committed_episode(tmp_path, 0, c, expect={"split": "train"})["episode_id"] == "ep0"
    assert verify_committed_episode(tmp_path, 1, c) is None
    with pytest.raises(ContractError, match="덮어쓰기"):
        commit_episode(tmp_path, episode_index=0, arrays=a, meta=_meta("ep0", "train"), contract=c)
    with pytest.raises(ContractError, match="계약"):
        verify_committed_episode(tmp_path, 0, _contract(image_id="other"))
    with pytest.raises(ContractError, match="split"):
        verify_committed_episode(tmp_path, 0, c, expect={"split": "val"})
    npz = tmp_path / "episodes" / rec["npz"]["file"]
    data = bytearray(npz.read_bytes())
    data[-10] ^= 0xFF
    npz.write_bytes(bytes(data))
    with pytest.raises(ContractError, match="sha256"):
        verify_committed_episode(tmp_path, 0, c)


def test_contract_rejects_other_split_sha():
    with pytest.raises(ContractError, match="v2"):
        _contract(aux_split_sha256="0" * 64)


# ---------------- loader (split 누수 방지) ----------------

@pytest.fixture(scope="module")
def split_doc():
    if not SPLIT_V2.is_file() or sha256_path(SPLIT_V2) != AUX_SPLIT_V2_SHA256:
        pytest.skip("aux split v2 없음")
    return json.loads(SPLIT_V2.read_text())


def _fake_extraction(root, split_doc, codec, *, picks, override_split=None):
    c = _contract()
    root.mkdir(parents=True, exist_ok=True)
    ensure_same_or_create(root / "extraction_contract.json", c, what="c")
    for k, e in enumerate(picks):
        split = (override_split or {}).get(e["episode_id"], e["split"])
        a = _arrays(e["episode_id"], 37 + 3 * k, codec, seed=k)
        commit_episode(root, episode_index=e["episode_index"], arrays=a,
                       meta=_meta(e["episode_id"], split, e["parquet_sha256"]), contract=c)
    return c


def _picks(split_doc):
    tr = next(e for e in split_doc["episodes"] if e["split"] == "train")
    va = next(e for e in split_doc["episodes"] if e["split"] == "val")
    return [tr, va]


def test_load_extraction_subset_and_leak_guard(tmp_path, split_doc):
    codec = _codec()
    picks = _picks(split_doc)
    _fake_extraction(tmp_path / "ok", split_doc, codec, picks=picks)
    ld = load_extraction(tmp_path / "ok", SPLIT_V2, codec=codec, raw_index=RAW_INDEX,
                         verify_source=False, allow_subset=True)
    assert [e.episode_id for e in ld.train] == [picks[0]["episode_id"]]
    assert [e.episode_id for e in ld.val] == [picks[1]["episode_id"]]
    assert ld.receipt["subset"] is True and ld.train[0].end_reason == "terminal"
    with pytest.raises(ContractError, match="split 전체"):
        load_extraction(tmp_path / "ok", SPLIT_V2, codec=codec, raw_index=RAW_INDEX,
                        verify_source=False, allow_subset=False)
    # val episode 를 train 으로 기록한 추출 → split 파일과 달라 거부
    _fake_extraction(tmp_path / "leak", split_doc, codec, picks=picks,
                     override_split={picks[1]["episode_id"]: "train"})
    with pytest.raises(ContractError, match="누수"):
        load_extraction(tmp_path / "leak", SPLIT_V2, codec=codec, raw_index=RAW_INDEX,
                        verify_source=False, allow_subset=True)
    # 다른 codec 으로 target 재계산 → 저장 target 과 다르면 거부
    other = ActionCodec(np.full(12, -3.0), np.full(12, 3.0), codec.modes)
    with pytest.raises(ContractError, match="expert target"):
        load_extraction(tmp_path / "ok", SPLIT_V2, codec=other, raw_index=RAW_INDEX,
                        verify_source=False, allow_subset=True)


# ---------------- validation-only selection ----------------

def test_eval_steps_schedule():
    from train_qam_baseline import eval_steps

    s = eval_steps(500_000, 50_000)
    assert s[0] == 0 and s[-1] == 500_000 and len(s) == 11
    assert eval_steps(23, 10) == [0, 10, 20, 23]
    with pytest.raises(ContractError):
        eval_steps(0, 10)


def test_select_best_uses_val_metric_only():
    from train_qam_baseline import VAL_METRIC, NonFinite, select_best

    ev = [{"step": 0, "metrics": {VAL_METRIC: 0.5, "train_like": 0.0}},
          {"step": 10, "metrics": {VAL_METRIC: 0.3, "train_like": 9.0}},
          {"step": 20, "metrics": {VAL_METRIC: 0.3, "train_like": -1.0}},
          {"step": 30, "metrics": {VAL_METRIC: 0.4, "train_like": -9.0}}]
    assert select_best(ev)["step"] == 10          # 동률 → 이른 step, 다른 지표 무시
    bad = copy.deepcopy(ev)
    bad[2]["metrics"][VAL_METRIC] = float("nan")
    with pytest.raises(NonFinite):
        select_best(bad)


# ---------------- membership purpose ----------------

def test_baseline_purpose_mode_separation(tmp_path, split_doc):
    from scripts.path_setup import DATA_ROOT
    from data_contract import (DataContractBlocked, OPENDRAWER_PACKAGE_REL, load_finetune_data_manifest,
                               package_smoke_manifest)

    if not (DATA_ROOT / OPENDRAWER_PACKAGE_REL).is_dir():
        pytest.skip("로컬 package 없음")
    fe = {"status": "confirmed", "feature_signature": SPEC.signature(),
          "capture_point": "action_head.action_decoder.input", "denoise_step": 0}
    paths = {}
    for mode in ("published_package_subset_for_smoke", "published_package_subset_for_baseline"):
        p = tmp_path / f"{mode}.json"
        p.write_text(json.dumps(package_smoke_manifest(SPLIT_V2, feature_extraction=fe, mode=mode)))
        paths[mode] = p
    ok = load_finetune_data_manifest(paths["published_package_subset_for_baseline"],
                                     feature_signature=SPEC.signature(),
                                     purpose="published_package_subset_for_baseline")
    assert ok["membership_claim"]["original_membership_confirmed"] is False
    assert "baseline" in ok["membership_claim"]["data_identity_claim"]
    for path, purpose in ((paths["published_package_subset_for_smoke"], "published_package_subset_for_baseline"),
                          (paths["published_package_subset_for_baseline"], "published_package_subset_for_smoke"),
                          (paths["published_package_subset_for_baseline"], "original_full_reproduction")):
        with pytest.raises(DataContractBlocked):
            load_finetune_data_manifest(path, feature_signature=SPEC.signature(), purpose=purpose)
    with pytest.raises(DataContractBlocked):
        package_smoke_manifest(SPLIT_V2, feature_extraction=fe, mode="historical_exact_100_demos")


# ---------------- CLI ----------------

@pytest.mark.parametrize("script", ["extract_opendrawer_latents.py", "train_qam_baseline.py"])
def test_cli_help(script):
    out = subprocess.run([sys.executable, str(ADAPTER_DIR / script), "--help"], capture_output=True,
                         text=True, timeout=120, cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert "--out" in out.stdout


def test_train_cli_smoke_immutable_outputs(tmp_path, split_doc, cached_checkpoint_dir):
    """synthetic latent 추출 (실제 split ID) → smoke 망 4 step. checkpoint/receipt 불변성·val-only 선택."""
    from scripts.path_setup import DATA_ROOT
    from contract import load_checkpoint_contract
    from data_contract import OPENDRAWER_PACKAGE_REL

    if not (DATA_ROOT / OPENDRAWER_PACKAGE_REL).is_dir():
        pytest.skip("로컬 package 없음")
    import train_qam_baseline as tqb

    codec = load_checkpoint_contract(cached_checkpoint_dir).codec
    modality = json.loads((DATA_ROOT / OPENDRAWER_PACKAGE_REL / "meta/modality.json").read_text())
    assert np.array_equal(raw_to_model_index(modality["action"]), RAW_INDEX)
    ext = tmp_path / "ext"
    _fake_extraction(ext, split_doc, codec, picks=_picks(split_doc))
    out = tmp_path / "run"
    argv = ["--extraction", str(ext), "--out", str(out), "--split", str(SPLIT_V2), "--steps", "4",
            "--eval-every", "2", "--log-every", "1", "--profile", "smoke_reduced_network",
            "--batch-size", "8", "--allow-subset-smoke", "--skip-source-verify"]
    assert tqb.main(argv) == 0
    final = json.loads((out / "selection_final.json").read_text())
    assert [e["step"] for e in final["evaluations"]] == [0, 2, 4]
    vals = {e["step"]: e["metrics"][tqb.VAL_METRIC] for e in final["evaluations"]}
    assert final["selected_step"] == min(vals, key=lambda s: (vals[s], s))
    assert final["production_run"] is False
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == ["step_0000000", "step_0000002",
                                                                       "step_0000004"]
    assert len(list((out / "selection").glob("after_step_*.json"))) == 3
    for e in final["evaluations"]:
        for name, f in e["checkpoint"]["files"].items():
            assert sha256_path(out / "checkpoints" / e["checkpoint"]["dir"] / name) == f["sha256"]
    lines = [json.loads(l) for l in (out / "progress.jsonl").read_text().splitlines()]
    assert lines[0]["event"] == "start" and lines[-1]["event"] == "done"
    assert sum(l["event"] == "train" for l in lines) == 4
    rm = json.loads((out / "run_manifest.json").read_text())
    # Synthetic train T=37: RL2 mask is zero at34..36; decisions30 and35 are terminal.
    assert rm["reward_contract"]["name"] == "rl2_success_expert_last3/v1"
    assert rm["transitions"]["train_terminal_rows"] == 2 and rm["membership_claim"]["purpose"] == \
        "published_package_subset_for_baseline"
    # 같은 out 재사용 거부, 저장 checkpoint 재로드
    with pytest.raises(SystemExit):
        tqb.main(argv)
    from qam_prefix import PrefixQAMTrainer

    sel = final["selected_checkpoint"]["dir"]
    PrefixQAMTrainer.load(out / "checkpoints" / sel, final["selected_step"], feature_signature=SPEC.signature())
    with pytest.raises(ContractError):
        tqb.save_checkpoint(PrefixQAMTrainer.load(out / "checkpoints" / "step_0000004", 4,
                                                  feature_signature=SPEC.signature()),
                            out / "checkpoints", feature_signature=SPEC.signature())
    assert contract_sha256(json.loads((ext / "extraction_contract.json").read_text()))
