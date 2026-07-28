"""scripts/scene_sae/* 회귀 테스트 — 2026-07-27 코드리뷰 지적 10건 고정.

번호는 리뷰 지적 번호 (#1 stats 자립, #2 split 지문 대조, #3 층화 alias, #4 라벨↔split
가드, #5 --window, #6 CV Pipeline, #7 순열 null 절차, #9 스캔 검증, #10 행 지문).
전부 CPU·소형 합성 데이터로 몇 초 안에 끝난다.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCENE_SAE = REPO / "scripts" / "scene_sae"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test",
                                                  SCENE_SAE / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


build = _load("build_sae_inputs")


def _eps(scene_layout: dict[int, tuple[int, int]], n_inf: int = 4,
         succ_of=lambda sc, j: int(j % 2 == 0)) -> list[dict]:
    """{scene: (layout, style)} → episode 목록 (scene 당 n_inf 판)."""
    out, ep = [], 0
    for sc, (lay, sty) in scene_layout.items():
        for j in range(n_inf):
            out.append({"episode_idx": ep, "success": succ_of(sc, j),
                        "scenario_seed": sc, "inference_seed": j * 1_000_000,
                        "layout_id": lay, "style_id": sty, "n_records": 5})
            ep += 1
    return out


# ------------------------------------------------------------------ #3 alias
def test_spread_slots_round_robin_aliases():
    """구 방식(그룹 round-robin × 전역 균등슬롯)이 실제로 한 그룹에 몰리는지 **재현**.

    20 scene = layout 2 그룹 × 10 이면 round-robin 순서에서 짝수 인덱스 = 그룹0,
    홀수 = 그룹1. _spread_slots(20, n_val=2, n_test=4) 의 test 슬롯은 2,8,12,18 로
    전부 짝수 = 그룹0 단독. 이 테스트가 통과하는 한 구 방식은 못 쓴다.
    """
    slots = build._spread_slots(20, 2, 4)
    test_pos = [i for i, s in enumerate(slots) if s == 2]
    val_pos = [i for i, s in enumerate(slots) if s == 1]
    assert test_pos == [2, 8, 12, 18]
    assert all(p % 2 == 0 for p in test_pos), "구 방식 alias 재현 실패 (전제가 바뀜)"
    assert all(p % 2 == 1 for p in val_pos)


def test_scene_split_two_layout_groups_are_stratified():
    """#3 실제 배정: 20 scene = 2 layout × 10 → test 4 개가 두 그룹에 나뉜다."""
    scene_layout = {100000 + i: ((0, 0) if i < 10 else (1, 1)) for i in range(20)}
    eps = _eps(scene_layout)
    s2s = build.assign_scene_splits(eps, seed=424101)

    groups = {sc: scene_layout[sc] for sc in s2s}
    test_groups = {groups[sc] for sc, s in s2s.items() if s == 2}
    val_groups = {groups[sc] for sc, s in s2s.items() if s == 1}
    assert sum(1 for s in s2s.values() if s == 2) == 4
    assert len(test_groups) == 2, f"test 가 단일 layout 에 몰림: {test_groups}"
    assert len(val_groups) == 2
    # coverage assert 자체도 통과해야 한다
    cov = build.assert_split_group_coverage(s2s, groups, "scene")
    assert len(cov["train"]) == 2 and len(cov["test"]) == 2


def test_scene_split_five_layout_groups_proportional():
    """5 layout × 4 scene: test 4 개가 서로 다른 4 layout 에 하나씩 (비례 배분)."""
    scene_layout = {100000 + i: (i % 5, i % 5) for i in range(20)}
    eps = _eps(scene_layout)
    s2s = build.assign_scene_splits(eps, seed=0)
    groups = {sc: scene_layout[sc] for sc in s2s}
    test_groups = [groups[sc] for sc, s in s2s.items() if s == 2]
    assert len(test_groups) == 4
    assert len(set(test_groups)) == 4, f"같은 layout 중복: {test_groups}"
    # 모든 그룹이 train 에 최소 1 개 남는다 (cap = size-1)
    train_groups = {groups[sc] for sc, s in s2s.items() if s == 0}
    assert len(train_groups) == 5


def test_split_group_coverage_rejects_single_group_split():
    """#3 coverage assert 가 실제로 걸린다 (한 split 이 단일 그룹일 때)."""
    item2split = {0: 0, 1: 0, 2: 2, 3: 2}
    group_of = {0: ("a",), 1: ("b",), 2: ("a",), 3: ("a",)}     # test 가 전부 a
    with pytest.raises(SystemExit, match="coverage"):
        build.assert_split_group_coverage(item2split, group_of, "scene")
    # 그룹이 1 개뿐이면 요구치가 1 이라 통과
    build.assert_split_group_coverage(item2split, {k: ("a",) for k in item2split}, "scene")


# --------------------------------------------------------------- #9 스캔 검증
def test_validate_episodes_rejects_duplicate_seed_pair():
    eps = _eps({100000: (1, 1)}, n_inf=2)
    eps[1]["inference_seed"] = eps[0]["inference_seed"]         # 쌍 중복
    with pytest.raises(SystemExit, match="중복"):
        build.validate_episodes(eps, require_scene_meta=True)


def test_validate_episodes_rejects_missing_seed():
    eps = _eps({100000: (1, 1)}, n_inf=2)
    eps[0]["scenario_seed"] = -1
    with pytest.raises(SystemExit, match="-1"):
        build.validate_episodes(eps, require_scene_meta=True)


def test_validate_episodes_rejects_inconsistent_fixture():
    eps = _eps({100000: (1, 1)}, n_inf=2)
    eps[1]["layout_id"] = 9                                     # 같은 scene, 다른 layout
    with pytest.raises(SystemExit, match="layout/style"):
        build.validate_episodes(eps, require_scene_meta=True)


def test_validate_episodes_accepts_clean_scan():
    build.validate_episodes(_eps({100000: (1, 1), 100001: (2, 2)}), require_scene_meta=True)


# ------------------------------------------------------------------ #10 지문
def test_row_fingerprint_sensitive_to_episode_set_and_records():
    a = _eps({100000: (1, 1), 100001: (2, 2)})
    b = [dict(e) for e in a]
    assert build.row_fingerprint(a) == build.row_fingerprint(b)
    assert build.row_fingerprint(a) == build.row_fingerprint(list(reversed(b)))  # 정렬 무관
    b[0]["n_records"] += 1
    assert build.row_fingerprint(a) != build.row_fingerprint(b)
    c = a[:-1]
    assert build.row_fingerprint(a) != build.row_fingerprint(c)
    assert len(build.row_fingerprint(a)) == 12


# ----------------------------------------------- probe: #4 라벨↔split, #6 CV
probe = _load("probe_scene")


def test_usable_cv_folds_skips_class_missing_fold():
    """#6 클래스가 빠진 fold 는 스킵된다."""
    y = np.array([0, 0, 1, 1, 2, 2])
    splits = [(np.array([0, 1, 2, 3]), np.array([4, 5])),        # 클래스 2 없음 → 스킵
              (np.array([0, 2, 4, 5]), np.array([1, 3]))]        # 전 클래스 有 → 유지
    keep, skipped = probe.usable_cv_folds(splits, y)
    assert len(keep) == 1 and len(skipped) == 1
    assert skipped[0]["missing_classes"] == [2]


def test_select_C_falls_back_when_all_folds_unusable(capsys):
    """#6 전 fold 스킵이면 C=1 fallback + 경고."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 5))
    y = np.repeat(np.arange(6), 10)          # 클래스 6 개, group=episode 도 6 개
    ep = np.repeat(np.arange(6), 10)         # fold 마다 클래스가 반드시 빠진다
    C, detail, warn = probe.select_C(Z, y, ep, [0.1, 1.0, 10.0], seed=0, max_iter=50,
                                     solver="lbfgs", folds=3, verbose=True)
    assert C == 1.0 and detail["fallback_C"] == 1.0
    assert "fallback" in capsys.readouterr().out


def test_probe_pipeline_scaler_fits_on_train_only():
    """#6 Pipeline 의 scaler 가 fit 된 행에서만 통계를 얻는다 (test 통계 누수 없음)."""
    rng = np.random.default_rng(0)
    Ztr = rng.normal(loc=0.0, scale=1.0, size=(200, 4))
    Zte = rng.normal(loc=50.0, scale=1.0, size=(50, 4))          # test 만 극단 평균
    y = (Ztr[:, 0] > 0).astype(int)
    _acc, pipe, _w = probe.fit_probe(Ztr, y, Zte, np.zeros(50, int), 0, 200, 1.0)
    sc = pipe.named_steps["scaler"]
    assert np.allclose(sc.mean_, Ztr.mean(0)), "scaler 가 test 통계에 오염됨"


def test_permutation_null_uses_true_test_labels():
    """#7 순열 null 은 train 라벨만 섞고 test 는 진짜 라벨로 평가한다.

    신호가 강한 합성 데이터에서 순열 null 평균은 우연 수준 근처여야 하고, 본 probe 정확도
    (≈1.0)와 확실히 떨어져야 한다.
    """
    rng = np.random.default_rng(0)
    n_ep, per_ep = 12, 20
    ep = np.repeat(np.arange(n_ep), per_ep)
    lab = np.repeat(np.arange(n_ep) % 2, per_ep)
    Z = rng.normal(size=(n_ep * per_ep, 3)) + 4.0 * lab[:, None]
    tr = ep < 8
    acc, _c, _w = probe.fit_probe(Z[tr], lab[tr], Z[~tr], lab[~tr], 0, 200, 1.0)
    nulls, _w = probe.permutation_null(Z[tr], Z[~tr], ep[tr], ep[~tr], lab[tr], lab[~tr],
                                       n_perm=12, seed=0, max_iter=200, C=1.0, verbose=False)
    assert acc > 0.95
    assert len(nulls) >= 8
    assert nulls.mean() < 0.8, f"null 이 너무 높다 ({nulls.mean():.3f}) — 절차 확인"


# ------------------------------------------------------- e2e (합성 pkl → probe)
def _write_fake_pkls(root: Path, n_scene=6, n_inf=4, D=24, T=4, K=2, cap=(0, 2),
                     n_rec=3, seed=0, dup_seed=False):
    """축소 합성 컬렉션 (계약만 동일: hidden_states/capture_layers/feature_phases/ep_meta)."""
    import pickle

    import torch
    rng = np.random.default_rng(seed)
    d = root / "cell"
    d.mkdir(parents=True, exist_ok=True)
    ep = 0
    for si in range(n_scene):
        sc = 100000 + si
        lay = si % 2
        for j in range(n_inf):
            succ = int(j % 2 == 0)
            nrec = n_rec if succ else n_rec + 2
            recs = [torch.from_numpy(
                (rng.normal(scale=0.5, size=(len(cap), K, T, D)) + 2.0 * lay
                 ).astype(np.float32)).to(torch.float16) for _ in range(nrec)]
            obj = {"hidden_states": recs, "capture_layers": list(cap),
                   "feature_phases": [f"p{r % 2}" for r in range(nrec)],
                   "episode_idx": ep, "episode_success": succ, "scenario_seed": sc,
                   "inference_seed": (0 if dup_seed else j) * 1_000_000,
                   "ep_meta": {"layout_id": lay, "style_id": lay}}
            (d / f"t--ep{ep}--succ{succ}.pkl").write_bytes(pickle.dumps(obj))
            ep += 1
    return d


def _run_build(scan_dir: Path, out_dir: Path, layers: str, split_by="scene"):
    return subprocess.run(
        [sys.executable, str(SCENE_SAE / "build_sae_inputs.py"),
         "--scan-dir", str(scan_dir), "--split-by", split_by, "--layers", layers,
         "--out-dir", str(out_dir), "--dtype", "fp32"],
        capture_output=True, text=True, cwd=str(REPO))


def test_build_append_fingerprint_mismatch_errors(tmp_path):
    """#10 같은 out-dir 이어쓰기: 행 구성이 같으면 통과, 바뀌면 에러."""
    scan = _write_fake_pkls(tmp_path / "coll")
    out = tmp_path / "out"
    r1 = _run_build(scan, out, "0")
    assert r1.returncode == 0, r1.stdout + r1.stderr
    assert (out / "X_L0.npz").exists() and (out / "meta.npz").exists()
    fp = str(np.load(out / "meta.npz")["row_fingerprint"])
    assert str(np.load(out / "X_L0.npz")["row_fingerprint"]) == fp
    assert json.loads((out / "split.json").read_text())["row_fingerprint"] == fp

    # (a) 같은 수집으로 다른 layer 이어쓰기 → 통과 + 같은 지문
    r2 = _run_build(scan, out, "2")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert str(np.load(out / "X_L2.npz")["row_fingerprint"]) == fp

    # (b) 수집이 바뀐 상태로 이어쓰기 → 에러
    scan2 = _write_fake_pkls(tmp_path / "coll2", n_rec=4, seed=1)
    r3 = _run_build(scan2, out, "2")
    assert r3.returncode != 0
    assert "fingerprint" in (r3.stdout + r3.stderr)


def test_build_rejects_duplicate_seed_pairs(tmp_path):
    """#9 (scenario_seed, inference_seed) 중복 수집은 빌드 단계에서 막힌다."""
    scan = _write_fake_pkls(tmp_path / "coll", dup_seed=True)
    r = _run_build(scan, tmp_path / "out", "0")
    assert r.returncode != 0
    assert "중복" in (r.stdout + r.stderr)


# ------------------------------------------- e2e: build → train → probe (#1/#2/#4/#5)
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """합성 scene-matched 컬렉션 1개로 build + train 을 한 번만 돌려 재사용."""
    root = tmp_path_factory.mktemp("e2e")
    scan = _write_fake_pkls(root / "coll", n_scene=8, n_inf=4, D=24, n_rec=4)
    out = root / "out"
    r = _run([sys.executable, str(SCENE_SAE / "build_sae_inputs.py"),
              "--scan-dir", str(scan), "--split-by", "scene", "--layers", "0",
              "--out-dir", str(out), "--dtype", "fp32"])
    assert r.returncode == 0, r.stdout + r.stderr
    ck = root / "ckpt"
    rt = _run([sys.executable, str(SCENE_SAE / "train_scene_sae.py"),
               "--x", str(out / "X_L0.npz"), "--meta", str(out / "meta.npz"),
               "--stats", str(out / "stats_L0.npz"), "--cell", "cell", "--layer", "0",
               "--m", "48", "--k", "4", "--epochs", "3", "--patience", "3",
               "--min-epochs", "1", "--batch-size", "256", "--device", "cpu",
               "--split-col", "split_scene", "--out-dir", str(ck)])
    assert rt.returncode == 0, rt.stdout + rt.stderr
    return {"out": out, "ckpt": ck, "root": root, "train_log": rt.stdout}


def test_train_computes_own_stats_and_records_split(e2e):
    """#1 학습이 자기 train split 행에서 mean/std 를 계산해 ckpt 에 저장 + #2 지문 기록."""
    st = np.load(e2e["ckpt"] / "stats.npz")
    meta = np.load(e2e["out"] / "meta.npz", allow_pickle=False)
    X = np.load(e2e["out"] / "X_L0.npz")["X"]
    tr = meta["split_scene"] == 0
    assert np.allclose(st["mean"], X[tr].astype(np.float64).mean(0), atol=1e-4)
    assert int(st["n_train_rows"]) == int(tr.sum())
    # 빌더 stats (split_scene 이 기본축이라 같은 값) 와도 대조된다
    assert "표준화 통계 자체 계산" in e2e["train_log"]

    cfg = json.loads((e2e["ckpt"] / "config.json").read_text())
    assert cfg["split_col"] == "split_scene"
    assert cfg["split_axis_scene_heldout"] is True
    assert len(cfg["train_episode_fingerprint"]) == 12
    assert cfg["train_scenes"] and cfg["stats_source"] == "self_computed_from_train_split"
    assert cfg["row_fingerprint"] == str(meta["row_fingerprint"])


def test_train_stats_differ_by_split_col(e2e, tmp_path):
    """#1 핵심: split_col 을 바꾸면 표준화 통계도 바뀐다 (빌더 stats 재사용은 누수)."""
    ck2 = tmp_path / "ckpt_ep"
    r = _run([sys.executable, str(SCENE_SAE / "train_scene_sae.py"),
              "--x", str(e2e["out"] / "X_L0.npz"), "--meta", str(e2e["out"] / "meta.npz"),
              "--cell", "cell", "--layer", "0", "--m", "48", "--k", "4",
              "--epochs", "2", "--patience", "2", "--min-epochs", "1",
              "--batch-size", "256", "--device", "cpu",
              "--split-col", "split_episode", "--out-dir", str(ck2)])
    assert r.returncode == 0, r.stdout + r.stderr
    a = np.load(e2e["ckpt"] / "stats.npz")["mean"]
    b = np.load(ck2 / "stats.npz")["mean"]
    assert not np.allclose(a, b), "split 축이 달라도 통계가 같다 — 자체 계산이 안 된 것"
    cfg_a = json.loads((e2e["ckpt"] / "config.json").read_text())
    cfg_b = json.loads((ck2 / "config.json").read_text())
    assert cfg_a["train_episode_fingerprint"] != cfg_b["train_episode_fingerprint"]


def _probe_cmd(e2e, out_json, *extra):
    return [sys.executable, str(SCENE_SAE / "probe_scene.py"),
            "--ckpt-dir", str(e2e["ckpt"]), "--x", str(e2e["out"] / "X_L0.npz"),
            "--meta", str(e2e["out"] / "meta.npz"), "--split-col", "split_scene",
            "--n-perm", "3", "--max-iter", "60", "--no-segments",
            "--out", str(out_json), *extra]


def test_probe_split_mismatch_errors_and_flag_overrides(e2e, tmp_path):
    """#2 probe 의 --split-col 이 ckpt 와 다르면 에러, --allow-split-mismatch 로만 강행."""
    cmd = _probe_cmd(e2e, tmp_path / "p.json")
    i = cmd.index("split_scene")
    cmd[i] = "split_episode"
    r = _run(cmd)
    assert r.returncode != 0
    assert "split" in (r.stdout + r.stderr)

    r2 = _run(cmd + ["--allow-split-mismatch"])
    assert r2.returncode == 0, r2.stdout + r2.stderr
    res = json.loads((tmp_path / "p.json").read_text())
    dc = res["data_check"]
    assert dc["split_mismatch_allowed"] is True
    assert any("MISMATCH" in w for w in dc["warnings"])
    # #4 layout_id × scene 공유 split → 경고가 json 에 남는다
    assert dc["scene_heldout"] is False
    assert any("암기" in w for w in dc["warnings"])


def test_probe_scenario_seed_label_with_scene_split_errors(e2e, tmp_path):
    """#4 scenario_seed 라벨 × scene held-out 은 평가 불가 → 에러."""
    r = _run(_probe_cmd(e2e, tmp_path / "p2.json", "--label", "scenario_seed"))
    assert r.returncode != 0
    assert "scenario_seed" in (r.stdout + r.stderr)


def test_probe_window_limits_records(e2e, tmp_path):
    """#5 --window N 은 record_idx < N 행만 사용하고 json 에 남는다."""
    r_all = _run(_probe_cmd(e2e, tmp_path / "w0.json", "--records-per-ep", "-1"))
    assert r_all.returncode == 0, r_all.stdout + r_all.stderr
    r_win = _run(_probe_cmd(e2e, tmp_path / "w2.json", "--records-per-ep", "-1",
                            "--window", "2"))
    assert r_win.returncode == 0, r_win.stdout + r_win.stderr
    a = json.loads((tmp_path / "w0.json").read_text())
    b = json.loads((tmp_path / "w2.json").read_text())
    assert b["window"] == 2 and a["window"] == 0
    assert b["n_train_rows"] < a["n_train_rows"]
    assert b["data_check"]["window"] == 2


def test_probe_uses_ckpt_stats(e2e, tmp_path):
    """#1 probe 가 빌더 stats 가 아니라 ckpt 의 stats.npz 를 쓴다."""
    r = _run(_probe_cmd(e2e, tmp_path / "p3.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads((tmp_path / "p3.json").read_text())
    assert res["data_check"]["stats_source"].endswith("ckpt/stats.npz")
    assert res["probe_version"] == 3
