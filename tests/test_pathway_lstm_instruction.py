"""per-instruction(ep_meta.lang) 분해 평가 단위 테스트.

통짜 multi-task 검출기는 그대로 두고 functional-CP 평가만 instruction별로 나누는
순수 로직(make_seqs lang 스레딩 / group_by_lang 필터 / who_first / md 렌더)을 검증.

torch 필요(모듈 top-level import). 실행:
  uv run --with numpy --with torch python -m pytest tests/test_pathway_lstm_instruction.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_MOD = (
    Path(__file__).resolve().parents[1]
    / "scripts/safe/groot_n16/robocasa/analyze/pathway_lstm_detector.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("pathway_lstm_detector", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pld = _load()


def _roll(success: int, lang: str, n: int = 5):
    return {
        "vl": np.random.rand(n, 4).astype(np.float32),
        "dit": np.random.rand(n, 7, 3).astype(np.float32),
        "success": success,
        "length": n,
        "task": "T",
        "lang": lang,
    }


def test_make_seqs_threads_lang():
    """make_seqs 가 (Xn, y, length, task, lang) 5-튜플로 lang 을 전달한다."""
    rolls = [_roll(1, "open the drawer"), _roll(0, "close the drawer")]
    out = pld.make_seqs(rolls, "vl", 0)
    assert len(out[0]) == 5, "tuple 에 lang 칸이 있어야 함"
    assert out[0][4] == "open the drawer"
    assert out[1][4] == "close the drawer"
    # y = 1 - success (failure=positive)
    assert out[0][1] == 0 and out[1][1] == 1


def test_group_by_lang_filters_small_subsets():
    """instruction 당 fail>=min & succ>=min 만 남기고 작은 subset 은 제외."""
    seqs = []
    for _ in range(8):
        seqs.append((None, 1, 5, "T", "A"))  # A: 8 fail
    for _ in range(8):
        seqs.append((None, 0, 5, "T", "A"))  # A: 8 succ  -> keep
    for _ in range(8):
        seqs.append((None, 1, 5, "T", "B"))  # B: 8 fail
    for _ in range(4):
        seqs.append((None, 0, 5, "T", "B"))  # B: 4 succ  -> drop
    g = pld.group_by_lang(seqs, min_fail=8, min_succ=8)
    assert set(g) == {"A"}
    assert len(g["A"]) == 16


def test_who_first_picks_earlier_tdet():
    """동일 instruction 에서 mean_tdet_fired 가 작은 pathway 가 'first'."""
    cp_vl = {"A": {"0.30": {"mean_tdet_fired": 0.30}},
             "B": {"0.30": {"mean_tdet_fired": None}}}
    cp_dit = {"A": {"0.30": {"mean_tdet_fired": 0.55}},
              "B": {"0.30": {"mean_tdet_fired": 0.40}}}
    w = pld.who_first(cp_vl, cp_dit, 0.3)
    assert w["A"]["first"] == "VL"   # 0.30 < 0.55
    assert w["B"]["first"] == "DiT"  # VL 미발화(None) → DiT


def test_render_per_instruction_md_contains_rows():
    """렌더된 markdown 에 instruction 문자열과 핵심 컬럼이 포함된다."""
    cell = {"0.30": {"tpr": 0.9, "fpr": 0.1, "bal_acc": 0.9,
                     "mean_tdet_fired": 0.4, "n_fail": 10, "n_succ": 12}}
    per = {
        "alpha_whofirst": 0.3,
        "pathways": {
            "dit": {"seen": {}, "unseen": {"open the drawer": cell}},
            "vl": {"seen": {}, "unseen": {"open the drawer": cell}},
        },
    }
    md = pld.render_per_instruction_md(per, "mlp")
    assert "open the drawer" in md
    assert "TPR" in md and "bal" in md.lower()
    assert "mlp" in md.lower()


# --------------------------------------------------------------------------- e2e

def _write_pkl(path, success, lang, n_steps, L=7, T=18, D=8, Dvl=6, rng=None):
    import pickle
    rng = rng or np.random
    hs = [rng.rand(L, T, D).astype(np.float32) for _ in range(n_steps)]
    vl = [rng.rand(Dvl).astype(np.float32) for _ in range(n_steps)]
    d = {
        "hidden_states": hs,
        "vl_hidden_states": vl,
        "episode_success": int(success),
        "task_id": 0,
        "ep_meta": {"lang": lang},
    }
    with open(path, "wb") as f:
        pickle.dump(d, f)


def test_main_split_instruction_end_to_end(tmp_path):
    """합성 rollout 으로 --split-instruction 전체 경로(load→train→per-instr CP→md) 검증."""
    rng = np.random.RandomState(0)
    run_dir = tmp_path / "raw_rollouts"
    # seen task: cal(성공 절반)≥3 확보 위해 성공 다수
    s_dir = run_dir / "S"; s_dir.mkdir(parents=True)
    for i in range(50):
        _write_pkl(s_dir / f"s{i}.pkl", 1, "seen instr", rng.randint(6, 13), rng=rng)
    for i in range(10):
        _write_pkl(s_dir / f"sf{i}.pkl", 0, "seen instr", rng.randint(6, 13), rng=rng)
    # unseen task: 2 instruction, 각 10 fail + 10 succ → fail≥8 & succ≥8 통과
    u_dir = run_dir / "U"; u_dir.mkdir(parents=True)
    for lang in ("open the top drawer", "open the bottom drawer"):
        tag = lang.split()[2]
        for i in range(10):
            _write_pkl(u_dir / f"u_{tag}_s{i}.pkl", 1, lang, rng.randint(6, 13), rng=rng)
        for i in range(10):
            _write_pkl(u_dir / f"u_{tag}_f{i}.pkl", 0, lang, rng.randint(6, 13), rng=rng)

    out = tmp_path / "analysis"
    argv = [
        "prog", "--run-dir", str(run_dir), "--out", str(out),
        "--seen", "S", "--unseen", "U", "--pathways", "dit,vl",
        "--epochs", "2", "--hidden", "8", "--detector-type", "mlp",
        "--split-instruction", "--min-instr-fail", "8", "--min-instr-succ", "8",
    ]
    import sys
    old = sys.argv
    sys.argv = argv
    try:
        pld.main()
    finally:
        sys.argv = old

    import json
    md_path = out / "detector_results_per_instruction.md"
    assert md_path.exists(), "per-instruction md 가 생성돼야 함"
    md = md_path.read_text()
    assert "open the top drawer" in md and "open the bottom drawer" in md
    assert "who fires first" in md

    res = json.loads((out / "pathway_lstm_detector.json").read_text())
    dit_unseen = res["per_instruction"]["pathways"]["dit"]["unseen"]
    assert set(dit_unseen) == {"open the top drawer", "open the bottom drawer"}
