"""wrong_grasp_vl_separation.py 로컬 synthetic smoke (torch 불필요, numpy 만).

검증 항목:
  1. FoldProjector.loo_auroc == ps.loo_auroc (수학 동치 — 캐시가 결과를 바꾸면 안 됨)
  2. window_indices: W_pre 가 첫 wg 이전 reach 만 / W_at / W_early
  3. build_design: budget auto = global-min(탈락 0), 고정 budget 은 미달 탈락 보고
  4. exact permutation: 7v5 → C(12,7)=792 전수
  5. 심은 신호 회수: wg 에 mean-shift 심으면 SIGNAL, null 데이터면 p 비유의(대개)
  6. spearman / trel_curve smoke

실행: python3 tests/test_wrong_grasp_vl_separation.py  (analyze/ 상위에서)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import phase_separation as ps  # noqa: E402
import wrong_grasp_vl_separation as wg  # noqa: E402

rng = np.random.default_rng(7)


def make_roll(name: str, success: int, phases: list[str], dvl: int = 32,
              shift: float = 0.0) -> dict:
    """synthetic rollout dict (ps.load_rollout 출력 형태)."""
    n = len(phases)
    vl = rng.normal(size=(n, dvl)).astype(np.float32)
    if shift:
        vl += shift  # 전 record 에 mean-shift (심은 신호)
    return {"name": name, "success": success, "length": n,
            "dit": rng.normal(size=(n, 2, 16)).astype(np.float32),
            "vl": vl, "phases": list(phases), "capture_layers": [0, 2]}


def synth_cell(shift_wg: float) -> list[dict]:
    """succ 12 / other_fail 4 / wg 5 의 미니 cell."""
    rolls = []
    for i in range(12):
        rolls.append(make_roll(f"succ{i}", 1,
                     ["reach-to-object"] * (4 + i % 3) + ["grasp"] * 3 + ["transport"] * 4))
    for i in range(4):
        rolls.append(make_roll(f"of{i}", 0, ["reach-to-object"] * 20))
    for i in range(5):
        pre = 6 + 2 * i
        rolls.append(make_roll(f"wg{i}", 0,
                     ["reach-to-object"] * pre + ["wrong-grasp"] * 5 + ["reach-to-object"] * 3,
                     shift=shift_wg))
    return rolls


def test_fold_projector_equiv():
    X = rng.normal(size=(14, 40))
    y = np.array([1] * 5 + [0] * 9)
    fp = wg.FoldProjector(X)
    a_cached = fp.loo_auroc(y)
    a_ref = ps.loo_auroc(X, y)
    assert abs(a_cached - a_ref) < 1e-9, (a_cached, a_ref)
    # 무작위 재라벨 몇 개에서도 동치
    for _ in range(5):
        yp = rng.permutation(y)
        assert abs(fp.loo_auroc(yp) - ps.loo_auroc(X, yp)) < 1e-9
    print("[ok] FoldProjector == ps.loo_auroc")


def test_windows():
    r = make_roll("a", 0, ["reach-to-object"] * 3 + ["grasp"] + ["wrong-grasp"] * 2
                  + ["reach-to-object"] * 2)
    assert wg.classify_episode(r) == "wg"
    assert wg.first_wg_index(r) == 4
    assert wg.window_indices(r, "W_pre") == [0, 1, 2]  # 첫 wg 이전 reach 만
    assert wg.window_indices(r, "W_at") == [4, 5]
    assert wg.window_indices(r, "W_early", k=3) == [0, 1, 2]
    s = make_roll("b", 1, ["reach-to-object"] * 2 + ["grasp"] * 2 + ["transport"])
    assert wg.classify_episode(s) == "succ"
    assert wg.window_indices(s, "W_pre") == [0, 1]      # 비-wg = reach 전체
    assert wg.window_indices(s, "W_at") == [2, 3, 4]     # grasp+transport
    print("[ok] window_indices")


def test_build_design_budget():
    rolls = synth_cell(0.0)
    sub, y = wg.comparison_labels(rolls, "rest_all")
    d = wg.build_design(sub, y, "W_pre", "VL", budget="auto")
    assert d is not None
    # auto = global-min reach count = succ 최소 4 → 탈락 0
    assert d["budget"] == 4, d["budget"]
    assert not d["dropped"]["under_budget"]
    assert len(d["names"]) == 21
    # 고정 budget 6 → reach 4~5 인 succ 탈락, 보고됨
    d6 = wg.build_design(sub, y, "W_pre", "VL", budget=6)
    assert d6["dropped"]["under_budget"], "고정 budget 미달 탈락이 기록돼야 함"
    assert len(d6["names"]) + len(d6["dropped"]["under_budget"]) == 21
    print("[ok] build_design budget rules")


def test_exact_perm_count():
    y = np.array([1] * 7 + [0] * 5)
    kind, gen, total = wg._perm_labelings(y, 1000, rng)
    assert kind == "exact" and total == math.comb(12, 7) == 792
    labelings = list(gen)
    assert len(labelings) == 792
    assert all(int(v.sum()) == 7 for v in labelings)
    print("[ok] exact permutation C(12,7)=792")


def test_signal_recovery():
    r = np.random.default_rng(0)
    # 심은 신호: wg VL 에 +1.2 mean shift → SIGNAL 기대
    rolls = synth_cell(shift_wg=1.2)
    sub, y = wg.comparison_labels(rolls, "rest_all")
    d = wg.build_design(sub, y, "W_pre", "VL", budget="auto")
    fp = wg.FoldProjector(d["X"])
    st = wg.perm_stats(fp, d["y"], 300, r)
    assert st["auroc"] > 0.9, st
    assert st["p_perm"] < 0.05, st
    # null 데이터 → 관측 dev 가 null 상단 아래 (확률적이라 p 로만 느슨히 확인)
    rolls0 = synth_cell(shift_wg=0.0)
    sub0, y0 = wg.comparison_labels(rolls0, "rest_all")
    d0 = wg.build_design(sub0, y0, "W_pre", "VL", budget="auto")
    st0 = wg.perm_stats(wg.FoldProjector(d0["X"]), d0["y"], 300, r)
    assert st0["p_perm"] > 0.01, st0  # null 에서 강한 유의가 나오면 문제
    print(f"[ok] signal recovery: planted auroc={st['auroc']:.3f} p={st['p_perm']:.4f} | "
          f"null p={st0['p_perm']:.3f}")


def test_postdrop_window():
    # wg: reach3 grasp5 place2 [drop@9] reach6(10-15) wg4(16-19) — postdrop = reach 10..15
    r = make_roll("w", 0, ["reach-to-object"] * 3 + ["grasp"] * 5 + ["place"] * 2
                  + ["reach-to-object"] * 6 + ["wrong-grasp"] * 4)
    r["drop_steps"] = [9]; r["grasp_steps"] = [3]
    idx, st = wg.postdrop_window(r)
    assert st == "ok" and idx == [10, 11, 12, 13, 14, 15], (idx, st)
    # drop 없는 wg → 제외
    r2 = make_roll("w2", 0, ["reach-to-object"] * 3 + ["wrong-grasp"] * 2)
    r2["drop_steps"] = []; r2["grasp_steps"] = []
    assert wg.postdrop_window(r2) == ([], "no_drop_before_wg")
    # 비-wg drop-재획득: [drop@5] reach(6-9) [grasp@10] — postdrop = 6..9
    s = make_roll("s", 1, ["reach-to-object"] * 3 + ["grasp"] * 3
                  + ["reach-to-object"] * 4 + ["grasp"] * 5)
    s["drop_steps"] = [5]; s["grasp_steps"] = [3, 10]
    idx, st = wg.postdrop_window(s)
    assert st == "ok" and idx == [6, 7, 8, 9], (idx, st)
    # drop 없는 비-wg → event-state 부재로 제외
    s2 = make_roll("s2", 1, ["reach-to-object"] * 2 + ["grasp"] * 3)
    s2["drop_steps"] = []; s2["grasp_steps"] = [2]
    assert wg.postdrop_window(s2) == ([], "no_drop")
    # build_postdrop_design: wg 2 + succ-drop 2 → budget=min, event-제외 목록 기록
    rolls = []
    for i in range(3):
        rr = make_roll(f"w{i}", 0, ["reach-to-object"] * 3 + ["grasp"] * 3 + ["place"] * 2
                       + ["reach-to-object"] * (4 + i) + ["wrong-grasp"] * 3, shift=1.0)
        rr["drop_steps"] = [7]; rr["grasp_steps"] = [3]
        rolls.append(rr)
    for i in range(3):
        ss = make_roll(f"s{i}", 1, ["reach-to-object"] * 3 + ["grasp"] * 3
                       + ["reach-to-object"] * (5 + i) + ["grasp"] * 4)
        ss["drop_steps"] = [5]; ss["grasp_steps"] = [3, 11 + i]
        rolls.append(ss)
    nd = make_roll("nodrop", 1, ["reach-to-object"] * 3 + ["grasp"] * 5)
    nd["drop_steps"] = []; nd["grasp_steps"] = [3]
    rolls.append(nd)
    d = wg.build_postdrop_design(rolls, "VL")
    assert d is not None and d["n_pos"] == 3 and d["n_neg"] == 3
    assert d["excluded"].get("no_drop") == ["nodrop(succ)"]
    assert d["budget"] == min(d["counts"]) and d["budget"] >= 4
    print("[ok] postdrop_window / build_postdrop_design")


def test_misc():
    a = np.array([1.0, 2, 3, 4, 5])
    assert abs(wg.spearman(a, a * 2 + 1) - 1.0) < 1e-9
    assert abs(wg.spearman(a, -a) + 1.0) < 1e-9
    rolls = synth_cell(0.6)
    tr = wg.trel_curve(rolls, "VL", max_back=4)
    assert set(tr) == {"-1", "-2", "-3", "-4"}
    assert all(("auroc" in v and "n_wg" in v) for v in tr.values())
    sc = wg.diag_mahal_scores(np.vstack([np.zeros((5, 8)), np.ones((3, 8)) * 3]),
                              np.array([0] * 5 + [1] * 3))
    assert sc[5:].mean() > sc[:5].mean()  # 양성이 참조분포에서 멀어야
    cen = wg.census(rolls)
    assert cen["n"] == {"succ": 12, "other_fail": 4, "wg": 5}
    print("[ok] spearman / trel / mahal / census")


if __name__ == "__main__":
    test_fold_projector_equiv()
    test_windows()
    test_build_design_budget()
    test_exact_perm_count()
    test_signal_recovery()
    test_postdrop_window()
    test_misc()
    print("\nALL OK")
