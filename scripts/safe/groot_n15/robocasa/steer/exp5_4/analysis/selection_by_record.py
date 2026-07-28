#!/usr/bin/env python3
"""exp5-4 후속 — 선택 신호를 record 0 이 아니라 record 1·2 에서 재평가.

동기: exp5-4 는 record 0 (t=0, 아직 아무 action 도 실행 전) 활성만 썼고, 13/13 적중은
seed 주효과로 설명됐다 (column 순열 p=.60, seed_only 동률). record r 은 5·10 env-step
을 **실제로 실행한 뒤**의 inference 활성이므로, 노이즈 draw 가 만든 궤적 차이가 활성에
반영돼 있을 수 있다. 여기서 묻는 것: r 이 커지면 선택이 (a) 한 seed column 붕괴에서
벗어나 scene-조건적이 되는가, (b) prospective 설정과 chunk baseline 을 이기는가.

표현 V (★ 유일한 변경점)
  · 기존: V = X[:W, li, :].mean(0)  (누적창, 기본 W=1 = record 0)
  · 여기: V = X[r, li, :]           (단일 record r ∈ {0,1,2}) — 방향 fit 도 같은 r 활성.
  · n_rec ≤ r 인 판이 있으면 `to_matrix(rec=r)` 이 에러 → 그 r 은 기록만 남기고 건너뜀.

r × L 마다 실행하는 3 블록
  1. prospective 게이트 (selection_gate.py 와 동일 설계): 8 seed 를 4+4 fold 로 나눠
     scene-out × seed-out 이중 제외 → held-out 4 후보 중 top-1 (K=4).
     학습축 Δ̂ (두 fold) + p_라벨셔플 재fit + p_seed순열, 그리고 pkl 캐시 기반
     chunk_tv / chunk_speed_mean / seed_only 를 **동일 split** 으로 비교.
     ※ baseline 의 chunk 특징은 record 0 (첫 chunk) 기준 그대로 — 비교 기준선 목적.
  2. in-fold K=8: 전 8 후보 대상 LOSO 선택 + 8!=40320 column 전수 순열 p
     + 라벨셔플 p + **선택 분포(chosen column histogram)**.
  3. drop-seed-0: column 0 을 fit·평가 양쪽에서 제거한 7 draw 버전의 Δ̂·선택 분포.

primary L0, 부표 L12.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sel_common import (load, to_matrix, loso_select, score_select,  # noqa: E402
                         delta_stat, seed_permutations)
from _sel_baselines import (seed_only_scores, pkl_pass, DEPLOYABLE)  # noqa: E402


# ── null 2종 ────────────────────────────────────────────────────────────
def seedperm_null(select_fn, Y, perms, obs_delta):
    """seed column 공통 순열 null (fit 포함 재실행)."""
    null = np.array([delta_stat(select_fn(Y[:, list(pi)]))["delta"] for pi in perms])
    return float((np.sum(null >= obs_delta) + 1) / (len(perms) + 1)), null


def labelshuffle_null(select_fn, Y, n_perm, rng, obs_delta):
    """scene 내 라벨셔플 + 재fit null (학습 선택자 primary)."""
    null = np.empty(n_perm)
    for k in range(n_perm):
        Yp = np.stack([rng.permutation(row) for row in Y])
        null[k] = delta_stat(select_fn(Yp))["delta"]
    return float((np.sum(null >= obs_delta) + 1) / (n_perm + 1)), null


def choice_summary(chosen, orig_idx):
    """선택 분포 요약 — 원 column index 기준 히스토그램 + 집중도."""
    ch = [int(orig_idx[c]) for c in np.asarray(chosen)]
    cnt = Counter(ch)
    n = len(ch)
    p = np.array([v / n for v in cnt.values()])
    ent = float(-(p * np.log(p)).sum())
    J = len(orig_idx)
    return dict(counts={str(k): int(v) for k, v in sorted(cnt.items())},
                n_distinct=len(cnt), top_col=int(max(cnt, key=cnt.get)),
                top_frac=float(max(cnt.values()) / n),
                entropy_nats=ent, entropy_max=float(np.log(J)),
                chosen_orig=ch)


def null_desc(x):
    return dict(mean=float(x.mean()), sd=float(x.std()),
                q=[float(q) for q in np.percentile(x, [5, 50, 95])])


# ── 블록 실행기 ─────────────────────────────────────────────────────────
def run_infold(A, Y, orig_idx, perms, exhaustive, n_perm_label, rng):
    sel = (lambda Yp, AA=A: loso_select(AA, Yp))
    res = sel(Y)
    st = delta_stat(res)
    t0 = time.time()
    p_seed, nullS = seedperm_null(sel, Y, perms, st["delta"])
    p_lab, nullL = labelshuffle_null(sel, Y, n_perm_label, rng, st["delta"])
    return dict(**st, expected_hits=float(res["base"][res["valid"]].sum()),
                p_seedperm=p_seed, p_labelshuffle=p_lab,
                n_perm_seed=len(perms), perm_exhaustive=exhaustive,
                n_perm_label=n_perm_label,
                null_seedperm=null_desc(nullS), null_labelshuffle=null_desc(nullL),
                choice=choice_summary(res["chosen"], orig_idx),
                sec=round(time.time() - t0, 1))


def run_prospective(A, Y, folds, seeds, perms, exhaustive, n_perm_label, rng):
    out = {}
    for fname, train_mask in folds:
        test_mask = ~train_mask
        sel = (lambda Yp, tr=train_mask, te=test_mask, AA=A:
               loso_select(AA, Yp, fit_seeds=tr, eval_seeds=te))
        res = sel(Y)
        st = delta_stat(res)
        t0 = time.time()
        p_seed, nullS = seedperm_null(sel, Y, perms, st["delta"])
        p_lab, nullL = labelshuffle_null(sel, Y, n_perm_label, rng, st["delta"])
        out[fname] = dict(
            train_seeds=[seeds[i] for i in np.where(train_mask)[0]],
            test_seeds=[seeds[i] for i in np.where(test_mask)[0]],
            **st, expected_hits=float(res["base"][res["valid"]].sum()),
            p_seedperm=p_seed, p_labelshuffle=p_lab,
            null_seedperm=null_desc(nullS), null_labelshuffle=null_desc(nullL),
            choice=choice_summary(res["chosen"], np.arange(Y.shape[1])),
            sec=round(time.time() - t0, 1))
    return out


def run_baselines(scores, Y, folds, perms, S):
    """동일 prospective split 에서 고정 선택자 baseline."""
    out = {}
    for bname, sc in sorted(scores.items()):
        for asc in (True, False):
            arr = []
            for fname, train_mask in folds:
                te = ~train_mask
                fn = (lambda Yp, t=te, s=sc, aa=asc:
                      score_select(s, Yp, eval_seeds=t, ascending=aa))
                r = fn(Y)
                st = delta_stat(r)
                p_sp, _ = seedperm_null(fn, Y, perms, st["delta"])
                arr.append(dict(fold=fname, **st, p_seedperm=p_sp))
            tag = "낮은쪽선택" if asc else "높은쪽선택"
            kind = ("negative_control" if bname == "seed_only" else
                    "deployable" if bname in DEPLOYABLE else "other")
            out[f"{bname}|{tag}"] = dict(
                delta_pooled=float(np.mean([x["delta"] for x in arr])), folds=arr,
                coverage_ok=all(x["n_scene"] == S for x in arr), kind=kind,
                deployable=bool(kind == "deployable"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/home/kimseungjun/sm_npz")
    ap.add_argument("--cells", default="pq3_drawer_right")
    ap.add_argument("--records", default="0,1,2")
    ap.add_argument("--layers", default="0,12", help="primary=첫번째")
    ap.add_argument("--drop-col", type=int, default=0, help="drop-seed ablation 대상 column")
    ap.add_argument("--n-perm-seed", type=int, default=40320)
    ap.add_argument("--n-perm-label", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--pkl-cache-dir", default="/home/kimseungjun/exp54_results/pkl_cache")
    ap.add_argument("--out", default="/home/kimseungjun/exp54_results/selection_by_record.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    recs = [int(x) for x in a.records.split(",")]
    want_layers = [int(x) for x in a.layers.split(",")]
    out = dict(config=vars(a), cells={})

    for cell in a.cells.split(","):
        eps, layers = load(Path(a.npz_dir), cell)
        if not eps:
            print(f"[skip] {cell}: npz 없음")
            continue
        _A0, Y, scenes, seeds, E = to_matrix(eps, 0)
        S, J = Y.shape
        nrec = np.array([e["X"].shape[0] for e in eps])
        print(f"\n{'='*104}\n{cell}: {S} scene × {J} seed = {S*J}판 · SR {Y.mean():.3f}"
              f" · layers {layers}")
        print(f"  n_rec: min {nrec.min()} / median {int(np.median(nrec))} / max {nrec.max()}"
              f" · n_rec<3 인 판 {int((nrec < 3).sum())}개")
        cell_out = dict(n_scene=S, n_seed=J, sr_all=float(Y.mean()),
                        seeds=[int(s) for s in seeds], layers=layers,
                        m_i=Y.sum(1).tolist(),
                        n_rec=dict(min=int(nrec.min()), max=int(nrec.max()),
                                   median=int(np.median(nrec)),
                                   lt3=int((nrec < 3).sum())),
                        per_column_sr=(Y.sum(0) / S).tolist(), runs={}, baselines={})

        folds = [("fold1", np.arange(J) < J // 2), ("fold2", np.arange(J) >= J // 2)]
        perms8, ex8 = seed_permutations(J, a.n_perm_seed, rng)
        keep = np.array([j for j in range(J) if j != a.drop_col])
        perms7, ex7 = seed_permutations(len(keep), a.n_perm_seed, rng)
        print(f"  column 순열 J={J}: {len(perms8)}({'전수' if ex8 else '부분'})"
              f" · J={len(keep)}: {len(perms7)}({'전수' if ex7 else '부분'})"
              f" · 라벨셔플 {a.n_perm_label}회")

        # ── baseline (record 무관, 1회) ───────────────────────────────
        if not a.no_baselines:
            ch, _oracle, diag = pkl_pass(cell, E, scenes,
                                         cache=f"{a.pkl_cache_dir}/{cell}_chunk.npz")
            print(f"  pkl chunk baseline: {diag['n_ok']}/{diag['n_total']}판"
                  f" (결측 {diag['n_miss']}) · 출처 {diag['source']}")
            cell_out["pkl_note"] = {k: v for k, v in diag.items() if k != "cached"}
            sc = dict(ch)
            sc["seed_only"] = seed_only_scores(seeds, S)
            cell_out["baselines"] = run_baselines(sc, Y, folds, perms8, S)
            for k, v in cell_out["baselines"].items():
                print(f"  [baseline] {k:28} {v['kind']:16} Δ̂ {v['delta_pooled']:+.3f}"
                      f"{'' if v['coverage_ok'] else ' [커버리지부족]'}")
        dep = {k: v["delta_pooled"] for k, v in cell_out["baselines"].items()
               if v["deployable"] and v["coverage_ok"]}
        best_b, best_v = (max(dep.items(), key=lambda kv: kv[1])
                          if dep else (None, float("nan")))
        neg = {k: v["delta_pooled"] for k, v in cell_out["baselines"].items()
               if v["kind"] == "negative_control"}
        best_neg = max(neg.values()) if neg else float("nan")
        cell_out["best_deployable"] = dict(name=best_b, delta=None if best_b is None else best_v)
        cell_out["best_seed_only"] = None if not neg else float(best_neg)

        # ── record × layer ────────────────────────────────────────────
        for r in recs:
            for L in want_layers:
                if L not in layers:
                    print(f"[skip] layer {L} 없음")
                    continue
                li = layers.index(L)
                key = f"r{r}_L{L}"
                try:
                    A, Yr, *_ = to_matrix(eps, li, rec=r)
                except ValueError as e:
                    print(f"  [skip] {key}: {e}")
                    cell_out["runs"][key] = dict(record=r, layer=L, error=str(e))
                    continue
                assert np.array_equal(Yr, Y)
                print(f"\n  ── {key} (V = X[{r}, L{L}]) ──")

                pro = run_prospective(A, Y, folds, seeds, perms8, ex8, a.n_perm_label, rng)
                pooled = float(np.mean([v["delta"] for v in pro.values()]))
                for fname, v in pro.items():
                    print(f"    [prospective {fname}] Δ̂ {v['delta']:+.3f} "
                          f"top1 {v['sr_top1']:.3f} (base {v['sr_base']:.3f}, "
                          f"적중 {v['hits']}/{v['n_scene']}, 기대 {v['expected_hits']:.1f}) "
                          f"p_lab {v['p_labelshuffle']:.4f} p_colperm {v['p_seedperm']:.4f} "
                          f"[{v['sec']:.0f}s]")

                inf = run_infold(A, Y, np.arange(J), perms8, ex8, a.n_perm_label, rng)
                print(f"    [in-fold K=8] Δ̂ {inf['delta']:+.3f} top1 {inf['sr_top1']:.3f} "
                      f"(base {inf['sr_base']:.3f}, 적중 {inf['hits']}/{inf['n_scene']}, "
                      f"기대 {inf['expected_hits']:.1f}) worst1 {inf['sr_worst1']:.3f} "
                      f"p_lab {inf['p_labelshuffle']:.4f} p_colperm {inf['p_seedperm']:.4f} "
                      f"[{inf['sec']:.0f}s]")
                print(f"      선택분포: {inf['choice']['counts']} "
                      f"(distinct {inf['choice']['n_distinct']}/{J}, "
                      f"top col {inf['choice']['top_col']} {inf['choice']['top_frac']:.2f}, "
                      f"H {inf['choice']['entropy_nats']:.2f}/{inf['choice']['entropy_max']:.2f})")

                dr = run_infold(A[:, keep], Y[:, keep], keep, perms7, ex7,
                                a.n_perm_label, rng)
                print(f"    [drop col {a.drop_col} K=7] Δ̂ {dr['delta']:+.3f} "
                      f"top1 {dr['sr_top1']:.3f} (base {dr['sr_base']:.3f}, "
                      f"적중 {dr['hits']}/{dr['n_scene']}) "
                      f"p_lab {dr['p_labelshuffle']:.4f} p_colperm {dr['p_seedperm']:.4f} "
                      f"[{dr['sec']:.0f}s]")
                print(f"      선택분포: {dr['choice']['counts']} "
                      f"(distinct {dr['choice']['n_distinct']}/{len(keep)}, "
                      f"top col {dr['choice']['top_col']} {dr['choice']['top_frac']:.2f})")

                verdict = ("prospective Δ̂ ≤ 0" if pooled <= 0 else
                           f"배포가능 baseline({best_b} {best_v:+.3f}) 이하"
                           if dep and pooled <= best_v else
                           "prospective Δ̂ > 0 이고 최강 배포가능 baseline 초과")
                print(f"    ★ prospective pooled Δ̂ {pooled:+.3f} vs baseline "
                      f"{best_b} {best_v:+.3f} / seed_only {best_neg:+.3f} → {verdict}")
                cell_out["runs"][key] = dict(
                    record=r, layer=L, prospective=pro, prospective_pooled=pooled,
                    infold_K8=inf, drop_col=a.drop_col, drop_K7=dr, verdict=verdict)

        out["cells"][cell] = cell_out

    Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).expanduser().write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
