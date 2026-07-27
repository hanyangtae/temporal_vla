#!/usr/bin/env python3
"""exp5-4 Step 1 — **선행 반증 게이트** (사전 등록 go/no-go).

질문: t=0 활성으로 노이즈 draw 를 고르는 것이 (a) 앞을 내다보는 prospective 설정에서
살아남는가, (b) 자명한 배포가능 baseline 을 이기는가.

설계 (double out — scene-out × seed-out)
  · 8 seed 를 4+4 두 fold 로 분할. fold 마다 test seed 4개는 fit 에서 완전 배제.
  · 평가 scene 의 판은 전부 fit 에서 배제 (LOSO) → 방향은 19 scene × train seed 4 로만 fit.
  · 그 scene 의 held-out 4 후보 중 top-1 선택 → prospective K=4.

통계량 Δ̂ = mean_i (y_top1,i − 후보평균 SR_i), 전 20 scene ITT (전패/전승 포함).
검정 1: 관측 m_i 조건부 exact randomization (p_i = m_i^test/4 Poisson-binomial DP).
검정 2: 8 seed column 공통 순열 전수(8!=40320) — scene 별 독립 순열이 아니라
        모든 scene 에 같은 순열을 적용(이 데이터는 8 seed 를 공유하므로 정직한 null).
        두 검정이 갈리면 seed 주효과(특정 noise seed 가 원래 잘함) 증거.

판정: pooled Δ̂ ≤ 0 이거나 최강 배포가능 baseline 이하 → **Phase A 중단 권고**.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sel_common import (load, to_matrix, loso_select, score_select,  # noqa: E402
                         delta_stat, pb_pvalue, seed_permutations)
from _sel_baselines import (activation_norm_scores, seed_only_scores,  # noqa: E402
                            action_scores, DEPLOYABLE)


def pb_p(res):
    """관측 m_i 조건부 exact p (P(H ≥ h_obs))."""
    v = res["valid"]
    p_i = res["base"][v].astype(float)
    h = int(res["top1"][v].sum())
    p, _pmf = pb_pvalue(p_i, h)
    return p, h, float(p_i.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/home/kimseungjun/sm_npz")
    ap.add_argument("--cells", default="pq3_drawer_right,pq3_ppcc_beer,exp41_mixer")
    ap.add_argument("--layers", default="0", help="쉼표 구분, 기본 primary L0")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--n-perm-seed", type=int, default=40320, help="8!=40320 이면 전수")
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--chunk", type=int, default=5, help="첫 chunk 로 볼 csv 행 수")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--out", default="/home/kimseungjun/exp54_results/selection_gate.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    want_layers = [int(x) for x in a.layers.split(",")]
    out = dict(config=vars(a), cells={})

    for cell in a.cells.split(","):
        eps, layers = load(Path(a.npz_dir), cell)
        if not eps:
            print(f"[skip] {cell}: npz 없음")
            continue
        A0, Y, scenes, seeds, E = to_matrix(eps, 0, W=a.window)
        S, J = Y.shape
        print(f"\n{'='*90}\n{cell}: {S} scene × {J} seed = {S*J}판 · 전체 SR {Y.mean():.3f}"
              f" · m_i {Y.sum(1).tolist()}")
        cell_out = dict(n_scene=S, n_seed=J, sr_all=float(Y.mean()),
                        m_i=Y.sum(1).tolist(), seeds=seeds, folds={}, baselines={})

        folds = [("fold1", np.arange(J) < J // 2), ("fold2", np.arange(J) >= J // 2)]
        # fold 이름 = train 쪽 표기. test 는 여집합.
        perms, exhaustive = seed_permutations(J, a.n_perm_seed, rng)
        print(f"  seed column 순열: {len(perms)}개 ({'전수' if exhaustive else '부분표본'})")

        # ── baseline 점수 준비 (fold 무관, split 은 동일하게 적용)
        scores = {}
        if not a.no_baselines:
            act_scores, note = action_scores(cell, E, chunk=a.chunk)
            print(f"  csv baseline: {note}")
            cell_out["csv_note"] = note
            scores.update(act_scores)
            scores["seed_only"] = seed_only_scores(seeds, S)

        for L in want_layers:
            li = layers.index(L)
            A, _Y2, _s, _sd, _E = to_matrix(eps, li, W=a.window)
            if not a.no_baselines:
                scores[f"act_norm_L{L}"] = activation_norm_scores(A)

            for fname, train_mask in folds:
                test_mask = ~train_mask
                res = loso_select(A, Y, fit_seeds=train_mask, eval_seeds=test_mask)
                st = delta_stat(res)
                p_pb, h_obs, exp_h = pb_p(res)

                # seed column 공통 순열 null
                t0 = time.time()
                null = np.empty(len(perms))
                for k, pi in enumerate(perms):
                    Yp = Y[:, list(pi)]
                    r = loso_select(A, Yp, fit_seeds=train_mask, eval_seeds=test_mask)
                    null[k] = delta_stat(r)["delta"]
                p_seed = float((np.sum(null >= st["delta"]) + 1) / (len(perms) + 1))
                sec = time.time() - t0

                key = f"L{L}_{fname}"
                cell_out["folds"][key] = dict(
                    layer=L, fold=fname,
                    train_seeds=[seeds[i] for i in np.where(train_mask)[0]],
                    test_seeds=[seeds[i] for i in np.where(test_mask)[0]],
                    **st, p_exact_pb=p_pb, expected_hits=exp_h,
                    p_seedperm=p_seed, n_perm=len(perms), perm_exhaustive=exhaustive,
                    null_seedperm_mean=float(null.mean()), null_seedperm_sd=float(null.std()),
                    null_seedperm_q=[float(q) for q in np.percentile(null, [5, 50, 95])],
                    sec=round(sec, 1))
                print(f"  [{key}] 학습축  Δ̂ {st['delta']:+.3f}  top1 {st['sr_top1']:.3f} "
                      f"(base {st['sr_base']:.3f}, 적중 {st['hits']}/{st['n_scene']}, "
                      f"기대 {exp_h:.1f})  worst1 {st['sr_worst1']:.3f}  "
                      f"p_exact {p_pb:.4f}  p_seedperm {p_seed:.4f}  [{sec:.0f}s]")

        # ── baseline: 동일 split
        for bname, sc in sorted(scores.items()):
            for asc in (True, False):
                arr = []
                for fname, train_mask in folds:
                    test_mask = ~train_mask
                    r = score_select(sc, Y, eval_seeds=test_mask, ascending=asc)
                    st = delta_stat(r)
                    p_pb, h_obs, exp_h = pb_p(r)
                    arr.append(dict(fold=fname, **st, p_exact_pb=p_pb))
                d = float(np.mean([x["delta"] for x in arr]))
                tag = "낮은쪽선택" if asc else "높은쪽선택"
                cell_out["baselines"][f"{bname}|{tag}"] = dict(
                    delta_pooled=d, folds=arr,
                    deployable=bool(bname.startswith("act_norm") or bname in DEPLOYABLE),
                    negative_control=bool(bname == "seed_only"))
                print(f"  [baseline] {bname:22} {tag}  Δ̂(fold평균) {d:+.3f}  "
                      + " ".join(f"{x['fold']}:{x['delta']:+.3f}(p{x['p_exact_pb']:.3f})"
                                 for x in arr))

        # ── pooled 판정
        Lp = want_layers[0]
        learned = [cell_out["folds"][f"L{Lp}_{f}"]["delta"] for f, _ in folds]
        learned_pooled = float(np.mean(learned))
        dep = {k: v["delta_pooled"] for k, v in cell_out["baselines"].items()
               if v["deployable"]}
        best_b, best_v = (max(dep.items(), key=lambda kv: kv[1]) if dep else (None, float("nan")))
        verdict = ("중단권고: prospective Δ̂ ≤ 0" if learned_pooled <= 0 else
                   f"중단권고: 배포가능 baseline({best_b}) 이하" if dep and learned_pooled <= best_v
                   else "통과: prospective Δ̂ > 0 이고 최강 배포가능 baseline 초과")
        cell_out["verdict"] = dict(primary_layer=Lp, learned_delta_pooled=learned_pooled,
                                   best_deployable_baseline=best_b,
                                   best_deployable_delta=None if best_b is None else best_v,
                                   text=verdict)
        print(f"  ★판정 (L{Lp}): 학습축 Δ̂ {learned_pooled:+.3f} vs 최강 배포가능 "
              f"baseline {best_b} {best_v:+.3f} → {verdict}")
        print("   (baseline 미달이어도 '기여 0' 이 아니라 'activation 고유 기여 근거 없음' 으로 읽는다)")
        out["cells"][cell] = cell_out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
