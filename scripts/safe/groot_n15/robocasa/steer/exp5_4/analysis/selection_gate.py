#!/usr/bin/env python3
"""exp5-4 Step 1 — **선행 반증 게이트** (사전 등록 go/no-go).

질문: t=0 활성으로 노이즈 draw 를 고르는 것이 (a) 앞을 내다보는 prospective 설정에서
살아남는가, (b) 자명한 배포가능 baseline 을 이기는가.

설계 (double out — scene-out × seed-out)
  · 8 seed 를 4+4 두 fold 로 분할. fold 마다 test seed 4개는 fit 에서 완전 배제.
  · 평가 scene 의 판은 전부 fit 에서 배제 (LOSO) → 방향은 19 scene × train seed 4 로만 fit.
  · 그 scene 의 held-out 4 후보 중 top-1 선택 → prospective K=4.

통계량 Δ̂ = mean_i (y_top1,i − 후보평균 SR_i), 전 20 scene ITT (전패/전승 포함).

검정 (★2026-07-28 정정 — Codex Gate2 리뷰)
  · **학습 선택자(LOSO)** 는 방향이 다른 scene 의 라벨에 의존 → scene 별 적중이 독립이
    아니므로 Poisson-binomial "exact" 는 성립하지 않는다. 학습 경로의 primary 검정은
    (i) scene 내 라벨셔플 + **전체 재fit** null, (ii) 8 seed column 공통 순열(전수 8!)
    + 재fit null 두 가지 순열검정이다. PB 값은 참고로만 찍고 `pb_valid=false` 로 표시.
  · **고정 선택자(baseline 점수)** 는 라벨과 무관하게 순위가 정해지므로 PB exact 가 유효
    (`pb_valid=true`). baseline 에도 seed column 순열 p 를 병기한다.

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
                            pkl_pass, load_handle_tsv, DEPLOYABLE)


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


def pb_p(res):
    """Poisson-binomial exact p — 고정 선택자에서만 유효."""
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
    ap.add_argument("--n-perm-label", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--pkl-cache-dir", default="/home/kimseungjun/exp54_results/pkl_cache")
    ap.add_argument("--with-oracle", action="store_true",
                    help="손잡이 기하 privileged oracle 포함 (handle tsv 에 있는 scene 만)")
    ap.add_argument("--handle-tsv", default="/home/kimseungjun/exp53_analysis/handle_all.tsv")
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
        _A0, Y, scenes, seeds, E = to_matrix(eps, 0, W=a.window)
        S, J = Y.shape
        print(f"\n{'='*94}\n{cell}: {S} scene × {J} seed = {S*J}판 · 전체 SR {Y.mean():.3f}"
              f" · m_i {Y.sum(1).tolist()}")
        cell_out = dict(n_scene=S, n_seed=J, sr_all=float(Y.mean()),
                        m_i=Y.sum(1).tolist(), seeds=seeds, folds={}, baselines={})

        folds = [("fold1", np.arange(J) < J // 2), ("fold2", np.arange(J) >= J // 2)]
        perms, exhaustive = seed_permutations(J, a.n_perm_seed, rng)
        print(f"  seed column 순열 {len(perms)}개({'전수' if exhaustive else '부분표본'})"
              f" · 라벨셔플 재fit {a.n_perm_label}회")

        scores = {}
        if not a.no_baselines:
            geom = (load_handle_tsv(a.handle_tsv)
                    if (a.with_oracle and Path(a.handle_tsv).expanduser().exists()) else None)
            ch, oracle, diag = pkl_pass(cell, E, scenes, geom=geom,
                                        cache=f"{a.pkl_cache_dir}/{cell}_chunk.npz")
            print(f"  pkl chunk baseline: {diag['n_ok']}/{diag['n_total']}판"
                  f" (결측 {diag['n_miss']}) · 출처 {diag['source']}"
                  + (f" · 평균 oracle cos {diag['mean_oracle_cos']:.3f}"
                     if diag.get("mean_oracle_cos") is not None else ""))
            cell_out["pkl_note"] = {k: v for k, v in diag.items() if k != "cached"}
            scores.update(ch)
            scores["seed_only"] = seed_only_scores(seeds, S)
            if geom is not None and np.isfinite(oracle).any():
                scores["oracle_handle_cos"] = oracle

        for L in want_layers:
            li = layers.index(L)
            A, *_ = to_matrix(eps, li, W=a.window)
            if not a.no_baselines:
                scores[f"act_norm_L{L}"] = activation_norm_scores(A)

            for fname, train_mask in folds:
                test_mask = ~train_mask
                sel = (lambda Yp, tr=train_mask, te=test_mask, AA=A:
                       loso_select(AA, Yp, fit_seeds=tr, eval_seeds=te))
                res = sel(Y)
                st = delta_stat(res)
                p_pb, h_obs, exp_h = pb_p(res)

                t0 = time.time()
                p_seed, nullS = seedperm_null(sel, Y, perms, st["delta"])
                p_lab, nullL = labelshuffle_null(sel, Y, a.n_perm_label, rng, st["delta"])
                sec = time.time() - t0

                key = f"L{L}_{fname}"
                cell_out["folds"][key] = dict(
                    layer=L, fold=fname,
                    train_seeds=[seeds[i] for i in np.where(train_mask)[0]],
                    test_seeds=[seeds[i] for i in np.where(test_mask)[0]],
                    **st, expected_hits=exp_h,
                    p_labelshuffle=p_lab, p_seedperm=p_seed,
                    p_exact_pb_reference_only=p_pb, pb_valid=False,
                    n_perm_seed=len(perms), perm_exhaustive=exhaustive,
                    n_perm_label=a.n_perm_label,
                    null_seedperm=[float(nullS.mean()), float(nullS.std())],
                    null_labelshuffle=[float(nullL.mean()), float(nullL.std())],
                    sec=round(sec, 1))
                print(f"  [{key}] 학습축 Δ̂ {st['delta']:+.3f} top1 {st['sr_top1']:.3f}"
                      f" (base {st['sr_base']:.3f}, 적중 {st['hits']}/{st['n_scene']},"
                      f" 기대 {exp_h:.1f}) worst1 {st['sr_worst1']:.3f}"
                      f"  p_라벨셔플 {p_lab:.4f}  p_seedperm {p_seed:.4f}"
                      f"  (PB {p_pb:.4f} — 학습선택자엔 부적합) [{sec:.0f}s]")

        # ── baseline: 동일 split (고정 선택자 → PB 유효)
        for bname, sc in sorted(scores.items()):
            for asc in (True, False):
                arr = []
                for fname, train_mask in folds:
                    test_mask = ~train_mask
                    fn = (lambda Yp, te=test_mask, s=sc, aa=asc:
                          score_select(s, Yp, eval_seeds=te, ascending=aa))
                    r = fn(Y)
                    st = delta_stat(r)
                    p_pb, h_obs, exp_h = pb_p(r)
                    p_sp, _n = seedperm_null(fn, Y, perms, st["delta"])
                    arr.append(dict(fold=fname, **st, p_exact_pb=p_pb, pb_valid=True,
                                    p_seedperm=p_sp))
                d = float(np.mean([x["delta"] for x in arr]))
                tag = "낮은쪽선택" if asc else "높은쪽선택"
                cov_ok = all(x["n_scene"] == S for x in arr)
                kind = ("negative_control" if bname == "seed_only" else
                        "oracle" if bname.startswith("oracle") else
                        "deployable" if (bname in DEPLOYABLE or bname.startswith("act_norm"))
                        else "other")
                cell_out["baselines"][f"{bname}|{tag}"] = dict(
                    delta_pooled=d, folds=arr, coverage_ok=cov_ok, kind=kind,
                    deployable=bool(kind == "deployable"))
                print(f"  [baseline] {bname:20} {tag} {kind:16} Δ̂ {d:+.3f}"
                      f"{'' if cov_ok else ' [커버리지부족 판정제외]'}  "
                      + " ".join(f"{x['fold']}:{x['delta']:+.3f}"
                                 f"(PB{x['p_exact_pb']:.3f}/sp{x['p_seedperm']:.3f})"
                                 for x in arr))

        Lp = want_layers[0]
        learned = [cell_out["folds"][f"L{Lp}_{f}"]["delta"] for f, _ in folds]
        learned_pooled = float(np.mean(learned))
        dep = {k: v["delta_pooled"] for k, v in cell_out["baselines"].items()
               if v["deployable"] and v["coverage_ok"]}
        best_b, best_v = (max(dep.items(), key=lambda kv: kv[1]) if dep else (None, float("nan")))
        verdict = ("중단권고: prospective Δ̂ ≤ 0" if learned_pooled <= 0 else
                   f"중단권고: 배포가능 baseline({best_b}) 이하" if dep and learned_pooled <= best_v
                   else "통과: prospective Δ̂ > 0 이고 최강 배포가능 baseline 초과")
        cell_out["verdict"] = dict(primary_layer=Lp, learned_delta_pooled=learned_pooled,
                                   best_deployable_baseline=best_b,
                                   best_deployable_delta=None if best_b is None else best_v,
                                   text=verdict)
        print(f"  ★판정 (L{Lp}): 학습축 Δ̂ {learned_pooled:+.3f} vs 최강 배포가능 baseline "
              f"{best_b} {best_v:+.3f} → {verdict}")
        print("   (baseline 미달 = '기여 0' 이 아니라 'activation 고유 기여 근거 없음')")
        out["cells"][cell] = cell_out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
