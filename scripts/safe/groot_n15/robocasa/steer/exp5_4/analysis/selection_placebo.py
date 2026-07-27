#!/usr/bin/env python3
"""exp5-4 Step 1 — 선택(top-1) 성능의 귀무분포 (in-fold, K=8 전체 후보).

기존 분석에는 AUROC 위약만 있고 **선택 성능 자체의 귀무**가 없었다. 여기서 셋을 만든다.

primary 통계량 Δ̂ = mean_i (y_top1,i − m_i/8), 전 20 scene ITT (전패/전승 포함).

  · 검정 A (exact randomization) : 관측 m_i 조건부. 무작위 선택이면 scene i 적중확률
        p_i = m_i/8 → 총 적중수 H 의 Poisson-binomial DP 정확 p. 셔플 불필요.
  · 검정 B (seed column 공통 순열) : 이 데이터는 20 scene 이 **같은 8 seed 를 공유**하므로
        scene 별 독립 순열은 seed 주효과를 지워버린다. 모든 scene 에 같은 순열을 가하는
        8!=40320 전수 순열이 정직한 null. A 와 B 가 갈리면 seed 주효과 증거.
  · 검정 C (라벨셔플 재fit, secondary) : scene 내 라벨 셔플 후 **방향 fit 부터 재실행**.

diagnostic: 혼재 scene top-1 적중수, worst-1 SR(음성 대조).
primary = layer L0, 나머지 6 layer 는 부표.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sel_common import (load, to_matrix, loso_select, delta_stat,  # noqa: E402
                         pb_pvalue, seed_permutations)


def shuffle_within_scene(Y, rng):
    Yp = Y.copy()
    for i in range(Y.shape[0]):
        Yp[i] = rng.permutation(Y[i])
    return Yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/home/kimseungjun/sm_npz")
    ap.add_argument("--cells", default="pq3_drawer_right,pq3_ppcc_beer,exp41_mixer")
    ap.add_argument("--primary-layer", type=int, default=0)
    ap.add_argument("--n-perm-seed", type=int, default=40320, help="primary layer 전수")
    ap.add_argument("--n-perm-seed-secondary", type=int, default=2000)
    ap.add_argument("--n-perm-label", type=int, default=500)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--out", default="/home/kimseungjun/exp54_results/selection_placebo.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = dict(config=vars(a), cells={})

    for cell in a.cells.split(","):
        eps, layers = load(Path(a.npz_dir), cell)
        if not eps:
            print(f"[skip] {cell}: npz 없음")
            continue
        _A0, Y, scenes, seeds, _E = to_matrix(eps, 0, W=a.window)
        S, J = Y.shape
        m = Y.sum(1)
        print(f"\n{'='*100}\n{cell}: {S} scene × {J} draw = {S*J}판 · 전체 SR {Y.mean():.3f}"
              f" · 혼재 scene {int(((m>0)&(m<J)).sum())} · layers {layers}")
        cell_out = dict(n_scene=S, n_seed=J, sr_all=float(Y.mean()), m_i=m.tolist(),
                        layers=layers, per_layer={})
        print(f"{'layer':>6} {'base':>7} {'top1':>7} {'Δ̂':>8} {'적중/기대':>12} "
              f"{'p_exact':>8} {'p_seedperm':>11} {'p_라벨셔플':>11} "
              f"{'혼재적중':>9} {'worst1':>7}")

        for li, L in enumerate(layers):
            A, _Y, _s, _sd, _E2 = to_matrix(eps, li, W=a.window)
            res = loso_select(A, Y)
            st = delta_stat(res)
            v = res["valid"]
            p_i = res["base"][v].astype(float)
            h_obs = int(res["top1"][v].sum())
            p_exact, _pmf = pb_pvalue(p_i, h_obs)
            mixed_hits = int(res["top1"][v & ((m > 0) & (m < J))].sum())
            mixed_n = int((v & ((m > 0) & (m < J))).sum())

            # 검정 B — seed column 공통 순열
            n_seed_perm = a.n_perm_seed if L == a.primary_layer else a.n_perm_seed_secondary
            perms, exhaustive = seed_permutations(J, n_seed_perm, rng)
            t0 = time.time()
            nullB = np.empty(len(perms))
            for k, pi in enumerate(perms):
                nullB[k] = delta_stat(loso_select(A, Y[:, list(pi)]))["delta"]
            pB = float((np.sum(nullB >= st["delta"]) + 1) / (len(perms) + 1))
            secB = time.time() - t0

            # 검정 C — scene 내 라벨셔플 재fit (secondary)
            t1 = time.time()
            nullC = np.empty(a.n_perm_label)
            for k in range(a.n_perm_label):
                nullC[k] = delta_stat(loso_select(A, shuffle_within_scene(Y, rng)))["delta"]
            pC = float((np.sum(nullC >= st["delta"]) + 1) / (a.n_perm_label + 1))
            secC = time.time() - t1

            tag = " ★primary" if L == a.primary_layer else ""
            print(f"  L{L:<4} {st['sr_base']:7.3f} {st['sr_top1']:7.3f} {st['delta']:+8.3f} "
                  f"{h_obs:4d}/{p_i.sum():6.1f} {p_exact:8.4f} {pB:11.4f} {pC:11.4f} "
                  f"{mixed_hits:4d}/{mixed_n:<4d} {st['sr_worst1']:7.3f}{tag}")

            cell_out["per_layer"][str(L)] = dict(
                observed=dict(**st, expected_hits=float(p_i.sum()),
                              mixed_hits=mixed_hits, mixed_n=mixed_n,
                              chosen_seed_idx=res["chosen"].tolist(),
                              per_scene_top1=res["top1"].tolist(),
                              per_scene_m=m.tolist()),
                testA_exact=dict(p=float(p_exact), method="Poisson-binomial DP, p_i=m_i/8"),
                testB_seedperm=dict(p=pB, n_perm=len(perms), exhaustive=exhaustive,
                                    null_mean=float(nullB.mean()), null_sd=float(nullB.std()),
                                    null_q=[float(q) for q in np.percentile(nullB, [5, 50, 95])],
                                    sec=round(secB, 1)),
                testC_labelshuffle=dict(p=pC, n_perm=a.n_perm_label,
                                        null_mean=float(nullC.mean()),
                                        null_sd=float(nullC.std()),
                                        null_q=[float(q) for q in
                                                np.percentile(nullC, [5, 50, 95])],
                                        sec=round(secC, 1)),
            )
        out["cells"][cell] = cell_out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
