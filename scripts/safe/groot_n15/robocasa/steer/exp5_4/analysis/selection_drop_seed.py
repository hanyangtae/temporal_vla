#!/usr/bin/env python3
"""exp5-4 후속 ablation — seed column 을 통째로 제거해도 t=0 선택이 되는가.

exp5-4 종결 판정: 13/13 적중은 seed 주효과(전역적으로 잘 되는 draw column 이 존재)로
설명 가능했다 (seed-column 공통 순열 p=.60, seed_only 대조 동률). 이 스크립트는 그
주효과의 **직접 제거 실험**이다.

  1. 제거 전 per-seed column SR 표 (J column × S scene) — "seed 0만 성공"인지 실측.
  2. `--drop` 으로 지정한 column 의 episode 를 **fit·평가 양쪽에서 완전 제거** 후
     표준 in-fold LOSO 선택(평가 scene 제외 방향 fit → record0 사영 → top-1) 재실행.
     · base(잔여 draw 평균 SR), top1 SR, Δ̂, worst1
     · 선택 분포 (scene 별 chosen seed — 또 한 column 에 몰리는가)
     · null 2종: scene-내 라벨셔플 재fit / 잔여 column 공통 순열 (J'! 전수)
     · seed_only 음성 대조(잔여 중 가장 낮은 seed 고정 선택) + 혼재 scene 적중률
  3. drop 세트는 기본 자동: {} (제거 전), {seed idx 0}, {최고 SR column},
     {상위 2 column}. `--drop-sets` 로 override ("-" = 제거 없음, "0" , "0+3" …).

primary = layer L0 (record 0 = t=0 활성), 부표 L12.
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
                         score_select, seed_permutations)


def shuffle_within_scene(Y, rng):
    Yp = Y.copy()
    for i in range(Y.shape[0]):
        Yp[i] = rng.permutation(Y[i])
    return Yp


def run_one(A, Y, seeds_kept, orig_idx, n_perm_label, n_perm_seed, rng, verbose_tag=""):
    """제거 후 행렬 [S,J']에 대한 선택 + null 2종. 반환 dict."""
    S, J = Y.shape
    m = Y.sum(1)
    mixed = (m > 0) & (m < J)

    res = loso_select(A, Y)
    st = delta_stat(res)
    v = res["valid"]
    h_obs = int(res["top1"][v].sum())
    exp_hits = float(res["base"][v].sum())

    # 선택 분포 (원래 seed index 기준)
    chosen_orig = [int(orig_idx[c]) for c in res["chosen"]]
    counts = {str(int(orig_idx[j])): int(np.sum(res["chosen"] == j)) for j in range(J)}

    # null B — 잔여 column 공통 순열 (J'! 전수 가능하면 전수)
    perms, exhaustive = seed_permutations(J, n_perm_seed, rng)
    t0 = time.time()
    nullB = np.empty(len(perms))
    for k, pi in enumerate(perms):
        nullB[k] = delta_stat(loso_select(A, Y[:, list(pi)]))["delta"]
    pB = float((np.sum(nullB >= st["delta"]) + 1) / (len(perms) + 1))
    secB = time.time() - t0

    # null C — scene 내 라벨셔플 재fit (primary)
    t1 = time.time()
    nullC = np.empty(n_perm_label)
    for k in range(n_perm_label):
        nullC[k] = delta_stat(loso_select(A, shuffle_within_scene(Y, rng)))["delta"]
    pC = float((np.sum(nullC >= st["delta"]) + 1) / (n_perm_label + 1))
    secC = time.time() - t1

    # seed_only 음성 대조 — 잔여 중 가장 낮은 seed 를 항상 선택
    so = score_select(np.tile(np.asarray(seeds_kept, float)[None, :], (S, 1)), Y)
    so_st = delta_stat(so)

    mixed_hits = int(res["top1"][v & mixed].sum())
    mixed_n = int((v & mixed).sum())

    return dict(
        n_scene=S, n_seed=J, seeds_kept=[int(s) for s in seeds_kept],
        orig_seed_idx=[int(i) for i in orig_idx],
        sr_all=float(Y.mean()), m_i=m.tolist(), n_mixed=int(mixed.sum()),
        observed=dict(**st, expected_hits=exp_hits, hits=h_obs,
                      mixed_hits=mixed_hits, mixed_n=mixed_n,
                      chosen_orig_seed_idx=chosen_orig,
                      chosen_counts_by_orig_idx=counts,
                      per_scene_top1=res["top1"].tolist()),
        seed_only=dict(sr_top1=so_st["sr_top1"], delta=so_st["delta"],
                       hits=so_st["hits"],
                       mixed_hits=int(so["top1"][so["valid"] & mixed].sum()),
                       mixed_n=int((so["valid"] & mixed).sum())),
        testB_seedperm=dict(p=pB, n_perm=len(perms), exhaustive=exhaustive,
                            null_mean=float(nullB.mean()), null_sd=float(nullB.std()),
                            null_q=[float(q) for q in np.percentile(nullB, [5, 50, 95])],
                            sec=round(secB, 1)),
        testC_labelshuffle_PRIMARY=dict(p=pC, n_perm=n_perm_label,
                                        null_mean=float(nullC.mean()),
                                        null_sd=float(nullC.std()),
                                        null_q=[float(q) for q in
                                                np.percentile(nullC, [5, 50, 95])],
                                        sec=round(secC, 1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/home/kimseungjun/sm_npz")
    ap.add_argument("--cells", default="pq3_drawer_right")
    ap.add_argument("--layers", default="0,12", help="분석 layer (primary=첫번째)")
    ap.add_argument("--drop-sets", default="auto",
                    help="'auto' 또는 '-,0,0+3' 형식 ('-'=제거없음)")
    ap.add_argument("--n-perm-seed", type=int, default=40320)
    ap.add_argument("--n-perm-label", type=int, default=2000)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--out", default="/home/kimseungjun/exp54_results/selection_drop_seed.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    want_layers = [int(x) for x in a.layers.split(",")]
    out = dict(config=vars(a), cells={})

    for cell in a.cells.split(","):
        eps, layers = load(Path(a.npz_dir), cell)
        if not eps:
            print(f"[skip] {cell}: npz 없음")
            continue
        _A0, Y0, scenes, seeds0, _E = to_matrix(eps, 0, W=a.window)
        S, J0 = Y0.shape
        col_succ = Y0.sum(0)
        col_sr = col_succ / S
        order = np.argsort(-col_sr)          # SR 내림차순

        print(f"\n{'='*100}\n{cell}: {S} scene × {J0} draw = {S*J0}판 · 전체 SR {Y0.mean():.3f}"
              f" · layers {layers}")
        print("\n[1] 제거 전 per-seed column SR")
        print(f"  {'col':>4} {'seed':>10} {'성공/판':>9} {'SR':>7}")
        for j in range(J0):
            print(f"  {j:>4} {seeds0[j]:>10} {int(col_succ[j]):>4}/{S:<4} {col_sr[j]:7.3f}")
        print(f"  SR 내림차순 column: {order.tolist()} "
              f"(최고 col {int(order[0])}, SR {col_sr[order[0]]:.3f})")

        # drop 세트 결정
        if a.drop_sets == "auto":
            drops = [(), (0,), (int(order[0]),), (int(order[0]), int(order[1]))]
            uniq, seen = [], set()
            for d in drops:
                k = tuple(sorted(d))
                if k not in seen:
                    seen.add(k)
                    uniq.append(k)
            drops = uniq
        else:
            drops = []
            for tok in a.drop_sets.split(","):
                tok = tok.strip()
                drops.append(() if tok == "-" else
                             tuple(sorted(int(x) for x in tok.split("+"))))

        cell_out = dict(n_scene=S, n_seed=J0, seeds=[int(s) for s in seeds0],
                        sr_all=float(Y0.mean()), layers=layers,
                        per_column=dict(succ=col_succ.tolist(), sr=col_sr.tolist()),
                        sr_desc_order=order.tolist(), runs={})

        for L in want_layers:
            if L not in layers:
                print(f"[skip] layer {L} 없음")
                continue
            li = layers.index(L)
            A0, _Y, _s, _sd, _E2 = to_matrix(eps, li, W=a.window)
            print(f"\n[2] L{L} — drop 별 결과")
            print(f"  {'drop':>10} {'J':>3} {'base':>7} {'top1':>7} {'Δ̂':>8} "
                  f"{'적중/기대':>12} {'p_라벨셔플':>11} {'p_colperm':>10} "
                  f"{'혼재적중':>9} {'worst1':>7} {'seedonly':>9}")
            for d in drops:
                keep = np.array([j for j in range(J0) if j not in d])
                A = A0[:, keep]
                Y = Y0[:, keep]
                r = run_one(A, Y, [seeds0[j] for j in keep], keep,
                            a.n_perm_label, a.n_perm_seed, rng)
                key = f"L{L}|drop={'-' if not d else '+'.join(map(str,d))}"
                cell_out["runs"][key] = dict(layer=L, dropped=list(d), **r)
                ob, so = r["observed"], r["seed_only"]
                print(f"  {('-' if not d else '+'.join(map(str,d))):>10} {r['n_seed']:>3} "
                      f"{ob['sr_base']:7.3f} {ob['sr_top1']:7.3f} {ob['delta']:+8.3f} "
                      f"{ob['hits']:4d}/{ob['expected_hits']:6.1f} "
                      f"{r['testC_labelshuffle_PRIMARY']['p']:11.4f} "
                      f"{r['testB_seedperm']['p']:10.4f} "
                      f"{ob['mixed_hits']:4d}/{ob['mixed_n']:<4d} {ob['sr_worst1']:7.3f} "
                      f"{so['sr_top1']:9.3f}")
                cnt = ob["chosen_counts_by_orig_idx"]
                print(f"      선택분포(원 col→횟수): "
                      f"{ {k: v for k, v in cnt.items() if v} }")
        out["cells"][cell] = cell_out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
