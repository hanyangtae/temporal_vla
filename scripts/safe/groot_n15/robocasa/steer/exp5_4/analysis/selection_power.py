#!/usr/bin/env python3
"""exp5-4 Step 1 — 검정력(power) 정확계산: "160판(20 scene × 8 draw)으로 충분한가".

모형: selector 가 혼재 scene 에서 top-1 을 성공 후보에 놓을 확률 q (격자 0.6/0.7/0.8).
      전패/전승 scene 은 선택과 무관하게 결과가 강제된다(ITT 에 그대로 포함).
귀무: 후보를 무작위로 하나 뽑음 → scene i 성공확률 p_i = m_i/K.
검정: 총 적중수 H 의 Poisson-binomial DP exact, 단측 α.

설계 2종
  · K=8  : 전체 8 draw 가 후보 (기존 상한 계산과 같은 in-fold 설정)
  · K=4  : gate 의 prospective 설정 (test seed 4개만 후보) — fold 별 m_i^test 사용
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sel_common import load, to_matrix, poisson_binomial_pmf  # noqa: E402


def crit_and_power(p_null, p_alt, alpha):
    pmf0 = poisson_binomial_pmf(p_null)
    tail0 = np.cumsum(pmf0[::-1])[::-1]           # tail0[h] = P(H>=h)
    h_star = int(np.argmax(tail0 <= alpha))       # 최초로 α 이하가 되는 h
    if tail0[h_star] > alpha:
        return None, 0.0, float(tail0[h_star])
    pmf1 = poisson_binomial_pmf(p_alt)
    tail1 = np.cumsum(pmf1[::-1])[::-1]
    return h_star, float(tail1[h_star]), float(tail0[h_star])


def alt_probs(m, K, q):
    """혼재 scene 은 q, 전패=0, 전승=1."""
    out = []
    for mi in m:
        if mi == 0:
            out.append(0.0)
        elif mi == K:
            out.append(1.0)
        else:
            out.append(q)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/home/kimseungjun/sm_npz")
    ap.add_argument("--cells", default="pq3_drawer_right,pq3_ppcc_beer,exp41_mixer")
    ap.add_argument("--q-grid", default="0.6,0.7,0.8")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--scene-multipliers", default="1,2,3", help="scene 수 배수 외삽")
    ap.add_argument("--out", default="/home/kimseungjun/exp54_results/selection_power.json")
    a = ap.parse_args()

    qs = [float(x) for x in a.q_grid.split(",")]
    mults = [int(x) for x in a.scene_multipliers.split(",")]
    out = dict(config=vars(a), cells={})

    for cell in a.cells.split(","):
        eps, _layers = load(Path(a.npz_dir), cell)
        if not eps:
            print(f"[skip] {cell}")
            continue
        _A, Y, scenes, seeds, _E = to_matrix(eps, 0)
        S, J = Y.shape
        m8 = Y.sum(1)
        designs = {"K=8(전체후보)": [(m8, J)]}
        half = J // 2
        designs["K=4(prospective fold평균)"] = [
            (Y[:, :half].sum(1), half), (Y[:, half:].sum(1), half)]

        print(f"\n{'='*84}\n{cell}: {S} scene × {J} draw · SR {Y.mean():.3f} · m_i {m8.tolist()}")
        cell_out = {}
        for dname, parts in designs.items():
            rows = {}
            for mult in mults:
                for q in qs:
                    pw, crits = [], []
                    for m, K in parts:
                        m_rep = np.tile(m, mult)
                        p0 = m_rep / K
                        p1 = alt_probs(m_rep, K, q)
                        h, power, ptail = crit_and_power(p0, p1, a.alpha)
                        pw.append(power); crits.append(h)
                    rows[f"S={S*mult},q={q}"] = dict(
                        power=float(np.mean(pw)), h_crit=crits,
                        n_ep=int(S * mult * J),
                        mixed=int(sum(int(((mm > 0) & (mm < K)).sum()) * mult
                                      for mm, K in parts) / len(parts)))
            cell_out[dname] = rows
            print(f"  [{dname}]")
            for k, v in rows.items():
                print(f"    {k:16} power {v['power']:.3f}  (임계 적중수 {v['h_crit']}, "
                      f"판수 {v['n_ep']}, 혼재 scene {v['mixed']})")
        out["cells"][cell] = cell_out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
