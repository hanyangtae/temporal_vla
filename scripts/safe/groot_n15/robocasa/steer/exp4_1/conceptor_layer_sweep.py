#!/usr/bin/env python3
"""exp4-1: A(conceptor) arm 의 layer 선정 — 분산(2차 모멘트) 차이 기준 (2026-07-22 사용자 결정).

setM 이 평균 분리도(사영 AUROC)로 layer 를 고르는 것과 대칭으로, conceptor 는 자기 연산자가
보는 것(클래스별 2차 모멘트 부분공간 차이)으로 고른다:
  layer 점수 = held-out record 에서의 C_steer R-가중 이득
               gain(C, X) = E[xᵀCx] / E[xᵀx]  (X = held-out **success** records)
  기준선 = episode-level 라벨순열 N_PERM 개로 같은 절차 → null 분포 → z-score.
α 는 α₀=10 고정 (exp3 Stage1 layer-선택 관례, COAST A.10.2). denoise 는 pool (선정용 —
배포 fit 은 fit_phase_conceptor_n15.py --denoise per_step 로 별도, exp3 배포 규약 유지).
truncation cap 은 setM fit 과 동일 (성공 길이 ceil(μ+1σ)) — 길이 confound 통제 일관.

알려진 상태: 자연실패 conceptor 는 퇴화(이득 ~0.006, memory conceptor-saturation-degenerate).
전 layer 퇴화(z≈0)면 그 자체를 기록하고 최대-점수 layer 를 선정 (A 는 legacy 참조선 역할).

사용 (승준):
  python conceptor_layer_sweep.py --manifest <task_PPCC_fit.tsv> --cell pq3_ppcc_bread \
    --out <...>/npz/pq3_ppcc_bread/conceptor_layer_sweep.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.conceptor import and_conceptor, compute_conceptor, not_conceptor  # noqa: E402
from fit_mean_diff import episode_records, gather, load_cell_rolls  # noqa: E402

# α 저역 그리드 (공유문서 §3: "α 그리드는 0.3 아래로 확장, C_succ 이득이 1에 붙으면 포화") —
# 실측 활성 스케일에선 α≥1 에서 C≈I 포화 → AND-NOT ≈0 (합성 재현 확인). layer 마다
# C_succ held-out 이득이 SUCC_BAND 에 드는 α 를 먼저 고르고 그 α 에서 C_steer 를 채점.
ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
SUCC_BAND = (0.5, 0.95)
N_PERM = 20
HELD_FRAC = 0.2
RNG_SEED = 424102


def c_steer(Xs: np.ndarray, Xf: np.ndarray, alpha: float) -> np.ndarray:
    return and_conceptor(
        compute_conceptor(Xs, alpha), not_conceptor(compute_conceptor(Xf, alpha))
    )


def gain(C: np.ndarray, X: np.ndarray) -> float:
    """R-가중 이득: E[xᵀCx]/E[xᵀx] — 07-20 산술 검사 지표의 스크립트화 (공유문서 §3)."""
    num = float(np.einsum("ni,ij,nj->", X, C, X))
    den = float(np.einsum("ni,ni->", X, X))
    return num / den if den > 0 else float("nan")


def split_data(rolls, labels, li: int, cap: int, rng):
    """episode 80/20 split → (Xs_tr, Xf_tr, X_held_succ) 또는 None."""
    idx_s = [i for i in range(len(rolls)) if labels[i] == 1]
    idx_f = [i for i in range(len(rolls)) if labels[i] == 0]
    if len(idx_s) < 3 or len(idx_f) < 2:
        return None
    rng.shuffle(idx_s)
    rng.shuffle(idx_f)
    h_s = idx_s[: max(1, int(len(idx_s) * HELD_FRAC))]
    h_f = idx_f[: max(1, int(len(idx_f) * HELD_FRAC))]
    held = set(h_s) | set(h_f)
    tr = [i for i in range(len(rolls)) if i not in held]
    Xs = gather([rolls[i] for i in tr], [labels[i] for i in tr], li, cap, 1)
    Xf = gather([rolls[i] for i in tr], [labels[i] for i in tr], li, cap, 0)
    if len(Xs) == 0 or len(Xf) == 0:
        return None
    X_held = np.concatenate([episode_records(rolls[i], li, cap) for i in h_s], axis=0)
    return Xs, Xf, X_held


def pick_alpha(Xs_tr: np.ndarray, X_held: np.ndarray):
    """C_succ held-out 이득이 SUCC_BAND 에 드는 α (0.8 근접 우선). 없으면 (closest, True)."""
    lo, hi = SUCC_BAND
    cands = []
    for a in ALPHAS:
        g = gain(compute_conceptor(Xs_tr, a), X_held)
        cands.append((a, g))
    in_band = [(a, g) for a, g in cands if lo <= g <= hi]
    if in_band:
        a, g = min(in_band, key=lambda t: abs(t[1] - 0.8))
        return a, g, False
    a, g = min(cands, key=lambda t: min(abs(t[1] - lo), abs(t[1] - hi)))
    return a, g, True  # saturated/degenerate flag


def split_score(rolls, labels, li: int, cap: int, rng, alpha: float) -> float:
    """episode 80/20 split: train C_steer(α) fit → held-out success record 이득."""
    got = split_data(rolls, labels, li, cap, rng)
    if got is None:
        return float("nan")
    Xs, Xf, X_held = got
    return gain(c_steer(Xs, Xf, alpha), X_held)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rolls = load_cell_rolls(args.manifest, args.cell)
    labels = [r["success"] for r in rolls]
    labels_arr = np.asarray(labels)
    cap_layers = [int(x) for x in rolls[0]["capture_layers"]]
    succ_lens = [r["length"] for r, y in zip(rolls, labels) if y == 1]
    cap = int(np.ceil(np.mean(succ_lens) + np.std(succ_lens)))

    rng = np.random.default_rng(RNG_SEED)
    perm_list = []
    for _ in range(N_PERM):
        p = labels_arr.copy()
        rng.shuffle(p)
        perm_list.append(p.tolist())

    sweep = []
    for blk in cap_layers:
        li = cap_layers.index(blk)
        got = split_data(rolls, labels, li, cap, np.random.default_rng(RNG_SEED + blk))
        if got is None:
            continue
        Xs_tr, _Xf_tr, X_held = got
        alpha, g_succ, saturated = pick_alpha(Xs_tr, X_held)
        g_true = split_score(rolls, labels, li, cap,
                             np.random.default_rng(RNG_SEED + blk), alpha)
        null = [
            split_score(rolls, pl, li, cap,
                        np.random.default_rng(RNG_SEED + blk + 31 * (pi + 1)), alpha)
            for pi, pl in enumerate(perm_list)
        ]
        null = [x for x in null if np.isfinite(x)]
        mu_n, sd_n = float(np.mean(null)), float(np.std(null))
        z = (g_true - mu_n) / sd_n if sd_n > 1e-12 else float("nan")
        sweep.append({"layer": blk, "alpha": alpha, "gain_succ_heldout": g_succ,
                      "alpha_saturated": saturated, "gain_heldout": g_true,
                      "null_mean": mu_n, "null_std": sd_n, "z": z})
        print(f"  L{blk}: α={alpha:g} g_succ={g_succ:.3f}{'(sat)' if saturated else ''} "
              f"gain={g_true:.5f} null={mu_n:.5f}±{sd_n:.5f} z={z:.2f}", flush=True)

    best = max(sweep, key=lambda r: (r["z"] if np.isfinite(r["z"]) else -1e9))
    degenerate = all(abs(r["z"]) < 2 for r in sweep if np.isfinite(r["z"]))
    out = {
        "cell": args.cell, "criterion": "held-out success-record R-weighted gain of "
        "C_steer(per-layer α: C_succ 이득 band 선택, denoise pool) vs "
        "episode-label-permutation null z",
        "cap_records": cap, "n_perm": N_PERM,
        "alpha_grid": ALPHAS, "succ_gain_band": list(SUCC_BAND),
        "sweep": sweep, "selected_layer": best["layer"], "selected_z": best["z"],
        "all_degenerate_z_lt_2": degenerate,
        "manifest_sha": hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:12],
        "rng_seed": RNG_SEED,
        "note": "전 layer 퇴화 예상(memory conceptor-saturation-degenerate) — 퇴화여도 "
                "최대-z layer 로 A arm 배포 (legacy 참조선; 배포 fit 은 per_step·table14)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[select] {args.cell}: layer=L{best['layer']} z={best['z']:.2f} "
          f"degenerate={degenerate} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
