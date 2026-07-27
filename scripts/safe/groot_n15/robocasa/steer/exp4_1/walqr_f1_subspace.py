#!/usr/bin/env python3
"""exp4-1 WA-LQR F1 게이트 ①: (layer-partition × denoise-t) SVD k≤64 + c_means 가
기존 수집(fit30 full-token)으로 fit 가능한지 — eval 0판 타당성 검사 (24a §5, 참고 24c §3).

WA-LQR 원설계는 clean-vs-perturbed 대조지만 게이트 ①은 succ/fail 대조로 fit 가능성만
본다 (24 공유문서 확정 주석: W 는 성공-perturb 쌍 데이터 필요 — 그건 exp4-2 산출 대기).
출력: partition×k 별 설명분산(k=64)·‖c_means‖·표본수 → JSON. scipy 없음 → numpy SVD
(표본 상한 4000 record 서브샘플 — 가능성 판정에 충분).

사용 (승준): python walqr_f1_subspace.py --manifest <task_PPCC_fit.tsv> --cell pq3_ppcc_bread \
  --out <...>/walqr_f1_subspace.json
"""
from __future__ import annotations

import argparse
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

from fit_mean_diff import load_cell_rolls  # noqa: E402

K_SVD = 64
MAX_REC = 4000
PARTITIONS = {"early": [0, 2, 4], "mid": [8, 10], "late": [12, 15]}  # capture 7층의 3분할
RNG_SEED = 424103


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    rolls = load_cell_rolls(args.manifest, args.cell)
    labels = [r["success"] for r in rolls]
    cap_layers = [int(x) for x in rolls[0]["capture_layers"]]
    n_k = rolls[0]["dit_k"].shape[2]
    out = {"cell": args.cell, "k_svd": K_SVD, "partitions": {}, "n_denoise": n_k}
    feasible = True
    for pname, blks in PARTITIONS.items():
        lis = [cap_layers.index(b) for b in blks if b in cap_layers]
        for k in range(n_k):
            Xs, Xf = [], []
            for r, y in zip(rolls, labels):
                v = r["dit_k"][:, lis, k, :].reshape(-1, r["dit_k"].shape[-1])
                (Xs if y == 1 else Xf).append(v)
            Xs = np.concatenate(Xs, axis=0)
            Xf = np.concatenate(Xf, axis=0)
            X = np.concatenate([Xs, Xf], axis=0)
            if len(X) > MAX_REC:
                X = X[rng.choice(len(X), MAX_REC, replace=False)]
            mu = X.mean(axis=0)
            _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            ev = float((S[:K_SVD] ** 2).sum() / (S ** 2).sum())
            V = Vt[:K_SVD].T
            c = (Xf.mean(axis=0) - Xs.mean(axis=0)) @ V  # 사영 대조평균 (WAM c_means 상당)
            entry = {"denoise_t": k, "n_succ": int(len(Xs)), "n_fail": int(len(Xf)),
                     "explained_var_k64": ev, "c_means_norm": float(np.linalg.norm(c)),
                     "c_means_frac": float(np.linalg.norm(c) /
                                           max(1e-9, np.linalg.norm(Xf.mean(0) - Xs.mean(0))))}
            out["partitions"].setdefault(pname, []).append(entry)
            if not np.isfinite(ev) or len(Xs) < 500 or len(Xf) < 500:
                feasible = False
    out["verdict_item1"] = {
        "fit_computable": feasible,
        "note": "succ/fail 대조로 부분공간·c_means 산출 가능 여부만 판정. W 원설계의 "
                "clean-vs-perturbed 쌍 대조는 exp4-2 P0 데이터 도착 시 재평가 (24 공유문서 주석).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for pname, rows in out["partitions"].items():
        r0 = rows[0]
        print(f"[{pname}] k0: ev64={r0['explained_var_k64']:.3f} "
              f"‖c‖={r0['c_means_norm']:.2f} frac={r0['c_means_frac']:.3f} "
              f"N={r0['n_succ']}+{r0['n_fail']}", flush=True)
    print(f"[F1-①] fit_computable={feasible} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
