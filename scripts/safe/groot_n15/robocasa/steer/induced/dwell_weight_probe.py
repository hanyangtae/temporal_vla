"""dwell 가중 실증 프로브 — record 서브샘플링/에피소드 가중이 fit 에 중요한가 (07-23 질문).

fail=timeout(144rec)·succ=조기종료라 record-pooled 추정량은 fail dwell 상태가 지배한다.
세 추정 방식을 같은 데이터로 대조:
  (a) pooled    : 전 record 풀 (현행 fit_phase_conceptor / setM 의 μ)
  (b) ep-equal  : episode 별 mean 후 클래스 mean (episode 동등 가중)
  (c) subsample : episode 당 고정 k record 결정적 서브샘플(균등 간격) 후 풀

지표:
  - setM 관련  : mean-diff 방향 r̂ 의 (a)↔(b), (a)↔(c) cosine (1이면 dwell 무관)
  - conceptor  : C_steer(a) vs C_steer(c) 의 상호 R-가중 이득 비 + quota 변화
    (compute_conceptor 는 record-pool 이므로 (b)는 (c)로 근사)

  docker exec lerobot python dwell_weight_probe.py --manifest m.tsv \
      [--record-start rs.tsv] --layers 8,12 [--k 20] --out probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from induced_common import load_roll_any, read_manifest, read_record_start  # noqa: E402
from src.conceptor import and_conceptor, compute_conceptor, conceptor_quota, not_conceptor  # noqa: E402


def _cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def _subsample(X: np.ndarray, k: int) -> np.ndarray:
    """결정적 균등 간격 서브샘플 (k 이상이면 k개, 미만이면 전부)."""
    n = X.shape[0]
    if n <= k:
        return X
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return X[idx]


def _gain(C: np.ndarray, R: np.ndarray) -> float:
    return float(np.trace(C @ R) / np.trace(R))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--record-start", default=None)
    ap.add_argument("--layers", default="8,12")
    ap.add_argument("--k", type=int, default=20, help="episode 당 서브샘플 record 수")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rs = read_record_start(args.record_start)
    rows = read_manifest(args.manifest)
    out = {"config": vars(args), "layers": {}}
    for layer in [x.strip() for x in args.layers.split(",") if x.strip()]:
        eps = []  # (X [n,D], label)
        for p, label in rows:
            r = load_roll_any(p)
            start = rs.get(str(p.resolve()), 0)
            if layer == "VL":
                if r.get("vl") is None:
                    raise SystemExit(f"{p.name}: VL 없음")
                X = np.asarray(r["vl"], dtype=np.float32)[start:]
            else:
                cap = r["capture_layers"]
                li = cap.index(int(layer))
                X = r["dit"][start:, li, :].astype(np.float32)
            if X.shape[0]:
                eps.append((X, int(label)))
        Xs_eps = [X for X, l in eps if l == 1]
        Xf_eps = [X for X, l in eps if l == 0]

        def mdiff(f_list, s_list, mode):
            if mode == "pooled":
                mf = np.concatenate(f_list).mean(axis=0)
                ms = np.concatenate(s_list).mean(axis=0)
            elif mode == "ep-equal":
                mf = np.mean([x.mean(axis=0) for x in f_list], axis=0)
                ms = np.mean([x.mean(axis=0) for x in s_list], axis=0)
            else:  # subsample
                mf = np.concatenate([_subsample(x, args.k) for x in f_list]).mean(axis=0)
                ms = np.concatenate([_subsample(x, args.k) for x in s_list]).mean(axis=0)
            return mf - ms

        d_pool = mdiff(Xf_eps, Xs_eps, "pooled")
        d_epeq = mdiff(Xf_eps, Xs_eps, "ep-equal")
        d_sub = mdiff(Xf_eps, Xs_eps, "subsample")

        # conceptor: pooled vs subsample fit — 상호 R-이득 대조
        Xs_pool, Xf_pool = np.concatenate(Xs_eps), np.concatenate(Xf_eps)
        Xs_sub = np.concatenate([_subsample(x, args.k) for x in Xs_eps])
        Xf_sub = np.concatenate([_subsample(x, args.k) for x in Xf_eps])
        C_pool = and_conceptor(compute_conceptor(Xs_pool, args.alpha),
                               not_conceptor(compute_conceptor(Xf_pool, args.alpha)))
        C_sub = and_conceptor(compute_conceptor(Xs_sub, args.alpha),
                              not_conceptor(compute_conceptor(Xf_sub, args.alpha)))
        R_s_pool = (Xs_pool.T @ Xs_pool / len(Xs_pool)).astype(np.float64)
        R_s_sub = (Xs_sub.T @ Xs_sub / len(Xs_sub)).astype(np.float64)

        out["layers"][str(layer)] = {
            "n_eps": [len(Xf_eps), len(Xs_eps)],
            "n_records_pooled": [int(Xf_pool.shape[0]), int(Xs_pool.shape[0])],
            "n_records_subsample": [int(Xf_sub.shape[0]), int(Xs_sub.shape[0])],
            "rhat_cos_pooled_vs_epequal": _cos(d_pool, d_epeq),
            "rhat_cos_pooled_vs_subsample": _cos(d_pool, d_sub),
            "rhat_cos_epequal_vs_subsample": _cos(d_epeq, d_sub),
            "conceptor_quota": {"pooled": float(conceptor_quota(C_pool.astype(np.float32))),
                                "subsample": float(conceptor_quota(C_sub.astype(np.float32)))},
            "gain_matrix": {
                "Cpool_on_Rpool": _gain(C_pool, R_s_pool),
                "Cpool_on_Rsub": _gain(C_pool, R_s_sub),
                "Csub_on_Rpool": _gain(C_sub, R_s_pool),
                "Csub_on_Rsub": _gain(C_sub, R_s_sub),
            },
        }
        lay = out["layers"][str(layer)]
        print(f"[L{layer}] r̂ cos pool↔epeq={lay['rhat_cos_pooled_vs_epequal']:.3f} "
              f"pool↔sub={lay['rhat_cos_pooled_vs_subsample']:.3f} | "
              f"gain pool→pool={lay['gain_matrix']['Cpool_on_Rpool']:.4f} "
              f"sub→sub={lay['gain_matrix']['Csub_on_Rsub']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
