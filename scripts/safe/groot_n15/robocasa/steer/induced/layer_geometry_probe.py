"""layer별 공간(기하) 변경 프로브 — 섭동이 activation 공간을 '어떻게' 바꾸는가 (07-27).

steering 설계 판정용 4지표, config × layer(0,2,4,8,10,12,15,VL):
1. mean_shift_dprime : ‖μ_p−μ_c‖ 를 그 방향의 pooled std 로 나눈 d' — 평균 이동의 유의성.
2. subspace_leak     : clean top-k PCA 부분공간(에너지 90%) 밖 잔차 에너지 —
   perturbed 평균 잔차 − clean held-out 평균 잔차 (**초과 누출**; ≈0 이면 같은 공간 안
   에서의 이동 → 되돌리기 기하학적으로 용이, ≫0 이면 공간 이탈).
3. delta_pca_evr     : paired Δh(C1/P1/P2 — trigger 전 bitwise 동일 검증) PCA top1/top5
   설명분산 — 섭동 효과가 저rank(소수 방향 = setM/ActAdd 로 조준 가능)인지 확산형인지.
4. spread_ratio      : perturbed/clean 공분산 participation ratio 비 — 공간 확장/수축.

dwell 통제: episode 당 균등 k=20 서브샘플, perturbed 는 시간분리 절단 적용.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from induced_common import load_roll_any, read_record_start  # noqa: E402

LAYER_KEYS = [0, 2, 4, 8, 10, 12, 15, "VL"]


def _ep_of(p: Path) -> int:
    return int(re.search(r"--ep(\d+)--", p.name).group(1))


def _feat(r: dict, lk) -> np.ndarray:
    if lk == "VL":
        return np.asarray(r["vl"], dtype=np.float32)
    return r["dit"][:, r["capture_layers"].index(lk), :].astype(np.float32)


def _sub(X: np.ndarray, k: int = 20) -> np.ndarray:
    n = X.shape[0]
    if n <= k:
        return X
    return X[np.linspace(0, n - 1, k).round().astype(int)]


def _pr(X: np.ndarray) -> float:
    """공분산 participation ratio (유효 차원)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float((s.sum() ** 2) / max((s**2).sum(), 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--record-start", required=True)
    ap.add_argument("--k-sub", type=int, default=20)
    ap.add_argument("--pca-energy", type=float, default=0.90)
    ap.add_argument("--onset-len", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rs = read_record_start(args.record_start)
    clean = {_ep_of(p): load_roll_any(p)
             for p in sorted(Path(args.clean_dir).glob("task*--ep*--succ1.pkl"))}
    ceps = sorted(clean)
    half = set(ceps[::2])  # clean 부분공간 fit 용 절반 / 나머지 = held-out 잔차 기준

    out = {"config": vars(args), "configs": {}}
    for cfg_dir in sorted(Path(args.capture_root).iterdir()):
        cfg = cfg_dir.name
        pert = []
        for p in sorted(cfg_dir.glob("raw_rollouts/*/*/task*--ep*--succ*.pkl")):
            r = load_roll_any(p)
            pert.append((_ep_of(p), r, rs.get(str(p.resolve()), 0)))
        if not pert:
            continue
        res = {}
        for lk in LAYER_KEYS:
            Xc_fit = np.concatenate([_sub(_feat(clean[e], lk), args.k_sub) for e in ceps if e in half])
            Xc_held = np.concatenate([_sub(_feat(clean[e], lk), args.k_sub) for e in ceps if e not in half])
            Xp = np.concatenate([_sub(_feat(r, lk)[s:], args.k_sub) for _e, r, s in pert
                                 if _feat(r, lk)[s:].shape[0] > 0])
            # 1. mean shift d'
            mu_c, mu_p = Xc_held.mean(axis=0), Xp.mean(axis=0)
            v = mu_p - mu_c
            nv = np.linalg.norm(v)
            vhat = v / max(nv, 1e-12)
            sd = np.sqrt(0.5 * (np.var(Xc_held @ vhat) + np.var(Xp @ vhat)))
            dprime = float(nv / max(sd, 1e-12))
            # 2. clean 부분공간 초과 누출
            Xf = Xc_fit - Xc_fit.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xf, full_matrices=False)
            k90 = int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), args.pca_energy) + 1)
            B = Vt[:k90]  # [k, D]
            mu0 = Xc_fit.mean(axis=0)

            def resid(X):
                Z = X - mu0
                proj = Z @ B.T
                return float(np.mean(1.0 - (proj**2).sum(1) / np.clip((Z**2).sum(1), 1e-12, None)))

            leak = resid(Xp) - resid(Xc_held)
            # 3. paired Δh 저rank 성 (C1/P1/P2)
            evr1 = evr5 = None
            if not cfg.startswith("g1"):
                dl = []
                for e, r, s in pert:
                    if e not in clean:
                        continue
                    on0 = 0 if cfg.startswith("c1") else max(s - (4 if cfg.startswith("p1") else 2), 0)
                    Xcl, Xpe = _feat(clean[e], lk), _feat(r, lk)
                    hi = min(on0 + args.onset_len, Xcl.shape[0], Xpe.shape[0])
                    if hi > on0:
                        dl.append(Xpe[on0:hi] - Xcl[on0:hi])
                if dl:
                    D = np.concatenate(dl)
                    D = D - D.mean(axis=0, keepdims=True)
                    s2 = np.linalg.svd(D, compute_uv=False) ** 2
                    evr1 = float(s2[0] / s2.sum())
                    evr5 = float(s2[:5].sum() / s2.sum())
            # 4. 유효 차원 비
            spread = _pr(Xp) / max(_pr(Xc_held), 1e-9)
            res[str(lk)] = {"mean_shift_dprime": dprime, "clean_k90": k90,
                            "subspace_leak_excess": leak,
                            "delta_evr_top1": evr1, "delta_evr_top5": evr5,
                            "spread_ratio_pr": float(spread)}
        out["configs"][cfg] = res
        for m, fmt in (("mean_shift_dprime", "{:.2f}"), ("subspace_leak_excess", "{:.4f}"),
                       ("delta_evr_top1", "{:.2f}"), ("spread_ratio_pr", "{:.2f}")):
            row = " ".join(
                f"L{lk}:" + (fmt.format(res[str(lk)][m]) if res[str(lk)][m] is not None else "--")
                for lk in LAYER_KEYS)
            print(f"[{cfg}] {m:22} {row}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
