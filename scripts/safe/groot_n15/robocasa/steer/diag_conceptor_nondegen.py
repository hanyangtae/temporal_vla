"""conceptor 비퇴화 진단 — 공유계획(24 §3) 하드 게이트. exp4-2 fit 산출물의 1차 가설 검정.

07-20 산술 검사의 스크립트화. 입력 = conceptors.npz + held-out activation manifest.
출력 JSON 4블록:
  1. spectrum      : C_steer 고유값(상위 64)·effective rank(participation ratio)·quota
  2. r_weighted    : R-가중 이득 tr(C_steer·R)/tr(R) — R 은 (a) fit C_success 역산
                     (자기참조) (b) held-out succ record 의 E[hhᵀ] 두 버전. exp3 기준값
                     0.006~0.007 병기.
  3. on_data       : held-out record 의 효과비 r(h)=‖(M−I)h‖/‖h‖ 분포 + Δh 방향 PCA
                     집중도 + C_succ held-out quota (1 근접 = 포화).
  4. perm_null     : label-permutation 재fit null 분포의 gain quantile + 관측 p —
                     **sanity gate 전용** (자기참조 지표라 단독 인과 검정 아님, Gate1).

pkl 로드에 torch 필요 — lerobot 컨테이너 실행. 예:
  docker exec lerobot python .../steer/diag_conceptor_nondegen.py \
    --npz <.../dit_L8/conceptors.npz> --capture-layer 8 \
    --fit-manifest fit.tsv --held-manifest held.tsv \
    [--record-start-manifest rs.tsv] [--beta 0.3] [--n-perm 200] --out diag.json
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

_INDUCED = Path(__file__).resolve().parent / "induced"
if str(_INDUCED) not in sys.path:
    sys.path.insert(0, str(_INDUCED))

from induced_common import REPO, load_roll_any  # noqa: E402  (sys.path 부수효과 포함)
from src.conceptor import (  # noqa: E402
    and_conceptor,
    compute_conceptor,
    conceptor_quota,
    not_conceptor,
)

EXP3_GAIN_REF = (0.006, 0.007)  # exp3 자연실패 fit 의 R-가중 이득 (퇴화 기준값)


def _read_manifest(path: str) -> list[tuple[Path, int]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if not p.is_absolute():
            p = REPO / p
        rows.append((p, int(parts[1])))
    return rows


def _read_rs(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if not p.is_absolute():
            p = REPO / p
        out[str(p.resolve())] = int(parts[1])
    return out


def _load_records(rows, rs_map, capture_layer: int, group: str):
    """manifest → episode 별 [n_i, D] record 행렬 리스트 (+라벨)."""
    eps = []
    for p, label in rows:
        r = load_roll_any(p)
        cap = r["capture_layers"]
        if capture_layer not in cap:
            raise SystemExit(f"{p.name}: capture layer {capture_layer} 없음 (cap={cap})")
        li = cap.index(capture_layer)
        start = rs_map.get(str(p.resolve()), 0)
        X = r["dit"][start:, li, :]  # [n, D]
        phases = r["phases"][start:]
        if group != "global":
            idx = [i for i, ph in enumerate(phases) if ph == group]
            X = X[idx]
        if X.shape[0]:
            eps.append((X.astype(np.float32), int(label)))
    return eps


def _refit_gain(eps, labels, alpha: float, R_ref: np.ndarray) -> float:
    """episode 리스트 + (순열)라벨로 C_steer 재fit 후 R_ref 가중 이득."""
    Xs_list = [X for (X, _), lab in zip(eps, labels) if lab == 1]
    Xf_list = [X for (X, _), lab in zip(eps, labels) if lab == 0]
    if not Xs_list or not Xf_list:
        return float("nan")
    Cs = compute_conceptor(np.concatenate(Xs_list, axis=0), alpha)
    Cf = compute_conceptor(np.concatenate(Xf_list, axis=0), alpha)
    Cst = and_conceptor(Cs, not_conceptor(Cf))
    return float(np.trace(Cst @ R_ref) / np.trace(R_ref))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--key", default=None, help="NPZ 키 접두 (기본: 첫 키의 α = 선택 α)")
    ap.add_argument("--capture-layer", type=int, required=True)
    ap.add_argument("--group", default="global")
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--fit-manifest", required=True, help="fit 에 쓴 episode (perm 재fit 원천)")
    ap.add_argument("--held-manifest", required=True, help="held-out episode (R_held·on-data)")
    ap.add_argument("--record-start-manifest", default=None)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--perm-seed", type=int, default=20260722)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    keys = [k for k in z.keys() if k.endswith("_C_steer")]
    if not keys:
        raise SystemExit(f"{args.npz}: *_C_steer 키 없음 ({sorted(z.keys())[:5]})")
    prefix = args.key or keys[0][: -len("_C_steer")]
    alpha = float(prefix.replace("alpha", ""))
    C_steer = np.asarray(z[f"{prefix}_C_steer"], dtype=np.float64)
    C_succ = np.asarray(z[f"{prefix}_C_success"], dtype=np.float64)
    D = C_steer.shape[0]

    # 1. spectrum
    ev = np.linalg.eigvalsh((C_steer + C_steer.T) / 2)[::-1]
    ev_pos = np.clip(ev, 0.0, None)
    eff_rank = float((ev_pos.sum() ** 2) / max((ev_pos**2).sum(), 1e-30))
    spectrum = {
        "alpha": alpha,
        "top_eigs": [float(v) for v in ev[:64]],
        "eig_max": float(ev[0]),
        "effective_rank": eff_rank,
        "quota_steer": float(conceptor_quota(C_steer.astype(np.float32))),
        "quota_success": float(conceptor_quota(C_succ.astype(np.float32))),
    }

    # 2a. R 복원 (fit C_success 역산 — 자기참조)
    w, U = np.linalg.eigh((C_succ + C_succ.T) / 2)
    w = np.clip(w, 0.0, 1.0 - 1e-9)
    R_fit = (U * (w / (1.0 - w) / (alpha**2))) @ U.T
    gain_fit = float(np.trace(C_steer @ R_fit) / np.trace(R_fit))

    # held-out 로드
    rs_map = _read_rs(args.record_start_manifest)
    held = _load_records(_read_manifest(args.held_manifest), rs_map, args.capture_layer, args.group)
    fit_eps = _load_records(_read_manifest(args.fit_manifest), rs_map, args.capture_layer, args.group)
    H = np.concatenate([X for X, _ in held], axis=0)
    Hs = np.concatenate([X for X, l in held if l == 1], axis=0) if any(l == 1 for _, l in held) else H
    # 2b. held-out succ 의 R
    R_held = (Hs.T @ Hs / max(Hs.shape[0], 1)).astype(np.float64)
    gain_held = float(np.trace(C_steer @ R_held) / np.trace(R_held))

    # 3. on-data 효과비 + 방향 구조성 + 포화
    M = (1.0 - args.beta) * np.eye(D) + args.beta * C_steer
    dH = H @ (M - np.eye(D)).T.astype(np.float32)
    r_h = np.linalg.norm(dH, axis=1) / np.clip(np.linalg.norm(H, axis=1), 1e-12, None)
    _, s_dh, _ = np.linalg.svd(dH - dH.mean(axis=0, keepdims=True), full_matrices=False)
    evr = (s_dh**2) / max((s_dh**2).sum(), 1e-30)
    q_held_succ = float(np.mean(np.einsum("nd,dk,nk->n", Hs, C_succ.astype(np.float32), Hs)
                                / np.clip((Hs * Hs).sum(axis=1), 1e-12, None)))
    on_data = {
        "n_held_records": int(H.shape[0]),
        "r_h_percentiles": {q: float(np.percentile(r_h, int(q))) for q in ("5", "25", "50", "75", "95")},
        "delta_pca_evr_top1": float(evr[0]),
        "delta_pca_evr_top5": float(evr[:5].sum()),
        "held_succ_Csucc_quota": q_held_succ,
        "saturated": bool(q_held_succ > 0.98),
    }

    # 4. permutation null (fit episode 라벨 순열 재fit → gain vs R_held)
    rng = np.random.default_rng(args.perm_seed)
    labels = [l for _, l in fit_eps]
    null = []
    for _ in range(args.n_perm):
        perm = list(rng.permutation(labels))
        null.append(_refit_gain(fit_eps, perm, alpha, R_held))
    null = np.asarray([v for v in null if np.isfinite(v)])
    p_val = float((np.sum(null >= gain_held) + 1) / (len(null) + 1)) if len(null) else float("nan")
    perm_null = {
        "n_perm": int(len(null)),
        "null_gain_q50": float(np.percentile(null, 50)) if len(null) else None,
        "null_gain_q95": float(np.percentile(null, 95)) if len(null) else None,
        "null_gain_q99": float(np.percentile(null, 99)) if len(null) else None,
        "observed_gain_held": gain_held,
        "p_upper": p_val,
        "note": "sanity gate 전용 — 자기참조 지표, 단독 인과 검정 아님 (공유계획 §3)",
    }

    out = {
        "npz": str(args.npz),
        "key_prefix": prefix,
        "capture_layer": args.capture_layer,
        "group": args.group,
        "beta": args.beta,
        "spectrum": spectrum,
        "r_weighted": {
            "gain_fit_R": gain_fit,
            "gain_held_R": gain_held,
            "exp3_natural_ref": list(EXP3_GAIN_REF),
        },
        "on_data": on_data,
        "perm_null": perm_null,
        "n_fit_eps": len(fit_eps),
        "n_held_eps": len(held),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[diag] eff_rank={eff_rank:.1f} gain_fit={gain_fit:.4f} gain_held={gain_held:.4f} "
          f"(exp3 ref {EXP3_GAIN_REF}) p_perm={p_val:.3f} saturated={on_data['saturated']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
