#!/usr/bin/env python3
"""Fit COAST contrastive conceptor steering matrices from GR00T N1.6 rollout features.

COAST App. A.9 (Eq. 4) + A.10.2 (Efficiently Selecting Hyperparameters):
  성공/실패 rollout 의 pre-velocity hidden state 분포에서 contrastive conceptor
  ``C_steer = C_success AND NOT C_failure`` 를 fit 하고, COAST 의 3-stage 절차로
  하이퍼파라미터를 선택한다.

  - Stage 1 (layer): quota(C_steer) at alpha0=10. **본 스크립트는 단일 layer
    데이터만 있어 layer 선택은 skip**, 대신 quota 를 진단값으로 기록한다(Eq. 10).
  - Stage 2 (aperture): alpha sweep 후 mean overlap(C_s, C_f) 가 sweet-spot band
    ``[0.85, 0.95]`` 에 드는 alpha 를 retain. 없으면 band 에 가장 가까운 alpha
    하나(Eq. 11). overlap-최소가 아니라 band — saturation 위/아래 모두 signal 이
    약해지기 때문(A.10.2 Stage 2).
  - Stage 3 (strength beta): rollout 이 필요한 유일 단계. 본 스크립트는 굽지 않고
    ``{0.1, 0.3}`` 를 권장값으로 기록한다(0.5 는 harmful 이라 drop).

Fit space:
  horizon/diff 둘 다 ``mean`` 으로 pool 한 ``z in R^1024`` (DiT pre-velocity hidden
  state). M=(1-β)I+β·C_steer 가 [1024,1024] 라서 inference 시 ``[K,H,D]`` 의 모든
  (k,h) slice 에 동일 적용 가능 → fit space 와 steer space 가 일치한다(A.9.2).

산출물 (scope 당, ``--per-task`` 시 task 당 추가):
  ``conceptors.npz``  — 선택된 alpha 별 ``alpha{a}_C_steer/_C_success/_C_failure``
                        (float32 [D,D]). COAST A.9.5 의 NPZ 저장 방식.
  ``metadata.json``   — alpha sweep(overlap·quota), 선택 alpha·band·mode,
                        layer-quota(α10), 권장 beta, step3 config 목록.

실행 (host, conda activate hyundai_aigs):
  python scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py \
      --scope train --per-task
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from run_config import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from safe_feature_vectors import (  # noqa: E402
    load_manifest,
    parse_aggregation_command,
    pooled_hidden_states,
)
from src.conceptor import (  # noqa: E402
    and_conceptor,
    as_float32,
    compute_conceptor,
    conceptor_overlap,
    conceptor_quota,
    eigenvalue_spectrum,
    not_conceptor,
)

DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_split_seen4_unseen2_openDrawer_pnpCab_100ep"
)
# COAST Table 14 — GR00T RoboCasa aperture grid (max 10).
DEFAULT_ALPHAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
# COAST A.10.2 Stage 2 — overlap sweet-spot band.
OVERLAP_BAND = (0.85, 0.95)
# COAST Eq. 10 / Table 18 — Stage-1 quota diagnostic at the sharpest aperture.
LAYER_QUOTA_ALPHA = 10.0
# COAST A.10.2 Stage 3 — keep these, drop 0.5 (harmful).
RECOMMENDED_BETAS = [0.1, 0.3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--scope",
        default="train",
        choices=("all", "train", "val_seen", "val_unseen"),
        help="conceptor fit 에 사용할 manifest split.",
    )
    parser.add_argument("--horizon-idx-rel", default="mean")
    parser.add_argument("--diff-idx-rel", default="mean")
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help=f"aperture sweep 그리드. 생략 시 COAST GR00T RoboCasa 그리드 {DEFAULT_ALPHAS}.",
    )
    parser.add_argument(
        "--overlap-band",
        type=float,
        nargs=2,
        default=list(OVERLAP_BAND),
        metavar=("LO", "HI"),
        help=f"Stage 2 sweet-spot band. 기본 {OVERLAP_BAND}.",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="scope 전체(global) 외에 task_id 별 conceptor 도 fit (mode-conditional).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="기본값: <split-root>/../conceptor_steering/<scope>_<agg-slug>.",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="smoke 용. scope 의 앞 N rollout 만 로드.",
    )
    args = parser.parse_args()
    if args.alpha is None:
        args.alpha = list(DEFAULT_ALPHAS)
    return args


def resolve_source_path(source_path: str, repo_root: Path) -> Path:
    """manifest 의 절대 source_path 를 현재 repo root 기준으로 재해석.

    manifest 가 다른 머신(예: pdk_ws)에서 생성돼 prefix 가 어긋나는 경우를 위해,
    ``outputs/`` 이하 tail 을 현재 repo root 에 재결합한다. 원본 경로가 그대로
    존재하면 그것을 우선 사용한다.
    """
    original = Path(source_path)
    if original.exists():
        return original
    parts = original.parts
    if "outputs" in parts:
        tail = Path(*parts[parts.index("outputs") :])
        candidate = repo_root / tail
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"source_path 를 찾을 수 없음: {source_path} (repo_root={repo_root})"
    )


def load_timestep_features(
    split_root: Path,
    scope: str,
    *,
    horizon_idx_rel: str,
    diff_idx_rel: str,
    max_rollouts: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """scope 의 모든 rollout 에서 per-timestep z 와 라벨을 로드.

    Returns:
        X: (N, D) per-timestep pooled hidden state (모든 rollout 의 timestep concat)
        meta: {"success", "failure", "task_id", "task_name", "episode_idx"} — 각 (N,)
    """
    horizon_cmd = parse_aggregation_command(horizon_idx_rel)
    diff_cmd = parse_aggregation_command(diff_idx_rel)
    rows = load_manifest(split_root, scope)
    if max_rollouts is not None:
        rows = rows[:max_rollouts]

    chunks: list[np.ndarray] = []
    success: list[np.ndarray] = []
    task_id: list[np.ndarray] = []
    task_name: list[str] = []
    episode_idx: list[np.ndarray] = []

    for row in rows:
        path = resolve_source_path(row["source_path"], REPO_ROOT)
        with path.open("rb") as f:
            record = pickle.load(f)
        feats = pooled_hidden_states(
            record, horizon_idx_rel=horizon_cmd, diff_idx_rel=diff_cmd
        )
        n = feats.shape[0]
        if n == 0:
            continue
        succ = int(row["success"])
        chunks.append(feats)
        success.append(np.full(n, succ, dtype=np.int64))
        task_id.append(np.full(n, int(row["task_id"]), dtype=np.int64))
        task_name.extend([row["task"]] * n)
        episode_idx.append(np.full(n, int(row["episode_idx"]), dtype=np.int64))

    if not chunks:
        raise ValueError(f"No features loaded for scope={scope}.")

    X = np.concatenate(chunks, axis=0)
    meta = {
        "success": np.concatenate(success),
        "failure": 1 - np.concatenate(success),
        "task_id": np.concatenate(task_id),
        "task_name": np.asarray(task_name),
        "episode_idx": np.concatenate(episode_idx),
    }
    return X, meta


def select_alphas_in_band(
    sweep: list[dict[str, float]], band: tuple[float, float]
) -> tuple[list[float], str]:
    """COAST A.10.2 Stage 2 — overlap 이 band 안인 alpha 를 retain.

    band 안에 드는 alpha 가 없으면 band 에 가장 가까운 alpha 하나를 택한다.

    Returns:
        (selected_alphas, mode) — mode 는 "band" 또는 "closest".
    """
    lo, hi = band
    in_band = sorted(r["alpha"] for r in sweep if lo <= r["overlap"] <= hi)
    if in_band:
        return in_band, "band"

    def band_distance(row: dict[str, float]) -> float:
        o = row["overlap"]
        if lo <= o <= hi:
            return 0.0
        return min(abs(o - lo), abs(o - hi))

    closest = min(sweep, key=band_distance)
    return [closest["alpha"]], "closest"


def fit_group(
    X_success: np.ndarray,
    X_failure: np.ndarray,
    alphas: list[float],
    band: tuple[float, float],
) -> tuple[dict[float, dict[str, Any]], dict[str, Any]]:
    """한 그룹(global 또는 task)에 대해 alpha sweep + band 선택 + contrastive fit.

    Returns:
        fits: 선택된 alpha 별 {C_success, C_failure, C_steer}
        metadata: sweep·선택·진단 정보
    """
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    sweep: list[dict[str, float]] = []
    for alpha in alphas:
        C_s = compute_conceptor(X_success, alpha)
        C_f = compute_conceptor(X_failure, alpha)
        cache[alpha] = (C_s, C_f)
        sweep.append(
            {
                "alpha": float(alpha),
                "overlap": conceptor_overlap(C_s, C_f),
                "quota_success": conceptor_quota(C_s),
                "quota_failure": conceptor_quota(C_f),
            }
        )

    selected, mode = select_alphas_in_band(sweep, band)

    # Stage 1 진단: contrastive conceptor quota at alpha0=10 (Eq. 10).
    if LAYER_QUOTA_ALPHA in cache:
        Cs0, Cf0 = cache[LAYER_QUOTA_ALPHA]
    else:
        Cs0 = compute_conceptor(X_success, LAYER_QUOTA_ALPHA)
        Cf0 = compute_conceptor(X_failure, LAYER_QUOTA_ALPHA)
    layer_quota = conceptor_quota(and_conceptor(Cs0, not_conceptor(Cf0)))

    fits: dict[float, dict[str, Any]] = {}
    per_alpha: dict[str, Any] = {}
    for alpha in selected:
        C_s, C_f = cache[alpha]
        C_steer = and_conceptor(C_s, not_conceptor(C_f))
        fits[alpha] = {"C_success": C_s, "C_failure": C_f, "C_steer": C_steer}
        per_alpha[f"{alpha:g}"] = {
            "overlap": conceptor_overlap(C_s, C_f),
            "quota_steer": conceptor_quota(C_steer),
            "spectrum_top20": eigenvalue_spectrum(C_steer)[:20].tolist(),
        }

    metadata = {
        "selected_alphas": [float(a) for a in selected],
        "selection_mode": mode,
        "overlap_band": list(band),
        "layer_quota_contrastive_at_alpha10": layer_quota,
        "recommended_betas": RECOMMENDED_BETAS,
        "step3_configs": [
            {"alpha": float(a), "beta": b} for a in selected for b in RECOMMENDED_BETAS
        ],
        "alpha_sweep": sweep,
        "per_alpha": per_alpha,
        "n_success": int(X_success.shape[0]),
        "n_failure": int(X_failure.shape[0]),
        "feature_dim": int(X_success.shape[1]),
    }
    return fits, metadata


def save_group(
    out_dir: Path,
    fits: dict[float, dict[str, Any]],
    metadata: dict[str, Any],
    base_meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for alpha, group in fits.items():
        key = f"alpha{alpha:g}"
        arrays[f"{key}_C_steer"] = as_float32(group["C_steer"])
        arrays[f"{key}_C_success"] = as_float32(group["C_success"])
        arrays[f"{key}_C_failure"] = as_float32(group["C_failure"])
    np.savez_compressed(out_dir / "conceptors.npz", **arrays)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump({**base_meta, **metadata}, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    band = (args.overlap_band[0], args.overlap_band[1])
    agg_slug = f"h{args.horizon_idx_rel}_d{args.diff_idx_rel}".replace("-", "")
    out_root = args.out_dir or (
        args.split_root.parent / "conceptor_steering" / f"{args.scope}_{agg_slug}"
    )

    X, meta = load_timestep_features(
        args.split_root,
        args.scope,
        horizon_idx_rel=args.horizon_idx_rel,
        diff_idx_rel=args.diff_idx_rel,
        max_rollouts=args.max_rollouts,
    )
    succ_mask = meta["success"] == 1
    print(
        f"[load] scope={args.scope} X={X.shape} "
        f"success_z={int(succ_mask.sum())} failure_z={int((~succ_mask).sum())}"
    )

    base_meta = {
        "scope": args.scope,
        "horizon_idx_rel": args.horizon_idx_rel,
        "diff_idx_rel": args.diff_idx_rel,
        "alpha_grid": args.alpha,
        "split_root": str(args.split_root),
        "repo_root": str(REPO_ROOT),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    def report(tag: str, m: dict[str, Any]) -> None:
        per = ", ".join(
            f"α{a}:ov={m['per_alpha'][f'{float(a):g}']['overlap']:.3f}"
            f"/q={m['per_alpha'][f'{float(a):g}']['quota_steer']:.3f}"
            for a in m["selected_alphas"]
        )
        print(
            f"[{tag}] selected α={m['selected_alphas']} ({m['selection_mode']}) "
            f"layer_quota(α10)={m['layer_quota_contrastive_at_alpha10']:.4f} | {per}"
        )

    # --- global ---
    fits, m = fit_group(X[succ_mask], X[~succ_mask], args.alpha, band)
    save_group(out_root / "global", fits, m, base_meta)
    report("global", m)

    # --- per-task (mode-conditional) ---
    if args.per_task:
        for tid in sorted(np.unique(meta["task_id"]).tolist()):
            tmask = meta["task_id"] == tid
            tname = meta["task_name"][tmask][0]
            t_succ = tmask & succ_mask
            t_fail = tmask & ~succ_mask
            if t_succ.sum() < 2 or t_fail.sum() < 2:
                print(
                    f"[task {tid} {tname}] skip — success_z={int(t_succ.sum())} "
                    f"failure_z={int(t_fail.sum())} (양쪽 >=2 필요)"
                )
                continue
            fits, m = fit_group(X[t_succ], X[t_fail], args.alpha, band)
            task_meta = {**base_meta, "task_id": int(tid), "task_name": str(tname)}
            save_group(out_root / f"task_{tid}_{tname}", fits, m, task_meta)
            report(f"task {tid} {tname}", m)

    print(f"[done] artifacts -> {out_root}")


if __name__ == "__main__":
    main()
