#!/usr/bin/env python3
"""Fit N1.5 instruction-cell conceptors from instruction pathway feature caches."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
N16_STEER_ROOT = REPO_ROOT / "scripts" / "safe" / "groot_n16" / "robocasa" / "steer"
if str(N16_STEER_ROOT) not in sys.path:
    sys.path.insert(0, str(N16_STEER_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fit_conceptor_steering import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_MAX_LEN,
    OVERLAP_BAND,
    fit_group,
    save_group,
)


@dataclass(frozen=True)
class CellDataset:
    cell_id: str
    cell_index: int
    task: str
    X: np.ndarray
    success: np.ndarray
    n_success_episodes: int
    n_failure_episodes: int


def _mode_tag(mode: str, max_len: int) -> str:
    return f"truncated_w{max_len}" if mode == "truncated" else mode


def _cache_run_root(cache_path: Path) -> Path:
    # <run_root>/analysis/pathway_separation/<cache>.npz
    return cache_path.parents[2]


def _as_str_array(values: np.ndarray) -> np.ndarray:
    return values.astype(str) if values.dtype.kind not in {"U", "S", "O"} else values


def _load_cache(cache_path: Path, pathway: str) -> dict[str, np.ndarray]:
    z = np.load(cache_path, allow_pickle=True)
    feature_key = pathway
    if feature_key not in z.files:
        raise ValueError(f"{cache_path} does not contain pathway key {feature_key!r}")
    required = {
        feature_key,
        "success",
        "cell_id",
        "cell_index",
        "task",
        "episode_idx",
        "scenario_seed",
        "step_idx",
    }
    missing = sorted(required.difference(z.files))
    if missing:
        raise ValueError(f"{cache_path} is missing required arrays: {', '.join(missing)}")
    return {
        "features": z[feature_key].astype(np.float64, copy=False),
        "success": z["success"].astype(np.int64, copy=False),
        "cell_id": _as_str_array(z["cell_id"]),
        "cell_index": z["cell_index"].astype(np.int64, copy=False),
        "task": _as_str_array(z["task"]),
        "episode_idx": z["episode_idx"].astype(np.int64, copy=False),
        "scenario_seed": z["scenario_seed"].astype(np.int64, copy=False),
        "step_idx": z["step_idx"].astype(np.int64, copy=False),
    }


def _validate_pathway_key(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            "pathway must be a cache key/path-safe token such as dit, vl, or dit_layer0"
        )
    return value


def _episode_keys(data: dict[str, np.ndarray], mask: np.ndarray) -> list[tuple[int, int]]:
    pairs = zip(data["episode_idx"][mask].tolist(), data["scenario_seed"][mask].tolist())
    return sorted({(int(ep), int(seed)) for ep, seed in pairs})


def _episode_class_counts(data: dict[str, np.ndarray], cell_mask: np.ndarray) -> tuple[int, int]:
    success_mask = cell_mask & (data["success"] == 1)
    failure_mask = cell_mask & (data["success"] == 0)
    return len(_episode_keys(data, success_mask)), len(_episode_keys(data, failure_mask))


def _episode_mean_dataset(data: dict[str, np.ndarray], cell_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    cell_indices = np.flatnonzero(cell_mask)
    groups: dict[tuple[int, int], list[int]] = {}
    for idx in cell_indices:
        key = (int(data["episode_idx"][idx]), int(data["scenario_seed"][idx]))
        groups.setdefault(key, []).append(int(idx))
    for indices in groups.values():
        group_idx = np.asarray(indices, dtype=np.int64)
        labels_in_group = np.unique(data["success"][group_idx])
        if labels_in_group.shape[0] != 1:
            raise ValueError("episode has mixed success labels")
        rows.append(data["features"][group_idx].mean(axis=0))
        labels.append(int(labels_in_group[0]))
    if not rows:
        return np.empty((0, data["features"].shape[1]), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def build_cell_dataset(
    data: dict[str, np.ndarray],
    cell_id: str,
    *,
    agg_mode: str,
    max_len: int,
) -> CellDataset:
    cell_mask = data["cell_id"] == cell_id
    if not np.any(cell_mask):
        raise ValueError(f"cell_id not found in cache: {cell_id}")

    n_success_eps, n_failure_eps = _episode_class_counts(data, cell_mask)
    first = int(np.flatnonzero(cell_mask)[0])
    cell_index = int(data["cell_index"][first])
    task = str(data["task"][first])

    if agg_mode == "coast":
        sample_mask = cell_mask
        X = data["features"][sample_mask]
        success = data["success"][sample_mask]
    elif agg_mode == "truncated":
        sample_mask = cell_mask & (data["step_idx"] < max_len)
        X = data["features"][sample_mask]
        success = data["success"][sample_mask]
    elif agg_mode == "episode_mean":
        X, success = _episode_mean_dataset(data, cell_mask)
    else:
        raise ValueError(f"unknown agg_mode: {agg_mode}")

    return CellDataset(
        cell_id=cell_id,
        cell_index=cell_index,
        task=task,
        X=X,
        success=success,
        n_success_episodes=n_success_eps,
        n_failure_episodes=n_failure_eps,
    )


def _safe_write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pathway",
        "agg_mode",
        "cell_id",
        "cell_index",
        "task",
        "status",
        "n_success_episodes",
        "n_failure_episodes",
        "n_success_samples",
        "n_failure_samples",
        "feature_dim",
        "selected_alphas",
        "selection_mode",
        "output_dir",
        "reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fit_cells(
    *,
    cache_path: Path,
    pathway: str,
    agg_mode: str,
    max_len: int,
    alphas: list[float],
    overlap_band: tuple[float, float],
    out_dir: Path,
    cell_ids: list[str] | None,
    min_episodes_per_class: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    data = _load_cache(cache_path, pathway)
    available_cells = sorted(str(c) for c in np.unique(data["cell_id"]))
    selected_cells = available_cells if cell_ids is None else cell_ids
    unknown = sorted(set(selected_cells).difference(available_cells))
    if unknown:
        raise ValueError(f"Unknown cell_id(s): {', '.join(unknown)}")

    mode_tag = _mode_tag(agg_mode, max_len)
    rows: list[dict[str, Any]] = []
    for cell_id in selected_cells:
        ds = build_cell_dataset(data, cell_id, agg_mode=agg_mode, max_len=max_len)
        s_mask = ds.success == 1
        n_success_samples = int(s_mask.sum())
        n_failure_samples = int((~s_mask).sum())
        cell_out = out_dir / pathway / mode_tag / cell_id
        base_row: dict[str, Any] = {
            "pathway": pathway,
            "agg_mode": mode_tag,
            "cell_id": ds.cell_id,
            "cell_index": ds.cell_index,
            "task": ds.task,
            "n_success_episodes": ds.n_success_episodes,
            "n_failure_episodes": ds.n_failure_episodes,
            "n_success_samples": n_success_samples,
            "n_failure_samples": n_failure_samples,
            "feature_dim": int(ds.X.shape[1]) if ds.X.ndim == 2 and ds.X.shape[0] else 0,
            "selected_alphas": "",
            "selection_mode": "",
            "output_dir": str(cell_out),
            "reason": "",
        }
        if (
            ds.n_success_episodes < min_episodes_per_class
            or ds.n_failure_episodes < min_episodes_per_class
        ):
            rows.append(
                {
                    **base_row,
                    "status": "skipped",
                    "reason": (
                        f"episode balance below threshold {min_episodes_per_class}: "
                        f"success={ds.n_success_episodes} failure={ds.n_failure_episodes}"
                    ),
                }
            )
            continue
        if n_success_samples < 2 or n_failure_samples < 2:
            rows.append(
                {
                    **base_row,
                    "status": "skipped",
                    "reason": (
                        "sample count below conceptor minimum: "
                        f"success={n_success_samples} failure={n_failure_samples}"
                    ),
                }
            )
            continue
        if dry_run:
            rows.append({**base_row, "status": "eligible", "reason": "dry-run"})
            continue

        fits, metadata = fit_group(ds.X[s_mask], ds.X[~s_mask], alphas, overlap_band)
        save_group(
            cell_out,
            fits,
            metadata,
            {
                "model_family": "lerobot_groot_n15",
                "group_key": "cell_id",
                "cell_id": ds.cell_id,
                "cell_index": ds.cell_index,
                "task": ds.task,
                "pathway": pathway,
                "agg_mode": mode_tag,
                "max_len": max_len if agg_mode == "truncated" else None,
                "min_episodes_per_class": min_episodes_per_class,
                "alpha_grid": alphas,
                "source": str(cache_path),
                "repo_root": str(REPO_ROOT),
                "created": datetime.now().isoformat(timespec="seconds"),
            },
        )
        rows.append(
            {
                **base_row,
                "status": "fit",
                "selected_alphas": ",".join(f"{a:g}" for a in metadata["selected_alphas"]),
                "selection_mode": metadata["selection_mode"],
            }
        )

    summary_name = "eligibility_summary.tsv" if dry_run else "fit_summary.tsv"
    summary_path = out_dir / pathway / mode_tag / summary_name
    _safe_write_summary(summary_path, rows)
    latest_summary_path = out_dir / summary_name
    _safe_write_summary(latest_summary_path, rows)
    print(f"wrote {len(rows)} rows -> {summary_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--pathway",
        required=True,
        help="Feature cache key to fit, e.g. dit, vl, or aligned layer key dit_layer0.",
    )
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument(
        "--agg-mode",
        choices=("coast", "truncated", "episode_mean"),
        default="truncated",
    )
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--alpha", type=float, nargs="+", default=None)
    parser.add_argument(
        "--overlap-band",
        type=float,
        nargs=2,
        default=list(OVERLAP_BAND),
        metavar=("LO", "HI"),
    )
    parser.add_argument("--min-episodes-per-class", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.alpha is None:
        args.alpha = list(DEFAULT_ALPHAS)
    if args.max_len <= 0:
        raise SystemExit("--max-len must be positive")
    if args.min_episodes_per_class < 1:
        raise SystemExit("--min-episodes-per-class must be >= 1")
    if args.out_dir is None:
        args.out_dir = _cache_run_root(args.cache) / "conceptor"
    try:
        args.pathway = _validate_pathway_key(args.pathway)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return args


def main() -> int:
    args = parse_args()
    fit_cells(
        cache_path=args.cache,
        pathway=args.pathway,
        agg_mode=args.agg_mode,
        max_len=args.max_len,
        alphas=list(args.alpha),
        overlap_band=(args.overlap_band[0], args.overlap_band[1]),
        out_dir=args.out_dir,
        cell_ids=args.cell_ids,
        min_episodes_per_class=args.min_episodes_per_class,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
