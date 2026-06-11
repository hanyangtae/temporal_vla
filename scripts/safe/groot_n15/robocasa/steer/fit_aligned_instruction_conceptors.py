#!/usr/bin/env python3
"""Fit all aligned N1.5 instruction conceptor pathways from one feature cache."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fit_instruction_conceptors import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_MAX_LEN,
    OVERLAP_BAND,
    _cache_run_root,
    _safe_write_summary,
    _validate_pathway_key,
    fit_cells,
)


LAYER_KEY_RE = re.compile(r"^dit_layer(\d+)$")


def discover_aligned_pathways(
    cache_path: Path,
    *,
    include_vl: bool = True,
) -> list[str]:
    z = np.load(cache_path, allow_pickle=True)
    layer_keys = []
    for key in z.files:
        match = LAYER_KEY_RE.fullmatch(key)
        if match:
            layer_keys.append((int(match.group(1)), key))
    pathways = [key for _layer, key in sorted(layer_keys)]
    if include_vl and "vl" in z.files:
        pathways.append("vl")
    if not pathways:
        raise ValueError(f"No aligned pathways found in cache: {cache_path}")
    return pathways


def fit_pathway_set(
    *,
    cache_path: Path,
    pathways: list[str] | None,
    include_vl: bool,
    agg_mode: str,
    max_len: int,
    alphas: list[float],
    overlap_band: tuple[float, float],
    out_dir: Path,
    cell_ids: list[str] | None,
    min_episodes_per_class: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    selected = (
        discover_aligned_pathways(cache_path, include_vl=include_vl)
        if pathways is None
        else [_validate_pathway_key(pathway) for pathway in pathways]
    )

    all_rows: list[dict[str, Any]] = []
    for pathway in selected:
        rows = fit_cells(
            cache_path=cache_path,
            pathway=pathway,
            agg_mode=agg_mode,
            max_len=max_len,
            alphas=alphas,
            overlap_band=overlap_band,
            out_dir=out_dir,
            cell_ids=cell_ids,
            min_episodes_per_class=min_episodes_per_class,
            dry_run=dry_run,
        )
        all_rows.extend(rows)

    summary_name = "eligibility_summary.tsv" if dry_run else "fit_summary.tsv"
    combined_path = out_dir / f"aligned_{summary_name}"
    _safe_write_summary(combined_path, all_rows)
    # Keep the top-level summary as the full aligned set after per-pathway calls.
    _safe_write_summary(out_dir / summary_name, all_rows)
    print(f"wrote {len(all_rows)} aligned rows across {len(selected)} pathways -> {combined_path}")
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--pathway",
        action="append",
        dest="pathways",
        help="Aligned cache key(s) to fit. Default discovers dit_layer* keys plus vl.",
    )
    parser.add_argument("--no-vl", action="store_true")
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
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
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
    return args


def main() -> int:
    args = parse_args()
    fit_pathway_set(
        cache_path=args.cache,
        pathways=args.pathways,
        include_vl=not bool(args.no_vl),
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
