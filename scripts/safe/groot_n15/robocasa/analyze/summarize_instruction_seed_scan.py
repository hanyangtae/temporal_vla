#!/usr/bin/env python3
"""Summarize instruction-fixed seed preselection progress."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CELLS_CONFIG = Path("configs/robocasa/n15_instruction_fixed_cells.tsv")

SUMMARY_COLUMNS = [
    "cell_index",
    "cell_id",
    "task",
    "target_count",
    "selected",
    "instruction_mismatch",
    "duplicate_seed_count",
    "ep_meta_missing",
    "scan_samples",
    "last_scanned_seed",
    "progress_found",
    "status",
]


def _read_tsv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_many_tsv(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_tsv(path))
    return rows


def _to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _normalize_instruction(value: str) -> str:
    return " ".join(value.strip().split())


def _default_scan_progress_path(selected_tsv: Path) -> Path:
    return selected_tsv.with_name(f"{selected_tsv.stem}_scan_progress.tsv")


def _default_scan_samples_path(selected_tsv: Path) -> Path:
    return selected_tsv.with_name(f"{selected_tsv.stem}_scan_samples.tsv")


def _default_shard_scan_paths(selected_tsv: Path, *, suffix: str) -> list[Path]:
    shard_dir = selected_tsv.parent / "shards"
    if not shard_dir.is_dir():
        return []
    return sorted(shard_dir.glob(f"*_{suffix}.tsv"))


def _resolve_artifact_path(path_base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else path_base / path


def summarize_seed_scan(
    *,
    cells_config: Path,
    selected_tsv: Path,
    scan_progress_tsv: Path | None = None,
    scan_samples_tsv: Path | None = None,
    path_base: Path | None = None,
) -> list[dict[str, Any]]:
    selected_rows = _read_tsv(selected_tsv)
    if scan_progress_tsv is None:
        progress_rows = _read_tsv(_default_scan_progress_path(selected_tsv))
        progress_rows.extend(
            _read_many_tsv(
                _default_shard_scan_paths(selected_tsv, suffix="scan_progress")
            )
        )
    else:
        progress_rows = _read_tsv(scan_progress_tsv)
    if scan_samples_tsv is None:
        sample_rows = _read_tsv(_default_scan_samples_path(selected_tsv))
        sample_rows.extend(
            _read_many_tsv(_default_shard_scan_paths(selected_tsv, suffix="scan_samples"))
        )
    else:
        sample_rows = _read_tsv(scan_samples_tsv)
    path_base = path_base or Path.cwd()

    selected_by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        selected_by_cell[row["cell_id"]].append(row)

    samples_by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        samples_by_cell[row["cell_id"]].append(row)

    progress_by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in progress_rows:
        progress_by_cell[row["cell_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for cell in _read_tsv(cells_config):
        cell_id = cell["cell_id"]
        canonical = _normalize_instruction(cell["canonical_instruction"])
        selected = selected_by_cell.get(cell_id, [])
        samples = samples_by_cell.get(cell_id, [])
        progress = progress_by_cell.get(cell_id, [])

        seeds = [_to_int(row["scenario_seed"]) for row in selected]
        sample_seeds = [_to_int(row["scenario_seed"]) for row in samples]
        selected_count = len(selected)
        target_count = _to_int(cell["target_episodes"])
        instruction_mismatch = sum(
            int(_normalize_instruction(row["instruction"]) != canonical)
            for row in selected
        )
        ep_meta_missing = sum(
            int(not _resolve_artifact_path(path_base, row["ep_meta_path"]).exists())
            for row in selected
        )
        duplicate_seed_count = selected_count - len(set(seeds))

        progress_found = max(
            selected_count,
            sum(_to_int(row.get("found")) for row in progress),
        )
        last_scanned_candidates = [
            _to_int(row.get("last_scanned_seed"))
            for row in progress
            if row.get("last_scanned_seed") not in (None, "")
        ]
        last_scanned_candidates.extend(sample_seeds)
        last_scanned_candidates.extend(seeds)
        last_scanned_seed = max(last_scanned_candidates) if last_scanned_candidates else ""

        if (
            selected_count >= target_count
            and instruction_mismatch == 0
            and duplicate_seed_count == 0
            and ep_meta_missing == 0
        ):
            status = "complete"
        elif selected_count > 0 or samples or progress:
            status = "in_progress"
        else:
            status = "not_started"

        rows.append(
            {
                "cell_index": _to_int(cell["cell_index"]),
                "cell_id": cell_id,
                "task": cell["task"],
                "target_count": target_count,
                "selected": selected_count,
                "instruction_mismatch": instruction_mismatch,
                "duplicate_seed_count": duplicate_seed_count,
                "ep_meta_missing": ep_meta_missing,
                "scan_samples": len(samples),
                "last_scanned_seed": last_scanned_seed,
                "progress_found": progress_found,
                "status": status,
            }
        )

    return rows


def write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-config", type=Path, default=DEFAULT_CELLS_CONFIG)
    parser.add_argument("--selected-seeds", type=Path, required=True)
    parser.add_argument("--scan-progress-tsv", type=Path, default=None)
    parser.add_argument("--scan-samples-tsv", type=Path, default=None)
    parser.add_argument(
        "--path-base",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative ep_meta_path values in selected manifest.",
    )
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summarize_seed_scan(
        cells_config=args.cells_config,
        selected_tsv=args.selected_seeds,
        scan_progress_tsv=args.scan_progress_tsv,
        scan_samples_tsv=args.scan_samples_tsv,
        path_base=args.path_base,
    )
    write_summary_tsv(args.output_tsv, rows)
    complete = sum(int(row["status"] == "complete") for row in rows)
    print(f"wrote {len(rows)} seed scan rows ({complete} complete) -> {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
