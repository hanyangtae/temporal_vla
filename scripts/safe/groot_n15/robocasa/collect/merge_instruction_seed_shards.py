#!/usr/bin/env python3
"""Merge instruction-fixed seed shard manifests into one selected-seed TSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CELLS_CONFIG = Path("configs/robocasa/n15_instruction_fixed_cells.tsv")

SELECTED_SEED_COLUMNS = [
    "cell_index",
    "cell_id",
    "task",
    "env_name",
    "scenario_seed",
    "instruction",
    "canonical_instruction",
    "ep_meta_path",
]


def _normalize_instruction(value: str) -> str:
    return " ".join(value.strip().split())


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _load_cells(path: Path) -> dict[str, dict[str, str]]:
    _fieldnames, rows = _read_tsv(path)
    cells = {row["cell_id"]: row for row in rows}
    if len(cells) != len(rows):
        raise ValueError(f"Duplicate cell_id in {path}")
    return cells


def discover_seed_shard_paths(shard_dir: Path) -> list[Path]:
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory does not exist: {shard_dir}")
    paths = [
        path
        for path in shard_dir.glob("*.tsv")
        if not path.stem.endswith("_scan_progress")
        and not path.stem.endswith("_scan_samples")
    ]
    return sorted(paths)


def merge_seed_shards(
    *,
    selected_seed_paths: list[Path],
    cells_config: Path,
    target_per_cell: int | None = None,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    cells = _load_cells(cells_config)
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()

    for path in selected_seed_paths:
        fieldnames, rows = _read_tsv(path)
        if "env_kwargs_json" in fieldnames:
            raise ValueError(
                f"{path}: env_kwargs_json is not supported; use natural seed scan shards"
            )
        missing = set(SELECTED_SEED_COLUMNS) - set(fieldnames)
        if missing:
            raise ValueError(f"{path}: missing selected seed columns: {', '.join(sorted(missing))}")

        for row in rows:
            cell_id = row["cell_id"]
            if cell_id not in cells:
                raise ValueError(f"{path}: unknown cell_id {cell_id!r}")
            cell = cells[cell_id]
            canonical = cell["canonical_instruction"]
            if _normalize_instruction(row["canonical_instruction"]) != _normalize_instruction(canonical):
                raise ValueError(f"{path}: canonical instruction mismatch for {cell_id}")
            if _normalize_instruction(row["instruction"]) != _normalize_instruction(canonical):
                raise ValueError(f"{path}: selected instruction mismatch for {cell_id}")

            scenario_seed = int(row["scenario_seed"])
            key = (cell_id, scenario_seed)
            if key in seen:
                raise ValueError(f"{path}: duplicate selected seed {cell_id}/{scenario_seed}")
            seen.add(key)
            by_cell[cell_id].append(
                {
                    "cell_index": int(cell["cell_index"]),
                    "cell_id": cell_id,
                    "task": cell["task"],
                    "env_name": cell["env_name"],
                    "scenario_seed": scenario_seed,
                    "instruction": row["instruction"],
                    "canonical_instruction": canonical,
                    "ep_meta_path": row["ep_meta_path"],
                }
            )

    merged: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for cell_id, cell in sorted(cells.items(), key=lambda item: int(item[1]["cell_index"])):
        target_count = target_per_cell or int(cell["target_episodes"])
        rows = sorted(by_cell.get(cell_id, []), key=lambda row: int(row["scenario_seed"]))
        if len(rows) < target_count:
            incomplete.append(f"{cell_id}={len(rows)}/{target_count}")
        merged.extend(rows[:target_count])

    if require_complete and incomplete:
        raise RuntimeError(f"incomplete selected seed shards: {', '.join(incomplete)}")

    return merged


def write_selected_seed_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_SEED_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-config", type=Path, default=DEFAULT_CELLS_CONFIG)
    parser.add_argument("--selected-seeds", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Directory containing per-cell seed shard TSV files.",
    )
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--target-per-cell", type=int, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selected_seeds is None and args.shard_dir is None:
        raise SystemExit("one of --selected-seeds or --shard-dir is required")
    selected_seed_paths = list(args.selected_seeds or [])
    if args.shard_dir is not None:
        selected_seed_paths.extend(discover_seed_shard_paths(args.shard_dir))
    rows = merge_seed_shards(
        selected_seed_paths=selected_seed_paths,
        cells_config=args.cells_config,
        target_per_cell=args.target_per_cell,
        require_complete=args.require_complete,
    )
    write_selected_seed_manifest(args.output_tsv, rows)
    print(f"merged {len(rows)} selected seeds -> {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
