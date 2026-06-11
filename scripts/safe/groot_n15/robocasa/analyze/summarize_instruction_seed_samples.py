#!/usr/bin/env python3
"""Summarize sampled instructions from instruction seed-scan audit logs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CELLS_CONFIG = Path("configs/robocasa/n15_instruction_fixed_cells.tsv")

SAMPLE_SUMMARY_COLUMNS = [
    "cell_index",
    "cell_id",
    "task",
    "canonical_instruction",
    "scanned",
    "matches",
    "match_rate",
    "unique_instructions",
    "top_instructions",
]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_cells(path: Path) -> list[dict[str, str]]:
    return _read_tsv(path)


def _sample_paths(shard_dir: Path) -> list[Path]:
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory does not exist: {shard_dir}")
    return sorted(shard_dir.glob("*_scan_samples.tsv"))


def _format_top(counter: Counter[str], *, top_n: int) -> str:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return " | ".join(f"{count}::{instruction}" for instruction, count in items)


def summarize_seed_samples(
    *,
    shard_dir: Path,
    cells_config: Path = DEFAULT_CELLS_CONFIG,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    cells = _load_cells(cells_config)
    instructions_by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    matches_by_cell: Counter[str] = Counter()

    for path in _sample_paths(shard_dir):
        for row in _read_tsv(path):
            cell_id = row["cell_id"]
            instructions_by_cell[cell_id][row["instruction"]] += 1
            if row.get("is_match") in {"1", "true", "True"}:
                matches_by_cell[cell_id] += 1

    rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = cell["cell_id"]
        counter = instructions_by_cell[cell_id]
        scanned = sum(counter.values())
        matches = matches_by_cell[cell_id]
        match_rate = (matches / scanned) if scanned else 0.0
        rows.append(
            {
                "cell_index": int(cell["cell_index"]),
                "cell_id": cell_id,
                "task": cell["task"],
                "canonical_instruction": cell["canonical_instruction"],
                "scanned": scanned,
                "matches": matches,
                "match_rate": f"{match_rate:.6f}",
                "unique_instructions": len(counter),
                "top_instructions": _format_top(counter, top_n=top_n),
            }
        )
    return rows


def write_sample_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_SUMMARY_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--cells-config", type=Path, default=DEFAULT_CELLS_CONFIG)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summarize_seed_samples(
        shard_dir=args.shard_dir,
        cells_config=args.cells_config,
        top_n=args.top_n,
    )
    write_sample_summary_tsv(args.output_tsv, rows)
    print(f"wrote {len(rows)} seed sample rows -> {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
