#!/usr/bin/env python3
"""Summarize instruction-fixed N1.5 RoboCasa rollout cells."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import pickle
from typing import Any


SUMMARY_COLUMNS = [
    "task",
    "cell_id",
    "cell_index",
    "canonical_instruction",
    "success",
    "total",
    "success_rate",
    "seed_min",
    "seed_max",
    "instruction_mismatch",
    "has_vl_hidden_states",
]


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _instruction_mismatch(payload: dict[str, Any]) -> bool:
    canonical = payload.get("canonical_instruction")
    if not canonical:
        return True
    task_description = payload.get("task_description")
    ep_meta = payload.get("ep_meta")
    ep_lang = ep_meta.get("lang") if isinstance(ep_meta, dict) else None
    return task_description != canonical or ep_lang != canonical


def summarize_instruction_cells(root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*/*/*.pkl")):
        payload = _load_pickle(path)
        task = payload.get("robocasa_task") or path.parent.parent.name
        cell_id = payload.get("cell_id") or path.parent.name
        grouped[(task, cell_id)].append(payload)

    rows: list[dict[str, Any]] = []
    for (task, cell_id), payloads in sorted(grouped.items()):
        total = len(payloads)
        success = sum(int(payload.get("episode_success", 0)) for payload in payloads)
        seeds = [
            int(payload["scenario_seed"])
            for payload in payloads
            if payload.get("scenario_seed") is not None
        ]
        first = payloads[0]
        rows.append(
            {
                "task": task,
                "cell_id": cell_id,
                "cell_index": int(first.get("cell_index", first.get("task_id", -1))),
                "canonical_instruction": first.get("canonical_instruction", ""),
                "success": success,
                "total": total,
                "success_rate": success / total if total else 0.0,
                "seed_min": min(seeds) if seeds else "",
                "seed_max": max(seeds) if seeds else "",
                "instruction_mismatch": sum(
                    int(_instruction_mismatch(payload)) for payload in payloads
                ),
                "has_vl_hidden_states": sum(
                    int(bool(payload.get("vl_hidden_states"))) for payload in payloads
                ),
            }
        )

    return rows


def write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["success_rate"] = f"{float(row['success_rate']):.4f}"
            writer.writerow(serializable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="raw_rollouts root with task/cell pkl layout")
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summarize_instruction_cells(args.root)
    write_summary_tsv(args.output_tsv, rows)
    print(f"wrote {len(rows)} instruction cell summaries -> {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
