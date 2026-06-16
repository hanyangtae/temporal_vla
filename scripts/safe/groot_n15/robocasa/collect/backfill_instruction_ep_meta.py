#!/usr/bin/env python3
"""Backfill instruction-fixed run-root ep_meta JSON manifests from rollout pkls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policies.groot.robocasa.scenario_replay import (  # noqa: E402
    ep_meta_manifest_path,
    write_ep_meta_manifest,
)


SUMMARY_KEYS = [
    "processed_pkls",
    "written_json",
    "missing_pkl_ep_meta",
    "missing_manifest_rows",
    "manifest_mismatches",
    "rewritten_manifest_rows",
]


def _read_selected_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _write_selected_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"rollout pkl is not a dict: {path}")
    return payload


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _selected_index(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    index: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        seed = _to_int(row.get("scenario_seed"))
        cell_id = row.get("cell_id")
        if cell_id is None or seed is None:
            continue
        index[(cell_id, seed)] = row
    return index


def _new_summary() -> dict[str, int]:
    return {key: 0 for key in SUMMARY_KEYS}


def _manifest_row_matches_payload(row: dict[str, str], payload: dict[str, Any]) -> bool:
    expected = {
        "task": payload.get("robocasa_task"),
        "env_name": payload.get("env_name"),
        "canonical_instruction": payload.get("canonical_instruction"),
    }
    return all(value is None or row.get(key) == str(value) for key, value in expected.items())


def backfill_ep_meta(
    *,
    raw_rollouts: Path,
    selected_seeds: Path,
    ep_meta_root: Path,
    rewrite_selected_seeds: bool = False,
) -> dict[str, int]:
    fieldnames, selected_rows = _read_selected_manifest(selected_seeds)
    selected_by_key = _selected_index(selected_rows)
    summary = _new_summary()

    for pkl_path in sorted(raw_rollouts.glob("*/*/*.pkl")):
        payload = _load_pickle(pkl_path)
        summary["processed_pkls"] += 1

        ep_meta = payload.get("ep_meta")
        if not isinstance(ep_meta, dict):
            summary["missing_pkl_ep_meta"] += 1
            continue

        cell_id = str(payload.get("cell_id") or pkl_path.parent.name)
        task = str(payload.get("robocasa_task") or pkl_path.parent.parent.name)
        env_name = payload.get("env_name")
        scenario_seed = _to_int(payload.get("scenario_seed", payload.get("seed")))
        if env_name is None or scenario_seed is None:
            summary["missing_pkl_ep_meta"] += 1
            continue

        row = selected_by_key.get((cell_id, scenario_seed))
        if row is None:
            summary["missing_manifest_rows"] += 1
        elif not _manifest_row_matches_payload(row, payload):
            summary["manifest_mismatches"] += 1

        target_path = ep_meta_manifest_path(
            ep_meta_root / task / cell_id,
            str(env_name),
            scenario_seed,
        )
        write_ep_meta_manifest(
            target_path,
            env_name=str(env_name),
            scenario_seed=scenario_seed,
            ep_meta=ep_meta,
            robocasa_env_source=str(payload.get("robocasa_env_source") or "robocasa365"),
            sort_keys=True,
        )
        summary["written_json"] += 1

        if rewrite_selected_seeds and row is not None and row.get("ep_meta_path") != str(target_path):
            row["ep_meta_path"] = str(target_path)
            summary["rewritten_manifest_rows"] += 1

    if rewrite_selected_seeds:
        _write_selected_manifest(selected_seeds, fieldnames, selected_rows)
    return summary


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.run_root is not None:
        return (
            args.run_root / "raw_rollouts",
            args.run_root / "manifests" / "selected_instruction_seeds.tsv",
            args.run_root / "ep_meta",
        )
    if args.raw_rollouts is None or args.selected_seeds is None or args.ep_meta_root is None:
        raise ValueError("Provide --run-root or all of --raw-rollouts/--selected-seeds/--ep-meta-root")
    return args.raw_rollouts, args.selected_seeds, args.ep_meta_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--raw-rollouts", type=Path, default=None)
    parser.add_argument("--selected-seeds", type=Path, default=None)
    parser.add_argument("--ep-meta-root", type=Path, default=None)
    parser.add_argument("--rewrite-selected-seeds", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_rollouts, selected_seeds, ep_meta_root = _resolve_paths(args)
    summary = backfill_ep_meta(
        raw_rollouts=raw_rollouts,
        selected_seeds=selected_seeds,
        ep_meta_root=ep_meta_root,
        rewrite_selected_seeds=args.rewrite_selected_seeds,
    )
    print(" ".join(f"{key}={summary[key]}" for key in SUMMARY_KEYS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
