#!/usr/bin/env python3
"""Materialize selected instruction seed ep_meta manifests before collection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[5]
COLLECT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))

from select_instruction_seeds import (  # noqa: E402
    extract_instruction,
    load_selected_seed_manifest,
    make_env,
    normalize_instruction,
)
from src.policies.groot.robocasa.scenario_replay import (  # noqa: E402
    get_robocasa_ep_meta,
    write_ep_meta_manifest,
)


SUMMARY_KEYS = [
    "selected_rows",
    "existing_json",
    "written_json",
    "dry_run_missing",
    "instruction_mismatches",
    "skipped_mismatches",
]


def _new_summary() -> dict[str, int]:
    return {key: 0 for key in SUMMARY_KEYS}


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def materialize_selected_ep_meta(
    *,
    selected_seeds: Path,
    cell_ids: list[str] | None = None,
    rewrite: bool = False,
    dry_run: bool = False,
    continue_on_mismatch: bool = False,
    robocasa_env_source: str = "robocasa365",
    make_env_fn: Callable[..., Any] = make_env,
) -> dict[str, int]:
    requested = None if cell_ids is None else set(cell_ids)
    rows = load_selected_seed_manifest(selected_seeds)
    summary = _new_summary()

    for row in rows:
        cell_id = str(row["cell_id"])
        if requested is not None and cell_id not in requested:
            continue
        summary["selected_rows"] += 1

        target_path = Path(str(row["ep_meta_path"]))
        if target_path.exists() and not rewrite:
            summary["existing_json"] += 1
            continue
        if dry_run:
            summary["dry_run_missing"] += int(not target_path.exists() or rewrite)
            continue

        scenario_seed = int(row["scenario_seed"])
        env_name = str(row["env_name"])
        task = str(row["task"])
        canonical = str(row["canonical_instruction"])
        env = make_env_fn(
            env_name,
            task=task,
            scenario_seed=scenario_seed,
            enable_render=False,
        )
        try:
            obs, _info = env.reset(seed=scenario_seed)
            ep_meta = get_robocasa_ep_meta(env)
            instruction = extract_instruction(obs, ep_meta)
        finally:
            _close_env(env)

        if normalize_instruction(instruction) != normalize_instruction(canonical):
            summary["instruction_mismatches"] += 1
            if continue_on_mismatch:
                summary["skipped_mismatches"] += 1
                continue
            raise RuntimeError(
                f"{cell_id} seed {scenario_seed}: instruction mismatch while "
                f"materializing ep_meta: {instruction!r} != {canonical!r}"
            )

        write_ep_meta_manifest(
            target_path,
            env_name=env_name,
            scenario_seed=scenario_seed,
            ep_meta=ep_meta,
            robocasa_env_source=robocasa_env_source,
            sort_keys=True,
        )
        summary["written_json"] += 1

    if requested is not None:
        found = {str(row["cell_id"]) for row in rows if str(row["cell_id"]) in requested}
        missing = requested - found
        if missing:
            raise ValueError(f"cell_id not found in selected manifest: {', '.join(sorted(missing))}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-seeds", type=Path, required=True)
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-mismatch",
        action="store_true",
        help="Continue scanning selected rows and skip rows whose current instruction mismatches.",
    )
    parser.add_argument("--robocasa-env-source", default="robocasa365")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = materialize_selected_ep_meta(
        selected_seeds=args.selected_seeds,
        cell_ids=args.cell_ids,
        rewrite=args.rewrite,
        dry_run=args.dry_run,
        continue_on_mismatch=args.continue_on_mismatch,
        robocasa_env_source=args.robocasa_env_source,
    )
    print(" ".join(f"{key}={summary[key]}" for key in SUMMARY_KEYS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
