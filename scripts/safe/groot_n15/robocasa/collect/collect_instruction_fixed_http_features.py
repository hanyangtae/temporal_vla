#!/usr/bin/env python3
"""Run N1.5 HTTP feature collection from an instruction-fixed seed manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
COLLECT_SCRIPT = Path(__file__).resolve().with_name("http_feature_collect.py")


@dataclass(frozen=True)
class SelectedSeedRow:
    cell_index: int
    cell_id: str
    task: str
    env_name: str
    scenario_seed: int
    instruction: str
    canonical_instruction: str
    ep_meta_path: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "SelectedSeedRow":
        return cls(
            cell_index=int(row["cell_index"]),
            cell_id=row["cell_id"],
            task=row["task"],
            env_name=row["env_name"],
            scenario_seed=int(row["scenario_seed"]),
            instruction=row["instruction"],
            canonical_instruction=row["canonical_instruction"],
            ep_meta_path=row["ep_meta_path"],
        )


@dataclass(frozen=True)
class CollectCommand:
    cell_id: str
    scenario_seed: int
    episode_idx: int
    argv: list[str]


def load_selected_seed_rows(
    path: Path,
    *,
    cell_ids: list[str] | None = None,
) -> list[SelectedSeedRow]:
    requested = None if cell_ids is None else set(cell_ids)
    with path.open(newline="") as f:
        rows = [SelectedSeedRow.from_row(row) for row in csv.DictReader(f, delimiter="\t")]
    if requested is None:
        return rows
    filtered = [row for row in rows if row.cell_id in requested]
    found = {row.cell_id for row in filtered}
    missing = requested - found
    if missing:
        raise ValueError(f"cell_id not found in selected manifest: {', '.join(sorted(missing))}")
    return filtered


def limit_rows_per_cell(
    rows: list[SelectedSeedRow],
    *,
    max_episodes_per_cell: int | None,
) -> list[SelectedSeedRow]:
    if max_episodes_per_cell is None:
        return rows
    if max_episodes_per_cell < 1:
        raise ValueError("--max-episodes-per-cell must be >= 1")

    counts_by_cell: dict[str, int] = {}
    limited: list[SelectedSeedRow] = []
    for row in rows:
        count = counts_by_cell.get(row.cell_id, 0)
        if count >= max_episodes_per_cell:
            continue
        counts_by_cell[row.cell_id] = count + 1
        limited.append(row)
    return limited


def build_collect_commands(
    rows: list[SelectedSeedRow],
    *,
    output_dir: Path,
    ep_meta_dir: Path,
    vla_server: str,
    n_action_steps: int,
    max_episode_steps: int | None,
    video_fps: int,
    steps_per_render: int,
    wait_ready: bool,
    replay_ep_meta: bool = False,
    skip_existing: bool = False,
    python_executable: str = sys.executable,
) -> list[CollectCommand]:
    counts_by_cell: dict[str, int] = {}
    commands: list[CollectCommand] = []

    for row in rows:
        episode_idx = counts_by_cell.get(row.cell_id, 0)
        counts_by_cell[row.cell_id] = episode_idx + 1
        if skip_existing and rollout_pkl_exists(
            output_dir=output_dir,
            task=row.task,
            cell_id=row.cell_id,
            cell_index=row.cell_index,
            episode_idx=episode_idx,
        ):
            continue
        argv = [
            python_executable,
            str(COLLECT_SCRIPT),
            "--vla-server",
            vla_server,
            "--task",
            row.task,
            "--env-name",
            row.env_name,
            "--output-dir",
            str(output_dir),
            "--task-id",
            str(row.cell_index),
            "--cell-id",
            row.cell_id,
            "--cell-index",
            str(row.cell_index),
            "--canonical-instruction",
            row.canonical_instruction,
            "--episode-start-idx",
            str(episode_idx),
            "--n-episodes",
            "1",
            "--seed",
            str(row.scenario_seed),
            "--inference-seed",
            str(row.scenario_seed),
            "--n-action-steps",
            str(n_action_steps),
            "--video-fps",
            str(video_fps),
            "--steps-per-render",
            str(steps_per_render),
            "--ep-meta-dir",
            str(ep_meta_dir),
        ]
        if replay_ep_meta:
            argv.extend(
                [
                    "--ep-meta-load-env-name",
                    row.env_name,
                ]
            )
        if max_episode_steps is not None:
            argv.extend(["--max-episode-steps", str(max_episode_steps)])
        if wait_ready:
            argv.append("--wait-ready")
        commands.append(
            CollectCommand(
                cell_id=row.cell_id,
                scenario_seed=row.scenario_seed,
                episode_idx=episode_idx,
                argv=argv,
            )
        )

    return commands


def rollout_pkl_exists(
    *,
    output_dir: Path,
    task: str,
    cell_id: str,
    cell_index: int,
    episode_idx: int,
) -> bool:
    cell_dir = output_dir / task / cell_id
    pattern = f"task{cell_index}--ep{episode_idx}--succ*.pkl"
    return any(cell_dir.glob(pattern))


def run_collect_commands(commands: list[CollectCommand], *, dry_run: bool) -> None:
    for command in commands:
        print(shlex.join(command.argv))
        if dry_run:
            continue
        subprocess.run(command.argv, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-seeds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ep-meta-dir", type=Path, required=True)
    parser.add_argument("--vla-server", default="http://127.0.0.1:8400")
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument(
        "--max-episodes-per-cell",
        type=int,
        default=None,
        help="Limit selected manifest rows per cell; useful for smoke runs.",
    )
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--wait-ready", action="store_true")
    parser.add_argument(
        "--replay-ep-meta",
        action="store_true",
        help=(
            "Import ep_meta manifests during collection. Off by default for "
            "instruction-fixed runs because some RoboCasa tasks do not encode "
            "all instruction-driving sampled state in ep_meta."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip selected manifest rows whose target task/cell episode pkl already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_selected_seed_rows(args.selected_seeds, cell_ids=args.cell_ids)
    rows = limit_rows_per_cell(rows, max_episodes_per_cell=args.max_episodes_per_cell)
    commands = build_collect_commands(
        rows,
        output_dir=args.output_dir,
        ep_meta_dir=args.ep_meta_dir,
        vla_server=args.vla_server,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        video_fps=args.video_fps,
        steps_per_render=args.steps_per_render,
        wait_ready=args.wait_ready,
        replay_ep_meta=args.replay_ep_meta,
        skip_existing=args.skip_existing,
    )
    run_collect_commands(commands, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
