#!/usr/bin/env python3
"""Launch N1.5 instruction-fixed seed preselection shards in tmux."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess


DEFAULT_CELLS_CONFIG = Path("configs/robocasa/n15_instruction_fixed_cells.tsv")
DEFAULT_RUN_ROOT = Path(
    "outputs/eval/robocasa/groot_n15/"
    "target_instruction_fixed15_pathway_seedscan50_sharded_natural"
)

SESSION_SLUGS = {
    "ppdc_wooden_spoon": "ppdc_spoon",
    "open_cabinet_door": "open_cabinet",
    "open_drawer_right": "drawer_right",
    "open_drawer_left": "drawer_left",
    "close_toaster_oven_door": "close_toaster",
    "turn_on_microwave": "microwave",
    "turn_on_sink_faucet": "sink_faucet",
    "navigate_coffee_machine": "navigate_coffee",
    "coffee_setup_mug": "coffee_mug",
}


@dataclass(frozen=True)
class InstructionCell:
    cell_id: str
    task: str
    target_episodes: int = 50


@dataclass(frozen=True)
class LaunchPlan:
    cell_id: str
    session_name: str
    command: list[str]
    reason: str = "launch"


def load_cells(path: Path) -> list[InstructionCell]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        InstructionCell(
            cell_id=row["cell_id"],
            task=row["task"],
            target_episodes=int(row.get("target_episodes") or 50),
        )
        for row in rows
    ]


def count_seed_rows(path: Path, *, cell_id: str) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and "env_kwargs_json" in reader.fieldnames:
            return 0
        return sum(1 for row in reader if row.get("cell_id") == cell_id)


def completed_cell_ids(cells: list[InstructionCell], *, run_root: Path) -> set[str]:
    completed: set[str] = set()
    shard_dir = run_root / "manifests" / "shards"
    for cell in cells:
        count = sum(
            count_seed_rows(shard_path, cell_id=cell.cell_id)
            for shard_path in shard_dir.glob(f"{cell.cell_id}*.tsv")
            if not shard_path.stem.endswith("_scan_progress")
            and not shard_path.stem.endswith("_scan_samples")
        )
        if count >= cell.target_episodes:
            completed.add(cell.cell_id)
    return completed


def session_name(cell_id: str, *, prefix: str, shard_suffix: str = "") -> str:
    return f"{prefix}{SESSION_SLUGS.get(cell_id, cell_id)}{shard_suffix}"


def list_tmux_sessions() -> set[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "no server running" in stderr:
            return set()
        raise RuntimeError(f"tmux list-sessions failed: {stderr or result.returncode}")
    if result.stderr.strip():
        raise RuntimeError(f"tmux list-sessions stderr: {result.stderr.strip()}")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _quote_path(path: Path | str) -> str:
    return shlex.quote(str(path))


def build_tmux_command(
    cell: InstructionCell,
    *,
    session: str,
    run_root: Path,
    container_name: str,
    repo_root: Path,
    container_workdir: str,
    target_per_cell: int,
    max_seeds_per_cell: int,
    progress_interval: int,
    shard_suffix: str = "",
    seed_start: int | None = None,
) -> list[str]:
    shard_stem = f"{cell.cell_id}{shard_suffix}"
    shard_tsv = run_root / "manifests" / "shards" / f"{shard_stem}.tsv"
    ep_meta_dir = run_root / "ep_meta"
    log_path = run_root / "logs" / f"{shard_stem}.log"
    seed_start_arg = "" if seed_start is None else f"--seed-start {seed_start} "
    inner_command = (
        f"cd {_quote_path(repo_root)} && "
        f"docker exec -w {_quote_path(container_workdir)} {_quote_path(container_name)} "
        "python scripts/safe/groot_n15/robocasa/collect/select_instruction_seeds.py "
        f"--output-tsv {_quote_path(shard_tsv)} "
        f"--ep-meta-dir {_quote_path(ep_meta_dir)} "
        f"--cell-id {_quote_path(cell.cell_id)} "
        f"--target-per-cell {target_per_cell} "
        f"{seed_start_arg}"
        f"--max-seeds-per-cell {max_seeds_per_cell} "
        f"--progress-interval {progress_interval} "
        f"--resume >> {_quote_path(log_path)} 2>&1"
    )
    return ["tmux", "new-session", "-d", "-s", session, inner_command]


def plan_launches(
    cells: list[InstructionCell],
    *,
    active_sessions: set[str],
    completed_cell_ids: set[str] | None = None,
    selected_cell_ids: set[str] | None,
    session_prefix: str,
    run_root: Path,
    container_name: str,
    repo_root: Path,
    container_workdir: str,
    target_per_cell: int,
    max_seeds_per_cell: int,
    progress_interval: int,
    shard_suffix: str = "",
    seed_start: int | None = None,
    max_new_sessions: int | None,
) -> list[LaunchPlan]:
    plans: list[LaunchPlan] = []
    launches = 0
    for cell in cells:
        if selected_cell_ids is not None and cell.cell_id not in selected_cell_ids:
            continue
        session = session_name(cell.cell_id, prefix=session_prefix, shard_suffix=shard_suffix)
        if session in active_sessions:
            plans.append(LaunchPlan(cell.cell_id, session, [], reason="active"))
            continue
        if completed_cell_ids is not None and cell.cell_id in completed_cell_ids:
            plans.append(LaunchPlan(cell.cell_id, session, [], reason="complete"))
            continue
        if max_new_sessions is not None and launches >= max_new_sessions:
            plans.append(LaunchPlan(cell.cell_id, session, [], reason="max_new_sessions"))
            continue
        command = build_tmux_command(
            cell,
            session=session,
            run_root=run_root,
            container_name=container_name,
            repo_root=repo_root,
            container_workdir=container_workdir,
            target_per_cell=target_per_cell,
            max_seeds_per_cell=max_seeds_per_cell,
            progress_interval=progress_interval,
            shard_suffix=shard_suffix,
            seed_start=seed_start,
        )
        plans.append(LaunchPlan(cell.cell_id, session, command))
        launches += 1
    return plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-config", type=Path, default=DEFAULT_CELLS_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--container-name", default="temporal_vla-robocasa-run-3705634bbbf6")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--container-workdir", default="/temporal_vla")
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--session-prefix", default="n15_seedscan_")
    parser.add_argument("--shard-suffix", default="")
    parser.add_argument("--target-per-cell", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--max-seeds-per-cell", type=int, default=10000)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-new-sessions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cells = load_cells(args.cells_config)
    selected = None if args.cell_ids is None else set(args.cell_ids)
    args.run_root.joinpath("manifests", "shards").mkdir(parents=True, exist_ok=True)
    args.run_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    plans = plan_launches(
        cells,
        active_sessions=list_tmux_sessions(),
        completed_cell_ids=completed_cell_ids(cells, run_root=args.run_root),
        selected_cell_ids=selected,
        session_prefix=args.session_prefix,
        run_root=args.run_root,
        container_name=args.container_name,
        repo_root=args.repo_root,
        container_workdir=args.container_workdir,
        target_per_cell=args.target_per_cell,
        max_seeds_per_cell=args.max_seeds_per_cell,
        progress_interval=args.progress_interval,
        shard_suffix=args.shard_suffix,
        seed_start=args.seed_start,
        max_new_sessions=args.max_new_sessions,
    )
    for plan in plans:
        if plan.reason != "launch":
            print(f"[skip:{plan.reason}] {plan.cell_id} session={plan.session_name}")
            continue
        print(shlex.join(plan.command))
        if not args.dry_run:
            subprocess.run(plan.command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
