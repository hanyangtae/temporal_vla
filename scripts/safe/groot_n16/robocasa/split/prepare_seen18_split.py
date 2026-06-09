#!/usr/bin/env python3
"""Prepare a seen/unseen SAFE split for the seen18 GR00T N1.6 RoboCasa run.

Generalizes ``prepare_seen4_unseen2_split.py`` (which hardcodes 6 tasks and a
strict open+pnp unseen pair) to the full 18-task ``target_atomic_seen18`` run:

* tasks are discovered from ``raw_rollouts/`` (one directory per task),
* any number of unseen tasks may be held out (default: a category-spread set
  of four), and
* the materialized layout (``train``/``val_seen``/``val_unseen`` symlink trees
  + ``manifest.tsv`` + ``summary.tsv``) is exactly what SAFE
  ``failure_prob/data/groot_n16.py`` consumes (it globs ``**/*.pkl`` and infers
  the split from the parent directory name).

Defaults hold out one task per major category (close/open/pnp/turn) while
keeping the singleton categories (navigate/slide/other) entirely in the seen
pool so train retains category coverage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import random
import re
import shutil


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
)
DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "safe_split_seen18_4unseen_100ep"
)
DEFAULT_UNSEEN_TASKS = (
    "CloseFridge",
    "OpenCabinet",
    "PickPlaceSinkToCounter",
    "TurnOnMicrowave",
)

ROLLOUT_RE = re.compile(r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$")


def category_of(task_name: str) -> str:
    name = task_name.lower()
    if name.startswith("pickplace") or name.startswith("pnp"):
        return "pnp"
    if name.startswith("open"):
        return "open"
    if name.startswith("close"):
        return "close"
    if name.startswith("turn"):
        return "turn"
    if name.startswith("navigate"):
        return "navigate"
    if name.startswith("slide"):
        return "slide"
    return "other"


@dataclass(frozen=True)
class RolloutFile:
    task_id: int
    task_name: str
    category: str
    episode_idx: int
    success: int
    pkl_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_RUN_ROOT / "raw_rollouts")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--unseen-task",
        action="append",
        default=None,
        help=f"Task name to hold out as unseen (repeatable). Default: {', '.join(DEFAULT_UNSEEN_TASKS)}.",
    )
    parser.add_argument("--seen-train-ratio", type=float, default=0.75)
    parser.add_argument("--rollouts-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing split root. Refuses paths not named safe_split_*.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned split without linking.")
    return parser.parse_args()


def discover_tasks(source_root: Path) -> dict[int, tuple[str, str]]:
    """Map task_id -> (task_name, category) by scanning rollout directories."""
    tasks: dict[int, tuple[str, str]] = {}
    for task_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        pkls = sorted(task_dir.glob("task*--ep*--succ*.pkl"))
        if not pkls:
            continue
        match = ROLLOUT_RE.match(pkls[0].name)
        if match is None:
            raise ValueError(f"Unexpected rollout filename: {pkls[0]}")
        task_id = int(match.group("task_id"))
        if task_id in tasks and tasks[task_id][0] != task_dir.name:
            raise ValueError(f"task_id {task_id} maps to both {tasks[task_id][0]} and {task_dir.name}")
        tasks[task_id] = (task_dir.name, category_of(task_dir.name))
    if not tasks:
        raise FileNotFoundError(f"No task directories with rollouts under {source_root}")
    return tasks


def parse_rollout(path: Path, task_id: int, task_name: str, category: str) -> RolloutFile:
    match = ROLLOUT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected rollout filename: {path}")
    if int(match.group("task_id")) != task_id:
        raise ValueError(f"{path} encodes task{match.group('task_id')} but lives under {task_name} (task{task_id})")
    return RolloutFile(
        task_id=task_id,
        task_name=task_name,
        category=category,
        episode_idx=int(match.group("episode_idx")),
        success=int(match.group("success")),
        pkl_path=path,
    )


def collect_rollouts(
    source_root: Path, tasks: dict[int, tuple[str, str]], rollouts_per_task: int
) -> dict[int, list[RolloutFile]]:
    by_task: dict[int, list[RolloutFile]] = {}
    for task_id, (task_name, category) in tasks.items():
        task_dir = source_root / task_name
        rollouts = [
            parse_rollout(pkl_path, task_id, task_name, category)
            for pkl_path in sorted(task_dir.glob("*.pkl"))
        ]
        rollouts = sorted(rollouts, key=lambda r: r.episode_idx)
        if len(rollouts) < rollouts_per_task:
            raise ValueError(f"{task_name}: need {rollouts_per_task} rollouts, found {len(rollouts)}")
        by_task[task_id] = rollouts[:rollouts_per_task]
    return by_task


def resolve_unseen_task_ids(unseen_task_names: list[str], tasks: dict[int, tuple[str, str]]) -> list[int]:
    name_to_id = {name: task_id for task_id, (name, _) in tasks.items()}
    unknown = sorted(set(unseen_task_names) - set(name_to_id))
    if unknown:
        raise ValueError(f"Unknown unseen task(s): {unknown}. Available: {sorted(name_to_id)}")
    if not unseen_task_names:
        raise ValueError("At least one unseen task is required.")
    return [name_to_id[name] for name in unseen_task_names]


def split_rollouts(
    by_task: dict[int, list[RolloutFile]],
    unseen_task_ids: list[int],
    seen_train_ratio: float,
    seed: int,
) -> dict[str, list[RolloutFile]]:
    rng = random.Random(seed)
    train: list[RolloutFile] = []
    val_seen: list[RolloutFile] = []
    val_unseen: list[RolloutFile] = []
    for task_id in sorted(by_task):
        rollouts = list(by_task[task_id])
        if task_id in unseen_task_ids:
            val_unseen.extend(rollouts)
            continue
        rng.shuffle(rollouts)
        n_train = int(seen_train_ratio * len(rollouts))
        train.extend(rollouts[:n_train])
        val_seen.extend(rollouts[n_train:])
    return {"train": train, "val_seen": val_seen, "val_unseen": val_unseen}


def safe_remove_split_root(split_root: Path) -> None:
    if not split_root.exists():
        return
    if not split_root.name.startswith("safe_split_"):
        raise ValueError(f"Refusing to remove non-split path: {split_root}")
    shutil.rmtree(split_root)


def symlink_relative(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src, dst.parent)
    if dst.is_symlink() and os.readlink(dst) == rel_src:
        return
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.symlink_to(rel_src)


def link_rollout_files(rollout: RolloutFile, dst_dir: Path) -> None:
    for src in sorted(rollout.pkl_path.parent.glob(f"{rollout.pkl_path.stem}.*")):
        symlink_relative(src, dst_dir / src.name)


def count_success(rollouts: list[RolloutFile]) -> tuple[int, int]:
    n_success = sum(r.success for r in rollouts)
    return n_success, len(rollouts) - n_success


def print_summary(splits: dict[str, list[RolloutFile]], tasks: dict[int, tuple[str, str]]) -> None:
    for split_name, rollouts in splits.items():
        n_success, n_fail = count_success(rollouts)
        print(f"{split_name}: {len(rollouts)} rollouts, {n_success} success/{n_fail} fail")
        for task_id in sorted({r.task_id for r in rollouts}):
            task_rollouts = [r for r in rollouts if r.task_id == task_id]
            task_name, category = tasks[task_id]
            t_succ, t_fail = count_success(task_rollouts)
            print(f"  task{task_id} {task_name} ({category}): {len(task_rollouts)} rollouts, {t_succ} success/{t_fail} fail")


def write_manifests(
    split_root: Path, splits: dict[str, list[RolloutFile]], tasks: dict[int, tuple[str, str]]
) -> None:
    manifest_path = split_root / "manifest.tsv"
    summary_path = split_root / "summary.tsv"
    split_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as manifest:
        manifest.write("split\ttask_id\ttask\tcategory\tepisode_idx\tsuccess\tsource_path\n")
        for split_name in ("train", "val_seen", "val_unseen"):
            for rollout in sorted(splits[split_name], key=lambda r: (r.task_id, r.episode_idx)):
                link_rollout_files(rollout, split_root / split_name / rollout.task_name)
                manifest.write(
                    f"{split_name}\t{rollout.task_id}\t{rollout.task_name}\t{rollout.category}\t"
                    f"{rollout.episode_idx}\t{rollout.success}\t{rollout.pkl_path}\n"
                )

    with summary_path.open("w") as summary:
        summary.write("split\ttask_id\ttask\tcategory\ttotal\tsuccess\tfailure\tsr\n")
        for split_name in ("train", "val_seen", "val_unseen"):
            rollouts = splits[split_name]
            n_success, n_fail = count_success(rollouts)
            sr = n_success / len(rollouts) if rollouts else 0.0
            summary.write(
                f"{split_name}\tALL\tALL\tALL\t{len(rollouts)}\t{n_success}\t{n_fail}\t"
                f"{sr:.6f}\n"
            )
            for task_id in sorted({r.task_id for r in rollouts}):
                task_rollouts = [r for r in rollouts if r.task_id == task_id]
                task_name, category = tasks[task_id]
                t_succ, t_fail = count_success(task_rollouts)
                t_sr = t_succ / len(task_rollouts) if task_rollouts else 0.0
                summary.write(
                    f"{split_name}\t{task_id}\t{task_name}\t{category}\t{len(task_rollouts)}\t"
                    f"{t_succ}\t{t_fail}\t{t_sr:.6f}\n"
                )

    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {args.source_root}")

    tasks = discover_tasks(args.source_root)
    unseen_task_names = args.unseen_task or list(DEFAULT_UNSEEN_TASKS)
    unseen_task_ids = resolve_unseen_task_ids(unseen_task_names, tasks)

    by_task = collect_rollouts(args.source_root, tasks, args.rollouts_per_task)
    splits = split_rollouts(by_task, unseen_task_ids, args.seen_train_ratio, args.seed)

    print(f"source_root={args.source_root}")
    print(f"split_root={args.split_root}")
    print(f"seed={args.seed}  seen_train_ratio={args.seen_train_ratio}")
    print(f"discovered {len(tasks)} tasks; unseen={[tasks[t][0] for t in unseen_task_ids]}")
    print_summary(splits, tasks)

    if args.dry_run:
        return

    if args.force:
        safe_remove_split_root(args.split_root)
    elif args.split_root.exists():
        raise FileExistsError(f"Split root already exists. Use --force to replace: {args.split_root}")

    write_manifests(args.split_root, splits, tasks)


if __name__ == "__main__":
    main()
