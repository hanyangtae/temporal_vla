#!/usr/bin/env python3
"""Prepare the seen4/unseen2 SAFE split for GR00T N1.6 RoboCasa rollouts.

The SAFE paper/repo split has three logical splits:

* train: seen-task rollouts used to train the detector
* val_seen: held-out seen-task rollouts used for validation and CP calibration
* val_unseen: unseen-task rollouts used for final evaluation

This script fixes the unseen tasks explicitly instead of relying on random
task selection. The current small GR00T N1.6 RoboCasa run holds out one Open
task and one PnP task.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys


ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
SAFE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SAFE_ROOT))
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from _common.split_lib import (  # noqa: E402
    RolloutFile,
    collect_rollouts,
    count_success,
    link_rollout_files,
    safe_remove_split_root,
)
from run_config import RUN_ROOT, SPLIT_ROOT  # noqa: E402


TASKS = {
    0: ("CoffeeSetupMug", "other"),
    1: ("OpenSingleDoor", "open"),
    2: ("PnPCounterToCab", "pnp"),
    3: ("PnPSinkToCounter", "pnp"),
    4: ("PnPCounterToStove", "pnp"),
    5: ("OpenDrawer", "open"),
}

DEFAULT_UNSEEN_TASKS = ("OpenDrawer", "PnPCounterToCab")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=RUN_ROOT / "raw_rollouts",
        help="Rollout source root containing one task directory per task.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=SPLIT_ROOT,
        help="Destination split root to create.",
    )
    parser.add_argument(
        "--unseen-task",
        action="append",
        default=None,
        help=(
            "Task name to hold out as unseen. Repeat exactly twice. "
            f"Default: {', '.join(DEFAULT_UNSEEN_TASKS)}."
        ),
    )
    parser.add_argument("--seen-train-ratio", type=float, default=0.75)
    parser.add_argument("--rollouts-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing split root. Refuses paths outside safe_split_*.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check counts and print the planned split without creating links.",
    )
    return parser.parse_args()

def task_ids_by_name(task_names: list[str]) -> list[int]:
    name_to_id = {task_name: task_id for task_id, (task_name, _) in TASKS.items()}
    unknown = sorted(set(task_names) - set(name_to_id))
    if unknown:
        raise ValueError(f"Unknown unseen task(s): {unknown}")
    return [name_to_id[name] for name in task_names]


def validate_unseen_tasks(unseen_task_ids: list[int]) -> None:
    if len(unseen_task_ids) != 2:
        raise ValueError(f"Expected exactly 2 unseen tasks, got {len(unseen_task_ids)}")
    categories = sorted(TASKS[task_id][1] for task_id in unseen_task_ids)
    if categories != ["open", "pnp"]:
        raise ValueError(f"Expected one open and one pnp unseen task, got categories={categories}")


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

    return {
        "train": train,
        "val_seen": val_seen,
        "val_unseen": val_unseen,
    }

def print_summary(splits: dict[str, list[RolloutFile]]) -> None:
    for split_name, rollouts in splits.items():
        n_success, n_fail = count_success(rollouts)
        print(f"{split_name}: {len(rollouts)} rollouts, {n_success} success/{n_fail} fail")
        for task_id in sorted({r.task_id for r in rollouts}):
            task_rollouts = [r for r in rollouts if r.task_id == task_id]
            task_name, category = TASKS[task_id]
            task_success, task_fail = count_success(task_rollouts)
            print(
                f"  task{task_id} {task_name} ({category}): "
                f"{len(task_rollouts)} rollouts, {task_success} success/{task_fail} fail"
            )


def write_manifests(split_root: Path, splits: dict[str, list[RolloutFile]]) -> None:
    manifest_path = split_root / "manifest.tsv"
    summary_path = split_root / "summary.tsv"
    split_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as manifest:
        manifest.write("split\ttask_id\ttask\tcategory\tepisode_idx\tsuccess\tsource_path\n")
        for split_name in ("train", "val_seen", "val_unseen"):
            for rollout in sorted(splits[split_name], key=lambda r: (r.task_id, r.episode_idx)):
                dst_dir = split_root / split_name / rollout.task_name
                link_rollout_files(rollout, dst_dir)
                manifest.write(
                    f"{split_name}\t{rollout.task_id}\t{rollout.task_name}\t{rollout.category}\t"
                    f"{rollout.episode_idx}\t{rollout.success}\t{rollout.pkl_path}\n"
                )

    with summary_path.open("w") as summary:
        summary.write("split\ttask_id\ttask\tcategory\ttotal\tsuccess\tfailure\tsr\n")
        for split_name in ("train", "val_seen", "val_unseen"):
            rollouts = splits[split_name]
            n_success, n_fail = count_success(rollouts)
            summary.write(
                f"{split_name}\tALL\tALL\tALL\t{len(rollouts)}\t{n_success}\t{n_fail}\t"
                f"{n_success / len(rollouts):.6f}\n"
            )
            for task_id in sorted({r.task_id for r in rollouts}):
                task_rollouts = [r for r in rollouts if r.task_id == task_id]
                task_name, category = TASKS[task_id]
                task_success, task_fail = count_success(task_rollouts)
                summary.write(
                    f"{split_name}\t{task_id}\t{task_name}\t{category}\t{len(task_rollouts)}\t"
                    f"{task_success}\t{task_fail}\t{task_success / len(task_rollouts):.6f}\n"
                )

    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")


def main() -> None:
    args = parse_args()
    unseen_task_names = args.unseen_task or list(DEFAULT_UNSEEN_TASKS)
    unseen_task_ids = task_ids_by_name(unseen_task_names)
    validate_unseen_tasks(unseen_task_ids)

    by_task = collect_rollouts(
        args.source_root,
        TASKS,
        rollouts_per_task=args.rollouts_per_task,
    )
    splits = split_rollouts(
        by_task=by_task,
        unseen_task_ids=unseen_task_ids,
        seen_train_ratio=args.seen_train_ratio,
        seed=args.seed,
    )

    print(f"source_root={args.source_root}")
    print(f"split_root={args.split_root}")
    print(f"seed={args.seed}")
    print(f"unseen_tasks={[TASKS[task_id][0] for task_id in unseen_task_ids]}")
    print_summary(splits)

    if args.dry_run:
        return

    if args.force:
        safe_remove_split_root(args.split_root, allowed_prefix="safe_split_")
    elif args.split_root.exists():
        raise FileExistsError(f"Split root already exists. Use --force to replace: {args.split_root}")

    write_manifests(args.split_root, splits)


if __name__ == "__main__":
    main()
