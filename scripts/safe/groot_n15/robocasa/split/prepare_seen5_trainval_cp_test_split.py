#!/usr/bin/env python3
"""Prepare GR00T N1.5 SAFE trainval/CP/test symlink splits.

SAFE's training script creates its own seen-task validation split from
``dataset.data_path`` using ``dataset.seen_train_ratio``. For the seen-task
reproduction layout, this script therefore creates only three physical splits:

* trainval: passed to SAFE train.py
* cp: used by external conformal calibration
* test: used by external final evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROBOCASA_ROOT = Path(__file__).resolve().parents[1]
SAFE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROBOCASA_ROOT))
sys.path.insert(0, str(SAFE_ROOT))

from run_config import SEEN5_SOURCE_ROOT, SEEN5_SPLIT_ROOT  # noqa: E402
from _common.split_lib import (  # noqa: E402
    RolloutFile,
    collect_rollouts,
    count_success,
    link_rollout_files,
    safe_remove_split_root,
)


TASKS = {
    0: "CloseFridge",
    1: "CloseToasterOvenDoor",
    2: "PickPlaceSinkToCounter",
    3: "OpenCabinet",
    4: "SlideDishwasherRack",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SEEN5_SOURCE_ROOT,
        help="Rollout source root containing one task directory per task.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=SEEN5_SPLIT_ROOT,
        help="Destination split root to create.",
    )
    parser.add_argument("--trainval-per-task", type=int, default=75)
    parser.add_argument("--cp-per-task", type=int, default=15)
    parser.add_argument("--test-per-task", type=int, default=10)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing split root. Refuses paths outside split_seen5_*.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check counts and print the planned split without creating links.",
    )
    return parser.parse_args()

def split_task(
    rollouts: list[RolloutFile],
    trainval_per_task: int,
    cp_per_task: int,
    test_per_task: int,
) -> dict[str, list[RolloutFile]]:
    total = trainval_per_task + cp_per_task + test_per_task
    if len(rollouts) < total:
        raise ValueError(f"Need {total} rollouts, found {len(rollouts)}")

    selected = rollouts[:total]
    return {
        "trainval": selected[:trainval_per_task],
        "cp": selected[trainval_per_task : trainval_per_task + cp_per_task],
        "test": selected[trainval_per_task + cp_per_task : total],
    }

def main() -> None:
    args = parse_args()
    by_task = collect_rollouts(args.source_root, TASKS)
    split_by_task = {
        task_id: split_task(
            rollouts,
            trainval_per_task=args.trainval_per_task,
            cp_per_task=args.cp_per_task,
            test_per_task=args.test_per_task,
        )
        for task_id, rollouts in by_task.items()
    }

    print(f"source_root={args.source_root}")
    print(f"split_root={args.split_root}")
    for task_id, task_name in TASKS.items():
        print(f"task{task_id} {task_name}: {len(by_task[task_id])} source rollouts")
        for split_name in ("trainval", "cp", "test"):
            split_rollouts = split_by_task[task_id][split_name]
            n_success, n_fail = count_success(split_rollouts)
            first_ep = split_rollouts[0].episode_idx
            last_ep = split_rollouts[-1].episode_idx
            print(
                f"  {split_name}: {len(split_rollouts)} "
                f"({n_success} success/{n_fail} fail), ep{first_ep:03d}..ep{last_ep:03d}"
            )

    if args.dry_run:
        return

    if args.force:
        safe_remove_split_root(args.split_root, allowed_prefix="split_seen5_")
    elif args.split_root.exists():
        raise FileExistsError(f"Split root already exists. Use --force to replace: {args.split_root}")

    manifest_path = args.split_root / "manifest.tsv"
    args.split_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as manifest:
        manifest.write("split\ttask_id\ttask\tepisode_idx\tsuccess\tpath\n")
        for task_id, task_name in TASKS.items():
            for split_name in ("trainval", "cp", "test"):
                dst_dir = args.split_root / split_name / f"task{task_id}"
                for rollout in split_by_task[task_id][split_name]:
                    link_rollout_files(rollout, dst_dir)
                    manifest.write(
                        f"{split_name}\t{task_id}\t{task_name}\t"
                        f"{rollout.episode_idx}\t{rollout.success}\t{rollout.pkl_path}\n"
                    )

    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
