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
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil


TASKS = {
    0: "CloseFridge",
    1: "CloseToasterOvenDoor",
    2: "PickPlaceSinkToCounter",
    3: "OpenCabinet",
    4: "SlideDishwasherRack",
}

ROLLOUT_RE = re.compile(r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$")


@dataclass(frozen=True)
class RolloutFile:
    task_id: int
    episode_idx: int
    success: int
    pkl_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n15/"
            "rollouts_seen5_100ep_per_task_subset100"
        ),
        help="Rollout source root containing one task directory per task.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(
            "/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n15/"
            "split_seen5_trainval75_cp15_test10_subset100"
        ),
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


def parse_rollout(path: Path) -> RolloutFile:
    match = ROLLOUT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected rollout filename: {path}")
    return RolloutFile(
        task_id=int(match.group("task_id")),
        episode_idx=int(match.group("episode_idx")),
        success=int(match.group("success")),
        pkl_path=path,
    )


def collect_rollouts(source_root: Path) -> dict[int, list[RolloutFile]]:
    by_task: dict[int, list[RolloutFile]] = {task_id: [] for task_id in TASKS}
    for task_id, task_name in TASKS.items():
        task_dir = source_root / task_name
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing task directory: {task_dir}")
        for pkl_path in sorted(task_dir.glob("*.pkl")):
            rollout = parse_rollout(pkl_path)
            if rollout.task_id != task_id:
                raise ValueError(f"{pkl_path} is under {task_name}, but has task{rollout.task_id}")
            by_task[task_id].append(rollout)

    for task_id, rollouts in by_task.items():
        by_task[task_id] = sorted(rollouts, key=lambda r: r.episode_idx)
    return by_task


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


def safe_remove_split_root(split_root: Path) -> None:
    if not split_root.exists():
        return
    if not split_root.name.startswith("split_seen5_"):
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


def link_rollout_files(rollout: RolloutFile, dst_dir: Path) -> list[Path]:
    linked = []
    for src in sorted(rollout.pkl_path.parent.glob(f"{rollout.pkl_path.stem}.*")):
        dst = dst_dir / src.name
        symlink_relative(src, dst)
        linked.append(dst)
    return linked


def count_success(rollouts: list[RolloutFile]) -> tuple[int, int]:
    n_success = sum(r.success for r in rollouts)
    return n_success, len(rollouts) - n_success


def main() -> None:
    args = parse_args()
    by_task = collect_rollouts(args.source_root)
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
        safe_remove_split_root(args.split_root)
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
