"""Shared rollout split helpers for SAFE RoboCasa scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Mapping


ROLLOUT_RE = re.compile(r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$")


@dataclass(frozen=True)
class RolloutFile:
    task_id: int
    episode_idx: int
    success: int
    pkl_path: Path
    task_name: str = ""
    category: str = ""


def task_name_category(value: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(value, tuple):
        return value
    return value, ""


def parse_rollout(
    path: Path,
    *,
    task_name: str = "",
    category: str = "",
) -> RolloutFile:
    match = ROLLOUT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected rollout filename: {path}")
    return RolloutFile(
        task_id=int(match.group("task_id")),
        episode_idx=int(match.group("episode_idx")),
        success=int(match.group("success")),
        pkl_path=path,
        task_name=task_name,
        category=category,
    )


def collect_rollouts(
    source_root: Path,
    tasks: Mapping[int, str | tuple[str, str]],
    *,
    rollouts_per_task: int | None = None,
) -> dict[int, list[RolloutFile]]:
    by_task: dict[int, list[RolloutFile]] = {}
    for task_id, task_spec in tasks.items():
        task_name, category = task_name_category(task_spec)
        task_dir = source_root / task_name
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing task directory: {task_dir}")
        rollouts = [
            parse_rollout(pkl_path, task_name=task_name, category=category)
            for pkl_path in sorted(task_dir.glob("*.pkl"))
        ]
        for rollout in rollouts:
            if rollout.task_id != task_id:
                raise ValueError(
                    f"{rollout.pkl_path} is under {task_name}, but has task{rollout.task_id}"
                )
        rollouts = sorted(rollouts, key=lambda r: r.episode_idx)
        if rollouts_per_task is not None:
            if len(rollouts) < rollouts_per_task:
                raise ValueError(
                    f"{task_name}: need {rollouts_per_task} rollouts, found {len(rollouts)}"
                )
            rollouts = rollouts[:rollouts_per_task]
        by_task[task_id] = rollouts
    return by_task


def safe_remove_split_root(split_root: Path, *, allowed_prefix: str) -> None:
    if not split_root.exists():
        return
    if not split_root.name.startswith(allowed_prefix):
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
