#!/usr/bin/env python3
"""Build deterministic SAFE train/CP/test splits for GR00T N1.5 rollout PKLs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys


SAFE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SAFE_ROOT))

from _common.split_lib import (  # noqa: E402
    RolloutFile,
    parse_rollout,
    rollout_stem,
    symlink_relative,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--cp-ratio", "--cal-ratio", dest="cp_ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def collect_rollouts(source_roots: list[Path]) -> list[RolloutFile]:
    rollouts: list[RolloutFile] = []
    for root in source_roots:
        for path in sorted(root.rglob("*.pkl")):
            try:
                rollout = parse_rollout(path.resolve(), source_root=root.resolve())
            except ValueError:
                continue
            rollouts.append(rollout)
    return rollouts


def _label_counts_for_target(groups: dict[int, list[RolloutFile]], target_count: int) -> dict[int, int]:
    total = sum(len(items) for items in groups.values())
    raw = {label: target_count * len(items) / total for label, items in groups.items()}
    counts = {label: min(int(value), len(groups[label])) for label, value in raw.items()}

    while sum(counts.values()) < target_count:
        candidates = [
            (raw[label] - counts[label], label)
            for label in groups
            if counts[label] < len(groups[label])
        ]
        if not candidates:
            break
        _, label = max(candidates)
        counts[label] += 1

    while sum(counts.values()) > target_count:
        candidates = [
            (counts[label] - raw[label], label)
            for label in groups
            if counts[label] > 0
        ]
        _, label = max(candidates)
        counts[label] -= 1

    return counts


def take_stratified(
    groups: dict[int, list[RolloutFile]],
    target_count: int,
) -> list[RolloutFile]:
    counts = _label_counts_for_target(groups, target_count)
    selected: list[RolloutFile] = []
    for label, count in counts.items():
        selected.extend(groups[label][:count])
        del groups[label][:count]
    return selected


def stratified_task_split(
    task_items: list[RolloutFile],
    rng: random.Random,
    ratios: tuple[float, float, float],
) -> dict[str, list[RolloutFile]]:
    n_total = len(task_items)
    train_n = round(n_total * ratios[0])
    cp_n = round(n_total * ratios[1])
    test_n = n_total - train_n - cp_n

    by_label = {
        0: [item for item in task_items if item.success == 0],
        1: [item for item in task_items if item.success == 1],
    }
    for label_items in by_label.values():
        rng.shuffle(label_items)

    out = {
        "train": take_stratified(by_label, train_n),
        "cp": take_stratified(by_label, cp_n),
    }
    out["test"] = by_label[0] + by_label[1]
    if len(out["test"]) != test_n:
        raise RuntimeError(f"Internal split error: expected {test_n} test items, got {len(out['test'])}")

    for split_items in out.values():
        split_items.sort(
            key=lambda item: (
                item.task_id,
                item.success,
                item.source_root.name if item.source_root is not None else "",
                item.episode_idx,
                str(item.pkl_path),
            )
        )
    return out


def ensure_empty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def link_split(output_root: Path, splits_by_task: dict[int, dict[str, list[RolloutFile]]]) -> list[dict[str, str | int]]:
    manifest_rows: list[dict[str, str | int]] = []
    next_episode_idx = {task_id: 0 for task_id in splits_by_task}

    for task_id in sorted(splits_by_task):
        for split_name in ("train", "cp", "test"):
            split_dir = output_root / split_name / f"task{task_id}"
            split_dir.mkdir(parents=True, exist_ok=True)
            for item in splits_by_task[task_id][split_name]:
                new_episode_idx = next_episode_idx[task_id]
                next_episode_idx[task_id] += 1
                new_name = f"{rollout_stem(task_id, new_episode_idx, item.success)}.pkl"
                link_path = split_dir / new_name
                symlink_relative(item.pkl_path, link_path)
                manifest_rows.append(
                    {
                        "split": split_name,
                        "task_id": task_id,
                        "success": item.success,
                        "new_episode_idx": new_episode_idx,
                        "new_path": str(link_path),
                        "source_root": str(item.source_root),
                        "source_path": str(item.pkl_path),
                        "source_filename": item.pkl_path.name,
                        "source_episode_idx": item.episode_idx,
                    }
                )
    return manifest_rows


def write_manifest(output_root: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path = output_root / "split_manifest.tsv"
    fieldnames = [
        "split",
        "task_id",
        "success",
        "new_episode_idx",
        "new_path",
        "source_root",
        "source_path",
        "source_filename",
        "source_episode_idx",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int]]) -> None:
    print("split\ttask_id\tn\tsuccess\tfail")
    for split_name in ("train", "cp", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        for task_id in sorted({int(row["task_id"]) for row in split_rows}):
            task_rows = [row for row in split_rows if int(row["task_id"]) == task_id]
            n_success = sum(int(row["success"]) for row in task_rows)
            print(f"{split_name}\t{task_id}\t{len(task_rows)}\t{n_success}\t{len(task_rows) - n_success}")


def main() -> None:
    args = parse_args()
    ratios = (args.train_ratio, args.cp_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    source_roots = [Path(path).expanduser() for path in args.source_root]
    output_root = Path(args.output_root).expanduser()
    ensure_empty_output(output_root)

    rollouts = collect_rollouts(source_roots)
    by_task = {}
    for item in rollouts:
        by_task.setdefault(item.task_id, []).append(item)

    rng = random.Random(args.seed)
    splits_by_task = {
        task_id: stratified_task_split(items, rng, ratios)
        for task_id, items in sorted(by_task.items())
    }
    rows = link_split(output_root, splits_by_task)
    write_manifest(output_root, rows)
    print(f"Wrote {len(rows)} linked rollouts to {output_root}")
    print_summary(rows)


if __name__ == "__main__":
    main()
