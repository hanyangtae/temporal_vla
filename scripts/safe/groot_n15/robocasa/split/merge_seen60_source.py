#!/usr/bin/env python3
"""Merge SAFE GR00T N1.5 rollout shards into one canonical source tree."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-split-root", required=True)
    parser.add_argument("--new-source-root", required=True)
    parser.add_argument("--new-split-root", required=True)
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _task_dir_name(row: dict[str, str]) -> str:
    return Path(row["source_path"]).parent.name


def _new_stem(task_id: int, episode_idx: int, success: int) -> str:
    return f"task{task_id}--ep{episode_idx:03d}--succ{success}"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.link(src, dst)


def _source_root_for(row: dict[str, str]) -> Path:
    return Path(row["source_root"]).resolve()


def main() -> None:
    args = parse_args()
    old_split_root = Path(args.old_split_root).resolve()
    new_source_root = Path(args.new_source_root).resolve()
    new_split_root = Path(args.new_split_root).resolve()
    manifest_path = old_split_root / "split_manifest.tsv"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if new_source_root.exists() and any(new_source_root.iterdir()):
        raise FileExistsError(f"New source root exists and is not empty: {new_source_root}")
    if new_split_root.exists() and any(new_split_root.iterdir()):
        raise FileExistsError(f"New split root exists and is not empty: {new_split_root}")

    rows = _load_manifest(manifest_path)
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 manifest rows, got {len(rows)}")

    source_manifest_rows: list[dict[str, str | int]] = []
    split_manifest_rows: list[dict[str, str | int]] = []

    for row in rows:
        split_name = "cp" if row["split"] == "cal" else row["split"]
        task_id = int(row["task_id"])
        success = int(row["success"])
        episode_idx = int(row["new_episode_idx"])
        task_dir = _task_dir_name(row)
        stem = _new_stem(task_id, episode_idx, success)
        old_pkl = Path(row["source_path"]).resolve()
        source_dir = new_source_root / task_dir
        new_pkl = source_dir / f"{stem}.pkl"

        for suffix in [".pkl", ".csv", ".mp4"]:
            _link_or_copy(old_pkl.with_suffix(suffix), source_dir / f"{stem}{suffix}", args.link_mode)

        old_log = old_pkl.parent / f"ep{int(row['source_episode_idx'])}.log"
        _link_or_copy(old_log, source_dir / f"{stem}.log", args.link_mode)

        split_dir = new_split_root / split_name / f"task{task_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_path = split_dir / f"{stem}.pkl"
        if split_path.exists() or split_path.is_symlink():
            split_path.unlink()
        split_path.symlink_to(new_pkl)

        source_manifest_rows.append(
            {
                "task_id": task_id,
                "task_dir": task_dir,
                "success": success,
                "episode_idx": episode_idx,
                "source_path": str(new_pkl),
                "old_source_root": row["source_root"],
                "old_source_path": row["source_path"],
                "old_source_filename": row["source_filename"],
                "old_source_episode_idx": row["source_episode_idx"],
                "old_split": row["split"],
                "split": split_name,
            }
        )
        split_manifest_rows.append(
            {
                "split": split_name,
                "task_id": task_id,
                "success": success,
                "new_episode_idx": episode_idx,
                "new_path": str(split_path),
                "source_root": str(new_source_root),
                "source_path": str(new_pkl),
                "source_filename": new_pkl.name,
                "source_episode_idx": episode_idx,
                "old_source_root": row["source_root"],
                "old_source_path": row["source_path"],
                "old_source_episode_idx": row["source_episode_idx"],
            }
        )

    source_manifest_path = new_source_root / "source_manifest.tsv"
    source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with source_manifest_path.open("w", newline="") as f:
        fieldnames = [
            "task_id",
            "task_dir",
            "success",
            "episode_idx",
            "source_path",
            "old_source_root",
            "old_source_path",
            "old_source_filename",
            "old_source_episode_idx",
            "old_split",
            "split",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(source_manifest_rows)

    split_manifest_path = new_split_root / "split_manifest.tsv"
    with split_manifest_path.open("w", newline="") as f:
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
            "old_source_root",
            "old_source_path",
            "old_source_episode_idx",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(split_manifest_rows)

    print(f"Wrote merged source: {new_source_root}")
    print(f"Wrote split: {new_split_root}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
