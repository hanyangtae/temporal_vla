from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_SEEN60_SOURCE = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "split"
    / "merge_seen60_source.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_merge_seen60_source_under_test",
        MERGE_SEEN60_SOURCE,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_old_split_manifest(old_split_root: Path, old_source_root: Path) -> None:
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
    task_dir = old_source_root / "OpenCabinet"
    task_dir.mkdir(parents=True)
    rows = []
    for idx in range(300):
        success = idx % 2
        stem = f"task2--ep{idx:03d}--succ{success}"
        pkl_path = task_dir / f"{stem}.pkl"
        pkl_path.write_bytes(f"pkl-{idx}".encode("utf-8"))
        rows.append(
            {
                "split": ["train", "cal", "test"][idx % 3],
                "task_id": "2",
                "success": str(success),
                "new_episode_idx": str(idx),
                "new_path": str(old_split_root / stem),
                "source_root": str(old_source_root),
                "source_path": str(pkl_path),
                "source_filename": pkl_path.name,
                "source_episode_idx": str(idx),
            }
        )

    old_split_root.mkdir(parents=True)
    with (old_split_root / "split_manifest.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_merge_seen60_source_links_new_split_pkls_relative(tmp_path, monkeypatch):
    module = _load_module()
    old_split_root = tmp_path / "old_split"
    old_source_root = tmp_path / "old_source"
    new_source_root = tmp_path / "new_source"
    new_split_root = tmp_path / "new_split"
    _write_old_split_manifest(old_split_root, old_source_root)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_seen60_source.py",
            "--old-split-root",
            str(old_split_root),
            "--new-source-root",
            str(new_source_root),
            "--new-split-root",
            str(new_split_root),
            "--link-mode",
            "copy",
        ],
    )

    module.main()

    split_link = new_split_root / "train" / "task2" / "task2--ep000--succ0.pkl"
    assert split_link.is_symlink()
    raw_target = Path(os.readlink(split_link))
    assert not raw_target.is_absolute()
    assert split_link.resolve() == (
        new_source_root / "OpenCabinet" / "task2--ep000--succ0.pkl"
    ).resolve()
