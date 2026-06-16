from __future__ import annotations

from pathlib import Path

import pytest

from scripts.safe._common.split_lib import (
    collect_rollouts,
    count_success,
    link_rollout_files,
    parse_rollout,
    rollout_stem,
    safe_remove_split_root,
)


def _write_rollout_files(task_dir: Path, stem: str) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = task_dir / f"{stem}.pkl"
    pkl_path.write_bytes(b"pkl")
    (task_dir / f"{stem}.mp4").write_bytes(b"mp4")
    (task_dir / f"{stem}.csv").write_text("t,ok\n", encoding="utf-8")
    return pkl_path


def test_parse_rollout_extracts_task_episode_success_and_metadata(tmp_path):
    path = tmp_path / "task3--ep042--succ1.pkl"
    source_root = tmp_path / "source"

    rollout = parse_rollout(
        path,
        task_name="OpenCabinet",
        category="open",
        source_root=source_root,
    )

    assert rollout.task_id == 3
    assert rollout.episode_idx == 42
    assert rollout.success == 1
    assert rollout.pkl_path == path
    assert rollout.task_name == "OpenCabinet"
    assert rollout.category == "open"
    assert rollout.source_root == source_root


def test_rollout_stem_formats_canonical_safe_filename_stem():
    assert rollout_stem(task_id=2, episode_idx=7, success=1) == "task2--ep007--succ1"


def test_collect_rollouts_sorts_limits_and_accepts_optional_category(tmp_path):
    source_root = tmp_path / "raw_rollouts"
    task_dir = source_root / "OpenCabinet"
    _write_rollout_files(task_dir, "task2--ep010--succ1")
    _write_rollout_files(task_dir, "task2--ep002--succ0")
    _write_rollout_files(task_dir, "task2--ep005--succ1")

    by_task = collect_rollouts(
        source_root,
        {2: ("OpenCabinet", "open")},
        rollouts_per_task=2,
    )

    assert list(by_task) == [2]
    assert [r.episode_idx for r in by_task[2]] == [2, 5]
    assert [r.success for r in by_task[2]] == [0, 1]
    assert [r.category for r in by_task[2]] == ["open", "open"]
    assert count_success(by_task[2]) == (1, 1)


def test_collect_rollouts_rejects_task_id_mismatch(tmp_path):
    source_root = tmp_path / "raw_rollouts"
    _write_rollout_files(source_root / "OpenCabinet", "task7--ep001--succ1")

    with pytest.raises(ValueError, match="has task7"):
        collect_rollouts(source_root, {2: "OpenCabinet"})


def test_link_rollout_files_links_all_sidecars_relative(tmp_path):
    source_root = tmp_path / "raw_rollouts"
    pkl_path = _write_rollout_files(
        source_root / "OpenCabinet",
        "task2--ep001--succ1",
    )
    rollout = parse_rollout(pkl_path, task_name="OpenCabinet")

    linked = link_rollout_files(rollout, tmp_path / "split" / "train" / "OpenCabinet")

    assert [p.name for p in linked] == [
        "task2--ep001--succ1.csv",
        "task2--ep001--succ1.mp4",
        "task2--ep001--succ1.pkl",
    ]
    for dst in linked:
        assert dst.is_symlink()
        assert dst.resolve().read_bytes() == (pkl_path.parent / dst.name).read_bytes()


def test_safe_remove_split_root_requires_allowed_prefix(tmp_path):
    split_root = tmp_path / "safe_split_seen4"
    split_root.mkdir()
    unsafe_root = tmp_path / "raw_rollouts"
    unsafe_root.mkdir()

    safe_remove_split_root(split_root, allowed_prefix="safe_split_")
    assert not split_root.exists()

    with pytest.raises(ValueError, match="Refusing to remove"):
        safe_remove_split_root(unsafe_root, allowed_prefix="safe_split_")
    assert unsafe_root.exists()
