from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import random


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SAFE_SPLITS = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "split"
    / "build_safe_splits.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_build_safe_splits_under_test",
        BUILD_SAFE_SPLITS,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rollout(root: Path, task_name: str, stem: str) -> Path:
    task_dir = root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = task_dir / f"{stem}.pkl"
    pkl_path.write_bytes(b"pkl")
    return pkl_path


def test_build_safe_splits_collects_shared_rollout_file_shape(tmp_path):
    module = _load_module()
    source_root = tmp_path / "source_a"
    pkl_path = _write_rollout(source_root, "OpenCabinet", "task2--ep007--succ1")

    rollouts = module.collect_rollouts([source_root])

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.task_id == 2
    assert rollout.episode_idx == 7
    assert rollout.success == 1
    assert rollout.pkl_path == pkl_path.resolve()
    assert rollout.source_root == source_root.resolve()


def test_build_safe_splits_link_manifest_preserves_source_metadata(tmp_path):
    module = _load_module()
    source_root = tmp_path / "source_a"
    for idx, success in enumerate([0, 1, 0, 1]):
        _write_rollout(source_root, "OpenCabinet", f"task2--ep{idx:03d}--succ{success}")

    rollouts = module.collect_rollouts([source_root])
    splits = {
        2: module.stratified_task_split(
            rollouts,
            random.Random(0),
            ratios=(0.5, 0.25, 0.25),
        )
    }
    rows = module.link_split(tmp_path / "split_out", splits)

    assert len(rows) == 4
    assert rows[0]["new_episode_idx"] == 0
    assert rows[0]["source_root"] == str(source_root.resolve())
    assert str(rows[0]["source_path"]).endswith(".pkl")
    link_path = Path(str(rows[0]["new_path"]))
    assert link_path.is_symlink()
    assert not Path(os.readlink(link_path)).is_absolute()
