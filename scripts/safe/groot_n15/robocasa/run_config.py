"""Shared paths and run identity for GR00T N1.5 RoboCasa scripts."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("GROOT_N15_REPO_ROOT", Path(__file__).resolve().parents[4])
)
OUT_ROOT = Path(
    os.environ.get(
        "GROOT_N15_OUT_ROOT",
        REPO_ROOT / "outputs/eval/robocasa/groot_n15",
    )
)

BASE_NEW_EMBODIMENT_CHECKPOINT = Path(
    os.environ.get(
        "GROOT_N15_BASE_NEW_EMBODIMENT_CHECKPOINT",
        REPO_ROOT / "outputs/checkpoints/groot_n15_base_pandaomron_new_embodiment",
    )
)

TARGET15_RUN_ID = os.environ.get("GROOT_N15_TARGET15_RUN_ID") or os.environ.get("RUN_ID")
TARGET15_RUN_ROOT = (
    OUT_ROOT / f"target15_seedpairs_{TARGET15_RUN_ID}"
    if TARGET15_RUN_ID
    else OUT_ROOT / "target15_seedpairs"
)

SEEN5_SOURCE_ROOT = Path(
    os.environ.get(
        "GROOT_N15_SEEN5_SOURCE_ROOT",
        OUT_ROOT / "rollouts_seen5_100ep_per_task_subset100",
    )
)
SEEN5_SPLIT_ROOT = Path(
    os.environ.get(
        "GROOT_N15_SEEN5_SPLIT_ROOT",
        OUT_ROOT / "split_seen5_trainval75_cp15_test10_subset100",
    )
)
