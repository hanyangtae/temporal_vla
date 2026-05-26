#!/usr/bin/env python3
"""Make 2 Hz mp4 videos from 11-milestone progress images.

Per-task videos:  6 (one per task) in rollout_per_task_progress_v2_m11/
All-task video:   1 in rollout_progress_7label_v2_m11/
Total:            7 videos
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
PER_TASK_DIR = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_feature_vis"
    / "seen4_unseen2_openDrawer_pnpCab_100ep"
    / "rollout_per_task_progress_v2_m11"
)
ALL_TASK_DIR = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_feature_vis"
    / "seen4_unseen2_openDrawer_pnpCab_100ep"
    / "rollout_progress_7label_v2_m11"
)
TASKS = ("CoffeeSetupMug", "OpenSingleDoor", "PnPCounterToCab",
         "PnPSinkToCounter", "PnPCounterToStove", "OpenDrawer")
MILESTONES_PCT = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fps", type=float, default=2.0)
    return p.parse_args()


def write_video(image_paths: List[Path], out_path: Path, fps: float) -> None:
    if not image_paths:
        raise FileNotFoundError("no images provided")
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise FileNotFoundError(f"failed to read {image_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")
    try:
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"failed to read {p}")
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
    print(f"  saved {out_path} ({len(image_paths)} frames @ {fps} fps)")


def main():
    args = parse_args()

    # per-task videos
    for task in TASKS:
        frames = [PER_TASK_DIR / f"{task}_p{pct:03d}.png" for pct in MILESTONES_PCT]
        missing = [p for p in frames if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{task}: missing frames -> {missing}")
        out = PER_TASK_DIR / f"{task}_progression.mp4"
        write_video(frames, out, args.fps)

    # all-task 7-label video
    frames = [ALL_TASK_DIR / f"progress_p{pct:03d}.png" for pct in MILESTONES_PCT]
    missing = [p for p in frames if not p.exists()]
    if missing:
        raise FileNotFoundError(f"all-task: missing frames -> {missing}")
    out = ALL_TASK_DIR / "progress_7label_progression.mp4"
    write_video(frames, out, args.fps)


if __name__ == "__main__":
    main()
