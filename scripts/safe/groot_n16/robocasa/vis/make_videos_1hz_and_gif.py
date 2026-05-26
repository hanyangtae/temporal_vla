#!/usr/bin/env python3
"""Re-encode v4 progress images into 1 Hz mp4 (7 videos) + 1 GIF (all-task)."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[5]
BASE = (REPO_ROOT / "outputs/eval/robocasa/groot_n16/safe_feature_vis"
        / "seen4_unseen2_openDrawer_pnpCab_100ep")
PER_TASK_DIR = BASE / "rollout_per_task_progress_v4_scaled"
ALL_TASK_DIR = BASE / "rollout_progress_7label_v4_scaled"
TASKS = ("CoffeeSetupMug", "OpenSingleDoor", "PnPCounterToCab",
         "PnPSinkToCounter", "PnPCounterToStove", "OpenDrawer")
MILESTONES_PCT = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
FPS = 1.0
FRAME_DURATION_MS = int(1000 / FPS)


def write_video(image_paths, out_path, fps):
    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for p in image_paths:
            img = cv2.imread(str(p))
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
    print(f"  saved {out_path}")


def write_gif(image_paths, out_path, duration_ms):
    frames = [Image.open(str(p)).convert("RGB") for p in image_paths]
    # downscale slightly for sane GIF size
    max_w = 1100
    if frames[0].width > max_w:
        ratio = max_w / frames[0].width
        new_size = (int(frames[0].width * ratio), int(frames[0].height * ratio))
        frames = [f.resize(new_size, Image.LANCZOS) for f in frames]
    # palette quantization for smaller file
    frames_p = [f.quantize(colors=256, method=Image.MEDIANCUT) for f in frames]
    frames_p[0].save(
        out_path,
        save_all=True,
        append_images=frames_p[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"  saved {out_path}")


def main():
    # Per-task mp4 (1 Hz)
    for task in TASKS:
        task_subdir = PER_TASK_DIR / task
        frames = [task_subdir / f"frame_p{pct:03d}.png" for pct in MILESTONES_PCT]
        out_mp4 = PER_TASK_DIR / f"{task}_progression.mp4"
        if out_mp4.exists():
            out_mp4.unlink()
        write_video(frames, out_mp4, FPS)

    # All-task mp4 (1 Hz)
    all_frames = [ALL_TASK_DIR / f"frame_p{pct:03d}.png" for pct in MILESTONES_PCT]
    out_mp4_all = ALL_TASK_DIR / "progress_7label_progression.mp4"
    if out_mp4_all.exists():
        out_mp4_all.unlink()
    write_video(all_frames, out_mp4_all, FPS)

    # All-task GIF (1 Hz = 1000 ms / frame)
    out_gif = ALL_TASK_DIR / "progress_7label_progression.gif"
    write_gif(all_frames, out_gif, FRAME_DURATION_MS)


if __name__ == "__main__":
    main()
