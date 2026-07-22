"""Load robocasa LeRobot *expert demo* datasets into the EpisodeData contract.

This is the cleaner counterpart to ``rollout_adapter.py``. Instead of our policy
rollouts (noisy, failures, kitchen tasks, 768-concatenated video), it reads the
robocasa v1.0 pretrain expert demonstrations stored in LeRobot v2 format at e.g.
``~/.cache/temporal_vla/datasets/robocasa/v1.0/pretrain/atomic/<Task>/<date>/lerobot/``.

Why these are a much better fit for INSIGHT segmentation than our rollouts:
  * EXPERT teleop (clean, all-success) — well-formed primitives, no stall/retry noise.
  * SEPARATE per-view mp4 (agentview_left/right, eye_in_hand), each 256x256 — no de-concat.
  * Full-rate 20fps trajectories (~200-250 frames) — finer segmentation granularity.
  * Real grasps in PickPlace tasks (action gripper_close has true transitions).
  * Native LeRobot format — frame index == step index (1:1).

state/action layout from ``meta/modality.json`` (PandaOmron, state[16] / action[12]):
  state:  base_position[0:3] base_rotation[3:7] eef_pos_rel[7:10]
          eef_rot_rel[10:14] gripper_qpos[14:16]
  action: base_motion[0:4] control_mode[4:5] eef_pos[5:8]
          eef_rot[8:11] gripper_close[11:12]

NOTE: these datasets already ship a per-frame ``progress`` column (task-level,
monotonic 0->1 over the whole episode — NOT INSIGHT's per-primitive reset). We
surface it in ``ep["meta"]["dataset_progress"]`` for comparison (see
docs/insight/02_progress_prediction.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

# modality.json dim slices (PandaOmron robocasa v1.0).
_S_EEF_POS = slice(7, 10)
_S_EEF_ROT = slice(10, 14)   # quaternion
_S_GRIPPER = 14              # gripper_qpos[0]
_A_EEF_POS = slice(5, 8)
_A_EEF_ROT = slice(8, 11)
_A_GRIPPER = 11             # gripper_close command

# LeRobot view key -> our EpisodeData view name.
_VIEW_MAP = {
    "observation.images.robot0_agentview_left": "exterior",
    "observation.images.robot0_agentview_right": "exterior2",
    "observation.images.robot0_eye_in_hand": "wrist",
}

_GRIPPER_TRANSITION_THRESH = 0.1


def _decode_video(path: Path) -> np.ndarray:
    """Decode an mp4 to [T,H,W,3] RGB uint8."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def _episode_paths(root: Path, episode_index: int) -> tuple[Path, dict[str, Path]]:
    """Resolve the parquet + per-view mp4 paths for an episode (chunked layout)."""
    info = json.loads((root / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunks_size
    pq = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    vids: dict[str, Path] = {}
    for vkey in _VIEW_MAP:
        vp = (root / "videos" / f"chunk-{chunk:03d}" / vkey
              / f"episode_{episode_index:06d}.mp4")
        if vp.is_file():
            vids[vkey] = vp
    return pq, vids


def _episode_task(root: Path, episode_index: int) -> str:
    """Language instruction for an episode (from meta/episodes.jsonl)."""
    ep_file = root / "meta" / "episodes.jsonl"
    for line in ep_file.read_text().splitlines():
        rec = json.loads(line)
        if int(rec["episode_index"]) == episode_index:
            tasks = rec.get("tasks") or []
            return tasks[0] if tasks else ""
    return ""


def load_lerobot_episode(
    root: str | Path,
    episode_index: int,
    views: Sequence[str] = ("exterior", "exterior2", "wrist"),
) -> dict[str, Any]:
    """Load one expert-demo episode into the EpisodeData contract.

    ``root`` is the ``.../lerobot/`` directory of a robocasa v1.0 pretrain task.
    Returns the same dict shape as ``rollout_adapter.load_episode``.
    """
    root = Path(root)
    pq, vids = _episode_paths(root, episode_index)
    df = pd.read_parquet(pq)
    n_steps = len(df)

    state = np.stack(df["observation.state"].values).astype(np.float32)   # [N,16]
    action = np.stack(df["action"].values).astype(np.float32)            # [N,12]

    ee_pos = state[:, _S_EEF_POS]                                        # [N,3]
    ee_quat = state[:, _S_EEF_ROT]                                       # [N,4]
    ee_delta = np.concatenate(
        [action[:, _A_EEF_POS], action[:, _A_EEF_ROT]], axis=1          # [N,6]
    )
    gripper = action[:, _A_GRIPPER].astype(np.float32)                   # [N]
    has_gripper = bool(np.any(np.abs(np.diff(gripper)) > _GRIPPER_TRANSITION_THRESH))

    # Decode requested views (skip any that are missing on disk).
    frames: dict[str, np.ndarray] = {}
    for vkey, vname in _VIEW_MAP.items():
        if vname in views and vkey in vids:
            frames[vname] = _decode_video(vids[vkey])

    # Full-rate LeRobot: frame index == step index (1:1).
    T = next((v.shape[0] for v in frames.values()), n_steps)
    frame_to_step = np.arange(T, dtype=int)
    if T != n_steps:  # be defensive if a view decoded short
        frame_to_step = np.clip(
            np.round(np.arange(T) * n_steps / max(T, 1)).astype(int), 0, n_steps - 1
        )

    dataset_progress = (df["progress"].to_numpy().astype(np.float32)
                        if "progress" in df.columns else None)

    return {
        "task": _episode_task(root, episode_index),
        "success": 1,  # expert demos are all successes
        "n_steps": int(n_steps),
        "frames": frames,
        "frame_to_step": frame_to_step,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "ee_delta": ee_delta,
        "gripper": gripper,
        "has_gripper": has_gripper,
        "meta": {
            "source": "lerobot_expert",
            "root": str(root),
            "episode_index": int(episode_index),
            "task_name": root.parent.parent.name,  # <Task>/<date>/lerobot -> <Task>
            "dataset_progress": dataset_progress,
        },
    }


def list_lerobot_episodes(root: str | Path) -> list[int]:
    """All episode indices present in a LeRobot dataset root."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    return list(range(int(info.get("total_episodes", 0))))


def find_task_roots(atomic_root: str | Path, task_substr: str | None = None) -> list[Path]:
    """Find ``.../lerobot/`` roots under a robocasa pretrain/atomic dir.

    Each task is ``<atomic_root>/<Task>/<date>/lerobot``.
    """
    atomic_root = Path(atomic_root)
    roots = sorted(atomic_root.glob("*/*/lerobot"))
    if task_substr:
        roots = [r for r in roots
                 if task_substr.lower() in r.parent.parent.name.lower()]
    return roots


def _selftest() -> None:
    base = Path.home() / ".cache/temporal_vla/datasets/robocasa/v1.0/pretrain/atomic"
    roots = find_task_roots(base)
    print(f"found {len(roots)} task roots; e.g. {[r.parent.parent.name for r in roots[:5]]}")
    pp = find_task_roots(base, "PickPlaceCounterToStove")[0]
    ep = load_lerobot_episode(pp, 0)
    print(f"task={ep['task']!r}")
    print(f"n_steps={ep['n_steps']} has_gripper={ep['has_gripper']}")
    for k, v in ep["frames"].items():
        print(f"  view {k}: {v.shape}")
    print(f"ee_pos {ep['ee_pos'].shape} ee_delta {ep['ee_delta'].shape} "
          f"gripper[min,max]=({ep['gripper'].min():.2f},{ep['gripper'].max():.2f})")
    dp = ep["meta"]["dataset_progress"]
    print(f"dataset_progress: {None if dp is None else (round(float(dp.min()),2), round(float(dp.max()),2))}")


if __name__ == "__main__":
    _selftest()
