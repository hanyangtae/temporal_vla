#!/usr/bin/env python3
"""Convert RoboCasa HDF5 train demos into GR00T-compatible LeRobot v2.1 datasets.

This writes one dataset directory per RoboCasa task with the following structure:

<output_root>/<task_name>/
  meta/
    info.json
    modality.json
    tasks.jsonl
    episodes.jsonl
    stats.json
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.robot0_agentview_left/episode_000000.mp4
  ...

The output schema targets Isaac-GR00T n1.5-release's LeRobot-compatible loader.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import h5py
import imageio.v2 as imageio
import jsonlines
import numpy as np
import pandas as pd


DEFAULT_CHUNK_SIZE = 1000
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

STATE_SPECS: list[tuple[str, str, int, str | None]] = [
    ("state.gripper_qpos", "robot0_gripper_qpos", 2, None),
    ("state.base_position", "robot0_base_pos", 3, None),
    ("state.base_rotation", "robot0_base_quat", 4, "quaternion"),
    ("state.end_effector_position_relative", "robot0_base_to_eef_pos", 3, None),
    ("state.end_effector_rotation_relative", "robot0_base_to_eef_quat", 4, "quaternion"),
    ("state.gripper_qvel", "robot0_gripper_qvel", 2, None),
    ("state.end_effector_position_absolute", "robot0_eef_pos", 3, None),
    ("state.end_effector_rotation_absolute", "robot0_eef_quat", 4, "quaternion"),
    ("state.joint_position", "robot0_joint_pos", 7, None),
    ("state.joint_position_cos", "robot0_joint_pos_cos", 7, None),
    ("state.joint_position_sin", "robot0_joint_pos_sin", 7, None),
    ("state.joint_velocity", "robot0_joint_vel", 7, None),
]

VIDEO_SPECS: list[tuple[str, str]] = [
    ("robot0_agentview_left", "robot0_agentview_left_image"),
    ("robot0_agentview_right", "robot0_agentview_right_image"),
    ("robot0_eye_in_hand", "robot0_eye_in_hand_image"),
]

ACTION_SPECS: list[tuple[str, int, bool, str | None]] = [
    ("action.eef_position", 3, False, None),
    ("action.eef_rotation", 3, False, "axis_angle"),
    ("action.gripper", 1, False, None),
    ("action.base", 4, False, None),
    ("action.control_mode", 1, False, None),
]
ACTION_DIM = sum(size for _, size, _, _ in ACTION_SPECS)
LANG_KEY = "annotation.human.action.task_description"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("."),
        help="Root directory containing RoboCasa HDF5 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("groot_lerobot_v21"),
        help="Directory where converted datasets will be written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "50_demos", "all"],
        help="HDF5 mask split to export. 'all' exports every existing demo in data/.",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Optional cap for quick sanity checks.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional list of task directory names to convert, e.g. PnPCabToCounter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output dataset directory before writing.",
    )
    return parser.parse_args()


def decode_str_list(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values.tolist()]


def find_hdf5_files(input_root: Path, selected_tasks: set[str] | None) -> list[Path]:
    files = sorted(input_root.glob("**/*.hdf5"))
    if selected_tasks is None:
        return files
    return [path for path in files if path.parts[0] in selected_tasks]


def infer_demo_names(file_handle: h5py.File, split: str) -> list[str]:
    demos = sorted(
        [name for name in file_handle["data"].keys() if name.startswith("demo")],
        key=lambda x: int(x.split("_")[1]),
    )
    if split == "all":
        return demos
    mask_values = decode_str_list(file_handle["mask"][split][()])
    existing = set(demos)
    filtered = [name for name in mask_values if name in existing]
    missing = [name for name in mask_values if name not in existing]
    if missing:
        print(
            f"[warn] {file_handle.filename}: mask/{split} references missing demos {missing}. "
            f"Exporting {len(filtered)} existing demos instead."
        )
    return filtered


def build_modality_meta() -> dict:
    state_meta = {}
    start = 0
    for full_key, _, size, rotation_type in STATE_SPECS:
        subkey = full_key.removeprefix("state.")
        entry = {
            "start": start,
            "end": start + size,
            "dtype": "float32",
            "absolute": True,
            "original_key": "observation.state",
        }
        if rotation_type is not None:
            entry["rotation_type"] = rotation_type
        state_meta[subkey] = entry
        start += size

    action_meta = {}
    start = 0
    for full_key, size, absolute, rotation_type in ACTION_SPECS:
        entry = {
            "start": start,
            "end": start + size,
            "dtype": "float32",
            "absolute": absolute,
            "original_key": "action",
        }
        if rotation_type is not None:
            entry["rotation_type"] = rotation_type
        action_meta[full_key.removeprefix("action.")] = entry
        start += size

    video_meta = {
        new_key: {"original_key": f"observation.images.{new_key}"} for new_key, _ in VIDEO_SPECS
    }

    annotation_meta = {
        "human.action.task_description": {"original_key": "task_index"},
    }

    return {
        "state": state_meta,
        "action": action_meta,
        "video": video_meta,
        "annotation": annotation_meta,
    }


def build_feature_meta(
    image_shape: tuple[int, int, int],
    fps: float,
) -> dict:
    height, width, channels = image_shape
    video_info = {
        "video.fps": float(fps),
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [sum(size for _, _, size, _ in STATE_SPECS)],
            "names": [f"state_{i}" for i in range(sum(size for _, _, size, _ in STATE_SPECS))],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [f"action_{i}" for i in range(ACTION_DIM)],
        },
        "timestamp": {"dtype": "float32", "shape": [1]},
        LANG_KEY: {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    for new_key, _ in VIDEO_SPECS:
        features[f"observation.images.{new_key}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "video_info": video_info,
        }
    return features


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def build_info(features: dict, fps: float, total_episodes: int, total_frames: int, total_tasks: int) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "PandaMobile",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(VIDEO_SPECS),
        "total_chunks": math.ceil(total_episodes / DEFAULT_CHUNK_SIZE) if total_episodes else 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": float(fps),
        "splits": {"train": "0:100"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }


def write_video(video_path: Path, frames: np.ndarray, fps: float) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        video_path,
        frames,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )


def concat_state(obs_group: h5py.Group) -> np.ndarray:
    parts = [np.asarray(obs_group[source_key][()], dtype=np.float32) for _, source_key, _, _ in STATE_SPECS]
    return np.concatenate(parts, axis=1)


def compute_stats(state_arrays: list[np.ndarray], action_arrays: list[np.ndarray]) -> dict:
    state = np.concatenate(state_arrays, axis=0).astype(np.float32)
    action = np.concatenate(action_arrays, axis=0).astype(np.float32)

    def summarize(array: np.ndarray) -> dict:
        return {
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }

    return {
        "observation.state": summarize(state),
        "action": summarize(action),
    }


def write_tasks(tasks_path: Path, ordered_tasks: list[str]) -> None:
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(tasks_path, mode="w") as writer:
        for index, task in enumerate(ordered_tasks):
            writer.write({"task_index": index, "task": task})


def convert_file(hdf5_path: Path, output_root: Path, split: str, max_episodes: int | None, overwrite: bool) -> None:
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        task_name = env_args["env_name"]
        dataset_root = output_root / task_name

        if dataset_root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output dataset already exists: {dataset_root}. Use --overwrite to replace it."
                )
            shutil.rmtree(dataset_root)

        demo_names = infer_demo_names(f, split)
        if max_episodes is not None:
            demo_names = demo_names[:max_episodes]
        if not demo_names:
            print(f"[skip] {hdf5_path}: no demos selected for split '{split}'")
            return

        fps = float(env_args["env_kwargs"].get("control_freq", 20))
        first_demo = f["data"][demo_names[0]]
        first_image = np.asarray(first_demo["obs"][VIDEO_SPECS[0][1]][0])
        features = build_feature_meta(first_image.shape, fps)
        modality_meta = build_modality_meta()

        ordered_tasks: list[str] = []
        task_to_index: dict[str, int] = {}
        episode_rows: list[dict] = []
        state_arrays: list[np.ndarray] = []
        action_arrays: list[np.ndarray] = []
        global_index = 0

        for episode_index, demo_name in enumerate(demo_names):
            demo = f["data"][demo_name]
            obs = demo["obs"]
            ep_meta = json.loads(demo.attrs["ep_meta"])
            task_text = ep_meta.get("lang", task_name)
            if task_text not in task_to_index:
                task_to_index[task_text] = len(ordered_tasks)
                ordered_tasks.append(task_text)
            task_index = task_to_index[task_text]

            state = concat_state(obs)
            action = np.asarray(demo["actions"][()], dtype=np.float32)
            rewards = np.asarray(demo["rewards"][()], dtype=np.float32)
            dones = np.asarray(demo["dones"][()], dtype=bool)
            num_frames = state.shape[0]
            timestamps = np.arange(num_frames, dtype=np.float32) / fps

            if action.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"{hdf5_path} {demo_name}: expected action dim {ACTION_DIM}, got {action.shape[1]}"
                )

            frame_df = pd.DataFrame(
                {
                    "observation.state": [row for row in state],
                    "action": [row for row in action],
                    "timestamp": timestamps,
                    LANG_KEY: np.full(num_frames, task_index, dtype=np.int64),
                    "task_index": np.full(num_frames, task_index, dtype=np.int64),
                    "episode_index": np.full(num_frames, episode_index, dtype=np.int64),
                    "frame_index": np.arange(num_frames, dtype=np.int64),
                    "index": np.arange(global_index, global_index + num_frames, dtype=np.int64),
                    "next.reward": rewards,
                    "next.done": dones,
                }
            )

            data_path = dataset_root / DATA_PATH_TEMPLATE.format(
                episode_chunk=episode_index // DEFAULT_CHUNK_SIZE,
                episode_index=episode_index,
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            frame_df.to_parquet(data_path, index=False)

            for new_key, source_key in VIDEO_SPECS:
                video_path = dataset_root / VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=episode_index // DEFAULT_CHUNK_SIZE,
                    video_key=f"observation.images.{new_key}",
                    episode_index=episode_index,
                )
                frames = np.asarray(obs[source_key][()], dtype=np.uint8)
                write_video(video_path, frames, fps=fps)

            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task_text],
                    "length": num_frames,
                    "source_demo": demo_name,
                }
            )

            state_arrays.append(state)
            action_arrays.append(action)
            global_index += num_frames

        stats = compute_stats(state_arrays, action_arrays)
        info = build_info(
            features=features,
            fps=fps,
            total_episodes=len(episode_rows),
            total_frames=global_index,
            total_tasks=len(ordered_tasks),
        )

        write_json(dataset_root / "meta" / "info.json", info)
        write_json(dataset_root / "meta" / "modality.json", modality_meta)
        write_json(dataset_root / "meta" / "stats.json", stats)
        write_tasks(dataset_root / "meta" / "tasks.jsonl", ordered_tasks)
        with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
            writer.write_all(episode_rows)

        print(
            f"[ok] {task_name}: wrote {len(episode_rows)} episodes, {global_index} frames to {dataset_root}"
        )


def main() -> None:
    args = parse_args()
    selected_tasks = set(args.tasks) if args.tasks else None
    files = find_hdf5_files(args.input_root, selected_tasks)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for hdf5_path in files:
        convert_file(
            hdf5_path=hdf5_path,
            output_root=args.output_root,
            split=args.split,
            max_episodes=args.max_episodes_per_task,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
