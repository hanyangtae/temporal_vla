#!/usr/bin/env python3
"""Merge RoboCasa LeRobot v2.1 task datasets into one GR00T dataset root."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.path_setup import DATA_ROOT

DEFAULT_TASKS = [
    "OpenDrawer",
    "CloseDrawer",
    "OpenCabinet",
    "CloseCabinet",
    "OpenFridge",
    "CloseFridge",
    "OpenMicrowave",
    "CloseMicrowave",
    "PickPlaceCounterToStove",
    "PickPlaceCounterToSink",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_or_copy(src: Path, dst: Path, use_hardlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_hardlinks:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def load_sources(source_root: Path, tasks: list[str], date: str) -> list[Path]:
    sources = []
    for task in tasks:
        dataset_root = source_root / task / date / "lerobot"
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
        sources.append(dataset_root)
    return sources


def validate_sources(sources: list[Path]) -> tuple[dict[str, Any], list[str]]:
    ref_info = read_json(sources[0] / "meta/info.json")
    ref_modality = read_json(sources[0] / "meta/modality.json")
    ref_embodiment = read_json(sources[0] / "meta/embodiment.json")

    if ref_info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected v2.1 dataset, got {ref_info.get('codebase_version')}")

    video_keys = [
        key for key, feat in ref_info["features"].items() if feat.get("dtype") == "video"
    ]
    required_meta = [
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "episodes_stats.jsonl",
        "modality.json",
        "embodiment.json",
        "stats.json",
        "relative_stats.json",
    ]
    for src in sources:
        for name in required_meta:
            if not (src / "meta" / name).exists():
                raise FileNotFoundError(f"Missing {name}: {src / 'meta' / name}")

        info = read_json(src / "meta/info.json")
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{src} is not v2.1: {info.get('codebase_version')}")
        if info.get("features") != ref_info.get("features"):
            raise ValueError(f"Feature schema mismatch: {src}")
        if info.get("chunks_size") != ref_info.get("chunks_size"):
            raise ValueError(f"chunks_size mismatch: {src}")
        if info.get("data_path") != ref_info.get("data_path"):
            raise ValueError(f"data_path pattern mismatch: {src}")
        if info.get("video_path") != ref_info.get("video_path"):
            raise ValueError(f"video_path pattern mismatch: {src}")
        if read_json(src / "meta/modality.json") != ref_modality:
            raise ValueError(f"modality.json mismatch: {src}")
        if read_json(src / "meta/embodiment.json") != ref_embodiment:
            raise ValueError(f"embodiment.json mismatch: {src}")

    return ref_info, video_keys


def map_tasks(sources: list[Path]) -> tuple[list[dict[str, Any]], dict[Path, dict[int, int]]]:
    merged_tasks: list[dict[str, Any]] = []
    task_to_new_index: dict[str, int] = {}
    per_source_task_maps: dict[Path, dict[int, int]] = {}

    for src in sources:
        src_map = {}
        for row in read_jsonl(src / "meta/tasks.jsonl"):
            task = row["task"]
            if task not in task_to_new_index:
                task_to_new_index[task] = len(merged_tasks)
                merged_tasks.append({"task_index": task_to_new_index[task], "task": task})
            src_map[int(row["task_index"])] = task_to_new_index[task]
        per_source_task_maps[src] = src_map

    return merged_tasks, per_source_task_maps


def rewrite_task_index_columns(df: pd.DataFrame, task_map: dict[int, int]) -> pd.DataFrame:
    rewritten = df.copy()
    index_columns = [
        "task_index",
        "annotation.human.task_description",
        "annotation.human.task_name",
    ]
    for column in index_columns:
        if column in rewritten.columns:
            rewritten[column] = rewritten[column].map(lambda x: task_map[int(x)]).astype("int64")
    return rewritten


def update_episode_stats_task_indices(
    stats_row: dict[str, Any],
    old_to_new_values: dict[str, int],
) -> dict[str, Any]:
    row = dict(stats_row)
    stats = dict(row.get("stats", {}))
    for key, new_value in old_to_new_values.items():
        if key not in stats:
            continue
        count = stats[key].get("count", [0])
        stats[key] = {
            "min": [new_value],
            "max": [new_value],
            "mean": [float(new_value)],
            "std": [0.0],
            "count": count,
        }
    row["stats"] = stats
    return row


def compute_stats(dataset_root: Path, info: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    float_features = [
        key for key, feat in info["features"].items() if "float" in feat.get("dtype", "")
    ]
    frames = []
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        frames.append(pd.read_parquet(parquet_path, columns=float_features))
    all_data = pd.concat(frames, axis=0)

    stats = {}
    for feature in float_features:
        values = np.vstack(
            [np.asarray(value, dtype=np.float32).reshape(-1) for value in all_data[feature]]
        )
        stats[feature] = {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return stats


def merge_datasets(
    sources: list[Path],
    output: Path,
    use_hardlinks: bool,
) -> None:
    ref_info, video_keys = validate_sources(sources)
    merged_tasks, task_maps = map_tasks(sources)

    chunks_size = int(ref_info["chunks_size"])
    data_path_pattern = ref_info["data_path"]
    video_path_pattern = ref_info["video_path"]

    output.mkdir(parents=True)
    (output / "meta").mkdir()

    shutil.copy2(sources[0] / "meta/modality.json", output / "meta/modality.json")
    shutil.copy2(sources[0] / "meta/embodiment.json", output / "meta/embodiment.json")
    write_jsonl(output / "meta/tasks.jsonl", merged_tasks)

    merged_episodes: list[dict[str, Any]] = []
    merged_episode_stats: list[dict[str, Any]] = []
    global_episode_index = 0
    global_frame_index = 0

    for src in sources:
        task_map = task_maps[src]
        info = read_json(src / "meta/info.json")
        episodes = read_jsonl(src / "meta/episodes.jsonl")
        episodes_stats = {
            int(row["episode_index"]): row
            for row in read_jsonl(src / "meta/episodes_stats.jsonl")
        }

        for src_episode_index in range(int(info["total_episodes"])):
            src_chunk = src_episode_index // chunks_size
            src_parquet_rel = data_path_pattern.format(
                episode_chunk=src_chunk,
                episode_index=src_episode_index,
            )
            src_parquet = src / src_parquet_rel
            if not src_parquet.exists():
                raise FileNotFoundError(f"Missing parquet: {src_parquet}")

            df = pd.read_parquet(src_parquet)
            df = rewrite_task_index_columns(df, task_map)
            episode_length = len(df)
            df["episode_index"] = global_episode_index
            df["index"] = np.arange(
                global_frame_index,
                global_frame_index + episode_length,
                dtype=np.int64,
            )

            dst_chunk = global_episode_index // chunks_size
            dst_parquet_rel = data_path_pattern.format(
                episode_chunk=dst_chunk,
                episode_index=global_episode_index,
            )
            dst_parquet = output / dst_parquet_rel
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_parquet, index=False)

            for video_key in video_keys:
                src_video_rel = video_path_pattern.format(
                    episode_chunk=src_chunk,
                    video_key=video_key,
                    episode_index=src_episode_index,
                )
                dst_video_rel = video_path_pattern.format(
                    episode_chunk=dst_chunk,
                    video_key=video_key,
                    episode_index=global_episode_index,
                )
                src_video = src / src_video_rel
                if not src_video.exists():
                    raise FileNotFoundError(f"Missing video: {src_video}")
                link_or_copy(src_video, output / dst_video_rel, use_hardlinks)

            episode_row = dict(episodes[src_episode_index])
            episode_row["episode_index"] = global_episode_index
            episode_row["length"] = episode_length
            merged_episodes.append(episode_row)

            stats_row = dict(episodes_stats[src_episode_index])
            stats_row["episode_index"] = global_episode_index
            old_to_new_values = {}
            for column in [
                "task_index",
                "annotation.human.task_description",
                "annotation.human.task_name",
            ]:
                if column in df.columns:
                    old_to_new_values[column] = int(df[column].iloc[0])
            merged_episode_stats.append(
                update_episode_stats_task_indices(stats_row, old_to_new_values)
            )

            global_episode_index += 1
            global_frame_index += episode_length

    merged_info = dict(ref_info)
    merged_info["total_episodes"] = global_episode_index
    merged_info["total_frames"] = global_frame_index
    merged_info["total_tasks"] = len(merged_tasks)
    merged_info["splits"] = {"train": f"0:{global_episode_index}"}
    merged_info["total_chunks"] = math.ceil(global_episode_index / chunks_size)
    merged_info["total_videos"] = global_episode_index * len(video_keys)

    write_json(output / "meta/info.json", merged_info)
    write_jsonl(output / "meta/episodes.jsonl", merged_episodes)
    write_jsonl(output / "meta/episodes_stats.jsonl", merged_episode_stats)
    write_json(output / "meta/stats.json", compute_stats(output, merged_info))
    write_json(output / "meta/relative_stats.json", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("src/benchmarks/robocasa/datasets/v1.0/pretrain/atomic"),
    )
    parser.add_argument("--date", default="20250819")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "datasets/robocasa_10tasks_lerobot_v21",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of hardlinking when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    tmp_output = output.with_name(output.name + ".tmp")

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
    if tmp_output.exists():
        raise FileExistsError(f"Temporary output already exists: {tmp_output}")
    if output.exists():
        shutil.rmtree(output)

    sources = load_sources(args.source_root, args.tasks, args.date)
    try:
        merge_datasets(sources, tmp_output, use_hardlinks=not args.copy_videos)
        tmp_output.rename(output)
    except Exception:
        if tmp_output.exists():
            shutil.rmtree(tmp_output)
        raise

    info = read_json(output / "meta/info.json")
    print(f"Merged {len(sources)} datasets into {output}")
    print(f"episodes={info['total_episodes']} frames={info['total_frames']}")
    print(f"tasks={info['total_tasks']} chunks={info['total_chunks']}")


if __name__ == "__main__":
    main()
