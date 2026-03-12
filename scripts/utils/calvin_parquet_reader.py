"""Calvin LeRobot parquet 읽기 유틸리티.

lerobot 없이 pandas/pyarrow만 사용. Python 3.8 호환.
데이터는 data/calvin_lerobot/fywang_ABCD_D/ 에 이미 존재한다고 가정.

사전 설치:
    pip install "pyarrow<15"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

EPISODES_PER_CHUNK = 1000  # chunk-N: 에피소드 N*1000 ~ (N+1)*1000-1


def _chunk_index(episode_index: int) -> int:
    return episode_index // EPISODES_PER_CHUNK


def get_parquet_path(data_dir: Path, episode_index: int) -> Path:
    chunk = _chunk_index(episode_index)
    return (
        data_dir
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def load_task_mapping(data_dir: Path) -> Dict[int, str]:
    """meta/tasks.jsonl → {task_index: task_name}.

    LeRobot v2.1 형식: {"task_index": 0, "task": "move_slider_right"}
    다운로드 후 실제 task name 형식 확인 필수 (CALVIN oracle 키와 일치 여부).
    """
    tasks_path = data_dir / "meta" / "tasks.jsonl"
    mapping: Dict[int, str] = {}
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_name = entry.get("task") or entry.get("task_name", "")
            mapping[entry["task_index"]] = task_name
    return mapping


def read_episode(data_dir: Path, episode_index: int) -> Dict[str, object]:
    """단일 에피소드 parquet 읽기.

    Returns:
        actions:    (T, 7)  float64 — relative action [-1,1], 마지막 dim gripper
        robot_obs:  (T, 15) float64 — observation.state
        scene_obs:  (T, 24) float64 — observation.environment_state
        task_index: int
        length:     int
    """
    df = pd.read_parquet(get_parquet_path(data_dir, episode_index))
    return {
        "actions": np.stack(df["action"].tolist()).astype(np.float64),
        "robot_obs": np.stack(df["observation.state"].tolist()).astype(np.float64),
        "scene_obs": np.stack(df["observation.environment_state"].tolist()).astype(
            np.float64
        ),
        "task_index": int(df["task_index"].iloc[0]),
        "length": len(df),
    }


def list_available_episodes(data_dir: Path) -> List[int]:
    """로컬에 존재하는 에피소드 index 목록."""
    episodes = []
    for parquet_file in sorted((data_dir / "data").glob("chunk-*/episode_*.parquet")):
        episodes.append(int(parquet_file.stem.split("_")[1]))
    return episodes
