"""SAFE rollout 수집 공통 유틸 (lerobot HTTP 경로).

groot_n16(``scripts/safe/groot_n16/robocasa/collect/collect_rollout.py``)의
per-episode pkl 스키마를 재사용하여, 기존 split/aggregation 스크립트
(``scripts/safe/groot_n16/.../safe_feature_vectors.py``, split helper)가
LIBERO/VLABench/RoboCasa 수집물도 동일하게 읽도록 한다.

데이터 한 점 = policy 추론(action chunk) 1회의 SAFE latent
  (= ``hidden_states`` 리스트의 한 원소; flow-matching ``[K,H,D]`` / pi0_fast ``[1,n_tokens,D]``).
라벨 = episode-level success (파일명 ``succ0/1`` + pkl ``episode_success``).

파일 규약 (groot_n16 와 동일):
  ``{output_dir}/rollouts_{run_id}/{task_name}/task{task_id}--ep{episode_idx}--succ{0|1}.pkl``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))[:80]


class SafeEpisodeCollector:
    """한 episode 동안 추론이 발화한 step 의 SAFE latent + action 을 누적.

    ``record`` 는 매 env step 마다 호출하되, 정책이 새 추론을 돌리지 않은 step
    (features is None) 은 무시한다. 따라서 ``hidden_states`` 는 추론 1회당 1개.
    """

    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        task_description: str,
        episode_idx: int,
        env_name: str = "",
    ):
        self.task_suite_name = task_suite_name
        self.task_id = int(task_id)
        self.task_description = task_description
        self.episode_idx = int(episode_idx)
        self.env_name = env_name
        self.hidden_states: list[np.ndarray] = []
        self.action_vectors: list[np.ndarray] = []
        self.actions: list[Any] = []
        self.feature_kind: str | None = None
        self.feature_axes: list[str] | None = None
        self.num_inference_timesteps: int | None = None
        self.model_action_horizon: int | None = None
        self.feature_dim: int | None = None

    def record(
        self,
        features: dict[str, Any] | None,
        action_vector: np.ndarray | None = None,
        action: Any | None = None,
    ) -> None:
        if features is None:
            return
        self.hidden_states.append(np.asarray(features["hidden_states"]))
        self.feature_kind = features.get("feature_kind", self.feature_kind)
        self.feature_axes = features.get("feature_axes", self.feature_axes)
        self.num_inference_timesteps = features.get(
            "num_inference_timesteps", self.num_inference_timesteps
        )
        self.model_action_horizon = features.get("action_horizon", self.model_action_horizon)
        self.feature_dim = features.get("feature_dim", self.feature_dim)
        if action_vector is not None:
            self.action_vectors.append(np.asarray(action_vector, dtype=np.float32))
        if action is not None:
            self.actions.append(action)

    def __len__(self) -> int:
        return len(self.hidden_states)

    def payload(self, success: bool, **extra: Any) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_suite_name": self.task_suite_name,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "episode_idx": self.episode_idx,
            "episode_success": int(bool(success)),
            "hidden_states": self.hidden_states,
            "actions": self.actions,
            "action_vectors": (
                np.stack(self.action_vectors)
                if self.action_vectors
                else np.empty((0,), dtype=np.float32)
            ),
            "feature_kind": self.feature_kind,
            "feature_axes": self.feature_axes,
            "num_inference_timesteps": self.num_inference_timesteps,
            "model_action_horizon": self.model_action_horizon,
            "feature_dim": self.feature_dim,
            "env_name": self.env_name,
        }
        d.update(extra)
        return d


def episode_path(
    output_dir: str | Path,
    run_id: str,
    task_name: str,
    task_id: int,
    episode_idx: int,
    success: bool,
) -> Path:
    succ = 1 if success else 0
    d = Path(output_dir) / f"rollouts_{run_id}" / _safe_name(task_name)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"task{int(task_id)}--ep{int(episode_idx)}--succ{succ}.pkl"


def write_episode(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def flatten_subkeys(action_subkeys: dict[str, Any], order: list[str]) -> np.ndarray:
    """action.* sub-key dict → emits 순서대로 concat 한 1D 벡터 (csv/vis 보조용)."""
    parts: list[np.ndarray] = []
    for k in order:
        if k in action_subkeys:
            parts.append(np.asarray(action_subkeys[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
