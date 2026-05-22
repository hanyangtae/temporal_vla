"""SAFE feature vector loading and aggregation for GR00T N1.6 RoboCasa rollouts."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_aggregation_command(value: str) -> float | str:
    if value == "mean" or value.startswith("concat"):
        return value
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Unknown aggregation command: {value}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"Relative aggregation index must be in [0, 1], got: {value}")
    return parsed


def parse_and_index_tensor_last(features: np.ndarray, command: str) -> np.ndarray:
    if command == "concat":
        return features.reshape(*features.shape[:-2], features.shape[-2] * features.shape[-1])

    prefix = "concat-"
    if not command.startswith(prefix):
        raise ValueError(f"Unknown concat aggregation command: {command}")
    sub_command = command[len(prefix) :]

    if ":" in sub_command:
        parts = sub_command.split(":")
        if len(parts) == 2:
            start_str, stop_str = parts
            step = None
        elif len(parts) == 3:
            start_str, stop_str, step_str = parts
            step = int(step_str) if step_str else None
        else:
            raise ValueError(f"Invalid concat slice command: {command}")
        start = int(start_str) if start_str else None
        stop = int(stop_str) if stop_str else None
        indexed = features[..., slice(start, stop, step), :]
    else:
        count = int(sub_command)
        if count < 2:
            raise ValueError(f"Uniform concat aggregation needs at least 2 positions: {command}")
        axis_size = features.shape[-2]
        indices = np.round(np.linspace(0, axis_size - 1, num=count)).astype(int)
        indexed = features[..., indices, :]

    return indexed.reshape(*indexed.shape[:-2], indexed.shape[-2] * indexed.shape[-1])


def process_tensor_idx_rel(features: np.ndarray, command: float | str) -> np.ndarray:
    if features.ndim < 2:
        raise ValueError(f"Expected rank >= 2 features, got {features.shape}")
    if isinstance(command, float):
        token_idx = round((features.shape[-2] - 1) * command)
        return features[..., token_idx, :]
    if command == "mean":
        return features.mean(axis=-2)
    if isinstance(command, str) and command.startswith("concat"):
        return parse_and_index_tensor_last(features, command)
    raise ValueError(f"Unknown aggregation command: {command}")


def aggregation_slug(horizon_idx_rel: str, diff_idx_rel: str) -> str:
    return f"{horizon_idx_rel}_{diff_idx_rel}"


def aggregation_space(horizon_idx_rel: str, diff_idx_rel: str, feature_dim: int) -> str:
    if horizon_idx_rel == "mean" and diff_idx_rel == "mean" and feature_dim == 1024:
        return "original_1024d_mean_over_diff_and_horizon"
    return f"original_{feature_dim}d_horizon_{horizon_idx_rel}_diff_{diff_idx_rel}"


def load_manifest(split_root: Path, scope: str) -> list[dict[str, str]]:
    manifest_path = split_root / "manifest.tsv"
    with manifest_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if scope == "all":
        selected = rows
    else:
        selected = [row for row in rows if row["split"] == scope]
    selected.sort(key=lambda row: (int(row["task_id"]), int(row["episode_idx"])))
    return selected


def pooled_hidden_states(
    record: dict[str, Any],
    *,
    horizon_idx_rel: float | str,
    diff_idx_rel: float | str,
) -> np.ndarray:
    features = []
    for hidden in record["hidden_states"]:
        hidden_np = tensor_to_numpy(hidden).astype(np.float32, copy=False)
        if hidden_np.ndim != 3:
            raise ValueError(f"Expected hidden state [K, H, D], got {hidden_np.shape}")
        hidden_np = process_tensor_idx_rel(hidden_np, horizon_idx_rel)
        hidden_np = process_tensor_idx_rel(hidden_np, diff_idx_rel)
        if hidden_np.ndim != 1:
            raise ValueError(f"Expected pooled hidden state [D], got {hidden_np.shape}")
        features.append(hidden_np)
    if not features:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(features, axis=0)


def load_scope_features(
    split_root: Path,
    scope: str,
    *,
    horizon_idx_rel: float | str,
    diff_idx_rel: float | str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = load_manifest(split_root, scope)
    feature_chunks: list[np.ndarray] = []
    success_labels: list[np.ndarray] = []
    failure_labels: list[np.ndarray] = []
    task_labels: list[np.ndarray] = []
    task_failure_labels: list[np.ndarray] = []
    task_names: list[str] = []
    episode_indices: list[np.ndarray] = []

    for row in rows:
        path = Path(row["source_path"])
        with path.open("rb") as f:
            record = pickle.load(f)
        feats = pooled_hidden_states(
            record,
            horizon_idx_rel=horizon_idx_rel,
            diff_idx_rel=diff_idx_rel,
        )
        n = feats.shape[0]
        if n == 0:
            continue

        task_id = int(row["task_id"])
        success = int(row["success"])
        failure = 1 - success
        feature_chunks.append(feats)
        success_labels.append(np.full(n, success, dtype=np.int64))
        failure_labels.append(np.full(n, failure, dtype=np.int64))
        task_labels.append(np.full(n, task_id, dtype=np.int64))
        task_failure_labels.append(np.full(n, task_id * 2 + failure, dtype=np.int64))
        task_names.extend([row["task"]] * n)
        episode_indices.append(np.full(n, int(row["episode_idx"]), dtype=np.int64))

    if not feature_chunks:
        raise ValueError(f"No features found for scope: {scope}")

    labels = {
        "success": np.concatenate(success_labels),
        "failure": np.concatenate(failure_labels),
        "task": np.concatenate(task_labels),
        "task_failure": np.concatenate(task_failure_labels),
        "episode_idx": np.concatenate(episode_indices),
        "task_name": np.asarray(task_names),
    }
    return np.concatenate(feature_chunks, axis=0), labels

