"""Schema helpers for GR00T N1.6 RoboCasa SAFE collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.policies.groot.robocasa.io import (
    GROOT_ACTION_KEYS,
    PRIMARY_ACTION_KEYS,
    REQUIRED_OBS_KEYS,
    VIDEO_RECORDING_KEYS,
    first_env_first_step,
    without_action_prefix,
)


SAFE_ACTION_COLUMNS = [
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
]


def _extract_safe_action_vector(action: dict[str, Any]) -> np.ndarray:
    pieces = []
    by_normalized_key = {without_action_prefix(k): v for k, v in action.items()}
    for key in PRIMARY_ACTION_KEYS:
        value = action.get(key)
        if value is None:
            value = by_normalized_key.get(without_action_prefix(key))
        if value is not None:
            pieces.append(first_env_first_step(value))

    if not pieces:
        for key in sorted(action):
            pieces.append(first_env_first_step(action[key]))

    vector = np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[: min(7, vector.shape[0])] = vector[:7]
    return out


def _extract_groot_action_vector(action: dict[str, Any]) -> np.ndarray:
    pieces = []
    by_normalized_key = {without_action_prefix(k): v for k, v in action.items()}
    for key in GROOT_ACTION_KEYS:
        value = action.get(key)
        if value is None:
            value = by_normalized_key.get(without_action_prefix(key))
        if value is not None:
            pieces.append(first_env_first_step(value))
    if not pieces:
        return _extract_safe_action_vector(action)
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def _to_pickleable_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_pickleable_numpy(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_pickleable_numpy(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_to_pickleable_numpy(val) for val in value)
    return value
