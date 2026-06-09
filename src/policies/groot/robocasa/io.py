"""GR00T RoboCasa IO adapter for native GrootRoboCasaEnv keys.

This module is the shared adapter between GrootRoboCasaEnv native keys and the
project FastAPI action interface. SAFE collection keeps its pkl-specific schema
helpers in scripts, but the observation/action IO mapping lives here.

The generic benchmark processor pipeline in ``src/processor`` serves arbitrary
VLA policies behind the project HTTP schema. GR00T `GrootRoboCasaEnv` eval and
SAFE wiring bypass that pipeline and use this adapter so native GR00T
observation/action keys stay aligned with the upstream evaluation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.schema import (
    GROOT_ENV_LANGUAGE_KEYS,
    GROOT_ENV_VIDEO_TO_UNIFIED_CAM,
    GROOT_TO_UNIFIED_ACTION,
    GROOT_TO_UNIFIED_STATE,
    UNIFIED_TO_GROOT_ACTION,
    extract_groot_instruction,
)


REQUIRED_UNIFIED_STATE_KEYS = (
    "observation.state.base_position",
    "observation.state.base_rotation",
    "observation.state.eef_pos_rel",
    "observation.state.eef_quat_rel",
    "observation.state.gripper_qpos",
)

PRIMARY_GROOT_ACTION_NAMES = (
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
)
PRIMARY_ACTION_KEYS = [
    UNIFIED_TO_GROOT_ACTION[GROOT_TO_UNIFIED_ACTION[groot_key]]
    for groot_key in PRIMARY_GROOT_ACTION_NAMES
]
GROOT_ACTION_KEYS = [
    f"action.{groot_key}" for groot_key in GROOT_TO_UNIFIED_ACTION
]
VIDEO_RECORDING_KEYS = {
    key for key in GROOT_ENV_VIDEO_TO_UNIFIED_CAM if key.startswith("video.res256_image")
}


def _invert_groups(items: dict[str, str]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for key, group in items.items():
        groups.setdefault(group, []).append(key)
    return {group: tuple(keys) for group, keys in groups.items()}


def _bidirectional_aliases(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for keys in groups.values():
        for src in keys:
            for dst in keys:
                if src != dst:
                    aliases[src] = dst
    return aliases


VIDEO_KEY_GROUPS = _invert_groups(GROOT_ENV_VIDEO_TO_UNIFIED_CAM)
OBS_ALIASES = _bidirectional_aliases(VIDEO_KEY_GROUPS)
OBS_ALIASES.update(
    {
        GROOT_ENV_LANGUAGE_KEYS[0]: GROOT_ENV_LANGUAGE_KEYS[1],
        GROOT_ENV_LANGUAGE_KEYS[1]: GROOT_ENV_LANGUAGE_KEYS[0],
    }
)
REQUIRED_OBS_KEYS = {
    *GROOT_ENV_VIDEO_TO_UNIFIED_CAM.keys(),
    *(GROOT_ENV_LANGUAGE_KEYS),
    *(
        groot_key
        for groot_key, unified_key in GROOT_TO_UNIFIED_STATE.items()
        if unified_key in REQUIRED_UNIFIED_STATE_KEYS
    ),
}


@dataclass(frozen=True)
class GrootRoboCasaHttpRequest:
    images: dict[str, np.ndarray]
    states: dict[str, np.ndarray]
    instruction: str


def without_action_prefix(key: str) -> str:
    return key[len("action.") :] if key.startswith("action.") else key


def first_env_first_step(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0].reshape(-1)
    return arr[0, 0].reshape(-1)


def latest_image_frame(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 5:
        return np.asarray(arr[0, -1], dtype=np.uint8)
    if arr.ndim == 4:
        return np.asarray(arr[-1], dtype=np.uint8)
    if arr.ndim == 3:
        return np.asarray(arr, dtype=np.uint8)
    raise ValueError(f"Expected image observation [H,W,C] or time/batch wrapped, got {arr.shape}")


def latest_state_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2:
        return arr[-1].reshape(-1)
    return arr[0, -1].reshape(-1)


def batched_action_chunk(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr[np.newaxis, np.newaxis, :]
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected action chunk rank 1-3, got {arr.shape}")


def prepare_groot_robocasa_observation(
    observation: dict[str, Any], *, strict: bool = True
) -> dict[str, Any]:
    aliased = dict(observation)
    for src, dst in OBS_ALIASES.items():
        if src in aliased and dst not in aliased:
            aliased[dst] = aliased[src]

    if strict:
        filtered = {key: aliased[key] for key in REQUIRED_OBS_KEYS if key in aliased}
        missing = sorted(REQUIRED_OBS_KEYS - filtered.keys())
        if missing:
            raise KeyError(f"Missing N1.6 RoboCasa observation keys: {missing}")
        return filtered

    return {key: value for key, value in aliased.items() if key in REQUIRED_OBS_KEYS}


def prepare_groot_robocasa_http_request(
    observation: dict[str, Any], *, strict: bool = True
) -> GrootRoboCasaHttpRequest:
    filtered = prepare_groot_robocasa_observation(observation, strict=strict)
    images: dict[str, np.ndarray] = {}
    for groot_key, unified_cam in GROOT_ENV_VIDEO_TO_UNIFIED_CAM.items():
        if groot_key in filtered:
            images.setdefault(unified_cam, latest_image_frame(filtered[groot_key]))

    states: dict[str, np.ndarray] = {}
    for groot_key, unified_key in GROOT_TO_UNIFIED_STATE.items():
        if groot_key in filtered:
            states[unified_key] = latest_state_vector(filtered[groot_key])

    return GrootRoboCasaHttpRequest(
        images=images,
        states=states,
        instruction=extract_groot_instruction(filtered),
    )


def http_server_url(host: str, port: int) -> str:
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:{port}"


def convert_http_actions_to_groot_chunk(actions: dict[str, Any]) -> dict[str, np.ndarray]:
    groot_action = {}
    for unified_key, groot_key in UNIFIED_TO_GROOT_ACTION.items():
        if unified_key in actions:
            groot_action[groot_key] = batched_action_chunk(actions[unified_key])
    if not groot_action:
        raise RuntimeError("HTTP response did not contain GR00T action sub-keys")
    return groot_action


def convert_http_actions_to_groot_step(actions: dict[str, Any]) -> dict[str, np.ndarray]:
    groot_action = {}
    for unified_key, groot_key in UNIFIED_TO_GROOT_ACTION.items():
        if unified_key in actions:
            groot_action[groot_key] = first_env_first_step(actions[unified_key])
    if not groot_action:
        raise RuntimeError("HTTP response did not contain GR00T action sub-keys")
    return groot_action
