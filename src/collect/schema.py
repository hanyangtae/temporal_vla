"""Schema helpers for GR00T N1.6 RoboCasa SAFE collection.

수집 산출물의 **계약 단일 출처**다. 액션 벡터 스키마와 수집 클라이언트가 갖춰야 할
속성 계약(``SafeFeaturePolicy``)을 함께 둔다 — 세 수집 모듈(``collect_artifacts`` /
``collect_policy_clients`` / n15 ``http_feature_collect``)이 여기서 import 한다.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.policies.groot.robocasa.io import (
    GROOT_ACTION_KEYS,
    PRIMARY_ACTION_KEYS,
    REQUIRED_OBS_KEYS,
    VIDEO_RECORDING_KEYS,
    first_env_first_step,
    without_action_prefix,
)


# ── 수집 클라이언트 계약 ──────────────────────────────────────────────────────
# write_safe_triplet(collect_artifacts)이 policy 객체에서 꺼내 쓰는 속성. 구 배선은
# 시그니처가 policy: Any 라 **본문을 읽어야만** 무엇이 필요한지 알 수 있었고, 새 클라이언트를
# 붙일 때 속성이 빠지면 AttributeError 로 늦게 터지거나 pkl 에 조용히 null 이 들어갔다.
#
# 구조적 타이핑(Protocol)이라 기존 클라이언트는 상속 없이 그대로 만족한다 — 선언은
# "무엇이 필요한지"의 문서이자 정적 검사 대상이고, 런타임 강제는 write_safe_triplet
# 진입부의 POLICY_REQUIRED_ATTRS 검사가 담당한다(아래에서 유도되므로 둘이 어긋날 수 없다).


@runtime_checkable
class SafeFeaturePolicy(Protocol):
    """SAFE triplet 을 쓰려면 수집 클라이언트가 갖춰야 하는 최소 속성."""

    records: list[dict[str, Any]]
    feature_kind: str | None
    feature_axes: list[str] | None
    feature_slice: str | None
    exported_action_token_count: int | None
    feature_action_horizon: int | None
    valid_action_horizon: int | None
    model_action_horizon: int | None
    num_inference_timesteps: int | None


# Protocol 에서 유도 — 선언이 하나뿐이라 목록이 어긋날 수 없다.
POLICY_REQUIRED_ATTRS: tuple[str, ...] = tuple(SafeFeaturePolicy.__annotations__)

# 선택 속성: 해당 캡처 모드에서만 존재하고 getattr 폴백으로 읽는다(없으면 그 필드 생략).
POLICY_OPTIONAL_ATTRS: tuple[str, ...] = (
    "capture_layers", "layer_indices", "layer_count", "token_count", "capture_token_mode",
    "vl_feature_kind", "vl_feature_axes", "vl_feature_dim",     # --capture-vl
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
