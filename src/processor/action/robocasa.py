"""RoboCasa 환경 action processor.

VLA 모델의 action을 PandaMobile의 12D action으로 매핑한다.

지원하는 입력 형식:
  1. Sub-keyed dict: {"action.eef_pos": (3,), "action.eef_euler": (3,), "action.gripper": (1,)}
     → 회전 포맷 자동 변환 (rot6d/quat → euler)
  2. Flat ndarray: shape (7,) — [arm(6), gripper(1)]

Python 3.8 compatible.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from ..base import ActionProcessorStep
from ..types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
)


class RoboCasaActionProcessor(ActionProcessorStep):
    """VLA action → PandaMobile 12D action.

    Input:  dict[str, np.ndarray] 또는 np.ndarray shape (7,)
    Output: np.ndarray shape (12,) — [arm(6), gripper(2), base(3), torso(1)]

    매핑:
        arm[0:6]   → eef_pos(3) + eef_euler(3) 또는 flat[:6]
        gripper    → 두 번 복제 (PandaMobile은 2-finger gripper)
        base[3]    → 0.0 (모바일 베이스 비사용)
        torso[1]   → 0.0 (torso 비사용)
    """

    def __init__(self, arm_dim: int = 6):
        self.arm_dim = arm_dim

    def process_action(self, action: Any) -> np.ndarray:
        if isinstance(action, dict):
            return self._process_subkeyed(action)
        return self._process_flat(action)

    def _process_subkeyed(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """action sub-key dict → PandaMobile 12D."""
        eef_pos = np.asarray(action_dict["action.eef_pos"], dtype=np.float32).flatten()

        if "action.eef_euler" in action_dict:
            eef_euler = np.asarray(action_dict["action.eef_euler"], dtype=np.float32).flatten()
        elif "action.eef_rot6d" in action_dict:
            rot6d = np.asarray(action_dict["action.eef_rot6d"], dtype=np.float32)
            eef_euler = _rot6d_to_euler(rot6d).flatten()
        elif "action.eef_quat" in action_dict:
            quat = np.asarray(action_dict["action.eef_quat"], dtype=np.float32)
            eef_euler = _quat_to_euler(quat).flatten()
        else:
            raise ValueError(
                "RoboCasaActionProcessor: rotation key 없음. "
                "action.eef_euler, action.eef_rot6d, action.eef_quat 중 하나 필요. "
                "받은 키: {}".format(list(action_dict.keys()))
            )

        gripper = np.asarray(action_dict["action.gripper"], dtype=np.float32).flatten()
        grip = float(gripper[0])

        arm = np.concatenate([eef_pos[:3], eef_euler[:3]])
        return np.concatenate([
            arm,
            [grip, grip],
            [0.0, 0.0, 0.0],
            [0.0],
        ]).astype(np.float32)

    def _process_flat(self, action: Any) -> np.ndarray:
        """Flat 7D ndarray → PandaMobile 12D."""
        action = np.asarray(action, dtype=np.float32)
        expected_dim = self.arm_dim + 1  # arm + gripper
        if action.shape[-1] != expected_dim:
            raise ValueError(
                "RoboCasaActionProcessor: expected {}D action (arm={} + gripper=1), "
                "got {}D".format(expected_dim, self.arm_dim, action.shape[-1])
            )
        arm = action[:self.arm_dim]
        grip = action[self.arm_dim]
        return np.concatenate([
            arm,
            [grip, grip],       # gripper × 2
            [0.0, 0.0, 0.0],   # base
            [0.0],              # torso
        ]).astype(np.float32)

    def transform_features(self, features: Features) -> Features:
        features = dict(features)
        features[PipelineFeatureType.ACTION] = {
            "action": PolicyFeature(FeatureType.ACTION, (12,)),
        }
        return features

    def get_config(self) -> Dict[str, Any]:
        return {"arm_dim": self.arm_dim}


# ─── 회전 변환 유틸 (Python 3.8 호환) ────────────────────────────────────────


def _rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation → euler (roll, pitch, yaw)."""
    a1 = rot6d[..., 0::2]
    a2 = rot6d[..., 1::2]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    pitch = -np.arcsin(np.clip(b1[..., 2], -1.0, 1.0))
    roll = np.arctan2(b2[..., 2], b3[..., 2])
    yaw = np.arctan2(b1[..., 1], b1[..., 0])

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → euler (roll, pitch, yaw)."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)
