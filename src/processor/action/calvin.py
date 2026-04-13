"""Calvin 환경 action processor.

VLA 모델이 반환하는 action sub-key dict 또는 flat ndarray를
Calvin env가 기대하는 7D [eef_pos(3), eef_euler(3), gripper(1)]로 변환한다.

지원하는 action sub-key:
  - action.eef_pos (3D) — 그대로 사용
  - action.eef_euler (3D) — 그대로 사용
  - action.eef_rot6d (6D) — euler(3D)로 변환
  - action.eef_quat (4D) — euler(3D)로 변환
  - action.gripper (1D) — threshold로 이산화 → {-1, 1}

Calvin robot.py는 gripper_action이 {-1, 1} 이산값이어야 한다.

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


class CalvinActionProcessor(ActionProcessorStep):
    """VLA action → Calvin env-compatible 7D action.

    두 가지 입력 형식을 지원:

    1. Sub-keyed dict (신규): {"action.eef_pos": (3,), "action.eef_rot6d": (6,), ...}
       → 회전 포맷 자동 변환 후 7D 조립

    2. Flat ndarray (하위호환): shape (7,) 또는 (N, 7)
       → gripper 이산화만 수행

    Args:
        threshold: gripper 이산화 임계값 (default 0.0).
                   relative 모델: 보통 0.0. absolute 모델(X-VLA 등): 0.8 권장.
        action_type: "relative" 또는 "absolute".
                     relative: flat 7D ndarray 반환 [pos(3)+euler(3)+gripper(1)]
                     absolute: 3-tuple 반환 (pos, euler, gripper)
                               Calvin env.apply_action()이 len==3이면 absolute로 처리.

    Input:  dict[str, np.ndarray] 또는 np.ndarray
    Output: relative → np.ndarray (7,)
            absolute → tuple(np.ndarray(3,), np.ndarray(3,), float)
    """

    def __init__(self, threshold: float = 0.0, action_type: str = "relative"):
        self.threshold = threshold
        self.action_type = action_type

    def process_action(self, action: Any) -> Any:
        if isinstance(action, dict):
            return self._process_subkeyed(action)
        return self._process_flat(action)

    # ------------------------------------------------------------------
    # Sub-keyed dict 처리 (신규)
    # ------------------------------------------------------------------

    def _process_subkeyed(self, action_dict: Dict[str, Any]) -> Any:
        """action sub-key dict → Calvin 호환 action.

        relative: flat 7D ndarray [pos(3), euler(3), gripper(1)]
        absolute: 3-tuple (pos(3), euler(3), gripper) — Calvin env가 직접 처리
        """
        # position (3D)
        eef_pos = np.asarray(action_dict["action.eef_pos"], dtype=np.float32).flatten()

        # rotation — 사용 가능한 포맷 우선순위: euler > rot6d > quat
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
                "CalvinActionProcessor: rotation key 없음. "
                "action.eef_euler, action.eef_rot6d, action.eef_quat 중 하나 필요. "
                "받은 키: {}".format(list(action_dict.keys()))
            )

        # gripper — 이산화
        # X-VLA sigmoid: 높은 값 = close(학습 라벨 1.0=closed), 낮은 값 = open
        # Calvin 규격: 1 = open, -1 = close
        # 따라서: sigmoid < threshold → open(1), sigmoid >= threshold → close(-1)
        gripper = np.asarray(action_dict["action.gripper"], dtype=np.float32).flatten()
        gripper_val = 1.0 if float(gripper[0]) < self.threshold else -1.0

        if self.action_type == "absolute":
            # Calvin env: len(action)==3 이면 absolute로 처리
            # action = ((x,y,z), (euler_x,euler_y,euler_z), (gripper,))
            return (tuple(eef_pos[:3]), tuple(eef_euler[:3]), (gripper_val,))

        # relative: flat 7D
        return np.concatenate([eef_pos[:3], eef_euler[:3], [gripper_val]]).astype(np.float32)

    # ------------------------------------------------------------------
    # Flat ndarray 처리 (하위호환)
    # ------------------------------------------------------------------

    def _process_flat(self, action: Any) -> np.ndarray:
        """Flat 7D ndarray → gripper 이산화."""
        action = np.array(action, dtype=np.float32).copy()
        if action.ndim == 1:
            action[-1] = 1.0 if action[-1] > self.threshold else -1.0
        else:
            action[:, -1] = np.where(action[:, -1] > self.threshold, 1.0, -1.0)
        return action

    # ------------------------------------------------------------------

    def transform_features(self, features: Features) -> Features:
        return features

    def get_config(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}


# ─── 회전 변환 유틸 (Python 3.8 호환) ────────────────────────────────────────


def _rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation → euler (roll, pitch, yaw).

    rot6d: (..., 6) — R.as_matrix()[:,:2].reshape(6) 형태.
    행 우선 flatten이므로 [R00, R01, R10, R11, R20, R21].
    열 벡터 복원은 interleaved 인덱싱: col0=[0,2,4], col1=[1,3,5].

    반환: (..., 3) — euler angles (Rz @ Ry @ Rx convention).
    """
    # interleaved extraction: rotation matrix column vectors
    a1 = rot6d[..., 0::2]  # [0, 2, 4] → R column 0
    a2 = rot6d[..., 1::2]  # [1, 3, 5] → R column 1

    # Gram-Schmidt orthogonalization → rotation matrix columns
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    # R = [b1, b2, b3] as columns: R[i,j] where column j = bj
    # Euler from Rz(yaw) @ Ry(pitch) @ Rx(roll):
    #   R[2,0] = -sin(pitch) → pitch = -arcsin(R[2,0])
    #   R[2,1] / R[2,2] = sin(roll)*cos(pitch) / cos(roll)*cos(pitch) → roll = atan2(R[2,1], R[2,2])
    #   R[1,0] / R[0,0] = sin(yaw)*cos(pitch) / cos(yaw)*cos(pitch) → yaw = atan2(R[1,0], R[0,0])
    # R[:,0] = b1, R[:,1] = b2, R[:,2] = b3
    # R[2,0] = b1[...,2], R[2,1] = b2[...,2], R[2,2] = b3[...,2]
    # R[1,0] = b1[...,1], R[0,0] = b1[...,0]

    pitch = -np.arcsin(np.clip(b1[..., 2], -1.0, 1.0))
    roll = np.arctan2(b2[..., 2], b3[..., 2])
    yaw = np.arctan2(b1[..., 1], b1[..., 0])

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → euler (roll, pitch, yaw).

    quat: (..., 4) — xyzw 순서.
    반환: (..., 3).
    """
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)
