"""Generic RoboCasa action processor for Project FastAPI evaluation.

VLA 모델의 action을 PandaMobile의 12D action으로 매핑한다.
GR00T `GrootRoboCasaEnv` native action conversion is intentionally handled by
`action.groot_robocasa`, not this generic benchmark processor.

지원하는 입력 형식:
  1. Sub-keyed dict:
       필수: action.eef_pos(3), action.gripper(1),
              그리고 rotation 중 하나 — action.eef_euler / .eef_axisangle / .eef_rot6d / .eef_quat
       옵션: action.base_motion(3 or 4)  — 없으면 base/torso = 0 + warning
              action.torso(1)            — 따로 보낼 경우 base_motion 보다 우선
  2. Flat ndarray: shape (7,) — [arm(6), gripper(1)]

회전 포맷 처리:
  - eef_euler / eef_axisangle: 둘 다 3D, robosuite OSC_POSE 가 받는 raw 형식 그대로 통과
    (의미는 다르지만 OSC_POSE 입력에서는 동일 슬롯에 들어가므로 변환 없이 사용)
  - eef_rot6d → euler 로 변환
  - eef_quat (x,y,z,w) → euler 로 변환

Python 3.8 compatible.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from ._rotation import quat_xyzw_to_euler, rot6d_to_euler
from ..base import ActionProcessorStep
from ..types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
)

logger = logging.getLogger(__name__)


class RoboCasaActionProcessor(ActionProcessorStep):
    """VLA action → PandaMobile 12D action.

    Input:  dict[str, np.ndarray] 또는 np.ndarray shape (7,)
    Output: np.ndarray shape (12,) — [arm(6), gripper(2), base(3), torso(1)]

    매핑:
        arm[0:6]   → eef_pos(3) + eef_rot(3) 또는 flat[:6]
        gripper    → 두 번 복제 (PandaMobile은 2-finger gripper)
        base[3]    → action.base_motion[:3] 있으면 사용, 없으면 0 (warning 1회)
        torso[1]   → action.torso 또는 action.base_motion[3:4] 있으면 사용, 없으면 0
    """

    def __init__(self, arm_dim: int = 6):
        self.arm_dim = arm_dim
        # 옵션 키 누락 warning 은 호출당 1번만 (반복 로그 폭주 방지)
        self._warned_missing_base = False
        self._warned_missing_torso = False

    def process_action(self, action: Any) -> np.ndarray:
        if isinstance(action, dict):
            return self._process_subkeyed(action)
        return self._process_flat(action)

    def _resolve_rotation(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """action_dict 에서 사용 가능한 회전 키를 찾아 3D float32 ndarray 로 반환.

        eef_euler 와 eef_axisangle 은 의미가 다르지만 둘 다 robosuite OSC_POSE
        (axis-angle) 슬롯에 같은 형식으로 들어가므로 변환 없이 통과.
        """
        if "action.eef_axisangle" in action_dict:
            return np.asarray(
                action_dict["action.eef_axisangle"], dtype=np.float32
            ).flatten()
        if "action.eef_euler" in action_dict:
            return np.asarray(
                action_dict["action.eef_euler"], dtype=np.float32
            ).flatten()
        if "action.eef_rot6d" in action_dict:
            rot6d = np.asarray(action_dict["action.eef_rot6d"], dtype=np.float32)
            return rot6d_to_euler(rot6d).flatten()
        if "action.eef_quat" in action_dict:
            quat = np.asarray(action_dict["action.eef_quat"], dtype=np.float32)
            return quat_xyzw_to_euler(quat).flatten()
        raise ValueError(
            "RoboCasaActionProcessor: rotation key 없음. "
            "action.eef_axisangle, action.eef_euler, action.eef_rot6d, action.eef_quat "
            "중 하나 필요. 받은 키: {}".format(list(action_dict.keys()))
        )

    def _resolve_base_torso(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """base(3) + torso(1) = 4D 를 action_dict 에서 추출.

        우선순위:
          1. action.base_motion (3D 또는 4D) + action.torso (있으면 torso override)
          2. action.base (3D) + action.torso (1D)
          3. 누락 시 0 + warning (호출 1회만)
        """
        base = np.zeros(3, dtype=np.float32)
        torso = np.zeros(1, dtype=np.float32)
        has_base = False
        has_torso = False

        if "action.base_motion" in action_dict:
            bm = np.asarray(action_dict["action.base_motion"], dtype=np.float32).flatten()
            if bm.shape[0] >= 3:
                base = bm[:3]
                has_base = True
            if bm.shape[0] >= 4:
                torso = bm[3:4]
                has_torso = True
        elif "action.base" in action_dict:
            base = np.asarray(action_dict["action.base"], dtype=np.float32).flatten()[:3]
            has_base = True

        if "action.torso" in action_dict:
            torso = np.asarray(action_dict["action.torso"], dtype=np.float32).flatten()[:1]
            has_torso = True

        if not has_base and not self._warned_missing_base:
            logger.warning(
                "RoboCasaActionProcessor: action.base_motion/.base 누락 → base = 0 처리. "
                "받은 action 키: %s", sorted(action_dict.keys()),
            )
            self._warned_missing_base = True
        if not has_torso and not self._warned_missing_torso:
            logger.warning(
                "RoboCasaActionProcessor: action.torso (또는 base_motion[3]) 누락 → torso = 0 처리.",
            )
            self._warned_missing_torso = True

        return np.concatenate([base, torso]).astype(np.float32)

    def _process_subkeyed(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """action sub-key dict → PandaMobile 12D."""
        eef_pos = np.asarray(action_dict["action.eef_pos"], dtype=np.float32).flatten()
        eef_rot = self._resolve_rotation(action_dict)

        gripper = np.asarray(action_dict["action.gripper"], dtype=np.float32).flatten()
        grip = float(gripper[0])

        base_torso = self._resolve_base_torso(action_dict)  # [base(3), torso(1)] = 4D

        arm = np.concatenate([eef_pos[:3], eef_rot[:3]])
        # PandaMobile 12D = [right(6), gripper(1), base(3), torso(1), base_mode(1)].
        # pi0.5 는 1D gripper + control_mode 를 내보냄 → control_mode 가 있으면 1D gripper +
        # base_mode(=control_mode) 로 조립. (GR00T 는 2D gripper 라 [grip, grip] 복제가 맞음.)
        if "action.control_mode" in action_dict:
            base_mode = float(
                np.asarray(action_dict["action.control_mode"], dtype=np.float32).flatten()[0]
            )
            return np.concatenate([
                arm,
                [grip],
                base_torso,
                [base_mode],
            ]).astype(np.float32)
        return np.concatenate([
            arm,
            [grip, grip],
            base_torso,
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
