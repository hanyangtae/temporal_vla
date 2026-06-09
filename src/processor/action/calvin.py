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

    def __init__(
        self,
        threshold: float = 0.0,
        action_type: str = "relative",
        gripper_invert: bool = False,
    ):
        """
        gripper_invert:
          False (default, **표준**): emit 값이 "클수록 open" 컨벤션.
            DreamVLA 처럼 ±1 binarized 후 +1=open, -1=close 로 emit 하는 모델.
            Calvin env 가 동일 부호 사용. → `g > threshold → open(+1)`.
          True: emit 값이 "클수록 close" 컨벤션 (X-VLA 같이 학습 라벨 1=closed).
            sigmoid 연속값을 그대로 emit 하는 경우 사용 (threshold ≈ 0.8 권장).
            → `g < threshold → open(+1)`.
        """
        self.threshold = threshold
        self.action_type = action_type
        self.gripper_invert = gripper_invert

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
            eef_euler = rot6d_to_euler(rot6d).flatten()
        elif "action.eef_quat" in action_dict:
            quat = np.asarray(action_dict["action.eef_quat"], dtype=np.float32)
            eef_euler = quat_xyzw_to_euler(quat).flatten()
        else:
            raise ValueError(
                "CalvinActionProcessor: rotation key 없음. "
                "action.eef_euler, action.eef_rot6d, action.eef_quat 중 하나 필요. "
                "받은 키: {}".format(list(action_dict.keys()))
            )

        # gripper — 컨벤션 분기
        # 표준 (gripper_invert=False): emit 클수록 open. DreamVLA 같은 ±1 binarized 출력.
        #   → g > threshold → open(+1), 아니면 close(-1).
        # invert (gripper_invert=True): emit 클수록 close. X-VLA 같은 sigmoid (1=closed).
        #   → g < threshold → open(+1), 아니면 close(-1).
        # Calvin env: +1 = open, -1 = close.
        gripper = np.asarray(action_dict["action.gripper"], dtype=np.float32).flatten()
        g = float(gripper[0])
        if self.gripper_invert:
            gripper_val = 1.0 if g < self.threshold else -1.0
        else:
            gripper_val = 1.0 if g > self.threshold else -1.0

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
        """Flat 7D ndarray → gripper 이산화. 컨벤션은 _process_subkeyed 와 동일."""
        action = np.array(action, dtype=np.float32).copy()
        if action.ndim == 1:
            if action.shape[-1] != 7:
                raise ValueError(
                    "CalvinActionProcessor: expected 7D action, got {}D".format(
                        action.shape[-1]
                    )
                )
        elif action.ndim == 2:
            if action.shape[-1] != 7:
                raise ValueError(
                    "CalvinActionProcessor: expected [N, 7] actions, got shape {}".format(
                        action.shape
                    )
                )
        else:
            raise ValueError(
                "CalvinActionProcessor: expected 1D or 2D action, got shape {}".format(
                    action.shape
                )
            )
        if self.gripper_invert:
            # high = close
            if action.ndim == 1:
                action[-1] = 1.0 if action[-1] < self.threshold else -1.0
            else:
                action[:, -1] = np.where(action[:, -1] < self.threshold, 1.0, -1.0)
        else:
            # 표준: high = open
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
