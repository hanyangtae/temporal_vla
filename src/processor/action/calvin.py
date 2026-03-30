"""Calvin 환경 action processor.

Calvin robot.py는 gripper_action이 {-1, 1} 이산값이어야 한다.
VLA 모델이 반환하는 연속 gripper 값을 이산화한다.

Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..base import ActionProcessorStep
from ..types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
)

# Calvin이 지원하는 action_dim. 확인된 구조만 허용.
_SUPPORTED_DIMS = (7,)


class CalvinActionProcessor(ActionProcessorStep):
    """VLA action → Calvin env-compatible 7D action.

    지원 action_model_dim:
        7D: [eef_pos(3), eef_euler(3), gripper(1)] — 표준 Calvin 포맷
            action[-1] → 1.0 if > threshold else -1.0

    미지원 dim(예: 6D bimanual 14D 등)은 명시적 ValueError.
    6D 지원은 smolvla/wall-x checkpoint 구조 확인 후 추가 예정.

    Args:
        action_model_dim: 모델 출력 차원. serve /health의 action_dim에서 읽어 전달.
        threshold: gripper 이산화 임계값.

    Input:  np.ndarray shape (action_model_dim,) 또는 (N, action_model_dim)
    Output: np.ndarray shape (7,) 또는 (N, 7), gripper 이산화됨
    """

    def __init__(self, action_model_dim: int = 7, threshold: float = 0.0):
        if action_model_dim not in _SUPPORTED_DIMS:
            raise ValueError(
                "CalvinActionProcessor: action_model_dim={}는 미지원. "
                "지원: {}. 6D 모델은 checkpoint action 구조 확인 후 추가 예정.".format(
                    action_model_dim, _SUPPORTED_DIMS
                )
            )
        self.action_model_dim = action_model_dim
        self.threshold = threshold

    def process_action(self, action: Any) -> np.ndarray:
        action = np.array(action, dtype=np.float32).copy()
        if action.ndim == 1:
            if action.shape[0] != self.action_model_dim:
                raise ValueError(
                    "CalvinActionProcessor: 런타임 action shape {}D ≠ "
                    "설정된 action_model_dim={}D".format(action.shape[0], self.action_model_dim)
                )
            action[-1] = 1.0 if action[-1] > self.threshold else -1.0
        else:
            if action.shape[-1] != self.action_model_dim:
                raise ValueError(
                    "CalvinActionProcessor: 런타임 action shape {}D ≠ "
                    "설정된 action_model_dim={}D".format(action.shape[-1], self.action_model_dim)
                )
            action[:, -1] = np.where(action[:, -1] > self.threshold, 1.0, -1.0)
        return action

    def transform_features(self, features: Features) -> Features:
        return features

    def get_config(self) -> Dict[str, Any]:
        return {"action_model_dim": self.action_model_dim, "threshold": self.threshold}
