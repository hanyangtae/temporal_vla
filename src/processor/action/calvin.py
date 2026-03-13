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


class CalvinActionProcessor(ActionProcessorStep):
    """VLA 7D action → Calvin env-compatible 7D action.

    변환:
        action[-1] (gripper) → 1.0 if > threshold else -1.0

    Input:  np.ndarray shape (7,) 또는 (N, 7)
    Output: np.ndarray shape 동일, gripper 이산화됨
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def process_action(self, action: Any) -> np.ndarray:
        action = np.array(action, dtype=np.float32).copy()
        if action.ndim == 1:
            action[-1] = 1.0 if action[-1] > self.threshold else -1.0
        else:
            action[:, -1] = np.where(
                action[:, -1] > self.threshold, 1.0, -1.0
            )
        return action

    def transform_features(self, features: Features) -> Features:
        # shape 불변: 7D → 7D (gripper 값만 이산화)
        return features

    def get_config(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}
