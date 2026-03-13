"""RoboCasa 환경 action processor.

VLA 모델의 7D action을 PandaMobile의 12D action으로 매핑한다.

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


class RoboCasaActionProcessor(ActionProcessorStep):
    """VLA 7D action → PandaMobile 12D action.

    Input:  np.ndarray shape (7,) — [arm(6), gripper(1)]
    Output: np.ndarray shape (12,) — [arm(6), gripper(2), base(3), torso(1)]

    매핑:
        arm[0:6]   → 그대로
        gripper[6] → 두 번 복제 (PandaMobile은 2-finger gripper)
        base[3]    → 0.0 (모바일 베이스 비사용)
        torso[1]   → 0.0 (torso 비사용)
    """

    def __init__(self, arm_dim: int = 6):
        self.arm_dim = arm_dim

    def process_action(self, action: Any) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
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
