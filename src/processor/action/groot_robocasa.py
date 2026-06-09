"""GR00T RoboCasa native action processor."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.policies.groot.robocasa.io import (
    convert_http_actions_to_groot_chunk,
    convert_http_actions_to_groot_step,
)

from ..base import ActionProcessorStep
from ..types import FeatureType, Features, PipelineFeatureType, PolicyFeature


_GROOT_ACTION_FEATURE_SHAPES = {
    "action.end_effector_position": (3,),
    "action.end_effector_rotation": (3,),
    "action.gripper_close": (1,),
    "action.base_motion": (4,),
    "action.control_mode": (1,),
}


class GrootRoboCasaActionProcessor(ActionProcessorStep):
    """Project HTTP action sub-keys -> GrootRoboCasaEnv native action dict.

    ``mode="step"`` consumes the first decoded action step for closed-loop eval.
    ``mode="chunk"`` keeps a batched chunk for GR00T SAFE rollout wrappers.
    """

    def __init__(self, mode: str = "step", action_horizon: int = 16):
        if mode not in {"step", "chunk"}:
            raise ValueError("mode must be 'step' or 'chunk', got {!r}".format(mode))
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive, got {}".format(action_horizon))
        self.mode = mode
        self.action_horizon = action_horizon

    def process_action(self, action: Any) -> Dict[str, np.ndarray]:
        if not isinstance(action, dict):
            raise RuntimeError(
                "GrootRoboCasaActionProcessor requires sub-keyed HTTP actions, got {}".format(
                    type(action).__name__
                )
            )
        if self.mode == "chunk":
            return convert_http_actions_to_groot_chunk(action)
        return convert_http_actions_to_groot_step(action)

    def transform_features(self, features: Features) -> Features:
        features = dict(features)
        if self.mode == "chunk":
            action_features = {
                key: PolicyFeature(FeatureType.ACTION, (1, self.action_horizon, shape[0]))
                for key, shape in _GROOT_ACTION_FEATURE_SHAPES.items()
            }
        else:
            action_features = {
                key: PolicyFeature(FeatureType.ACTION, shape)
                for key, shape in _GROOT_ACTION_FEATURE_SHAPES.items()
            }
        features[PipelineFeatureType.ACTION] = action_features
        return features

    def get_config(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "action_horizon": self.action_horizon,
        }
