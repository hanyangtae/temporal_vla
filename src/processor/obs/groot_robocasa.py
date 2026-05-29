"""GR00T RoboCasa native observation processor.

This processor gives ``GrootRoboCasaEnv`` the same pipeline-shaped interface
as the generic benchmark processors while keeping the key mapping in
``src.policies.groot.robocasa_io`` as the single implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from src.policies.groot.robocasa_io import prepare_groot_robocasa_http_request

from ..base import ObservationProcessorStep
from ..types import FeatureType, Features, PipelineFeatureType, PolicyFeature


class GrootRoboCasaObsProcessor(ObservationProcessorStep):
    """GrootRoboCasaEnv obs dict -> project HTTP observation payload."""

    def __init__(self, strict: bool = True, image_size: int = 256):
        self.strict = strict
        self.image_size = image_size

    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        request = prepare_groot_robocasa_http_request(
            observation,
            strict=self.strict,
        )
        result = {
            "observation.images.{}".format(name): image
            for name, image in request.images.items()
        }
        result.update(request.states)
        result["task"] = request.instruction
        return result

    def transform_features(self, features: Features) -> Features:
        sz = self.image_size
        features = dict(features)
        features[PipelineFeatureType.OBSERVATION] = {
            "observation.images.left": PolicyFeature(FeatureType.VISUAL, (sz, sz, 3)),
            "observation.images.right": PolicyFeature(FeatureType.VISUAL, (sz, sz, 3)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (sz, sz, 3)),
            "observation.state.base_position": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.state.base_rotation": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.state.eef_pos_rel": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.state.eef_quat_rel": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.state.gripper_qpos": PolicyFeature(FeatureType.STATE, (2,)),
            "task": PolicyFeature(FeatureType.LANGUAGE, (1,)),
        }
        return features

    def get_config(self) -> Dict[str, Any]:
        return {
            "strict": self.strict,
            "image_size": self.image_size,
        }
