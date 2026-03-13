"""RoboCasa 환경 observation processor.

RoboCasa(robosuite) env는 이미지를 HWC uint8로 반환하므로 변환 없이 리매핑만 수행.

Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..base import ObservationProcessorStep
from ..types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
)


class RoboCasaObsProcessor(ObservationProcessorStep):
    """RoboCasa raw obs → VLAClient-compatible dict.

    Input (robosuite env obs):
        obs["robot0_agentview_left_image"]  — HWC uint8 (224×224×3)
        obs["robot0_eye_in_hand_image"]     — HWC uint8 (224×224×3)
        obs["robot0_proprio-state"]         — float32 (32,)

    Output:
        {"observation.images.static": uint8 HWC numpy,
         "observation.images.wrist":  uint8 HWC numpy,
         "observation.state":         float32 numpy}
    """

    def __init__(
        self,
        static_cam: str = "robot0_agentview_left",
        wrist_cam: str = "robot0_eye_in_hand",
        state_key: str = "robot0_proprio-state",
        image_size: int = 224,
    ):
        self.static_cam = static_cam
        self.wrist_cam = wrist_cam
        self.state_key = state_key
        self.image_size = image_size

    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        sz = self.image_size
        fallback = np.zeros((sz, sz, 3), dtype=np.uint8)

        result = {
            "observation.images.static": observation.get(
                "{}_image".format(self.static_cam), fallback.copy()
            ),
            "observation.images.wrist": observation.get(
                "{}_image".format(self.wrist_cam), fallback.copy()
            ),
        }

        state = observation.get(self.state_key)
        if state is not None:
            result["observation.state"] = np.asarray(state, dtype=np.float32)

        return result

    def transform_features(self, features: Features) -> Features:
        sz = self.image_size
        obs_features = {
            "observation.images.static": PolicyFeature(
                FeatureType.VISUAL, (sz, sz, 3)
            ),
            "observation.images.wrist": PolicyFeature(
                FeatureType.VISUAL, (sz, sz, 3)
            ),
            "observation.state": PolicyFeature(
                FeatureType.STATE, (32,)
            ),
        }

        features = dict(features)
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features

    def get_config(self) -> Dict[str, Any]:
        return {
            "static_cam": self.static_cam,
            "wrist_cam": self.wrist_cam,
            "state_key": self.state_key,
            "image_size": self.image_size,
        }
