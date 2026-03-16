"""Calvin 환경 observation processor.

Calvin env는 rgb_obs를 [-1, 1] 범위의 CHW tensor로 반환한다.
이를 VLAClient가 기대하는 uint8 HWC numpy로 변환한다.

Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..base import ObservationProcessorStep
from ..types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
)


class CalvinObsProcessor(ObservationProcessorStep):
    """Calvin raw obs → VLAClient-compatible dict.

    Input (Calvin env obs):
        obs["rgb_obs"]["rgb_static"]  — CHW tensor, [-1, 1]
        obs["rgb_obs"]["rgb_gripper"] — CHW tensor, [-1, 1] (optional)
        obs["robot_obs"]             — 15D float (optional, robot state)

    Output:
        {"observation.images.static": uint8 HWC numpy}
        {"observation.images.wrist":  uint8 HWC numpy}  (if use_wrist=True)
        {"observation.state":         float32 numpy 7D}  (if use_state=True)
    """

    def __init__(self, use_wrist: bool = False, use_state: bool = True, image_size: int = 200):
        self.use_wrist = use_wrist
        self.use_state = use_state
        self.image_size = image_size

    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        rgb_obs = observation["rgb_obs"]

        result = {
            "observation.images.static": self._convert(rgb_obs["rgb_static"]),
        }
        if self.use_wrist:
            result["observation.images.wrist"] = self._convert(rgb_obs["rgb_gripper"])

        if self.use_state and "robot_obs" in observation:
            robot_obs = np.asarray(observation["robot_obs"], dtype=np.float32)
            # 15D → 7D: arm(6) + gripper(1), DreamVLA 원본 eval과 동일
            result["observation.state"] = np.concatenate(
                [robot_obs[:6], robot_obs[-1:]]
            )

        return result

    def transform_features(self, features: Features) -> Features:
        obs_features = {
            "observation.images.static": PolicyFeature(
                FeatureType.VISUAL, (self.image_size, self.image_size, 3)
            ),
        }
        if self.use_wrist:
            obs_features["observation.images.wrist"] = PolicyFeature(
                FeatureType.VISUAL, (self.image_size, self.image_size, 3)
            )
        if self.use_state:
            obs_features["observation.state"] = PolicyFeature(
                FeatureType.STATE, (7,)
            )

        features = dict(features)
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features

    def get_config(self) -> Dict[str, Any]:
        return {"use_wrist": self.use_wrist, "use_state": self.use_state, "image_size": self.image_size}

    # ------------------------------------------------------------------

    @staticmethod
    def _convert(img: Any) -> np.ndarray:
        """CHW tensor in [-1, 1] → HWC uint8 numpy in [0, 255]."""
        if hasattr(img, "numpy"):  # torch tensor
            img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        elif isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] in (1, 3):
            # numpy CHW → HWC
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = ((np.clip(img, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
        return img
