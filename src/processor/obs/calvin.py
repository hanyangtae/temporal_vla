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
    """Calvin raw obs → observation.state.* sub-keys.

    Input (Calvin env obs):
        obs["rgb_obs"]["rgb_static"]  — CHW tensor, [-1, 1]
        obs["rgb_obs"]["rgb_gripper"] — CHW tensor, [-1, 1] (optional)
        obs["robot_obs"]             — 15D float (optional, robot state)

    robot_obs 15D layout:
        [ 0: 3] tcp_pos          — EEF 위치 (xyz)
        [ 3: 6] tcp_orn          — EEF 자세 (euler rpy)
        [ 6]    gripper_opening  — 그리퍼 열림 폭 (연속값)
        [ 7:14] arm_joint_pos    — 7 관절 위치
        [14]    gripper_action   — 그리퍼 액션 (-1 close / 1 open)

    Output keys:
        observation.images.static              — uint8 HWC numpy
        observation.images.wrist               — uint8 HWC numpy  (if use_wrist)
        observation.state.eef_pos              — float32 (3,)
        observation.state.eef_euler            — float32 (3,)  rpy
        observation.state.gripper_opening      — float32 (1,)
        observation.state.joint_pos            — float32 (7,)
        observation.state.gripper_action       — float32 (1,)  {-1, 1}
    """

    def __init__(self, use_wrist: bool = False, use_state: bool = True, image_size: int = 200):
        self.use_wrist = use_wrist
        self.use_state = use_state
        self.image_size = image_size

    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        rgb_obs = observation["rgb_obs"]

        result = {
            "observation.images.static": self._convert(rgb_obs["rgb_static"]),
        }  # type: Dict[str, Any]
        if self.use_wrist:
            result["observation.images.wrist"] = self._convert(rgb_obs["rgb_gripper"])

        if self.use_state and "robot_obs" in observation:
            robot_obs = np.asarray(observation["robot_obs"], dtype=np.float32)
            result["observation.state.eef_pos"] = robot_obs[0:3].copy()
            result["observation.state.eef_euler"] = robot_obs[3:6].copy()
            result["observation.state.gripper_opening"] = robot_obs[6:7].copy()
            result["observation.state.joint_pos"] = robot_obs[7:14].copy()
            result["observation.state.gripper_action"] = robot_obs[14:15].copy()

        return result

    def transform_features(self, features: Features) -> Features:
        sz = self.image_size
        obs_features = {
            "observation.images.static": PolicyFeature(
                FeatureType.VISUAL, (sz, sz, 3)
            ),
        }  # type: Dict[str, PolicyFeature]
        if self.use_wrist:
            obs_features["observation.images.wrist"] = PolicyFeature(
                FeatureType.VISUAL, (sz, sz, 3)
            )
        if self.use_state:
            obs_features["observation.state.eef_pos"] = PolicyFeature(FeatureType.STATE, (3,))
            obs_features["observation.state.eef_euler"] = PolicyFeature(FeatureType.STATE, (3,))
            obs_features["observation.state.gripper_opening"] = PolicyFeature(FeatureType.STATE, (1,))
            obs_features["observation.state.joint_pos"] = PolicyFeature(FeatureType.STATE, (7,))
            obs_features["observation.state.gripper_action"] = PolicyFeature(FeatureType.STATE, (1,))

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
