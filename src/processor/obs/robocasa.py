"""Generic RoboCasa observation processor for Project FastAPI evaluation.

robosuite obs dict에서 개별 observable 키를 사용하여 state를
``observation.state.*`` sub-key로 분리 출력한다.
모델 서버가 필요한 성분만 골라 쓰고, 변환(quat→euler 등)도 모델 쪽에서 수행.

GR00T `GrootRoboCasaEnv` native observation conversion is intentionally kept
out of this generic processor; use `obs.groot_robocasa` for that pipeline
adapter.

Python 3.8 compatible.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

from ..base import ObservationProcessorStep
from ..types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
)

# 68D proprio-state 내부 인덱스 (PandaMobile 기준, fallback용)
_PROPRIO_SLICES = {
    "eef_pos":       (35, 38),   # (3,)
    "eef_quat":      (38, 42),   # (4,) xyzw
    "gripper_qpos":  (46, 48),   # (2,)
    "joint_pos":     (0,  7),    # (7,)
    "joint_vel":     (21, 28),   # (7,)
    "base_pos":      (50, 53),   # (3,)
    "base_quat":     (53, 57),   # (4,)
}
_PROPRIO_MIN_DIM = 57


class RoboCasaObsProcessor(ObservationProcessorStep):
    """RoboCasa raw obs → observation.state.* sub-keys.

    robosuite obs dict의 개별 observable 키에서 state 성분을 추출한다.
    개별 키가 없으면 flat ``robot0_proprio-state`` 에서 인덱스로 파싱 (fallback).

    Output keys:
        observation.images.static        — uint8 HWC numpy
        observation.images.wrist         — uint8 HWC numpy
        observation.state.eef_pos        — float32 (3,)
        observation.state.eef_quat       — float32 (4,)  xyzw
        observation.state.gripper_qpos   — float32 (2,)
        observation.state.joint_pos      — float32 (7,)
        observation.state.joint_vel      — float32 (7,)
        observation.state.base_pos       — float32 (3,)  mobile only
        observation.state.base_quat      — float32 (4,)  mobile only
        observation.state.base_to_eef_pos  — float32 (3,)  base-frame EEF pos
        observation.state.base_to_eef_quat — float32 (4,)  base-frame EEF quat xyzw
    """

    _STATE_COMPONENTS = (
        "eef_pos", "eef_quat", "gripper_qpos",
        "joint_pos", "joint_vel",
        "base_pos", "base_quat",
        # base-frame relative EEF (robocasa pi05 학습 포맷). robosuite mobile_robot
        # observable robot0_base_to_eef_pos/_quat 를 그대로 emit (계산 불필요).
        "base_to_eef_pos", "base_to_eef_quat",
    )

    def __init__(
        self,
        static_cam: str = "robot0_agentview_left",
        static_cam2: str = None,
        wrist_cam: str = "robot0_eye_in_hand",
        robot_prefix: str = "robot0_",
        image_size: int = 224,
        state_key: str = "robot0_proprio-state",
    ):
        self.static_cam = static_cam
        self.static_cam2 = static_cam2
        self.wrist_cam = wrist_cam
        self.robot_prefix = robot_prefix
        self.image_size = image_size
        self.state_key = state_key
        self._warned_fallback = False

    def _extract_named_states(self, observation):
        # type: (Dict[str, Any]) -> Dict[str, np.ndarray]
        pf = self.robot_prefix
        states = {}  # type: Dict[str, np.ndarray]
        for comp in self._STATE_COMPONENTS:
            val = observation.get("{}{}".format(pf, comp))
            if val is not None:
                states["observation.state.{}".format(comp)] = np.asarray(val, dtype=np.float32)
        return states

    def _fallback_from_proprio(self, observation):
        # type: (Dict[str, Any]) -> Dict[str, np.ndarray]
        raw = observation.get(self.state_key)
        if raw is None:
            return {}
        raw = np.asarray(raw, dtype=np.float32)

        if not self._warned_fallback:
            logger.warning(
                "개별 obs 키(eef_pos 등) 없음 — proprio-state %dD에서 인덱스 파싱 (fallback)",
                raw.shape[0],
            )
            self._warned_fallback = True

        states = {}  # type: Dict[str, np.ndarray]
        for comp, (start, end) in _PROPRIO_SLICES.items():
            if raw.shape[0] >= end:
                states["observation.state.{}".format(comp)] = raw[start:end].copy()
        return states

    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        sz = self.image_size
        fallback = np.zeros((sz, sz, 3), dtype=np.uint8)

        static_key = "{}_image".format(self.static_cam)
        wrist_key = "{}_image".format(self.wrist_cam)

        if static_key not in observation:
            logger.warning("카메라 '%s' 누락, zero 이미지로 대체", static_key)
        if wrist_key not in observation:
            logger.warning("카메라 '%s' 누락, zero 이미지로 대체", wrist_key)

        if self.static_cam2 is not None:
            # 3-camera 모드: left(=static alias), right, wrist 의 의미 명확한 키로 emit.
            # robocasa native 카메라 = robot0_agentview_left/_right/_eye_in_hand 셋.
            # GR00T ROBOCASA_PANDA_OMRON schema 와 정합되도록 side_0/side_1/wrist_0 alias 도 emit.
            # `static` 은 `left` alias 로 유지하여 기존 2-camera 모델(xvla/dreamvla)의 통일 API 호환성 유지.
            static2_key = "{}_image".format(self.static_cam2)
            if static2_key not in observation:
                logger.warning("카메라 '%s' 누락, zero 이미지로 대체", static2_key)
            left_img = observation.get(static_key, fallback.copy())
            right_img = observation.get(static2_key, fallback.copy())
            wrist_img = observation.get(wrist_key, fallback.copy())
            result = {
                "observation.images.static":  left_img,   # left alias (2-camera 모델 호환)
                "observation.images.left":    left_img,
                "observation.images.side_0":  left_img,   # GR00T schema alias
                "observation.images.right":   right_img,
                "observation.images.side_1":  right_img,  # GR00T schema alias
                "observation.images.wrist":   wrist_img,
                "observation.images.wrist_0": wrist_img,  # GR00T schema alias
            }  # type: Dict[str, Any]
        else:
            result = {
                "observation.images.static": observation.get(static_key, fallback.copy()),
                "observation.images.wrist": observation.get(wrist_key, fallback.copy()),
            }  # type: Dict[str, Any]

        states = self._extract_named_states(observation)
        if not states:
            states = self._fallback_from_proprio(observation)
        result.update(states)

        return result

    def transform_features(self, features: Features) -> Features:
        sz = self.image_size
        obs_features = {
            "observation.images.static": PolicyFeature(FeatureType.VISUAL, (sz, sz, 3)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (sz, sz, 3)),
            "observation.state.eef_pos": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.state.eef_quat": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.state.gripper_qpos": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.state.joint_pos": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.state.joint_vel": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.state.base_pos": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.state.base_quat": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.state.base_to_eef_pos": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.state.base_to_eef_quat": PolicyFeature(FeatureType.STATE, (4,)),
        }
        features = dict(features)
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features

    def get_config(self) -> Dict[str, Any]:
        return {
            "static_cam": self.static_cam,
            "wrist_cam": self.wrist_cam,
            "robot_prefix": self.robot_prefix,
            "image_size": self.image_size,
            "state_key": self.state_key,
        }
