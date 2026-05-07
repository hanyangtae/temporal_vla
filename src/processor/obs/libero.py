"""LIBERO 환경 observation processor.

LIBERO env (robosuite OffScreenRenderEnv) 가 반환하는 dict 의 raw key 들을
통일 API observation 키로 변환한다.

이미지 회전 처리:
  LIBERO bddl 학습 데이터는 180도 회전된 이미지로 학습됨. 업스트림 `get_libero_image`
  도 `[::-1, ::-1]` 로 회전을 적용. 우리는 raw 이미지를 그대로 보내고
  serve 측 `image_preprocess.rotate_180=true` 가 회전을 수행 → 학습 시점과 동일.

State 처리:
  openvla-oft LIBERO 학습 proprio = pos(3) + axisangle(3) + gripper_qpos(2) = 8D.
  여기서는 raw 형태(`eef_quat` 4D xyzw + `gripper_qpos` 2D) 그대로 보내고
  serve 가 `quat_to_axisangle` 변환 후 8D 조립. → normalize_proprio 가
  학습 시 사용된 dataset_statistics 의 평균/스케일을 그대로 적용.

Python 3.10+ assumed (LIBERO 컨테이너 = openvla_oft, py3.10).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..base import ObservationProcessorStep
from ..types import Features, FeatureType, PipelineFeatureType, PolicyFeature


class LIBEROObsProcessor(ObservationProcessorStep):
    """LIBERO raw obs → unified observation sub-keys.

    Input (LIBERO `OffScreenRenderEnv.step`/`set_init_state` 반환 dict):
        obs["agentview_image"]          — (H, W, 3) uint8, robosuite raw orientation
        obs["robot0_eye_in_hand_image"] — (H, W, 3) uint8
        obs["robot0_eef_pos"]           — (3,) float64, EEF position
        obs["robot0_eef_quat"]          — (4,) float64, xyzw quaternion
        obs["robot0_gripper_qpos"]      — (2,) float64, two-finger joint pos

    Output keys (통일 API):
        observation.images.static       — (H, W, 3) uint8, **raw orientation** (serve 가 rotate)
        observation.images.wrist        — (H, W, 3) uint8, **raw orientation**
        observation.state.eef_pos       — (3,)  float32
        observation.state.eef_quat      — (4,)  float32 xyzw
        observation.state.gripper_qpos  — (2,)  float32

    image_size: 명시 안 하면 입력 그대로 통과.
    """

    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "observation.images.static": self._to_uint8(observation["agentview_image"]),
            "observation.images.wrist": self._to_uint8(
                observation["robot0_eye_in_hand_image"]
            ),
            "observation.state.eef_pos": np.asarray(
                observation["robot0_eef_pos"], dtype=np.float32
            ).copy(),
            "observation.state.eef_quat": np.asarray(
                observation["robot0_eef_quat"], dtype=np.float32
            ).copy(),
            "observation.state.gripper_qpos": np.asarray(
                observation["robot0_gripper_qpos"], dtype=np.float32
            ).copy(),
        }
        return result

    def transform_features(self, features: Features) -> Features:
        sz = self.image_size
        obs_features: Dict[str, PolicyFeature] = {
            "observation.images.static": PolicyFeature(
                FeatureType.VISUAL, (sz, sz, 3)
            ),
            "observation.images.wrist": PolicyFeature(
                FeatureType.VISUAL, (sz, sz, 3)
            ),
            "observation.state.eef_pos": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.state.eef_quat": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.state.gripper_qpos": PolicyFeature(FeatureType.STATE, (2,)),
        }
        features = dict(features)
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features

    def get_config(self) -> Dict[str, Any]:
        return {"image_size": self.image_size}

    @staticmethod
    def _to_uint8(img: Any) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= -1.0:
                if arr.min() < 0.0:
                    arr = (arr + 1.0) / 2.0
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return arr
