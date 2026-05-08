"""LIBERO 환경 action processor.

VLA 서버가 반환하는 action sub-key dict 를 LIBERO env 가 기대하는
7D action `[eef_pos(3), eef_axisangle(3), gripper(1)]` 으로 변환한다.

학습 시 normalize 된 action 의 디노말은 모두 **serve 측** (`predict_action(unnorm_key=...)`)
에서 끝난 상태로 sub-key 가 emit 된다. gripper 도 `gripper_encoding.binarize +
sign_flip` 까지 serve 에서 적용되어 LIBERO env 기대값 (close=+1, open=-1) 그대로 도착.
따라서 여기서는 **차원 합성 + euler→axisangle 회전 변환** 만 수행한다.

지원하는 rotation sub-key:
  - action.eef_axisangle (3D) — 그대로 사용 (정확)
  - action.eef_euler (3D)     — Rotation.from_euler("xyz") → as_rotvec() (round-trip OK)
  - action.eef_quat (4D xyzw) — Rotation.from_quat → as_rotvec()
  - action.eef_rot6d (6D)     — Gram-Schmidt → as_rotvec()

openvla_oft 프로파일은 기본적으로 axisangle → euler 로 변환해 emit 하므로
실 운용에서는 `action.eef_euler` 경로가 사용된다.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from ..base import ActionProcessorStep
from ..types import Features


class LIBEROActionProcessor(ActionProcessorStep):
    """unified action sub-keys → LIBERO 7D action ndarray.

    Input  (dict):
        {"action.eef_pos":         (3,) | [[3]],
         "action.eef_euler"|...:   회전 sub-key 중 하나,
         "action.gripper":         (1,) | [[1]] in {-1, +1}}

    Output (np.ndarray, shape (7,)):
        [pos_x, pos_y, pos_z, axisangle_x, axisangle_y, axisangle_z, gripper]
    """

    def process_action(self, action: Any) -> Any:
        if not isinstance(action, dict):
            return self._process_flat(action)
        return self._process_subkeyed(action)

    def _process_subkeyed(self, action_dict: Dict[str, Any]) -> np.ndarray:
        pos = np.asarray(
            action_dict["action.eef_pos"], dtype=np.float32
        ).flatten()[:3]
        axisangle = self._to_axisangle(action_dict)
        grip = float(
            np.asarray(action_dict["action.gripper"], dtype=np.float32).flatten()[0]
        )
        return np.concatenate([pos, axisangle, [grip]]).astype(np.float32)

    @staticmethod
    def _to_axisangle(action_dict: Dict[str, Any]) -> np.ndarray:
        if "action.eef_axisangle" in action_dict:
            aa = np.asarray(
                action_dict["action.eef_axisangle"], dtype=np.float32
            ).flatten()[:3]
            return aa
        if "action.eef_euler" in action_dict:
            euler = np.asarray(
                action_dict["action.eef_euler"], dtype=np.float32
            ).flatten()[:3]
            return Rotation.from_euler("xyz", euler).as_rotvec().astype(np.float32)
        if "action.eef_quat" in action_dict:
            quat = np.asarray(
                action_dict["action.eef_quat"], dtype=np.float32
            ).flatten()[:4]
            return Rotation.from_quat(quat).as_rotvec().astype(np.float32)
        if "action.eef_rot6d" in action_dict:
            rot6d = np.asarray(
                action_dict["action.eef_rot6d"], dtype=np.float32
            ).flatten()[:6]
            a1 = rot6d[0::2]
            a2 = rot6d[1::2]
            b1 = a1 / (np.linalg.norm(a1) + 1e-8)
            b2 = a2 - np.dot(b1, a2) * b1
            b2 = b2 / (np.linalg.norm(b2) + 1e-8)
            b3 = np.cross(b1, b2)
            mat = np.stack([b1, b2, b3], axis=1)
            return Rotation.from_matrix(mat).as_rotvec().astype(np.float32)
        raise ValueError(
            "LIBEROActionProcessor: rotation sub-key 없음. action.eef_axisangle / "
            "action.eef_euler / action.eef_quat / action.eef_rot6d 중 하나 필요. "
            f"받은 키: {list(action_dict.keys())}"
        )

    @staticmethod
    def _process_flat(action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).flatten()
        if arr.shape[0] != 7:
            raise ValueError(
                f"LIBEROActionProcessor flat input must be 7-D, got {arr.shape}"
            )
        return arr

    def transform_features(self, features: Features) -> Features:
        return features

    def get_config(self) -> Dict[str, Any]:
        return {}
