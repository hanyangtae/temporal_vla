"""GR00T 입출력 schema 매핑 테이블 (PandaOmron embodiment 기준).

서빙 layer 가 통일 API ↔ GR00T native key 변환에 사용.
학습 adapter 도 robosuite raw key → GR00T key 매핑이 필요할 때 참고
가능 (단, adapter 는 robocasa submodule 의 `PandaOmronKeyConverter` 를
직접 호출하는 게 더 안전 — 이번 매핑 테이블은 통일 HTTP API 한정).

robocasa native 카메라 ↔ GR00T modality key 대응:
    robot0_agentview_left  ↔ video.res256_image_side_0
    robot0_agentview_right ↔ video.res256_image_side_1
    robot0_eye_in_hand     ↔ video.res256_image_wrist_0
"""

from __future__ import annotations

# 통일 API 이미지 키 → GR00T video 키 (PandaOmron 기준).
# `static`/`left` 는 2-camera 모델 호환 alias. side_0/side_1/wrist_0 는
# GR00T schema 정합 alias.
UNIFIED_TO_VIDEO_KEY: dict[str, str] = {
    "observation.images.left":    "video.res256_image_side_0",
    "observation.images.static":  "video.res256_image_side_0",  # 2-camera 모델 호환
    "observation.images.side_0":  "video.res256_image_side_0",
    "observation.images.right":   "video.res256_image_side_1",
    "observation.images.side_1":  "video.res256_image_side_1",
    "observation.images.wrist":   "video.res256_image_wrist_0",
    "observation.images.wrist_0": "video.res256_image_wrist_0",
}

# 통일 API state 키 → GR00T state 키 (PandaOmronKeyConverter 매핑 기준).
UNIFIED_TO_STATE_KEY: dict[str, str] = {
    "observation.state.gripper_qpos":   "state.gripper_qpos",
    "observation.state.base_position":  "state.base_position",
    "observation.state.base_rotation":  "state.base_rotation",
    "observation.state.eef_pos_rel":    "state.end_effector_position_relative",
    "observation.state.eef_quat_rel":   "state.end_effector_rotation_relative",
    "observation.state.gripper_qvel":   "state.gripper_qvel",
    "observation.state.eef_pos":        "state.end_effector_position_absolute",
    "observation.state.eef_quat":       "state.end_effector_rotation_absolute",
    "observation.state.joint_pos":      "state.joint_position",
    "observation.state.joint_pos_cos":  "state.joint_position_cos",
    "observation.state.joint_pos_sin":  "state.joint_position_sin",
    "observation.state.joint_vel":      "state.joint_velocity",
}

# GR00T native action key → 통일 sub-key (RoboCasaActionProcessor 가 인식).
GROOT_TO_UNIFIED_ACTION: dict[str, str] = {
    "end_effector_position": "action.eef_pos",
    "end_effector_rotation": "action.eef_axisangle",
    "gripper_close":         "action.gripper",
    "base_motion":           "action.base_motion",
    "control_mode":          "action.control_mode",
}


def normalize_modality_key(key: str, prefix: str) -> str:
    """modality_keys 가 'res256_image_side_0' 형태로 prefix 없이 올 수 있으므로 보장.

    이미 'video.' / 'state.' 접두사가 있으면 그대로, 없으면 추가.
    """
    return key if key.startswith(prefix + ".") else f"{prefix}.{key}"
