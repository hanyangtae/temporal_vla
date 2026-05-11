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


# ──────────────────────────────────────────────────────────────────────────
# 동적 modality 매핑 (옵션 B) — 로드된 모델의 modality_keys 에서 추론.
# ──────────────────────────────────────────────────────────────────────────


# pretrain (NVIDIA) 와 finetune (LeRobot raw) 양쪽 컨벤션을 다 흡수하는 패턴.
_VIDEO_ROLE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("left",  ("side_0", "agentview_left", "_left", "static")),
    ("right", ("side_1", "agentview_right", "_right")),
    ("wrist", ("wrist_0", "wrist_1", "eye_in_hand", "wrist")),
]

# role → 통일 API 입력 키 (alias 들; 첫 항목이 canonical).
_ROLE_TO_UNIFIED: dict[str, tuple[str, ...]] = {
    "left":  ("observation.images.left",  "observation.images.static", "observation.images.side_0"),
    "right": ("observation.images.right", "observation.images.side_1"),
    "wrist": ("observation.images.wrist", "observation.images.wrist_0"),
}


def _classify_video_modality_key(modality_key: str) -> str | None:
    """모델이 노출한 video modality_key (예: 'res256_image_side_0' 또는
    'robot0_agentview_left') 가 left/right/wrist 중 어느 role 인지 분류.

    매칭되는 role 이 없으면 None — 이 경우 정체불명 카메라이므로 호출 측에서
    identity alias 만 등록.
    """
    mk = modality_key.lower()
    for role, patterns in _VIDEO_ROLE_PATTERNS:
        for p in patterns:
            if p in mk:
                return role
    return None


def build_video_mapping(modality_keys: list[str]) -> dict[str, str]:
    """``policy.modality_configs["video"].modality_keys`` 를 받아 통일 API 입력 키
    → ``video.<modality_key>`` 매핑을 동적으로 구성한다.

    pretrain ckpt (``res256_image_side_0/...``) 든 finetune ckpt
    (``robot0_agentview_left/...``) 든 같은 통일 API 입력으로 서빙 가능.

    추가로 raw key 자체에 대해서도 ``observation.images.<modality_key>`` 형태의
    identity alias 를 등록해 호출자가 원본 키를 그대로 보내고 싶을 때도 지원.
    """
    mapping: dict[str, str] = {}
    for mk in modality_keys:
        groot_key = normalize_modality_key(mk, "video")

        role = _classify_video_modality_key(mk)
        if role is not None:
            for unified in _ROLE_TO_UNIFIED[role]:
                # 같은 unified 키에 다른 modality 가 이미 매핑돼 있으면 첫 등록 우선.
                mapping.setdefault(unified, groot_key)

        # identity alias — 호출자가 raw modality_key 를 그대로 보낸 경우.
        mapping.setdefault(f"observation.images.{mk}", groot_key)

    return mapping
