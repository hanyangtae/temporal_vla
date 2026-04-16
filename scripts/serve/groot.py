"""
GR00T N1.6 추론 서버 (통일 API).

groot 컨테이너 내에서 실행:
  python /temporal_vla/scripts/serve/groot.py \
    --model-path /temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B \
    --port 8500

통일 API:
  POST /act     <- obs dict (이미지: base64 PNG, state: 배열) -> {"action": [[...], ...]}
  POST /reset   <- 에피소드 시작 시 policy state 초기화
  GET  /health  <- 모델/embodiment 정보
"""

import argparse
import base64
import io
import sys
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, "/temporal_vla/src/policies/Isaac-GR00T")

app = FastAPI(title="GR00T N1.6 Inference Server")

_policy = None
_embodiment_tag = None
_modality_configs = None

FINAL_IMAGE_RESOLUTION = (256, 256)

# 통일 API 이미지 키 -> GR00T video 키 (PandaOmron 기준)
UNIFIED_TO_VIDEO_KEY = {
    "observation.images.side_0": "video.res256_image_side_0",
    "observation.images.static": "video.res256_image_side_0",
    "observation.images.side_1": "video.res256_image_side_1",
    "observation.images.wrist": "video.res256_image_wrist_0",
}

# 통일 API state 키 -> GR00T state 키 (PandaOmronKeyConverter 기준)
UNIFIED_TO_STATE_KEY = {
    "observation.state.gripper_qpos": "state.gripper_qpos",
    "observation.state.base_position": "state.base_position",
    "observation.state.base_rotation": "state.base_rotation",
    "observation.state.eef_pos_rel": "state.end_effector_position_relative",
    "observation.state.eef_quat_rel": "state.end_effector_rotation_relative",
    "observation.state.gripper_qvel": "state.gripper_qvel",
    "observation.state.eef_pos": "state.end_effector_position_absolute",
    "observation.state.eef_quat": "state.end_effector_rotation_absolute",
    "observation.state.joint_pos": "state.joint_position",
    "observation.state.joint_pos_cos": "state.joint_position_cos",
    "observation.state.joint_pos_sin": "state.joint_position_sin",
    "observation.state.joint_vel": "state.joint_velocity",
}


def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """base64 PNG -> HxWx3 uint8 numpy."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))


def _process_img(img: np.ndarray) -> np.ndarray:
    """GrootRoboCasaEnv.process_img 동일 로직: 정사각형 패딩 + 256x256 리사이즈."""
    h, w, _ = img.shape
    if h != w:
        dim = max(h, w)
        y_offset = (dim - h) // 2
        x_offset = (dim - w) // 2
        img = np.pad(img, ((y_offset, y_offset), (x_offset, x_offset), (0, 0)))
    if img.shape[:2] != FINAL_IMAGE_RESOLUTION:
        img = cv2.resize(img, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)
    return np.copy(img)


def _build_groot_obs(payload: dict) -> dict:
    """통일 API payload -> Gr00tSimPolicyWrapper 입력 형식.

    video:    np.ndarray(B=1, T=1, H, W, C) uint8
    state:    np.ndarray(B=1, T=1, D) float32
    language: tuple[str] (B,)
    """
    obs = {}

    # 이미지
    for unified_key, groot_key in UNIFIED_TO_VIDEO_KEY.items():
        b64_str = payload.get(unified_key)
        if b64_str is None:
            continue
        if groot_key in obs:
            continue  # static/side_0 중복 방지
        np_img = _process_img(_b64_to_numpy(b64_str))
        obs[groot_key] = np_img[np.newaxis, np.newaxis, ...]

    # state
    for unified_key, groot_key in UNIFIED_TO_STATE_KEY.items():
        val = payload.get(unified_key)
        if val is not None:
            arr = np.array(val, dtype=np.float32).flatten()
            obs[groot_key] = arr[np.newaxis, np.newaxis, :]

    # language
    obs["annotation.human.action.task_description"] = (payload.get("task", ""),)

    return obs


def _flatten_action(action_dict: dict) -> np.ndarray:
    """action dict -> 2D array (T, total_dim).

    PandaOmron action keys:
      action.base_motion [4], action.control_mode [1],
      action.end_effector_position [3], action.end_effector_rotation [3],
      action.gripper_close [1]
    """
    if not action_dict:
        return np.zeros((1, 7), dtype=np.float32)

    parts = []
    for k in sorted(action_dict.keys()):
        arr = action_dict[k]
        if arr.ndim == 3:
            arr = arr[0]  # (B, T, D) -> (T, D)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 0:
            arr = arr.reshape(1, 1)
        parts.append(arr)

    return np.concatenate(parts, axis=-1).astype(np.float32)


@app.on_event("startup")
def load_model():
    global _policy, _embodiment_tag, _modality_configs

    args = getattr(app.state, "args", None)
    if args is None:
        return

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    _embodiment_tag = EmbodimentTag[args.embodiment_tag]

    print(f"Loading GR00T N1.6...")
    print(f"  Model: {args.model_path}")
    print(f"  Embodiment: {_embodiment_tag}")
    print(f"  Device: {args.device}")

    base_policy = Gr00tPolicy(
        embodiment_tag=_embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=not args.no_strict,
    )
    _policy = Gr00tSimPolicyWrapper(base_policy, strict=not args.no_strict)
    _modality_configs = _policy.get_modality_config()

    for modality, cfg in _modality_configs.items():
        print(f"  [{modality}] keys={cfg.modality_keys}")

    print("GR00T N1.6 loaded.")


@app.post("/reset")
async def reset():
    if _policy is not None:
        _policy.reset()
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    if _policy is None:
        return {"error": "model not loaded"}

    t0 = time.time()
    groot_obs = _build_groot_obs(payload)
    action_dict, info = _policy.get_action(groot_obs)
    latency_ms = (time.time() - t0) * 1000

    # GR00T native action dict → action.* sub-key로 반환
    result = {"latency_ms": latency_ms}
    for k in sorted(action_dict.keys()):
        arr = action_dict[k]
        if arr.ndim == 3:
            arr = arr[0]  # (B, T, D) -> (T, D)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 0:
            arr = arr.reshape(1, 1)
        # "action.base_motion" 등 이미 action. 접두사가 있으면 그대로, 없으면 추가
        key = k if k.startswith("action.") else "action.{}".format(k)
        result[key] = arr.tolist()
    return result


@app.get("/health")
async def health():
    action_horizon = 0
    if _modality_configs is not None:
        action_horizon = len(_modality_configs["action"].delta_indices)

    action_keys = []
    if _modality_configs is not None and "action" in _modality_configs:
        action_keys = [
            k if k.startswith("action.") else "action.{}".format(k)
            for k in _modality_configs["action"].modality_keys
        ]

    return {
        "status": "ok" if _policy is not None else "not_loaded",
        "model": "groot-n1.6",
        "embodiment_tag": _embodiment_tag.value if _embodiment_tag else None,
        "n_action_steps": action_horizon,
        "action_type": "relative",
        "action_keys": action_keys,
    }


def main():
    parser = argparse.ArgumentParser(description="GR00T N1.6 추론 서버 (통일 API)")
    parser.add_argument("--model-path", type=str,
                        default="/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B")
    parser.add_argument("--embodiment-tag", type=str, default="ROBOCASA_PANDA_OMRON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-strict", action="store_true",
                        help="observation/action 검증 비활성화")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8500)
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
