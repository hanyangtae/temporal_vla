"""
GR00T N1.6 추론 서버 (통일 API, port 8500).

프로파일 기반 동작: embodiment_tag, device 등은 configs/checkpoints/*.yaml 에 선언.

groot 컨테이너에서 실행:
  docker compose run --rm groot \
    python /temporal_vla/scripts/serve/groot.py \
    --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml

통일 API:
  POST /act     ← {"observation.images.*": b64png, "observation.state.*": [...],
                    "task": "..."}
                → 프로파일 emits_subkeys 규약에 따른 sub-key dict 반환
  POST /reset   ← 에피소드 시작 시 policy state 초기화
  GET  /health  ← 프로파일 + 모델 modality_configs 반영
"""

import argparse
import base64
import io
import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, "/temporal_vla/src/policies/Isaac-GR00T")

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="GR00T N1.6 Inference Server")

# ─── 글로벌 ──────────────────────────────────────────────────────────────────

_policy = None
_embodiment_tag = None
_modality_configs = None
_profile: Optional[CheckpointProfile] = None
_device = "cuda"

FINAL_IMAGE_RESOLUTION = (256, 256)

# 통일 API 이미지 키 → GR00T video 키 (PandaOmron 기준).
# robocasa native 카메라: left(=agentview_left) / right(=agentview_right) / wrist(=eye_in_hand).
# GR00T modality keys: side_0 / side_1 / wrist_0 ← left / right / eye_in_hand 학습 매핑.
UNIFIED_TO_VIDEO_KEY = {
    "observation.images.left":   "video.res256_image_side_0",
    "observation.images.static": "video.res256_image_side_0",  # left alias (2-camera 모델 호환)
    "observation.images.side_0": "video.res256_image_side_0",
    "observation.images.right":  "video.res256_image_side_1",
    "observation.images.side_1": "video.res256_image_side_1",
    "observation.images.wrist":   "video.res256_image_wrist_0",
    "observation.images.wrist_0": "video.res256_image_wrist_0",
}

# 통일 API state 키 → GR00T state 키 (PandaOmronKeyConverter 기준).
UNIFIED_TO_STATE_KEY = {
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

# GR00T native action key → 통일 sub-key (RoboCasaActionProcessor 가 인식하는 형식).
GROOT_TO_UNIFIED_ACTION = {
    "end_effector_position": "action.eef_pos",
    "end_effector_rotation": "action.eef_axisangle",
    "gripper_close":         "action.gripper",
    "base_motion":           "action.base_motion",
    "control_mode":          "action.control_mode",
}


# ─── 유틸 ────────────────────────────────────────────────────────────────────


def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """base64 PNG → HxWx3 uint8 numpy."""
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


_warned_missing_video_keys: set = set()
_warned_missing_state_keys: set = set()
# embodiment 별 state key → dim. statistics.json 에서 모델 로드 시 채워짐.
_state_dims: dict = {}


def _normalize_modality_key(key: str, prefix: str) -> str:
    """modality_keys 가 'res256_image_side_0' 형태일 수 있으므로 prefix 보장."""
    return key if key.startswith(prefix + ".") else "{}.{}".format(prefix, key)


def _build_groot_obs(payload: dict) -> dict:
    """통일 API payload → Gr00tSimPolicyWrapper 입력 형식.

    video:    np.ndarray(B=1, T=1, H, W, C) uint8
    state:    np.ndarray(B=1, T=1, D) float32
    language: tuple[str] (B,)

    GR00T 가 modality_configs 로 요구하는 video/state 키가 payload 에 없으면
    zero image / zero state 로 fallback (strict assert 통과용). 첫 누락 시 warning.
    """
    obs = {}

    for unified_key, groot_key in UNIFIED_TO_VIDEO_KEY.items():
        b64 = payload.get(unified_key)
        if b64 is None or groot_key in obs:
            continue
        np_img = _process_img(_b64_to_numpy(b64))
        obs[groot_key] = np_img[np.newaxis, np.newaxis, ...]

    for unified_key, groot_key in UNIFIED_TO_STATE_KEY.items():
        val = payload.get(unified_key)
        if val is not None:
            arr = np.array(val, dtype=np.float32).flatten()
            obs[groot_key] = arr[np.newaxis, np.newaxis, :]

    # modality_configs 에 선언된 필수 video/state 키 중 누락 분 zero fallback.
    if _modality_configs is not None:
        h, w = FINAL_IMAGE_RESOLUTION
        zero_img = np.zeros((1, 1, h, w, 3), dtype=np.uint8)

        for vk_raw in _modality_configs["video"].modality_keys:
            vk = _normalize_modality_key(vk_raw, "video")
            if vk not in obs:
                obs[vk] = zero_img
                if vk not in _warned_missing_video_keys:
                    logger.warning(
                        "video key %s missing → zero image fallback", vk,
                    )
                    _warned_missing_video_keys.add(vk)

        for sk_raw in _modality_configs["state"].modality_keys:
            sk = _normalize_modality_key(sk_raw, "state")
            if sk not in obs:
                # state dim 은 statistics.json 에서 가져옴, 없으면 1D
                local = sk[len("state."):]
                dim = _state_dims.get(local, 1)
                obs[sk] = np.zeros((1, 1, dim), dtype=np.float32)
                if sk not in _warned_missing_state_keys:
                    logger.warning(
                        "state key %s missing → zero(%dD) fallback", sk, dim,
                    )
                    _warned_missing_state_keys.add(sk)

    obs["annotation.human.action.task_description"] = (payload.get("task", ""),)
    return obs


def _load_state_dims_from_statistics(model_path: str, embodiment_value: str) -> dict:
    """checkpoints 의 statistics.json 에서 embodiment 의 state key dim map 추출."""
    import json
    import os

    stats_path = os.path.join(model_path, "statistics.json")
    if not os.path.exists(stats_path):
        logger.warning("statistics.json not found under %s", model_path)
        return {}
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        emb = stats.get(embodiment_value, {})
        state = emb.get("state", {})
        dims = {}
        for k, v in state.items():
            if isinstance(v, dict) and "mean" in v and hasattr(v["mean"], "__len__"):
                dims[k] = len(v["mean"])
        return dims
    except Exception as e:  # pragma: no cover
        logger.warning("failed to parse statistics.json: %s", e)
        return {}


# ─── 모델 로딩 ───────────────────────────────────────────────────────────────


@app.on_event("startup")
def load_model():
    try:
        _load_model_impl()
    except Exception:
        import traceback
        sys.stderr.write("=== load_model FAILED ===\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _load_model_impl():
    global _policy, _embodiment_tag, _modality_configs, _device

    args = app.state.args
    profile = _profile
    assert profile is not None, "profile must be set before load_model"

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    ms = profile.model_specific
    _device = ms.get("device", args.device)
    _embodiment_tag = EmbodimentTag[ms["embodiment_tag"]]

    model_path = profile.checkpoint_source.id
    strict = not bool(ms.get("no_strict", False))
    logger.info(
        "Loading GR00T from %s (profile=%s, embodiment=%s, device=%s, strict=%s)",
        model_path, profile.name, _embodiment_tag.value, _device, strict,
    )

    base_policy = Gr00tPolicy(
        embodiment_tag=_embodiment_tag,
        model_path=model_path,
        device=_device,
        strict=strict,
    )
    _policy = Gr00tSimPolicyWrapper(base_policy, strict=strict)
    _modality_configs = _policy.get_modality_config()

    for modality, cfg in _modality_configs.items():
        logger.info("[%s] modality_keys=%s", modality, cfg.modality_keys)

    horizon = len(_modality_configs["action"].delta_indices)

    # state dim map (fallback zero state 시 정확한 dim 으로 채우기 위함)
    global _state_dims
    _state_dims = _load_state_dims_from_statistics(model_path, _embodiment_tag.value)
    logger.info("state dims loaded from statistics.json: %s", _state_dims)

    logger.info(
        "GR00T loaded. action_horizon=%d (profile.n_action_steps=%d)",
        horizon, profile.n_action_steps,
    )


# ─── FastAPI 엔드포인트 ──────────────────────────────────────────────────────


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
    profile = _profile
    assert profile is not None

    groot_obs = _build_groot_obs(payload)
    action_dict, info = _policy.get_action(groot_obs)
    latency_ms = (time.time() - t0) * 1000

    # GR00T native action dict → 통일 sub-key dict.
    # action_dict 의 key 는 "end_effector_position" 또는 "action.end_effector_position" 형태로 들어올 수 있음.
    # 이름 정규화 후 GROOT_TO_UNIFIED_ACTION 으로 매핑.
    out = {"latency_ms": latency_ms}
    for k_raw, arr in action_dict.items():
        k = k_raw[len("action."):] if k_raw.startswith("action.") else k_raw
        unified = GROOT_TO_UNIFIED_ACTION.get(k)
        if unified is None:
            # 알 수 없는 키 → action. 접두사 유지하여 그대로 보존
            unified = "action.{}".format(k)
        if arr.ndim == 3:
            arr = arr[0]   # (B, T, D) → (T, D)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 0:
            arr = arr.reshape(1, 1)
        out[unified] = arr.tolist()

    # 프로파일이 emits_subkeys 로 선언한 키가 빠졌으면 빈 array 라도 채워주기 (디버깅 명확성)
    horizon = len(_modality_configs["action"].delta_indices) if _modality_configs else 1
    for sk in profile.emits_subkeys:
        if sk not in out:
            local = sk[len("action."):]
            if local in {a.name for a in profile.action_layout}:
                dim = profile.dim_slice(local).stop - profile.dim_slice(local).start
            else:
                dim = 1
            out[sk] = [[0.0] * dim for _ in range(horizon)]
            logger.warning("emit sub-key %s missing from GR00T output, filled zeros", sk)

    # 디버그 (첫 5 호출)
    step_count = getattr(app.state, "_step", 0) + 1
    app.state._step = step_count
    if step_count <= 5:
        logger.info(
            "call=%d horizon=%d pos=%s grip=%s task=%r",
            step_count, horizon,
            np.array(out.get("action.eef_pos", [[0]*3]))[0].round(4).tolist(),
            np.array(out.get("action.gripper", [[0]]))[0].round(4).tolist(),
            payload.get("task", "")[:40],
        )

    return out


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "groot"}
    horizon = (
        len(_modality_configs["action"].delta_indices)
        if _modality_configs is not None else _profile.n_action_steps
    )
    return {
        "status": "ok" if _policy is not None else "not_loaded",
        "model": "groot-n1.6",
        "profile": _profile.name,
        "embodiment_tag": _embodiment_tag.value if _embodiment_tag else None,
        "n_action_steps": horizon,
        "action_type": _profile.action_type,
        "action_keys": list(_profile.emits_subkeys),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    global _profile

    try:
        from src.utils.common.logger import create_module_logger

        create_module_logger("groot_serve")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GR00T N1.6 추론 서버 (port 8500)")
    parser.add_argument(
        "--profile", type=str, required=True,
        help="체크포인트 프로파일 YAML 경로 (configs/checkpoints/*.yaml)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _profile = load_profile(args.profile)
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "groot", (
        f"profile.base_model={_profile.base_model!r}, but this server is groot"
    )

    app.state.args = args
    app.state._step = 0
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
