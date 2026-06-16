"""
DreamVLA 추론 서버 (통일 API, port 8200).

프로파일 기반 동작: 모든 모델 하이퍼파라미터(sequence_length, hidden_dim,
DiT type 등) 와 체크포인트 경로는 configs/checkpoints/*.yaml 에 선언.

dreamvla 컨테이너에서 실행:
  docker compose run --rm dreamvla \
    python /temporal_vla/scripts/serve/dreamvla.py \
    --profile /temporal_vla/configs/checkpoints/dreamvla__calvin_dynamic_depth_semantic.yaml

통일 API:
  POST /act     ← {"observation.images.*": b64png, "observation.state.*": [...],
                    "task": "..."}
                → 프로파일 emits_subkeys 규약대로 sub-key dict 반환
  POST /reset   ← 에피소드 시작 시 history queue 초기화
  GET  /health  ← 프로파일 기반 서버 상태
"""

import argparse
import logging
import math
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, "/temporal_vla/src/policies/dreamvla")

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

from src.utils.common.image import decode_b64_image  # noqa: E402
from src.utils.common.serving import (  # noqa: E402
    add_server_args,
    health_response,
    run_uvicorn,
    setup_serve_logging,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="DreamVLA Inference Server")

# ─── 글로벌 ──────────────────────────────────────────────────────────────────

_model = None
_clip_model = None
_image_processor = None
_tokenizer = None
_cast_dtype = torch.float32
_profile: Optional[CheckpointProfile] = None
_device = "cuda"

# History queues (프로파일의 history_len 으로 lazy 초기화)
_img_queue: Optional[deque] = None
_gripper_queue: Optional[deque] = None
_state_queue: Optional[deque] = None
_text_queue: Optional[deque] = None


# ─── 유틸 ────────────────────────────────────────────────────────────────────


def _quat_xyzw_to_euler(quat) -> np.ndarray:
    """Quaternion (x,y,z,w) → euler (roll, pitch, yaw)."""
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


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
    global _model, _clip_model, _image_processor, _tokenizer, _cast_dtype, _device
    global _img_queue, _gripper_queue, _state_queue, _text_queue

    args = app.state.args
    profile = _profile
    assert profile is not None, "profile must be set before load_model"
    _device = args.device

    ms = profile.model_specific
    history_len = int(ms.get("history_len", ms.get("sequence_length", 10)))
    _img_queue = deque(maxlen=history_len)
    _gripper_queue = deque(maxlen=history_len)
    _state_queue = deque(maxlen=history_len)
    _text_queue = deque(maxlen=history_len)

    import clip as clip_lib
    from models.dreamvla_model import DreamVLA

    checkpoint_path = profile.checkpoint_source.id
    logger.info("Loading DreamVLA from %s (profile=%s)", checkpoint_path, profile.name)

    _clip_model, _image_processor = clip_lib.load("ViT-B/32", device="cpu")
    _tokenizer = clip_lib

    _model = DreamVLA(
        finetune_type="",
        clip_device="cpu",
        vit_checkpoint_path=ms["vit_checkpoint"],
        sequence_length=int(ms.get("sequence_length", 10)),
        num_resampler_query=int(ms.get("num_resampler_query", 16)),
        num_obs_token_per_image=int(ms.get("num_obs_token_per_image", 9)),
        calvin_input_image_size=int(ms.get("image_size", profile.image_preprocess.resolution)),
        patch_size=int(ms.get("patch_size", 16)),
        action_pred_steps=int(ms.get("action_pred_steps", 3)),
        obs_pred=bool(ms.get("obs_pred", True)),
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=int(ms.get("atten_goal", 0)),
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        transformer_layers=int(ms.get("transformer_layers", 24)),
        hidden_dim=int(ms.get("hidden_dim", 1024)),
        transformer_heads=int(ms.get("transformer_heads", 16)),
        phase=ms.get("phase", "evaluate"),
        gripper_width=False,
        depth_pred=bool(ms.get("depth_pred", True)),
        trajectory_pred=False,
        pred_num=int(ms.get("pred_num", 1)),
        use_trajectory_query=False,
        track_label_patch_size=16,
        use_dinosiglip=False,
        use_dit_head=bool(ms.get("use_dit_head", True)),
        dino_feat_pred=False,
        sam_feat_pred=bool(ms.get("sam_feat_pred", True)),
        no_pred_gripper_traj=True,
        no_unshuffle=False,
        use_gpt2_pretrained=False,
        share_query=False,
        attn_implementation=ms.get("attn_implementation", "sdpa"),
        dit_type=ms.get("dit_type", "DiT-B"),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    _model.load_state_dict(state_dict, strict=False)

    precision = ms.get("precision", "fp32")
    if precision == "bf16":
        _cast_dtype = torch.bfloat16
        _model = _model.bfloat16()
    elif precision == "fp16":
        _cast_dtype = torch.float16
        _model = _model.half()
    else:
        _cast_dtype = torch.float32

    _model = _model.to(_device).eval()
    _model._init_model_type()
    logger.info(
        "DreamVLA loaded. precision=%s, sequence_length=%d, action_dim=%d",
        precision, int(ms.get("sequence_length", 10)), profile.action_dim,
    )


# ─── 전처리 ──────────────────────────────────────────────────────────────────


def _preprocess_image(np_image: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 → [1, 1, C, H, W]"""
    pil = Image.fromarray(np_image)
    return _image_processor(pil).unsqueeze(0).unsqueeze(0).to(dtype=_cast_dtype)


def _assemble_state(payload: dict) -> list:
    """payload obs → DreamVLA 7D state: eef_pos(3) + eef_euler(3) + gripper(1).

    fallback:
      eef_euler 없음 → eef_quat 변환
      gripper_action 없음 → gripper_qpos 임계값
    """
    eef_pos = payload.get("observation.state.eef_pos")
    if eef_pos is None:
        return [0.0] * 7

    eef_euler = payload.get("observation.state.eef_euler")
    if eef_euler is None:
        eef_quat = payload.get("observation.state.eef_quat")
        eef_euler = _quat_xyzw_to_euler(eef_quat).tolist() if eef_quat else [0.0, 0.0, 0.0]

    gripper_action = payload.get("observation.state.gripper_action")
    if gripper_action is not None:
        grip = float(gripper_action[0]) if hasattr(gripper_action, "__len__") else float(gripper_action)
    else:
        gripper_qpos = payload.get("observation.state.gripper_qpos")
        grip = 1.0 if (gripper_qpos is not None and float(gripper_qpos[0]) > 0.01) else -1.0

    pos = list(eef_pos) if not isinstance(eef_pos, list) else eef_pos
    orn = list(eef_euler) if not isinstance(eef_euler, list) else eef_euler
    return pos[:3] + orn[:3] + [grip]


def _preprocess_state(state_7d: list) -> torch.Tensor:
    """7D state → [1, 1, 7] tensor."""
    return torch.tensor(state_7d, dtype=_cast_dtype).unsqueeze(0).unsqueeze(0)


def _preprocess_text(instruction: str) -> torch.Tensor:
    """str → [1, 1, 77]"""
    return _tokenizer.tokenize([instruction], truncate=True).unsqueeze(1)


# ─── FastAPI 엔드포인트 ──────────────────────────────────────────────────────


def _reset_queues():
    global _img_queue, _gripper_queue, _state_queue, _text_queue
    profile = _profile
    assert profile is not None
    history_len = int(profile.model_specific.get(
        "history_len", profile.model_specific.get("sequence_length", 10),
    ))
    _img_queue = deque(maxlen=history_len)
    _gripper_queue = deque(maxlen=history_len)
    _state_queue = deque(maxlen=history_len)
    _text_queue = deque(maxlen=history_len)


@app.post("/reset")
async def reset():
    """에피소드 시작 시 history 초기화."""
    _reset_queues()
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    """통일 API: 이미지 + state + instruction → action sub-keys."""
    if _model is None:
        return {"error": "model not loaded"}

    t0 = time.time()
    profile = _profile
    assert profile is not None
    ms = profile.model_specific

    static_b64 = payload.get("observation.images.static")
    wrist_b64 = payload.get("observation.images.wrist")
    img_size = int(ms.get("image_size", profile.image_preprocess.resolution))
    static_np = decode_b64_image(static_b64) if static_b64 else np.zeros((img_size, img_size, 3), dtype=np.uint8)
    wrist_np = decode_b64_image(wrist_b64) if wrist_b64 else np.zeros((img_size, img_size, 3), dtype=np.uint8)
    state_list = _assemble_state(payload)
    instruction = payload.get("task", "")

    image_x = _preprocess_image(static_np).to(_device)
    gripper_x = _preprocess_image(wrist_np).to(_device)
    state_x = _preprocess_state(state_list).to(_device)
    text_x = _preprocess_text(instruction).to(_device)

    _img_queue.append(image_x)
    _gripper_queue.append(gripper_x)
    _state_queue.append(state_x)
    if len(_text_queue) == 0:
        for _ in range(_text_queue.maxlen or 10):
            _text_queue.append(text_x)

    image_primary = torch.cat(list(_img_queue), dim=1)
    image_wrist = torch.cat(list(_gripper_queue), dim=1)
    state = torch.cat(list(_state_queue), dim=1)
    text_token = torch.cat(list(_text_queue), dim=1)

    seq_len = int(ms.get("sequence_length", 10))
    num_step = image_primary.shape[1]
    if num_step < seq_len:
        pad = seq_len - num_step
        image_primary = torch.cat([image_primary, image_primary[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        image_wrist = torch.cat([image_wrist, image_wrist[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        state = torch.cat([state, state[:, -1:].expand(-1, pad, -1)], dim=1)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=_cast_dtype):
        arm_pred, gripper_pred, image_pred, *_ = _model(
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            action=torch.zeros(1, seq_len, profile.action_dim, device=_device, dtype=_cast_dtype),
            mode="test",
        )

    # arm_pred: [1, seq_len, pred_num, 6], gripper_pred: [1, seq_len, pred_num, 1]
    arm = arm_pred[0, :, 0, :]                          # [seq_len, 6]
    gripper = (gripper_pred[0, :, 0, :] > 0.5).float()  # binary 0/1
    gripper = (gripper - 0.5) * 2                       # → -1 / 1
    all_actions = torch.cat([arm, gripper], dim=-1)     # [seq_len, 7]

    if num_step < seq_len:
        action = all_actions[num_step - 1]
    else:
        action = all_actions[-1]
    action = action.cpu().float().numpy()  # shape [action_dim]

    latency_ms = (time.time() - t0) * 1000

    # Sub-key emit (n_action_steps=1 → 길이 1 list 로 wrap)
    out = {}
    for sk in profile.emits_subkeys:
        local_name = sk[len("action."):]
        sl = profile.dim_slice(local_name)
        out[sk] = [action[sl].tolist()]  # [[dim]]

    # 디버그 (첫 5 step)
    if len(_img_queue) <= 5:
        logger.info(
            "step=%d num_step=%d state=%s action=%s task=%r",
            len(_img_queue), num_step,
            [round(v, 4) for v in state_list],
            action.round(4).tolist(),
            instruction[:30],
        )

    out["latency_ms"] = latency_ms
    return out


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "dreamvla"}
    return health_response(
        policy=_model,
        model="dreamvla",
        profile=_profile,
        n_action_steps=_profile.n_action_steps,
        action_type=_profile.action_type,
        action_keys=list(_profile.emits_subkeys),
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    global _profile

    setup_serve_logging("dreamvla_serve")

    parser = argparse.ArgumentParser(description="DreamVLA 추론 서버 (port 8200)")
    parser.add_argument(
        "--profile", type=str, required=True,
        help="체크포인트 프로파일 YAML 경로 (configs/checkpoints/*.yaml)",
    )
    add_server_args(parser, default_port=8200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _profile = load_profile(args.profile)
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "dreamvla", (
        f"profile.base_model={_profile.base_model!r}, but this server is dreamvla"
    )

    app.state.args = args
    run_uvicorn(app, args)


if __name__ == "__main__":
    main()
