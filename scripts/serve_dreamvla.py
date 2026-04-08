"""
DreamVLA 추론 서버 (통일 API).

dreamvla 컨테이너 내에서 실행:
  docker compose run --rm dreamvla \
    python /temporal_vla/scripts/serve_dreamvla.py \
    --checkpoint /temporal_vla/checkpoints/dreamvla/dreamvla_dynamic_depth_semantic-001.pth \
    --vit-checkpoint /temporal_vla/checkpoints/mae_pretrain_vit_base.pth \
    --precision fp32 \
    --obs-pred --depth-pred --sam-feat-pred --use-dit-head \
    --pred-num 1 --attn-implementation sdpa --phase evaluate

통일 API (LeRobot 컨벤션):
  POST /act     ← {"observation.images.static": b64png, "observation.images.wrist": b64png,
                    "observation.state": [...], "task": "..."}
                → {"action": [[7 floats]], "latency_ms": float}
  POST /reset   ← 에피소드 시작 시 히스토리 초기화
  GET  /health  ← feature 정보 포함
"""

import argparse
import base64
import io
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, "/temporal_vla/src/policies/dreamvla")

app = FastAPI(title="DreamVLA Inference Server")

model = None
clip_model = None
image_processor = None
tokenizer = None
cast_dtype = torch.float32

HISTORY_LEN = 10
img_queue: deque = deque(maxlen=HISTORY_LEN)
gripper_queue: deque = deque(maxlen=HISTORY_LEN)
state_queue: deque = deque(maxlen=HISTORY_LEN)
text_queue: deque = deque(maxlen=HISTORY_LEN)

def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """base64 PNG → HxWx3 uint8 numpy."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))


@app.on_event("startup")
def load_model():
    global model, clip_model, image_processor, tokenizer, cast_dtype

    args = app.state.args
    print(f"Loading DreamVLA from {args.checkpoint} ...")

    import clip as clip_lib
    from models.dreamvla_model import DreamVLA

    clip_model, image_processor = clip_lib.load("ViT-B/32", device="cpu")
    tokenizer = clip_lib

    model = DreamVLA(
        finetune_type="",
        clip_device="cpu",
        vit_checkpoint_path=args.vit_checkpoint,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=args.atten_goal,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=False,
        depth_pred=args.depth_pred,
        trajectory_pred=False,
        pred_num=args.pred_num,
        use_trajectory_query=False,
        track_label_patch_size=16,
        use_dinosiglip=False,
        use_dit_head=args.use_dit_head,
        dino_feat_pred=False,
        sam_feat_pred=args.sam_feat_pred,
        no_pred_gripper_traj=True,
        no_unshuffle=False,
        use_gpt2_pretrained=False,
        share_query=False,
        attn_implementation=args.attn_implementation,
        dit_type=args.dit_type,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    if args.precision == "bf16":
        cast_dtype = torch.bfloat16
        model = model.bfloat16()
    elif args.precision == "fp16":
        cast_dtype = torch.float16
        model = model.half()

    model = model.cuda().eval()
    model._init_model_type()
    print("DreamVLA loaded.")


def _reset_queues():
    global img_queue, gripper_queue, state_queue, text_queue
    img_queue = deque(maxlen=HISTORY_LEN)
    gripper_queue = deque(maxlen=HISTORY_LEN)
    state_queue = deque(maxlen=HISTORY_LEN)
    text_queue = deque(maxlen=HISTORY_LEN)


def _preprocess_image(np_image: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 → [1, 1, C, H, W]"""
    pil = Image.fromarray(np_image)
    return image_processor(pil).unsqueeze(0).unsqueeze(0).to(dtype=cast_dtype)


def _quat_xyzw_to_euler(quat):
    """Quaternion (x,y,z,w) → euler (roll, pitch, yaw)."""
    import math
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _assemble_state_7d(payload: dict) -> list:
    """payload의 observation.state.* sub-keys에서 DreamVLA용 7D state 조립.

    DreamVLA는 Calvin 학습 기준: eef_pos(3) + eef_euler(3) + gripper(1) = 7D.
    벤치마크에 따라 euler/quat, gripper 키가 다를 수 있으므로 fallback 포함.
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
        grip = float(gripper_action[0]) if hasattr(gripper_action, '__len__') else float(gripper_action)
    else:
        gripper_qpos = payload.get("observation.state.gripper_qpos")
        grip = 1.0 if (gripper_qpos is not None and float(gripper_qpos[0]) > 0.01) else -1.0

    pos = list(eef_pos) if not isinstance(eef_pos, list) else eef_pos
    orn = list(eef_euler) if not isinstance(eef_euler, list) else eef_euler
    return pos[:3] + orn[:3] + [grip]


def _preprocess_state(state_7d: list) -> torch.Tensor:
    """7D state → [1, 1, 7] tensor."""
    return torch.tensor(state_7d, dtype=cast_dtype).unsqueeze(0).unsqueeze(0)


def _preprocess_text(instruction: str) -> torch.Tensor:
    """str → [1, 1, 77]"""
    return tokenizer.tokenize([instruction], truncate=True).unsqueeze(1)


@app.post("/reset")
async def reset():
    """에피소드 시작 시 히스토리 초기화."""
    _reset_queues()
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    """
    통일 API:
      요청: {"observation.images.static": b64png, "observation.images.wrist": b64png,
             "observation.state.eef_pos": [...], "observation.state.eef_quat": [...], ...,
             "task": "..."}
      응답: {"action": [[7 floats]], "latency_ms": float}
    """
    if model is None:
        return {"error": "model not loaded"}

    t0 = time.time()

    static_b64 = payload.get("observation.images.static")
    wrist_b64 = payload.get("observation.images.wrist")
    static_np = _b64_to_numpy(static_b64) if static_b64 else np.zeros((224, 224, 3), dtype=np.uint8)
    wrist_np = _b64_to_numpy(wrist_b64) if wrist_b64 else np.zeros((224, 224, 3), dtype=np.uint8)
    state_list = _assemble_state_7d(payload)
    instruction = payload.get("task", "")

    image_x = _preprocess_image(static_np).cuda()
    gripper_x = _preprocess_image(wrist_np).cuda()
    state_x = _preprocess_state(state_list).cuda()
    text_x = _preprocess_text(instruction).cuda()

    img_queue.append(image_x)
    gripper_queue.append(gripper_x)
    state_queue.append(state_x)
    if len(text_queue) == 0:
        for _ in range(HISTORY_LEN):
            text_queue.append(text_x)

    image_primary = torch.cat(list(img_queue), dim=1)
    image_wrist = torch.cat(list(gripper_queue), dim=1)
    state = torch.cat(list(state_queue), dim=1)
    text_token = torch.cat(list(text_queue), dim=1)

    seq_len = app.state.args.sequence_length
    num_step = image_primary.shape[1]
    if num_step < seq_len:
        pad = seq_len - num_step
        image_primary = torch.cat([image_primary, image_primary[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        image_wrist = torch.cat([image_wrist, image_wrist[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        state = torch.cat([state, state[:, -1:].expand(-1, pad, -1)], dim=1)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=cast_dtype):
        arm_pred, gripper_pred, image_pred, *_ = model(
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            action=torch.zeros(1, seq_len, 7, device="cuda", dtype=cast_dtype),
            mode="test",
        )

    # 원본 eval과 동일: 전체 pred_steps의 action을 반환
    # arm_pred: [1, seq_len, pred_num, 6], gripper_pred: [1, seq_len, pred_num, 1]
    arm = arm_pred[0, :, 0, :]        # [seq_len, 6]
    gripper = (gripper_pred[0, :, 0, :] > 0.5).float()
    gripper = (gripper - 0.5) * 2     # [seq_len, 1]
    all_actions = torch.cat([arm, gripper], dim=-1)  # [seq_len, 7]

    if num_step < seq_len:
        action = all_actions[num_step - 1]  # [7]
    else:
        action = all_actions[-1]            # [7]

    action = action.cpu().float().numpy()

    # 디버그 로그 (첫 5스텝만)
    if len(img_queue) <= 5:
        print(f"[DEBUG] step={len(img_queue)} num_step={num_step} "
              f"state_7d={[round(v, 4) for v in state_list]} "
              f"action={action.round(4).tolist()} "
              f"task='{instruction[:30]}'")

    return {
        "action.eef_pos": [action[:3].tolist()],
        "action.eef_euler": [action[3:6].tolist()],
        "action.gripper": [action[6:7].tolist()],
        "latency_ms": (time.time() - t0) * 1000,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "not_loaded",
        "model": "dreamvla",
        "n_action_steps": 1,
        "action_type": "relative",
        "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
    }


def main():
    parser = argparse.ArgumentParser(description="DreamVLA 추론 서버")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="학습된 체크포인트 경로 (.pt / .pth)")
    parser.add_argument("--vit-checkpoint", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--action-pred-steps", type=int, default=3)
    parser.add_argument("--atten-goal", type=int, default=0)
    parser.add_argument("--num-resampler-query", type=int, default=16)
    parser.add_argument("--num-obs-token-per-image", type=int, default=9)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--transformer-layers", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--transformer-heads", type=int, default=16)
    parser.add_argument("--obs-pred", action="store_true")
    parser.add_argument("--depth-pred", action="store_true")
    parser.add_argument("--sam-feat-pred", action="store_true")
    parser.add_argument("--use-dit-head", action="store_true")
    parser.add_argument("--pred-num", type=int, default=1)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--dit-type", type=str, default="DiT-B", choices=["DiT-S", "DiT-B", "DiT-L"])
    parser.add_argument("--phase", type=str, default="evaluate")
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
