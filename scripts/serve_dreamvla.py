"""
DreamVLA 추론 서버.

dreamvla 컨테이너 내에서 실행:
  docker compose --profile dreamvla run --rm dreamvla \
    python /temporal_vla/scripts/serve_dreamvla.py \
      --checkpoint /temporal_vla/checkpoints/dreamvla/checkpoint.pt \
      --vit-checkpoint /temporal_vla/checkpoints/vit_mae.pth

robocasa 컨테이너에서 HTTP로 액션 요청:
  POST http://localhost:8200/act
  POST http://localhost:8200/reset  (에피소드 시작 시 히스토리 초기화)
"""

import argparse
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

# 전역 모델 상태
model = None
clip_model = None
image_processor = None
tokenizer = None
cast_dtype = torch.float32

# 히스토리 큐 (에피소드 단위 상태)
HISTORY_LEN = 10
img_queue: deque = deque(maxlen=HISTORY_LEN)
gripper_queue: deque = deque(maxlen=HISTORY_LEN)
state_queue: deque = deque(maxlen=HISTORY_LEN)
text_queue: deque = deque(maxlen=HISTORY_LEN)

# RoboCasa 카메라 → DreamVLA 키 매핑
STATIC_CAM = "robot0_agentview_left"
WRIST_CAM = "robot0_eye_in_hand"


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
        depth_pred=False,
        trajectory_pred=False,
        pred_num=args.action_pred_steps,
        use_trajectory_query=False,
        track_label_patch_size=16,
        use_dinosiglip=False,
        use_dit_head=False,
        dino_feat_pred=False,
        sam_feat_pred=False,
        no_pred_gripper_traj=True,
        no_unshuffle=False,
        use_gpt2_pretrained=False,
        share_query=False,
        attn_implementation="eager",
        dit_type="dit",
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    # DDP로 저장된 경우 "module." prefix 제거
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    if args.precision == "bf16":
        cast_dtype = torch.bfloat16
        model = model.bfloat16()
    elif args.precision == "fp16":
        cast_dtype = torch.float16
        model = model.half()

    model = model.cuda().eval()
    print("DreamVLA loaded.")


def _reset_queues():
    global img_queue, gripper_queue, state_queue, text_queue
    img_queue = deque(maxlen=HISTORY_LEN)
    gripper_queue = deque(maxlen=HISTORY_LEN)
    state_queue = deque(maxlen=HISTORY_LEN)
    text_queue = deque(maxlen=HISTORY_LEN)


def _preprocess_image(np_image: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 → [1, 1, C, H, W] 텐서."""
    pil = Image.fromarray(np_image)
    tensor = image_processor(pil).unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
    return tensor.to(dtype=cast_dtype)


def _preprocess_state(state_list: list) -> torch.Tensor:
    """proprioceptive state → [1, 1, 7] 텐서 (6 arm + 1 gripper)."""
    state = np.array(state_list, dtype=np.float32)
    # RoboCasa state가 7차원보다 길 수 있으므로 앞 6개 + 마지막 1개 사용
    state7 = np.concatenate([state[:6], state[[-1]]])
    tensor = torch.from_numpy(state7).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
    return tensor.to(dtype=cast_dtype)


def _preprocess_text(instruction: str) -> torch.Tensor:
    """str → [1, 1, 77] CLIP 토큰."""
    tokens = tokenizer.tokenize([instruction], truncate=True)  # [1, 77]
    return tokens.unsqueeze(1)  # [1, 1, 77]


@app.post("/reset")
async def reset():
    """에피소드 시작 시 히스토리 초기화."""
    _reset_queues()
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    """
    Payload:
      - images: dict of camera_name -> HxWx3 uint8 list
        필수 키: "robot0_agentview_left", "robot0_eye_in_hand"
      - state: list[float]  (proprioceptive state, 최소 7차원)
      - language_instruction: str
    Returns:
      - action: list[float]  (7-DoF: 6 arm + 1 gripper [-1 or 1])
      - world_pred: list | null  (이미지 예측 토큰, 옵션)
      - latency_ms: float
    """
    if model is None:
        return {"error": "model not loaded"}

    t0 = time.time()

    images = payload.get("images", {})
    static_np = np.array(images.get(STATIC_CAM, images.get("static", np.zeros((224, 224, 3), dtype=np.uint8))), dtype=np.uint8)
    wrist_np = np.array(images.get(WRIST_CAM, images.get("gripper", np.zeros((224, 224, 3), dtype=np.uint8))), dtype=np.uint8)
    state_list = payload.get("state", [0.0] * 7)
    instruction = payload.get("language_instruction", "")

    image_x = _preprocess_image(static_np).cuda()
    gripper_x = _preprocess_image(wrist_np).cuda()
    state_x = _preprocess_state(state_list).cuda()
    text_x = _preprocess_text(instruction)

    img_queue.append(image_x)
    gripper_queue.append(gripper_x)
    state_queue.append(state_x)

    if len(text_queue) == 0:
        for _ in range(HISTORY_LEN):
            text_queue.append(text_x)

    image_primary = torch.cat(list(img_queue), dim=1)    # [1, T, C, H, W]
    image_wrist = torch.cat(list(gripper_queue), dim=1)
    state = torch.cat(list(state_queue), dim=1)          # [1, T, 7]
    text_token = torch.cat(list(text_queue), dim=1)      # [1, T, 77]

    # 히스토리가 sequence_length보다 짧으면 마지막 프레임으로 패딩
    seq_len = app.state.args.sequence_length
    num_step = image_primary.shape[1]
    if num_step < seq_len:
        pad = seq_len - num_step
        image_primary = torch.cat([image_primary, image_primary[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        image_wrist = torch.cat([image_wrist, image_wrist[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        state = torch.cat([state, state[:, -1:].expand(-1, pad, -1)], dim=1)

    with torch.inference_mode():
        (
            arm_pred,
            gripper_pred,
            image_pred,
            _arm_state,
            _gripper_state,
            _loss,
            _depth,
            _traj,
            _dino,
            _sam,
        ) = model(
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            action=torch.zeros(1, seq_len, 7, device="cuda", dtype=cast_dtype),
            mode="test",
        )

    # 현재 스텝의 action 추출
    step_idx = min(num_step, seq_len) - 1
    arm = arm_pred[0, step_idx, 0, :]          # [6]
    gripper = (gripper_pred[0, step_idx, 0, :] > 0.5).float()  # [1]
    gripper = (gripper - 0.5) * 2              # {0,1} → {-1,1}
    action = torch.cat([arm, gripper], dim=-1).cpu().float().numpy()  # [7]

    # world_pred: image_pred 텐서 (선택적 반환)
    world_pred = None
    if image_pred is not None:
        world_pred = image_pred[0, step_idx].cpu().float().numpy().tolist()

    latency_ms = (time.time() - t0) * 1000

    return {
        "action": action.tolist(),
        "world_pred": world_pred,
        "latency_ms": latency_ms,
    }


@app.get("/health")
async def health():
    return {"status": "ok" if model is not None else "not_loaded", "model": "dreamvla"}


def main():
    parser = argparse.ArgumentParser(description="DreamVLA 추론 서버")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="학습된 체크포인트 경로 (.pt / .pth)")
    parser.add_argument("--vit-checkpoint", type=str, default=None,
                        help="ViT-MAE 체크포인트 경로 (없으면 랜덤 초기화)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    # 모델 구조 (학습 시 사용한 값과 일치해야 함)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--action-pred-steps", type=int, default=1)
    parser.add_argument("--atten-goal", type=int, default=0)
    parser.add_argument("--num-resampler-query", type=int, default=9)
    parser.add_argument("--num-obs-token-per-image", type=int, default=9)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--obs-pred", action="store_true")
    parser.add_argument("--phase", type=str, default="pretrain")
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
