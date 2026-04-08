"""
X-VLA 추론 서버 (통일 API, port 8100).

xvla 컨테이너에서 실행:
  docker compose run --rm xvla \
    python /temporal_vla/scripts/serve_xvla.py \
    --pretrained-path 2toINF/X-VLA-Calvin-ABC_D

통일 API:
  POST /act     ← {"observation.images.static": b64png, ...,
                    "observation.state.eef_pos": [...], ...,
                    "task": "..."}
                → {"action.eef_pos": [[3]], "action.eef_rot6d": [[6]],
                    "action.gripper": [[1]], "latency_ms": float}
  POST /reset   ← no-op (X-VLA는 히스토리 없음)
  GET  /health  ← 서버 상태 + 모델 정보

X-VLA Calvin 체크포인트는 ee6d action mode (20D, dual-arm).
서버는 모델 native format 그대로 sub-key로 분리하여 반환한다.
  - action.eef_pos: arm1 position (3D, absolute)
  - action.eef_rot6d: arm1 6D rotation (6D, absolute)
  - action.gripper: arm1 gripper sigmoid (1D)
회전 변환(rot6d→euler 등)은 벤치마크 측 ActionProcessor가 담당.
"""

import argparse
import base64
import io
import math
import sys
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, "/temporal_vla/lerobot/src")

app = FastAPI(title="X-VLA Inference Server")

# ─── 글로벌 ──────────────────────────────────────────────────────────────────

model = None
_tokenizer = None
_n_action_steps = 30
_denoising_steps = 10
_max_views = 3
_device = "cuda"
_domain_id = 2  # Calvin domain ID (X-VLA domain_config.py 기준)
_image_size = 224  # X-VLA Calvin 학습 해상도

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── 유틸 ────────────────────────────────────────────────────────────────────


def _b64_to_pil(b64_str: str) -> Image.Image:
    """base64 PNG → PIL Image (RGB)."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _euler_to_rot6d(euler: np.ndarray) -> np.ndarray:
    """Euler angles (roll, pitch, yaw) → 6D rotation representation.

    Calvin euler 순서: roll(x), pitch(y), yaw(z).
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll), rot6d = [R[:,0], R[:,1]].
    """
    r, p, y = float(euler[0]), float(euler[1]), float(euler[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float32)

    return np.concatenate([R[:, 0], R[:, 1]])  # (6,)


# ─── 전처리 ──────────────────────────────────────────────────────────────────


def _preprocess_images(pil_images: list) -> tuple:
    """PIL images → (image_input [1, V, C, H, W], image_mask [1, V]).

    224×224로 리사이즈 후 ImageNet 정규화.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    tensors = []
    for img in pil_images:
        img = img.resize((_image_size, _image_size), Image.BILINEAR)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        t = (t - mean) / std
        tensors.append(t)
    masks = [True] * len(tensors)

    while len(tensors) < _max_views:
        tensors.append(torch.zeros_like(tensors[0]))
        masks.append(False)

    image_input = torch.stack(tensors).unsqueeze(0).to(_device)
    image_mask = torch.tensor([masks], dtype=torch.bool, device=_device)
    return image_input, image_mask


def _assemble_proprio(payload: dict) -> torch.Tensor:
    """Calvin obs sub-keys → ee6d proprio tensor [1, 20].

    ee6d layout (dual-arm, Calvin은 single-arm이므로 arm2는 전부 0):
      Arm1 [0:10]:  eef_pos(3) + eef_rot6d(6) + gripper(1)
      Arm2 [10:20]: zeros(10)

    Calvin obs sub-keys → ee6d 변환:
      eef_pos(3)   → 그대로
      eef_euler(3) → rot6d(6) 변환
      gripper      → binary (preprocess에서 0으로 마스킹됨)
    """
    eef_pos = payload.get("observation.state.eef_pos", [0.0, 0.0, 0.0])
    eef_pos = list(eef_pos)[:3]

    eef_euler = payload.get("observation.state.eef_euler", [0.0, 0.0, 0.0])
    eef_euler = np.array(eef_euler, dtype=np.float32)[:3]
    eef_rot6d = _euler_to_rot6d(eef_euler)

    gripper_val = payload.get("observation.state.gripper_opening", [0.0])
    if hasattr(gripper_val, '__len__'):
        gripper_val = float(gripper_val[0])
    else:
        gripper_val = float(gripper_val)

    arm1 = eef_pos + eef_rot6d.tolist() + [gripper_val]
    arm2 = [0.0] * 10
    proprio = arm1 + arm2  # 20D

    return torch.tensor([proprio], dtype=torch.float32, device=_device)


def _tokenize(instruction: str) -> torch.Tensor:
    """Instruction string → input_ids [1, max_len]."""
    tokens = _tokenizer(
        instruction,
        max_length=50,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokens["input_ids"].to(_device)


# ─── 모델 로딩 ───────────────────────────────────────────────────────────────


@app.on_event("startup")
def load_model():
    global model, _tokenizer, _n_action_steps, _denoising_steps, _device

    args = app.state.args
    _device = args.device

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading X-VLA from {args.pretrained_path} ...")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModel.from_pretrained(
        args.pretrained_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(_device).eval()

    _tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    _n_action_steps = args.n_action_steps
    _denoising_steps = args.denoising_steps

    num_actions = getattr(model.config, "num_actions", _n_action_steps)
    print(
        f"X-VLA loaded.  num_actions(chunk)={num_actions}  "
        f"n_action_steps={_n_action_steps}  denoising_steps={_denoising_steps}  "
        f"action_mode={model.config.action_mode}  dtype={dtype}  "
        f"domain_id={_domain_id}  image_size={_image_size}"
    )


# ─── FastAPI 엔드포인트 ──────────────────────────────────────────────────────


@app.post("/reset")
async def reset():
    """X-VLA는 히스토리 없음. no-op."""
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    """통일 API: 이미지 + state + instruction → action sub-keys."""
    if model is None:
        return {"error": "model not loaded"}

    t0 = time.time()

    # 1) 이미지: base64 → PIL → 224×224 resize → ImageNet normalize
    images = []
    for k in sorted(k for k in payload if k.startswith("observation.images.")):
        images.append(_b64_to_pil(payload[k]))
    if not images:
        return {"error": "no images in payload"}

    image_input, image_mask = _preprocess_images(images)

    # 2) Instruction tokenize
    input_ids = _tokenize(payload.get("task", ""))

    # 3) Proprio (Calvin obs → ee6d 20D)
    proprio = _assemble_proprio(payload)

    # 4) Domain ID
    domain_id = torch.tensor([_domain_id], dtype=torch.long, device=_device)

    # 5) Action 생성
    cast_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        actions = model.generate_actions(
            input_ids=input_ids,
            image_input=image_input.to(dtype=cast_dtype),
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio.to(dtype=cast_dtype),
            steps=_denoising_steps,
        )
    # actions: [1, num_actions, 20] — ee6d postprocessed (gripper에 sigmoid 적용됨)

    # 6) Arm1만 추출하여 sub-key로 분리 (native format 그대로)
    arm1 = actions[0, :_n_action_steps, :10].cpu().float().numpy()  # [N, 10]

    latency_ms = (time.time() - t0) * 1000

    # 디버그 로그 (첫 5 호출)
    step_count = getattr(app.state, "_step", 0) + 1
    app.state._step = step_count
    if step_count <= 5:
        print(
            f"[DEBUG] call={step_count}  "
            f"n_actions={arm1.shape[0]}  "
            f"pos={arm1[0, :3].round(4).tolist()}  "
            f"grip={arm1[0, 9]:.4f}  "
            f"task='{payload.get('task', '')[:40]}'"
        )

    return {
        "action.eef_pos": arm1[:, :3].tolist(),       # [N, 3] absolute position
        "action.eef_rot6d": arm1[:, 3:9].tolist(),    # [N, 6] absolute rotation 6D
        "action.gripper": arm1[:, 9:10].tolist(),      # [N, 1] sigmoid [0,1]
        "latency_ms": latency_ms,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "not_loaded",
        "model": "xvla",
        "n_action_steps": _n_action_steps,
        "action_type": "absolute",
        "action_keys": ["action.eef_pos", "action.eef_rot6d", "action.gripper"],
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="X-VLA 추론 서버 (통일 API, port 8100)")
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="2toINF/X-VLA-Calvin-ABC_D",
        help="HuggingFace repo ID 또는 로컬 체크포인트 경로",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=30,
        help="예측마다 반환할 action 수 (기본: 30, 체크포인트 chunk_size와 동일)",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=10,
        help="Flow matching denoising steps (기본: 10)",
    )
    args = parser.parse_args()

    app.state.args = args
    app.state._step = 0
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
