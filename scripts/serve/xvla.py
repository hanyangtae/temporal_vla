"""
X-VLA 추론 서버 (통일 API, port 8100).

프로파일 기반 동작: 체크포인트별 domain_id, denoising_steps, tokenizer,
image normalization, proprio layout 등은 configs/checkpoints/*.yaml 에 선언.

xvla 컨테이너에서 실행:
  docker compose run --rm xvla \
    python /temporal_vla/scripts/serve/xvla.py \
    --profile /temporal_vla/configs/checkpoints/xvla__calvin_abc_d.yaml

통일 API:
  POST /act     ← {"observation.images.*": b64png, "observation.state.*": [...],
                    "task": "..."}
                → 프로파일 emits_subkeys 규약에 따라 sub-key dict 반환
  POST /reset   ← no-op (X-VLA 는 히스토리 없음)
  GET  /health  ← 프로파일 기반 서버 상태
"""

import argparse
import logging
import math
import sys
import time
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, "/temporal_vla/lerobot/src")

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

from src.utils.common.image import decode_b64_pil  # noqa: E402
from src.utils.common.serving import health_response  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="X-VLA Inference Server")

# ─── 글로벌 ──────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_profile: Optional[CheckpointProfile] = None
_device = "cuda"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── 유틸 ────────────────────────────────────────────────────────────────────


def _euler_to_rot6d(euler: np.ndarray) -> np.ndarray:
    """Euler angles (roll, pitch, yaw) → 6D rotation representation.

    Calvin euler 순서: roll(x), pitch(y), yaw(z).
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    X-VLA rot6d 포맷: R[:, :2].reshape(6) = [R00, R01, R10, R11, R20, R21]
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

    return R[:, :2].reshape(6)


# ─── 전처리 ──────────────────────────────────────────────────────────────────


def _preprocess_images(pil_images: list) -> tuple:
    """PIL images → (image_input [1, V, C, H, W], image_mask [1, V]).

    프로파일의 resolution 으로 resize 후 정규화. max_views 까지 zero padding.
    """
    profile = _profile
    assert profile is not None
    resolution = profile.image_preprocess.resolution
    max_views = int(profile.model_specific.get("max_views", 3))
    norm_scheme = profile.model_specific.get("image_normalization", "imagenet")

    if norm_scheme == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    else:
        raise NotImplementedError(f"image_normalization={norm_scheme!r} not supported")

    tensors = []
    for img in pil_images:
        img = img.resize((resolution, resolution), Image.BILINEAR)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        t = (t - mean) / std
        tensors.append(t)
    masks = [True] * len(tensors)

    while len(tensors) < max_views:
        tensors.append(torch.zeros_like(tensors[0]))
        masks.append(False)

    image_input = torch.stack(tensors).unsqueeze(0).to(_device)
    image_mask = torch.tensor([masks], dtype=torch.bool, device=_device)
    return image_input, image_mask


def _assemble_proprio(payload: dict) -> torch.Tensor:
    """payload obs → proprio tensor.

    현재 지원 레이아웃: `ee6d_dual_arm`
      Arm1 [0:10]: eef_pos(3) + eef_rot6d(6) + gripper(1)
      Arm2 [10:20]: zeros (single-arm 체크포인트용 padding)
    """
    profile = _profile
    assert profile is not None
    layout = profile.model_specific.get("proprio_layout", "ee6d_dual_arm")

    if layout != "ee6d_dual_arm":
        raise NotImplementedError(f"proprio_layout={layout!r} not supported")

    eef_pos = list(payload.get("observation.state.eef_pos", [0.0, 0.0, 0.0]))[:3]
    eef_euler = np.array(
        payload.get("observation.state.eef_euler", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    )[:3]
    eef_rot6d = _euler_to_rot6d(eef_euler)

    gripper_val = payload.get("observation.state.gripper_opening", [0.0])
    if hasattr(gripper_val, "__len__"):
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
    try:
        _load_model_impl()
    except Exception:
        import traceback
        sys.stderr.write("=== load_model FAILED ===\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _load_model_impl():
    global _model, _tokenizer, _device

    args = app.state.args
    profile = _profile
    assert profile is not None, "profile must be set before load_model"

    _device = args.device
    from transformers import AutoModel, AutoTokenizer

    checkpoint_id = profile.checkpoint_source.id
    logger.info("Loading X-VLA from %s (profile=%s)", checkpoint_id, profile.name)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    _model = AutoModel.from_pretrained(
        checkpoint_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    _model = _model.to(_device).eval()

    tokenizer_id = profile.model_specific.get("tokenizer", "facebook/bart-large")
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    if profile.lora is not None:
        from peft import PeftModel

        src = profile.checkpoint_source
        if src.type == "hf_repo":
            adapter_arg = src.id
            kw = {"subfolder": profile.lora.subfolder}
        else:
            import os as _os
            adapter_arg = _os.path.join(src.id, profile.lora.subfolder)
            kw = {}
        logger.info("Applying LoRA adapter subfolder=%s", profile.lora.subfolder)
        _model = PeftModel.from_pretrained(_model, adapter_arg, **kw).merge_and_unload()

    num_actions = getattr(_model.config, "num_actions", profile.n_action_steps)
    logger.info(
        "X-VLA loaded. num_actions(chunk)=%s, n_action_steps=%d, "
        "denoising_steps=%s, domain_id=%s, tokenizer=%s",
        num_actions,
        profile.n_action_steps,
        profile.model_specific.get("denoising_steps"),
        profile.model_specific.get("domain_id"),
        tokenizer_id,
    )


# ─── FastAPI 엔드포인트 ──────────────────────────────────────────────────────


@app.post("/reset")
async def reset():
    """X-VLA 는 히스토리 없음. no-op."""
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    """통일 API: 이미지 + state + instruction → action sub-keys."""
    if _model is None:
        return {"error": "model not loaded"}

    t0 = time.time()
    profile = _profile
    assert profile is not None

    # 1) 이미지 — 프로파일 images 순서대로 로드
    pil_images = []
    for view in profile.observation_requirements.images:
        b64 = payload.get(f"observation.images.{view}")
        if b64 is not None:
            pil_images.append(decode_b64_pil(b64))
    if not pil_images:
        return {"error": "no images in payload"}
    image_input, image_mask = _preprocess_images(pil_images)

    # 2) Instruction tokenize
    input_ids = _tokenize(payload.get("task", ""))

    # 3) Proprio
    proprio = _assemble_proprio(payload)

    # 4) Domain ID (프로파일 기반)
    domain_id_val = int(profile.model_specific.get("domain_id", 0))
    domain_id = torch.tensor([domain_id_val], dtype=torch.long, device=_device)

    # 5) Flow matching denoising
    denoising_steps = int(profile.model_specific.get("denoising_steps", 10))
    cast_dtype = next(_model.parameters()).dtype
    with torch.inference_mode():
        actions = _model.generate_actions(
            input_ids=input_ids,
            image_input=image_input.to(dtype=cast_dtype),
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio.to(dtype=cast_dtype),
            steps=denoising_steps,
        )
    # actions: [1, num_actions, 20] dual-arm

    # 6) 프로파일 action_dim 만큼 앞에서 추출 (arm1 10D = Calvin single-arm)
    n_steps = profile.n_action_steps
    dim = profile.action_dim
    arm1 = actions[0, :n_steps, :dim].cpu().float().numpy()  # [N, dim]

    latency_ms = (time.time() - t0) * 1000

    # 7) Sub-key 분리 (layout 기반)
    out = {}
    for sk in profile.emits_subkeys:
        local_name = sk[len("action."):]  # ex: "eef_pos"
        if local_name in {a.name for a in profile.action_layout}:
            sl = profile.dim_slice(local_name)
            out[sk] = arm1[:, sl].tolist()
        else:
            raise ValueError(
                f"emit sub-key {sk} has no matching action_layout entry"
            )

    # 디버그 로그 (첫 5 호출)
    step_count = getattr(app.state, "_step", 0) + 1
    app.state._step = step_count
    if step_count <= 5:
        logger.info(
            "call=%d n_actions=%d pos=%s grip=%.4f task=%r",
            step_count, arm1.shape[0],
            arm1[0, profile.dim_slice("eef_pos")].round(4).tolist(),
            float(arm1[0, profile.dim_slice("gripper")][0]),
            payload.get("task", "")[:40],
        )

    out["latency_ms"] = latency_ms
    return out


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "xvla"}
    return health_response(
        policy=_model,
        model="xvla",
        profile=_profile,
        n_action_steps=_profile.n_action_steps,
        action_type=_profile.action_type,
        action_keys=list(_profile.emits_subkeys),
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    global _profile

    try:
        from src.utils.common.logger import create_module_logger

        create_module_logger("xvla_serve")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="X-VLA 추론 서버 (port 8100)")
    parser.add_argument(
        "--profile", type=str, required=True,
        help="체크포인트 프로파일 YAML 경로 (configs/checkpoints/*.yaml)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"],
    )
    args = parser.parse_args()

    _profile = load_profile(args.profile)
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "xvla", (
        f"profile.base_model={_profile.base_model!r}, but this server is xvla"
    )

    app.state.args = args
    app.state._step = 0
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
