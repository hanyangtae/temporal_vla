"""
UP-VLA 추론 서버 (통일 API, port 8300).

upvla 컨테이너에서 실행:
  docker compose run --rm upvla \
    python /temporal_vla/scripts/serve/upvla.py \
    --model-config /temporal_vla/src/policies/UP-VLA/policy_rollout/upvla_model.yaml

통일 API (LeRobot 컨벤션):
  POST /act     ← {"observation.images.static": b64png, "observation.images.wrist": b64png,
                    "observation.state": [...], "task": "..."}
                → {"action": [[7 floats] x act_step], "latency_ms": float}
  POST /reset   ← no-op (히스토리 없음)
  GET  /health  ← feature 정보 포함
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

app = FastAPI(title="UP-VLA Inference Server")

_model = None
_vq_model = None
_uni_prompting = None
_config = None
_device = None


def _b64_to_pil(b64_str: str) -> Image.Image:
    """base64 PNG 문자열 → PIL Image (RGB)."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


@app.on_event("startup")
def load_model():
    global _model, _vq_model, _uni_prompting, _config, _device

    upvla_root = Path(app.state.upvla_root)

    # UP-VLA 소스를 sys.path에 추가하고 작업 디렉토리 변경
    # (upvla_model.yaml의 상대 경로 ./showlab/... 가 upvla_root 기준으로 해석됨)
    sys.path.insert(0, str(upvla_root))
    os.chdir(upvla_root)

    from llava.llava import conversation as conversation_lib
    from models import MAGVITv2, Upvla
    from omegaconf import OmegaConf
    from training.prompting_utils import UniversalPrompting_w_action
    from transformers import AutoTokenizer

    conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

    config = OmegaConf.load(app.state.model_config_path)
    _config = config
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer: {config.model.showo.llm_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path, padding_side="left"
    )
    _uni_prompting = UniversalPrompting_w_action(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        future_steps=config.act_step,
    )

    print(f"Loading MAGVITv2: {config.model.vq_model.vq_model_name}")
    _vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(
        _device
    )
    _vq_model.requires_grad_(False)
    _vq_model.eval()

    print(f"Loading Upvla backbone: {config.model.showo.pretrained_model_path}")
    _model = Upvla.from_pretrained(
        config.model.showo.pretrained_model_path,
        low_cpu_mem_usage=False,
        act_step=config.act_step,
    ).to(_device)

    ckpt_path = (
        f"{config.model.showo.tuned_model_path}/unwrapped_model/pytorch_model.bin"
    )
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    _model.load_state_dict(state_dict, strict=True)
    del state_dict
    _model.eval()

    print(
        f"UP-VLA ready  act_step={config.act_step}  "
        f"resolution={config.dataset.preprocessing.resolution}  "
        f"num_view={config.model.vla.num_view}  device={_device}"
    )


@app.post("/act")
async def predict_action(payload: dict):
    """
    통일 API (LeRobot 컨벤션):
      요청: {"observation.images.static": b64png, "observation.images.wrist": b64png,
             "observation.state": [...], "task": "..."}
      응답: {"action": [[7 floats] x act_step], "latency_ms": float}
    """
    from training.prompting_utils import (
        create_attention_mask_predict_next_for_future_prediction,
    )
    from training.utils import image_transform

    t0 = time.time()
    config = _config
    resolution = config.dataset.preprocessing.resolution

    # static view 처리
    static_b64 = payload.get("observation.images.static")
    img_static = _b64_to_pil(static_b64)
    pv_static = (
        image_transform(img_static, resolution=resolution).to(_device).unsqueeze(0)
    )
    image_tokens = _vq_model.get_code(pv_static) + len(_uni_prompting.text_tokenizer)

    # wrist view 처리 (num_view=2 && 클라이언트가 전송한 경우)
    wrist_b64 = payload.get("observation.images.wrist")
    if config.model.vla.num_view == 2 and wrist_b64:
        img_wrist = _b64_to_pil(wrist_b64)
        pv_wrist = (
            image_transform(img_wrist, resolution=resolution).to(_device).unsqueeze(0)
        )
        tokens_wrist = _vq_model.get_code(pv_wrist) + len(_uni_prompting.text_tokenizer)
        image_tokens = torch.cat([image_tokens, tokens_wrist], dim=1)

    instruction = payload.get("task", "")
    input_ids, _ = _uni_prompting(([instruction], image_tokens), "pre_gen")
    attention_mask = create_attention_mask_predict_next_for_future_prediction(
        input_ids,
        pad_id=int(_uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(_uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(_uni_prompting.sptids_dict["<|eoi|>"]),
        rm_pad_in_image=True,
    )

    with torch.no_grad():
        gen_token_ids, actions = _model.pre_pad_predict(
            input_ids=input_ids,
            uncond_input_ids=None,
            attention_mask=attention_mask,
            guidance_scale=None,
            temperature=None,
            timesteps=None,
            noise_schedule=None,
            noise_type=None,
            predict_all_tokens=None,
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=_uni_prompting,
            config=config,
            return_actions=True,
        )

    # actions shape: [1, act_step, 7] 또는 [act_step, 7]
    actions_np = actions.squeeze().detach().cpu().numpy()  # [act_step, 7]
    if actions_np.ndim == 1:
        actions_np = actions_np[np.newaxis, :]
    latency_ms = (time.time() - t0) * 1000

    return {
        "action.eef_pos": actions_np[:, :3].tolist(),
        "action.eef_euler": actions_np[:, 3:6].tolist(),
        "action.gripper": actions_np[:, 6:7].tolist(),
        "latency_ms": latency_ms,
    }


@app.post("/reset")
async def reset():
    """통일 API: no-op (UP-VLA는 히스토리 없음)."""
    return {"status": "reset"}


@app.get("/health")
async def health():
    act_step = int(_config.act_step) if _config is not None else None
    return {
        "status": "ok" if _model is not None else "not_loaded",
        "model": "upvla",
        "n_action_steps": act_step,
        "action_type": "relative",
        "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
    }


def main():
    parser = argparse.ArgumentParser(description="UP-VLA 추론 서버 (port 8300)")
    parser.add_argument(
        "--model-config",
        type=str,
        default="/temporal_vla/src/policies/UP-VLA/policy_rollout/upvla_model.yaml",
        help="upvla_model.yaml 경로",
    )
    parser.add_argument(
        "--upvla-root",
        type=str,
        default="/temporal_vla/src/policies/UP-VLA",
        help="UP-VLA 소스 루트 (상대 경로 resolve 기준 및 sys.path 추가)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    app.state.model_config_path = args.model_config
    app.state.upvla_root = args.upvla_root
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
