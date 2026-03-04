"""
UP-VLA 추론 서버.

upvla 컨테이너 내에서 실행:
  docker compose run --rm upvla python /temporal_vla/scripts/serve_upvla.py

robocasa 컨테이너(또는 벤치마크 스크립트)에서 HTTP로 액션 요청:
  POST http://localhost:8300/act

모델 설정은 --config-path 로 지정한 OmegaConf yaml 파일을 통해 관리.
기본값: /temporal_vla/upvla/policy_rollout/upvla_model.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="UP-VLA Inference Server")

# (model_config, model, uni_prompting, vq_model, mask_token_id)
agent = None


@app.on_event("startup")
def load_model():
    global agent

    config_path = app.state.config_path
    device_id = app.state.device_id

    print(f"Loading UP-VLA from config: {config_path} on cuda:{device_id} ...")

    sys.path.insert(0, "/temporal_vla/upvla")
    try:
        from omegaconf import OmegaConf
        from models import Upvla, MAGVITv2
        from training.prompting_utils import (
            UniversalPrompting_w_action,
            create_attention_mask_predict_next_for_future_prediction,
        )
        from training.utils import get_config, image_transform
        from transformers import AutoTokenizer

        model_config = OmegaConf.load(config_path)
        config = model_config
        device = torch.device(f"cuda:{device_id}")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.showo.llm_model_path, padding_side="left"
        )
        uni_prompting = UniversalPrompting_w_action(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob,
        )

        vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()

        model = Upvla.from_pretrained(
            config.model.showo.pretrained_model_path,
            low_cpu_mem_usage=False,
            act_step=config.act_step,
        ).to(device)

        tuned_path = config.model.showo.get("tuned_model_path", None)
        if tuned_path:
            ckpt = Path(tuned_path) / "unwrapped_model" / "pytorch_model.bin"
            print(f"Loading checkpoint: {ckpt}")
            state_dict = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            del state_dict

        model.eval()
        mask_token_id = model.config.mask_token_id

        app.state.image_transform = image_transform
        app.state.create_attn_mask = create_attention_mask_predict_next_for_future_prediction

        agent = (model_config, model, uni_prompting, vq_model, mask_token_id)
        print("UP-VLA model loaded.")

    except ImportError as e:
        print(f"WARNING: Failed to import UP-VLA modules: {e}")
        print("Make sure /temporal_vla/upvla is present and dependencies are installed.")
        agent = None


@app.post("/act")
async def predict_action(payload: dict):
    """
    Payload keys:
      - image: HxWx3 uint8 list (third-person camera)
      - image_gripper: HxWx3 uint8 list (optional, wrist camera)
      - language_instruction: str
    Returns:
      - action: list of [act_step, action_dim] float32
    """
    if agent is None:
        return {"action": [[0.0] * 7], "error": "model not loaded"}

    model_config, model, uni_prompting, vq_model, mask_token_id = agent
    image_transform = app.state.image_transform
    create_attn_mask = app.state.create_attn_mask

    from PIL import Image as PILImage

    device = next(model.parameters()).device
    resolution = model_config.dataset.preprocessing.resolution
    num_view = model_config.model.vla.num_view

    image_np = np.array(payload.get("image"), dtype=np.uint8)
    instruction = payload.get("language_instruction", "")

    pixel_values = image_transform(
        PILImage.fromarray(image_np), resolution=resolution
    ).to(device).unsqueeze(0)

    with torch.no_grad():
        image_tokens = vq_model.get_code(pixel_values) + len(uni_prompting.text_tokenizer)

        if num_view == 2 and payload.get("image_gripper") is not None:
            image_gripper_np = np.array(payload["image_gripper"], dtype=np.uint8)
            pixel_values_gripper = image_transform(
                PILImage.fromarray(image_gripper_np), resolution=resolution
            ).to(device).unsqueeze(0)
            image_tokens_gripper = vq_model.get_code(pixel_values_gripper) + len(uni_prompting.text_tokenizer)
            image_tokens = torch.cat([image_tokens, image_tokens_gripper], dim=1)

        input_ids, _ = uni_prompting(([instruction], image_tokens), "pre_gen")
        attention_mask = create_attn_mask(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        )

        _, actions = model.pre_pad_predict(
            input_ids=input_ids,
            uncond_input_ids=None,
            attention_mask=attention_mask,
            guidance_scale=None,
            temperature=None,
            timesteps=None,
            noise_schedule=None,
            noise_type=None,
            predict_all_tokens=None,
            seq_len=model_config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=model_config,
            return_actions=True,
        )

    # actions: [1, act_step, action_dim] → [act_step, action_dim]
    actions = actions.squeeze().cpu().numpy().astype(np.float32)
    return {"action": actions.tolist()}


@app.get("/health")
async def health():
    loaded = agent is not None
    return {"status": "ok" if loaded else "stub", "model": "upvla"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="/temporal_vla/upvla/policy_rollout/upvla_model.yaml",
        help="OmegaConf yaml config (upvla_model.yaml)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    app.state.config_path = args.config_path
    app.state.device_id = args.device
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
