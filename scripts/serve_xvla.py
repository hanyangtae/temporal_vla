"""
X-VLA 추론 서버 (통일 API, LeRobot 내장 버전).

xvla 컨테이너 내에서 실행:
  docker compose --profile xvla run --rm xvla \
    python /temporal_vla/scripts/serve_xvla.py --model-path lerobot/xvla-base

통일 API (LeRobot 컨벤션):
  POST /act     ← {"observation.images.static": b64png, ..., "observation.state": [...], "task": "..."}
                → {"action": [[floats], ...], "latency_ms": float}
  POST /reset   ← no-op (히스토리 없음)
  GET  /health  ← feature 정보 포함
"""

import argparse
import base64
import io
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image


def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """base64 PNG → HxWx3 uint8 numpy."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))

app = FastAPI(title="X-VLA Inference Server")

policy = None
policy_cfg = None


@app.on_event("startup")
def load_model():
    global policy, policy_cfg
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import init_hydra_config

    model_path = app.state.model_path
    print(f"Loading X-VLA policy from {model_path}...")

    policy = make_policy(
        hydra_cfg=None,
        pretrained_name_or_path=model_path,
    )
    policy = policy.to("cuda").eval()
    print(f"X-VLA policy loaded. Parameters: {sum(p.numel() for p in policy.parameters()):,}")


@app.post("/act")
async def predict_action(payload: dict):
    """
    통일 API (LeRobot 컨벤션):
      요청: {"observation.images.static": b64png, ..., "observation.state": [...], "task": "..."}
      응답: {"action": [[floats], ...], "latency_ms": float}
    """
    t0 = time.time()

    observation = {}
    # observation.images.* 키에서 이미지 추출
    for key, value in payload.items():
        if key.startswith("observation.images."):
            img = _b64_to_numpy(value)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            observation[key] = img_tensor.cuda()

    if "observation.state" in payload:
        state = torch.tensor(payload["observation.state"], dtype=torch.float32).unsqueeze(0).cuda()
        observation["observation.state"] = state

    if "task" in payload:
        observation["task"] = payload["task"]

    with torch.inference_mode():
        action = policy.select_action(observation)

    action_np = action.cpu().numpy()
    if action_np.ndim == 1:
        action_np = action_np[np.newaxis, :]
    latency_ms = (time.time() - t0) * 1000

    return {
        "action": action_np.tolist(),
        "latency_ms": latency_ms,
    }


@app.post("/reset")
async def reset():
    """통일 API: no-op (X-VLA는 히스토리 없음)."""
    return {"status": "reset"}


@app.get("/health")
async def health():
    info = {
        "status": "ok" if policy is not None else "not_loaded",
        "model": "xvla",
    }
    if policy is not None and hasattr(policy, "config"):
        cfg = policy.config
        info["input_features"] = {
            k: {"type": v.type if hasattr(v, "type") else "UNKNOWN", "shape": list(v.shape) if hasattr(v, "shape") else []}
            for k, v in getattr(cfg, "input_features", {}).items()
        }
        info["output_features"] = {
            k: {"type": v.type if hasattr(v, "type") else "UNKNOWN", "shape": list(v.shape) if hasattr(v, "shape") else []}
            for k, v in getattr(cfg, "output_features", {}).items()
        }
        info["n_action_steps"] = getattr(cfg, "n_action_steps", 1)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="lerobot/xvla-base",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    app.state.model_path = args.model_path
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
