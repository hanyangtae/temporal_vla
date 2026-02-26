"""
X-VLA 추론 서버 (LeRobot 내장 버전).

xvla 컨테이너 내에서 실행:
  docker compose --profile xvla run --rm xvla \
    python /temporal_vla/scripts/serve_xvla.py --model-path lerobot/xvla-base

robocasa 컨테이너에서 HTTP로 액션 요청:
  POST http://localhost:8100/act
"""

import argparse
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

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
    Payload:
      - images: dict of camera_name -> HxWx3 uint8 list
      - state: list[float] (proprioceptive state)
      - language_instruction: str
    Returns:
      - action: list[list[float]]
      - latency_ms: float
    """
    t0 = time.time()

    observation = {}
    for cam_name, img_data in payload.get("images", {}).items():
        img = np.array(img_data, dtype=np.uint8)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        observation[f"observation.images.{cam_name}"] = img_tensor.cuda()

    if "state" in payload:
        state = torch.tensor(payload["state"], dtype=torch.float32).unsqueeze(0).cuda()
        observation["observation.state"] = state

    if "language_instruction" in payload:
        observation["task"] = payload["language_instruction"]

    with torch.inference_mode():
        action = policy.select_action(observation)

    action_np = action.cpu().numpy()
    latency_ms = (time.time() - t0) * 1000

    return {
        "action": action_np.tolist(),
        "latency_ms": latency_ms,
    }


@app.get("/health")
async def health():
    return {"status": "ok" if policy is not None else "not_loaded", "model": "xvla"}


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
