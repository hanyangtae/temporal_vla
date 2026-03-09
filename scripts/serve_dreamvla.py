"""
DreamVLA 추론 서버.

dreamvla 컨테이너 내에서 실행:
  docker compose --profile dreamvla run --rm dreamvla python /temporal_vla/scripts/serve_dreamvla.py

robocasa 컨테이너(또는 벤치마크 스크립트)에서 HTTP로 액션 요청:
  POST http://localhost:8200/act
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="DreamVLA Inference Server")

model = None


@app.on_event("startup")
def load_model():
    global model
    model_path = app.state.model_path
    print(f"Loading DreamVLA model from {model_path}...")

    # DreamVLA uses custom loading; adjust import path once repo is cloned
    sys.path.insert(0, "/temporal_vla/dreamvla")
    try:
        from models.dreamvla import DreamVLA as DreamVLAModel
        model = DreamVLAModel.from_pretrained(model_path)
        model = model.cuda().eval()
    except ImportError:
        print(
            "WARNING: DreamVLA model code not found. "
            "Clone the repo first: git clone https://github.com/Zhangwenyao1/DreamVLA /temporal_vla/dreamvla"
        )
        model = None
    print("DreamVLA model loaded." if model else "DreamVLA model NOT loaded (stub mode).")


@app.post("/act")
async def predict_action(payload: dict):
    """
    Payload keys:
      - images: list of HxWx3 uint8 arrays
      - proprio: list[float]  (proprioceptive state)
      - language_instruction: str
    """
    if model is None:
        return {"action": [[0.0] * 7], "error": "model not loaded"}

    images = [np.array(img, dtype=np.uint8) for img in payload.get("images", [])]
    proprio = np.array(payload.get("proprio", []), dtype=np.float32)
    instruction = payload.get("language_instruction", "")

    # DreamVLA forward: will be adapted once exact API is confirmed from the repo
    with torch.inference_mode():
        action = model.predict(
            images=images,
            proprio=proprio,
            instruction=instruction,
        )

    actions = np.array(action, dtype=np.float32)
    return {"action": actions.tolist()}


@app.get("/health")
async def health():
    loaded = model is not None
    return {"status": "ok" if loaded else "stub", "model": "dreamvla"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="WenyaoZhang/DreamVLA",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    app.state.model_path = args.model_path
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
