"""
LeRobot policy 추론 서버 (통일 API).

lerobot 컨테이너 내에서 실행:
  docker compose run --rm lerobot \
    python /temporal_vla/scripts/serve_lerobot.py \
    --policy-type pi0 \
    --pretrained-path lerobot/pi0_base \
    --port 8400

  wrist 카메라 포함 시:
    --is-wrist 추가 (static 1장 + gripper 1장 → policy input_features 순서대로 매핑)

통일 API:
  POST /act     ← {"observation.images.static": b64png, ...,
                    "observation.state.eef_pos": [...], ...,
                    "task": "..."}
                → {"action": [[7 floats], ...], "latency_ms": float}
  POST /reset   ← 에피소드 시작 시 policy 히스토리 초기화
  GET  /health  ← 서버 상태 + 모델 정보
"""

import argparse
import base64
import io
import sys
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

app = FastAPI(title="LeRobot Inference Server")

# 모듈 레벨 글로벌: startup 시 또는 테스트에서 직접 주입
policy = None
preprocessor = None
postprocessor = None
_policy_type = "unknown"
_n_action_steps = 1
_camera_key_map: dict = (
    {}
)  # 통일 키 → policy 키 (e.g. observation.images.static → observation.images.top)
_state_dim: int = 0  # >0이면 observation.state를 앞 N차원으로 슬라이싱

# Calvin 기준 고정 state sub-key 순서 (eef_pos(3)+eef_euler(3)+gripper_opening(1)+joint_pos(7)+gripper_action(1)=15D)
STATE_KEY_ORDER = [
    "observation.state.eef_pos",
    "observation.state.eef_euler",
    "observation.state.gripper_opening",
    "observation.state.joint_pos",
    "observation.state.gripper_action",
]


# ─── 변환 유틸 ────────────────────────────────────────────────────────────────


def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """base64 PNG → HxWx3 uint8 numpy."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))


def _build_remap_config(visual_keys: list, state_feat, is_wrist: bool) -> tuple:
    """policy input_features에서 카메라 키맵과 state dim을 도출.

    Args:
        visual_keys: policy.config.input_features에서 VISUAL 타입 키 목록 (순서 보존).
        state_feat:  policy.config.robot_state_feature (없으면 None).
        is_wrist:    wrist 카메라 사용 여부.

    Returns:
        (camera_key_map, state_dim)
        - camera_key_map: 통일 API 키 → policy 키 dict (src==dst인 항목은 제거)
        - state_dim: policy가 기대하는 state 차원 (0이면 슬라이싱 없음)
    """
    if is_wrist:
        raw = {
            "observation.images.static": visual_keys[0],
            "observation.images.gripper": visual_keys[1],
        }
    else:
        raw = {"observation.images.static": visual_keys[0]}
    camera_key_map = {s: d for s, d in raw.items() if s != d}
    state_dim = state_feat.shape[0] if state_feat is not None else 0
    return camera_key_map, state_dim


def _apply_input_remap(batch: dict) -> dict:
    """통일 API batch → policy input_features 키 형식으로 변환.

    1. 카메라 키 리맵핑: _camera_key_map에 따라 observation.images.* 키를 rename.
       e.g. observation.images.static → observation.images.top
    2. state 차원 슬라이싱: _state_dim > 0이면 observation.state[:, :_state_dim]으로 truncate.
       e.g. Calvin 15D → policy가 학습된 7D
    """
    if _camera_key_map:
        for src, dst in _camera_key_map.items():
            if src in batch:
                batch[dst] = batch.pop(src)
    if _state_dim > 0 and "observation.state" in batch:
        batch["observation.state"] = batch["observation.state"][:, :_state_dim]
    return batch


def parse_payload(payload: dict) -> dict:
    """HTTP JSON payload → LeRobot batch dict.

    변환 규칙:
      observation.images.*  : base64 PNG → [1, C, H, W] float32 tensor, [0.0, 1.0]
      observation.state.*   : STATE_KEY_ORDER 순서로 concatenate → [1, D] float32 tensor
                              STATE_KEY_ORDER에 없는 sub-key는 무시
      task                  : str 그대로 (없으면 "")
      그 외 키              : 무시
    """
    batch = {}

    # 이미지: base64 → CHW float tensor
    for k, v in payload.items():
        if k.startswith("observation.images."):
            np_img = _b64_to_numpy(v)
            t = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # CHW
            batch[k] = t.unsqueeze(0)  # [1, C, H, W]

    # state: STATE_KEY_ORDER 순서, 없는 키는 skip
    state_parts = []
    for key in STATE_KEY_ORDER:
        if key in payload:
            state_parts.append(np.array(payload[key], dtype=np.float32))
    if state_parts:
        batch["observation.state"] = torch.from_numpy(
            np.concatenate(state_parts)
        ).unsqueeze(
            0
        )  # [1, D]

    # instruction
    batch["task"] = payload.get("task", "")

    return batch


# ─── FastAPI 엔드포인트 ───────────────────────────────────────────────────────


@app.post("/reset")
async def reset():
    """에피소드 시작 시 policy 히스토리 초기화."""
    if policy is not None:
        policy.reset()
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    """
    통일 API:
      요청: {"observation.images.static": b64png, "observation.state.eef_pos": [...], ..., "task": "..."}
      응답: {"action": [[float...], ...], "latency_ms": float}  ← action 항상 2D
    """
    if policy is None:
        return {"error": "model not loaded"}

    t0 = time.time()

    batch = parse_payload(payload)
    batch = _apply_input_remap(batch)

    if preprocessor is not None:
        batch = preprocessor(batch)

    with torch.inference_mode():
        action = policy.select_action(batch)

    if postprocessor is not None:
        action = postprocessor(action)

    action_np = action.detach().cpu().float().numpy()
    if action_np.ndim == 1:
        action_np = action_np[np.newaxis, :]  # [action_dim] → [1, action_dim]

    return {
        "action": action_np.tolist(),
        "latency_ms": (time.time() - t0) * 1000,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if policy is not None else "not_loaded",
        "model": _policy_type,
        "n_action_steps": _n_action_steps,
    }


# ─── 모델 로딩 ────────────────────────────────────────────────────────────────


@app.on_event("startup")
def load_model():
    global policy, preprocessor, postprocessor, _policy_type, _n_action_steps, _camera_key_map, _state_dim

    args = getattr(app.state, "args", None)
    if args is None:
        return  # 테스트 환경: args 없으면 로딩 skip

    _policy_type = args.policy_type

    sys.path.insert(0, "/temporal_vla/lerobot/src")
    from lerobot.configs.types import FeatureType
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    if not args.pretrained_path:
        raise ValueError("--pretrained-path is required for inference server")

    # dataset_stats 로딩 (옵션: 없으면 체크포인트에서 자동 로드)
    dataset_stats = None
    if args.dataset_stats:
        import json

        with open(args.dataset_stats) as f:
            raw = json.load(f)
        dataset_stats = {
            k: {sk: torch.tensor(sv) for sk, sv in sv_dict.items()}
            for k, sv_dict in raw.items()
        }

    # make_policy는 ds_meta/env_cfg 필수라 추론 서버에서 사용 불가.
    # 예제(lerobot/examples/tutorial/pi0)와 동일하게 from_pretrained 직접 호출.
    policy_cls = get_policy_class(args.policy_type)
    policy = policy_cls.from_pretrained(args.pretrained_path)
    policy.config.device = args.device
    policy.to(args.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.pretrained_path,
        dataset_stats=dataset_stats,
    )
    _n_action_steps = getattr(policy.config, "n_action_steps", 1)

    # 카메라 키 리맵핑 + state dim: policy config에서 자동 도출
    visual_keys = [
        k
        for k, v in policy.config.input_features.items()
        if v.type == FeatureType.VISUAL
    ]
    _camera_key_map, _state_dim = _build_remap_config(
        visual_keys, getattr(policy.config, "robot_state_feature", None), args.is_wrist
    )

    print(
        f"LeRobot '{args.policy_type}' loaded from {args.pretrained_path} "
        f"(n_action_steps={_n_action_steps}, visual_keys={visual_keys}, state_dim={_state_dim})"
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LeRobot policy 추론 서버 (통일 API)")
    parser.add_argument(
        "--policy-type",
        type=str,
        required=True,
        help="LeRobot policy 이름 (e.g. pi0, groot, act, smolvla)",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        required=False,
        help="체크포인트 경로 또는 HuggingFace repo ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--dataset-stats",
        type=str,
        default=None,
        help="정규화 통계 JSON 경로 (없으면 체크포인트에서 로드)",
    )
    parser.add_argument(
        "--is-wrist",
        action="store_true",
        help="wrist 카메라 사용 여부. 지정하면 static+gripper 2장, 미지정이면 static 1장.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
