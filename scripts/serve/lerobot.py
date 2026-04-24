"""
LeRobot policy 추론 서버 (통일 API).

프로파일 기반 동작: 체크포인트별 policy_type, dataset_stats 경로, 외부
체크포인트 fallback config 등은 configs/checkpoints/*.yaml 에 선언.

lerobot 컨테이너에서 실행:
  docker compose run --rm lerobot \
    python /temporal_vla/scripts/serve/lerobot.py \
    --profile /temporal_vla/configs/checkpoints/lerobot_pi05__calvin_sft.yaml

카메라 키 매핑: policy input_features 의 visual key 순서대로 통일 키에 자동 매핑.
  1번째 → observation.images.static
  2번째 → observation.images.wrist
  3번째 → observation.images.wrist2

통일 API:
  POST /act     ← {"observation.images.static": b64png, ...,
                    "observation.state.eef_pos": [...], ..., "task": "..."}
                → 프로파일 emits_subkeys 규약에 따라 sub-key dict 반환
  POST /reset   ← 에피소드 시작 시 policy 히스토리 초기화
  GET  /health  ← 프로파일 기반 서버 상태 + 모델 정보
"""

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="LeRobot Inference Server")

# 모듈 레벨 글로벌
policy = None
preprocessor = None
postprocessor = None
_profile: Optional[CheckpointProfile] = None
_policy_type = "unknown"
_n_action_steps = 1
_action_dim: int = 7
_camera_key_map: dict = {}
_state_dim: int = 0

# payload 의 observation.state.* 서브키를 lerobot observation.state 로 합칠 때 사용할
# canonical 정렬 순서 (벤치 공통). 체크포인트가 학습된 state dim 만큼 앞에서 truncate.
STATE_KEY_ORDER = [
    "observation.state.eef_pos",
    "observation.state.eef_euler",
    "observation.state.eef_quat",
    "observation.state.gripper_opening",
    "observation.state.gripper_qpos",
    "observation.state.joint_pos",
    "observation.state.joint_vel",
    "observation.state.gripper_action",
    "observation.state.base_pos",
    "observation.state.base_quat",
]


# ─── 변환 유틸 ────────────────────────────────────────────────────────────────


def _b64_to_numpy(b64_str: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))


def _build_remap_config(visual_keys: list, state_feat) -> tuple:
    """policy input_features 에서 camera key map 과 state dim 도출."""
    unified_keys = [
        "observation.images.static",
        "observation.images.wrist",
        "observation.images.wrist2",
    ]
    raw = {
        unified_keys[i]: vk for i, vk in enumerate(visual_keys) if i < len(unified_keys)
    }
    camera_key_map = {s: d for s, d in raw.items() if s != d}
    state_dim = state_feat.shape[0] if state_feat is not None else 0
    return camera_key_map, state_dim


def _apply_input_remap(batch: dict) -> dict:
    """통일 API batch → policy input_features 키 형식 변환."""
    if _camera_key_map:
        for src, dst in _camera_key_map.items():
            if src in batch:
                batch[dst] = batch.pop(src)
    if _state_dim > 0 and "observation.state" in batch:
        batch["observation.state"] = batch["observation.state"][:, :_state_dim]
    return batch


def parse_payload(payload: dict) -> dict:
    """HTTP JSON payload → LeRobot batch dict."""
    batch = {}

    for k, v in payload.items():
        if k.startswith("observation.images."):
            np_img = _b64_to_numpy(v)
            t = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
            batch[k] = t.unsqueeze(0)  # [1, C, H, W]

    state_parts = []
    for key in STATE_KEY_ORDER:
        if key in payload:
            state_parts.append(np.array(payload[key], dtype=np.float32))
    if state_parts:
        batch["observation.state"] = torch.from_numpy(
            np.concatenate(state_parts)
        ).unsqueeze(0)

    batch["task"] = payload.get("task", "")
    return batch


# ─── FastAPI 엔드포인트 ───────────────────────────────────────────────────────


@app.post("/reset")
async def reset():
    if policy is not None:
        policy.reset()
    return {"status": "reset"}


def _emit_subkeys(action_np: np.ndarray, profile: CheckpointProfile) -> dict:
    """프로파일 action_layout 에 따라 raw action 벡터를 sub-key dict 로 분리."""
    out = {}
    for sk in profile.emits_subkeys:
        local = sk[len("action."):]
        names = {a.name for a in profile.action_layout}
        if local in names:
            sl = profile.dim_slice(local)
            out[sk] = action_np[:, sl].tolist()
        else:
            raise ValueError(f"emit sub-key {sk} has no matching action_layout entry")
    return out


@app.post("/act")
async def predict_action(payload: dict):
    """통일 API: observation → action sub-keys."""
    if policy is None:
        return {"error": "model not loaded"}

    t0 = time.time()
    profile = _profile
    assert profile is not None

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
        action_np = action_np[np.newaxis, :]

    result = _emit_subkeys(action_np, profile)
    result["latency_ms"] = (time.time() - t0) * 1000
    return result


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "lerobot"}
    return {
        "status": "ok" if policy is not None else "not_loaded",
        "model": _policy_type,
        "profile": _profile.name,
        "n_action_steps": _n_action_steps,
        "action_type": _profile.action_type,
        "action_keys": list(_profile.emits_subkeys),
    }


# ─── 모델 로딩 ────────────────────────────────────────────────────────────────


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
    global policy, preprocessor, postprocessor, _policy_type, _n_action_steps
    global _action_dim, _camera_key_map, _state_dim

    args = getattr(app.state, "args", None)
    if args is None:
        return  # 테스트 환경: args 없으면 로딩 skip

    profile = _profile
    assert profile is not None

    ms = profile.model_specific
    _policy_type = ms.get("policy_type", "pi0")

    sys.path.insert(0, "/temporal_vla/lerobot/src")

    # 체크포인트 경로 결정
    src = profile.checkpoint_source
    pretrained_path = src.id  # local path or HF repo id
    logger.info(
        "Loading LeRobot policy_type=%s from %s (profile=%s)",
        _policy_type, pretrained_path, profile.name,
    )

    # dataset_stats 로딩 (프로파일에 명시된 경우)
    dataset_stats = None
    stats_path = ms.get("dataset_stats_path")
    if stats_path:
        import json
        with open(stats_path) as f:
            raw = json.load(f)
        dataset_stats = {
            k: {sk: torch.tensor(sv) for sk, sv in sv_dict.items()}
            for k, sv_dict in raw.items()
        }

    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_cls = get_policy_class(_policy_type)

    # 외부 체크포인트 분기: lerobot config.json 없는 경우
    ckpt_dir = Path(pretrained_path)
    has_lerobot_config = (
        (ckpt_dir / "config.json").exists() or not ckpt_dir.is_dir()
    )
    if not has_lerobot_config and (ckpt_dir / "model.safetensors").exists():
        logger.info("lerobot config 없음 — 외부 체크포인트 수동 로딩: %s", ckpt_dir)
        from safetensors.torch import load_file
        from lerobot.configs.policies import PolicyFeature
        from lerobot.configs.types import FeatureType

        ext = ms.get("external_config", {})
        image_shape = list(ext.get("image_shape", [3, 200, 200]))
        state_shape = list(ext.get("state_shape", [7]))
        action_shape = list(ext.get("action_shape", [7]))
        camera_key = ext.get("camera_key", "observation.images.top")

        cfg = policy_cls.config_class()
        cfg.input_features = {
            camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
        }
        cfg.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape),
        }

        # dataset_stats 가 아직 없으면 norm_stats.json 재귀 검색
        if dataset_stats is None:
            norm_paths = list(ckpt_dir.rglob("norm_stats.json"))
            if norm_paths:
                import json as _json
                with open(norm_paths[0]) as _f:
                    raw_norm = _json.load(_f)
                if "norm_stats" in raw_norm:
                    raw_norm = raw_norm["norm_stats"]
                dataset_stats = {}
                if "state" in raw_norm:
                    dataset_stats["observation.state"] = {
                        sk: torch.tensor(sv) for sk, sv in raw_norm["state"].items()
                    }
                if "actions" in raw_norm:
                    dataset_stats["action"] = {
                        sk: torch.tensor(sv) for sk, sv in raw_norm["actions"].items()
                    }
                logger.info(
                    "norm_stats loaded from %s: keys=%s",
                    norm_paths[0], list(dataset_stats.keys()),
                )

        policy = policy_cls(cfg)
        state_dict = load_file(str(ckpt_dir / "model.safetensors"))
        # weight tying 호환: lm_head ↔ embed_tokens (PaliGemma)
        embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
        if lm_head_key in state_dict and embed_key not in state_dict:
            state_dict[embed_key] = state_dict[lm_head_key]
        policy.model.load_state_dict(state_dict, strict=True)
        logger.info("외부 체크포인트 로드 성공")
    else:
        policy = policy_cls.from_pretrained(pretrained_path)

    policy.config.device = args.device
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=pretrained_path if has_lerobot_config else None,
        dataset_stats=dataset_stats,
    )

    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import ACTION

    _n_action_steps = getattr(policy.config, "n_action_steps", 1)

    if ACTION in policy.config.output_features:
        feat = policy.config.output_features[ACTION]
        _action_dim = feat["shape"][0] if isinstance(feat, dict) else feat.shape[0]
    else:
        _action_dim = 7

    visual_keys = [
        k for k, v in policy.config.input_features.items()
        if (v.get("type") if isinstance(v, dict) else v.type) == (
            FeatureType.VISUAL if not isinstance(v, dict) else "VISUAL"
        )
    ]
    state_feat = getattr(policy.config, "robot_state_feature", None)
    if state_feat is None and "observation.state" in policy.config.input_features:
        sf = policy.config.input_features["observation.state"]
        _state_dim = sf["shape"][0] if isinstance(sf, dict) else sf.shape[0]
    _camera_key_map, sd = _build_remap_config(visual_keys, state_feat)
    if sd > 0:
        _state_dim = sd

    logger.info(
        "LeRobot '%s' loaded from %s "
        "(n_action_steps=%d, visual_keys=%s, state_dim=%d, action_dim=%d)",
        _policy_type, pretrained_path, _n_action_steps,
        visual_keys, _state_dim, _action_dim,
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    global _profile

    try:
        from src.utils.common.logger import create_module_logger

        create_module_logger("lerobot_serve")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="LeRobot policy 추론 서버 (통일 API)")
    parser.add_argument(
        "--profile", type=str, required=True,
        help="체크포인트 프로파일 YAML 경로 (configs/checkpoints/*.yaml)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    args = parser.parse_args()

    _profile = load_profile(args.profile)
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "lerobot", (
        f"profile.base_model={_profile.base_model!r}, but this server is lerobot"
    )

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
