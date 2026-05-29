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
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

# SAFE feature hook (scripts/serve 는 스크립트 실행 시 sys.path[0])
import safe_hooks  # noqa: E402

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
# SAFE 수집 전용 모드. True 면 /act(hook 없는 추론)를 거부한다.
# 이유: compile_model=True 인 정책은 sample_actions 가 "처음" compile 될 때 hook 이
# 등록돼 있어야 SAFE forward hook 이 발화한다. /act 가 먼저 돌면 hook 없는 그래프가
# 캐시돼 이후 /act_with_features 의 hook 이 무시(features=None)된다. 수집 serve 는
# /act_with_features 만 받아 첫 compile 이 hook 과 함께 일어나도록 강제한다.
_collect_mode: bool = False

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


def _encode_ndarray(arr: np.ndarray) -> str:
    """ndarray → base64(np.save bytes). dtype/shape 보존, JSON list 대비 경량."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


# state sub-key → dim. 프로파일 기반 state 조립에 사용 (openvla_oft serve 와 동일 규약).
# eef_quat 은 axisangle(3D)로 내부 변환되어 들어감.
_STATE_DIM: dict = {
    "eef_pos": 3,
    "eef_euler": 3,
    "eef_axisangle": 3,
    "eef_quat": 3,
    "eef_rot6d": 6,
    "gripper_qpos": 2,
    "gripper_opening": 1,
    "gripper_action": 1,
    "joint_pos": 7,
    "joint_vel": 7,
}


def _quat_xyzw_to_axisangle(quat) -> np.ndarray:
    """Quaternion (x,y,z,w) → axis-angle (3D). lerobot LiberoProcessorStep._quat2axisangle 와 동일."""
    quat = np.asarray(quat, dtype=np.float64)
    w = max(-1.0, min(1.0, float(quat[3])))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(w) / den).astype(np.float32)


def _build_state_from_profile(payload: dict, profile: CheckpointProfile) -> np.ndarray:
    """프로파일 observation_requirements.state 순서대로 state 벡터 조립 (선언된 변환 수행).

    각 모델이 학습된 layout 을 프로파일에 명시 → serve 가 모델별로 맞춰 조립.
    예) LIBERO pi05: [eef_pos, eef_axisangle, gripper_qpos], allow_conversions=[quat_to_axisangle]
        → 들어온 eef_quat 을 axisangle 로 변환해 8D 조립.
    """
    conversions = set(profile.observation_requirements.allow_conversions)
    parts: list[np.ndarray] = []
    for key in profile.observation_requirements.state:
        dim = _STATE_DIM.get(key, 0)
        raw = payload.get(f"observation.state.{key}")
        if raw is not None:
            arr = np.array(raw, dtype=np.float32).flatten()
            if key == "eef_quat":  # quat 직접 제공 → axisangle(3D)
                arr = _quat_xyzw_to_axisangle(raw)
        elif key == "eef_axisangle":
            quat = payload.get("observation.state.eef_quat")
            euler = payload.get("observation.state.eef_euler")
            if quat is not None and "quat_to_axisangle" in conversions:
                arr = _quat_xyzw_to_axisangle(quat)
            elif euler is not None and "euler_to_axisangle" in conversions:
                from scipy.spatial.transform import Rotation

                arr = Rotation.from_euler("xyz", euler).as_rotvec().astype(np.float32)
            else:
                arr = np.zeros(dim, dtype=np.float32)
        elif key == "gripper_qpos":
            ga = payload.get("observation.state.gripper_action")
            if ga is not None:
                g = float(np.array(ga).flatten()[0])
                arr = np.array([g, g], dtype=np.float32)
            else:
                arr = np.zeros(dim, dtype=np.float32)
        else:
            arr = np.zeros(dim, dtype=np.float32)

        if key == "gripper_qpos" and len(arr) == 1:
            arr = np.array([arr[0], arr[0]], dtype=np.float32)
        if dim and len(arr) != dim:
            arr = arr[:dim] if len(arr) > dim else np.concatenate(
                [arr, np.zeros(dim - len(arr), dtype=np.float32)]
            )
        parts.append(arr.astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


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

    rotate_180 = bool(
        _profile is not None and _profile.image_preprocess.rotate_180
    )
    for k, v in payload.items():
        if k.startswith("observation.images."):
            np_img = _b64_to_numpy(v)
            t = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
            if rotate_180:
                # 학습 데이터(LIBERO)는 180° 회전 이미지 사용. lerobot LiberoProcessorStep
                # 과 동일하게 H,W 축을 뒤집어 학습 시점 orientation 으로 맞춤.
                t = torch.flip(t, dims=[1, 2])
            batch[k] = t.unsqueeze(0)  # [1, C, H, W]

    # state: 프로파일이 layout 을 선언했으면 그에 맞춰 조립(변환 포함),
    # 아니면 STATE_KEY_ORDER 단순 concat fallback (기존 동작 보존).
    state_np = None
    if _profile is not None and _profile.observation_requirements.state:
        state_np = _build_state_from_profile(payload, _profile)
    if state_np is None or state_np.size == 0:
        state_parts = [
            np.array(payload[key], dtype=np.float32)
            for key in STATE_KEY_ORDER
            if key in payload
        ]
        state_np = np.concatenate(state_parts) if state_parts else None
    if state_np is not None and state_np.size > 0:
        batch["observation.state"] = torch.from_numpy(
            state_np.astype(np.float32)
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
    if _collect_mode:
        # SAFE 수집 serve 에서 /act(hook 없는 추론)가 먼저 돌면 compile 그래프가 hook
        # 없이 캐시돼 /act_with_features 가 features=None 이 된다. 조용한 실패를 막기
        # 위해 명시적으로 거부한다. 수집 시엔 /act_with_features 만 사용할 것.
        raise HTTPException(
            status_code=409,
            detail="serve is in --collect mode; use /act_with_features (not /act). "
            "Running /act first poisons the compiled graph and disables SAFE hooks.",
        )

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


@app.post("/act_with_features")
async def predict_action_with_features(payload: dict):
    """SAFE 수집용: /act 와 동일하되 추론이 발화한 step 에서 SAFE hidden_states 동봉.

    lerobot 정책은 action queue 가 빌 때(매 n_action_steps)만 새 추론을 돌리므로,
    그 step 에만 has_feature=True 와 hidden_states_b64(=[K,H,D] 또는 [1,n_tokens,D])
    가 채워진다. 그 외 step 은 버퍼된 action 만 반환(has_feature=False).
    """
    if policy is None:
        return {"error": "model not loaded"}
    profile = _profile
    assert profile is not None
    if _policy_type not in safe_hooks.SUPPORTED_TYPES:
        return {"error": f"SAFE features unsupported for policy_type={_policy_type}"}

    t0 = time.time()
    batch = parse_payload(payload)
    batch = _apply_input_remap(batch)

    if preprocessor is not None:
        batch = preprocessor(batch)

    action, hidden, _axes, meta = safe_hooks.run_with_features(policy, batch, _policy_type)

    if postprocessor is not None:
        action = postprocessor(action)

    action_np = action.detach().cpu().float().numpy()
    if action_np.ndim == 1:
        action_np = action_np[np.newaxis, :]

    result = _emit_subkeys(action_np, profile)
    if hidden is not None:
        result["has_feature"] = True
        result["hidden_states_b64"] = _encode_ndarray(hidden)
        result.update(meta)  # feature_kind, feature_axes, num_inference_timesteps, ...
    else:
        result["has_feature"] = False
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
        "collect_mode": _collect_mode,
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

    # repo root 기준 경로 (host conda / Docker 컨테이너 양쪽 지원)
    _repo_root = Path(__file__).resolve().parents[2]
    _lerobot_src = _repo_root / "lerobot" / "src"
    if _lerobot_src.is_dir():
        sys.path.insert(0, str(_lerobot_src))

    # 체크포인트 경로 결정
    src = profile.checkpoint_source
    pretrained_path = src.id  # local path or HF repo id
    # 프로파일의 컨테이너 절대경로(/temporal_vla/...)를 host repo root 로 remap.
    # 컨테이너에서는 _repo_root == /temporal_vla 이므로 identity.
    if isinstance(pretrained_path, str) and pretrained_path.startswith("/temporal_vla/"):
        _mapped = _repo_root / pretrained_path[len("/temporal_vla/") :]
        if _mapped.exists():
            pretrained_path = str(_mapped)
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
    # 프로파일이 n_action_steps 를 지정하면 policy config 에 적용
    # (예: pi05 LIBERO=10, OpenPI/lerobot 재현 기준. 체크포인트 기본값 50 override).
    _prof_nas = getattr(profile, "n_action_steps", None)
    if _prof_nas and hasattr(policy.config, "n_action_steps"):
        policy.config.n_action_steps = int(_prof_nas)
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
    global _profile, _collect_mode

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
    parser.add_argument(
        "--collect",
        action="store_true",
        help="SAFE 수집 전용 모드. /act 를 거부하고 /act_with_features 만 허용한다. "
        "compile_model=True 정책에서 SAFE hook 이 첫 compile 에 포함되도록 보장 "
        "(/act 선행 시 hook 없는 그래프가 캐시돼 features=None). compile 은 유지된다.",
    )
    args = parser.parse_args()

    _collect_mode = bool(args.collect)
    _profile = load_profile(args.profile)
    if _collect_mode:
        logger.info(
            "SAFE collect mode ON: /act 거부, /act_with_features 만 허용 (compile 유지)."
        )
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "lerobot", (
        f"profile.base_model={_profile.base_model!r}, but this server is lerobot"
    )

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
