"""
LeRobot policy 추론 서버 (통일 API).

프로파일 기반 동작: 체크포인트별 policy_type, dataset_stats 경로, 외부
체크포인트 fallback config 등은 configs/checkpoints/*.yaml 에 선언.

lerobot 컨테이너에서 실행:
  docker compose run --rm lerobot \
    python /temporal_vla/scripts/serve/lerobot.py \
    --profile /temporal_vla/configs/checkpoints/lerobot_pi05__calvin_sft.yaml

카메라/state 키 매핑: profile 의 policy_type 에 맞는 adapter 가 담당.
  - pi 계열: observation.images.static/wrist/wrist2 순서 매핑
  - groot: side_0/side_1/wrist_0 및 left/right/wrist alias, RoboCasa 20D state 조립

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
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

_SERVE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_UTILS_ROOT = _SCRIPTS_ROOT / "utils"
for _path in (_REPO_ROOT, _SCRIPTS_ROOT, _UTILS_ROOT, _SERVE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402
from lerobot_adapters import (  # noqa: E402
    STATE_DIM,
    load_dataset_stats,
    make_policy_adapter,
    preprocess_image_numpy,
)
from lerobot_adapters.pi import PiPolicyAdapter  # noqa: E402
from lerobot_adapters.rotation import quat_xyzw_to_axisangle  # noqa: E402
from src.utils.common.image import decode_b64_image  # noqa: E402
from src.utils.common.serving import (  # noqa: E402
    add_server_args,
    health_response,
    reset_policy,
    run_uvicorn,
    setup_serve_logging,
)

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
_policy_adapter = None
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


def _encode_ndarray(arr: np.ndarray) -> str:
    """ndarray → base64(np.save bytes). dtype/shape 보존, JSON list 대비 경량."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def _state_payload_keys(key: str) -> tuple[str, ...]:
    if _policy_adapter is not None:
        return _policy_adapter.state_payload_keys(key)
    return (f"observation.state.{key}",)


def _build_state_from_profile(payload: dict, profile: CheckpointProfile) -> np.ndarray:
    """프로파일 observation_requirements.state 순서대로 state 벡터 조립 (선언된 변환 수행).

    각 모델이 학습된 layout 을 프로파일에 명시 → serve 가 모델별로 맞춰 조립.
    예) LIBERO pi05: [eef_pos, eef_axisangle, gripper_qpos], allow_conversions=[quat_to_axisangle]
        → 들어온 eef_quat 을 axisangle 로 변환해 8D 조립.
    """
    conversions = set(profile.observation_requirements.allow_conversions)
    parts: list[np.ndarray] = []
    for key in profile.observation_requirements.state:
        dim = STATE_DIM.get(key, 0)
        if _policy_adapter is not None:
            dim = _policy_adapter.state_dim(key, dim)
        raw = None
        for payload_key in _state_payload_keys(key):
            if payload_key in payload:
                raw = payload[payload_key]
                break
        if raw is not None:
            if _policy_adapter is not None:
                raw = _policy_adapter.transform_state_value(key, raw)
            arr = np.array(raw, dtype=np.float32).flatten()
            if key == "eef_quat":  # quat 직접 제공 → axisangle(3D)
                arr = quat_xyzw_to_axisangle(raw)
        elif key == "eef_axisangle":
            quat = payload.get("observation.state.eef_quat")
            euler = payload.get("observation.state.eef_euler")
            if quat is not None and "quat_to_axisangle" in conversions:
                arr = quat_xyzw_to_axisangle(quat)
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


def _build_remap_config(visual_keys: list, state_feat, policy_type: str | None = None) -> tuple:
    """policy input_features 에서 camera key map 과 state dim 도출."""
    if policy_type is not None:
        adapter = make_policy_adapter(policy_type)
    else:
        adapter = _policy_adapter or PiPolicyAdapter()
    return adapter.build_remap_config(visual_keys, state_feat)


def _apply_input_remap(batch: dict) -> dict:
    """통일 API batch → policy input_features 키 형식 변환."""
    if _camera_key_map:
        for src, dst in _camera_key_map.items():
            if src in batch:
                value = batch.pop(src)
                if dst not in batch:
                    batch[dst] = value
    if _state_dim > 0 and "observation.state" in batch:
        batch["observation.state"] = batch["observation.state"][:, :_state_dim]
    return batch


def _apply_inference_seed(payload: dict) -> int | None:
    """Apply optional per-request sampling seed for stochastic policies."""
    raw_seed = payload.get("inference_seed")
    if raw_seed is None:
        return None
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="inference_seed must be an integer") from exc
    if seed < 0:
        raise HTTPException(status_code=400, detail="inference_seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def parse_payload(payload: dict) -> dict:
    """HTTP JSON payload → LeRobot batch dict."""
    batch = {}

    image_preprocess = getattr(_profile, "image_preprocess", None)
    rotate_180 = bool(getattr(image_preprocess, "rotate_180", False))
    for k, v in payload.items():
        if k.startswith("observation.images."):
            np_img = decode_b64_image(v)
            np_img = preprocess_image_numpy(np_img, image_preprocess)
            t = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
            if rotate_180:
                # 학습 데이터(LIBERO)는 180° 회전 이미지 사용. lerobot LiberoProcessorStep
                # 과 동일하게 H,W 축을 뒤집어 학습 시점 orientation 으로 맞춤.
                t = torch.flip(t, dims=[1, 2])
            batch[k] = t.unsqueeze(0)  # [1, C, H, W]

    # state: 프로파일이 layout 을 선언했으면 그에 맞춰 조립(변환 포함),
    # 아니면 STATE_KEY_ORDER 단순 concat fallback (기존 동작 보존).
    state_np = None
    obs_req = getattr(_profile, "observation_requirements", None)
    if obs_req is not None and getattr(obs_req, "state", None):
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
    return reset_policy(policy)


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

    inference_seed = _apply_inference_seed(payload)
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
    if inference_seed is not None:
        result["inference_seed"] = inference_seed
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
    inference_seed = _apply_inference_seed(payload)
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
    if inference_seed is not None:
        result["inference_seed"] = inference_seed
    result["latency_ms"] = (time.time() - t0) * 1000
    return result


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "lerobot"}
    return health_response(
        policy=policy,
        model=_policy_type,
        profile=_profile,
        n_action_steps=_n_action_steps,
        action_type=_profile.action_type,
        action_keys=list(_profile.emits_subkeys),
        collect_mode=_collect_mode,
    )


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
    global policy, preprocessor, postprocessor
    global _policy_type, _policy_adapter, _n_action_steps
    global _action_dim, _camera_key_map, _state_dim

    args = getattr(app.state, "args", None)
    if args is None:
        return  # 테스트 환경: args 없으면 로딩 skip

    profile = _profile
    assert profile is not None

    ms = profile.model_specific
    _policy_type = ms.get("policy_type", "pi0")
    _policy_adapter = None
    _camera_key_map = {}
    _state_dim = 0

    # repo root 기준 경로 (host conda / Docker 컨테이너 양쪽 지원)
    _repo_root = Path(__file__).resolve().parents[2]
    _lerobot_src = _repo_root / "lerobot" / "src"
    if _lerobot_src.is_dir():
        sys.path.insert(0, str(_lerobot_src))

    adapter = make_policy_adapter(_policy_type)
    _policy_adapter = adapter
    pretrained_path = adapter.resolve_pretrained_path(profile, _repo_root)

    logger.info(
        "Loading LeRobot policy_type=%s from %s (profile=%s, adapter=%s)",
        _policy_type, pretrained_path, profile.name, type(adapter).__name__,
    )

    dataset_stats = load_dataset_stats(profile)

    from lerobot.policies.factory import get_policy_class

    policy_cls = get_policy_class(_policy_type)
    loaded = adapter.load(
        profile=profile,
        policy_cls=policy_cls,
        pretrained_path=pretrained_path,
        dataset_stats=dataset_stats,
        device=args.device,
    )
    policy = loaded.policy
    preprocessor = loaded.preprocessor
    postprocessor = loaded.postprocessor

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
    _camera_key_map, sd = adapter.build_remap_config(visual_keys, state_feat)
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

    setup_serve_logging("lerobot_serve")

    parser = argparse.ArgumentParser(description="LeRobot policy 추론 서버 (통일 API)")
    parser.add_argument(
        "--profile", type=str, required=True,
        help="체크포인트 프로파일 YAML 경로 (configs/checkpoints/*.yaml)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    add_server_args(parser, default_port=8400)
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
    run_uvicorn(app, args)


if __name__ == "__main__":
    main()
