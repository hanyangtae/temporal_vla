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
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

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
from src.utils.common.feature_blob import (  # noqa: E402
    encode_feature_blob,
    encode_legacy_feature_array,
)
from src.policies.safe_metadata import (  # noqa: E402
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES,
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND,
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_AXES,
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_KIND,
    GROOT_N15_VL_FEATURE_KIND,
    GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_AXES,
    GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_KIND,
    GROOT_VL_FEATURE_AXES,
    PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES,
    PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND,
    lerobot_feature_axes,
    lerobot_feature_kind,
    normalize_feature_metadata,
)
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
_capture_vl_features: bool = False
_groot_dit_capture_layers: tuple[int, ...] | None = None
_pi05_expert_capture_layers: tuple[int, ...] | None = None
_groot_dit_token_pool: str = "action_token_mean"  # pq3: "all_token_full" = full-token 수집
_groot_vl_capture_point: str = "vlln_mean"  # pq3: "post_vl_sa_full" = cross-attn 입력 full-token
_steering = []  # list of registered steering hooks (multi-layer 지원)
_gated_registry: dict = {}  # oracle phase-gated steering: {"hooks":{layer:hook},"matrices":{layer:{phase:M|M_seq}},"identity":{layer:I|[I]*K}}


def _reset_steering_step_counters() -> None:
    """Per-Step steering 의 denoise call 카운터를 요청 시작 시 리셋.

    phase 는 요청 단위(/steering_phase), step 은 요청 내 denoise call 단위라 직교 —
    /act·/act_with_features 진입부에서 매 요청 호출한다 (global M 단일 hook 은 no-op).
    """
    for hook in _steering:
        reset = getattr(hook, "reset_step_counter", None)
        if reset is not None:
            reset()

# payload 의 observation.state.* 서브키를 lerobot observation.state 로 합칠 때 사용할
# canonical 정렬 순서 (벤치 공통). 체크포인트가 학습된 state dim 만큼 앞에서 truncate.
STATE_KEY_ORDER = [
    "observation.state.eef_pos",
    "observation.state.eef_euler",
    "observation.state.eef_quat",
    "observation.state.base_to_eef_pos",
    "observation.state.base_to_eef_quat",
    "observation.state.gripper_opening",
    "observation.state.gripper_qpos",
    "observation.state.joint_pos",
    "observation.state.joint_vel",
    "observation.state.gripper_action",
    "observation.state.base_pos",
    "observation.state.base_quat",
]


# ─── 변환 유틸 ────────────────────────────────────────────────────────────────


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
        # base-frame relative 키가 없으면 world-frame alias로 fallback (safety net).
        # RoboCasaObsProcessor는 robot0_base_to_eef_pos/_quat 를 직접 emit 하므로
        # robocasa pi05는 이 fallback이 발동하지 않는다. 다른 벤치마크 대비 안전망.
        if raw is None and key in ("eef_pos_rel", "base_to_eef_pos"):
            raw = payload.get("observation.state.eef_pos")
        if raw is None and key in ("eef_quat_rel", "base_to_eef_quat"):
            raw = payload.get("observation.state.eef_quat")

        if raw is not None:
            if _policy_adapter is not None:
                raw = _policy_adapter.transform_state_value(key, raw)
            arr = np.array(raw, dtype=np.float32).flatten()
            # eef_quat_rel: 4D quat 그대로 사용 (no conversion)
            # eef_quat (STATE_DIM=3 모델): axisangle 변환 — 이 분기는 legacy 동작
            if key == "eef_quat" and dim == 3:
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
        st = batch["observation.state"]
        cur_dim = st.shape[-1]
        if cur_dim > _state_dim:
            st = st[..., :_state_dim]
        elif cur_dim < _state_dim:
            # pi05 robocasa 등 max_state_dim 에 zero-pad (openpi pad_to_dim 동일)
            import torch as _torch
            pad = _torch.zeros(*st.shape[:-1], _state_dim - cur_dim, dtype=st.dtype, device=st.device)
            st = _torch.cat([st, pad], dim=-1)
        batch["observation.state"] = st
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


def _parse_groot_dit_capture_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("--groot-dit-capture-layers must be a comma-separated int list")
    layers = tuple(int(part) for part in parts)
    if len(layers) == 0:
        raise ValueError("--groot-dit-capture-layers must not be empty")
    return layers


def _parse_pi05_expert_capture_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("--pi05-expert-capture-layers must be a comma-separated int list")
    layers = tuple(int(part) for part in parts)
    if len(layers) == 0:
        raise ValueError("--pi05-expert-capture-layers must not be empty")
    return layers


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


@app.post("/steering_phase")
def steering_phase(payload: dict):
    """Oracle phase-gated steering: 현재 phase 의 conceptor 로 hook M 을 스위칭.

    수집 client 가 매 get_action 전에 POST {"phase": "<reach-to-object|transport|...>"}.
    등록된 phase 가 없으면 identity(=no steer). --steering-phase-npz-base 로 활성화.
    """
    if not _gated_registry:
        raise HTTPException(status_code=409, detail="gated steering not enabled")
    phase = str(payload.get("phase", ""))
    for layer, hook in _gated_registry["hooks"].items():
        M = _gated_registry["matrices"][layer].get(phase)
        # set_matrices 가 M(단일) / M_seq(per-step 리스트) 모두 수용, 텐서 캐시·step
        # 카운터도 함께 리셋한다 (구 ``hook.M=...; hook._Mt=None`` 배선 대체).
        hook.set_matrices(M if M is not None else _gated_registry["identity"][layer])
    _gated_registry["current"] = phase
    return {"ok": True, "phase": phase, "gated": phase in next(iter(_gated_registry["matrices"].values()))}


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


def _postprocess_action_preserve_chunk(action: torch.Tensor) -> torch.Tensor:
    """Run the LeRobot postprocessor without collapsing a [B,H,D] action chunk."""
    if postprocessor is None:
        return action
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        return postprocessor(action)

    processed_steps = []
    for step_idx in range(action.shape[1]):
        processed_steps.append(postprocessor(action[:, step_idx, :]))
    return torch.stack(processed_steps, dim=1)


def _action_to_emit_array(action: torch.Tensor) -> np.ndarray:
    action_np = action.detach().cpu().float().numpy()
    if action_np.ndim == 1:
        return action_np[np.newaxis, :]
    if action_np.ndim == 2:
        return action_np
    if action_np.ndim == 3:
        if action_np.shape[0] != 1:
            raise ValueError(f"Only batch size 1 action chunks are supported, got {action_np.shape}")
        return action_np[0]
    raise ValueError(f"Unsupported action tensor shape: {action_np.shape}")


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
    _reset_steering_step_counters()
    batch = parse_payload(payload)
    batch = _apply_input_remap(batch)

    if preprocessor is not None:
        batch = preprocessor(batch)

    with torch.inference_mode():
        action = policy.select_action(batch)

    action = _postprocess_action_preserve_chunk(action)
    action_np = _action_to_emit_array(action)

    result = _emit_subkeys(action_np, profile)
    if inference_seed is not None:
        result["inference_seed"] = inference_seed
    result["latency_ms"] = (time.time() - t0) * 1000
    return result


@app.post("/act_with_features")
async def predict_action_with_features(payload: dict):
    """SAFE 수집용: /act 와 동일하되 추론이 발화한 step 에서 SAFE hidden_states 동봉.

    GR00T N1.5 collect는 N1.6 SAFE collector와 같은 chunk execution을 맞추기 위해
    predict_action_chunk를 hook 아래에서 직접 호출하고 [H,D] action subkeys를 반환한다.
    다른 lerobot 정책은 action queue가 빌 때만 새 추론을 돌리므로 그 step에만
    has_feature=True, legacy hidden_states_b64, unified features.hidden_states blob이
    채워진다.
    """
    if policy is None:
        return {"error": "model not loaded"}
    profile = _profile
    assert profile is not None
    if _policy_type not in safe_hooks.SUPPORTED_TYPES:
        return {"error": f"SAFE features unsupported for policy_type={_policy_type}"}

    t0 = time.time()
    inference_seed = _apply_inference_seed(payload)
    _reset_steering_step_counters()
    batch = parse_payload(payload)
    batch = _apply_input_remap(batch)

    if preprocessor is not None:
        batch = preprocessor(batch)

    action, hidden, _axes, meta = safe_hooks.run_with_features(
        policy,
        batch,
        _policy_type,
        capture_vl=_capture_vl_features,
        groot_dit_layers=_groot_dit_capture_layers,
        pi05_expert_layers=_pi05_expert_capture_layers,
        groot_dit_token_pool=_groot_dit_token_pool,
        vl_capture_point=_groot_vl_capture_point,
    )

    action = _postprocess_action_preserve_chunk(action)
    action_np = _action_to_emit_array(action)

    result = _emit_subkeys(action_np, profile)
    if hidden is not None:
        result["has_feature"] = True
        hidden_np = np.asarray(hidden)
        # Keep the legacy keys for existing collectors, and also emit the
        # unified /act_with_features contract used by VLAClient and GR00T HTTP.
        result["hidden_states_b64"] = encode_legacy_feature_array(hidden_np)
        result["features.hidden_states"] = encode_feature_blob(hidden_np)
        vl_hidden = meta.get("vl_hidden_states")
        result.update(
            {k: v for k, v in meta.items() if k != "vl_hidden_states"}
        )  # feature_kind, feature_axes, num_inference_timesteps, ...
        metadata = normalize_feature_metadata(meta)
        result["features.kind"] = metadata.feature_kind
        result["features.axes"] = metadata.feature_axes
        result["exported_action_token_count"] = metadata.exported_action_token_count
        result["features.exported_action_token_count"] = (
            metadata.exported_action_token_count
        )
        result["features.feature_action_horizon"] = metadata.feature_action_horizon
        result["features.model_action_horizon"] = metadata.model_action_horizon
        result["features.num_inference_timesteps"] = metadata.num_inference_timesteps
        if vl_hidden is not None:
            result["features.vl_hidden_states"] = encode_feature_blob(
                np.asarray(vl_hidden)
            )
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
    feature_metadata = _health_feature_metadata()
    return health_response(
        policy=policy,
        model=_policy_type,
        profile=_profile,
        n_action_steps=_n_action_steps,
        action_type=_profile.action_type,
        action_keys=list(_profile.emits_subkeys),
        collect_mode=_collect_mode,
        capture_vl=_capture_vl_features,
        **feature_metadata,
    )


def _health_feature_metadata() -> dict[str, Any]:
    if _policy_type not in safe_hooks.SUPPORTED_TYPES:
        return {}

    _groot_block_mode = _policy_type == "groot" and _groot_dit_capture_layers is not None
    _pi05_block_mode = _policy_type == "pi05" and _pi05_expert_capture_layers is not None
    if _groot_block_mode:
        _full = _groot_dit_token_pool == "all_token_full"
        metadata: dict[str, Any] = {
            "supports_features": True,
            "feature_kind": (
                GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_KIND
                if _full
                else GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND
            ),
            "feature_axes": list(
                GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_AXES
                if _full
                else GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES
            ),
            "feature_dtype": "float32",
            "model_action_horizon": _n_action_steps,
            "groot_dit_capture_layers": [int(layer) for layer in _groot_dit_capture_layers],
            "capture_token_mode": _groot_dit_token_pool,
        }
    elif _pi05_block_mode:
        metadata = {
            "supports_features": True,
            "feature_kind": PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND,
            "feature_axes": list(PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES),
            "feature_dtype": "float32",
            "model_action_horizon": _n_action_steps,
            "pi05_expert_capture_layers": [
                int(layer) for layer in _pi05_expert_capture_layers
            ],
        }
    else:
        metadata = {
            "supports_features": True,
            "feature_kind": lerobot_feature_kind(_policy_type),
            "feature_axes": lerobot_feature_axes(_policy_type),
            "feature_dtype": "float32",
        }
    if (
        _policy_type in safe_hooks.FLOW_MATCHING_TYPES
        and not (_groot_block_mode or _pi05_block_mode)
    ):
        metadata["feature_action_horizon"] = _n_action_steps
        metadata["model_action_horizon"] = _n_action_steps
    if _policy_type == "groot" and _capture_vl_features:
        _vl_full = _groot_vl_capture_point == "post_vl_sa_full"
        metadata.update(
            {
                "vl_feature_kind": (
                    GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_KIND
                    if _vl_full
                    else GROOT_N15_VL_FEATURE_KIND
                ),
                "vl_feature_axes": list(
                    GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_AXES
                    if _vl_full
                    else GROOT_VL_FEATURE_AXES
                ),
                "vl_feature_dim": 2048,
                "vl_capture_point": _groot_vl_capture_point,
            }
        )
    return metadata


def _register_steering_if_requested(loaded_policy, args):
    global _steering
    steering_npz = getattr(args, "steering_npz", None)
    steering_npz_dir = getattr(args, "steering_npz_dir", None)
    steering_layers = getattr(args, "steering_layers", None)
    if not steering_npz and not steering_npz_dir and not getattr(args, "steering_phase_npz_base", None):
        return None
    if _policy_type not in ("groot", "pi05"):
        raise ValueError("Conceptor steering requires policy_type in {'groot', 'pi05'}")

    from steering_hooks import (
        ConceptorSteering,
        Pi05ConceptorSteering,
        load_steering_matrices_per_step,
        load_steering_matrix,
    )

    # unregister any previously-registered hooks (reload-safe)
    for _h in _steering:
        _h.unregister()
    _steering = []

    beta = getattr(args, "steering_beta", 0.3)
    alpha = getattr(args, "steering_alpha", None)
    key = getattr(args, "steering_key", "C_steer")
    # pq3: token_select 는 default None(pathway 기본 보존 — dit=last_horizon, vl=all),
    # denoise 는 global(구 단일 M) | per_step(step k 에 M_k 스와핑, groot dit 전용).
    token_select = getattr(args, "steering_token_select", None)
    denoise = getattr(args, "steering_denoise", "global") or "global"
    per_step = denoise == "per_step"
    expected_steps = None
    if per_step:
        if _policy_type != "groot":
            raise ValueError("--steering-denoise per_step 은 groot 전용")
        _gm = getattr(loaded_policy, "_groot_model", None)
        if _gm is None:
            raise ValueError("GR00T LeRobot policy is missing _groot_model for steering")
        expected_steps = int(_gm.action_head.num_inference_timesteps)

    def _load_matrices(npz_path):
        """denoise 모드에 맞는 M(단일) 또는 M_seq(list) 로드 + preflight 로그."""
        if per_step:
            return load_steering_matrices_per_step(
                str(npz_path), beta=beta, alpha=alpha, key=key, num_steps=expected_steps
            )
        return load_steering_matrix(str(npz_path), beta=beta, alpha=alpha, key=key)

    # --- Oracle phase-gated multi-layer steering: /steering_phase 로 M 스위칭 ---
    phase_base = getattr(args, "steering_phase_npz_base", None)
    if phase_base and steering_layers:
        global _gated_registry
        if _policy_type != "groot":
            raise ValueError("--steering-phase-npz-base 는 groot dit pathway 전용")
        groot_model = getattr(loaded_policy, "_groot_model", None)
        if groot_model is None:
            raise ValueError("GR00T LeRobot policy is missing _groot_model for steering")
        import numpy as _np

        layers = [int(x) for x in str(steering_layers).split(",") if x.strip()]
        base = Path(phase_base)
        phases = sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / f"dit_L{layers[0]}" / "conceptors.npz").exists()
        )
        if not phases:
            raise FileNotFoundError(f"phase 서브디렉토리 없음: {base}")
        hooks, matrices, identity = {}, {}, {}
        for lyr in layers:
            matrices[lyr] = {}
            for ph in phases:
                npz_path = base / ph / f"dit_L{lyr}" / "conceptors.npz"
                if npz_path.exists():
                    matrices[lyr][ph] = _load_matrices(npz_path)
            first = next(iter(matrices[lyr].values()))
            dim = (first[0] if isinstance(first, list) else first).shape[0]
            # per-step 이면 identity 도 [I]×K 로 통일 (전 요청에서 카운터 배선 동일 검증)
            identity[lyr] = (
                [_np.eye(dim)] * expected_steps if per_step else _np.eye(dim)
            )
            hook = ConceptorSteering(
                groot_model, identity[lyr], pathway="dit", layer=lyr,
                token_select=token_select,
            ).register()
            hooks[lyr] = hook
            _steering.append(hook)
        _gated_registry = {"hooks": hooks, "matrices": matrices, "identity": identity, "current": None}
        logger.info(
            "Phase-gated conceptor steering registered: base=%s layers=%s phases=%s "
            "beta=%s token_select=%s denoise=%s",
            phase_base, layers, phases, beta,
            token_select or "last_horizon(default)", denoise,
        )
        # 러너 preflight 대조용 (module logger 는 serve 로그 파일에 안 남음 — print 필수)
        print(
            f"[steer-registered] path=gated layers={','.join(str(x) for x in layers)} "
            f"phases={','.join(phases)} beta={beta:g} "
            f"token_select={token_select or 'last_horizon(default)'} denoise={denoise}",
            flush=True,
        )
        return _steering

    # --- Multi-layer DiT steering (net-new): layer 마다 hook 하나씩 ---
    if steering_npz_dir and steering_layers:
        if _policy_type != "groot":
            raise ValueError("--steering-npz-dir/--steering-layers 는 groot dit pathway 전용")
        groot_model = getattr(loaded_policy, "_groot_model", None)
        if groot_model is None:
            raise ValueError("GR00T LeRobot policy is missing _groot_model for steering")
        layers = [int(x) for x in str(steering_layers).split(",") if x.strip()]
        for lyr in layers:
            npz_path = Path(steering_npz_dir) / f"dit_L{lyr}" / "conceptors.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"multi-layer steering npz 없음: {npz_path}")
            mat = _load_matrices(npz_path)
            _steering.append(
                ConceptorSteering(
                    groot_model, mat, pathway="dit", layer=lyr,
                    token_select=token_select,
                ).register()
            )
        logger.info(
            "Multi-layer conceptor steering registered: dir=%s layers=%s beta=%s key=%s "
            "token_select=%s denoise=%s",
            steering_npz_dir, layers, beta, key,
            token_select or "last_horizon(default)", denoise,
        )
        print(
            f"[steer-registered] path=multi layers={','.join(str(x) for x in layers)} "
            f"beta={beta:g} key={key} "
            f"token_select={token_select or 'last_horizon(default)'} denoise={denoise}",
            flush=True,
        )
        return _steering

    # --- Single hook (--steering-npz) ---
    if _policy_type == "pi05":
        if per_step:
            raise ValueError("--steering-denoise per_step 은 pi05 미지원 (groot dit 전용)")
        # COAST A.7.1 global: action expert decoder layer ℓ(default 11) residual stream.
        matrix = load_steering_matrix(steering_npz, beta=beta, alpha=alpha, key=key)
        layer = getattr(args, "steering_layer", None)
        if layer is None:
            layer = 11
        _steering.append(Pi05ConceptorSteering(loaded_policy, matrix, layer=int(layer)).register())
        logger.info(
            "Pi05 conceptor steering registered: npz=%s beta=%s alpha=%s key=%s layer=%s",
            steering_npz, beta, alpha, key, layer,
        )
        return _steering

    matrix = _load_matrices(steering_npz)

    groot_model = getattr(loaded_policy, "_groot_model", None)
    if groot_model is None:
        raise ValueError("GR00T LeRobot policy is missing _groot_model for steering")

    pathway = getattr(args, "steering_pathway", "dit")
    if per_step and pathway != "dit":
        raise ValueError("--steering-denoise per_step 은 pathway='dit' 전용")
    layer = None if pathway == "vl" else getattr(args, "steering_layer", None)
    _steering.append(
        ConceptorSteering(
            groot_model, matrix, pathway=pathway, layer=layer,
            token_select=token_select,
        ).register()
    )
    logger.info(
        "Conceptor steering registered: npz=%s pathway=%s beta=%s alpha=%s key=%s "
        "layer=%s token_select=%s denoise=%s",
        steering_npz, pathway, beta, alpha, key, layer,
        token_select or f"{'all' if pathway == 'vl' else 'last_horizon'}(default)", denoise,
    )
    _ts = token_select or f"{'all' if pathway == 'vl' else 'last_horizon'}(default)"
    print(
        f"[steer-registered] path=single pathway={pathway} layer={layer} beta={beta:g} "
        f"key={key} token_select={_ts} denoise={denoise}",
        flush=True,
    )
    return _steering


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
    _register_steering_if_requested(policy, args)

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
    global _profile, _collect_mode, _capture_vl_features, _groot_dit_capture_layers
    global _pi05_expert_capture_layers, _groot_dit_token_pool, _groot_vl_capture_point

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
    parser.add_argument(
        "--capture-vl",
        action="store_true",
        help=(
            "GR00T N1.5 /act_with_features 에서 VL(goal) pathway feature"
            "(action_head.vlln seq-mean-pool)도 함께 반환한다. 기본은 DiT-only."
        ),
    )
    parser.add_argument(
        "--groot-dit-capture-layers",
        default=None,
        help=(
            "Comma-separated GR00T N1.5 DiT transformer_block indices to capture. "
            "지정 시 /act_with_features 의 DiT feature가 final action-token output "
            "대신 block residual [layer, model_token, feature_dim]가 된다."
        ),
    )
    parser.add_argument(
        "--pi05-expert-capture-layers",
        default=None,
        help=(
            "Comma-separated pi05 action expert(Gemma2) decoder layer indices to capture "
            "(COAST A.7.1, e.g. '0,5,11,17'). 지정 시 /act_with_features 의 pi05 feature가 "
            "action_out_proj pre-velocity 대신 expert block residual "
            "[layer, denoise_step, feature_dim](마지막 chunk_size action token mean-pool)가 된다."
        ),
    )
    parser.add_argument(
        "--steering-npz",
        default=None,
        help="Conceptor npz path. 지정 시 GR00T N1.5 HTTP server에 steering hook을 등록한다.",
    )
    parser.add_argument("--steering-beta", type=float, default=0.3)
    parser.add_argument("--steering-alpha", type=float, default=None)
    parser.add_argument(
        "--steering-key",
        choices=("C_steer", "C_success", "C_failure"),
        default="C_steer",
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="DiT block index to steer (pathway=dit). None=action_head.model output.",
    )
    parser.add_argument(
        "--steering-pathway",
        choices=("dit", "vl"),
        default="dit",
        help="Steering pathway: dit=motor action tokens, vl=goal pathway action_head.vlln.",
    )
    parser.add_argument(
        "--steering-layers",
        default=None,
        help=(
            "Multi-layer DiT steering: comma-separated block indices (예: '4,8,12'). "
            "--steering-npz-dir 와 함께 사용하며 각 layer L 의 conceptor 를 "
            "<npz-dir>/dit_L{L}/conceptors.npz 에서 로드해 layer 마다 hook 을 건다."
        ),
    )
    parser.add_argument(
        "--steering-npz-dir",
        default=None,
        help=(
            "Multi-layer steering 용 group 디렉토리 (예: .../conceptor_steering_n15/<cell>/transport). "
            "--steering-layers 의 각 layer 서브디렉토리(dit_L{n}/conceptors.npz)를 로드."
        ),
    )
    parser.add_argument(
        "--steering-phase-npz-base",
        default=None,
        help=(
            "Oracle phase-gated steering: <base>/<phase>/dit_L{n}/conceptors.npz 를 phase 별로 로드하고 "
            "/steering_phase POST 로 매 요청 전 conceptor 를 스위칭. --steering-layers 필요. "
            "등록 안 된 phase 는 identity(no steer)."
        ),
    )
    parser.add_argument(
        "--groot-dit-token-pool",
        choices=("action_token_mean", "all_token_full"),
        default="action_token_mean",
        help=(
            "GR00T DiT block residual 캡처의 token 풀링 (pq3 COAST 토큰 축 정렬). "
            "action_token_mean=구·default([L,K,D]) | all_token_full=전체 토큰 보존"
            "([L,K,T,D] fp16, fit 수집 전용 — mean 은 fit 시점에)."
        ),
    )
    parser.add_argument(
        "--groot-vl-capture-point",
        choices=("vlln_mean", "post_vl_sa_full"),
        default="vlln_mean",
        help=(
            "GR00T VL pathway 캡처 지점. vlln_mean=구·default(vlln 출력 seq-mean [D]) | "
            "post_vl_sa_full=vl_self_attention 출력(=DiT cross-attn 입력) full-token [T_vl,D]."
        ),
    )
    parser.add_argument(
        "--steering-token-select",
        choices=("last_horizon", "all"),
        default=None,
        help=(
            "Steering hook 의 적용 토큰. 미지정(None)=pathway 기본 보존"
            "(dit=last_horizon, vl=all). pq3 COAST 정렬은 dit 에 all 을 명시 주입."
        ),
    )
    parser.add_argument(
        "--steering-denoise",
        choices=("global", "per_step"),
        default="global",
        help=(
            "denoise 축 steering 모드. global=구·default(전 step 같은 M) | per_step="
            "step k 에 M_k 스와핑 (NPZ 키 step{k}_alpha{a}_*, groot dit 전용, "
            "요청 시작마다 카운터 리셋)."
        ),
    )
    args = parser.parse_args()

    _collect_mode = bool(args.collect)
    _capture_vl_features = bool(args.capture_vl)
    _groot_dit_capture_layers = _parse_groot_dit_capture_layers(
        args.groot_dit_capture_layers
    )
    _pi05_expert_capture_layers = _parse_pi05_expert_capture_layers(
        args.pi05_expert_capture_layers
    )
    _groot_dit_token_pool = str(args.groot_dit_token_pool)
    _groot_vl_capture_point = str(args.groot_vl_capture_point)
    _profile = load_profile(args.profile)
    if _collect_mode:
        logger.info(
            "SAFE collect mode ON: /act 거부, /act_with_features 만 허용 (compile 유지)."
        )
    if _capture_vl_features:
        logger.info(
            "SAFE VL capture ON: /act_with_features returns features.vl_hidden_states."
        )
    if _groot_dit_capture_layers is not None:
        logger.info(
            "SAFE GR00T DiT block residual capture ON: layers=%s token_pool=%s",
            ",".join(str(layer) for layer in _groot_dit_capture_layers),
            _groot_dit_token_pool,
        )
    if _capture_vl_features and _groot_vl_capture_point != "vlln_mean":
        logger.info("SAFE GR00T VL capture point: %s", _groot_vl_capture_point)
    if _pi05_expert_capture_layers is not None:
        logger.info(
            "SAFE pi05 expert block residual capture ON: layers=%s",
            ",".join(str(layer) for layer in _pi05_expert_capture_layers),
        )
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "lerobot", (
        f"profile.base_model={_profile.base_model!r}, but this server is lerobot"
    )

    app.state.args = args
    run_uvicorn(app, args)


if __name__ == "__main__":
    main()
