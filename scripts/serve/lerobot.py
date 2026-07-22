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
import uuid
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
_groot_dit_token_pool: str = "action_token_mean"  # exp3(구 pq3): "all_token_full" = full-token 수집
_groot_vl_capture_point: str = "vlln_mean"  # exp3: "post_vl_sa_full" = cross-attn 입력 full-token
# DiT cross-attention 카메라 뷰별 mass 캡처 (attn_hooks.CrossAttnCapture, groot 전용).
# boot 시 1회 install — per-request 설치/해제는 compile 캐시와 상호작용할 수 있어 피한다.
_cross_attn_capture = None
_steering = []  # list of registered steering hooks (multi-layer 지원)
_gated_registry: dict = {}  # oracle phase-gated steering: {"hooks":{layer:hook},"matrices":{layer:{phase:M|M_seq}},"identity":{layer:I|[I]*K}}
# 프로세스 지문: 러너가 "포트의 기존 서버"를 새 serve 로 오인하는 사고 방지 (Gate 2 치명#3)
# — 로그의 [serve-boot] id 와 /health 의 boot_id 가 일치해야 같은 프로세스.
_BOOT_ID = uuid.uuid4().hex[:12]
_steering_spec: dict = {}  # /health 노출용 스티어링 지문 (mode/layers/β/npz sha/…)
# patchceil donor-trajectory transplant (docs/collab/2026-07-16-patching-transplant-gate1.md)
_patch_hooks: dict = {}  # {layer(int): PatchSteering}
_patch_spec: dict = {}  # /health 노출용 patch 지문 (layers/token/K/armed tag/npz sha)
_patch_donor_arrays: dict = {}  # 마지막 로드 donor {layer: [R,K,T,D]} — npz 생략 재-arm 용


def _reset_steering_step_counters() -> None:
    """Per-Step steering 의 denoise call 카운터를 요청 시작 시 리셋.

    phase 는 요청 단위(/steering_phase), step 은 요청 내 denoise call 단위라 직교 —
    /act·/act_with_features 진입부에서 매 요청 호출한다 (global M 단일 hook 은 no-op).
    """
    for hook in _steering:
        reset = getattr(hook, "reset_step_counter", None)
        if reset is not None:
            reset()
    # patch hook 은 같은 호출이 k 리셋 + record cursor 전진을 겸한다
    # (요청 1개 = record 1개 규약, patching_hooks.PatchSteering docstring)
    for hook in _patch_hooks.values():
        hook.reset_step_counter()


def _has_per_step_steering() -> bool:
    return any(getattr(h, "per_step", False) for h in _steering)


def _assert_per_step_hook_counts() -> None:
    """chunk 추론 직후 Per-Step hook 이 정확히 K회 발화했는지 검증 (미발화 무음 방지).

    초과 발화는 hook 자체가 RuntimeError — **미발화**(hook suppression·경로 분기·
    torch.compile 변화)는 여기서 잡는다 (Gate 2 높음#1). per-step hook 이 없으면 no-op.
    chunk 추론이 보장되는 경로(predict_action_chunk)에서만 호출할 것.
    """
    for hook in _steering:
        if getattr(hook, "per_step", False):
            expected = len(hook._M_seq)
            if hook._k != expected:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"per-step steering under-fire: fired {hook._k}/{expected} "
                        f"(layer={hook.layer}) — denoise hook 배선 확인 필요"
                    ),
                )


def _assert_patch_hook_counts() -> None:
    """chunk 추론 직후 patch hook 이 정확히 K회 발화했는지 검증 (미발화 무음 방지).

    over-fire 는 hook 자체가 RuntimeError. 미발화(hook suppression·compile 경로 변화)는
    여기서 잡는다 — per-step steering 의 _assert_per_step_hook_counts 와 동일 규약.
    """
    for layer, hook in _patch_hooks.items():
        # DiT hook 은 요청당 K회(denoise), VL hook 은 요청당 1회 (expected_fires 우선).
        expected = getattr(hook, "expected_fires", None) or hook.expected_k
        if hook._k != expected:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"patch hook under-fire: fired {hook._k}/{expected} "
                    f"(layer={layer}) — hook 배선 확인 필요"
                ),
            )

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


@app.post("/patch_arm")
def patch_arm(payload: dict):
    """patchceil: rollout 1개 분의 transplant 파라미터를 원자적으로 arm.

    러너가 collector 기동 **직전** 호출한다 (collector 의 policy.reset() → /reset 은
    카운터만 리셋하고 arm 은 유지). payload:
      {"npz": donor NPZ 경로(생략 시 직전 로드 재사용), "start_record": int,
       "donor_start": int=0, "patch_len": int=-1(-1=donor 고갈까지), "tag": str}
    """
    if not _patch_hooks:
        raise HTTPException(status_code=409, detail="patch hooks not enabled (--patch-layers)")
    from patching_hooks import load_donor_npz, load_vl_donor_npz

    try:
        start_record = int(payload["start_record"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="start_record(int) 필수") from exc
    donor_start = int(payload.get("donor_start", 0))
    patch_len = int(payload.get("patch_len", -1))
    tag = str(payload.get("tag", "")) or None

    npz = payload.get("npz")
    is_vl = _patch_spec.get("pathway") == "vl"
    if npz:
        try:
            if is_vl:
                vl_arr, meta, sha12 = load_vl_donor_npz(npz)
                arrays = {"VL": vl_arr}
            else:
                expected_k = next(iter(_patch_hooks.values())).expected_k
                arrays, meta, sha12 = load_donor_npz(
                    npz, list(_patch_hooks.keys()), expected_k=expected_k
                )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"donor npz 로드 실패: {exc}") from exc
        _patch_donor_arrays.clear()
        _patch_donor_arrays.update(arrays)
        _patch_spec["donor_npz_sha"] = sha12
        _patch_spec["donor_meta"] = {
            k: meta.get(k)
            for k in ("cell", "episode_idx", "scenario_seed", "inference_seed", "n_records")
        }
    if not _patch_donor_arrays:
        raise HTTPException(status_code=409, detail="donor 미로드 — payload 에 npz 경로 필요")

    try:
        for layer, hook in _patch_hooks.items():
            hook.arm(
                _patch_donor_arrays[layer],
                start_record=start_record,
                donor_start=donor_start,
                patch_len=patch_len,
                tag=tag,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _patch_spec.update(
        {
            "armed_tag": tag,
            "start_record": start_record,
            "donor_start": donor_start,
            "patch_len": patch_len,
        }
    )
    logger.info(
        "[patch-arm] tag=%s start_record=%d donor_start=%d patch_len=%d sha=%s",
        tag, start_record, donor_start, patch_len, _patch_spec.get("donor_npz_sha"),
    )
    return {"ok": True, "boot_id": _BOOT_ID, "patch": dict(_patch_spec)}


@app.post("/patch_disarm")
def patch_disarm():
    """patchceil: no-patch 대조(재실행) rollout 용 — donor 를 내리고 카운터 초기화."""
    if not _patch_hooks:
        raise HTTPException(status_code=409, detail="patch hooks not enabled (--patch-layers)")
    for hook in _patch_hooks.values():
        hook.disarm()
    _patch_spec["armed_tag"] = None
    return {"ok": True, "boot_id": _BOOT_ID}


@app.get("/patch_status")
def patch_status():
    """patchceil: rollout 종료 후 러너가 실제 발화 창을 기대와 대조 (무음 오적용 방지)."""
    if not _patch_hooks:
        raise HTTPException(status_code=409, detail="patch hooks not enabled (--patch-layers)")
    return {
        "boot_id": _BOOT_ID,
        "patch": dict(_patch_spec),
        "hooks": {str(layer): hook.status() for layer, hook in _patch_hooks.items()},
    }


@app.post("/reset")
async def reset():
    # patchceil: 에피소드 경계 — record cursor·발화 로그만 초기화, arm 은 유지
    # (러너의 /patch_arm → collector 기동(내부 /reset) 순서 때문).
    for hook in _patch_hooks.values():
        hook.reset_episode()
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

    if _has_per_step_steering():
        # groot select_action 은 16-큐 팝 — 추론이 매 콜 발생하지 않아 per-step M_k
        # 스와핑과 양립 불가 (무음 오적용 방지, Gate 2 치명#1/높음#1).
        raise HTTPException(
            status_code=409,
            detail="per-step steering serve 는 /act(큐 팝) 미지원 — "
            "/act_with_features (skip_features=1) 를 사용하라",
        )
    if _patch_hooks:
        # record cursor 는 "요청 1개 = record 1개" 를 전제 — /act 큐 팝(16콜당 1추론)과
        # 양립 불가 (무음 커서 어긋남 방지).
        raise HTTPException(
            status_code=409,
            detail="patch serve 는 /act(큐 팝) 미지원 — "
            "/act_with_features (skip_features=1) 를 사용하라",
        )
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

    if payload.get("skip_features"):
        if _collect_mode:
            # 수집 serve 에서 hook 없는 첫 compile 이 캐시되면 이후 캡처가 무음 미발화
            # (/act 거부와 같은 이유 — Gate 2 R2 중간#4)
            raise HTTPException(
                status_code=409,
                detail="collect mode 에서 skip_features 금지 (hook 없는 compile 오염)",
            )
        # exp3 eval 캡처-OFF: hook 없이 **캡처 경로와 동일한 chunk 추론 단위**를 사용
        # (groot select_action 은 16-큐 팝이라 16콜당 1회만 추론 — noise pairing·실행
        # 단위가 캡처 경로와 어긋남, Gate 2 치명#1). predict_action_chunk 를 직접 호출.
        with torch.inference_mode():
            if _policy_type == "groot" and hasattr(policy, "predict_action_chunk"):
                action = policy.predict_action_chunk(batch)
            else:
                action = policy.select_action(batch)
        _assert_per_step_hook_counts()
        _assert_patch_hook_counts()
        action = _postprocess_action_preserve_chunk(action)
        action_np = _action_to_emit_array(action)
        result = _emit_subkeys(action_np, profile)
        result["has_feature"] = False
        result["skip_features"] = True
        if inference_seed is not None:
            result["inference_seed"] = inference_seed
        result["latency_ms"] = (time.time() - t0) * 1000
        return result

    action, hidden, _axes, meta = safe_hooks.run_with_features(
        policy,
        batch,
        _policy_type,
        capture_vl=_capture_vl_features,
        groot_dit_layers=_groot_dit_capture_layers,
        pi05_expert_layers=_pi05_expert_capture_layers,
        groot_dit_token_pool=_groot_dit_token_pool,
        vl_capture_point=_groot_vl_capture_point,
        cross_attn_capture=_cross_attn_capture,
    )
    if _policy_type == "groot":
        _assert_per_step_hook_counts()
    # ndarray 는 JSON 직렬화 불가 — meta 에서 분리해 feature blob 으로 동봉.
    cross_attn_arr = meta.pop("cross_attn", None)
    cross_attn_maps_arr = meta.pop("cross_attn_maps", None)

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
    if cross_attn_arr is not None:
        result["features.cross_attn"] = encode_feature_blob(np.asarray(cross_attn_arr))
        if cross_attn_maps_arr is not None:
            result["features.cross_attn_maps"] = encode_feature_blob(
                np.asarray(cross_attn_maps_arr)
            )
        if hidden is None:
            # hidden 없는 경로에서도 cross-attn 축/그룹 메타는 동봉 (groot 는 비발생).
            result.update({k: v for k, v in meta.items() if k.startswith("cross_attn") or k == "view_token_spans"})
    if inference_seed is not None:
        result["inference_seed"] = inference_seed
    result["latency_ms"] = (time.time() - t0) * 1000
    return result


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "lerobot", "boot_id": _BOOT_ID}
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
        # 러너 preflight: 로그의 [serve-boot] id 와 대조해 "포트의 남의 서버" 오인 방지
        boot_id=_BOOT_ID,
        steering=_steering_spec or None,
        patch=_patch_spec or None,
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
            # wire dtype: assemble 이 fp16 으로 내보냄 — full 모드는 광고도 fp16 으로
            # (구 모드 float32 광고는 legacy 계약 유지, Gate 2 높음#3)
            "feature_dtype": "float16" if _full else "float32",
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
    if _policy_type == "groot" and _cross_attn_capture is not None:
        metadata.update(
            {
                "capture_cross_attn": True,
                "cross_attn_blocks": [
                    int(i) for i in _cross_attn_capture.cross_block_indices
                ],
                "cross_attn_qgroups": list(_cross_attn_capture.qgroups),
                "cross_attn_kgroups": list(_cross_attn_capture.kgroups),
                "cross_attn_keep_heads": bool(_cross_attn_capture.keep_heads),
                "cross_attn_full_maps": bool(_cross_attn_capture.full_maps),
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
    # exp3: token_select 는 default None(pathway 기본 보존 — dit=last_horizon, vl=all),
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

    loaded_npz_shas: list[str] = []

    def _load_matrices(npz_path):
        """denoise 모드에 맞는 M(단일) 또는 M_seq(list) 로드 + preflight 로그 + sha 수집."""
        import hashlib as _hashlib

        loaded_npz_shas.append(
            _hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()[:12]
        )
        if per_step:
            return load_steering_matrices_per_step(
                str(npz_path), beta=beta, alpha=alpha, key=key, num_steps=expected_steps
            )
        return load_steering_matrix(str(npz_path), beta=beta, alpha=alpha, key=key)

    def _set_steering_spec(mode: str, layers, phases=None):
        """/health 노출용 스티어링 지문 (러너가 프로세스·설정 오인 방지에 사용)."""
        global _steering_spec
        _steering_spec = {
            "mode": mode,
            "layers": [int(x) for x in layers] if layers else [],
            "beta": float(beta),
            "alpha": None if alpha is None else float(alpha),
            "key": key,
            "token_select": token_select,
            "denoise": denoise,
            "npz_shas": sorted(set(loaded_npz_shas)),
            "phases": sorted(phases) if phases else None,
        }

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
        # 기대 phase 목록이 주어지면 발견 집합과 정확히 일치해야 함 (부분 로드 무음 방지)
        expected_phases = getattr(args, "steering_phases", None)
        if expected_phases:
            want = sorted(p.strip() for p in str(expected_phases).split(",") if p.strip())
            if want != phases:
                raise ValueError(
                    f"--steering-phases 불일치: 기대 {want} != 발견 {phases} ({base})"
                )
        hooks, matrices, identity = {}, {}, {}
        for lyr in layers:
            matrices[lyr] = {}
            for ph in phases:
                npz_path = base / ph / f"dit_L{lyr}" / "conceptors.npz"
                if not npz_path.exists():
                    # layer×phase Cartesian 완전성 강제 — 일부 layer 만 조향되는
                    # 부분-gated arm 이 정상 등록되는 사고 방지 (Gate 2 치명#2)
                    raise FileNotFoundError(
                        f"gated NPZ 누락: layer {lyr} 에 phase '{ph}' 없음 ({npz_path}) — "
                        f"phase 집합 {phases} 은 전 layer 에 존재해야 한다"
                    )
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
        _set_steering_spec("gated", layers, phases)
        # 러너 preflight 대조용 (module logger 는 serve 로그 파일에 안 남음 — print 필수)
        print(
            f"[steer-registered] path=gated layers={','.join(str(x) for x in layers)} "
            f"phases={','.join(phases)} beta={beta:g} "
            f"token_select={token_select or 'last_horizon(default)'} denoise={denoise}",
            flush=True,
        )
        # dose 로깅 근거 (Gate 2 높음#10): phase×step 별 ‖M−I‖F — 사이드카의
        # feature_phases + phase_gated_flags 와 조합해 오프라인에서 누적 dose 재구성.
        for lyr in layers:
            for ph in phases:
                mats = matrices[lyr][ph]
                seq = mats if isinstance(mats, list) else [mats]
                for k, M in enumerate(seq):
                    dI = float(_np.linalg.norm(M - _np.eye(M.shape[0]), "fro"))
                    print(f"[steer-norms] layer={lyr} phase={ph} step={k} fro_M_minus_I={dI:.6f}",
                          flush=True)
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
        _set_steering_spec("multi", layers)
        print(
            f"[steer-registered] path=multi layers={','.join(str(x) for x in layers)} "
            f"beta={beta:g} key={key} "
            f"token_select={token_select or 'last_horizon(default)'} denoise={denoise}",
            flush=True,
        )
        import numpy as _np2
        for hook in _steering:
            for k, M in enumerate(getattr(hook, "_M_seq", []) or []):
                dI = float(_np2.linalg.norm(M - _np2.eye(M.shape[0]), "fro"))
                print(f"[steer-norms] layer={hook.layer} step={k} fro_M_minus_I={dI:.6f}",
                      flush=True)
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
    _set_steering_spec("single", [] if layer is None else [layer])
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
    _register_patching_if_requested(policy, args)

    global _cross_attn_capture
    if getattr(args, "capture_cross_attn", False):
        if _policy_type != "groot":
            raise ValueError("--capture-cross-attn requires policy_type='groot'")
        import attn_hooks

        _cross_attn_capture = attn_hooks.CrossAttnCapture(
            policy,
            keep_heads=bool(getattr(args, "cross_attn_keep_heads", False)),
            full_maps=bool(getattr(args, "cross_attn_full_maps", False)),
        ).install()
        # 러너 preflight 대조용 지문 한 줄.
        print(
            f"[attn-preflight] cross_blocks={_cross_attn_capture.cross_block_indices} "
            f"image_token_index={_cross_attn_capture.image_token_index} "
            f"qgroups={_cross_attn_capture.qgroups} kgroups={_cross_attn_capture.kgroups} "
            f"keep_heads={_cross_attn_capture.keep_heads} "
            f"full_maps={_cross_attn_capture.full_maps}",
            flush=True,
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


def _register_patching_if_requested(loaded_policy, args):
    """patchceil donor-trajectory transplant hook 등록 (--patch-layers).

    conceptor steering 과 달리 M 변환이 아니라 donor activation 대입이며, rollout record
    cursor 상태를 가진다 (patching_hooks.PatchSteering). patch rollout 은 캡처 OFF 가
    표준이라 --collect 와 동시 사용을 금지하고, 해석 오염 방지를 위해 --steering-* 와도
    상호 배타다.
    """
    global _patch_hooks, _patch_spec
    patch_layers = getattr(args, "patch_layers", None)
    patch_npz = getattr(args, "patch_npz", None)
    patch_pathway = getattr(args, "patch_pathway", "dit") or "dit"
    if patch_pathway == "vl" and patch_layers:
        raise ValueError(
            "--patch-pathway vl 은 --patch-layers 와 상호 배타 "
            "(vl 주입은 vl_self_attention 단일 지점 — layer 개념 없음)"
        )
    if not patch_layers and patch_pathway != "vl":
        if patch_npz:
            raise ValueError("--patch-npz 는 --patch-layers 와 함께 지정해야 한다")
        return None
    if _policy_type != "groot":
        raise ValueError("patch hook 은 groot (GR00T N1.5) 전용")
    if _collect_mode and not getattr(args, "patch_allow_collect", False):
        raise ValueError(
            "patch hook 은 --collect 와 동시 사용 금지 — patch rollout 은 캡처 OFF "
            "(/act_with_features skip_features=1) 표준. anchor(A2/A3) 검증처럼 emitted "
            "actions 저장이 필요한 경우에만 --patch-allow-collect 로 명시 허용."
        )
    if _steering or _gated_registry:
        raise ValueError("patch hook 은 --steering-* 와 동시 사용 금지 (해석 오염)")

    from patching_hooks import PatchSteering, PatchSteeringVL, load_donor_npz, load_vl_donor_npz

    _gm = getattr(loaded_policy, "_groot_model", None)
    if _gm is None:
        raise ValueError("GR00T LeRobot policy is missing _groot_model for patching")
    expected_k = int(_gm.action_head.num_inference_timesteps)
    token_select = getattr(args, "patch_token_select", "all") or "all"

    for _h in _patch_hooks.values():
        _h.unregister()
    _patch_hooks = {}
    if patch_pathway == "vl":
        layers = ["VL"]
        _patch_hooks["VL"] = PatchSteeringVL(_gm).register()
        _patch_spec = {
            "mode": "transplant",
            "pathway": "vl",
            "layers": layers,
            "token_select": "all",
            "expected_fires": 1,
            "armed_tag": None,
            "donor_npz_sha": None,
        }
    else:
        layers = [int(x) for x in str(patch_layers).split(",") if x.strip() != ""]
        if not layers:
            raise ValueError("--patch-layers 가 비어 있다")
        for layer in layers:
            hook = PatchSteering(
                _gm, layer=layer, expected_k=expected_k, token_select=token_select
            ).register()
            _patch_hooks[layer] = hook
        _patch_spec = {
            "mode": "transplant",
            "pathway": "dit",
            "layers": layers,
            "token_select": token_select,
            "expected_k": expected_k,
            "armed_tag": None,
            "donor_npz_sha": None,
        }

    # 정적 arm (스모크·anchor 용): --patch-npz + --patch-start-record 지정 시 기동 즉시 arm.
    # 본 실행은 rollout 마다 /patch_arm 으로 동적 arm 한다.
    if patch_npz:
        if patch_pathway == "vl":
            vl_arr, meta, sha12 = load_vl_donor_npz(patch_npz)
            arrays = {"VL": vl_arr}
        else:
            arrays, meta, sha12 = load_donor_npz(patch_npz, layers, expected_k=expected_k)
        _patch_donor_arrays.clear()
        _patch_donor_arrays.update(arrays)
        _patch_spec["donor_npz_sha"] = sha12
        _patch_spec["donor_meta"] = {
            k: meta.get(k)
            for k in ("cell", "episode_idx", "scenario_seed", "inference_seed", "n_records")
        }
        start = getattr(args, "patch_start_record", None)
        if start is not None:
            for layer in layers:
                _patch_hooks[layer].arm(
                    _patch_donor_arrays[layer],
                    start_record=int(start),
                    donor_start=int(getattr(args, "patch_donor_start", 0) or 0),
                    patch_len=int(getattr(args, "patch_len", -1)),
                    tag="static",
                )
            _patch_spec.update(
                {
                    "armed_tag": "static",
                    "start_record": int(start),
                    "donor_start": int(getattr(args, "patch_donor_start", 0) or 0),
                    "patch_len": int(getattr(args, "patch_len", -1)),
                }
            )
    logger.info(
        "[patch-preflight] pathway=%s layers=%s token_select=%s K=%d npz=%s sha=%s armed=%s",
        patch_pathway, layers, token_select, expected_k, patch_npz,
        _patch_spec.get("donor_npz_sha"), _patch_spec.get("armed_tag"),
    )
    return _patch_hooks


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
        "--capture-cross-attn",
        action="store_true",
        help=(
            "GR00T N1.5 /act_with_features 에서 DiT cross-attention 의 카메라 뷰별 "
            "attention mass(features.cross_attn, [n_cross_blocks, K, qgroup, kgroup])를 "
            "함께 반환한다. kgroup=(text,left,right,wrist). 출력 action 은 불변."
        ),
    )
    parser.add_argument(
        "--cross-attn-full-maps",
        action="store_true",
        help="cross-attn 원맵 [T_q, S](head-mean, fp16)도 반환 (smoke/정성 확인 전용 — 용량 큼).",
    )
    parser.add_argument(
        "--cross-attn-keep-heads",
        action="store_true",
        help="cross-attn mass 의 head 축 보존 ([n_blocks, K, heads, qgroup, kgroup]).",
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
            "GR00T DiT block residual 캡처의 token 풀링 (exp3 COAST 토큰 축 정렬). "
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
            "(dit=last_horizon, vl=all). exp3 COAST 정렬은 dit 에 all 을 명시 주입."
        ),
    )
    parser.add_argument(
        "--steering-phases",
        default=None,
        help=(
            "gated 전용: 기대 phase 목록(콤마). 지정 시 NPZ 디렉토리에서 발견된 phase "
            "집합과 정확히 일치하지 않으면 기동 abort (부분 gated arm 무음 방지)."
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
    parser.add_argument(
        "--patch-layers",
        default=None,
        help=(
            "patchceil transplant 주입 layer (콤마, DiT transformer_block idx). 지정 시 "
            "donor-trajectory transplant hook 등록 — --steering-*/--collect 와 상호 배타. "
            "rollout 별 창은 /patch_arm 으로 동적 설정."
        ),
    )
    parser.add_argument(
        "--patch-pathway",
        choices=("dit", "vl"),
        default="dit",
        help=(
            "transplant 주입 pathway (exp4-2). dit=기존 --patch-layers 경로 | "
            "vl=action_head.vl_self_attention 출력 통째 교체 (B1 — donor NPZ 키 VL="
            "[R,T_vl,D], 요청당 1 fire, --patch-layers 와 상호 배타)."
        ),
    )
    parser.add_argument(
        "--patch-npz",
        default=None,
        help="donor NPZ 경로 (키 L{layer}=[R,K,T,D] fp16 + meta_json). 기동 시 preload.",
    )
    parser.add_argument(
        "--patch-token-select",
        choices=("all", "action"),
        default="all",
        help="대입 토큰: all=전 토큰(기본, full-token donor 필수) | action=마지막 horizon 개.",
    )
    parser.add_argument(
        "--patch-start-record",
        type=int,
        default=None,
        help="정적 arm 스모크용 t0 (record idx). 본 실행은 /patch_arm 사용.",
    )
    parser.add_argument("--patch-donor-start", type=int, default=0)
    parser.add_argument(
        "--patch-allow-collect",
        action="store_true",
        help="anchor(A2/A3) 전용: --collect 캡처 serve 에 patch hook 동시 허용 "
        "(emitted actions 를 pkl 로 남겨 donor/baseline 과 수치 대조).",
    )
    parser.add_argument(
        "--patch-len",
        type=int,
        default=-1,
        help="패치 창 길이 (records). -1=donor 고갈까지 (고갈 후 합성 없음 — 기록만).",
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
    # 프로세스 지문 — 러너가 fresh 로그의 이 라인과 /health boot_id 를 대조 (치명#3 가드)
    print(f"[serve-boot] id={_BOOT_ID} port={args.port}", flush=True)

    app.state.args = args
    run_uvicorn(app, args)


if __name__ == "__main__":
    main()
