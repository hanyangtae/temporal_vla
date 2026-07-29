"""GR00T N1.5 (+TTT) 추론 서버 — port 8510, 통일 API.

N1.6 serve (``scripts/serve/groot.py``) 와 분리한 이유:
  - N1.5 ``Gr00tPolicy`` 생성자 시그니처가 다름 — modality_config + modality_transform
    직접 주입 필수 (N1.6 의 ``load_groot_policy`` 헬퍼 사용 불가).
  - N1.5 ``EmbodimentTag`` enum 에 ``ROBOCASA_PANDA_OMRON`` 없음 → ``new_embodiment``.
  - TTT attach: ``attach_ttt_to_n1d5`` (``groot_wrapper_n1d5.py``).
  - VideoResize 224x224 (N1.6 256 과 다름).

TTT 없는 vanilla N1.5 서빙도 동일 스크립트 — profile.model_specific.predictor_path
가 비어있으면 attach 단계를 건너뜀.

실행 (groot_serve_n1d5 컨테이너):
  docker exec groot_serve_n1d5 \\
    python /temporal_vla/scripts/serve/groot_n1d5_ttt.py \\
    --profile /temporal_vla/configs/checkpoints/groot_n1d5_ttt__robocasa_target15.yaml \\
    --port 8510

통일 API:
  POST /act    ← observation sub-keys + task
              → action sub-keys (eef_pos, eef_axisangle, gripper, base_motion, control_mode)
  POST /reset  ← 에피소드 시작 시 policy state 초기화 (TTT inner-loop reset 포함)
  GET  /health ← profile + 모델 modality_configs
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI

# scripts/utils 가 PYTHONPATH 에 있어야 함 (CheckpointProfile 로더).
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

# N1.5 codebase + configs/policies + repo root 경로 보장.
for _p in (
    "/temporal_vla/src/policies/Isaac-GR00T-N1.5",
    "/temporal_vla/configs/policies",
    "/temporal_vla",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# pytorch3d stub — N1.5 state_action transform 의 import 만 통과시키기 위한
# 더미. 실제 회전 변환은 N1.5 transform pipeline 에서 호출되지 않음 (RoboCasa).
# launch_finetune_ttt_n1d5.py 와 동일 패턴.
import types as _types  # noqa: E402
if "pytorch3d" not in sys.modules:
    _pt = _types.ModuleType("pytorch3d")
    _pt.transforms = _types.ModuleType("pytorch3d.transforms")
    for _name in (
        "matrix_to_quaternion", "quaternion_to_matrix",
        "matrix_to_euler_angles", "euler_angles_to_matrix",
        "matrix_to_rotation_6d", "rotation_6d_to_matrix",
        "quaternion_apply", "axis_angle_to_quaternion",
    ):
        setattr(_pt.transforms, _name, lambda *a, **k: None)
    sys.modules["pytorch3d"] = _pt
    sys.modules["pytorch3d.transforms"] = _pt.transforms

# 통일 API ↔ GR00T modality 매핑 (N1.6 helper 재사용 — state/action 키 동일).
from src.policies.groot.preprocess import decode_b64_image  # noqa: E402
from src.policies.groot.schema import (  # noqa: E402
    GROOT_TO_UNIFIED_ACTION,
    UNIFIED_TO_STATE_KEY,
    build_video_mapping,
    normalize_modality_key,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="GR00T N1.5 (+TTT) Inference Server")

# ─── 글로벌 ───────────────────────────────────────────────────────────────────

_policy = None                       # gr00t.model.policy.Gr00tPolicy (N1.5)
_modality_configs = None
_profile: Optional[CheckpointProfile] = None
_device = "cuda"

_warned_missing_video_keys: set = set()
_warned_missing_state_keys: set = set()
_state_dims: dict = {}
_video_key_mapping: dict = {}

# N1.5 VideoResize 224x224 (configs/policies/robocasa_n15_panda_omron_data_config.py).
_IMAGE_H = 224
_IMAGE_W = 224


def _resize_img(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 → 224x224x3 uint8 (square pad + resize).

    N1.5 의 modality_transform 이 다시 224 로 resize 하므로 idempotent.
    serve 가 224 로 미리 만들면 라운드트립 비용 절약.
    """
    import cv2
    h, w = img.shape[:2]
    if h != w:
        dim = max(h, w)
        y_off = (dim - h) // 2
        x_off = (dim - w) // 2
        img = np.pad(img, ((y_off, y_off), (x_off, x_off), (0, 0)))
    if img.shape[:2] != (_IMAGE_H, _IMAGE_W):
        img = cv2.resize(img, (_IMAGE_W, _IMAGE_H), cv2.INTER_AREA)
    return np.copy(img)


def _build_n1d5_obs(payload: dict) -> dict:
    """통일 API payload → N1.5 Gr00tPolicy.get_action 입력 형식.

    video:    np.ndarray(B=1, T=1, H=224, W=224, C=3) uint8
    state:    np.ndarray(B=1, T=1, D) float32
    annotation: tuple[str] (B,)
    """
    obs = {}

    # 이미지
    for unified_key, groot_key in _video_key_mapping.items():
        b64 = payload.get(unified_key)
        if b64 is None or groot_key in obs:
            continue
        np_img = _resize_img(decode_b64_image(b64))
        obs[groot_key] = np_img[np.newaxis, np.newaxis, ...]   # (1,1,H,W,3)

    # state — N1.6 helper 의 UNIFIED_TO_STATE_KEY 가 N1.5 key 와 동일
    # ('state.end_effector_position_relative' 등) 이라 그대로 재사용.
    for unified_key, groot_key in UNIFIED_TO_STATE_KEY.items():
        val = payload.get(unified_key)
        if val is not None:
            arr = np.array(val, dtype=np.float32).flatten()
            obs[groot_key] = arr[np.newaxis, np.newaxis, :]

    # modality_configs 의 필수 키 중 누락분 zero fallback (transform strict assert 통과).
    if _modality_configs is not None:
        zero_img = np.zeros((1, 1, _IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)

        for vk_raw in _modality_configs["video"].modality_keys:
            vk = normalize_modality_key(vk_raw, "video")
            if vk not in obs:
                obs[vk] = zero_img
                if vk not in _warned_missing_video_keys:
                    logger.warning("video key %s missing → zero image fallback", vk)
                    _warned_missing_video_keys.add(vk)

        for sk_raw in _modality_configs["state"].modality_keys:
            sk = normalize_modality_key(sk_raw, "state")
            if sk not in obs:
                local = sk[len("state."):]
                dim = _state_dims.get(local, 1)
                obs[sk] = np.zeros((1, 1, dim), dtype=np.float32)
                if sk not in _warned_missing_state_keys:
                    logger.warning("state key %s missing → zero(%dD) fallback", sk, dim)
                    _warned_missing_state_keys.add(sk)

    # N1.5 language key.
    obs["annotation.human.task_description"] = (payload.get("task", ""),)
    return obs


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
    global _policy, _modality_configs, _device, _state_dims, _video_key_mapping

    args = app.state.args
    profile = _profile
    assert profile is not None

    ms = profile.model_specific or {}
    _device = ms.get("device", args.device)
    model_path = profile.checkpoint_source.id
    embodiment_tag_str = ms.get("embodiment_tag", "new_embodiment")
    data_config_spec = ms.get(
        "data_config",
        "robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig",
    )

    # N1.5 codebase import (sys.path 이미 보장됨).
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.policy import Gr00tPolicy

    embodiment_tag = EmbodimentTag(embodiment_tag_str)

    # data_config 모듈:Class 에서 modality_config + transforms 로드.
    mod_name, cls_name = data_config_spec.split(":", 1)
    data_config_cls = getattr(importlib.import_module(mod_name), cls_name)()
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    logger.info(
        "[n1d5] loading from %s (embodiment=%s, device=%s)",
        model_path, embodiment_tag_str, _device,
    )

    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_configs,
        modality_transform=transforms,
        device=_device,
    )

    # TTT attach 분기.
    predictor_path = ms.get("predictor_path", "")
    if predictor_path:
        from src.ttt.integrations.groot_wrapper_n1d5 import attach_ttt_to_n1d5

        attach_ttt_to_n1d5(
            policy.model,                         # GR00T_N1_5 instance
            predictor_input_dim=int(ms.get("predictor_input_dim", 2048)),
            predictor_proj_dim=int(ms.get("predictor_proj_dim", 1536)),
            predictor_inner_model=ms.get("predictor_inner_model", "linear"),
            predictor_eta_base=float(ms.get("predictor_eta_base", 0.1)),
            predictor_state_path=predictor_path,
            ttt_update_in_train=False,
            ttt_update_at_inference=bool(ms.get("ttt_update_at_inference", True)),
        )
        logger.info("[ttt] N1.5 model → TTT-augmented (predictor=%s)", predictor_path)
    else:
        logger.info("[n1d5] no predictor_path → vanilla N1.5 (no TTT)")

    _policy = policy
    _modality_configs = modality_configs

    # state dim 추출 (zero fallback 크기 결정).
    _state_dims = _extract_state_dims(modality_configs, model_path, embodiment_tag.value)

    # video key 매핑 동적 구성 (N1.5 의 robot0_agentview_* 가 left/right 로 classify).
    video_modality_keys = list(modality_configs["video"].modality_keys)
    _video_key_mapping = build_video_mapping(video_modality_keys)
    logger.info("[video] modality_keys=%s → mapping=%s",
                video_modality_keys, _video_key_mapping)

    horizon = len(modality_configs["action"].delta_indices)
    logger.info("[n1d5] loaded. action_horizon=%d", horizon)


def _extract_state_dims(modality_configs, model_path: str, embodiment_value: str) -> dict:
    """experiment_cfg/metadata.json 에서 embodiment 별 state key dim 추출."""
    import json
    import os
    dims = {}
    meta_path = os.path.join(model_path, "experiment_cfg", "metadata.json")
    if not os.path.exists(meta_path):
        logger.warning("metadata.json not found at %s", meta_path)
        return dims
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        emb = meta.get(embodiment_value, {})
        stats = emb.get("statistics", {}).get("state", {})
        for k, v in stats.items():
            if isinstance(v, dict) and "mean" in v:
                mean = v["mean"]
                if hasattr(mean, "__len__"):
                    dims[k] = len(mean)
    except Exception as e:
        logger.warning("failed to parse metadata.json: %s", e)
    return dims


# ─── FastAPI 엔드포인트 ───────────────────────────────────────────────────────


@app.post("/reset")
async def reset():
    if _policy is not None:
        # N1.5 Gr00tPolicy 에 reset() 없음. model 에 reset_predictor 가 있으면 호출.
        model = getattr(_policy, "model", None)
        if model is not None and hasattr(model, "reset_predictor"):
            model.reset_predictor()
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    if _policy is None:
        return {"error": "model not loaded"}

    t0 = time.time()
    profile = _profile
    assert profile is not None

    n1d5_obs = _build_n1d5_obs(payload)
    action_dict = _policy.get_action(n1d5_obs)
    latency_ms = (time.time() - t0) * 1000

    out = {"latency_ms": latency_ms}

    # action_dict 의 key 는 "action.end_effector_position" 같은 형태.
    for k_raw, arr in action_dict.items():
        k = k_raw[len("action."):] if k_raw.startswith("action.") else k_raw
        unified = GROOT_TO_UNIFIED_ACTION.get(k)
        if unified is None:
            unified = f"action.{k}"
        if hasattr(arr, "ndim"):
            arr_np = np.asarray(arr)
            if arr_np.ndim == 3:
                arr_np = arr_np[0]    # (B, T, D) → (T, D)
            elif arr_np.ndim == 1:
                arr_np = arr_np[np.newaxis, :]
            elif arr_np.ndim == 0:
                arr_np = arr_np.reshape(1, 1)
            out[unified] = arr_np.tolist()
        else:
            out[unified] = list(arr)

    # profile emits_subkeys 중 누락 키 zero 패딩 (디버깅 명확성).
    horizon = len(_modality_configs["action"].delta_indices) if _modality_configs else 1
    for sk in profile.emits_subkeys:
        if sk not in out:
            local = sk[len("action."):]
            if local in {a.name for a in profile.action_layout}:
                dim = profile.dim_slice(local).stop - profile.dim_slice(local).start
            else:
                dim = 1
            out[sk] = [[0.0] * dim for _ in range(horizon)]
            logger.warning("emit sub-key %s missing from N1.5 output → filled zeros", sk)

    # 디버그 (첫 5 호출).
    step_count = getattr(app.state, "_step", 0) + 1
    app.state._step = step_count
    if step_count <= 5:
        logger.info(
            "call=%d horizon=%d pos=%s grip=%s task=%r",
            step_count, horizon,
            np.array(out.get("action.eef_pos", [[0] * 3]))[0].round(4).tolist(),
            np.array(out.get("action.gripper", [[0]]))[0].round(4).tolist(),
            payload.get("task", "")[:40],
        )

    return out


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "groot-n1.5"}
    horizon = (
        len(_modality_configs["action"].delta_indices)
        if _modality_configs is not None
        else _profile.n_action_steps
    )
    ms = _profile.model_specific or {}
    has_ttt = bool(ms.get("predictor_path", ""))
    return {
        "status": "ok" if _policy is not None else "not_loaded",
        "model": "groot-n1.5-ttt" if has_ttt else "groot-n1.5",
        "profile": _profile.name,
        "n_action_steps": horizon,
        "action_type": _profile.action_type,
        "action_keys": list(_profile.emits_subkeys),
        "ttt_enabled": has_ttt,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    global _profile

    try:
        from src.utils.common.logger import create_module_logger
        create_module_logger("groot_n1d5_serve")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GR00T N1.5 (+TTT) 추론 서버 (port 8510)")
    parser.add_argument("--profile", type=str, required=True,
                        help="체크포인트 프로파일 YAML (configs/checkpoints/*.yaml)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8510)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _profile = load_profile(args.profile)
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "groot_n1d5_ttt", (
        f"profile.base_model={_profile.base_model!r}, expected 'groot_n1d5_ttt'"
    )

    app.state.args = args
    app.state._step = 0
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
