"""
GR00T N1.6 추론 서버 (통일 API, port 8500).

프로파일 기반 동작: embodiment_tag, device 등은 configs/checkpoints/*.yaml 에 선언.

groot 컨테이너에서 실행:
  docker compose run --rm groot \
    python /temporal_vla/scripts/serve/groot.py \
    --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml

통일 API:
  POST /act     ← {"observation.images.*": b64png, "observation.state.*": [...],
                    "task": "..."}
                → 프로파일 emits_subkeys 규약에 따른 sub-key dict 반환
  POST /reset   ← 에피소드 시작 시 policy state 초기화
  GET  /health  ← 프로파일 + 모델 modality_configs 반영
"""

import argparse
import logging
import sys
import time
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

# 학습 adapter 와 공유하는 GR00T helper (src/policies/groot/).
sys.path.insert(0, "/temporal_vla")
from src.policies.groot.preprocess import (  # noqa: E402
    FINAL_IMAGE_RESOLUTION,
    decode_b64_image,
    process_img,
)
from src.policies.groot.schema import (  # noqa: E402
    GROOT_TO_UNIFIED_ACTION,
    UNIFIED_TO_STATE_KEY,
    UNIFIED_TO_VIDEO_KEY,
    normalize_modality_key,
)
from src.policies.groot.loader import load_groot_policy  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="GR00T N1.6 Inference Server")

# ─── 글로벌 ──────────────────────────────────────────────────────────────────

_policy = None
_embodiment_tag = None
_modality_configs = None
_profile: Optional[CheckpointProfile] = None
_device = "cuda"

_warned_missing_video_keys: set = set()
_warned_missing_state_keys: set = set()
# embodiment 별 state key → dim. statistics.json 에서 모델 로드 시 채워짐.
_state_dims: dict = {}


def _build_groot_obs(payload: dict) -> dict:
    """통일 API payload → Gr00tSimPolicyWrapper 입력 형식.

    video:    np.ndarray(B=1, T=1, H, W, C) uint8
    state:    np.ndarray(B=1, T=1, D) float32
    language: tuple[str] (B,)

    GR00T 가 modality_configs 로 요구하는 video/state 키가 payload 에 없으면
    zero image / zero state 로 fallback (strict assert 통과용). 첫 누락 시 warning.
    """
    obs = {}

    for unified_key, groot_key in UNIFIED_TO_VIDEO_KEY.items():
        b64 = payload.get(unified_key)
        if b64 is None or groot_key in obs:
            continue
        np_img = process_img(decode_b64_image(b64))
        obs[groot_key] = np_img[np.newaxis, np.newaxis, ...]

    for unified_key, groot_key in UNIFIED_TO_STATE_KEY.items():
        val = payload.get(unified_key)
        if val is not None:
            arr = np.array(val, dtype=np.float32).flatten()
            obs[groot_key] = arr[np.newaxis, np.newaxis, :]

    # modality_configs 에 선언된 필수 video/state 키 중 누락 분 zero fallback.
    if _modality_configs is not None:
        h, w = FINAL_IMAGE_RESOLUTION
        zero_img = np.zeros((1, 1, h, w, 3), dtype=np.uint8)

        for vk_raw in _modality_configs["video"].modality_keys:
            vk = normalize_modality_key(vk_raw, "video")
            if vk not in obs:
                obs[vk] = zero_img
                if vk not in _warned_missing_video_keys:
                    logger.warning(
                        "video key %s missing → zero image fallback", vk,
                    )
                    _warned_missing_video_keys.add(vk)

        for sk_raw in _modality_configs["state"].modality_keys:
            sk = normalize_modality_key(sk_raw, "state")
            if sk not in obs:
                # state dim 은 statistics.json 에서 가져옴, 없으면 1D
                local = sk[len("state."):]
                dim = _state_dims.get(local, 1)
                obs[sk] = np.zeros((1, 1, dim), dtype=np.float32)
                if sk not in _warned_missing_state_keys:
                    logger.warning(
                        "state key %s missing → zero(%dD) fallback", sk, dim,
                    )
                    _warned_missing_state_keys.add(sk)

    obs["annotation.human.action.task_description"] = (payload.get("task", ""),)
    return obs


# ─── 모델 로딩 ───────────────────────────────────────────────────────────────


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
    global _policy, _embodiment_tag, _modality_configs, _device, _state_dims

    args = app.state.args
    profile = _profile
    assert profile is not None, "profile must be set before load_model"

    loaded = load_groot_policy(profile, device=args.device)
    _policy = loaded.policy
    _embodiment_tag = loaded.embodiment_tag
    _modality_configs = loaded.modality_configs
    _state_dims = loaded.state_dims
    _device = loaded.device


# ─── FastAPI 엔드포인트 ──────────────────────────────────────────────────────


@app.post("/reset")
async def reset():
    if _policy is not None:
        _policy.reset()
    return {"status": "reset"}


@app.post("/act")
async def predict_action(payload: dict):
    if _policy is None:
        return {"error": "model not loaded"}

    t0 = time.time()
    profile = _profile
    assert profile is not None

    groot_obs = _build_groot_obs(payload)
    action_dict, info = _policy.get_action(groot_obs)
    latency_ms = (time.time() - t0) * 1000

    # GR00T native action dict → 통일 sub-key dict.
    # action_dict 의 key 는 "end_effector_position" 또는 "action.end_effector_position" 형태로 들어올 수 있음.
    # 이름 정규화 후 GROOT_TO_UNIFIED_ACTION 으로 매핑.
    out = {"latency_ms": latency_ms}
    for k_raw, arr in action_dict.items():
        k = k_raw[len("action."):] if k_raw.startswith("action.") else k_raw
        unified = GROOT_TO_UNIFIED_ACTION.get(k)
        if unified is None:
            # 알 수 없는 키 → action. 접두사 유지하여 그대로 보존
            unified = "action.{}".format(k)
        if arr.ndim == 3:
            arr = arr[0]   # (B, T, D) → (T, D)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 0:
            arr = arr.reshape(1, 1)
        out[unified] = arr.tolist()

    # 프로파일이 emits_subkeys 로 선언한 키가 빠졌으면 빈 array 라도 채워주기 (디버깅 명확성)
    horizon = len(_modality_configs["action"].delta_indices) if _modality_configs else 1
    for sk in profile.emits_subkeys:
        if sk not in out:
            local = sk[len("action."):]
            if local in {a.name for a in profile.action_layout}:
                dim = profile.dim_slice(local).stop - profile.dim_slice(local).start
            else:
                dim = 1
            out[sk] = [[0.0] * dim for _ in range(horizon)]
            logger.warning("emit sub-key %s missing from GR00T output, filled zeros", sk)

    # 디버그 (첫 5 호출)
    step_count = getattr(app.state, "_step", 0) + 1
    app.state._step = step_count
    if step_count <= 5:
        logger.info(
            "call=%d horizon=%d pos=%s grip=%s task=%r",
            step_count, horizon,
            np.array(out.get("action.eef_pos", [[0]*3]))[0].round(4).tolist(),
            np.array(out.get("action.gripper", [[0]]))[0].round(4).tolist(),
            payload.get("task", "")[:40],
        )

    return out


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "groot"}
    horizon = (
        len(_modality_configs["action"].delta_indices)
        if _modality_configs is not None else _profile.n_action_steps
    )
    return {
        "status": "ok" if _policy is not None else "not_loaded",
        "model": "groot-n1.6",
        "profile": _profile.name,
        "embodiment_tag": _embodiment_tag.value if _embodiment_tag else None,
        "n_action_steps": horizon,
        "action_type": _profile.action_type,
        "action_keys": list(_profile.emits_subkeys),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    global _profile

    try:
        from src.utils.common.logger import create_module_logger

        create_module_logger("groot_serve")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GR00T N1.6 추론 서버 (port 8500)")
    parser.add_argument(
        "--profile", type=str, required=True,
        help="체크포인트 프로파일 YAML 경로 (configs/checkpoints/*.yaml)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _profile = load_profile(args.profile)
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "groot", (
        f"profile.base_model={_profile.base_model!r}, but this server is groot"
    )

    app.state.args = args
    app.state._step = 0
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
