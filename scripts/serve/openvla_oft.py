"""
OpenVLA-OFT 추론 서버 (통일 API, port 8400).

프로파일 기반 동작: 체크포인트별 normalization, sub-key 매핑, state 요구사항은
configs/checkpoints/*.yaml 로 선언되고 서버는 해당 프로파일 필드에 따라 분기한다.

openvla_oft 컨테이너에서 실행:
  docker compose run --rm openvla_oft \
    python /temporal_vla/scripts/serve/openvla_oft.py \
    --profile /temporal_vla/configs/checkpoints/openvla_oft__moojink_libero_spatial.yaml

통일 API:
  POST /act     ← {"observation.images.static": b64png, "observation.images.wrist": b64png,
                    "observation.state.*": [...], "task": "..."}
                → {"action.eef_pos": [[...]], "action.eef_euler": [[...]],
                   "action.gripper": [[...]], "latency_ms": float}
  POST /reset   ← 에피소드 시작 시 action queue 초기화
  GET  /health  ← 프로파일 기반 모델/액션 정보
"""

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

# openvla-oft 소스 경로
OPENVLA_OFT_ROOT = "/temporal_vla/src/policies/openvla-oft"
sys.path.insert(0, OPENVLA_OFT_ROOT)

# 프로파일 로더 (scripts/utils 는 PYTHONPATH 에 포함됨)
from checkpoint_profile import CheckpointProfile, load_profile  # noqa: E402

from src.utils.common.image import decode_b64_image  # noqa: E402
from src.utils.common.serving import health_response  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="OpenVLA-OFT Inference Server")

# 전역 상태
_model = None
_processor = None
_action_head = None
_proprio_projector = None
# 입력 텐서를 올릴 device. single-GPU 면 DEVICE(cuda:0), multi-GPU split 이면
# embed_tokens(첫 shard)/proprio_projector 가 있는 device. _load_model_impl 에서 설정.
_input_device = None
_action_queue: deque = deque()
_cfg = None
_profile: Optional[CheckpointProfile] = None


def _quat_xyzw_to_axisangle(quat) -> np.ndarray:
    """Quaternion (x,y,z,w) → axis-angle (3D)."""
    quat = np.array(quat, dtype=np.float64)
    w = float(quat[3])
    w = max(-1.0, min(1.0, w))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(w) / den).astype(np.float32)


def _axisangle_to_euler(aa) -> np.ndarray:
    """Axis-angle (3D) → euler (roll, pitch, yaw)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(aa).as_euler("xyz").astype(np.float32)


# state sub-key → proprio 벡터 dim 매핑. _assemble_proprio 와 proprio_projector init 에서 공유.
_STATE_DIM: Dict[str, int] = {
    "eef_pos": 3,
    "eef_euler": 3,
    "eef_axisangle": 3,
    "eef_quat": 3,        # quat 은 axisangle 로 내부 변환돼 3D 로 들어감
    "eef_rot6d": 6,
    "gripper_qpos": 2,    # LIBERO 2-finger 기본 (1D 입력은 padding)
    "gripper_action": 1,
    "joint_pos": 7,
}


def _proprio_dim(profile: CheckpointProfile) -> int:
    return sum(_STATE_DIM[k] for k in profile.observation_requirements.state)


def _select_unnorm_key(profile: CheckpointProfile, available: List[str]) -> str:
    """프로파일 key_selection fallback chain 에서 실제 존재하는 첫 key 선택."""
    for key in profile.normalization.key_selection:
        if key == "<first_available>":
            if available:
                logger.warning(
                    "unnorm_key fallback to first available: %s (chain exhausted)",
                    available[0],
                )
                return available[0]
            raise RuntimeError("norm_stats is empty; cannot select unnorm_key")
        if key in available:
            return key
    raise RuntimeError(
        f"None of profile.normalization.key_selection found in norm_stats. "
        f"chain={profile.normalization.key_selection}, available={available}"
    )


@app.on_event("startup")
def load_model():
    # uvicorn startup hook 은 exception 을 삼키는 경향이 있어 traceback 을 별도 보장.
    try:
        _load_model_impl()
    except Exception:
        import traceback
        sys.stderr.write("=== load_model FAILED ===\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _load_model_impl():
    global _model, _processor, _action_head, _proprio_projector, _cfg, _input_device

    args = app.state.args
    profile = _profile
    assert profile is not None, "profile must be set before load_model"

    checkpoint_id = profile.checkpoint_source.id
    logger.info("Loading OpenVLA-OFT from %s (profile=%s)", checkpoint_id, profile.name)

    # openvla-oft 코드 기준 working directory 변경 (상대 경로 resolve)
    os.chdir(OPENVLA_OFT_ROOT)

    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import get_model, DEVICE

    # 간이 config 객체 생성 (draccus dataclass 대신)
    class Config:
        model_family = "openvla"
        pretrained_checkpoint = checkpoint_id
        use_l1_regression = True
        use_diffusion = False
        num_diffusion_steps_train = 50
        num_diffusion_steps_inference = 50
        use_film = False
        num_images_in_input = len(profile.observation_requirements.images)
        use_proprio = True
        center_crop = profile.image_preprocess.center_crop
        num_open_loop_steps = profile.n_action_steps
        lora_rank = 32
        unnorm_key = ""
        load_in_8bit = args.load_in_8bit
        load_in_4bit = args.load_in_4bit
        # openvla-oft 내부 코드가 task_suite_name 을 참조하는 곳이 있어 fallback 으로 채워둠
        task_suite_name = profile.normalization.key_selection[0]

    cfg = Config()
    _cfg = cfg

    # 모델 로드
    _model = get_model(cfg)

    # LoRA adapter 적용 (프로파일에 lora 필드가 있을 때)
    if profile.lora is not None:
        from peft import PeftModel

        src = profile.checkpoint_source
        if src.type == "hf_repo":
            adapter_arg = src.id
            subfolder_kwarg = {"subfolder": profile.lora.subfolder}
        else:
            adapter_arg = os.path.join(src.id, profile.lora.subfolder)
            subfolder_kwarg = {}
        logger.info(
            "Applying LoRA adapter from %s (subfolder=%s)",
            adapter_arg, profile.lora.subfolder,
        )
        _model = PeftModel.from_pretrained(_model, adapter_arg, **subfolder_kwarg)
        _model = _model.merge_and_unload()
        logger.info("LoRA adapter merged into base model")

    # unnorm_key 결정 — 프로파일의 key_selection fallback chain 사용
    available = list(_model.norm_stats.keys())
    cfg.unnorm_key = _select_unnorm_key(profile, available)

    # processor
    _processor = get_processor(cfg)

    # action head (L1 regression)
    _action_head = get_action_head(cfg, _model.llm_dim)

    # proprio projector
    proprio_dim = _proprio_dim(profile)
    _proprio_projector = get_proprio_projector(cfg, _model.llm_dim, proprio_dim=proprio_dim)

    # device_map="auto" 로 model 이 multi-GPU 에 split 된 경우, get_action_head/
    # get_proprio_projector 는 default DEVICE(cuda:0) 에 모듈을 만들어 두는데,
    # action_head 의 입력 hidden_states 는 lm_head 가 있는 last device 에서 나오고
    # proprio_projector 의 입력 proprio 는 embed_tokens 가 있는 first device 에 있어야 함.
    if hasattr(_model, "hf_device_map") and _model.hf_device_map:
        lm_head_dev = next(_model.language_model.lm_head.parameters()).device
        embed_dev = next(_model.language_model.model.embed_tokens.parameters()).device
        _action_head = _action_head.to(lm_head_dev)
        _proprio_projector = _proprio_projector.to(embed_dev)
        # 입력(프로세서 출력·proprio)은 첫 shard(embed) device 에 올린다. device_map 이
        # cuda:0 을 첫 shard 로 쓰지 않는 커스텀 맵이어도 proprio_projector 와 device 가 맞음.
        _input_device = embed_dev
        logger.info(
            "Multi-GPU split detected (hf_device_map=%s). "
            "Moved action_head→%s, proprio_projector→%s, inputs→%s",
            _model.hf_device_map, lm_head_dev, embed_dev, _input_device,
        )
    else:
        _input_device = DEVICE

    logger.info(
        "OpenVLA-OFT loaded. unnorm_key=%s, num_open_loop_steps=%d, proprio_dim=%d",
        cfg.unnorm_key, cfg.num_open_loop_steps, proprio_dim,
    )


def _assemble_proprio(payload: dict, profile: CheckpointProfile) -> np.ndarray:
    """프로파일의 observation_requirements.state 순서대로 proprio 벡터 조립.

    지원 키와 차원은 _STATE_DIM 참조. 누락된 키는 프로파일의 allow_conversions 를 참고해
    fallback 변환을 시도하고, 모두 실패하면 zero padding.
    """
    from scipy.spatial.transform import Rotation

    conversions = set(profile.observation_requirements.allow_conversions)
    parts: List[np.ndarray] = []

    for key in profile.observation_requirements.state:
        dim = _STATE_DIM[key]
        raw = payload.get(f"observation.state.{key}")

        if raw is not None:
            arr = np.array(raw, dtype=np.float32).flatten()
        elif key == "eef_axisangle":
            # quat 또는 euler 로부터 변환 허용
            quat = payload.get("observation.state.eef_quat")
            euler = payload.get("observation.state.eef_euler")
            if quat is not None and "quat_to_axisangle" in conversions:
                arr = _quat_xyzw_to_axisangle(quat)
            elif euler is not None and "euler_to_axisangle" in conversions:
                arr = Rotation.from_euler("xyz", euler).as_rotvec().astype(np.float32)
            else:
                arr = np.zeros(dim, dtype=np.float32)
        elif key == "eef_quat":
            # 요청에는 quat 이 있지만 payload 에는 없을 때 — 현재 openvla-oft 는 quat→axisangle 로
            # 내부 사용하므로 dim 은 3 으로 맞춰야 함.
            euler = payload.get("observation.state.eef_euler")
            if euler is not None:
                arr = Rotation.from_euler("xyz", euler).as_rotvec().astype(np.float32)
            else:
                arr = np.zeros(dim, dtype=np.float32)
        elif key == "gripper_qpos":
            # 1D action 으로부터 2D qpos padding 허용
            gripper_action = payload.get("observation.state.gripper_action")
            if gripper_action is not None:
                g = float(np.array(gripper_action).flatten()[0])
                arr = np.array([g, g], dtype=np.float32)
            else:
                arr = np.zeros(dim, dtype=np.float32)
        else:
            arr = np.zeros(dim, dtype=np.float32)

        # quat 요청이면 axisangle(3D) 로 내부 변환되어 들어감
        if key == "eef_quat" and raw is not None:
            arr = _quat_xyzw_to_axisangle(raw)

        # gripper_qpos 1D padding
        if key == "gripper_qpos" and len(arr) == 1:
            arr = np.array([arr[0], arr[0]], dtype=np.float32)

        # dim 이 예상치와 다르면 잘라내거나 pad
        if len(arr) != dim:
            if len(arr) > dim:
                arr = arr[:dim]
            else:
                arr = np.concatenate([arr, np.zeros(dim - len(arr), dtype=np.float32)])

        parts.append(arr.astype(np.float32))

    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _predict_action_chunk(payload: dict) -> np.ndarray:
    """OpenVLA-OFT로 action chunk 예측. [num_open_loop_steps, action_dim] 반환."""
    from experiments.robot.openvla_utils import (
        normalize_proprio,
        prepare_images_for_vla,
    )
    from experiments.robot.robot_utils import DEVICE

    profile = _profile
    assert profile is not None
    resolution = profile.image_preprocess.resolution
    rotate_180 = profile.image_preprocess.rotate_180

    # 이미지 전처리 — 프로파일 images 순서대로 로드
    images_np: List[np.ndarray] = []
    for view in profile.observation_requirements.images:
        b64 = payload.get(f"observation.images.{view}")
        if b64:
            arr = decode_b64_image(b64)
        else:
            arr = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        if rotate_180:
            arr = arr[::-1, ::-1].copy()
        images_np.append(arr)

    # 이미지 준비 (resize + center crop) — openvla-oft 의 공용 전처리 유틸
    all_images = prepare_images_for_vla(images_np, _cfg)
    primary_image = all_images[0]
    wrist_images = all_images[1:]

    # 프롬프트
    instruction = payload.get("task", "")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    # processor를 통한 입력 생성 — multi-GPU split 시 첫 shard device, 아니면 DEVICE(cuda:0)
    dev = _input_device if _input_device is not None else DEVICE
    inputs = _processor(prompt, primary_image).to(dev, dtype=torch.bfloat16)

    # wrist 이미지 결합
    if wrist_images:
        wrist_inputs = [
            _processor(prompt, img).to(dev, dtype=torch.bfloat16)
            for img in wrist_images
        ]
        primary_pv = inputs["pixel_values"]
        wrist_pvs = [wi["pixel_values"] for wi in wrist_inputs]
        inputs["pixel_values"] = torch.cat([primary_pv] + wrist_pvs, dim=1)

    # proprio 처리
    proprio = _assemble_proprio(payload, profile)
    proprio_norm_stats = _model.norm_stats[_cfg.unnorm_key]["proprio"]
    proprio = normalize_proprio(proprio, proprio_norm_stats)

    # 추론
    with torch.inference_mode():
        actions, _ = _model.predict_action(
            **inputs,
            unnorm_key=_cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=_proprio_projector,
            action_head=_action_head,
        )

    # actions: list of np.ndarray, 각 [7]
    return np.array(actions)  # [num_actions, 7]


def _process_action(action: np.ndarray, profile: CheckpointProfile) -> np.ndarray:
    """프로파일의 gripper_encoding 기준으로 gripper 차원 후처리."""
    action = action.copy()
    ge = profile.gripper_encoding
    grip_slice = profile.dim_slice("gripper")
    grip = action[grip_slice]

    if ge.binarize:
        if ge.range == "[0,1]":
            grip = 2.0 * grip - 1.0
        # "[-1,1]" / "{-1,1}" 은 이미 [-1,1] 범위로 가정
        # threshold 가 0 아닌 경우 적용 가능하나 현재는 np.sign 으로 통일
        grip = np.sign(grip - ge.threshold)
    if ge.sign_flip:
        grip = -grip

    action[grip_slice] = grip
    return action


def _rotation_key(profile: CheckpointProfile) -> str:
    """action_layout 에서 rotation 을 담당하는 엔트리 name 을 찾음."""
    for a in profile.action_layout:
        if a.name in ("eef_euler", "eef_axisangle", "eef_quat", "eef_rot6d"):
            return a.name
    raise RuntimeError("action_layout has no rotation component")


def _rotation_to_euler(rot_vec: np.ndarray, rot_key: str) -> np.ndarray:
    """rotation 차원을 xyz euler 로 변환."""
    from scipy.spatial.transform import Rotation

    if rot_key == "eef_euler":
        return rot_vec.astype(np.float32)
    if rot_key == "eef_axisangle":
        return _axisangle_to_euler(rot_vec)
    if rot_key == "eef_quat":
        return Rotation.from_quat(rot_vec).as_euler("xyz").astype(np.float32)
    if rot_key == "eef_rot6d":
        # Gram-Schmidt 로 6D → rotation matrix → euler
        a1 = rot_vec[:3]
        a2 = rot_vec[3:6]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        mat = np.stack([b1, b2, b3], axis=1)
        return Rotation.from_matrix(mat).as_euler("xyz").astype(np.float32)
    raise ValueError(f"unsupported rotation key: {rot_key}")


def _emit_subkeys(action: np.ndarray, profile: CheckpointProfile) -> Dict[str, list]:
    """action 벡터를 프로파일 emits_subkeys 규약에 맞춰 dict 로 변환.

    각 값은 [N, dim] 의 2D list 여야 하나 서버가 queue 에서 1개씩 반환하므로 N=1.
    """
    rot_key = _rotation_key(profile)
    rot_vec = action[profile.dim_slice(rot_key)]

    out: Dict[str, list] = {}
    for sk in profile.emits_subkeys:
        if sk == "action.eef_pos":
            out[sk] = [action[profile.dim_slice("eef_pos")].tolist()]
        elif sk == "action.eef_euler":
            out[sk] = [_rotation_to_euler(rot_vec, rot_key).tolist()]
        elif sk == "action.eef_axisangle":
            if rot_key == "eef_axisangle":
                out[sk] = [rot_vec.tolist()]
            else:
                raise NotImplementedError(
                    f"emit action.eef_axisangle from {rot_key} not supported yet"
                )
        elif sk == "action.eef_quat":
            from scipy.spatial.transform import Rotation

            euler = _rotation_to_euler(rot_vec, rot_key)
            quat = Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)
            out[sk] = [quat.tolist()]
        elif sk == "action.eef_rot6d":
            if rot_key == "eef_rot6d":
                out[sk] = [rot_vec.tolist()]
            else:
                raise NotImplementedError(
                    f"emit action.eef_rot6d from {rot_key} not supported yet"
                )
        elif sk == "action.gripper":
            out[sk] = [action[profile.dim_slice("gripper")].tolist()]
        else:
            raise ValueError(f"unsupported emit sub-key: {sk}")
    return out


@app.post("/act")
async def predict_action(payload: dict):
    """통일 API: observation → action sub-keys."""
    if _model is None:
        return {"error": "model not loaded"}

    t0 = time.time()
    profile = _profile
    assert profile is not None

    try:
        # action queue가 비었으면 새로 예측
        if len(_action_queue) == 0:
            action_chunk = _predict_action_chunk(payload)
            for i in range(len(action_chunk)):
                _action_queue.append(action_chunk[i])

        # queue에서 하나씩 꺼내서 반환
        action = _action_queue.popleft()
        action = _process_action(action, profile)

        out = _emit_subkeys(action, profile)
        out["latency_ms"] = (time.time() - t0) * 1000
        return out
    except Exception:
        # uvicorn 기본 핸들러가 traceback 을 stdout 으로 안 흘려서 진단 불가.
        # 직접 stderr 로 dump 후 500 반환.
        import traceback as _tb
        _tb.print_exc()
        sys.stderr.flush()
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(status_code=500, detail=_tb.format_exc())


@app.post("/reset")
async def reset():
    """에피소드 시작 시 action queue 초기화."""
    _action_queue.clear()
    return {"status": "reset"}


@app.get("/health")
async def health():
    if _profile is None:
        return {"status": "not_loaded", "model": "openvla-oft"}
    return health_response(
        policy=_model,
        model="openvla-oft",
        profile=_profile,
        # serve 는 내부 chunk 를 queue 에서 1개씩 반환하므로 외부 API 기준 1
        n_action_steps=1,
        action_type=_profile.action_type,
        action_keys=list(_profile.emits_subkeys),
    )


def main():
    global _profile

    try:
        from src.utils.common.logger import create_module_logger

        create_module_logger("openvla_oft_serve")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="OpenVLA-OFT 추론 서버 (port 8400)")
    parser.add_argument(
        "--profile", type=str, required=True,
        help="체크포인트 프로파일 YAML 경로 (configs/checkpoints/*.yaml)",
    )
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    args = parser.parse_args()

    _profile = load_profile(args.profile)
    logger.info("Loaded profile %s from %s", _profile.name, args.profile)
    assert _profile.base_model == "openvla_oft", (
        f"profile.base_model={_profile.base_model!r}, but this server is openvla_oft"
    )

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
