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
import contextlib
import logging
import os
import random
import sys
import time
import types
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
from src.utils.common.feature_blob import encode_feature_blob  # noqa: E402
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
    serve_provenance,
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
_steering = []  # list of registered steering hooks (multi-layer 지원)
_gated_registry: dict = {}
# 통일 arm 레지스트리 (2026-08-10 배선반 통일 1단계): target(pathway,layer) →
# {"hook","family","per_step"}. 4모드 전부 여기 등록되고, /steer_arm 이 재기동 없이
# 같은 family(conceptor↔conceptor, setpoint↔setpoint) 안에서 연산자를 교체한다 —
# 동적 큐의 슬롯 가동률 전제. family 교체는 hook 클래스가 달라 재기동 필요(명시 에러).
_condg_hooks: list = []  # condg(상태-조건부 대조 guidance) hook — /act 마다 상태 주입 필요
_arm_registry: dict = {}  # oracle phase-gated steering: {"hooks":{layer:hook},"matrices":{layer:{phase:M|M_seq}},"identity":{layer:I|[I]*K}}
# 프로세스 지문: 러너가 "포트의 기존 서버"를 새 serve 로 오인하는 사고 방지 (Gate 2 치명#3)
# — 로그의 [serve-boot] id 와 /health 의 boot_id 가 일치해야 같은 프로세스.
_BOOT_ID = uuid.uuid4().hex[:12]
_steering_spec: dict = {}  # /health 노출용 스티어링 지문 (mode/layers/β/npz sha/…)
# patchceil donor-trajectory transplant (docs/collab/2026-07-16-patching-transplant-gate1.md)
_patch_hooks: dict = {}  # {layer(int): PatchSteering}
_patch_spec: dict = {}  # /health 노출용 patch 지문 (layers/token/K/armed tag/npz sha)
_patch_donor_arrays: dict = {}  # 마지막 로드 donor {layer: [R,K,T,D]} — npz 생략 재-arm 용

# ---------------------------------------------------------------- online phase readout
# 추론 중(inference-time) action-phase 판정. DiT layer residual → task_classification 의
# 사후-이산화 AE/SAE(kmeans) 로 현재 phase 를 읽어 /act_with_features 응답에 실어 보낸다.
# 학습 데이터와 동일하게: layer L 의 마지막 denoise step, 전체 토큰 mean-pool → [D] → clf.
# all_token_full 캡처(=학습 데이터 규격) + 해당 layer 캡처가 켜져 있어야 한다.
_phase_readouts: list = []       # [{"name": "ae"|"sae", "clf": OnlinePhaseClassifier}]
_phase_layer: int = 12           # 물리 DiT block layer (capture_layers 에 존재해야 함)
_phase_denoise: int | None = None  # denoise step index (None=마지막). 학습=3=마지막(K=4)
_phase_spec: dict = {}           # /health 노출용


def _phase_layer_index() -> int:
    """물리 layer(_phase_layer) → features.hidden_states 의 L축 인덱스.

    hidden_states 는 --groot-dit-capture-layers 순서대로 layer 축을 쌓는다.
    """
    layers = list(_groot_dit_capture_layers or [])
    if _phase_layer not in layers:
        raise RuntimeError(
            f"phase-readout: layer {_phase_layer} 가 capture layers {layers} 에 없음 "
            f"(--groot-dit-capture-layers 에 {_phase_layer} 포함 필요)"
        )
    return layers.index(_phase_layer)


def _phase_from_hidden(hidden_np) -> dict | None:
    """features.hidden_states [L, K, T, D] → {name: {cluster, phase, purity}}.

    학습 파이프라인과 동일한 축약: layer L, 마지막(또는 지정) denoise step, 전체 T 토큰
    mean-pool → [D]. clf 가 내부에서 PCA-64 → encode → kmeans → phase 로 잇는다.
    """
    if not _phase_readouts:
        return None
    arr = np.asarray(hidden_np)
    if arr.ndim != 4:                       # all_token_full 이 아니면 [L,K,D] 등 — 미지원
        return None
    li = _phase_layer_index()
    ki = _phase_denoise if _phase_denoise is not None else arr.shape[1] - 1  # 마지막 denoise
    h = arr[li, ki].astype(np.float32).mean(axis=0)   # [T, D] → [D] (전체 토큰 평균)
    return {r["name"]: r["clf"].infer(h) for r in _phase_readouts}


def _load_phase_readouts_if_requested(args) -> None:
    """--phase-readout 시 AE/SAE 분류기를 로드하고 캡처 설정 정합을 검증한다.

    readout 은 학습 데이터와 같은 tensor(전체 토큰 보존 residual)가 필요하므로
    all_token_full 캡처 + 해당 layer 캡처가 켜져 있어야 한다. 안 맞으면 조용히
    틀린 phase 를 내기보다 즉시 실패한다.
    """
    global _phase_readouts, _phase_layer, _phase_denoise, _phase_spec
    if not getattr(args, "phase_readout", False):
        return
    if _policy_type != "groot":
        raise ValueError("--phase-readout 은 policy_type='groot' 전용")
    if _groot_dit_capture_layers is None:
        raise ValueError(
            "--phase-readout 은 --groot-dit-capture-layers 필요 "
            "(DiT residual 캡처가 켜져야 phase 를 읽는다)"
        )
    if _groot_dit_token_pool != "all_token_full":
        raise ValueError(
            "--phase-readout 은 --groot-dit-token-pool all_token_full 필요 "
            f"(현재 {_groot_dit_token_pool!r}) — clf 는 전체 토큰 mean 으로 학습됨"
        )
    _phase_layer = int(args.phase_layer)
    _phase_denoise = None if args.phase_denoise is None else int(args.phase_denoise)
    _phase_layer_index()  # capture layers 에 _phase_layer 존재하는지 preflight

    from src.phase_online import OnlinePhaseClassifier

    runs = [r.strip() for r in str(args.phase_run_dirs).split(",") if r.strip()]
    _phase_readouts = []
    for run in runs:
        run_path = Path(run)
        if not run_path.is_absolute():
            run_path = Path(__file__).resolve().parents[2] / run
        name = run_path.name.split("-")[0]  # ae-log_likelihood-s0 → "ae"
        clf = OnlinePhaseClassifier.from_run(
            run_dir=run_path, pca_path=args.phase_pca, map_path=args.phase_map,
            device=args.phase_device,
        )
        _phase_readouts.append({"name": name, "clf": clf})
    _phase_spec = {
        "enabled": True, "layer": _phase_layer,
        "denoise": "last" if _phase_denoise is None else _phase_denoise,
        "runs": [r["name"] for r in _phase_readouts],
        "device": args.phase_device,
    }
    logger.info(
        "online phase readout ON: layer=%s denoise=%s runs=%s",
        _phase_layer, _phase_spec["denoise"], _phase_spec["runs"],
    )


# ---------------------------------------------------------------- online failure detector
# SAFE 식 causal failure detector(`scripts/analysis/grid_phase/failure_detector_sim.py` 산출
# `detector_<arm>_<model>_<slug|all>.pt`)를 추론 경로에 얹어, 매 record 마다 score 와
# CP 밴드 발화(score > δ_t)를 응답에 실어 보낸다 — 클라이언트의 online gating 신호.
# phase readout 과 같은 자리(같은 hidden [L,K,T,D])에서 계산하며 좌표는 ckpt 가 정한다.
_failure_detector = None         # OnlineFailureDetector | None
_failure_layer_idx: int | None = None
_failure_spec: dict = {}         # /health 노출용


# ---------------------------------------------------------------- cluster phase assigner
# per-step 게이트의 phase 를 GT POST 값 대신 serve 자체 activation cluster 로 정한다
# (docs/steering/47 후속, cluster-k8 라운드). 번들 = ae_cluster.py --export-bundle 산출 NPZ.
_cluster_assigner = None         # ClusterPhaseAssigner | None
_cluster_layer_idx: int | None = None
_cluster_spec: dict = {}         # /health 노출용


def _cluster_phase_from_hidden(hidden_np) -> dict | None:
    """features.hidden_states [L,K,T,D] → {"name":"c3","idx":3,"dist":float}.

    detector 와 달리 상태가 없다 (per-record 독립) — /reset 에서 할 일도 없다.
    좌표(L12·마지막 denoise·49토큰 mean)는 detector 와 동일해야 한다.
    """
    if _cluster_assigner is None:
        return None
    feat = _cluster_assigner.feature_from_hidden(
        np.asarray(hidden_np), _cluster_layer_idx)
    return _cluster_assigner.assign(feat)


def _load_cluster_phase_if_requested(args) -> None:
    """--cluster-phase-bundle 시 판정기를 로드하고 캡처 설정 정합을 검증한다.

    detector 와 **같은 좌표 전제**(groot + all_token_full + 번들 layer 가 capture layers
    에 포함)를 fail-loud 로 건다 — 조용히 틀린 좌표로 phase 를 붙이는 것 방지.
    """
    global _cluster_assigner, _cluster_layer_idx, _cluster_spec
    path = getattr(args, "cluster_phase_bundle", None)
    if not path:
        return
    if _policy_type != "groot":
        raise ValueError("--cluster-phase-bundle 는 policy_type='groot' 전용")
    if _groot_dit_capture_layers is None:
        raise ValueError(
            "--cluster-phase-bundle 는 --groot-dit-capture-layers 필요 "
            "(DiT residual 캡처가 켜져야 cluster feature 를 만든다)"
        )
    if _groot_dit_token_pool != "all_token_full":
        raise ValueError(
            "--cluster-phase-bundle 는 --groot-dit-token-pool all_token_full 필요 "
            f"(현재 {_groot_dit_token_pool!r}) — 번들은 49토큰 mean 으로 학습됨"
        )
    from src.failure_online.cluster_phase import ClusterPhaseAssigner

    bundle_path = Path(path)
    if not bundle_path.is_absolute():
        bundle_path = Path(__file__).resolve().parents[2] / bundle_path
    _cluster_assigner = ClusterPhaseAssigner.from_bundle(
        bundle_path,
        task=getattr(args, "cluster_phase_task", None) or None,
        device=str(getattr(args, "cluster_phase_device", "cpu")),
    )
    _cluster_layer_idx = _cluster_assigner.resolve_layer_index(
        list(_groot_dit_capture_layers)
    )  # preflight (capture layers 에 번들 layer 존재)
    _cluster_spec = {"enabled": True, **_cluster_assigner.spec()}
    logger.info(
        "cluster phase assigner ON: bundle=%s slug=%s k=%s latent=%s layer=%s "
        "denoise_index=%s seg=%s",
        _cluster_spec["bundle"], _cluster_spec["slug"], _cluster_spec["k"],
        _cluster_spec["latent"], _cluster_spec["layer"],
        _cluster_spec["denoise_index"], _cluster_spec["seg"],
    )


# ---------------------------------------------------------------- LLR 재샘플 채점기
# best-of-N 재샘플(op=rsn_llr)에서 후보 활성화를 phase 조건부 성공/실패 가우시안
# 로그우도비로 채점한다. NPZ 계약의 단일 출처 = src/failure_online/llr_scorer.py docstring.
_llr_scorer = None               # LLRScorer | None
_llr_scene: str | None = None    # 기동 시 고정하는 scene ("s3") — 등록 단위가 (scene, phase)
_llr_spec: dict = {}             # /health 노출용


def _raw_feature_from_hidden(hidden_np) -> np.ndarray:
    """features.hidden_states [L,K,T,D] → LLR 입력 raw feature [1536] (표준화 전).

    좌표 추출은 **기존 헬퍼만** 쓴다 (새 추출 경로를 만들면 학습 좌표와 조용히 어긋난다):
    cluster 번들이 있으면 그 좌표, 없으면 detector 좌표 — 둘은 같은 좌표 전제다.
    """
    if _cluster_assigner is not None:
        return _cluster_assigner.feature_from_hidden(np.asarray(hidden_np), _cluster_layer_idx)
    if _failure_detector is not None:
        return _failure_detector.feature_from_hidden(np.asarray(hidden_np), _failure_layer_idx)
    raise HTTPException(
        status_code=409,
        detail="LLR 재샘플: raw feature 좌표를 정할 판정기가 없음 "
        "(--cluster-phase-bundle 또는 --failure-detector 필요)",
    )


def _load_llr_scorer_if_requested(args) -> None:
    """--llr-bundle 시 채점기를 로드한다 (미지정이면 None — rsn_llr 요청 시 409).

    연산자 등록 단위가 **(scene, phase)** 라 번들은 task 당 1개이고 scene 은 기동
    인자(--llr-scene)로 고정한다 — 없으면 기동 실패(어느 scene 의 가우시안으로 채점하는지
    불명인 채 도는 것 방지).
    """
    global _llr_scorer, _llr_scene, _llr_spec
    path = getattr(args, "llr_bundle", None)
    if not path:
        return
    if _policy_type != "groot":
        raise ValueError("--llr-bundle 는 policy_type='groot' 전용")
    if _groot_dit_capture_layers is None or _groot_dit_token_pool != "all_token_full":
        raise ValueError(
            "--llr-bundle 는 --groot-dit-capture-layers + "
            "--groot-dit-token-pool all_token_full 필요 (후보 활성화 캡처 좌표)"
        )
    from src.failure_online.llr_scorer import LLRScorer, normalize_scene

    scene_arg = getattr(args, "llr_scene", None)
    if scene_arg in (None, ""):
        raise ValueError(
            "--llr-bundle 는 --llr-scene 필요 (등록 단위가 (scene, phase) — "
            "어느 scene 의 가우시안으로 채점할지 명시할 것)"
        )
    _llr_scene = normalize_scene(scene_arg)

    bundle_path = Path(path)
    if not bundle_path.is_absolute():
        bundle_path = Path(__file__).resolve().parents[2] / bundle_path
    _llr_scorer = LLRScorer.from_bundle(bundle_path)
    if _llr_scene not in _llr_scorer.scenes():
        raise ValueError(
            f"--llr-scene {_llr_scene} 이 번들에 없음 (있는 scene: {_llr_scorer.scenes()})"
        )
    # basename 만 기록 (docs/04 §8 — 산출물 안 절대경로 금지)
    _llr_spec = {"enabled": True, "bundle": bundle_path.name, "scene": _llr_scene,
                 **_llr_scorer.spec()}
    logger.info("LLR resample scorer ON: bundle=%s scene=%s registered=%s",
                _llr_spec["bundle"], _llr_scene, _llr_spec["registered"])


def _failure_from_hidden(hidden_np) -> dict | None:
    """features.hidden_states [L,K,T,D] → {"score","fired","delta","t"} (1 step 전진).

    detector 는 **episode 안에서 상태를 이어간다** (LSTM (h,c) / MLP 누적평균 / step
    카운터) — /reset 이 리셋한다. 요청 1개 = record 1개 규약을 전제로 한 step 씩 전진.
    """
    if _failure_detector is None:
        return None
    feat = _failure_detector.feature_from_hidden(np.asarray(hidden_np), _failure_layer_idx)
    return _failure_detector.step(feat)


def _load_failure_detector_if_requested(args) -> None:
    """--failure-detector 시 detector 를 로드하고 캡처 설정 정합을 검증한다.

    phase readout 과 같은 fail-loud 기준: groot + all_token_full 캡처 + ckpt 가 지정한
    물리 layer 가 capture layers 에 있어야 한다 (조용히 틀린 좌표로 점수 내는 것 방지).
    """
    global _failure_detector, _failure_layer_idx, _failure_spec
    path = getattr(args, "failure_detector", None)
    if not path:
        return
    if _policy_type != "groot":
        raise ValueError("--failure-detector 는 policy_type='groot' 전용")
    if _groot_dit_capture_layers is None:
        raise ValueError(
            "--failure-detector 는 --groot-dit-capture-layers 필요 "
            "(DiT residual 캡처가 켜져야 detector feature 를 만든다)"
        )
    if _groot_dit_token_pool != "all_token_full":
        raise ValueError(
            "--failure-detector 는 --groot-dit-token-pool all_token_full 필요 "
            f"(현재 {_groot_dit_token_pool!r}) — detector 는 토큰 세그먼트 mean 으로 학습됨"
        )
    from src.failure_online import OnlineFailureDetector

    ckpt_path = Path(path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parents[2] / ckpt_path
    _failure_detector = OnlineFailureDetector.from_checkpoint(
        ckpt_path,
        alpha=float(args.failure_alpha),
        task=getattr(args, "failure_task", None) or None,
        device=str(getattr(args, "failure_device", "cpu")),
    )
    _failure_layer_idx = _failure_detector.resolve_layer_index(
        list(_groot_dit_capture_layers)
    )  # preflight (capture layers 에 ckpt layer 존재)
    _failure_detector.reset()
    _failure_spec = {"enabled": True, **_failure_detector.spec()}
    logger.info(
        "online failure detector ON: ckpt=%s model=%s layer=%s denoise=%s seg=%s "
        "task=%s alpha=%s band_L=%s",
        _failure_spec["ckpt"], _failure_spec["model"], _failure_spec["layer"],
        _failure_spec["denoise"], _failure_spec["seg"], _failure_spec["task"],
        _failure_spec["alpha"], _failure_spec["band_L"],
    )


def _reset_steering_step_counters(*, include_patch: bool = True) -> None:
    """Per-Step steering 의 denoise call 카운터를 요청 시작 시 리셋.

    phase 는 요청 단위(/steering_phase), step 은 요청 내 denoise call 단위라 직교 —
    /act·/act_with_features 진입부에서 매 요청 호출한다 (global M 단일 hook 은 no-op).

    ``include_patch=False``: per-step 게이트의 2차 pass(같은 record 재실행)용 —
    patch hook 의 reset 은 record cursor 전진을 겸하므로 요청당 1회만 불러야 한다.
    """
    for hook in _steering:
        reset = getattr(hook, "reset_step_counter", None)
        if reset is not None:
            reset()
    if not include_patch:
        return
    # patch hook 은 같은 호출이 k 리셋 + record cursor 전진을 겸한다
    # (요청 1개 = record 1개 규약, patching_hooks.PatchSteering docstring)
    for hook in _patch_hooks.values():
        hook.reset_step_counter()


CONDG_STATE_KEYS = (
    "observation.state.eef_pos_rel",     # 3
    "observation.state.eef_quat_rel",    # 4
    "observation.state.gripper_qpos",    # 2
)


def _inject_condg_state(payload: dict) -> None:
    """condg hook 에 raw proprio 9차원을 주입 (요청 1개 = record 1개 규약).

    속도는 hook 내부 버퍼의 직전 record 차분이라 **요청마다 정확히 1회** 불러야 한다
    (docs/steering/44 §1). 키가 없으면 조용히 0 을 넣지 않고 즉시 실패 — 상태가
    비면 setpoint 가 통째로 틀어진다.
    """
    if not _condg_hooks:
        return
    parts = []
    for key in CONDG_STATE_KEYS:
        raw = payload.get(key)
        if raw is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"condg steering: payload 에 {key} 없음 "
                    f"(필요 키 {list(CONDG_STATE_KEYS)}) — GR00T RoboCasa io 경로 필요"
                ),
            )
        parts.append(np.asarray(raw, dtype=np.float64).reshape(-1))
    p9 = np.concatenate(parts)
    for hook in _condg_hooks:
        hook.set_state(p9)


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


def _apply_steering_phase_state(phase: str | None, scene: int | None) -> None:
    """등록된 gated hook 을 (phase, scene) 상태로 스위칭 — 실제 배선 본문.

    ``/steering_phase`` 핸들러와 per-step 게이트의 2차 pass 가 공유한다
    (docs/steering/47 §3-3). ``phase`` 가 None/빈 문자열이거나 미등록 phase 면
    무개입(identity / set_vector(None) / set_phase(None)) — 즉 off 와 같은 경로다.
    ``_gated_registry["current"]`` 는 여기서 건드리지 않는다 (핸들러 책임).
    """
    for layer, hook in _gated_registry["hooks"].items():
        if hasattr(hook, "set_phase"):
            # condg: hook 이 전 phase 파라미터를 들고 있어 이름·scene 만 스위칭.
            hook.set_phase(phase or None, scene)
            continue
        M = _gated_registry["matrices"][layer].get(phase) if phase else None
        if hasattr(hook, "set_vector"):
            # setpoint hook: 등록 phase → 활성, 미등록 → 비활성(no-op).
            # 4-튜플=세그먼트 연산자(v_seg,s_tok,bounds,mask), 2-튜플=구 pooled (r̂,s)
            if M is None:
                hook.set_segment(None)
                hook.set_vector(None)
            elif len(M) == 4:
                hook.set_segment(M)
            else:
                hook.set_vector(*M)
        else:
            # set_matrices 가 M(단일) / M_seq(per-step 리스트) 모두 수용, 텐서 캐시·step
            # 카운터도 함께 리셋한다 (구 ``hook.M=...; hook._Mt=None`` 배선 대체).
            hook.set_matrices(M if M is not None else _gated_registry["identity"][layer])


def _steering_phase_off() -> None:
    """전 gated hook 무개입 전환 (미등록 phase 폴백과 동일 경로). 등록 없으면 no-op.

    per-step 게이트 규약: 요청 시작 시 무조건 off → 1차 pass 는 자연 활성화,
    발화한 record 만 2차 pass 에서 잠깐 on 했다가 다시 off (latch 없음).
    """
    if not _gated_registry:
        return
    _apply_steering_phase_state(None, None)


@app.post("/steering_phase")
def steering_phase(payload: dict):
    """Oracle phase-gated steering: 현재 phase 의 conceptor 로 hook M 을 스위칭.

    수집 client 가 매 get_action 전에 POST {"phase": "<reach-to-object|transport|...>"}.
    등록된 phase 가 없으면 identity(=no steer). --steering-phase-npz-base 로 활성화.

    condg(op=condg)는 phase 에 더해 optional ``"scene": int`` 를 받아 scene별 중심화
    파라미터를 고른다 (미지 scene = global fallback). 기존 호출자는 scene 없이 그대로.
    """
    if not _gated_registry:
        raise HTTPException(status_code=409, detail="gated steering not enabled")
    phase = str(payload.get("phase", ""))
    scene = payload.get("scene")
    scene = None if scene is None else int(scene)
    # 비-perstep 경로는 현행 그대로: POST 즉시 적용 (per-step 게이트는 /act 진입부에서
    # off 시킨 뒤 발화한 record 의 2차 pass 에서만 이 상태를 다시 적용한다).
    _apply_steering_phase_state(phase, scene)
    _gated_registry["current"] = phase
    _gated_registry["current_scene"] = scene
    return {"ok": True, "phase": phase, "scene": scene,
            "gated": phase in next(iter(_gated_registry["matrices"].values()))}


@app.post("/steer_arm")
def steer_arm(payload: dict):
    """재기동 없이 연산자 교체 (배선반 통일 1단계, 2026-08-10 — 동적 큐의 슬롯 가동률 전제).

    payload: {"op": "conceptor"|"setpoint"|"setpoint_seg", "beta": float,
              "alpha"?: float, "key"?: str, "denoise"?: "global"|"per_step",
              "token_select"?: str,
              "bindings": [{"pathway":"dit"|"vl", "layer": int|None, "npz": path}, ...]}

    제약 (명시 에러 — 무음 오적용 방지):
      - target(pathway,layer) 은 **기동 시 설치된 hook 집합 안**이어야 한다
        (hook 추가·family 교체는 compile 캐시 상호작용 회피를 위해 재기동).
      - family(conceptor↔setpoint) 는 기존 hook 클래스와 일치해야 한다.
      - gated(_gated_registry) 상태에서는 사용 불가 — gated arm 은 재기동으로 교체.
    """
    import hashlib as _hashlib

    from steering_hooks import (
        load_steering_matrices_per_step,
        load_steering_matrix,
        load_steering_segment,
        load_steering_setpoint,
    )

    if _gated_registry:
        raise HTTPException(status_code=409, detail="gated 등록 상태 — /steer_arm 불가(재기동 필요)")
    if not _arm_registry:
        raise HTTPException(status_code=409, detail="기동 시 설치된 steering hook 없음(재기동 필요)")

    op = payload.get("op")
    beta = float(payload.get("beta", 0.3))
    alpha = payload.get("alpha")
    key = payload.get("key", "C_steer")
    denoise = payload.get("denoise", "global") or "global"
    token_select = payload.get("token_select")
    bindings = payload.get("bindings") or []
    if op not in ("conceptor", "setpoint", "setpoint_seg"):
        raise HTTPException(status_code=422, detail=f"op 불명: {op!r}")
    fam = "setpoint" if op.startswith("setpoint") else "conceptor"
    per_step = denoise == "per_step"
    if per_step and fam == "setpoint":
        raise HTTPException(status_code=422, detail="setpoint 는 denoise=global 전용")

    # 검증 먼저 전부 — 일부만 갈아끼운 채 실패하는 부분 적용 방지
    plans = []
    for b in bindings:
        tgt = (b.get("pathway", "dit"), b.get("layer"))
        ent = _arm_registry.get(tgt)
        if ent is None:
            raise HTTPException(
                status_code=409,
                detail=f"target {tgt} 은 기동 시 설치 안 됨 — 설치 집합 {sorted(_arm_registry)} (재기동 필요)")
        if ent["family"] != fam:
            raise HTTPException(
                status_code=409,
                detail=f"target {tgt} family {ent['family']} != 요청 {fam} — hook 클래스가 달라 재기동 필요")
        npz = b.get("npz")
        if not npz or not Path(npz).exists():
            raise HTTPException(status_code=422, detail=f"npz 없음: {npz}")
        plans.append((tgt, ent, npz))

    expected_steps = None
    if per_step:
        _gm = getattr(policy, "_groot_model", None)
        expected_steps = int(_gm.action_head.num_inference_timesteps)

    shas, layers = [], []
    for tgt, ent, npz in plans:
        shas.append(_hashlib.sha256(Path(npz).read_bytes()).hexdigest()[:12])
        hook = ent["hook"]
        if fam == "setpoint":
            if op == "setpoint_seg":
                hook.set_segment(load_steering_segment(str(npz)))
            else:
                v, sp = load_steering_setpoint(str(npz), alpha=alpha)
                hook.set_vector(v, sp)
            hook.beta = beta
        else:
            if per_step:
                m = load_steering_matrices_per_step(
                    str(npz), beta=beta, alpha=alpha, key=key, num_steps=expected_steps)
            else:
                m = load_steering_matrix(str(npz), beta=beta, alpha=alpha, key=key)
            hook.set_matrices(m)
        if tgt[1] is not None:
            layers.append(int(tgt[1]))
        print(f"[steer-rearm] target={tgt} op={op} beta={beta:g} npz={npz}", flush=True)

    _update_steering_spec(mode="rearm", op=op, layers=layers, beta=beta, alpha=alpha,
                          key=key, token_select=token_select, denoise=denoise, npz_shas=shas)
    return {"status": "armed", "steering": _steering_spec}


@app.post("/steer_disarm")
def steer_disarm():
    """전 target 무개입 전환 (identity/None — off≡identity 규약). 재기동 없이."""
    if _gated_registry:
        raise HTTPException(status_code=409, detail="gated 등록 상태 — /steering_phase 로 제어")
    n = 0
    for (pathway, layer), ent in _arm_registry.items():
        hook = ent["hook"]
        if ent["family"] == "setpoint":
            if hasattr(hook, "set_segment"):
                try:
                    hook.set_segment(None)
                except Exception:  # noqa: BLE001 — segment 미사용 hook
                    pass
            hook.set_vector(None)
        else:
            first = hook._M_seq[0]
            eye = np.eye(first.shape[0])
            hook.set_matrices([eye] * len(hook._M_seq) if hook.per_step else eye)
        n += 1
        print(f"[steer-rearm] target=({pathway},{layer}) disarmed", flush=True)
    _update_steering_spec(mode="disarmed", op="none", layers=[], beta=0.0, alpha=None,
                          key=None, token_select=None, denoise=None, npz_shas=[])
    return {"status": "disarmed", "targets": n}


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
    # online failure detector: episode 경계에서 (h,c)·누적평균·step 카운터 리셋.
    # 리셋을 빠뜨리면 이전 판의 상태가 이어져 발화 시점이 오염된다.
    if _failure_detector is not None:
        _failure_detector.reset()
    # condg: 속도 차분 버퍼(직전 record 상태) 초기화 — 판 경계를 넘겨 이으면 첫 record
    # 속도가 이전 판의 잔재가 된다.
    for hook in _condg_hooks:
        hook.reset_state()
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


# ─── per-step 게이트 (docs/steering/47) ────────────────────────────────────────
# 규약: 1차 pass 는 **hook 전부 off** 로 돌려 자연 활성화 x_t 를 얻고(detector 순환
# 차단, §1-2), detector 가 발화한 record 에서만 **DiT-only 2차 pass** 로 개입한다.
# 개입은 그 record 1회성 — 다음 record 는 다시 무개입이 기본값(latch 폐기).
_PERSTEP_OPS = ("setm", "condg", "reseed", "rsn_llr", "rsn_rand")
# rsn_* = best-of-N 재샘플: 후보 n 개를 DiT-only 로 뽑아 하나만 실행(순수 재샘플 —
# setm/condg 훅은 적용하지 않는다). 선택 규칙만 다르다 (llr argmin vs 무작위).
_PERSTEP_RESAMPLE_OPS = ("rsn_llr", "rsn_rand")
_PERSTEP_DEFAULT_RESEED_OFFSET = 900000
# setm/condg 가 발화했는데 phase 미등록일 때: skip=무개입 | reseed=reseed 로 대체 개입
_PERSTEP_FALLBACKS = ("skip", "reseed")
_PERSTEP_DEFAULT_N = 8
_PERSTEP_MAX_N = 32
# action_head.get_action 호출 인자 캐시 — backbone(VL) 재실행 없이 DiT 만 다시 돌린다.
_dit_call_cache: dict = {}


def _shallow_copy_model_inputs(obj):
    """``BatchFeature`` (또는 dict) 얕은 복사 — 키 in-place 교체로부터 캐시 보호.

    값(텐서)은 공유하고 매핑만 새로 만든다. ``process_backbone_output`` 이 하는 건
    ``backbone_output["backbone_features"] = ...`` (키 재바인딩)이라 이걸로 충분하다.
    """
    if type(obj) is dict:
        return dict(obj)
    return type(obj)(data=dict(obj))  # BatchFeature(data=...) — 속성 접근 보존


def _parse_perstep_gate(payload: dict) -> dict | None:
    """payload 의 ``perstep_gate`` / ``perstep_debug_rerun`` 파싱 + 사전 배선 검증.

    payload 계약:
      ``perstep_gate``: {"op": "setm"|"condg"|"reseed"|"rsn_llr"|"rsn_rand"|null,
                         "reseed_offset": int (기본 900000),
                         "n": int (rsn_* 후보 수, 기본 8, 1≤n≤32),
                         "fallback": "skip"(기본)|"reseed" — setm/condg 가 발화했는데
                           현재 phase 에 연산자가 없을 때의 처리. skip=무개입,
                           reseed=reseed 2차 pass 로 대체 개입. rsn_*/reseed 는 무관}
      ``perstep_debug_rerun``: bool — 발화 무관하게 hook off·같은 seed 로 2차 실행해
      배관 동치(max|Δaction|)만 재는 스모크 모드 (응답 action 은 1차 것 유지).

    없으면 None (기존 경로 그대로). 오배선은 전부 명시 에러 — 조용히 개입 없는
    "개입 arm" 이 도는 사고 방지.
    """
    raw = payload.get("perstep_gate")
    debug = bool(payload.get("perstep_debug_rerun"))
    if raw is None and not debug:
        return None
    if raw is not None and not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="perstep_gate 는 dict 여야 한다")
    cfg = dict(raw or {})

    op = cfg.get("op")
    if op is not None:
        op = str(op).strip().lower()
        if op in ("", "none", "null"):
            op = None
    if op is not None and op not in _PERSTEP_OPS:
        raise HTTPException(
            status_code=422,
            detail=f"perstep_gate.op 불명: {op!r} (허용 {list(_PERSTEP_OPS)} 또는 null)",
        )

    if _policy_type != "groot":
        # DiT-only 재실행은 GR00T action_head 구조 전용 (fail-loud, 무음 no-op 금지)
        raise HTTPException(
            status_code=409,
            detail=f"per-step 게이트는 policy_type='groot' 전용 (현재 {_policy_type!r})",
        )
    if _failure_detector is None and not debug:
        raise HTTPException(
            status_code=409,
            detail="per-step 게이트는 --failure-detector 필요 (게이트 신호 없음)",
        )
    if _patch_hooks:
        # patch hook 은 "요청 1개 = record 1개" 커서 규약 — 2차 pass 재실행이 발화
        # 횟수를 2배로 만들어 over-fire 가드를 터뜨린다 (동시 사용 금지).
        raise HTTPException(
            status_code=409,
            detail="per-step 게이트는 patch serve(--patch-layers)와 동시 사용 불가",
        )
    if payload.get("skip_features") and _failure_detector is None:
        # detector 없는 skip_features 는 hook 없는 chunk 경로로 빠져 인자 캐시가 안 찬다
        raise HTTPException(
            status_code=409,
            detail="per-step 게이트 + skip_features 는 detector 켜진 serve 에서만 가능",
        )
    if op in ("setm", "condg"):
        if not _gated_registry:
            raise HTTPException(
                status_code=409,
                detail=f"perstep op={op} 인데 gated steering 미등록 (serve 재기동 필요)",
            )
        want_family = "condg" if op == "condg" else "setpoint"
        fams = {ent["family"] for ent in _arm_registry.values()}
        if want_family not in fams:
            raise HTTPException(
                status_code=409,
                detail=f"perstep op={op}(family={want_family}) != 등록 family {sorted(fams)}",
            )

    if op == "rsn_llr" and _llr_scorer is None:
        # 채점기 없이 rsn_llr 를 돌리면 조용히 "후보 0 고정"=reseed 1회 arm 이 된다.
        raise HTTPException(
            status_code=409,
            detail="perstep op=rsn_llr 인데 LLR 채점기 미로드 — serve 를 --llr-bundle 로 재기동",
        )

    try:
        offset = int(cfg.get("reseed_offset", _PERSTEP_DEFAULT_RESEED_OFFSET))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="perstep_gate.reseed_offset 은 int") from exc

    fallback = cfg.get("fallback", "skip")
    fallback = str(fallback).strip().lower() if fallback is not None else "skip"
    if fallback not in _PERSTEP_FALLBACKS:
        raise HTTPException(
            status_code=422,
            detail=f"perstep_gate.fallback 불명: {fallback!r} (허용 {list(_PERSTEP_FALLBACKS)})",
        )

    try:
        n_cand = int(cfg.get("n", _PERSTEP_DEFAULT_N))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="perstep_gate.n 은 int") from exc
    if not (1 <= n_cand <= _PERSTEP_MAX_N):
        raise HTTPException(
            status_code=422,
            detail=f"perstep_gate.n={n_cand} 범위 밖 (1≤n≤{_PERSTEP_MAX_N})",
        )

    return {"op": op, "reseed_offset": offset, "n": n_cand, "fallback": fallback,
            "debug_rerun": debug}


def _ensure_dit_rerun_wrap() -> None:
    """``action_head.get_action`` 을 감싸 호출 인자를 캐시 (2차 pass 재실행용).

    ``flow_matching_action_head.process_backbone_output`` 이
    ``backbone_output["backbone_features"]`` 를 **in-place 로 덮어쓴다** — 그래서
    원함수 호출 **전에** 얕은 복사를 떠 둔다. 아니면 2차 pass 가 vlln·vl_self_attention
    을 두 번 먹인 값을 받아 조용히 다른 조건이 된다.
    """
    if _dit_call_cache.get("installed"):
        return
    groot_model = getattr(policy, "_groot_model", None)
    if groot_model is None:
        raise HTTPException(status_code=500, detail="per-step 게이트: policy._groot_model 없음")
    head = groot_model.action_head
    orig = head.get_action  # 클래스 구현에 바인딩된 원함수 (래핑 전에 확보 — 재귀 방지)

    def _capturing_get_action(self, backbone_output, action_input):  # noqa: ARG001
        _dit_call_cache["backbone_output"] = _shallow_copy_model_inputs(backbone_output)
        _dit_call_cache["action_input"] = _shallow_copy_model_inputs(action_input)
        return orig(backbone_output, action_input)

    head.get_action = types.MethodType(_capturing_get_action, head)
    _dit_call_cache["orig"] = orig
    _dit_call_cache["installed"] = True
    logger.info("per-step 게이트: action_head.get_action 인자 캐시 wrap 설치")


def _rerun_dit_only(*, capture: bool) -> tuple[torch.Tensor, np.ndarray | None]:
    """캐시된 (backbone_output, action_input) 으로 action_head 만 재실행 (2차 pass).

    backbone(VL)은 다시 돌지 않는다 — 같은 관측·같은 VL 조건 위에서 denoise 만 다시
    한다. 반환 action 은 ``predict_action_chunk`` 와 같은 [B,H,D] raw 텐서(원 action_dim
    슬라이스 완료). ``capture=True`` 면 별도 ``SafeFeatureCapture`` 로 2차 활성화
    ([L,K,T,D])를 함께 반환한다 (1차 캡처 컨텍스트 밖 — [K]→[2K] 오염 방지).
    """
    if "backbone_output" not in _dit_call_cache:
        raise HTTPException(
            status_code=500,
            detail="per-step 게이트: 1차 pass 의 action_head 인자 캐시가 비었다 (wrap 미발화)",
        )
    bo_cached = _dit_call_cache["backbone_output"]
    ai_cached = _dit_call_cache["action_input"]
    # 2차 호출도 복사본으로 — get_action 이 backbone_features 를 다시 in-place 덮어쓴다
    bo = _shallow_copy_model_inputs(bo_cached)
    ai = _shallow_copy_model_inputs(ai_cached)
    orig = _dit_call_cache["orig"]

    device = next(policy.parameters()).device
    use_bf16 = bool(getattr(policy.config, "use_bf16", False))
    # per-step hook 의 denoise 카운터·condg call 카운터를 2차 pass 용으로 리셋
    # (안 하면 over-fire 가드가 터지거나 apply_call 지점이 어긋난다).
    # patch hook 은 제외 — 그 reset 은 record cursor 전진을 겸한다(요청당 1회).
    _reset_steering_step_counters(include_patch=False)

    cap = None
    if capture:
        cap = safe_hooks.SafeFeatureCapture(
            policy,
            _policy_type,
            capture_vl=_capture_vl_features,
            groot_dit_layers=_groot_dit_capture_layers,
            pi05_expert_layers=_pi05_expert_capture_layers,
            groot_dit_token_pool=_groot_dit_token_pool,
            vl_capture_point=_groot_vl_capture_point,
        )
    with torch.inference_mode():
        with (cap if cap is not None else contextlib.nullcontext()):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = orig(bo, ai)
    _assert_per_step_hook_counts()

    actions = out["action_pred"]
    from lerobot.utils.constants import ACTION as _ACTION  # noqa: PLC0415

    original_action_dim = int(policy.config.output_features[_ACTION].shape[0])
    actions = actions[:, :, :original_action_dim]

    hidden2 = None
    if cap is not None:
        hidden2 = cap.assemble_blocks()
        if hidden2 is None:
            raise HTTPException(
                status_code=500, detail="per-step 게이트: 2차 pass 활성화 캡처 실패(무발화)"
            )
    return actions, hidden2


def _gated_phase_registered(phase: str | None) -> bool:
    """해당 phase 에 실제 연산자가 등록돼 있는지 (미등록이면 적용해도 identity)."""
    if not phase or not _gated_registry:
        return False
    return any(phase in table for table in _gated_registry["matrices"].values())


def _run_resample_gate(
    cfg: dict, op: str, seed1: int, extras: dict
) -> tuple[torch.Tensor, Any, int]:
    """best-of-N 재샘플 2차 pass: 후보 n 개 → 1개 선택. 반환 (action, hidden, seed).

    후보 i 의 seed = seed1 + reseed_offset + i (n=1 이면 기존 reseed 와 동일한 후보).
    **선택된 후보 = 실제로 실행된 세계** — detector 재step·응답 action 모두 그 후보 것을
    쓴다. setm/condg 훅은 얹지 않는다 (순수 재샘플 arm).

    발동 조건은 detector 발화뿐 — phase 조건은 없다. 채점 entry 는 후보 latent 의
    최근접 비-OOD 등록 entry (`LLRScorer.score_nearest`).
    """
    n_cand = int(cfg["n"])
    base = seed1 + int(cfg["reseed_offset"])
    cands: list[tuple[int, torch.Tensor, Any]] = []
    cand_ms: list[float] = []      # 후보별 DiT-only rerun 소요 (제어 주기 영향 정량화용)
    for i in range(n_cand):
        seed_i = base + i
        # 후보마다 전역 RNG 를 다시 심는다 (앞 후보가 소모한 상태를 물려받지 않도록).
        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
        # step counter reset 은 _rerun_dit_only 내부에서 수행 — 여기서 중복 호출 금지.
        _t_cand = time.perf_counter()
        act_i, hid_i = _rerun_dit_only(capture=True)
        cand_ms.append((time.perf_counter() - _t_cand) * 1e3)
        cands.append((seed_i, act_i, hid_i))

    cand_logs: list | None = None
    llrs: list[float | None] | None = None
    rejects: list[str | None] | None = None
    cand_entries: list[str | None] | None = None
    skipped: str | None = None
    scored = None
    # 채점은 **후보 자신의 latent 위치**로 entry 를 정한다 (score_nearest) — "현재 cluster
    # phase" 로 entry 를 조회하지 않는다. 발화 시점 online cluster 가 등록 entry 와 전면
    # 불일치해 (scene, phase) 조회는 전 케이스 fallback 으로 퇴화했다(설계 세션 실측).
    # scene 에 entry 가 없는 경우는 기동 시 --llr-scene 검증에서 이미 걸러진다.
    if _llr_scorer is not None:
        scored = [_llr_scorer.score_nearest(_raw_feature_from_hidden(h), _llr_scene)
                  for _, _, h in cands]
        llrs = [None if s["llr"] is None else float(s["llr"]) for s in scored]
        rejects = ["ood" if s["ood_reject"] else None for s in scored]
        cand_entries = [s["entry"] for s in scored]
        # 외삽 깊이 사후분석용 (연산자 설계 요청): 후보별 [log_s, log_f]
        cand_logs = [[float(s["log_s"]), float(s["log_f"])] for s in scored]

    if op == "rsn_rand":
        # 위약 arm: 같은 후보 풀에서 무작위 선택 (seed1 고정 → 재현 가능).
        # 채점기가 있으면 llr/entry 는 **기록만** 한다 (선택에는 쓰지 않는다).
        sel = int(np.random.RandomState(seed1).randint(n_cand))
    else:
        if scored is None:  # _parse_perstep_gate 가 이미 막지만 무음 퇴화 방지
            raise HTTPException(
                status_code=409, detail="perstep op=rsn_llr 인데 LLR 채점기 미로드")
        keep = [i for i, s in enumerate(scored) if not s["ood_reject"]]
        if not keep:
            # 전 후보 기각 → 후보 0 으로 **개입은 한다**(=reseed 1회). 선택이 LLR 이
            # 아니었음을 별도 필드로 남긴다 (무음 no-op 금지).
            sel = 0
            skipped = "llr_all_ood"
        else:
            sel = min(keep, key=lambda i: llrs[i])

    extras["features.perstep_cand_n"] = n_cand
    extras["features.perstep_cand_ms"] = [float(v) for v in cand_ms]
    extras["features.perstep_cand_llr"] = llrs
    extras["features.perstep_cand_entry"] = cand_entries
    extras["features.perstep_cand_logs"] = cand_logs
    extras["features.perstep_cand_sel"] = int(sel)
    extras["features.perstep_cand_reject"] = rejects
    if skipped is not None:
        # 주의: gate_skipped 가 아니다 — fallback 도 후보 0 으로 **개입은 일어난다**.
        # gate_skipped 는 "무개입" 전용(집계 applied_count 가 ¬skipped 로 세므로),
        # LLR 선별 불가 사유는 별도 필드로 남긴다.
        extras["features.perstep_llr_fallback"] = skipped
    seed_sel, action_sel, hidden_sel = cands[sel]
    return action_sel, hidden_sel, seed_sel


def _run_perstep_gate(
    cfg: dict, action1: torch.Tensor, hidden1, inference_seed: int | None
) -> tuple[torch.Tensor, Any, dict, dict | None]:
    """1차 pass 결과 → detector → (발화 시) 2차 pass. 반환 (action, hidden, extras, y_t).

    - detector 입력은 **항상 1차 pass(pre-hook)** 활성화 (docs/steering/47 §1-2).
    - 2차 pass 후 detector 상태를 1차 step 직전으로 되돌린 뒤 x_t' 로 다시 step —
      즉 상태 커밋은 실제로 실행된 활성화 쪽(h_t = h′).
    """
    extras: dict[str, Any] = {
        "features.perstep_fired": False,
        "features.perstep_op": None,
        "features.perstep_seed2": None,
    }
    snap = None
    fail1 = None
    if _failure_detector is not None:
        if hidden1 is None:
            raise HTTPException(status_code=500, detail="per-step 게이트: 1차 pass hidden 없음")
        snap = _failure_detector.snapshot()
        fail1 = _failure_from_hidden(np.asarray(hidden1))
        extras["features.perstep_fired"] = bool(fail1["fired"])

    # cluster phase 판정: **발화 여부와 무관하게 매 요청** 1차 pass(pre-hook) 활성화로
    # 계산해 응답에 싣는다 (클라이언트가 feature_phases 에 기록해야 하므로).
    cluster1 = None
    if _cluster_assigner is not None:
        if hidden1 is None:
            raise HTTPException(
                status_code=500, detail="per-step 게이트: cluster phase 용 1차 pass hidden 없음")
        cluster1 = _cluster_phase_from_hidden(np.asarray(hidden1))
        extras["features.perstep_cluster"] = cluster1["name"]
        extras["features.perstep_cluster_dist"] = cluster1["dist"]

    if cfg["debug_rerun"]:
        # 배관 동치 스모크: hook off·seed1 로 2차 실행 → raw action 동치 확인만.
        # 응답 action 은 1차 것 유지(개입 아님), detector 도 1차 step 만 커밋.
        if inference_seed is not None:
            torch.manual_seed(int(inference_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(inference_seed))
        action2, _ = _rerun_dit_only(capture=False)
        diff = float((action2.detach().float() - action1.detach().float()).abs().max().item())
        extras["features.perstep_debug_max_action_diff"] = diff
        return action1, hidden1, extras, fail1

    op = cfg["op"]
    if op is None or not extras["features.perstep_fired"]:
        return action1, hidden1, extras, fail1

    if inference_seed is None:
        raise HTTPException(
            status_code=422,
            detail="per-step 게이트 2차 pass 는 payload.inference_seed 필요 "
            "(noise 재현/재추첨의 기준 seed)",
        )
    seed1 = int(inference_seed)
    if cluster1 is not None:
        # 번들이 있으면 phase 는 serve 자체 판정("c0".."c{k-1}") — 클라이언트가 POST 한
        # GT phase 값은 **무시**한다 (online 자립 조건). 번들이 없으면 현행 POST current.
        cur_phase = cluster1["name"]
    else:
        cur_phase = _gated_registry.get("current") if _gated_registry else None

    # 2차 pass 전체 소요 (rsn 후보 루프+채점 포함, detector 재step 직전까지) — 제어
    # 주기 영향 정량화용. 무발화 record 는 여기 오지 않으므로 필드 자체가 없다.
    _t_rerun = time.perf_counter()
    if op in _PERSTEP_RESAMPLE_OPS:
        # 발동 조건 = SAFE 발화뿐 (phase 분기 없음). cur_phase 는 채점에 쓰지 않는다 —
        # 응답의 perstep_cluster 기록은 위에서 이미 실었다.
        action2, hidden2, seed2 = _run_resample_gate(cfg, op, seed1, extras)
    else:
        # 1차 pass 가 전역 RNG 를 소모했으므로 2차 직전 반드시 재설정:
        #   setm/condg → seed1 (1차와 같은 noise, 차이는 개입뿐)
        #   reseed     → seed1+offset (denoise noise 재추첨 자체가 개입)
        seed2 = seed1 + int(cfg["reseed_offset"]) if op == "reseed" else seed1
        torch.manual_seed(seed2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed2)

        if op in ("setm", "condg") and not _gated_phase_registered(cur_phase):
            # 현재 phase 에 연산자가 없으면 setm/condg 2차 pass 는 identity.
            if cfg["fallback"] != "reseed":
                # 개입 없음으로 커밋하고 사유를 데이터에 남긴다 (무음 no-op 금지).
                extras["features.perstep_gate_skipped"] = f"phase_unregistered:{cur_phase!r}"
                return action1, hidden1, extras, fail1
            # fallback=reseed: 연산자 대신 **reseed 2차 pass 로 대체 개입**(후보 1개).
            # 개입이 실제로 일어나므로 gate_skipped 가 아니라 별도 필드에 남긴다
            # (llr_fallback 규약과 동일 논리 — 집계 applied_count 가 ¬skipped 로 센다).
            seed2 = seed1 + int(cfg["reseed_offset"])
            torch.manual_seed(seed2)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed2)
            extras["features.perstep_fallback"] = f"reseed:phase_unregistered:{cur_phase}"
            # 아래 공통 2차 pass 로 내려간다 — 연산자 적용만 건너뛴다(=reseed 와 동일).
            op_apply = None
        else:
            op_apply = op

        applied = False
        try:
            if op_apply in ("setm", "condg"):
                # 마지막으로 POST 된 phase/scene 상태를 그 record 에만 적용
                _apply_steering_phase_state(cur_phase, _gated_registry.get("current_scene"))
                applied = True
            action2, hidden2 = _rerun_dit_only(capture=True)
        finally:
            if applied:
                _steering_phase_off()

    extras["features.perstep_rerun_ms"] = (time.perf_counter() - _t_rerun) * 1e3

    if _failure_detector is not None:
        _failure_detector.restore(snap)
        fail2 = _failure_from_hidden(np.asarray(hidden2))
        extras["features.failure_score_post"] = fail2["score"]
    extras["features.perstep_op"] = op
    extras["features.perstep_seed2"] = seed2
    return action2, hidden2, extras, fail1


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
    if payload.get("perstep_gate") is not None or payload.get("perstep_debug_rerun"):
        # per-step 게이트는 활성화(detector 입력)가 필요 — /act 는 16-큐 팝이라 불가
        raise HTTPException(
            status_code=409,
            detail="per-step 게이트는 /act 미지원 — /act_with_features 를 사용하라",
        )
    inference_seed = _apply_inference_seed(payload)
    _reset_steering_step_counters()
    _inject_condg_state(payload)
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
    has_feature=True 와 unified features.hidden_states blob 이
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
    _inject_condg_state(payload)
    batch = parse_payload(payload)
    batch = _apply_input_remap(batch)

    if preprocessor is not None:
        batch = preprocessor(batch)

    # per-step 게이트 (docs/steering/47): 1차 pass 는 반드시 무개입이어야 하므로
    # 요청 시작에 hook 을 전부 off 하고, action_head 인자 캐시 wrap 을 보장한다.
    perstep_cfg = _parse_perstep_gate(payload)
    if perstep_cfg is not None:
        _ensure_dit_rerun_wrap()
        _steering_phase_off()

    # detector 가 켜져 있으면 skip_features 는 "hook 없이 돌라"가 아니라 **"blob 만 빼라"**
    # 로 해석한다 — detector 는 hidden 이 있어야 점수를 낸다. 아래 hook 경로가 hidden 을
    # 만들고 응답에서 blob 만 억제하므로 --no-features eval 의 응답 크기 이점은 유지된다
    # (chunk 추론 경로도 동일: run_with_features 도 predict_action_chunk 를 부른다).
    if payload.get("skip_features") and _failure_detector is None:
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
    )
    if _policy_type == "groot":
        _assert_per_step_hook_counts()

    perstep_extras: dict = {}
    fail_precomputed: dict | None = None
    if perstep_cfg is not None:
        # 발화 시 action·hidden 이 2차 pass 것으로 교체된다 (응답 action = 실행된 것).
        action, hidden, perstep_extras, fail_precomputed = _run_perstep_gate(
            perstep_cfg, action, hidden, inference_seed
        )

    action = _postprocess_action_preserve_chunk(action)
    action_np = _action_to_emit_array(action)

    result = _emit_subkeys(action_np, profile)
    # blob 억제 모드(detector ON + skip_features): 점수만 싣고 hidden 은 안 보낸다.
    suppress_blob = bool(payload.get("skip_features")) and _failure_detector is not None
    if hidden is not None:
        hidden_np = np.asarray(hidden)
        # online failure detector — blob 억제 여부와 무관하게 매 record 1 step 전진.
        if _failure_detector is not None:
            # per-step 게이트에서는 1차 pass(pre-hook, y_t) 점수를 그대로 싣는다 —
            # hidden_np 는 2차 pass 것이라 여기서 다시 step 하면 순환·이중 전진.
            fail = (
                fail_precomputed
                if fail_precomputed is not None
                else _failure_from_hidden(hidden_np)
            )
            result["features.failure_score"] = fail["score"]
            result["features.failure_fired"] = fail["fired"]
            result["features.failure_delta"] = fail["delta"]
            result["features.failure_step"] = fail["t"]
        # cluster phase — per-step 게이트 경로에서는 1차 pass 것을 perstep_extras 가
        # 이미 싣는다 (여기 hidden_np 는 2차 pass 라 다시 재면 좌표가 어긋난다).
        if _cluster_assigner is not None and perstep_cfg is None:
            cl = _cluster_phase_from_hidden(hidden_np)
            result["features.perstep_cluster"] = cl["name"]
            result["features.perstep_cluster_dist"] = cl["dist"]
    if suppress_blob and hidden is not None:
        # 클라이언트(VLAClient)는 blob 이 없고 has_feature=False 면 features=None 을 받는다
        # → --no-features 수집기의 "skip_features 인데 features 가 왔다" 가드와 양립.
        result["has_feature"] = False
        result["skip_features"] = True
    elif hidden is not None:
        result["has_feature"] = True
        # 통일 /act_with_features 계약(VLAClient·GR00T HTTP)만 발송. legacy
        # hidden_states_b64 이중 발송은 2026-08-10 제거 — 같은 배열을 두 번 실어
        # 응답이 2배였다 (VLAClient 는 통일 blob 우선이라 무영향, 폴백은 클라이언트에 잔존).
        result["features.hidden_states"] = encode_feature_blob(hidden_np)
        # inference-time action-phase readout (DiT residual → AE/SAE kmeans phase).
        if _phase_readouts:
            phase = _phase_from_hidden(hidden_np)
            if phase is not None:
                result["features.phase"] = phase
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
    # per-step 게이트 보고 필드 (발화 여부·적용 연산자·2차 seed·post 점수·디버그 diff)
    result.update(perstep_extras)
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
        # condg 는 상태 의존이라 정적 지문만으론 부족 — 현재 arm/phase·게이트 통계 노출
        condg=[h.status() for h in _condg_hooks] or None,
        patch=_patch_spec or None,
        phase_readout=_phase_spec or None,
        failure_detector=_failure_spec or None,
        cluster_phase=_cluster_spec or None,
        # 러너 preflight: rsn_llr arm 발사 전 채점기 로드 여부를 /health 로 확인
        llr_scorer=_llr_spec or None,
        # exp4-1: client 가 사이드카에 GPU 를 기록해 arm×GPU confound 를 사후 감사
        serve_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        # docs/04 규약 — rollout 인덱스의 machine·ckpt 열 원천 (헬퍼가 단일 출처)
        **serve_provenance(_profile),
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
    return metadata


def _update_steering_spec(*, mode, op, layers, beta, alpha, key,
                          token_select, denoise, npz_shas, phases=None, extra=None):
    """/health 스티어링 지문 갱신 — 기동·재무장(/steer_arm) 공용 (armsig 의 원천).

    ``extra``: 연산자 고유 필드(예 condg 의 mode/gate)를 그대로 얹는다.
    """
    global _steering_spec
    _steering_spec = {
        "mode": mode,
        "op": op,
        "layers": [int(x) for x in layers] if layers else [],
        "beta": float(beta),
        "alpha": None if alpha is None else float(alpha),
        "key": key,
        "token_select": token_select,
        "denoise": denoise,
        "npz_shas": sorted(set(npz_shas)),
        "phases": sorted(phases) if phases else None,
    }
    if extra:
        _steering_spec.update(extra)


def _register_steering_if_requested(loaded_policy, args):
    global _steering
    steering_npz = getattr(args, "steering_npz", None)
    steering_npz_dir = getattr(args, "steering_npz_dir", None)
    steering_layers = getattr(args, "steering_layers", None)
    condg_npz = getattr(args, "condg_npz", None)
    if (not steering_npz and not steering_npz_dir and not condg_npz
            and not getattr(args, "steering_phase_npz_base", None)):
        return None
    if _policy_type not in ("groot", "pi05"):
        raise ValueError("Conceptor steering requires policy_type in {'groot', 'pi05'}")

    from steering_hooks import (
        CondGuidanceSteering,
        ConceptorSteering,
        Pi05ConceptorSteering,
        SetpointSteering,
        load_cond_guidance,
        load_steering_matrices_per_step,
        load_steering_matrix,
        load_steering_segment,
        load_steering_setpoint,
    )

    # unregister any previously-registered hooks (reload-safe)
    for _h in _steering:
        _h.unregister()
    _steering = []
    _arm_registry.clear()
    _condg_hooks.clear()

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

    def _set_steering_spec(mode: str, layers, phases=None, op: str = "conceptor"):
        _update_steering_spec(mode=mode, op=op, layers=layers, beta=beta, alpha=alpha,
                              key=key, token_select=token_select, denoise=denoise,
                              npz_shas=loaded_npz_shas, phases=phases)

    # --- condg: 상태-조건부 대조 guidance (docs/steering/44) -----------------------
    # 단일 NPZ 안에 phase 전부(W_s/W_f/τ) + scene별 중심화 파라미터가 들어 있어
    # hook 1개가 /steering_phase 로 phase·scene 만 스위칭한다 (gated 계열과 동거하되
    # NPZ 디렉토리 계약은 쓰지 않는다).
    if condg_npz:
        global _gated_registry
        _want_op = getattr(args, "steering_op", "auto") or "auto"
        if _want_op not in ("auto", "condg"):
            raise ValueError(f"--condg-npz 는 --steering-op condg 전용 (got {_want_op})")
        if _policy_type != "groot":
            raise ValueError("--condg-npz 는 groot dit pathway 전용")
        if steering_npz or steering_npz_dir or getattr(args, "steering_phase_npz_base", None):
            raise ValueError("--condg-npz 는 다른 --steering-npz*/phase-base 와 상호 배타")
        if per_step:
            raise ValueError("condg 는 --steering-denoise global 전용 (마지막 denoise call 한정)")
        groot_model = getattr(loaded_policy, "_groot_model", None)
        if groot_model is None:
            raise ValueError("GR00T LeRobot policy is missing _groot_model for steering")
        import hashlib as _hashlib_cg

        loaded_npz_shas.append(
            _hashlib_cg.sha256(Path(condg_npz).read_bytes()).hexdigest()[:12]
        )
        params = load_cond_guidance(condg_npz)
        # scene 목록은 phase별(중심화가 phase 단위) — 진단용으로 합집합만 노출.
        _cg_scenes = sorted({sc for v in params["phases"].values()
                             for sc in v["scenes"]})
        # layer: --steering-layer 우선, 미지정이면 fit 표적 L12
        _cg_layer = getattr(args, "steering_layer", None)
        _cg_layer = 12 if _cg_layer is None else int(_cg_layer)
        _cg_mode = getattr(args, "condg_mode", "condg") or "condg"
        _cg_gate = bool(getattr(args, "condg_gate", True))
        _cg_token = token_select or "all"
        _cg_apply = getattr(args, "condg_apply_call", "last") or "last"
        hook = CondGuidanceSteering(
            groot_model, params, beta, layer=_cg_layer, mode=_cg_mode,
            token_select=_cg_token, gate=_cg_gate, apply_call=_cg_apply,
        ).register()
        _steering.append(hook)
        _condg_hooks.append(hook)
        _arm_registry[("dit", _cg_layer)] = {
            "hook": hook, "family": "condg", "per_step": False,
        }
        _registered_phases = sorted(p for p, v in params["phases"].items() if v["registered"])
        # /steering_phase 가 hooks 를 순회하며 set_phase 로 스위칭한다. matrices 는
        # 응답의 gated 플래그 계산용(등록 phase 집합)으로만 쓰인다.
        _gated_registry = {
            "hooks": {_cg_layer: hook},
            "matrices": {_cg_layer: {ph: None for ph in _registered_phases}},
            "identity": {_cg_layer: None},
            "current": None,
        }
        _update_steering_spec(
            mode="gated", op="condg", layers=[_cg_layer], beta=beta, alpha=alpha,
            key=None, token_select=_cg_token, denoise="last_call",
            npz_shas=loaded_npz_shas, phases=_registered_phases,
            extra={"condg_mode": _cg_mode, "condg_gate": _cg_gate,
                   "condg_apply_call": _cg_apply,
                   "condg_state_dim": hook.state_dim,
                   "condg_num_denoise": hook.num_denoise,
                   "condg_scenes": _cg_scenes},
        )
        logger.info(
            "condg steering registered: npz=%s layer=%s mode=%s gate=%s beta=%s "
            "token_select=%s phases=%s scenes=%s",
            condg_npz, _cg_layer, _cg_mode, _cg_gate, beta, _cg_token,
            _registered_phases, _cg_scenes,
        )
        print(
            f"[steer-registered] path=condg op=condg mode={_cg_mode} gate={_cg_gate} "
            f"layer={_cg_layer} beta={beta:g} token_select={_cg_token} "
            f"phases={','.join(_registered_phases)} "
            f"unregistered={','.join(sorted(set(params['phases']) - set(_registered_phases)))} "
            f"num_denoise={hook.num_denoise} dim={hook.expected_dim}",
            flush=True,
        )
        for ph in sorted(params["phases"]):
            ent = params["phases"][ph]
            print(f"[steer-norms] op=condg phase={ph} registered={ent['registered']} "
                  f"tau={ent['tau']:.6f} B={ent['B']} "
                  f"‖W_s‖={float(np.linalg.norm(ent['W_s'])):.4f} "
                  f"‖W_f‖={float(np.linalg.norm(ent['W_f'])):.4f}", flush=True)
        return _steering

    # --- Oracle phase-gated multi-layer steering: /steering_phase 로 M 스위칭 ---
    phase_base = getattr(args, "steering_phase_npz_base", None)
    if phase_base and steering_layers:
        # (_gated_registry 는 위 condg 분기에서 이미 global 선언됨)
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
        # 연산자 판정 (exp4-1): NPZ 키 스니핑(*_v_steer=setpoint) + --steering-op assert.
        # 러너가 arm 마다 op 를 명시해 NPZ 오배치를 기동 시점에 잡는다 (무음 오적용 방지).
        first_npz = base / phases[0] / f"dit_L{layers[0]}" / "conceptors.npz"
        _first_keys = _np.load(first_npz).files
        if any(k.endswith("_v_seg") for k in _first_keys):
            detected_op = "setpoint_seg"      # exp4-1 v2: 세그먼트 방향 + 토큰별 setpoint
        elif any(k.endswith("_v_steer") for k in _first_keys):
            detected_op = "setpoint"          # v1 pooled (배포 금지 — 공간 불일치)
        else:
            detected_op = "conceptor" 
        want_op = getattr(args, "steering_op", "auto") or "auto"
        if want_op != "auto" and want_op != detected_op:
            raise ValueError(
                f"--steering-op {want_op} != NPZ 감지 {detected_op} ({first_npz})"
            )
        op = detected_op
        if op == "setpoint" and per_step:
            raise ValueError("setpoint(setM) 은 --steering-denoise global 전용")

        def _load_setpoint(npz_path):
            import hashlib as _hashlib

            loaded_npz_shas.append(
                _hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()[:12]
            )
            if op == "setpoint_seg":
                # fit(토큰 보존) ↔ serve(token_select) 정합 게이트: 세그먼트 연산자는
                # 전 토큰에 위치별로 적용해야 한다 (2026-07-23 배선 회귀 재발 방지)
                if (token_select or "last_horizon") != "all":
                    raise ValueError(
                        "setpoint_seg 는 --steering-token-select all 필수 "
                        f"(현재 {token_select!r}) — fit 은 전 토큰 위치별 s_t 를 산출했다")
                return load_steering_segment(str(npz_path))
            return load_steering_setpoint(str(npz_path), alpha=alpha)

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
                matrices[lyr][ph] = (
                    _load_setpoint(npz_path) if op in ("setpoint", "setpoint_seg")
                    else _load_matrices(npz_path)
                )
            if op in ("setpoint", "setpoint_seg"):
                # off = set_vector/set_segment(None) 명시적 no-op — identity 행렬 불필요
                identity[lyr] = None
                hook = SetpointSteering(
                    groot_model, None, beta, layer=lyr, token_select=token_select,
                ).register()
            else:
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
            _arm_registry[("dit", lyr)] = {
                "hook": hook,
                "family": "setpoint" if op.startswith("setpoint") else "conceptor",
                "per_step": per_step,
            }
        _gated_registry = {"hooks": hooks, "matrices": matrices, "identity": identity, "current": None}
        logger.info(
            "Phase-gated %s steering registered: base=%s layers=%s phases=%s "
            "beta=%s token_select=%s denoise=%s",
            op, phase_base, layers, phases, beta,
            token_select or "last_horizon(default)", denoise,
        )
        _set_steering_spec("gated", layers, phases, op=op)
        # 러너 preflight 대조용 (module logger 는 serve 로그 파일에 안 남음 — print 필수)
        print(
            f"[steer-registered] path=gated op={op} "
            f"layers={','.join(str(x) for x in layers)} "
            f"phases={','.join(phases)} beta={beta:g} "
            f"token_select={token_select or 'last_horizon(default)'} denoise={denoise}",
            flush=True,
        )
        # dose 로깅 근거 (Gate 2 높음#10): phase×step 별 ‖M−I‖F — 사이드카의
        # feature_phases + phase_gated_flags 와 조합해 오프라인에서 누적 dose 재구성.
        # setpoint 는 실 dose 가 상태 의존(β|(h·r̂)−s|)이라 β·‖r̂‖·s 만 로그.
        for lyr in layers:
            for ph in phases:
                mats = matrices[lyr][ph]
                if op == "setpoint_seg":
                    v_seg, s_tok, _b, mk = mats
                    print(f"[steer-norms] layer={lyr} phase={ph} op=setpoint_seg "
                          f"beta={beta:g} S={v_seg.shape[0]} T={s_tok.shape[0]} "
                          f"mask={list(mk)} s_tok[min,max]="
                          f"[{float(s_tok.min()):.2f},{float(s_tok.max()):.2f}]", flush=True)
                    continue
                if op == "setpoint":
                    v, s = mats
                    print(
                        f"[steer-norms] layer={lyr} phase={ph} op=setpoint beta={beta:g} "
                        f"norm_v={float(_np.linalg.norm(v)):.6f} s={s:.6f} "
                        f"dose=state-dependent", flush=True,
                    )
                    continue
                seq = mats if isinstance(mats, list) else [mats]
                for k, M in enumerate(seq):
                    dI = float(_np.linalg.norm(M - _np.eye(M.shape[0]), "fro"))
                    print(f"[steer-norms] layer={lyr} phase={ph} step={k} fro_M_minus_I={dI:.6f}",
                          flush=True)
        return _steering

    # --- VL setpoint (exp5-2 setM VL): 단일 NPZ + pathway=vl, 토큰-평균 이동 -------------
    # dit setpoint 는 phase 디렉토리(gated) 계약이지만 VL 은 phase 게이팅 없이
    # 서버 수명 내내 상시 적용(C1 = 전 구간 섭동)이라 단일 NPZ 경로를 쓴다.
    if steering_npz and _policy_type == "groot" and not steering_npz_dir:
        import numpy as _np_vl

        _keys = _np_vl.load(steering_npz).files
        _pathway_arg = getattr(args, "steering_pathway", "dit")
        if any(k.endswith("_v_seg") for k in _keys):
            _detected = "setpoint_seg"
        elif any(k.endswith("_v_steer") for k in _keys):
            _detected = "setpoint_vl" if _pathway_arg == "vl" else "setpoint"
        else:
            _detected = "conceptor"
        _want_op = getattr(args, "steering_op", "auto") or "auto"
        if _want_op != "auto" and _want_op != _detected:
            raise ValueError(
                f"--steering-op {_want_op} != NPZ 감지 {_detected} ({steering_npz}, "
                f"pathway={_pathway_arg})"
            )
        if _detected == "setpoint_vl":
            if per_step:
                raise ValueError("setpoint_vl 은 --steering-denoise global 전용")
            groot_model = getattr(loaded_policy, "_groot_model", None)
            if groot_model is None:
                raise ValueError("GR00T LeRobot policy is missing _groot_model for steering")
            import hashlib as _hashlib_vl

            loaded_npz_shas.append(
                _hashlib_vl.sha256(Path(steering_npz).read_bytes()).hexdigest()[:12]
            )
            v_vl, s_vl = load_steering_setpoint(str(steering_npz), alpha=alpha)
            _vl_hook = SetpointSteering(
                groot_model, (v_vl, s_vl), beta, pathway="vl",
                token_select=token_select,
            ).register()
            _steering.append(_vl_hook)
            _arm_registry[("vl", None)] = {
                "hook": _vl_hook, "family": "setpoint", "per_step": False,
            }
            logger.info(
                "VL setpoint steering registered: npz=%s beta=%s alpha=%s dim=%s s=%.4f",
                steering_npz, beta, alpha, v_vl.shape[0], s_vl,
            )
            _set_steering_spec("single", [], op="setpoint_vl")
            print(
                f"[steer-registered] path=single op=setpoint_vl pathway=vl "
                f"beta={beta:g} dim={v_vl.shape[0]} s={s_vl:.4f} token_select=all "
                f"denoise={denoise}",
                flush=True,
            )
            # setpoint 는 실 dose 가 상태 의존(β|(m·r̂)−s|)이라 β·s 만 기록.
            print(f"[steer-norms] op=setpoint_vl beta={beta:g} s={s_vl:.4f} "
                  f"‖r̂‖={float(_np_vl.linalg.norm(v_vl)):.6f}", flush=True)
            return _steering

    # setpoint(setM) 은 gated 경로 전용 — 이하 경로에서 지정 시 fail loud
    if (getattr(args, "steering_op", "auto") or "auto") in ("setpoint", "setpoint_seg"):
        raise ValueError("--steering-op setpoint 는 --steering-phase-npz-base(gated) 전용")
    if (getattr(args, "steering_op", "auto") or "auto") == "setpoint_vl":
        raise ValueError(
            "--steering-op setpoint_vl 은 --steering-npz(단일) + --steering-pathway vl 전용"
        )

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
            _ml_hook = ConceptorSteering(
                groot_model, mat, pathway="dit", layer=lyr,
                token_select=token_select,
            ).register()
            _steering.append(_ml_hook)
            _arm_registry[("dit", lyr)] = {
                "hook": _ml_hook, "family": "conceptor", "per_step": per_step,
            }
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
    _single_hook = ConceptorSteering(
        groot_model, matrix, pathway=pathway, layer=layer,
        token_select=token_select,
    ).register()
    _steering.append(_single_hook)
    _arm_registry[(pathway, layer)] = {
        "hook": _single_hook, "family": "conceptor", "per_step": per_step,
    }
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
    _load_phase_readouts_if_requested(args)
    _load_failure_detector_if_requested(args)
    _load_cluster_phase_if_requested(args)
    _load_llr_scorer_if_requested(args)

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
        "--phase-readout",
        action="store_true",
        help=(
            "inference-time action-phase 판정 ON: DiT residual → task_classification "
            "AE/SAE(kmeans) 로 현재 phase 를 읽어 /act_with_features 응답의 "
            "features.phase 로 노출. --groot-dit-capture-layers(해당 layer 포함) + "
            "--groot-dit-token-pool all_token_full 필요."
        ),
    )
    parser.add_argument(
        "--phase-run-dirs",
        type=str,
        default="task_classification/runs/ae-log_likelihood-s0,"
                "task_classification/runs/sae-log_likelihood-s0",
        help="phase 분류기 run 디렉토리(model.pt) 콤마 구분. 이름은 basename 접두어(ae/sae).",
    )
    parser.add_argument(
        "--phase-pca",
        type=str,
        default="task_classification/datasets_local/phase_cls_pq3/"
                "derived/L12-D3-pca64w/pca.npz",
        help="PCA-whiten 통계(pca.npz) 경로 (mu/V/sqrt_lam).",
    )
    parser.add_argument(
        "--phase-map",
        type=str,
        default="task_classification/runs/cluster_phase_map.json",
        help="cluster→phase 이름 매핑 json (선택). 없으면 cluster id 만 반환.",
    )
    parser.add_argument(
        "--phase-layer",
        type=int,
        default=12,
        help="phase readout 에 쓰는 물리 DiT block layer (capture layers 에 있어야 함).",
    )
    parser.add_argument(
        "--phase-denoise",
        type=int,
        default=None,
        help="denoise step index (기본 None=마지막 step). 학습=마지막(K=4 의 3).",
    )
    parser.add_argument(
        "--phase-device",
        type=str,
        default="cpu",
        help="phase 분류기 device (cpu 권장 — 연산량 미미, GPU 경합 회피).",
    )
    parser.add_argument(
        "--failure-detector",
        type=str,
        default=None,
        metavar="CKPT.pt",
        help=(
            "online failure detector ON: failure_detector_sim.py 가 저장한 "
            "detector_<arm>_<model>_<slug|all>.pt 를 얹어 /act_with_features 응답에 "
            "features.failure_score / features.failure_fired 를 노출. "
            "--groot-dit-capture-layers(ckpt 의 layer 포함) + "
            "--groot-dit-token-pool all_token_full 필요."
        ),
    )
    parser.add_argument(
        "--failure-alpha",
        type=float,
        default=0.2,
        help="발화 임계 δ_t 의 CP 유의수준(=FPR 목표). ckpt cp_bands 에 있는 값이어야 함.",
    )
    parser.add_argument(
        "--failure-task",
        type=str,
        default=None,
        help=(
            "cp_bands 에서 쓸 task slug (mixed arm 은 밴드가 task 별 보정 — 필수). "
            "밴드가 하나뿐이면 생략 가능."
        ),
    )
    parser.add_argument(
        "--cluster-phase-bundle",
        type=str,
        default=None,
        metavar="BUNDLE.npz",
        help=(
            "per-step 게이트의 phase 를 GT POST 값 대신 serve 자체 activation cluster "
            "판정으로 정한다: ae_cluster.py --export-bundle 산출 NPZ(표준화+encoder+"
            "instruction 별 centers)를 얹어 phase 이름 'c0'..'c{k-1}' 을 낸다. "
            "응답에 features.perstep_cluster / features.perstep_cluster_dist 노출. "
            "--groot-dit-capture-layers(번들 layer=12 포함) + "
            "--groot-dit-token-pool all_token_full 필요. 미지정 시 현행(POST current) 유지."
        ),
    )
    parser.add_argument(
        "--llr-bundle",
        type=str,
        default=None,
        metavar="BUNDLE.npz",
        help=(
            "best-of-N 재샘플(perstep_gate.op=rsn_llr)의 후보 채점기 NPZ. phase 조건부 "
            "성공/실패 가우시안 로그우도비로 후보를 고르고(llr argmin), OOD 후보는 기각. "
            "NPZ 계약 = src/failure_online/llr_scorer.py docstring (단일 출처, "
            "등록 단위 (scene, phase) — --llr-scene 동반 필수). "
            "--groot-dit-capture-layers + --groot-dit-token-pool all_token_full 필요. "
            "미지정 시 rsn_llr 요청은 409 (rsn_rand 는 채점 없이 동작)."
        ),
    )
    parser.add_argument(
        "--llr-scene",
        type=str,
        default=None,
        metavar="SCENE",
        help=(
            "LLR 채점에 쓸 scene (int 또는 's3'). 번들 등록 단위가 (scene, phase) 라 "
            "--llr-bundle 지정 시 **필수** — 없거나 번들에 없는 scene 이면 기동 실패."
        ),
    )
    parser.add_argument(
        "--cluster-phase-task",
        type=str,
        default=None,
        help=(
            "번들에서 쓸 instruction slug (centers 는 instruction 별로 따로 적합 — 필수). "
            "번들에 slug 가 하나뿐이면 생략 가능."
        ),
    )
    parser.add_argument(
        "--cluster-phase-device",
        type=str,
        default="cpu",
        help="cluster phase 판정기 device (cpu 권장 — 연산량 미미, GPU 경합 회피).",
    )
    parser.add_argument(
        "--failure-device",
        type=str,
        default="cpu",
        help="detector device (cpu 권장 — 연산량 미미, GPU 경합 회피).",
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
        choices=("last_horizon", "all", "future"),
        default=None,
        help=(
            "Steering hook 의 적용 토큰. 미지정(None)=pathway 기본 보존"
            "(dit=last_horizon, vl=all). exp3 COAST 정렬은 dit 에 all 을 명시 주입. "
            "future=future 세그먼트만([1:T−horizon]) — setM future_only 와 정렬한 "
            "conceptor 계열 future arm (hook 은 이미 지원, steering_hooks.TOKEN_SELECTS)."
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
        "--steering-op",
        choices=("auto", "conceptor", "setpoint", "setpoint_seg", "setpoint_vl", "condg"),
        default="auto",
        help=(
            "steering 연산자 (exp4-1). auto=NPZ 키로 감지(*_v_steer=setpoint). "
            "명시 시 감지 결과와 불일치하면 기동 abort — 러너가 arm 마다 명시해 "
            "NPZ 오배치를 잡는다. setpoint(setM)는 gated 경로 전용, h'=h−β[(h·r̂)−s]r̂. "
            "setpoint_vl(exp5-2)은 --steering-npz 단일 NPZ + --steering-pathway vl 전용 — "
            "vlln 출력의 **토큰-평균**을 setpoint 로 이동. "
            "condg(docs/steering/44)는 --condg-npz 전용 — 상태-조건부 대조 guidance."
        ),
    )
    parser.add_argument(
        "--condg-npz",
        default=None,
        help=(
            "상태-조건부 대조 guidance(condg) NPZ (docs/steering/44 §4). phase별 "
            "W_s/W_f/tau/registered + scene별 mh/mp/sp + global fallback 을 담은 단일 "
            "파일. /steering_phase {\"phase\":…, \"scene\": int} 로 스위칭하고, /act 요청의 "
            "observation.state.{eef_pos_rel,eef_quat_rel,gripper_qpos} 로 상태를 주입한다. "
            "layer 는 --steering-layer(기본 12), β 는 --steering-beta."
        ),
    )
    parser.add_argument(
        "--condg-mode",
        choices=("condg", "hs"),
        default="condg",
        help=(
            "condg 적용식. condg=대조 투영(Δ=β⟨h̃−ĥ_s,d̂⟩d̂, d̂=normalize(ĥ_f−ĥ_s)) | "
            "hs=성공-모방 단독 ablation((1−β)h̃+β·ĥ_s)."
        ),
    )
    parser.add_argument(
        "--condg-apply-call",
        choices=("last", "first"),
        default="last",
        help=(
            "condg 개입 denoise call. last=k==K−1 (fit denoise_step 3 표적, 기본) | "
            "first=k==0 — fit --denoise-step 0 NPZ 전용 (τ·W 가 step-0 공간이어야 함)."
        ),
    )
    parser.add_argument(
        "--condg-gate",
        dest="condg_gate",
        action="store_true",
        default=True,
        help="condg margin 게이트 ON (기본) — m>τ 인 record 에서만 개입.",
    )
    parser.add_argument(
        "--no-condg-gate",
        dest="condg_gate",
        action="store_false",
        help="무게이트 ablation: 발화 후 전 record 상시 개입 (g≡1, 44 §3).",
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
