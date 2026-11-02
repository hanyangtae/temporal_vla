"""GR00T N1.5 RoboCasa365 checkpoint-120000 계약: config/metadata 고정, action codec, QAM flags 검증.

정본 설명은 같은 폴더 CONTRACT.md. 이 모듈은 numpy/json만 쓰므로 host CPU에서 검증 가능하다.
수치 출처 (2026-10-21 cached artifact 실측):
- config.json      sha256 6713ae6e... : action_dim 32, action_horizon 16, num_inference_timesteps 4,
                                         input_embedding_dim 1536 (=32 heads x 48), DiT 16 layers,
                                         DiT output_dim 1024 (proj_out_2) = hidden_size 1024
                                         (action_decoder 입력 폭).
- metadata.json    sha256 8be0fc60... : new_embodiment action 5 key = 12D, min_max/binary 통계.
- action 순서/모드  configs/policies/robocasa_n15_panda_omron_data_config.py (PandaOmron 순서).
  metadata.json의 dict 삽입 순서(알파벳)는 학습 concat 순서가 아니다.
- raw LeRobot action 12D 순서 OpenDrawer/20250819 meta/modality.json 실측:
  base_motion 0:4 · control_mode 4:5 · eef_pos 5:8 · eef_rot 8:11 · gripper 11:12.
  모델 순서로 바꿀 때는 이 metadata slice 로 명시적으로 재배열한다 (위치 가정 금지).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class ContractError(ValueError):
    """계약 위반. 호출자는 잡아서 우회하지 말고 중단한다."""


CHECKPOINT_SUBFOLDER = "gr00t_n1-5/multitask_learning/checkpoint-120000"
HF_REPO_CACHE_DIR = "models--robocasa--robocasa365_checkpoints"
# snapshot 14895998... 와 c484448a... 가 같은 blob 을 가리킴 (revision 이 아니라 내용으로 고정).
CONFIG_SHA256 = "6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526"
METADATA_SHA256 = "8be0fc606c9220356bad497feefc4ae4daaa05670acccc31caecaa7ae5590b69"
EMBODIMENT_TAG = "new_embodiment"

# (metadata key, dims, upstream normalization mode) — PandaOmron data config concat 순서.
ACTION_KEYS: tuple[tuple[str, int, str], ...] = (
    ("end_effector_position", 3, "min_max"),
    ("end_effector_rotation", 3, "min_max"),
    ("gripper_close", 1, "binary"),
    ("base_motion", 4, "min_max"),
    ("control_mode", 1, "binary"),
)
ENV_ACTION_DIM = 12
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 16
NUM_INFERENCE_STEPS = 4
NUM_TIMESTEP_BUCKETS = 1000
# DiT 내부 block residual 폭 (32 heads x 48). QAM 관측이 아니다 — config 검사용.
DIT_WIDTH = 1536
DIT_NUM_LAYERS = 16
# DiT proj_out_2 출력 = action_decoder 입력 폭 (config dit.output_dim = action_head hidden_size).
DIT_OUTPUT_DIM = 1024
VLM_CONTEXT_DIM = 2048  # backbone_embedding_dim / cross_attention_dim — 역시 QAM 관측 아님
# DiT 입력 시퀀스 = state 1 + future_tokens 32 + action 16 = 49 (docs/04 §6).
TOKEN_SLICES = {
    "state": slice(0, 1),
    "future": slice(1, 33),
    "action": slice(33, 49),
    "all": slice(0, 49),
}
DIT_SEQ_LEN = 49
# robocasa gym_wrapper: binary 출력 {0,1} → env {-1,+1} (임계 0.5).
ENV_BINARY_THRESHOLD = 0.5

# raw LeRobot action 12D 의 key 별 slice (meta/modality.json "action" 블록 형식).
RAW_LEROBOT_ACTION_SLICES: dict[str, tuple[int, int]] = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}

QAM_OB_DIM = DIT_OUTPUT_DIM
QAM_ACTION_DIM = ENV_ACTION_DIM
QAM_HORIZON = ACTION_HORIZON
QAM_FLOW_STEPS = NUM_INFERENCE_STEPS
CONTRACT_KEY = "groot_robocasa_contract"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_cached_checkpoint_dir(datasets_root: str | Path | None = None) -> Path:
    """HF cache 안에서 sha 가 일치하는 checkpoint-120000 폴더를 찾는다 (다운로드 없음)."""
    if datasets_root is None:
        from scripts.path_setup import DATA_ROOT  # 경로 단일 출처

        datasets_root = DATA_ROOT
    snapshots = Path(datasets_root) / "huggingface" / "hub" / HF_REPO_CACHE_DIR / "snapshots"
    for snap in sorted(snapshots.glob("*")):
        cand = snap / CHECKPOINT_SUBFOLDER
        cfg, meta = cand / "config.json", cand / "experiment_cfg" / "metadata.json"
        if cfg.is_file() and meta.is_file():
            if sha256_file(cfg) == CONFIG_SHA256 and sha256_file(meta) == METADATA_SHA256:
                return cand
    raise ContractError(f"cached checkpoint-120000 (sha 고정) 없음: {snapshots}")


def validate_model_config(config: Mapping[str, Any]) -> None:
    """action32/H16/4step/DiT1536 이 아닌 checkpoint (예: 7D, H4) 를 거부한다."""
    head = config.get("action_head_cfg") or {}
    dit = head.get("diffusion_model_cfg") or {}
    checks = {
        "model_type": (config.get("model_type"), "gr00t_n1_5"),
        "action_dim": (config.get("action_dim"), MODEL_ACTION_DIM),
        "action_horizon": (config.get("action_horizon"), ACTION_HORIZON),
        "action_head_cfg.action_dim": (head.get("action_dim"), MODEL_ACTION_DIM),
        "action_head_cfg.max_action_dim": (head.get("max_action_dim"), MODEL_ACTION_DIM),
        "action_head_cfg.action_horizon": (head.get("action_horizon"), ACTION_HORIZON),
        "action_head_cfg.num_inference_timesteps": (
            head.get("num_inference_timesteps"), NUM_INFERENCE_STEPS),
        "action_head_cfg.num_timestep_buckets": (
            head.get("num_timestep_buckets"), NUM_TIMESTEP_BUCKETS),
        "action_head_cfg.input_embedding_dim": (head.get("input_embedding_dim"), DIT_WIDTH),
        "action_head_cfg.hidden_size": (head.get("hidden_size"), DIT_OUTPUT_DIM),
        "action_head_cfg.backbone_embedding_dim": (
            head.get("backbone_embedding_dim"), VLM_CONTEXT_DIM),
        "dit.output_dim": (dit.get("output_dim"), DIT_OUTPUT_DIM),
        "dit.num_layers": (dit.get("num_layers"), DIT_NUM_LAYERS),
        "dit.inner_dim": (
            (dit.get("num_attention_heads") or 0) * (dit.get("attention_head_dim") or 0),
            DIT_WIDTH),
        "action_head_cfg.num_target_vision_tokens": (head.get("num_target_vision_tokens"), 32),
        "action_head_cfg.add_pos_embed": (head.get("add_pos_embed"), True),
    }
    bad = {k: got for k, (got, want) in checks.items() if got != want}
    if bad:
        raise ContractError(
            f"GR00T config 계약 불일치 (action32/H16/4step/DiT1536/decoder입력1024 필요): {bad}")


@dataclass(frozen=True)
class ActionCodec:
    """Upstream Normalizer(min_max/binary) 의 numpy 미러 + active12/padding 마스크."""

    min: np.ndarray            # (12,)
    max: np.ndarray            # (12,)
    modes: tuple[str, ...]     # 길이 12, "min_max" | "binary"
    names: tuple[str, ...] = field(default=())

    @property
    def binary_mask(self) -> np.ndarray:
        return np.array([m == "binary" for m in self.modes])

    @property
    def degenerate_mask(self) -> np.ndarray:
        """min==max 인 min_max 차원 (upstream 은 normalize 시 0, inverse 시 min 상수)."""
        return (~self.binary_mask) & (self.min == self.max)

    @staticmethod
    def active_mask() -> np.ndarray:
        m = np.zeros(MODEL_ACTION_DIM, dtype=bool)
        m[:ENV_ACTION_DIM] = True
        return m

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        raw = _finite(raw, "raw action", last_dim=ENV_ACTION_DIM).astype(np.float64)
        out = np.zeros_like(raw)
        mm = (~self.binary_mask) & (self.min != self.max)
        out[..., mm] = 2 * (raw[..., mm] - self.min[mm]) / (self.max[mm] - self.min[mm]) - 1
        out[..., self.binary_mask] = (raw[..., self.binary_mask] > 0.5).astype(np.float64)
        return out  # degenerate 차원은 0 (upstream 동일)

    def unapply(self, model_action: np.ndarray) -> np.ndarray:
        """(..., H, 32) 정규화 출력 → (..., H, 12) raw. padding 20차원은 읽지 않는다.

        upstream inverse 와 동일하게 clip 하지 않는다 (합성 velocity 가 [-1,1] 밖으로 밀면
        학습 범위 밖 raw 가 나온다 → out_of_range_fraction 으로 진단).
        """
        a = np.asarray(model_action)
        if a.ndim < 2 or a.shape[-1] != MODEL_ACTION_DIM or a.shape[-2] != ACTION_HORIZON:
            raise ContractError(
                f"model action shape {a.shape} != (..., {ACTION_HORIZON}, {MODEL_ACTION_DIM})")
        active = _finite(a[..., :ENV_ACTION_DIM], "model action (active12)").astype(np.float64)
        out = (active + 1) / 2 * (self.max - self.min) + self.min
        out[..., self.binary_mask] = (active[..., self.binary_mask] > 0.5).astype(np.float64)
        return out

    def out_of_range_fraction(self, model_action: np.ndarray) -> float:
        a = np.asarray(model_action)[..., :ENV_ACTION_DIM]
        mm = (~self.binary_mask) & (~self.degenerate_mask)
        return float(np.mean(np.abs(a[..., mm]) > 1.0))

    def split_env_action(self, raw12: np.ndarray) -> dict[str, np.ndarray]:
        """raw 12D → robocasa GrootRoboCasaEnv action.* dict (binary 는 {0,1} 그대로)."""
        raw12 = _finite(raw12, "raw action", last_dim=ENV_ACTION_DIM)
        out, off = {}, 0
        for key, dims, _ in ACTION_KEYS:
            out[f"action.{key}"] = raw12[..., off:off + dims]
            off += dims
        return out


def raw_to_model_index(slices: Mapping[str, Any] | None = None) -> np.ndarray:
    """raw LeRobot12 → 모델 순서(eef, rot, gripper, base, mode) gather index.

    slices: meta/modality.json 의 "action" 블록 ({key: {start, end}}) 또는 {key: (start, end)}.
    None 이면 OpenDrawer/20250819 실측값(RAW_LEROBOT_ACTION_SLICES). key 집합·폭·겹침을 검사한다.
    """
    src = RAW_LEROBOT_ACTION_SLICES if slices is None else slices
    norm: dict[str, tuple[int, int]] = {}
    for key, v in src.items():
        lo, hi = (v["start"], v["end"]) if isinstance(v, Mapping) else tuple(v)
        norm[key] = (int(lo), int(hi))
    if set(norm) != {k for k, _, _ in ACTION_KEYS}:
        raise ContractError(f"raw action slice key 불일치: {sorted(norm)}")
    idx: list[int] = []
    for key, dims, _ in ACTION_KEYS:
        lo, hi = norm[key]
        if hi - lo != dims:
            raise ContractError(f"raw slice {key}={lo}:{hi} 폭 != {dims}")
        idx += list(range(lo, hi))
    if sorted(idx) != list(range(ENV_ACTION_DIM)):
        raise ContractError(f"raw action slice 가 0:12 를 정확히 덮지 않음: {norm}")
    return np.asarray(idx, dtype=np.int64)


def reorder_raw_to_model(raw12: np.ndarray, index: np.ndarray) -> np.ndarray:
    """raw LeRobot 순서 (..., 12) → 모델 순서 (..., 12). 값은 바꾸지 않는다 (순열만)."""
    a = _finite(raw12, "raw LeRobot action", last_dim=ENV_ACTION_DIM)
    return a[..., index]


def reorder_model_to_raw(model12: np.ndarray, index: np.ndarray) -> np.ndarray:
    a = _finite(model12, "model-order action", last_dim=ENV_ACTION_DIM)
    out = np.empty_like(a)
    out[..., index] = a
    return out


def build_action_codec(metadata: Mapping[str, Any]) -> ActionCodec:
    try:
        stats = metadata[EMBODIMENT_TAG]["statistics"]["action"]
        modal = metadata[EMBODIMENT_TAG]["modalities"]["action"]
    except KeyError as exc:
        raise ContractError(f"metadata 에 {EMBODIMENT_TAG} action 통계 없음") from exc
    if set(stats) != {k for k, _, _ in ACTION_KEYS}:
        raise ContractError(f"metadata action key 불일치: {sorted(stats)}")
    mins, maxs, modes, names = [], [], [], []
    for key, dims, mode in ACTION_KEYS:
        if list(modal[key]["shape"]) != [dims]:
            raise ContractError(f"{key} shape {modal[key]['shape']} != [{dims}]")
        lo = np.asarray(stats[key]["min"], dtype=np.float64)
        hi = np.asarray(stats[key]["max"], dtype=np.float64)
        if lo.shape != (dims,) or hi.shape != (dims,) or np.any(hi < lo):
            raise ContractError(f"{key} min/max 통계 이상")
        mins.append(lo)
        maxs.append(hi)
        modes += [mode] * dims
        names += [f"{key}[{i}]" for i in range(dims)]
    return ActionCodec(np.concatenate(mins), np.concatenate(maxs), tuple(modes), tuple(names))


@dataclass(frozen=True)
class CheckpointContract:
    checkpoint_dir: Path
    config_sha256: str
    metadata_sha256: str
    codec: ActionCodec


def load_checkpoint_contract(checkpoint_dir: str | Path, *, require_pinned: bool = True
                             ) -> CheckpointContract:
    d = Path(checkpoint_dir)
    cfg_path, meta_path = d / "config.json", d / "experiment_cfg" / "metadata.json"
    if not cfg_path.is_file() or not meta_path.is_file():
        raise ContractError(f"config.json/metadata.json 없음: {d}")
    cfg_sha, meta_sha = sha256_file(cfg_path), sha256_file(meta_path)
    if require_pinned and (cfg_sha, meta_sha) != (CONFIG_SHA256, METADATA_SHA256):
        raise ContractError(f"checkpoint-120000 고정 sha 불일치: config={cfg_sha[:12]} "
                            f"metadata={meta_sha[:12]}")
    validate_model_config(json.loads(cfg_path.read_text()))
    codec = build_action_codec(json.loads(meta_path.read_text()))
    return CheckpointContract(d, cfg_sha, meta_sha, codec)


FEATURE_INTERFACE = "action_decoder_input_last16_mean_denoise0"


def validate_qam_flags(flags: Mapping[str, Any], *, feature_signature: str,
                       config_sha256: str = CONFIG_SHA256,
                       metadata_sha256: str = METADATA_SHA256) -> None:
    """GR00T 계약으로 학습된 QAM 만 허용.

    pi0 Bridge QAM 도 ob 1024 이므로 차원만으로는 구분되지 않는다 → bridge 표식, 7D/H4/flow10,
    그리고 GR00T provenance 블록(feature_interface 포함) 부재로 거부한다.
    """
    agent = flags.get("agent") or {}
    bridge_markers = [
        str(flags.get("env_name", "")).lower() == "bridge",
        bool(flags.get("bridge_dataset_dir")),
        "bridge" in str(flags.get("run_group", "")).lower(),
    ]
    if any(bridge_markers):
        raise ContractError("pi0 Bridge QAM 가중치 재사용 금지 (env_name/bridge_dataset_dir/run_group)")
    ob = agent.get("ob_dims")
    if ob == [DIT_WIDTH] or ob == [VLM_CONTEXT_DIM]:
        raise ContractError(
            f"ob_dims {ob}: DiT block residual(1536)/VLM context(2048) 는 다른 인터페이스. "
            f"이 계약의 QAM 관측은 action_decoder 입력 last16 평균 {QAM_OB_DIM} 뿐이다")
    want = {
        "ob_dims": [QAM_OB_DIM],
        "action_dim": QAM_ACTION_DIM,
        "horizon_length": QAM_HORIZON,
        "action_chunking": True,
        "flow_steps": QAM_FLOW_STEPS,
        "best_of_n": 1,
    }
    bad = {k: agent.get(k) for k, v in want.items() if agent.get(k) != v}
    if bad:
        raise ContractError(f"QAM agent 계약 불일치 (want {want}): {bad}")
    block = flags.get(CONTRACT_KEY)
    if not isinstance(block, Mapping):
        raise ContractError(f"flags.{CONTRACT_KEY} 없음 — GR00T 계약 provenance 없는 QAM 거부")
    prov = {
        "config_sha256": config_sha256,
        "metadata_sha256": metadata_sha256,
        "feature_signature": feature_signature,
        "feature_interface": FEATURE_INTERFACE,
        "flow_time_convention": "t=i/flow_steps, noise(t=0)->action(t=1), v=action-noise",
        "action_space": "groot_normalized_active12",
    }
    bad = {k: block.get(k) for k, v in prov.items() if block.get(k) != v}
    if bad:
        raise ContractError(f"QAM provenance 불일치: {bad}")


def _finite(x: Any, what: str, last_dim: int | None = None) -> np.ndarray:
    a = np.asarray(x)
    if last_dim is not None and (a.ndim == 0 or a.shape[-1] != last_dim):
        raise ContractError(f"{what} last dim {a.shape} != {last_dim}")
    if not np.all(np.isfinite(a)):
        raise ContractError(f"{what} 에 비유한 값")
    return a
