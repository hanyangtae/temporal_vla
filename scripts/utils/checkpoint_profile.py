"""VLA 체크포인트 프로파일 로더.

`configs/checkpoints/*.yaml` 을 파싱해 `CheckpointProfile` dataclass 로 반환.
serve 스크립트는 `--profile <yaml>` 인자로 이 로더를 호출해 동작을 결정한다.

스키마 문서: `configs/checkpoints/README.md`
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_ACTION_TYPES = {"relative", "absolute"}
_ROTATION_ENCODINGS = {
    "euler",
    "quat_xyzw",
    "quat_wxyz",
    "rot6d",
    "axisangle",
    "none",
}
_NORM_SCHEMES = {"none", "min_max", "q01_q99", "mean_std"}
_GRIPPER_RANGES = {"[-1,1]", "[0,1]", "{-1,1}"}
_SOURCE_TYPES = {"hf_repo", "local"}


@dataclass
class ActionDim:
    name: str
    dims: int


@dataclass
class CheckpointSource:
    type: str
    id: str


@dataclass
class GripperEncoding:
    range: str
    binarize: bool = False
    sign_flip: bool = False
    threshold: float = 0.0


@dataclass
class Normalization:
    scheme: str
    stats_file: Optional[str] = None
    key_selection: List[str] = field(default_factory=list)


@dataclass
class ObservationRequirements:
    images: List[str]
    state: List[str]
    allow_conversions: List[str] = field(default_factory=list)


@dataclass
class ImagePreprocess:
    resolution: int = 224
    rotate_180: bool = False
    center_crop: bool = False


@dataclass
class LoraAdapter:
    """PEFT LoRA 어댑터 로드 명세. `checkpoint_source` 와 동일 repo/path 아래의
    `subfolder` 에서 `adapter_config.json` + `adapter_model.safetensors` 를 찾는다.
    """
    subfolder: str


@dataclass
class CheckpointProfile:
    name: str
    base_model: str
    checkpoint_source: CheckpointSource
    compatible_benchmarks: List[str]
    action_type: str
    action_layout: List[ActionDim]
    rotation_encoding: str
    gripper_encoding: GripperEncoding
    normalization: Normalization
    observation_requirements: ObservationRequirements
    n_action_steps: int
    image_preprocess: ImagePreprocess
    emits_subkeys: List[str]
    lora: Optional[LoraAdapter] = None
    # 모델 아키텍처 고유 설정 (domain_id, denoising_steps, embodiment 등).
    # 각 serve 스크립트가 자기가 필요한 key 만 참조하고 나머지는 무시.
    model_specific: Dict[str, Any] = field(default_factory=dict)

    @property
    def action_dim(self) -> int:
        return sum(a.dims for a in self.action_layout)

    def dim_slice(self, name: str) -> slice:
        """layout 의 `name` 구간을 모델 출력 벡터 slice 로 반환."""
        offset = 0
        for a in self.action_layout:
            if a.name == name:
                return slice(offset, offset + a.dims)
            offset += a.dims
        raise KeyError(
            f"{name!r} not in action_layout {[a.name for a in self.action_layout]}"
        )


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"[checkpoint_profile] {msg}")


def load_profile(path: str | Path) -> CheckpointProfile:
    path = Path(path)
    _require(path.is_file(), f"profile not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)

    # top-level
    _require(
        data.get("action_type") in _ACTION_TYPES,
        f"action_type must be in {_ACTION_TYPES}, got {data.get('action_type')}",
    )
    _require(
        data.get("rotation_encoding") in _ROTATION_ENCODINGS,
        f"rotation_encoding must be in {_ROTATION_ENCODINGS}, "
        f"got {data.get('rotation_encoding')}",
    )

    src = data["checkpoint_source"]
    _require(
        src.get("type") in _SOURCE_TYPES,
        f"checkpoint_source.type must be in {_SOURCE_TYPES}, got {src.get('type')}",
    )

    layout = [ActionDim(**item) for item in data["action_layout"]]
    _require(len(layout) > 0, "action_layout must have at least one entry")

    norm = data["normalization"]
    _require(
        norm.get("scheme") in _NORM_SCHEMES,
        f"normalization.scheme must be in {_NORM_SCHEMES}, got {norm.get('scheme')}",
    )

    grip = data["gripper_encoding"]
    _require(
        grip.get("range") in _GRIPPER_RANGES,
        f"gripper_encoding.range must be in {_GRIPPER_RANGES}, got {grip.get('range')}",
    )

    obs = data["observation_requirements"]
    img = data.get("image_preprocess", {})
    lora_data = data.get("lora")
    lora_obj = LoraAdapter(**lora_data) if lora_data else None

    profile = CheckpointProfile(
        name=data["name"],
        base_model=data["base_model"],
        checkpoint_source=CheckpointSource(**src),
        compatible_benchmarks=list(data["compatible_benchmarks"]),
        action_type=data["action_type"],
        action_layout=layout,
        rotation_encoding=data["rotation_encoding"],
        gripper_encoding=GripperEncoding(**grip),
        normalization=Normalization(**norm),
        observation_requirements=ObservationRequirements(**obs),
        n_action_steps=int(data["n_action_steps"]),
        image_preprocess=ImagePreprocess(**img),
        emits_subkeys=list(data["emits_subkeys"]),
        lora=lora_obj,
        model_specific=dict(data.get("model_specific", {})),
    )

    # cross-field sanity
    for sk in profile.emits_subkeys:
        _require(
            sk.startswith("action."),
            f"emits_subkey must start with 'action.': {sk!r}",
        )
    _require(
        profile.name == path.stem,
        f"profile.name ({profile.name!r}) must match filename stem ({path.stem!r})",
    )

    logger.debug("Loaded checkpoint profile %s from %s", profile.name, path)
    return profile


def main():
    try:
        from src.utils.common.logger import create_module_logger

        log = create_module_logger("checkpoint_profile")
    except ImportError:
        # colorlog / src 미가용 환경(호스트 dry-run)에서는 표준 logging 으로 fallback.
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        log = logging.getLogger("checkpoint_profile")

    if len(sys.argv) != 2:
        log.error("usage: python checkpoint_profile.py <profile.yaml>")
        sys.exit(1)
    p = load_profile(sys.argv[1])
    log.info("Loaded: %s", p.name)
    log.info("  base_model        : %s", p.base_model)
    log.info("  checkpoint_source : %s:%s", p.checkpoint_source.type, p.checkpoint_source.id)
    log.info("  action_type       : %s", p.action_type)
    log.info("  action_dim        : %d", p.action_dim)
    log.info("  action_layout     : %s", [(a.name, a.dims) for a in p.action_layout])
    log.info("  rotation_encoding : %s", p.rotation_encoding)
    log.info("  gripper           : %s", p.gripper_encoding)
    log.info("  normalization     : %s", p.normalization)
    log.info("  n_action_steps    : %d", p.n_action_steps)
    log.info("  emits_subkeys     : %s", p.emits_subkeys)
    log.info("  benchmarks        : %s", p.compatible_benchmarks)


if __name__ == "__main__":
    main()
