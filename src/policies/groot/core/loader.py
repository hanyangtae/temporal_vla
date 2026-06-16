"""GR00T 모델 로더.

`scripts/serve/groot.py` 와 학습 코드 양쪽이 동일한 절차로 GR00T 체크포인트를
로드할 수 있도록 분리. 프로파일 (`configs/checkpoints/*.yaml`) 을 single source
of truth 로 사용.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _prepend_existing_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


# Isaac-GR00T submodule 경로 보장 (gr00t.* import 가능하도록).
# docker 안의 /temporal_vla와 host checkout 경로를 모두 지원한다.
for _path in (
    Path("/temporal_vla/src/policies/Isaac-GR00T"),
    _REPO_ROOT / "src" / "policies" / "Isaac-GR00T",
):
    _prepend_existing_path(_path)

if TYPE_CHECKING:
    from checkpoint_profile import CheckpointProfile

logger = logging.getLogger(__name__)


@dataclass
class LoadedGrootModel:
    """`load_groot_policy` 반환값.

    policy:           Gr00tSimPolicyWrapper. `.get_action(obs)` 호출 가능.
    modality_configs: dict[modality_name -> ModalityConfig]. video/state/action 의
                      modality_keys + delta_indices 정보.
    embodiment_tag:   gr00t.data.embodiment_tags.EmbodimentTag enum 값.
    state_dims:       state key (e.g. "gripper_qpos") -> dim 매핑. fallback 시 사용.
    device:           "cuda" / "cpu" / 명시 device.
    """

    policy: Any
    modality_configs: dict
    embodiment_tag: Any
    state_dims: dict[str, int]
    device: str


def resolve_model_path(model_path: str) -> str:
    """container profile path를 현재 checkout path로 보정한다.

    checkpoint profile은 docker 기준 `/temporal_vla/...` 절대 경로를 쓰는 경우가 많다.
    host에서 같은 repo checkout을 직접 실행하면 그 경로가 없으므로, 현재 checkout 아래
    대응 경로가 존재할 때만 치환한다. 이미 존재하는 경로나 HF repo id는 그대로 둔다.
    """
    path = Path(model_path)
    if path.exists() or not path.is_absolute():
        return model_path

    try:
        rel = path.relative_to("/temporal_vla")
    except ValueError:
        return model_path

    local_path = _REPO_ROOT / rel
    if local_path.exists():
        return str(local_path)
    return model_path


def resolve_device(model_specific: dict[str, Any], device: Optional[str] = None) -> str:
    """명시 device 인자를 profile default보다 우선한다."""
    if device is not None:
        return device
    return str(model_specific.get("device", "cuda"))


def load_state_dims_from_statistics(model_path: str, embodiment_value: str) -> dict[str, int]:
    """`{model_path}/statistics.json` 에서 embodiment 의 state key dim map 추출.

    statistics.json 구조:
        {<embodiment>: {"state": {<key>: {"mean": [...], "std": [...]}, ...}, "action": {...}}}

    각 state key 의 mean 길이가 dim. fallback zero state 채울 때 사용.
    파일이 없거나 파싱 실패하면 빈 dict 반환 (warning).
    """
    model_path = resolve_model_path(model_path)
    stats_path = os.path.join(model_path, "statistics.json")
    if not os.path.exists(stats_path):
        logger.warning("statistics.json not found under %s", model_path)
        return {}
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        emb = stats.get(embodiment_value, {})
        state = emb.get("state", {})
        dims: dict[str, int] = {}
        for k, v in state.items():
            if isinstance(v, dict) and "mean" in v and hasattr(v["mean"], "__len__"):
                dims[k] = len(v["mean"])
        return dims
    except Exception as e:  # pragma: no cover
        logger.warning("failed to parse statistics.json: %s", e)
        return {}


def load_groot_policy(
    profile: "CheckpointProfile",
    device: Optional[str] = None,
) -> LoadedGrootModel:
    """프로파일 → Gr00tSimPolicyWrapper 로드.

    profile.model_specific:
      - embodiment_tag: EmbodimentTag enum name (예: "ROBOCASA_PANDA_OMRON")
      - device:         "cuda" / "cpu" (생략 시 인자 device 또는 "cuda")
      - no_strict:      True 면 Gr00tPolicy(strict=False) — 디버깅용

    profile.checkpoint_source.id 가 모델 디렉토리 경로 (HF cache 또는 local 절대 경로).
    """
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    ms = profile.model_specific or {}
    embodiment_tag = EmbodimentTag[ms["embodiment_tag"]]
    resolved_device = resolve_device(ms, device)
    model_path = resolve_model_path(profile.checkpoint_source.id)
    strict = not bool(ms.get("no_strict", False))

    logger.info(
        "Loading GR00T from %s (profile=%s, embodiment=%s, device=%s, strict=%s)",
        model_path, profile.name, embodiment_tag.value, resolved_device, strict,
    )

    base_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=resolved_device,
        strict=strict,
    )
    policy = Gr00tSimPolicyWrapper(base_policy, strict=strict)
    modality_configs = policy.get_modality_config()

    for modality, cfg in modality_configs.items():
        logger.info("[%s] modality_keys=%s", modality, cfg.modality_keys)

    state_dims = load_state_dims_from_statistics(model_path, embodiment_tag.value)
    logger.info("state dims loaded from statistics.json: %s", state_dims)

    horizon = len(modality_configs["action"].delta_indices)
    logger.info(
        "GR00T loaded. action_horizon=%d (profile.n_action_steps=%d)",
        horizon, profile.n_action_steps,
    )

    return LoadedGrootModel(
        policy=policy,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        state_dims=state_dims,
        device=resolved_device,
    )
