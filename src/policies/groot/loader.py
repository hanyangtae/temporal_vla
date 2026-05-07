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
from typing import TYPE_CHECKING, Any, Optional

# Isaac-GR00T submodule 경로 보장 (gr00t.* import 가능하도록).
_GR00T_PATH = "/temporal_vla/src/policies/Isaac-GR00T"
if _GR00T_PATH not in sys.path:
    sys.path.insert(0, _GR00T_PATH)

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


def load_state_dims_from_statistics(model_path: str, embodiment_value: str) -> dict[str, int]:
    """`{model_path}/statistics.json` 에서 embodiment 의 state key dim map 추출.

    statistics.json 구조:
        {<embodiment>: {"state": {<key>: {"mean": [...], "std": [...]}, ...}, "action": {...}}}

    각 state key 의 mean 길이가 dim. fallback zero state 채울 때 사용.
    파일이 없거나 파싱 실패하면 빈 dict 반환 (warning).
    """
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
    resolved_device = ms.get("device", device or "cuda")
    model_path = profile.checkpoint_source.id
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
