"""Shared SAFE feature metadata naming helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FLOW_FEATURE_AXES = ["denoising_step", "action_step", "feature_dim"]
FAST_FEATURE_AXES = ["token_singleton", "action_token", "feature_dim"]
GROOT_N16_VALID_FEATURE_AXES = [
    "denoising_step",
    "valid_action_step",
    "feature_dim",
]
GROOT_N16_ALL_FEATURE_AXES = [
    "denoising_step",
    "model_action_token",
    "feature_dim",
]
GROOT_N15_DIT_BLOCK_RESIDUAL_FEATURE_AXES = [
    "layer",
    "model_token",
    "feature_dim",
]
# COAST-faithful (A.7.2): action-token mean-pooled, one vector per denoising step,
# denoise axis preserved. Shape per env-step = [layer, denoise_step, feature_dim].
GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES = [
    "layer",
    "denoise_step",
    "feature_dim",
]
# pq3 full-token capture: 전 시퀀스 토큰(state+future+action, T=token_count) 보존,
# mean 은 fit 시점에 수행. Shape per env-step = [layer, denoise_step, model_token,
# feature_dim] (fp16 저장, fit 수집 전용). axes[0]="layer" 라 collect_artifacts 의
# action-token horizon 불변식에서 면제된다.
GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_AXES = [
    "layer",
    "denoise_step",
    "model_token",
    "feature_dim",
]

GROOT_N16_VALID_FEATURE_KIND = "groot_n16_dit_valid_action_tokens_pre_velocity"
GROOT_N16_ALL_FEATURE_KIND = "groot_n16_dit_all_action_tokens_pre_velocity"
GROOT_N15_DIT_BLOCK_RESIDUAL_FEATURE_KIND = "groot_n15_dit_block_residual_tokens"
GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND = (
    "groot_n15_dit_block_residual_action_tokens_denoise"
)
GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_KIND = (
    "groot_n15_dit_block_residual_full_tokens_denoise"
)
GROOT_VL_FEATURE_AXES = ["feature_dim"]
GROOT_N15_VL_FEATURE_KIND = "groot_n15_vlln_seq_meanpool"
GROOT_N16_VL_FEATURE_KIND = "groot_n16_vlln_seq_meanpool"
# pq3: post-VL-SA(=DiT cross-attn 입력, vlln→vl_self_attention 출력) full-token 캡처.
# Shape per env-step = [vl_token, feature_dim] (T_vl 은 런타임 기록).
GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_AXES = ["vl_token", "feature_dim"]
GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_KIND = "groot_n15_vl_post_sa_full_tokens"

# COAST A.7.1 pi05: π0.5 action expert(Gemma2, 18 layer, d=1024)의 decoder layer
# residual stream에서 마지막 chunk_size action token을 mean-pool, denoise step K 보존.
# GR00T DiT block residual 과 동일한 [layer, denoise_step, feature_dim] 축.
PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES = [
    "layer",
    "denoise_step",
    "feature_dim",
]
PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND = (
    "pi05_expert_block_residual_action_tokens_denoise"
)

_LEROBOT_FEATURE_KINDS = {
    "pi0": "pi0_action_expert_pre_velocity",
    "pi05": "pi05_action_expert_pre_velocity",
    "xvla": "xvla_transformer_pre_action_decoder",
    "groot": "groot_n15_dit_action_tokens_pre_decode",
    "pi0_fast": "pi0_fast_prelogit_action_tokens",
}


@dataclass(frozen=True)
class SafeFeatureMetadata:
    feature_kind: str | None
    feature_axes: list[str] | None
    feature_slice: str | None
    exported_action_token_count: int | None
    feature_action_horizon: int | None
    valid_action_horizon: int | None
    model_action_horizon: int | None
    num_inference_timesteps: int | None

    def asdict(self) -> dict[str, Any]:
        return {
            "feature_kind": self.feature_kind,
            "feature_axes": self.feature_axes,
            "feature_slice": self.feature_slice,
            "exported_action_token_count": self.exported_action_token_count,
            "feature_action_horizon": self.feature_action_horizon,
            "valid_action_horizon": self.valid_action_horizon,
            "model_action_horizon": self.model_action_horizon,
            "num_inference_timesteps": self.num_inference_timesteps,
        }


def lerobot_feature_kind(policy_type: str) -> str:
    try:
        return _LEROBOT_FEATURE_KINDS[policy_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported policy_type: {policy_type}") from exc


def lerobot_feature_axes(policy_type: str) -> list[str]:
    if policy_type == "pi0_fast":
        return list(FAST_FEATURE_AXES)
    if policy_type in _LEROBOT_FEATURE_KINDS:
        return list(FLOW_FEATURE_AXES)
    raise ValueError(f"Unsupported policy_type: {policy_type}")


def groot_n16_feature_metadata(feature_slice: str) -> tuple[str, list[str]]:
    if feature_slice == "valid":
        return GROOT_N16_VALID_FEATURE_KIND, list(GROOT_N16_VALID_FEATURE_AXES)
    if feature_slice == "all":
        return GROOT_N16_ALL_FEATURE_KIND, list(GROOT_N16_ALL_FEATURE_AXES)
    raise ValueError(f"Unsupported feature slice: {feature_slice}")


def _first_non_none(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def normalize_feature_metadata(payload: dict[str, Any]) -> SafeFeatureMetadata:
    """Normalize ZMQ, HTTP-response, or VLAClient feature metadata keys.

    Canonical fields match the SAFE rollout pkl schema. Accepted aliases include
    ``feature_kind`` / ``kind`` / ``features.kind`` and equivalent patterns for
    axes, slice, and horizon metadata. Legacy LeRobot responses use
    ``action_horizon`` or ``n_action_tokens`` for the exported feature horizon.
    """
    feature_action_horizon = _optional_int(
        _first_non_none(
            payload,
            "feature_action_horizon",
            "features.feature_action_horizon",
            "action_horizon",
            "features.action_horizon",
            "n_action_tokens",
            "features.n_action_tokens",
        )
    )
    exported_action_token_count = _optional_int(
        _first_non_none(
            payload,
            "exported_action_token_count",
            "features.exported_action_token_count",
        )
    )
    if exported_action_token_count is None:
        exported_action_token_count = feature_action_horizon

    model_action_horizon = _optional_int(
        _first_non_none(
            payload,
            "model_action_horizon",
            "features.model_action_horizon",
        )
    )
    if model_action_horizon is None:
        model_action_horizon = feature_action_horizon

    axes = _first_non_none(payload, "feature_axes", "axes", "features.axes")
    if axes is not None:
        axes = list(axes)

    return SafeFeatureMetadata(
        feature_kind=_first_non_none(payload, "feature_kind", "kind", "features.kind"),
        feature_axes=axes,
        feature_slice=_first_non_none(payload, "feature_slice", "slice", "features.slice"),
        exported_action_token_count=exported_action_token_count,
        feature_action_horizon=feature_action_horizon,
        valid_action_horizon=_optional_int(
            _first_non_none(payload, "valid_action_horizon", "features.valid_action_horizon")
        ),
        model_action_horizon=model_action_horizon,
        num_inference_timesteps=_optional_int(
            _first_non_none(
                payload,
                "num_inference_timesteps",
                "features.num_inference_timesteps",
            )
        ),
    )
