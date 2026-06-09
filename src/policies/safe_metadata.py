"""Shared SAFE feature metadata naming helpers."""

from __future__ import annotations

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

GROOT_N16_VALID_FEATURE_KIND = "groot_n16_dit_valid_action_tokens_pre_velocity"
GROOT_N16_ALL_FEATURE_KIND = "groot_n16_dit_all_action_tokens_pre_velocity"

_LEROBOT_FEATURE_KINDS = {
    "pi0": "pi0_action_expert_pre_velocity",
    "pi05": "pi05_action_expert_pre_velocity",
    "xvla": "xvla_transformer_pre_action_decoder",
    "groot": "groot_n15_dit_action_tokens_pre_decode",
    "pi0_fast": "pi0_fast_prelogit_action_tokens",
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
