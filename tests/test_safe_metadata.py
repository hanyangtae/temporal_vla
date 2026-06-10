from __future__ import annotations

import numpy as np

from src.policies.safe_metadata import (
    FAST_FEATURE_AXES,
    FLOW_FEATURE_AXES,
    GROOT_N16_ALL_FEATURE_AXES,
    GROOT_N16_ALL_FEATURE_KIND,
    GROOT_N16_VALID_FEATURE_AXES,
    GROOT_N16_VALID_FEATURE_KIND,
    lerobot_feature_axes,
    lerobot_feature_kind,
    normalize_feature_metadata,
)


def test_normalize_feature_metadata_accepts_canonical_zmq_keys():
    metadata = normalize_feature_metadata(
        {
            "feature_kind": GROOT_N16_VALID_FEATURE_KIND,
            "feature_axes": GROOT_N16_VALID_FEATURE_AXES,
            "feature_slice": "valid",
            "exported_action_token_count": 8,
            "feature_action_horizon": 8,
            "valid_action_horizon": 16,
            "model_action_horizon": 50,
            "num_inference_timesteps": 4,
        }
    )

    assert metadata.feature_kind == GROOT_N16_VALID_FEATURE_KIND
    assert metadata.feature_axes == GROOT_N16_VALID_FEATURE_AXES
    assert metadata.feature_slice == "valid"
    assert metadata.exported_action_token_count == 8
    assert metadata.feature_action_horizon == 8
    assert metadata.valid_action_horizon == 16
    assert metadata.model_action_horizon == 50
    assert metadata.num_inference_timesteps == 4


def test_normalize_feature_metadata_accepts_vla_client_feature_keys():
    metadata = normalize_feature_metadata(
        {
            "kind": GROOT_N16_VALID_FEATURE_KIND,
            "axes": tuple(GROOT_N16_VALID_FEATURE_AXES),
            "slice": "valid",
            "feature_action_horizon": np.int64(8),
            "valid_action_horizon": 16,
            "model_action_horizon": 50,
            "num_inference_timesteps": 4,
        }
    )

    assert metadata.feature_kind == GROOT_N16_VALID_FEATURE_KIND
    assert metadata.feature_axes == GROOT_N16_VALID_FEATURE_AXES
    assert metadata.feature_slice == "valid"
    assert metadata.exported_action_token_count == 8


def test_normalize_feature_metadata_accepts_flat_features_namespace_keys():
    metadata = normalize_feature_metadata(
        {
            "features.kind": GROOT_N16_ALL_FEATURE_KIND,
            "features.axes": GROOT_N16_ALL_FEATURE_AXES,
            "features.slice": "all",
            "features.exported_action_token_count": 50,
            "features.feature_action_horizon": 50,
            "features.valid_action_horizon": 16,
            "features.model_action_horizon": 50,
            "features.num_inference_timesteps": 6,
        }
    )

    assert metadata.feature_kind == GROOT_N16_ALL_FEATURE_KIND
    assert metadata.feature_axes == GROOT_N16_ALL_FEATURE_AXES
    assert metadata.feature_slice == "all"
    assert metadata.exported_action_token_count == 50
    assert metadata.num_inference_timesteps == 6


def test_normalize_feature_metadata_accepts_legacy_lerobot_horizon_alias():
    metadata = normalize_feature_metadata(
        {
            "kind": "pi05_action_expert_pre_velocity",
            "axes": ["denoising_step", "action_step", "feature_dim"],
            "action_horizon": 50,
            "num_inference_timesteps": 10,
        }
    )

    assert metadata.feature_kind == "pi05_action_expert_pre_velocity"
    assert metadata.feature_axes == ["denoising_step", "action_step", "feature_dim"]
    assert metadata.exported_action_token_count == 50
    assert metadata.feature_action_horizon == 50
    assert metadata.model_action_horizon == 50
    assert metadata.num_inference_timesteps == 10


def test_normalize_feature_metadata_accepts_autoregressive_token_alias():
    metadata = normalize_feature_metadata(
        {
            "kind": "pi0_fast_prelogit_action_tokens",
            "axes": ["token_singleton", "action_token", "feature_dim"],
            "n_action_tokens": 12,
            "num_inference_timesteps": None,
        }
    )

    assert metadata.feature_kind == "pi0_fast_prelogit_action_tokens"
    assert metadata.feature_axes == ["token_singleton", "action_token", "feature_dim"]
    assert metadata.exported_action_token_count == 12
    assert metadata.feature_action_horizon == 12
    assert metadata.model_action_horizon == 12
    assert metadata.num_inference_timesteps is None


def test_normalize_feature_metadata_skips_none_alias_values():
    metadata = normalize_feature_metadata(
        {
            "kind": "pi0_fast_prelogit_action_tokens",
            "axes": ["token_singleton", "action_token", "feature_dim"],
            "action_horizon": None,
            "n_action_tokens": 12,
            "model_action_horizon": None,
        }
    )

    assert metadata.exported_action_token_count == 12
    assert metadata.feature_action_horizon == 12
    assert metadata.model_action_horizon == 12


def test_normalize_feature_metadata_accepts_namespaced_legacy_horizon_aliases():
    metadata = normalize_feature_metadata(
        {
            "features.kind": "pi0_fast_prelogit_action_tokens",
            "features.axes": ["token_singleton", "action_token", "feature_dim"],
            "features.action_horizon": None,
            "features.n_action_tokens": 12,
            "features.model_action_horizon": None,
        }
    )

    assert metadata.feature_kind == "pi0_fast_prelogit_action_tokens"
    assert metadata.feature_axes == ["token_singleton", "action_token", "feature_dim"]
    assert metadata.exported_action_token_count == 12
    assert metadata.feature_action_horizon == 12
    assert metadata.model_action_horizon == 12


def test_lerobot_feature_metadata_names_flow_and_autoregressive_axes():
    assert lerobot_feature_kind("groot") == "groot_n15_dit_action_tokens_pre_decode"
    assert lerobot_feature_axes("groot") == FLOW_FEATURE_AXES

    assert lerobot_feature_kind("pi0_fast") == "pi0_fast_prelogit_action_tokens"
    assert lerobot_feature_axes("pi0_fast") == FAST_FEATURE_AXES
