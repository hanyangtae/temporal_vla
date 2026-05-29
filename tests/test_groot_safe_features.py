"""Unit tests for the shared GR00T SAFE feature module."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.policies.groot.safe_features import (
    SAFE_FEATURE_AXES_ALL,
    SAFE_FEATURE_AXES_VALID,
    SAFE_FEATURE_KIND_ALL,
    SAFE_FEATURE_KIND_VALID,
    capture_dit_features,
    cast_feature_tensor,
    decode_features_from_base64,
    encode_features_to_base64,
    feature_metadata,
    normalize_feature_metadata,
    resolve_feature_action_horizon,
    SafeFeatureExtractor,
)


class _ModalityConfig:
    def __init__(self, delta_indices: list[int]):
        self.delta_indices = delta_indices


class _HookHandle:
    def __init__(self, target):
        self.target = target

    def remove(self):
        self.target.hook = None


class _HookTarget:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _HookHandle(self)

    def emit(self):
        if self.hook is not None:
            self.hook(self, (), self.output)


class _ActionHead:
    def __init__(self, output: torch.Tensor):
        self.action_horizon = int(output.shape[1])
        self.model = _HookTarget(output)


class _InnerPolicy:
    def __init__(self, output: torch.Tensor):
        self.model = type("_Model", (), {"action_head": _ActionHead(output)})()


class _SimPolicy:
    def __init__(self, output: torch.Tensor, valid_horizon: int, n_emits: int = 1):
        self.policy = _InnerPolicy(output)
        self._valid_horizon = valid_horizon
        self._n_emits = n_emits
        self.last_obs = None
        self.last_options = None

    def get_modality_config(self):
        return {"action": _ModalityConfig(list(range(self._valid_horizon)))}

    def get_action(self, observation, options=None):
        self.last_obs = observation
        self.last_options = options
        for _ in range(self._n_emits):
            self.policy.model.action_head.model.emit()
        return ({"action.eef_pos": np.array([[[1.0, 2.0, 3.0]]])}, {})


class TestFeatureMetadata(unittest.TestCase):
    def test_valid_metadata(self):
        kind, axes = feature_metadata("valid")
        self.assertEqual(kind, SAFE_FEATURE_KIND_VALID)
        self.assertEqual(axes, SAFE_FEATURE_AXES_VALID)

    def test_all_metadata(self):
        kind, axes = feature_metadata("all")
        self.assertEqual(kind, SAFE_FEATURE_KIND_ALL)
        self.assertEqual(axes, SAFE_FEATURE_AXES_ALL)

    def test_unknown_slice_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported feature slice"):
            feature_metadata("bogus")


class TestNormalizeFeatureMetadata(unittest.TestCase):
    def test_normalizes_zmq_canonical_metadata(self):
        metadata = normalize_feature_metadata(
            {
                "feature_kind": SAFE_FEATURE_KIND_VALID,
                "feature_axes": SAFE_FEATURE_AXES_VALID,
                "feature_slice": "valid",
                "exported_action_token_count": 8,
                "feature_action_horizon": 8,
                "valid_action_horizon": 16,
                "model_action_horizon": 50,
                "num_inference_timesteps": 4,
            }
        )

        self.assertEqual(metadata.feature_kind, SAFE_FEATURE_KIND_VALID)
        self.assertEqual(metadata.feature_axes, SAFE_FEATURE_AXES_VALID)
        self.assertEqual(metadata.feature_slice, "valid")
        self.assertEqual(metadata.exported_action_token_count, 8)
        self.assertEqual(metadata.feature_action_horizon, 8)
        self.assertEqual(metadata.valid_action_horizon, 16)
        self.assertEqual(metadata.model_action_horizon, 50)
        self.assertEqual(metadata.num_inference_timesteps, 4)

    def test_normalizes_http_feature_metadata_aliases(self):
        metadata = normalize_feature_metadata(
            {
                "kind": SAFE_FEATURE_KIND_VALID,
                "axes": tuple(SAFE_FEATURE_AXES_VALID),
                "slice": "valid",
                "feature_action_horizon": np.int64(8),
                "valid_action_horizon": 16,
                "model_action_horizon": 50,
                "num_inference_timesteps": 4,
            }
        )

        self.assertEqual(metadata.feature_kind, SAFE_FEATURE_KIND_VALID)
        self.assertEqual(metadata.feature_axes, SAFE_FEATURE_AXES_VALID)
        self.assertEqual(metadata.feature_slice, "valid")
        self.assertEqual(metadata.exported_action_token_count, 8)

    def test_normalizes_vla_client_flattened_feature_metadata(self):
        metadata = normalize_feature_metadata(
            {
                "features.kind": SAFE_FEATURE_KIND_ALL,
                "features.axes": SAFE_FEATURE_AXES_ALL,
                "features.slice": "all",
                "features.exported_action_token_count": 50,
                "features.feature_action_horizon": 50,
                "features.valid_action_horizon": 16,
                "features.model_action_horizon": 50,
                "features.num_inference_timesteps": 6,
            }
        )

        self.assertEqual(metadata.feature_kind, SAFE_FEATURE_KIND_ALL)
        self.assertEqual(metadata.feature_axes, SAFE_FEATURE_AXES_ALL)
        self.assertEqual(metadata.feature_slice, "all")
        self.assertEqual(metadata.exported_action_token_count, 50)
        self.assertEqual(metadata.num_inference_timesteps, 6)


class TestResolveFeatureActionHorizon(unittest.TestCase):
    def test_valid_defaults_to_valid_horizon(self):
        fah = resolve_feature_action_horizon(
            feature_slice="valid",
            model_action_horizon=50,
            valid_action_horizon=16,
            feature_action_horizon=None,
        )
        self.assertEqual(fah, 16)

    def test_all_defaults_to_model_horizon(self):
        fah = resolve_feature_action_horizon(
            feature_slice="all",
            model_action_horizon=50,
            valid_action_horizon=16,
            feature_action_horizon=None,
        )
        self.assertEqual(fah, 50)

    def test_explicit_horizon_within_bound(self):
        fah = resolve_feature_action_horizon(
            feature_slice="valid",
            model_action_horizon=50,
            valid_action_horizon=16,
            feature_action_horizon=8,
        )
        self.assertEqual(fah, 8)

    def test_exceeds_bound_raises(self):
        with self.assertRaisesRegex(ValueError, "exceeds exportable horizon"):
            resolve_feature_action_horizon(
                feature_slice="valid",
                model_action_horizon=50,
                valid_action_horizon=16,
                feature_action_horizon=17,
            )

    def test_negative_horizon_raises(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_feature_action_horizon(
                feature_slice="valid",
                model_action_horizon=50,
                valid_action_horizon=16,
                feature_action_horizon=-1,
            )

    def test_unknown_slice_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported feature slice"):
            resolve_feature_action_horizon(
                feature_slice="bogus",
                model_action_horizon=50,
                valid_action_horizon=16,
                feature_action_horizon=None,
            )


class TestCaptureDitFeatures(unittest.TestCase):
    def test_valid_slice_captures_leading_valid_tokens_across_denoising(self):
        # model_action_horizon=4, valid_action_horizon=2.
        output = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        sim = _SimPolicy(output, valid_horizon=2, n_emits=3)

        result = capture_dit_features(sim, {"obs": 1}, feature_slice="valid")

        self.assertEqual(result["model_action_horizon"], 4)
        self.assertEqual(result["valid_action_horizon"], 2)
        self.assertEqual(result["feature_action_horizon"], 2)
        self.assertEqual(result["num_inference_timesteps"], 3)
        # [B=1, K=3, H=2, D=3] — leading 2 tokens of the trailing model horizon.
        self.assertEqual(tuple(result["features"].shape), (1, 3, 2, 3))
        np.testing.assert_allclose(
            result["features"][0, 0].numpy(),
            output[0, -4:][:2].numpy(),
        )

    def test_all_slice_captures_full_model_horizon(self):
        output = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        sim = _SimPolicy(output, valid_horizon=2, n_emits=1)

        result = capture_dit_features(sim, {"obs": 1}, feature_slice="all")

        self.assertEqual(result["feature_action_horizon"], 4)
        self.assertEqual(tuple(result["features"].shape), (1, 1, 4, 3))

    def test_hook_removed_after_call(self):
        output = torch.zeros((1, 4, 3), dtype=torch.float32)
        sim = _SimPolicy(output, valid_horizon=2, n_emits=1)

        capture_dit_features(sim, {"obs": 1})
        # Hook removed even on success.
        self.assertIsNone(sim.policy.model.action_head.model.hook)

    def test_hook_removed_on_resolution_error(self):
        output = torch.zeros((1, 4, 3), dtype=torch.float32)
        sim = _SimPolicy(output, valid_horizon=2, n_emits=1)

        with self.assertRaises(ValueError):
            capture_dit_features(
                sim,
                {"obs": 1},
                feature_slice="valid",
                feature_action_horizon=99,
            )
        # ValueError fires before hook registration so the target should be
        # unchanged. The point is no leaked hook.
        self.assertIsNone(sim.policy.model.action_head.model.hook)

    def test_no_emit_raises(self):
        output = torch.zeros((1, 4, 3), dtype=torch.float32)
        sim = _SimPolicy(output, valid_horizon=2, n_emits=0)

        with self.assertRaisesRegex(RuntimeError, "Failed to capture"):
            capture_dit_features(sim, {"obs": 1})


class TestSafeFeatureExtractor(unittest.TestCase):
    def test_capture_returns_cast_hidden_states_and_normalized_metadata(self):
        output = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        sim = _SimPolicy(output, valid_horizon=2, n_emits=1)
        extractor = SafeFeatureExtractor(
            sim,
            feature_dtype="float32",
            feature_slice="valid",
            feature_action_horizon=1,
        )

        result = extractor.capture({"obs": 1})

        self.assertIn("action.eef_pos", result.action)
        self.assertEqual(tuple(result.hidden_states.shape), (1, 1, 1, 3))
        self.assertEqual(result.hidden_states.dtype, torch.float32)
        self.assertEqual(result.metadata.feature_kind, SAFE_FEATURE_KIND_VALID)
        self.assertEqual(result.metadata.feature_axes, SAFE_FEATURE_AXES_VALID)
        self.assertEqual(result.metadata.feature_slice, "valid")
        self.assertEqual(result.metadata.exported_action_token_count, 1)
        self.assertEqual(result.metadata.feature_action_horizon, 1)
        self.assertEqual(result.metadata.valid_action_horizon, 2)
        self.assertEqual(result.metadata.model_action_horizon, 4)
        self.assertEqual(result.metadata.num_inference_timesteps, 1)


class TestFeatureEncoding(unittest.TestCase):
    def test_float16_round_trip(self):
        arr = np.linspace(-1.0, 1.0, 12, dtype=np.float16).reshape(1, 2, 2, 3)
        tensor = torch.from_numpy(arr.astype(np.float32))

        blob = encode_features_to_base64(tensor, "float16")
        decoded = decode_features_from_base64(blob)

        self.assertEqual(blob["dtype"], "float16")
        self.assertEqual(list(blob["shape"]), [1, 2, 2, 3])
        np.testing.assert_array_equal(decoded, arr)

    def test_float32_round_trip(self):
        arr = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(1, 1, 2, 3)
        tensor = torch.from_numpy(arr)

        blob = encode_features_to_base64(tensor, "float32")
        decoded = decode_features_from_base64(blob)

        self.assertEqual(blob["dtype"], "float32")
        np.testing.assert_array_equal(decoded, arr)

    def test_cast_unsupported_dtype_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported feature dtype"):
            cast_feature_tensor(torch.zeros(1), "int8")

    def test_decoded_array_is_writable(self):
        arr = np.zeros((2,), dtype=np.float16)
        blob = encode_features_to_base64(torch.from_numpy(arr.astype(np.float32)), "float16")
        decoded = decode_features_from_base64(blob)

        # frombuffer returns a read-only view; decode should return a copy.
        decoded[0] = 1.0
        self.assertEqual(decoded[0], 1.0)


if __name__ == "__main__":
    unittest.main()
