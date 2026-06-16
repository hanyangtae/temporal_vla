"""SAFE GR00T N1.6 feature endpoint tests."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SERVER_PATH = (
    REPO_ROOT / "scripts" / "safe" / "groot_n16" / "robocasa" / "serve" / "feature_server.py"
)


class _ModalityConfig:
    def __init__(self, delta_indices: list[int]):
        self.delta_indices = delta_indices


class _HookHandle:
    def __init__(self, target: "_HookTarget"):
        self.target = target

    def remove(self) -> None:
        self.target.hook = None


class _HookTarget:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _HookHandle(self)

    def emit(self) -> None:
        assert self.hook is not None
        self.hook(self, (), self.output)


class _ActionHead:
    def __init__(self, output: torch.Tensor):
        self.action_horizon = int(output.shape[1])
        self.model = _HookTarget(output)


class _RootModel:
    def __init__(self, output: torch.Tensor):
        self.action_head = _ActionHead(output)


class _InnerPolicy:
    def __init__(self, output: torch.Tensor):
        self.model = _RootModel(output)


class _FakeSimPolicy:
    def __init__(self, output: torch.Tensor, valid_horizon: int):
        self.policy = _InnerPolicy(output)
        self.valid_horizon = valid_horizon

    def get_modality_config(self):
        return {"action": _ModalityConfig(list(range(self.valid_horizon)))}

    def reset(self, options=None):
        return {"status": "reset"}

    def get_action(self, observation, options=None):
        self.policy.model.action_head.model.emit()
        return {
            "action.end_effector_position": torch.randn(
                1,
                self.valid_horizon,
                3,
            ).numpy()
        }, {}


class _TransformerBlocks:
    def __init__(self, outputs: list[torch.Tensor]):
        self.transformer_blocks = [_HookTarget(output) for output in outputs]


class _PathwayActionHead:
    def __init__(self, layer_outputs: list[torch.Tensor], vl_output: torch.Tensor):
        self.action_horizon = int(layer_outputs[0].shape[1])
        self.model = _TransformerBlocks(layer_outputs)
        self.vlln = _HookTarget(vl_output)


class _PathwayRootModel:
    def __init__(self, layer_outputs: list[torch.Tensor], vl_output: torch.Tensor):
        self.action_head = _PathwayActionHead(layer_outputs, vl_output)


class _PathwayInnerPolicy:
    def __init__(self, layer_outputs: list[torch.Tensor], vl_output: torch.Tensor):
        self.model = _PathwayRootModel(layer_outputs, vl_output)


class _FakePathwaySimPolicy:
    def __init__(
        self,
        layer_outputs: list[torch.Tensor],
        vl_output: torch.Tensor,
        valid_horizon: int,
    ):
        self.policy = _PathwayInnerPolicy(layer_outputs, vl_output)
        self.valid_horizon = valid_horizon

    def get_modality_config(self):
        return {"action": _ModalityConfig(list(range(self.valid_horizon)))}

    def reset(self, options=None):
        return {"status": "reset"}

    def get_action(self, observation, options=None):
        action_head = self.policy.model.action_head
        for block in action_head.model.transformer_blocks:
            block.emit()
        if action_head.vlln.hook is not None:
            action_head.vlln.emit()
        return {
            "action.end_effector_position": torch.randn(
                1,
                self.valid_horizon,
                3,
            ).numpy()
        }, {}


def _import_feature_server():
    if importlib.util.find_spec("msgpack") is None:
        raise unittest.SkipTest("msgpack is not installed in this Python env")
    if importlib.util.find_spec("zmq") is None:
        raise unittest.SkipTest("pyzmq is not installed in this Python env")

    spec = importlib.util.spec_from_file_location(
        "safe_groot_feature_server_under_test", FEATURE_SERVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSafeN16FeaturePolicy(unittest.TestCase):
    def setUp(self):
        self.module = _import_feature_server()

    def test_get_action_with_features_returns_valid_horizon_features(self):
        output = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
        sim_policy = _FakeSimPolicy(output, valid_horizon=3)
        policy = self.module.SafeN16FeaturePolicy(
            sim_policy,
            feature_dtype="float32",
            feature_slice="valid",
        )

        response = policy.get_action_with_features({"obs": "value"})

        np.testing.assert_allclose(
            response["hidden_states"],
            output[:, :3, :].reshape(1, 1, 3, 2).numpy(),
        )
        self.assertIn("action", response)
        self.assertEqual(response["feature_kind"], self.module.SAFE_FEATURE_KIND_VALID)
        self.assertEqual(response["feature_axes"], self.module.SAFE_FEATURE_AXES_VALID)
        self.assertEqual(response["exported_action_token_count"], 3)
        self.assertEqual(response["feature_action_horizon"], 3)
        self.assertEqual(response["valid_action_horizon"], 3)
        self.assertEqual(response["model_action_horizon"], 5)
        self.assertEqual(response["num_inference_timesteps"], 1)

    def test_get_action_with_features_honors_inference_seed_without_leaking_rng(self):
        output = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
        policy = self.module.SafeN16FeaturePolicy(
            _FakeSimPolicy(output, valid_horizon=3),
            feature_dtype="float32",
            feature_slice="valid",
        )

        torch.manual_seed(111)
        before = torch.rand(3)
        seeded_a = policy.get_action_with_features(
            {"obs": "value"},
            options={"inference_seed": 4242},
        )["action"]["action.end_effector_position"]
        after_a = torch.rand(3)

        torch.manual_seed(111)
        expected_before = torch.rand(3)
        expected_after = torch.rand(3)
        seeded_b = policy.get_action_with_features(
            {"obs": "value"},
            options={"inference_seed": 4242},
        )["action"]["action.end_effector_position"]

        np.testing.assert_allclose(seeded_a, seeded_b)
        torch.testing.assert_close(before, expected_before)
        torch.testing.assert_close(after_a, expected_after)

    def test_feature_action_horizon_cannot_exceed_valid_horizon(self):
        output = torch.zeros((1, 5, 2), dtype=torch.float32)
        sim_policy = _FakeSimPolicy(output, valid_horizon=3)
        policy = self.module.SafeN16FeaturePolicy(
            sim_policy,
            feature_dtype="float32",
            feature_slice="valid",
            feature_action_horizon=4,
        )

        with self.assertRaisesRegex(ValueError, "exceeds exportable horizon"):
            policy.get_action_with_features({})

    def test_multilayer_features_default_to_dit_only(self):
        layer_outputs = [
            torch.ones((1, 5, 2), dtype=torch.float32),
            torch.full((1, 5, 2), 3.0, dtype=torch.float32),
        ]
        vl_output = torch.ones((1, 2, 4), dtype=torch.float32)
        policy = self.module.SafeN16FeaturePolicy(
            _FakePathwaySimPolicy(layer_outputs, vl_output, valid_horizon=3),
            feature_dtype="float32",
            feature_slice="valid",
        )
        policy.capture_layers = [0, 1]

        response = policy.get_action_with_multilayer_features({"obs": "value"})

        self.assertEqual(tuple(response["hidden_states"].shape), (2, 2))
        self.assertEqual(
            response["feature_kind"],
            self.module.MULTILAYER_FEATURE_KIND,
        )
        self.assertNotIn("vl_hidden_states", response)
        self.assertNotIn("vl_feature_kind", response)
        self.assertNotIn("vl_feature_dim", response)

    def test_multilayer_features_include_vl_only_when_enabled(self):
        layer_outputs = [
            torch.ones((1, 5, 2), dtype=torch.float32),
            torch.full((1, 5, 2), 3.0, dtype=torch.float32),
        ]
        vl_output = torch.tensor(
            [[[1.0, 3.0, 5.0, 7.0], [3.0, 5.0, 7.0, 9.0]]],
            dtype=torch.float32,
        )
        policy = self.module.SafeN16FeaturePolicy(
            _FakePathwaySimPolicy(layer_outputs, vl_output, valid_horizon=3),
            feature_dtype="float32",
            feature_slice="valid",
        )
        policy.capture_layers = [0, 1]
        policy.capture_vl = True

        response = policy.get_action_with_multilayer_features({"obs": "value"})

        self.assertEqual(tuple(response["hidden_states"].shape), (2, 2))
        np.testing.assert_allclose(
            response["vl_hidden_states"],
            np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
        )
        self.assertEqual(response["vl_feature_kind"], self.module.VL_FEATURE_KIND)
        self.assertEqual(response["vl_feature_axes"], self.module.VL_FEATURE_AXES)
        self.assertEqual(response["vl_feature_dim"], 4)


if __name__ == "__main__":
    unittest.main()
