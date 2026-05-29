"""SAFE GR00T N1.6 collector endpoint contract tests."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECT_ROLLOUT_PATH = (
    REPO_ROOT / "scripts" / "safe" / "groot_n16" / "robocasa" / "collect" / "collect_rollout.py"
)


def _import_collect_rollout():
    for module_name in ("gymnasium", "msgpack", "torch", "tqdm", "zmq"):
        if importlib.util.find_spec(module_name) is None:
            raise unittest.SkipTest(f"{module_name} is not installed in this Python env")

    spec = importlib.util.spec_from_file_location(
        "safe_groot_collect_rollout_under_test", COLLECT_ROLLOUT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _raw_observation() -> dict:
    return {
        "video.robot0_agentview_left": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
        "state.base_position": np.zeros((1, 1, 3), dtype=np.float32),
        "state.base_rotation": np.zeros((1, 1, 4), dtype=np.float32),
        "state.end_effector_position_relative": np.zeros((1, 1, 3), dtype=np.float32),
        "state.end_effector_rotation_relative": np.zeros((1, 1, 4), dtype=np.float32),
        "state.gripper_qpos": np.zeros((1, 1, 2), dtype=np.float32),
        "annotation.human.task_description": ("open the cabinet",),
    }


class TestSafeCollectEndpointContract(unittest.TestCase):
    def setUp(self):
        self.module = _import_collect_rollout()

    def test_headless_rendering_defaults_to_egl_backend(self):
        with mock.patch.dict(self.module.os.environ, {}, clear=True):
            self.module._configure_headless_rendering()

            self.assertEqual(self.module.os.environ["MUJOCO_GL"], "egl")
            self.assertEqual(self.module.os.environ["PYOPENGL_PLATFORM"], "egl")

    def test_prepare_observation_adds_raw_and_res256_aliases(self):
        filtered = self.module._prepare_observation(_raw_observation())

        for key in self.module.REQUIRED_OBS_KEYS:
            self.assertIn(key, filtered)
        self.assertIs(
            filtered["video.res256_image_side_0"],
            filtered["video.robot0_agentview_left"],
        )
        self.assertEqual(
            filtered["annotation.human.action.task_description"],
            ("open the cabinet",),
        )

    def test_collecting_client_uses_get_action_with_features_endpoint(self):
        module = self.module

        class _FakeClient(module.N16SafeCollectingPolicyClient):
            def __init__(self):
                self.endpoint = "get_action_with_features"
                self.inference_seed = 4242
                self.records = []
                self.task_description = None
                self.feature_kind = None
                self.feature_axes = None
                self.feature_slice = None
                self.exported_action_token_count = None
                self.feature_action_horizon = None
                self.valid_action_horizon = None
                self.model_action_horizon = None
                self.num_inference_timesteps = None
                self.last_endpoint = None
                self.last_data = None

            def call_endpoint(self, endpoint, data=None, requires_input=True):
                self.last_endpoint = endpoint
                self.last_data = data
                return {
                    "action": {
                        "action.end_effector_position": np.ones((1, 16, 3), dtype=np.float32),
                        "action.end_effector_rotation": np.ones((1, 16, 3), dtype=np.float32),
                        "action.gripper_close": np.ones((1, 16, 1), dtype=np.float32),
                    },
                    "hidden_states": np.zeros((1, 4, 8, 2), dtype=np.float32),
                    "feature_kind": "groot_n16_dit_valid_action_tokens_pre_velocity",
                    "feature_axes": [
                        "denoising_step",
                        "valid_action_step",
                        "feature_dim",
                    ],
                    "feature_slice": "valid",
                    "exported_action_token_count": 8,
                    "feature_action_horizon": 8,
                    "valid_action_horizon": 16,
                    "model_action_horizon": 50,
                    "num_inference_timesteps": 4,
                }

        client = _FakeClient()
        action, _ = client.get_action(_raw_observation())

        self.assertEqual(client.last_endpoint, "get_action_with_features")
        self.assertIn("observation", client.last_data)
        self.assertEqual(client.last_data["options"]["inference_seed"], 4242)
        self.assertEqual(client.task_description, "open the cabinet")
        self.assertIn("action.end_effector_position", action)
        self.assertEqual(tuple(client.records[0]["hidden_state"].shape), (4, 8, 2))
        self.assertEqual(client.exported_action_token_count, 8)
        self.assertEqual(client.feature_action_horizon, 8)
        self.assertEqual(client.valid_action_horizon, 16)
        self.assertEqual(client.model_action_horizon, 50)

        client.get_action(_raw_observation())
        self.assertEqual(client.last_data["options"]["inference_seed"], 4243)

    def test_http_collecting_client_records_act_with_features_as_safe_schema(self):
        module = self.module
        calls = []

        def _fake_reset(self):
            self.reset_count = getattr(self, "reset_count", 0) + 1

        def _fake_predict_with_features(
            self,
            images,
            states,
            instruction,
            inference_seed=None,
        ):
            calls.append(
                {
                    "images": dict(images),
                    "states": dict(states),
                    "instruction": instruction,
                    "inference_seed": inference_seed,
                }
            )
            return (
                {
                    "action.eef_pos": np.ones((16, 3), dtype=np.float32),
                    "action.eef_axisangle": np.ones((16, 3), dtype=np.float32) * 2,
                    "action.gripper": np.ones((16, 1), dtype=np.float32) * 3,
                    "action.base_motion": np.zeros((16, 4), dtype=np.float32),
                    "action.control_mode": np.ones((16, 1), dtype=np.float32),
                },
                {
                    "hidden_states": np.zeros((1, 4, 8, 2), dtype=np.float16),
                    "kind": "groot_n16_dit_valid_action_tokens_pre_velocity",
                    "axes": [
                        "denoising_step",
                        "valid_action_step",
                        "feature_dim",
                    ],
                    "slice": "valid",
                    "feature_action_horizon": 8,
                    "valid_action_horizon": 16,
                    "model_action_horizon": 50,
                    "num_inference_timesteps": 4,
                },
                12.5,
            )

        vla_client_cls = sys.modules["collect_policy_clients"].VLAClient
        with mock.patch.object(
            vla_client_cls,
            "reset",
            _fake_reset,
        ), mock.patch.object(
            vla_client_cls,
            "predict_with_features",
            _fake_predict_with_features,
        ):
            client = module.HttpN16SafeCollectingPolicyClient(
                "127.0.0.1",
                8500,
                timeout_s=3.0,
                inference_seed=4242,
            )
            client.reset()
            action, _ = client.get_action(_raw_observation())

        self.assertIsInstance(client, vla_client_cls)
        self.assertEqual(client.url, "http://127.0.0.1:8500")
        self.assertEqual(client.timeout, 3.0)
        self.assertEqual(client.reset_count, 1)
        call = calls[0]
        self.assertEqual(set(call["images"]), {"left", "right", "wrist"})
        self.assertIn("observation.state.eef_pos_rel", call["states"])
        self.assertEqual(call["instruction"], "open the cabinet")
        self.assertEqual(call["inference_seed"], 4242)
        self.assertEqual(tuple(action["action.end_effector_position"].shape), (1, 16, 3))
        self.assertEqual(tuple(client.records[0]["hidden_state"].shape), (4, 8, 2))
        self.assertEqual(client.feature_kind, "groot_n16_dit_valid_action_tokens_pre_velocity")
        self.assertEqual(client.feature_slice, "valid")
        self.assertEqual(client.exported_action_token_count, 8)
        self.assertEqual(client.feature_action_horizon, 8)
        self.assertEqual(client.valid_action_horizon, 16)
        self.assertEqual(client.model_action_horizon, 50)


if __name__ == "__main__":
    unittest.main()
