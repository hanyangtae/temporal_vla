from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECT_COMMON_PATH = REPO_ROOT / "scripts" / "safe" / "lerobot" / "collect_common.py"


def _load_collect_common():
    spec = importlib.util.spec_from_file_location(
        "safe_lerobot_collect_common_under_test",
        COLLECT_COMMON_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSafeEpisodeCollector(unittest.TestCase):
    def setUp(self):
        self.module = _load_collect_common()

    def test_record_accepts_vla_client_feature_metadata(self):
        collector = self.module.SafeEpisodeCollector(
            task_suite_name="libero_object",
            task_id=0,
            task_description="pick up the alphabet soup",
            episode_idx=2,
            env_name="libero/libero_object",
        )
        hidden = np.zeros((10, 50, 1024), dtype=np.float16)

        collector.record(
            {
                "hidden_states": hidden,
                "kind": "pi05_action_expert_pre_velocity",
                "axes": ["denoising_step", "action_step", "feature_dim"],
                "num_inference_timesteps": 10,
                "exported_action_token_count": 50,
                "feature_action_horizon": 50,
                "valid_action_horizon": 50,
            },
            action_vector=np.ones(7, dtype=np.float32),
            action={"action.eef_pos": [0.0, 0.0, 0.0]},
        )
        payload = collector.payload(success=True)

        self.assertEqual(payload["episode_success"], 1)
        self.assertEqual(payload["feature_kind"], "pi05_action_expert_pre_velocity")
        self.assertEqual(
            payload["feature_axes"], ["denoising_step", "action_step", "feature_dim"]
        )
        self.assertEqual(payload["num_inference_timesteps"], 10)
        self.assertEqual(payload["exported_action_token_count"], 50)
        self.assertEqual(payload["feature_action_horizon"], 50)
        self.assertEqual(payload["valid_action_horizon"], 50)
        self.assertEqual(payload["model_action_horizon"], 50)
        self.assertEqual(payload["feature_dim"], 1024)
        self.assertEqual(len(payload["hidden_states"]), 1)
        np.testing.assert_array_equal(payload["hidden_states"][0], hidden)

    def test_record_preserves_zero_metadata_values(self):
        collector = self.module.SafeEpisodeCollector(
            task_suite_name="libero_object",
            task_id=0,
            task_description="pick up the alphabet soup",
            episode_idx=2,
            env_name="libero/libero_object",
        )
        collector.num_inference_timesteps = 10
        collector.exported_action_token_count = 50
        collector.feature_action_horizon = 50
        collector.valid_action_horizon = 50
        collector.model_action_horizon = 50

        collector.record(
            {
                "hidden_states": np.zeros((1, 1, 2), dtype=np.float16),
                "kind": "pi05_action_expert_pre_velocity",
                "axes": ["denoising_step", "action_step", "feature_dim"],
                "num_inference_timesteps": 0,
                "exported_action_token_count": 0,
                "feature_action_horizon": 0,
                "valid_action_horizon": 0,
                "model_action_horizon": 0,
            }
        )
        payload = collector.payload(success=True)

        self.assertEqual(payload["num_inference_timesteps"], 0)
        self.assertEqual(payload["exported_action_token_count"], 0)
        self.assertEqual(payload["feature_action_horizon"], 0)
        self.assertEqual(payload["valid_action_horizon"], 0)
        self.assertEqual(payload["model_action_horizon"], 0)

    def test_record_falls_back_to_hidden_shape_when_feature_dim_is_none(self):
        collector = self.module.SafeEpisodeCollector(
            task_suite_name="libero_object",
            task_id=0,
            task_description="pick up the alphabet soup",
            episode_idx=2,
            env_name="libero/libero_object",
        )

        collector.record(
            {
                "hidden_states": np.zeros((2, 3, 17), dtype=np.float16),
                "feature_dim": None,
            }
        )
        payload = collector.payload(success=True)

        self.assertEqual(payload["feature_dim"], 17)


if __name__ == "__main__":
    unittest.main()
