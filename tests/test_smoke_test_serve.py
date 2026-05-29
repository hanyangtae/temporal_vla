"""Smoke-test helper contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))

import smoke_test_serve  # noqa: E402


def _profile() -> SimpleNamespace:
    return SimpleNamespace(
        image_preprocess=SimpleNamespace(resolution=8),
        observation_requirements=SimpleNamespace(
            images=["side_0", "side_1", "wrist_0"],
            state=[
                "eef_pos_rel",
                "eef_quat_rel",
                "gripper_qpos",
                "base_position",
                "base_rotation",
            ],
        ),
        emits_subkeys=[
            "action.eef_pos",
            "action.eef_axisangle",
            "action.gripper",
            "action.base_motion",
            "action.control_mode",
        ],
    )


class SmokeTestServeContractTests(unittest.TestCase):
    def test_build_dummy_payload_covers_groot_robocasa_state_keys(self):
        images, states = smoke_test_serve._build_dummy_payload(_profile())

        self.assertEqual(sorted(images), ["side_0", "side_1", "wrist_0"])
        self.assertEqual(images["side_0"].shape, (8, 8, 3))
        self.assertEqual(images["side_0"].dtype, np.uint8)

        expected_shapes = {
            "observation.state.eef_pos_rel": (3,),
            "observation.state.eef_quat_rel": (4,),
            "observation.state.gripper_qpos": (2,),
            "observation.state.base_position": (3,),
            "observation.state.base_rotation": (4,),
        }
        self.assertEqual(set(states), set(expected_shapes))
        for key, shape in expected_shapes.items():
            self.assertEqual(states[key].shape, shape)
            self.assertEqual(states[key].dtype, np.float32)

    def test_check_action_validates_groot_optional_action_dims(self):
        action = {
            "action.eef_pos": np.zeros((2, 3), dtype=np.float32),
            "action.eef_axisangle": np.zeros((2, 3), dtype=np.float32),
            "action.gripper": np.zeros((2, 1), dtype=np.float32),
            "action.base_motion": np.zeros((2, 4), dtype=np.float32),
            "action.control_mode": np.zeros((2, 1), dtype=np.float32),
        }

        smoke_test_serve._check_action(action, _profile())

        action["action.base_motion"] = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(
            smoke_test_serve.SmokeTestFailure,
            "action.base_motion dim mismatch",
        ):
            smoke_test_serve._check_action(action, _profile())


if __name__ == "__main__":
    unittest.main()
