"""GR00T RoboCasa HTTP IO adapter tests."""

from __future__ import annotations

import unittest

import numpy as np

from src.policies.groot.robocasa.io import (
    convert_http_actions_to_groot_chunk,
    convert_http_actions_to_groot_step,
    http_server_url,
    prepare_groot_robocasa_http_request,
    prepare_groot_robocasa_observation,
)


def _raw_observation() -> dict:
    return {
        "video.robot0_agentview_left": np.full((1, 2, 4, 4, 3), 1, dtype=np.uint8),
        "video.robot0_agentview_right": np.full((1, 2, 4, 4, 3), 2, dtype=np.uint8),
        "video.robot0_eye_in_hand": np.full((1, 2, 4, 4, 3), 3, dtype=np.uint8),
        "state.base_position": np.ones((1, 2, 3), dtype=np.float32),
        "state.base_rotation": np.ones((1, 2, 4), dtype=np.float32) * 2,
        "state.end_effector_position_relative": np.ones((1, 2, 3), dtype=np.float32) * 3,
        "state.end_effector_rotation_relative": np.ones((1, 2, 4), dtype=np.float32) * 4,
        "state.gripper_qpos": np.ones((1, 2, 2), dtype=np.float32) * 5,
        "annotation.human.task_description": ("open the cabinet",),
    }


class TestGrootRoboCasaObservation(unittest.TestCase):
    def test_strict_observation_adds_raw_res256_and_language_aliases(self):
        filtered = prepare_groot_robocasa_observation(_raw_observation(), strict=True)

        self.assertIs(
            filtered["video.res256_image_side_0"],
            filtered["video.robot0_agentview_left"],
        )
        self.assertIs(
            filtered["video.res256_image_side_1"],
            filtered["video.robot0_agentview_right"],
        )
        self.assertIs(
            filtered["video.res256_image_wrist_0"],
            filtered["video.robot0_eye_in_hand"],
        )
        self.assertEqual(
            filtered["annotation.human.action.task_description"],
            ("open the cabinet",),
        )

    def test_strict_observation_reports_missing_contract_keys(self):
        with self.assertRaisesRegex(KeyError, "state.base_rotation"):
            prepare_groot_robocasa_observation(
                {"annotation.human.task_description": "open the cabinet"},
                strict=True,
            )

    def test_http_request_uses_latest_frame_and_state_vectors(self):
        request = prepare_groot_robocasa_http_request(_raw_observation(), strict=True)

        self.assertEqual(set(request.images), {"left", "right", "wrist"})
        np.testing.assert_array_equal(
            request.images["left"],
            np.full((4, 4, 3), 1, dtype=np.uint8),
        )
        self.assertEqual(
            request.states["observation.state.eef_pos_rel"].shape,
            (3,),
        )
        np.testing.assert_array_equal(
            request.states["observation.state.gripper_qpos"],
            np.ones((2,), dtype=np.float32) * 5,
        )
        self.assertEqual(request.instruction, "open the cabinet")

    def test_non_strict_request_allows_partial_eval_observation(self):
        request = prepare_groot_robocasa_http_request(
            {
                "video.res256_image_side_0": np.zeros((4, 4, 3), dtype=np.uint8),
                "annotation.human.action.task_description": "close the drawer",
            },
            strict=False,
        )

        self.assertEqual(set(request.images), {"left"})
        self.assertEqual(request.states, {})
        self.assertEqual(request.instruction, "close the drawer")


class TestGrootRoboCasaActions(unittest.TestCase):
    def test_converts_http_action_chunk_to_native_batched_chunk(self):
        chunk = convert_http_actions_to_groot_chunk(
            {
                "action.eef_pos": np.ones((16, 3), dtype=np.float32),
                "action.eef_axisangle": np.ones((16, 3), dtype=np.float32) * 2,
                "action.gripper": np.ones((16, 1), dtype=np.float32) * 3,
            }
        )

        self.assertEqual(
            tuple(chunk["action.end_effector_position"].shape),
            (1, 16, 3),
        )
        self.assertEqual(
            tuple(chunk["action.end_effector_rotation"].shape),
            (1, 16, 3),
        )
        self.assertEqual(tuple(chunk["action.gripper_close"].shape), (1, 16, 1))

    def test_converts_http_action_chunk_to_native_first_step(self):
        step = convert_http_actions_to_groot_step(
            {
                "action.eef_pos": np.arange(48, dtype=np.float32).reshape(16, 3),
                "action.gripper": np.ones((16, 1), dtype=np.float32),
            }
        )

        np.testing.assert_array_equal(
            step["action.end_effector_position"],
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            step["action.gripper_close"],
            np.array([1.0], dtype=np.float32),
        )

    def test_http_url_accepts_full_url_or_host_port(self):
        self.assertEqual(http_server_url("127.0.0.1", 8500), "http://127.0.0.1:8500")
        self.assertEqual(http_server_url("http://localhost:8500/", 123), "http://localhost:8500")


if __name__ == "__main__":
    unittest.main()
