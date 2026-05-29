"""GR00T schema helper tests."""

from __future__ import annotations

import unittest

import numpy as np

from src.policies.groot.schema import (
    GROOT_ENV_VIDEO_TO_UNIFIED_CAM,
    GROOT_TO_UNIFIED_STATE,
    UNIFIED_TO_GROOT_ACTION,
    build_video_mapping,
    extract_groot_instruction,
)


class TestGrootEnvSchema(unittest.TestCase):
    def test_raw_robocasa_cameras_map_to_unified_aliases(self):
        self.assertEqual(
            GROOT_ENV_VIDEO_TO_UNIFIED_CAM["video.robot0_agentview_left"], "left"
        )
        self.assertEqual(
            GROOT_ENV_VIDEO_TO_UNIFIED_CAM["video.robot0_agentview_right"], "right"
        )
        self.assertEqual(
            GROOT_ENV_VIDEO_TO_UNIFIED_CAM["video.robot0_eye_in_hand"], "wrist"
        )

    def test_state_and_action_reverse_maps_are_derived_from_shared_schema(self):
        self.assertEqual(
            GROOT_TO_UNIFIED_STATE["state.base_position"],
            "observation.state.base_position",
        )
        self.assertEqual(
            UNIFIED_TO_GROOT_ACTION["action.eef_pos"],
            "action.end_effector_position",
        )

    def test_server_video_mapping_accepts_smoke_and_identity_aliases(self):
        mapping = build_video_mapping(
            [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ]
        )

        self.assertEqual(
            mapping["observation.images.side_0"],
            "video.robot0_agentview_left",
        )
        self.assertEqual(
            mapping["observation.images.side_1"],
            "video.robot0_agentview_right",
        )
        self.assertEqual(
            mapping["observation.images.wrist_0"],
            "video.robot0_eye_in_hand",
        )
        self.assertEqual(
            mapping["observation.images.robot0_agentview_left"],
            "video.robot0_agentview_left",
        )

    def test_instruction_accepts_new_and_old_language_keys(self):
        self.assertEqual(
            extract_groot_instruction(
                {"annotation.human.task_description": ("open the cabinet",)}
            ),
            "open the cabinet",
        )
        self.assertEqual(
            extract_groot_instruction(
                {
                    "annotation.human.action.task_description": [
                        "close the toaster",
                    ],
                    "annotation.human.task_description": ("ignored fallback",),
                }
            ),
            "close the toaster",
        )
        self.assertEqual(
            extract_groot_instruction(
                {"annotation.human.task_description": np.array(["turn on the sink"])}
            ),
            "turn on the sink",
        )


if __name__ == "__main__":
    unittest.main()
