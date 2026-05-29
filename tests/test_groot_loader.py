"""GR00T loader path resolution tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.policies.groot.loader import (
    load_state_dims_from_statistics,
    resolve_device,
    resolve_model_path,
)


class TestResolveModelPath(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.local_checkpoint = Path(self.tmpdir.name) / "checkpoint"
        self.local_checkpoint.mkdir()
        self.container_checkpoint = "/temporal_vla/{}".format(
            self.local_checkpoint.relative_to(Path.cwd()).as_posix()
        )
        with (self.local_checkpoint / "statistics.json").open("w") as f:
            json.dump(
                {
                    "robocasa_panda_omron": {
                        "state": {
                            "gripper_qpos": {"mean": [0.0, 0.0]},
                            "end_effector_position_relative": {
                                "mean": [0.0, 0.0, 0.0],
                            },
                        }
                    }
                },
                f,
            )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_existing_local_path_is_unchanged(self):
        path = self.local_checkpoint.resolve()
        self.assertEqual(resolve_model_path(str(path)), str(path))

    def test_temporal_vla_container_path_maps_to_current_checkout(self):
        resolved = resolve_model_path(self.container_checkpoint)
        self.assertEqual(resolved, str(self.local_checkpoint.resolve()))

    def test_unknown_absolute_path_is_unchanged(self):
        path = "/does/not/exist/GR00T-N1.6-3B"
        self.assertEqual(resolve_model_path(path), path)

    def test_hf_repo_id_is_unchanged(self):
        repo_id = "nvidia/GR00T-N1.6-3B"
        self.assertEqual(resolve_model_path(repo_id), repo_id)

    def test_state_dims_load_from_resolved_container_path(self):
        dims = load_state_dims_from_statistics(
            self.container_checkpoint,
            "robocasa_panda_omron",
        )

        self.assertEqual(dims["gripper_qpos"], 2)
        self.assertEqual(dims["end_effector_position_relative"], 3)

    def test_explicit_device_overrides_profile_default(self):
        self.assertEqual(resolve_device({"device": "cuda"}, "cpu"), "cpu")

    def test_profile_device_used_when_no_override(self):
        self.assertEqual(resolve_device({"device": "cuda"}, None), "cuda")


if __name__ == "__main__":
    unittest.main()
