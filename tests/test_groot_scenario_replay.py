"""GR00T RoboCasa scenario replay helper tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.policies.groot.robocasa.scenario_replay import (
    ep_meta_manifest_path,
    get_robocasa_ep_meta,
    json_safe,
    load_ep_meta_manifest,
    set_robocasa_ep_meta,
    write_ep_meta_manifest,
)


class _LeafEnv:
    def __init__(self):
        self.ep_meta = {"layout_id": np.int64(7), "pose": np.array([1.0, 2.0])}
        self.received = None

    def get_ep_meta(self):
        return self.ep_meta

    def set_ep_meta(self, ep_meta):
        self.received = ep_meta


class _Wrapper:
    def __init__(self, env):
        self.env = env


class TestScenarioReplay(unittest.TestCase):
    def test_json_safe_converts_numpy_payloads(self):
        self.assertEqual(
            json_safe({"a": np.array([1, 2]), "b": np.float32(3.5)}),
            {"a": [1, 2], "b": 3.5},
        )

    def test_manifest_round_trip_validates_env_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = ep_meta_manifest_path(Path(tmp), "CloseSingleDoor", 123)
            write_ep_meta_manifest(
                path,
                env_name="CloseSingleDoor",
                scenario_seed=123,
                ep_meta={"layout_id": np.int64(7), "pose": np.array([1.0, 2.0])},
                robocasa_env_source="robocasa365",
                sort_keys=True,
            )

            self.assertEqual(
                load_ep_meta_manifest(path, env_name="CloseSingleDoor", scenario_seed=123),
                {"layout_id": 7, "pose": [1.0, 2.0]},
            )
            payload = json.loads(path.read_text())
            self.assertEqual(payload["format"], "robocasa_ep_meta_manifest.v1")
            self.assertEqual(payload["robocasa_env_source"], "robocasa365")
            with self.assertRaisesRegex(ValueError, "env mismatch"):
                load_ep_meta_manifest(path, env_name="OtherTask", scenario_seed=123)
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                load_ep_meta_manifest(path, env_name="CloseSingleDoor", scenario_seed=124)

    def test_get_and_set_ep_meta_walk_wrapper_stack_and_copy_payload(self):
        leaf = _LeafEnv()
        wrapped = _Wrapper(_Wrapper(leaf))

        self.assertEqual(
            get_robocasa_ep_meta(wrapped),
            {"layout_id": 7, "pose": [1.0, 2.0]},
        )

        source = {"layout_id": 9, "nested": {"value": [1]}}
        set_robocasa_ep_meta(wrapped, source)
        source["nested"]["value"].append(2)

        self.assertEqual(leaf.received, {"layout_id": 9, "nested": {"value": [1]}})


if __name__ == "__main__":
    unittest.main()
