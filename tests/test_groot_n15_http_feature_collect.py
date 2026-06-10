from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECT_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "collect"
    / "http_feature_collect.py"
)


def _load_module():
    if importlib.util.find_spec("torch") is None:
        raise unittest.SkipTest("torch is not installed in this Python env")
    spec = importlib.util.spec_from_file_location(
        "groot_n15_http_feature_collect_under_test",
        COLLECT_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_observation() -> dict:
    return {
        "video.robot0_agentview_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((8, 8, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((8, 8, 3), dtype=np.uint8),
        "annotation.human.task_description": "open the fridge",
        "state.end_effector_position_relative": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "state.end_effector_rotation_relative": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "state.gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
        "state.base_position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "state.base_rotation": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }


def test_n15_http_feature_client_records_feature_steps_as_safe_records():
    module = _load_module()
    calls = []

    def fake_predict_with_features(self, images, states, instruction, inference_seed=None):
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
                "hidden_states": np.ones((4, 16, 2), dtype=np.float16),
                "kind": "groot_n15_dit_action_tokens_pre_decode",
                "axes": ["denoising_step", "action_step", "feature_dim"],
                "num_inference_timesteps": 4,
                "feature_action_horizon": 16,
                "exported_action_token_count": 16,
                "model_action_horizon": 16,
            },
            12.5,
        )

    with mock.patch.object(module.VLAClient, "predict_with_features", fake_predict_with_features):
        client = module.N15LerobotHttpFeatureClient(
            "http://127.0.0.1:8400",
            inference_seed=5000,
        )
        action, _ = client.get_action(_raw_observation())

    assert calls[0]["instruction"] == "open the fridge"
    assert calls[0]["inference_seed"] == 5000
    assert set(calls[0]["images"]) == {"side_0", "side_1", "wrist_0"}
    assert "observation.state.eef_pos_rel" in calls[0]["states"]
    assert set(action) == {
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    }
    assert tuple(action["action.end_effector_position"].shape) == (16, 3)
    assert tuple(action["action.gripper_close"].shape) == (16, 1)
    assert len(client.records) == 1
    assert tuple(client.records[0]["hidden_state"].shape) == (4, 16, 2)
    assert client.feature_kind == "groot_n15_dit_action_tokens_pre_decode"
    assert client.feature_axes == ["denoising_step", "action_step", "feature_dim"]
    assert client.exported_action_token_count == 16
    assert client.feature_action_horizon == 16
    assert client.model_action_horizon == 16


def test_n15_http_feature_client_skips_buffered_steps_without_features():
    module = _load_module()

    def fake_predict_with_features(self, images, states, instruction, inference_seed=None):
        return (
            {
                "action.eef_pos": np.ones((1, 3), dtype=np.float32),
                "action.eef_axisangle": np.ones((1, 3), dtype=np.float32),
                "action.gripper": np.ones((1, 1), dtype=np.float32),
                "action.base_motion": np.zeros((1, 4), dtype=np.float32),
                "action.control_mode": np.ones((1, 1), dtype=np.float32),
            },
            None,
            3.0,
        )

    with mock.patch.object(module.VLAClient, "predict_with_features", fake_predict_with_features):
        client = module.N15LerobotHttpFeatureClient("http://127.0.0.1:8400")
        action, _ = client.get_action(_raw_observation())

    assert "action.end_effector_position" in action
    assert client.records == []


def test_n15_http_feature_run_passes_replay_meta_and_upstream_video(tmp_path: Path):
    module = _load_module()
    output_dir = tmp_path / "rollouts" / "CloseFridge"
    ep_meta_dir = tmp_path / "ep_meta"
    env_name = "robocasa_panda_omron/CloseFridge_PandaOmron_Env"
    ep_meta = {"layout_id": 7, "fixture_refs": {"fridge": "fixture_a"}}
    manifest_path = module.ep_meta_manifest_path(ep_meta_dir, env_name, 100000)
    module.write_collect_ep_meta_manifest(
        manifest_path,
        env_name=env_name,
        scenario_seed=100000,
        ep_meta=ep_meta,
        robocasa_env_source="robocasa365",
    )

    class FakePolicy:
        def __init__(self, *args, **kwargs):
            self.records = [{"hidden_state": np.zeros((1,), dtype=np.float32)}]
            self.task_description = "close the fridge"
            self.exported_action_token_count = 16
            self.feature_kind = "groot_n15_dit_action_tokens_pre_decode"
            self.feature_axes = ["denoising_step", "action_step", "feature_dim"]
            self.feature_slice = None
            self.feature_action_horizon = 16
            self.valid_action_horizon = None
            self.model_action_horizon = 16
            self.num_inference_timesteps = 4

        def wait_until_ready(self, max_wait):
            return {"status": "ok"}

        def reset(self):
            return None

        def get_action(self, obs):
            return {"action.end_effector_position": np.zeros((1, 3), dtype=np.float32)}, {}

    class FakeEnv:
        def __init__(self, video_dir: Path):
            self.video_dir = video_dir
            self.replayed = None

        def set_ep_meta(self, value):
            self.replayed = value

        def get_ep_meta(self):
            return {"layout_id": 99}

        def reset(self, seed=None):
            return {"obs": seed}, {}

        def step(self, action):
            return {"obs": 1}, 0.0, True, False, {}

        def render(self):
            path = self.video_dir / "upstream.mp4"
            path.write_bytes(b"fake mp4")
            return str(path)

        def close(self):
            return None

    captured = {}

    def fake_make_env(*args, **kwargs):
        captured["make_env_kwargs"] = kwargs
        return FakeEnv(kwargs["video_dir"])

    def fake_write_safe_triplet(**kwargs):
        captured["triplet"] = kwargs

    args = argparse.Namespace(
        vla_server="http://127.0.0.1:8400",
        task="CloseFridge",
        env_name=env_name,
        split="target",
        output_dir=str(output_dir),
        task_id=0,
        task_description=None,
        episode_start_idx=0,
        n_episodes=1,
        max_episode_steps=1,
        seed=100000,
        inference_seed=424242,
        n_action_steps=16,
        ep_meta_dir=ep_meta_dir,
        ep_meta_load_env_name=env_name,
        timeout=10.0,
        wait_ready=False,
        video_fps=20,
        steps_per_render=2,
    )

    with (
        mock.patch.object(module, "parse_args", return_value=args),
        mock.patch.object(module, "N15LerobotHttpFeatureClient", FakePolicy),
        mock.patch.object(module, "make_env", fake_make_env),
        mock.patch.object(module, "task_horizon", return_value=1),
        mock.patch.object(module, "step_success", return_value=True),
        mock.patch.object(module, "write_safe_triplet", side_effect=fake_write_safe_triplet),
    ):
        result = module.run()

    assert result["episodes"][0]["success"] is True
    assert captured["make_env_kwargs"]["env_name"] == env_name
    assert captured["make_env_kwargs"]["scenario_seed"] == 100000
    assert captured["make_env_kwargs"]["n_action_steps"] == 16
    assert captured["make_env_kwargs"]["max_episode_steps"] == 1
    assert captured["triplet"]["env_name"] == env_name
    assert captured["triplet"]["ep_meta"] == ep_meta
    assert captured["triplet"]["upstream_video_path"].name == "upstream.mp4"
    assert captured["triplet"]["n_action_steps"] == 16
