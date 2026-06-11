"""SAFE GR00T N1.6 collector endpoint contract tests."""

from __future__ import annotations

import importlib.util
import argparse
import pickle
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECT_ROLLOUT_PATH = (
    REPO_ROOT / "scripts" / "safe" / "groot_n16" / "robocasa" / "collect" / "collect_rollout.py"
)
COLLECT_POLICY_CLIENTS_PATH = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n16"
    / "robocasa"
    / "collect"
    / "collect_policy_clients.py"
)
COLLECT_ARTIFACTS_PATH = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n16"
    / "robocasa"
    / "collect"
    / "collect_artifacts.py"
)


def _import_collect_rollout():
    for module_name in ("gymnasium", "torch", "tqdm", "zmq"):
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


def _import_collect_policy_clients_direct():
    for path in (
        COLLECT_POLICY_CLIENTS_PATH.parent,
        REPO_ROOT / "scripts" / "utils",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "safe_groot_collect_policy_clients_direct_under_test",
        COLLECT_POLICY_CLIENTS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _import_collect_artifacts_direct():
    collect_dir = COLLECT_ARTIFACTS_PATH.parent
    if str(collect_dir) not in sys.path:
        sys.path.insert(0, str(collect_dir))
    spec = importlib.util.spec_from_file_location(
        "safe_groot_collect_artifacts_direct_under_test",
        COLLECT_ARTIFACTS_PATH,
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
                self._init_safe_feature_records()
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


def test_feature_record_mixin_preserves_zero_metadata_values():
    clients_module = _import_collect_policy_clients_direct()
    mixin_cls = clients_module.SafeFeatureRecordMixin

    class _FakeClient(mixin_cls):
        pass

    client = _FakeClient()
    client._init_safe_feature_records()
    client.exported_action_token_count = 8
    client.feature_action_horizon = 8
    client.valid_action_horizon = 16
    client.model_action_horizon = 50
    client.num_inference_timesteps = 4

    client._update_safe_feature_metadata(
        {
            "feature_kind": "groot_n16_dit_valid_action_tokens_pre_velocity",
            "feature_axes": [
                "denoising_step",
                "valid_action_step",
                "feature_dim",
            ],
            "feature_slice": "valid",
            "exported_action_token_count": 0,
            "feature_action_horizon": 0,
            "valid_action_horizon": 0,
            "model_action_horizon": 0,
            "num_inference_timesteps": 0,
        }
    )

    assert client.exported_action_token_count == 0
    assert client.feature_action_horizon == 0
    assert client.valid_action_horizon == 0
    assert client.model_action_horizon == 0
    assert client.num_inference_timesteps == 0


def test_feature_record_mixin_preserves_block_residual_metadata():
    clients_module = _import_collect_policy_clients_direct()
    mixin_cls = clients_module.SafeFeatureRecordMixin

    class _FakeClient(mixin_cls):
        pass

    client = _FakeClient()
    client._init_safe_feature_records()

    client._update_safe_feature_metadata(
        {
            "feature_kind": "groot_n16_dit_block_residual_kmean_perT_multilayer",
            "feature_axes": ["layer", "token_pos", "feature_dim"],
            "layer_indices": [0, 2, 31],
            "layer_count": 3,
            "token_count": 51,
            "num_inference_timesteps": 4,
        }
    )

    assert client.feature_kind == "groot_n16_dit_block_residual_kmean_perT_multilayer"
    assert client.feature_axes == ["layer", "token_pos", "feature_dim"]
    assert client.layer_indices == [0, 2, 31]
    assert client.capture_layers == [0, 2, 31]
    assert client.layer_count == 3
    assert client.token_count == 51
    assert client.num_inference_timesteps == 4


def test_safe_triplet_writer_records_model_family_and_transport_metadata():
    module = _import_collect_artifacts_direct()

    class _FakePolicy:
        feature_kind = "groot_n16_dit_valid_action_tokens_pre_velocity"
        feature_axes = ["denoising_step", "valid_action_step", "feature_dim"]
        feature_slice = "valid"
        exported_action_token_count = 8
        feature_action_horizon = 8
        valid_action_horizon = 16
        model_action_horizon = 50
        num_inference_timesteps = 4
        records = [
            {
                "hidden_state": np.zeros((4, 8, 2), dtype=np.float32),
                "action_vector": np.zeros(7, dtype=np.float32),
                "groot_action_vector": np.zeros(12, dtype=np.float32),
                "action": {"action.end_effector_position": np.zeros(3, dtype=np.float32)},
            }
        ]

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        source_video = output_dir / "source.mp4"
        source_video.write_bytes(b"video")

        module.write_safe_triplet(
            output_dir=output_dir,
            stem="task0--ep0--succ1",
            policy=_FakePolicy(),
            task_id=0,
            task_description="open the cabinet",
            episode_idx=0,
            scenario_seed=100000,
            episode_success=True,
            env_name="robocasa_panda_omron/OpenCabinet_PandaOmron_Env",
            upstream_video_path=source_video,
            ep_meta={"lang": "open the cabinet"},
            n_action_steps=8,
            robocasa_env_source="robocasa365",
            max_episode_steps=720,
            video_fps=20,
            steps_per_render=2,
            inference_seed=4242,
            model_family="groot_n16",
            policy_transport="http",
            task_suite_name="groot_n16_robocasa",
        )
        with (output_dir / "task0--ep0--succ1.pkl").open("rb") as f:
            payload = pickle.load(f)

    assert payload["model_family"] == "groot_n16"
    assert payload["policy_transport"] == "http"
    assert payload["task_suite_name"] == "groot_n16_robocasa"
    assert payload["video_source"] == "groot_upstream_video_recording_wrapper"
    assert payload["max_episode_steps"] == 720
    assert payload["video_fps"] == 20
    assert payload["steps_per_render"] == 2
    assert payload["inference_seed"] == 4242


def test_safe_triplet_writer_allows_block_residual_token_features():
    module = _import_collect_artifacts_direct()

    class _FakePolicy:
        feature_kind = "groot_n15_dit_block_residual_tokens"
        feature_axes = ["layer", "model_token", "feature_dim"]
        feature_slice = None
        exported_action_token_count = None
        feature_action_horizon = None
        valid_action_horizon = None
        model_action_horizon = 16
        num_inference_timesteps = 4
        capture_layers = [0, 2]
        layer_count = 2
        token_count = 5
        records = [
            {
                "hidden_state": np.zeros((2, 5, 3), dtype=np.float32),
                "action_vector": np.zeros(7, dtype=np.float32),
                "groot_action_vector": np.zeros(12, dtype=np.float32),
                "action": {"action.end_effector_position": np.zeros(3, dtype=np.float32)},
            }
        ]

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        source_video = output_dir / "source.mp4"
        source_video.write_bytes(b"video")

        module.write_safe_triplet(
            output_dir=output_dir,
            stem="task0--ep0--succ1",
            policy=_FakePolicy(),
            task_id=0,
            task_description="open the cabinet",
            episode_idx=0,
            scenario_seed=100000,
            episode_success=True,
            env_name="robocasa_panda_omron/OpenCabinet_PandaOmron_Env",
            upstream_video_path=source_video,
            ep_meta={"lang": "open the cabinet"},
            n_action_steps=16,
            robocasa_env_source="robocasa365",
            max_episode_steps=720,
            video_fps=20,
            steps_per_render=2,
            inference_seed=4242,
            model_family="lerobot_groot_n15",
            policy_transport="http",
            task_suite_name="lerobot_groot_n15_robocasa",
        )
        with (output_dir / "task0--ep0--succ1.pkl").open("rb") as f:
            payload = pickle.load(f)

    assert payload["feature_kind"] == "groot_n15_dit_block_residual_tokens"
    assert payload["feature_axes"] == ["layer", "model_token", "feature_dim"]
    assert payload["exported_action_token_count"] is None
    assert payload["feature_action_horizon"] is None
    assert payload["model_action_horizon"] == 16
    assert payload["capture_layers"] == [0, 2]
    assert payload["layer_count"] == 2
    assert payload["token_count"] == 5


def test_collect_main_advances_scenario_seed_per_episode():
    module = _import_collect_rollout()

    rollout_calls = []
    triplet_calls = []
    manifest_calls = []

    class _FakePolicy:
        def __init__(self, host, port, endpoint=None, inference_seed=None):
            self.host = host
            self.port = port
            self.endpoint = endpoint
            self.inference_seed = inference_seed
            self.records = [object()]
            self.task_description = "close the fridge"
            self.feature_kind = "groot_n16_dit_valid_action_tokens_pre_velocity"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_video = tmp_path / "source.mp4"
        source_video.write_bytes(b"video")
        args = argparse.Namespace(
            policy_client_host="127.0.0.1",
            policy_client_port=5557,
            policy_transport="zmq",
            feature_endpoint="get_action_with_features",
            env_name="robocasa_panda_omron/CloseFridge_PandaOmron_Env",
            output_dir=str(tmp_path / "out"),
            task_id=4,
            task_description=None,
            episode_start_idx=10,
            n_episodes=3,
            n_action_steps=8,
            max_episode_steps=None,
            seed=100,
            inference_seed=4242,
            ep_meta_dir=tmp_path / "ep_meta",
            video_fps=20,
            steps_per_render=2,
        )

        def _fake_run_single_rollout(
            env_name,
            policy,
            wrapper_configs,
            scenario_seed,
            replay_ep_meta,
        ):
            rollout_calls.append(
                {
                    "env_name": env_name,
                    "policy": policy,
                    "scenario_seed": scenario_seed,
                    "replay_ep_meta": replay_ep_meta,
                    "video_fps": wrapper_configs.video.fps,
                    "steps_per_render": wrapper_configs.video.steps_per_render,
                    "video_max_episode_steps": wrapper_configs.video.max_episode_steps,
                    "multistep_max_episode_steps": wrapper_configs.multistep.max_episode_steps,
                    "n_action_steps": wrapper_configs.multistep.n_action_steps,
                }
            )
            return (None, [False], None, source_video, {"scenario_seed": scenario_seed})

        def _fake_write_safe_triplet(**kwargs):
            triplet_calls.append(kwargs)

        def _fake_write_ep_meta_manifest(path, **kwargs):
            manifest_calls.append({"path": path, **kwargs})

        def _fake_ep_meta_manifest_path(ep_meta_dir, env_name, scenario_seed):
            return ep_meta_dir / f"{scenario_seed}.json"

        with mock.patch.object(module, "parse_args", return_value=args), mock.patch.object(
            module,
            "N16SafeCollectingPolicyClient",
            _FakePolicy,
        ), mock.patch.object(
            module,
            "_run_single_rollout",
            _fake_run_single_rollout,
        ), mock.patch.object(
            module,
            "_write_safe_triplet",
            _fake_write_safe_triplet,
        ), mock.patch.object(
            module,
            "_write_ep_meta_manifest",
            _fake_write_ep_meta_manifest,
        ), mock.patch.object(
            module,
            "_ep_meta_manifest_path",
            _fake_ep_meta_manifest_path,
        ):
            module.main()

    assert [call["scenario_seed"] for call in rollout_calls] == [100, 101, 102]
    assert [call["scenario_seed"] for call in triplet_calls] == [100, 101, 102]
    assert [call["scenario_seed"] for call in manifest_calls] == [100, 101, 102]
    assert [call["episode_idx"] for call in triplet_calls] == [10, 11, 12]
    assert [call["policy"].inference_seed for call in rollout_calls] == [4242, 4242, 4242]
    assert [call["n_action_steps"] for call in rollout_calls] == [8, 8, 8]
    assert [call["video_fps"] for call in rollout_calls] == [20, 20, 20]
    assert [call["steps_per_render"] for call in rollout_calls] == [2, 2, 2]
    assert [call["video_max_episode_steps"] for call in rollout_calls] == [720, 720, 720]
    assert [call["multistep_max_episode_steps"] for call in rollout_calls] == [720, 720, 720]
    assert [call["max_episode_steps"] for call in triplet_calls] == [720, 720, 720]
    assert [call["video_fps"] for call in triplet_calls] == [20, 20, 20]
    assert [call["steps_per_render"] for call in triplet_calls] == [2, 2, 2]
    assert [call["inference_seed"] for call in triplet_calls] == [4242, 4242, 4242]


if __name__ == "__main__":
    unittest.main()
