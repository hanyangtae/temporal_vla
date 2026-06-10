"""GR00T HTTP eval path tests for RoboCasa."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOCASA_EVAL_PATH = REPO_ROOT / "scripts" / "eval" / "robocasa_eval.py"


class _NoopLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _import_robocasa_eval():
    modules = {
        "robosuite": types.ModuleType("robosuite"),
        "robocasa": types.ModuleType("robocasa"),
        "robocasa.utils": types.ModuleType("robocasa.utils"),
        "robocasa.utils.dataset_registry": types.ModuleType(
            "robocasa.utils.dataset_registry"
        ),
        "robocasa.utils.gym_utils": types.ModuleType("robocasa.utils.gym_utils"),
        "gymnasium": types.ModuleType("gymnasium"),
        "src.utils.common.logger": types.ModuleType("src.utils.common.logger"),
        "src.processor.factory": types.ModuleType("src.processor.factory"),
        "src.processor.types": types.ModuleType("src.processor.types"),
    }
    modules["robosuite"].load_part_controller_config = lambda default_controller: {
        "controller": default_controller,
    }
    modules["robosuite"].make = lambda **_kwargs: None
    modules["robocasa.utils.dataset_registry"].TASK_SET_REGISTRY = {}
    modules["robocasa.utils.gym_utils"].GrootRoboCasaEnv = object
    modules["gymnasium"].make = lambda *_args, **_kwargs: None
    modules["src.utils.common.logger"].create_module_logger = lambda _name: _NoopLogger()

    class _TransitionKey:
        OBSERVATION = "observation"
        ACTION = "action"

    modules["src.processor.types"].TransitionKey = _TransitionKey

    class _FakeGrootObsPipeline:
        def __call__(self, data):
            obs = data[_TransitionKey.OBSERVATION]
            instruction = obs.get(
                "annotation.human.action.task_description",
                obs.get("annotation.human.task_description", ""),
            )
            if isinstance(instruction, np.ndarray):
                instruction = instruction.reshape(-1)[0] if instruction.size else ""
            elif isinstance(instruction, (list, tuple)):
                instruction = instruction[0] if instruction else ""
            return {
                _TransitionKey.OBSERVATION: {
                    "observation.images.left": obs.get("video.robot0_agentview_left"),
                    "observation.images.right": obs.get("video.robot0_agentview_right"),
                    "observation.images.wrist": obs.get("video.robot0_eye_in_hand"),
                    "observation.state.eef_pos_rel": obs.get(
                        "state.end_effector_position_relative"
                    ),
                    "observation.state.eef_quat_rel": obs.get(
                        "state.end_effector_rotation_relative"
                    ),
                    "observation.state.gripper_qpos": obs.get("state.gripper_qpos"),
                    "observation.state.base_position": obs.get("state.base_position"),
                    "observation.state.base_rotation": obs.get("state.base_rotation"),
                    "task": str(instruction),
                }
            }

    class _FakeGrootActionPipeline:
        def __call__(self, data):
            actions = data[_TransitionKey.ACTION]

            def first_step(value):
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 1:
                    return arr
                if arr.ndim == 2:
                    return arr[0]
                return arr[0, 0]

            return {
                _TransitionKey.ACTION: {
                    "action.end_effector_position": first_step(actions["action.eef_pos"]),
                    "action.end_effector_rotation": first_step(
                        actions["action.eef_axisangle"]
                    ),
                    "action.gripper_close": first_step(actions["action.gripper"]),
                    "action.base_motion": first_step(actions["action.base_motion"]),
                    "action.control_mode": first_step(actions["action.control_mode"]),
                }
            }

    modules["src.processor.factory"].make_robocasa_processors = lambda **_kwargs: (
        None,
        None,
    )
    modules["src.processor.factory"].make_groot_robocasa_processors = (
        lambda **_kwargs: (_FakeGrootObsPipeline(), _FakeGrootActionPipeline())
    )

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "robocasa_eval_under_test", ROBOCASA_EVAL_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class _FakeVLAClient:
    def __init__(self):
        self.reset_count = 0
        self.calls = []

    def reset(self):
        self.reset_count += 1

    def predict(self, images, states, instruction, inference_seed=None):
        self.calls.append(
            {
                "images": dict(images),
                "states": dict(states),
                "instruction": instruction,
                "inference_seed": inference_seed,
            }
        )
        return (
            {
                "action.eef_pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                "action.eef_axisangle": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                "action.gripper": np.array([[0.5]], dtype=np.float32),
                "action.base_motion": np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                "action.control_mode": np.array([[1.0]], dtype=np.float32),
            },
            12.5,
        )


class _FakeGrootEnv:
    def __init__(self):
        self.step_actions = []
        self.reset_seeds = []
        self.ep_meta = {"lang": "open the cabinet", "layout_id": 1, "style_id": 1}
        self.set_ep_meta_calls = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return (
            {
                "video.robot0_agentview_left": np.full((8, 8, 3), 1, dtype=np.uint8),
                "video.robot0_agentview_right": np.full((8, 8, 3), 2, dtype=np.uint8),
                "video.robot0_eye_in_hand": np.full((8, 8, 3), 3, dtype=np.uint8),
                "annotation.human.task_description": np.array(["open the cabinet"]),
                "state.end_effector_position_relative": np.array([0.1, 0.2, 0.3]),
                "state.end_effector_rotation_relative": np.array([0.0, 0.0, 0.0, 1.0]),
                "state.gripper_qpos": np.array([0.4, 0.5]),
                "state.base_position": np.array([0.0, 0.0, 0.0]),
                "state.base_rotation": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            {},
        )

    def step(self, action):
        self.step_actions.append(action)
        return (
            {
                "video.res256_image_side_0": np.stack(
                    [
                        np.full((8, 8, 3), 4, dtype=np.uint8),
                        np.full((8, 8, 3), 5, dtype=np.uint8),
                    ]
                ),
                "video.robot0_agentview_left": np.full((8, 8, 3), 4, dtype=np.uint8),
            },
            0.0,
            False,
            False,
            {"success": True},
        )

    def get_ep_meta(self):
        return self.ep_meta

    def set_ep_meta(self, ep_meta):
        self.set_ep_meta_calls.append(ep_meta)
        self.ep_meta = ep_meta

    def close(self):
        pass


class TestRunVlaRolloutsGroot(unittest.TestCase):
    def test_groot_env_default_server_uses_groot_http_port(self):
        module = _import_robocasa_eval()

        self.assertEqual(module._default_vla_server(use_groot_env=True), "http://localhost:8500")
        self.assertEqual(module._default_vla_server(use_groot_env=False), "http://localhost:8200")

        args = module.build_parser().parse_args(
            ["--task", "OpenDrawer", "--use-groot-env"]
        )
        self.assertIsNone(args.vla_server)
        self.assertEqual(module._default_vla_server(args.use_groot_env), "http://localhost:8500")

    def test_raw_robocasa_obs_maps_to_http_payload_and_native_action(self):
        module = _import_robocasa_eval()
        env = _FakeGrootEnv()
        client = _FakeVLAClient()

        result = module.run_vla_rollouts_groot(
            env,
            client,
            num_rollouts=1,
            num_steps=2,
            seed=100000,
        )

        self.assertEqual(client.reset_count, 1)
        self.assertEqual(env.reset_seeds, [100000])
        self.assertEqual(result["num_success"], 1)
        self.assertEqual(result["rollouts"][0]["seed"], 100000)
        call = client.calls[0]
        self.assertEqual(call["instruction"], "open the cabinet")
        self.assertIsNone(call["inference_seed"])
        self.assertEqual(set(call["images"]), {"left", "right", "wrist"})
        self.assertIn("observation.state.eef_pos_rel", call["states"])
        self.assertIn("observation.state.base_rotation", call["states"])
        np.testing.assert_allclose(
            env.step_actions[0]["action.end_effector_position"],
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            env.step_actions[0]["action.base_motion"],
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            env.step_actions[0]["action.control_mode"],
            np.array([1.0], dtype=np.float32),
        )

    def test_groot_env_eval_passes_seed_to_env_construction_and_reset(self):
        module = _import_robocasa_eval()
        env = _FakeGrootEnv()
        make_calls = []

        def fake_make(*args, **kwargs):
            make_calls.append({"args": args, "kwargs": kwargs})
            return env

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = fake_make
        fake_robocasa = types.ModuleType("robocasa")
        fake_robocasa_utils = types.ModuleType("robocasa.utils")
        fake_gym_utils = types.ModuleType("robocasa.utils.gym_utils")
        fake_gym_utils.GrootRoboCasaEnv = object
        fake_robosuite = types.ModuleType("robosuite")
        client = _FakeVLAClient()

        with mock.patch.dict(
            sys.modules,
            {
                "gymnasium": fake_gym,
                "robocasa": fake_robocasa,
                "robocasa.utils": fake_robocasa_utils,
                "robocasa.utils.gym_utils": fake_gym_utils,
                "robosuite": fake_robosuite,
            },
        ):
            result = module.evaluate_task(
                "CloseFridge",
                client,
                obs_pipeline=None,
                action_pipeline=None,
                num_rollouts=1,
                num_steps=1,
                seed=100000,
                use_groot_env=True,
            )

        self.assertEqual(result["mode"], "vla_groot_env")
        self.assertEqual(make_calls[0]["args"][0], "robocasa_panda_omron/CloseFridge_PandaOmron_Env")
        self.assertEqual(make_calls[0]["kwargs"]["seed"], 100000)
        self.assertEqual(env.reset_seeds, [100000])

    def test_groot_env_rollout_exports_ep_meta_manifest_and_inference_seed(self):
        module = _import_robocasa_eval()
        env = _FakeGrootEnv()
        client = _FakeVLAClient()
        env_name = "robocasa_panda_omron/CloseFridge_PandaOmron_Env"

        with tempfile.TemporaryDirectory() as tmp:
            result = module.run_vla_rollouts_groot(
                env,
                client,
                num_rollouts=1,
                num_steps=2,
                seed=100000,
                env_name=env_name,
                ep_meta_dir=Path(tmp),
                inference_seed=4242,
            )
            manifest = Path(result["rollouts"][0]["ep_meta_manifest"])
            payload = json.loads(manifest.read_text())

        self.assertEqual(result["rollouts"][0]["ep_meta_mode"], "exported")
        self.assertEqual(result["rollouts"][0]["scenario_seed"], 100000)
        self.assertEqual(result["rollouts"][0]["inference_seed"], 4242)
        self.assertEqual(client.calls[0]["inference_seed"], 4242)
        self.assertEqual(payload["format"], "robocasa_ep_meta_manifest.v1")
        self.assertEqual(payload["env_name"], env_name)
        self.assertEqual(payload["scenario_seed"], 100000)
        self.assertEqual(payload["ep_meta"], env.ep_meta)

    def test_groot_env_eval_imports_ep_meta_manifest_before_reset(self):
        module = _import_robocasa_eval()
        env = _FakeGrootEnv()
        env_name = "robocasa_panda_omron/CloseFridge_PandaOmron_Env"
        replay_ep_meta = {"lang": "close the fridge", "layout_id": 7, "style_id": 10}
        make_calls = []

        def fake_make(*args, **kwargs):
            make_calls.append({"args": args, "kwargs": kwargs})
            return env

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = fake_make
        fake_robocasa = types.ModuleType("robocasa")
        fake_robocasa_utils = types.ModuleType("robocasa.utils")
        fake_gym_utils = types.ModuleType("robocasa.utils.gym_utils")
        fake_gym_utils.GrootRoboCasaEnv = object
        fake_robosuite = types.ModuleType("robosuite")

        with tempfile.TemporaryDirectory() as tmp:
            ep_meta_dir = Path(tmp)
            manifest = module._ep_meta_manifest_path(ep_meta_dir, env_name, 100000)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "format": "robocasa_ep_meta_manifest.v1",
                        "env_name": env_name,
                        "scenario_seed": 100000,
                        "ep_meta": replay_ep_meta,
                    }
                )
            )
            with mock.patch.dict(
                sys.modules,
                {
                    "gymnasium": fake_gym,
                    "robocasa": fake_robocasa,
                    "robocasa.utils": fake_robocasa_utils,
                    "robocasa.utils.gym_utils": fake_gym_utils,
                    "robosuite": fake_robosuite,
                },
            ):
                result = module.evaluate_task(
                    "CloseFridge",
                    _FakeVLAClient(),
                    obs_pipeline=None,
                    action_pipeline=None,
                    num_rollouts=1,
                    num_steps=1,
                    seed=100000,
                    use_groot_env=True,
                    ep_meta_dir=ep_meta_dir,
                    inference_seed=5000,
                )

        self.assertEqual(make_calls[0]["kwargs"]["seed"], 100000)
        self.assertEqual(env.reset_seeds, [100000])
        self.assertEqual(env.set_ep_meta_calls, [replay_ep_meta])
        self.assertEqual(result["rollouts"][0]["ep_meta_mode"], "imported")
        self.assertEqual(result["rollouts"][0]["inference_seed"], 5000)

    def test_groot_env_http_video_contract_is_episode_files_and_manifest(self):
        module = _import_robocasa_eval()
        env = _FakeGrootEnv()
        client = _FakeVLAClient()

        class FakeWriter:
            def __init__(self, path, fps):
                self.path = Path(path)
                self.fps = fps
                self.frames = []
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_bytes(b"")

            def append_data(self, frame):
                self.frames.append(frame)

            def close(self):
                pass

        fake_imageio = types.ModuleType("imageio")
        writers = []

        def get_writer(path, fps):
            writer = FakeWriter(path, fps)
            writers.append(writer)
            return writer

        fake_imageio.get_writer = get_writer

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sys.modules,
            {"imageio": fake_imageio},
        ):
            video_dir = Path(tmp) / "videos" / "OpenFridge"
            result = module.run_vla_rollouts_groot(
                env,
                client,
                num_rollouts=2,
                num_steps=2,
                seed=0,
                video_dir=str(video_dir),
            )

            manifest_lines = (video_dir / "per_episode.tsv").read_text().splitlines()

        self.assertEqual(len(writers), 2)
        self.assertEqual(result["num_success"], 2)
        self.assertEqual(len(result["video_paths"]), 2)
        self.assertTrue(all(path.endswith("_s1.mp4") for path in result["video_paths"]))
        self.assertTrue(all(writer.frames[-1].shape == (8, 8, 3) for writer in writers))
        self.assertTrue(all(np.all(writer.frames[-1] == 5) for writer in writers))
        self.assertEqual(
            manifest_lines,
            [
                "episode_idx\tsuccess\tlanguage",
                "0\t1\topen the cabinet",
                "1\t1\topen the cabinet",
            ],
        )


if __name__ == "__main__":
    unittest.main()
