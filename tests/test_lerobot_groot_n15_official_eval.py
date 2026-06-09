from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "safe" / "groot_n15" / "robocasa" / "eval" / "lerobot_http_eval.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("lerobot_groot_n15_lerobot_http_eval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_official_obs_maps_to_lerobot_http_inputs():
    module = _load_module()
    obs = {
        "video.robot0_agentview_left": np.full((4, 4, 3), 1, dtype=np.uint8),
        "video.robot0_agentview_right": np.full((4, 4, 3), 2, dtype=np.uint8),
        "video.robot0_eye_in_hand": np.full((4, 4, 3), 3, dtype=np.uint8),
        "annotation.human.task_description": "Open the fridge.",
        "state.end_effector_position_relative": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "state.end_effector_rotation_relative": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "state.gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
        "state.base_position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "state.base_rotation": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }

    images, states, instruction = module.official_obs_to_lerobot_inputs(obs)

    assert set(images) == {"side_0", "side_1", "wrist_0"}
    assert instruction == "Open the fridge."
    assert set(states) == {
        "observation.state.eef_pos_rel",
        "observation.state.eef_quat_rel",
        "observation.state.gripper_qpos",
        "observation.state.base_position",
        "observation.state.base_rotation",
    }
    np.testing.assert_allclose(states["observation.state.eef_pos_rel"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(states["observation.state.base_rotation"], [0.0, 0.0, 1.0, 0.0])


def test_lerobot_action_maps_to_official_robocasa_action():
    module = _load_module()
    action = {
        "action.eef_pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "action.eef_axisangle": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "action.gripper": np.array([[0.5]], dtype=np.float32),
        "action.base_motion": np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "action.control_mode": np.array([[1.0]], dtype=np.float32),
    }

    mapped = module.lerobot_action_to_official_action(action)

    assert set(mapped) == {
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    }
    np.testing.assert_allclose(mapped["action.end_effector_position"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mapped["action.control_mode"], [1.0])


def test_step_success_uses_sparse_reward_when_raw_robocasa_info_has_no_success():
    module = _load_module()

    assert module.step_success(1.0, {}) is True
    assert module.step_success(0.0, {}) is False
    assert module.step_success(0.0, {"success": np.array([True])}) is True


def test_step_success_can_fallback_to_env_private_checker():
    module = _load_module()

    class RawEnv:
        def _check_success(self):
            return True

    class WrappedEnv:
        unwrapped = RawEnv()

    assert module.step_success(0.0, {}, env=WrappedEnv()) is True
