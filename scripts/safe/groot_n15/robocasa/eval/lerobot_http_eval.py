#!/usr/bin/env python
"""RoboCasa benchmark-style eval client for LeRobot GR00T N1.5 HTTP serve.

This keeps the upstream RoboCasa task convention used by
``robocasa-benchmark/Isaac-GR00T``: ``gym.make("robocasa/<Task>", split=...)``.
The policy transport is this repo's unified LeRobot HTTP ``/act`` API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import uuid
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
for path in (
    REPO_ROOT / "scripts" / "utils",
    REPO_ROOT / "src" / "policies" / "Isaac-GR00T-N1.5",
    REPO_ROOT / "src" / "benchmarks" / "robocasa",
    REPO_ROOT / "src" / "benchmarks" / "robosuite",
    REPO_ROOT,
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from vla_client import VLAClient  # noqa: E402


OFFICIAL_IMAGE_TO_LEROBOT = {
    "video.robot0_agentview_left": "side_0",
    "video.robot0_agentview_right": "side_1",
    "video.robot0_eye_in_hand": "wrist_0",
}
OFFICIAL_STATE_TO_LEROBOT = {
    "state.end_effector_position_relative": "observation.state.eef_pos_rel",
    "state.end_effector_rotation_relative": "observation.state.eef_quat_rel",
    "state.gripper_qpos": "observation.state.gripper_qpos",
    "state.base_position": "observation.state.base_position",
    "state.base_rotation": "observation.state.base_rotation",
}
LEROBOT_ACTION_TO_OFFICIAL = {
    "action.eef_pos": "action.end_effector_position",
    "action.eef_axisangle": "action.end_effector_rotation",
    "action.gripper": "action.gripper_close",
    "action.base_motion": "action.base_motion",
    "action.control_mode": "action.control_mode",
}


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return str(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return "" if not value else str(value[0])
    return str(value)


def _first_step(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0]
    return arr[0, 0]


def official_obs_to_lerobot_inputs(
    obs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    images: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}

    for src, dst in OFFICIAL_IMAGE_TO_LEROBOT.items():
        if src not in obs:
            raise KeyError(f"Missing official RoboCasa image key: {src}")
        images[dst] = np.asarray(obs[src], dtype=np.uint8)

    for src, dst in OFFICIAL_STATE_TO_LEROBOT.items():
        if src not in obs:
            raise KeyError(f"Missing official RoboCasa state key: {src}")
        states[dst] = np.asarray(obs[src], dtype=np.float32)

    instruction = _scalar_text(
        obs.get(
            "annotation.human.task_description",
            obs.get("annotation.human.action.task_description", ""),
        )
    )
    return images, states, instruction


def lerobot_action_to_official_action(action: dict[str, Any]) -> dict[str, np.ndarray]:
    missing = sorted(set(LEROBOT_ACTION_TO_OFFICIAL) - set(action))
    if missing:
        raise KeyError(f"Missing LeRobot action keys: {missing}")
    return {
        dst: _first_step(action[src])
        for src, dst in LEROBOT_ACTION_TO_OFFICIAL.items()
    }


def env_check_success(env: Any) -> bool:
    raw_env = getattr(env, "unwrapped", env)
    check_success = getattr(raw_env, "_check_success", None)
    if not callable(check_success):
        return False
    return bool(check_success())


def step_success(reward: Any, info: dict[str, Any], env: Any | None = None) -> bool:
    if "success" in info and bool(np.any(info["success"])):
        return True

    # RoboCasa's raw Gym wrapper forwards robosuite info unchanged. For these
    # sparse-reward tasks, reward is float(env._check_success()).
    if bool(np.any(np.asarray(reward, dtype=np.float32) > 0.5)):
        return True

    return env is not None and env_check_success(env)


def fixture_open_state(env: Any) -> dict[str, float]:
    raw_env = getattr(env, "unwrapped", env)
    fixture = getattr(raw_env, "fxtr", None)
    if fixture is None or not hasattr(fixture, "get_joint_state"):
        return {}

    joint_names = getattr(fixture, "_fridge_door_joint_names", None)
    if joint_names is None:
        joint_names = getattr(fixture, "door_joint_names", None)
    if joint_names is None:
        return {}

    joint_state = fixture.get_joint_state(raw_env, joint_names)
    return {name: float(value) for name, value in joint_state.items()}


def _save_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    if not frames:
        return
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-server", default="http://127.0.0.1:8400")
    parser.add_argument("--task", default="OpenFridge")
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--wait-ready", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument(
        "--success-debug-every",
        type=int,
        default=0,
        help="If >0, print sparse reward and fixture open joint state every N env steps.",
    )
    return parser.parse_args()


def make_env(task: str, split: str):
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robosuite  # noqa: F401

    return gym.make(f"robocasa/{task}", split=split, enable_render=True)


def task_horizon(task: str) -> int:
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    return int(get_task_horizon(task))


def run() -> dict[str, Any]:
    args = parse_args()
    client = VLAClient(args.vla_server, timeout=args.timeout)
    if args.wait_ready:
        client.wait_until_ready(max_wait=args.timeout)

    max_steps = args.max_episode_steps or task_horizon(args.task)
    successes: list[bool] = []
    episode_lengths: list[int] = []
    first_success_steps: list[int | None] = []
    max_rewards: list[float] = []
    max_fixture_open: list[float | None] = []
    video_paths: list[str] = []
    start = time.time()

    env = make_env(args.task, args.split)
    try:
        for episode_i in range(args.n_episodes):
            client.reset()
            reset_seed = None if args.seed is None else args.seed + episode_i
            obs, _info = env.reset(seed=reset_seed)
            frames: list[np.ndarray] = []
            success = False
            first_success_step = None
            episode_max_reward = float("-inf")
            episode_max_open = float("-inf")
            step_i = 0

            while step_i < max_steps:
                if args.video_dir is not None and step_i % args.steps_per_render == 0:
                    frames.append(env.render())
                images, states, instruction = official_obs_to_lerobot_inputs(obs)
                inference_seed = None
                if args.inference_seed is not None:
                    inference_seed = args.inference_seed + episode_i * max_steps + step_i
                action, _latency_ms = client.predict(
                    images,
                    states,
                    instruction,
                    inference_seed=inference_seed,
                )
                if not isinstance(action, dict):
                    raise RuntimeError("LeRobot GR00T N1.5 server must return sub-keyed actions")
                official_action = lerobot_action_to_official_action(action)
                obs, reward, terminated, truncated, info = env.step(official_action)
                step_i += 1
                reward_value = float(np.max(np.asarray(reward, dtype=np.float32)))
                episode_max_reward = max(episode_max_reward, reward_value)
                open_state = fixture_open_state(env)
                if open_state:
                    episode_max_open = max(episode_max_open, max(open_state.values()))
                success_now = step_success(reward, info, env=env)
                if success_now and first_success_step is None:
                    first_success_step = step_i
                success = success or success_now
                if args.success_debug_every > 0 and (
                    step_i == 1 or step_i % args.success_debug_every == 0 or success_now
                ):
                    open_text = (
                        ", ".join(f"{name}={value:.3f}" for name, value in open_state.items())
                        if open_state
                        else "n/a"
                    )
                    print(
                        f"[success-debug] episode={episode_i + 1} step={step_i} "
                        f"reward={reward_value:.1f} success={success_now} open={open_text} "
                        f"info_keys={sorted(info.keys())}"
                    )
                if terminated or truncated or success:
                    break

            if args.video_dir is not None:
                frames.append(env.render())
                suffix = f"success{int(success)}"
                video_path = args.video_dir / f"{uuid.uuid4()}_{suffix}.mp4"
                _save_video(frames, video_path, args.video_fps)
                video_paths.append(str(video_path))

            successes.append(success)
            episode_lengths.append(step_i)
            first_success_steps.append(first_success_step)
            max_rewards.append(episode_max_reward if episode_max_reward > float("-inf") else 0.0)
            max_fixture_open.append(
                episode_max_open if episode_max_open > float("-inf") else None
            )
            cumulative = float(np.mean(successes))
            print(f"EP {episode_i + 1} success: {success}; Cumulative success rate: {cumulative}")
    finally:
        env.close()

    result = {
        "env_name": f"robocasa/{args.task}",
        "split": args.split,
        "num_episodes": len(successes),
        "successes": successes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "episode_lengths": episode_lengths,
        "first_success_steps": first_success_steps,
        "max_rewards": max_rewards,
        "max_fixture_open": max_fixture_open,
        "video_paths": video_paths,
        "elapsed_sec": time.time() - start,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
