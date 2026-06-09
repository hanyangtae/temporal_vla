"""RoboCasa env construction and one-episode rollout execution."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.vector.utils import concatenate
import numpy as np
from tqdm import tqdm

from gr00t.eval.rollout_policy import WrapperConfigs

from src.policies.groot.robocasa_env_wrappers import wrap_groot_robocasa_eval_env
from src.policies.groot.scenario_replay import (
    get_robocasa_ep_meta,
    json_safe,
    set_robocasa_ep_meta,
)


def configure_headless_rendering() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class NoAutoResetSyncVectorEnv(gym.vector.SyncVectorEnv):
    """SyncVectorEnv variant that returns terminal observations without resetting."""

    def step_wait(self):
        observations = []
        infos = {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space,
            observations,
            self.observations,
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )


def coerce_success(value: Any) -> bool:
    if isinstance(value, list):
        return bool(np.any(value))
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, (bool, int, np.bool_, np.integer)):
        return bool(value)
    raise ValueError(f"Unknown success dtype: {type(value)}")


def find_latest_upstream_video(wrapper_configs: WrapperConfigs) -> Path | None:
    video_dir = wrapper_configs.video.video_dir
    if video_dir is None:
        return None
    videos = sorted(Path(video_dir).glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    return videos[-1] if videos else None


def make_robocasa_env(env_name: str, scenario_seed: int | None) -> gym.Env:
    configure_headless_rendering()

    import robocasa  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401

    return gym.make(env_name, enable_render=True, seed=scenario_seed)


def create_safe_eval_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    wrapper_configs: WrapperConfigs,
    scenario_seed: int | None,
) -> gym.Env:
    del env_idx, total_n_envs
    env = make_robocasa_env(env_name, scenario_seed=scenario_seed)
    return wrap_groot_robocasa_eval_env(env, wrapper_configs)


def run_single_rollout(
    env_name: str,
    policy: Any,
    wrapper_configs: WrapperConfigs,
    scenario_seed: int | None,
    replay_ep_meta: dict[str, Any] | None,
) -> tuple[str, list[bool], dict[str, list[Any]], Path | None, dict[str, Any]]:
    env = NoAutoResetSyncVectorEnv(
        [
            lambda: create_safe_eval_env(
                env_name=env_name,
                env_idx=0,
                total_n_envs=1,
                wrapper_configs=wrapper_configs,
                scenario_seed=scenario_seed,
            )
        ]
    )

    current_success = False
    current_length = 0
    episode_infos: dict[str, list[Any]] = defaultdict(list)
    upstream_video_path: Path | None = None
    ep_meta: dict[str, Any] | None = None

    try:
        if replay_ep_meta is not None:
            set_robocasa_ep_meta(env.envs[0], replay_ep_meta)
        observations, _ = env.reset(seed=scenario_seed)
        captured_ep_meta = get_robocasa_ep_meta(env.envs[0])
        ep_meta = json_safe(replay_ep_meta) if replay_ep_meta is not None else captured_ep_meta
        policy.reset()

        pbar = tqdm(total=1, desc="Episodes")
        while True:
            actions, _ = policy.get_action(observations)
            next_obs, _rewards, terminations, truncations, env_infos = env.step(actions)
            current_length += 1

            if "success" in env_infos:
                current_success |= coerce_success(env_infos["success"][0])

            if "final_info" in env_infos and env_infos["final_info"][0] is not None:
                final_info = env_infos["final_info"][0]
                if "success" in final_info:
                    current_success |= coerce_success(final_info["success"])

            done = bool(terminations[0] or truncations[0])
            if done:
                if "final_info" in env_infos and env_infos["final_info"][0] is not None:
                    final_info = env_infos["final_info"][0]
                    if "task_progress" in final_info:
                        episode_infos["task_progress"].append(final_info["task_progress"])
                    if "q_score" in final_info:
                        episode_infos["q_score"].append(final_info["q_score"])
                    if "valid" in final_info:
                        episode_infos["valid"].append(final_info["valid"])
                episode_infos["length"].append(current_length)
                pbar.update(1)
                pbar.close()
                rendered_video = env.envs[0].render()
                upstream_video_path = find_latest_upstream_video(wrapper_configs)
                if upstream_video_path is None and rendered_video is not None:
                    upstream_video_path = Path(rendered_video)
                break

            observations = next_obs

    finally:
        env.close()

    if ep_meta is None:
        raise RuntimeError("RoboCasa ep_meta was not captured during rollout reset")
    return env_name, [current_success], dict(episode_infos), upstream_video_path, ep_meta
