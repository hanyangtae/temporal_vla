"""Shared GR00T RoboCasa eval wrappers.

These helpers keep project-local RoboCasa eval paths aligned with the upstream
GR00T rollout stack while enforcing the canonical 3-view video recording
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

from gr00t.eval.rollout_policy import WrapperConfigs
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.eval.sim.wrapper.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)

from .io import VIDEO_RECORDING_KEYS


class CanonicalRoboCasaVideoObservationFilter(gym.ObservationWrapper):
    """Keep rollout recording on the three canonical RoboCasa res256 views."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict(
                {
                    key: space
                    for key, space in env.observation_space.spaces.items()
                    if not key.startswith("video.") or key in VIDEO_RECORDING_KEYS
                }
            )

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in observation.items()
            if not key.startswith("video.") or key in VIDEO_RECORDING_KEYS
        }


def wrap_groot_robocasa_eval_env(
    env: gym.Env,
    wrapper_configs: WrapperConfigs,
) -> gym.Env:
    """Apply GR00T RoboCasa recording and multistep wrappers.

    The upstream ``GrootRoboCasaEnv`` exposes both ``res256`` and ``res512``
    camera observations. The policy payload is filtered separately before ZMQ
    send; this wrapper filters only the video recorder observation path so
    saved rollout videos stay 3-view instead of 6-view.
    """

    if wrapper_configs.video.video_dir is not None:
        env = CanonicalRoboCasaVideoObservationFilter(env)
        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            # choke point 강제 OFF: upstream wrapper 의 overlay 는 프레임 하단 장면
            # 위에 caption 을 구워 복구 불가(2026-08 kanu 219판 사고). 호출측 설정과
            # 무관하게 항상 클린 영상 — instruction 은 ep_meta/사이드카에만 남긴다.
            overlay_text=False,
        )

    return MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
