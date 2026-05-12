#!/usr/bin/env python3
"""RoboCasa eval client for a remote GR00T ZMQ policy server.

The fine-tuned RoboCasa checkpoint uses the training modality keys
``robot0_agentview_left/right`` and ``robot0_eye_in_hand``.  The upstream
GrootRoboCasaEnv emits the GR00T sim aliases ``res256_image_side_0/1`` and
``res256_image_wrist_0``.  This client keeps upstream rollout logic but aliases
those keys before sending observations to the ZMQ policy server.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/policies/Isaac-GR00T"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/benchmarks/robocasa"))

import numpy as np

from gr00t.eval.rollout_policy import (  # noqa: E402
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    run_rollout_gymnasium_policy,
)
from gr00t.policy.server_client import PolicyClient  # noqa: E402


OBS_ALIASES = {
    "video.res256_image_side_0": "video.robot0_agentview_left",
    "video.res256_image_side_1": "video.robot0_agentview_right",
    "video.res256_image_wrist_0": "video.robot0_eye_in_hand",
    "annotation.human.action.task_description": "annotation.human.task_description",
}


class AliasedPolicyClient:
    def __init__(self, host: str, port: int):
        self.client = PolicyClient(host=host, port=port)

    def reset(self, options=None):
        return self.client.reset(options=options)

    def get_action(self, observation, options=None):
        aliased = dict(observation)
        for src, dst in OBS_ALIASES.items():
            if src in aliased and dst not in aliased:
                aliased[dst] = aliased[src]
        return self.client.get_action(aliased, options=options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T RoboCasa ZMQ evaluation with key aliases")
    parser.add_argument("--policy-client-host", required=True)
    parser.add_argument("--policy-client-port", type=int, required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--video-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir
    if video_dir is None:
        env_tag = args.env_name.replace("/", "_")
        video_dir = f"/tmp/sim_eval_videos_{env_tag}_ac{args.n_action_steps}_{uuid.uuid4()}"
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=args.max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
    )
    policy = AliasedPolicyClient(args.policy_client_host, args.policy_client_port)
    results = run_rollout_gymnasium_policy(
        env_name=args.env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
    )
    print("Video saved to: ", video_dir)
    print("results: ", results)
    print("success rate: ", np.mean(results[1]) if results[1] else 0.0)


if __name__ == "__main__":
    main()
