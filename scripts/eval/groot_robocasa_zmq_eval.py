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
import inspect
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
    def __init__(self, host: str, port: int, inference_seed_base: int | None = None):
        self.client = PolicyClient(host=host, port=port)
        # inference_seed_base: get_action 호출마다 inference_seed = base + call_idx 자동 주입.
        # None 이면 비결정 (server flow-matching noise 매번 다름).
        self.inference_seed_base = inference_seed_base
        self._call_idx = 0

    def reset(self, options=None):
        self._call_idx = 0
        return self.client.reset(options=options)

    def get_action(self, observation, options=None):
        aliased = dict(observation)
        for src, dst in OBS_ALIASES.items():
            if src in aliased and dst not in aliased:
                aliased[dst] = aliased[src]
        opts = dict(options) if options else {}
        if self.inference_seed_base is not None and "inference_seed" not in opts:
            opts["inference_seed"] = int(self.inference_seed_base) + self._call_idx
        self._call_idx += 1
        return self.client.get_action(aliased, options=opts or None)


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
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="고정 seed (env_idx 별 [seed, seed+1, ...]). 같은 (env, seed) 는 같은 episode 시리즈.",
    )
    parser.add_argument(
        "--per-episode-csv",
        type=str,
        default=None,
        help="per-episode success+language tsv 출력 경로. 미지정 시 video-dir/per_episode.tsv.",
    )
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
    policy = AliasedPolicyClient(
        args.policy_client_host,
        args.policy_client_port,
        inference_seed_base=args.eval_seed,
    )
    rollout_kwargs = {
        "env_name": args.env_name,
        "policy": policy,
        "wrapper_configs": wrapper_configs,
        "n_episodes": args.n_episodes,
        "n_envs": args.n_envs,
    }
    if "eval_seed" in inspect.signature(run_rollout_gymnasium_policy).parameters:
        rollout_kwargs["eval_seed"] = args.eval_seed
    results = run_rollout_gymnasium_policy(**rollout_kwargs)
    env_name, successes, infos = results
    print("Video saved to: ", video_dir)
    print("results: ", results)
    print("success rate: ", np.mean(successes) if successes else 0.0)

    # per-episode tsv: episode_idx, success, language (instruction)
    per_ep_csv = args.per_episode_csv or str(Path(video_dir) / "per_episode.tsv")
    Path(per_ep_csv).parent.mkdir(parents=True, exist_ok=True)
    langs = infos.get("language", [""] * len(successes))
    with open(per_ep_csv, "w") as f:
        f.write("episode_idx\tsuccess\tlanguage\n")
        for idx, (s, lang) in enumerate(zip(successes, langs)):
            f.write(f"{idx}\t{int(bool(s))}\t{str(lang)}\n")
    print(f"per-episode tsv saved to: {per_ep_csv}")


if __name__ == "__main__":
    main()
