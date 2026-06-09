#!/usr/bin/env python3
"""RoboCasa rollout client for a GR00T N1.5 ZMQ policy server.

N1.5's ``scripts/inference_service.py`` exposes a simpler ZMQ protocol than
the N1.6 ``PolicyServer`` used by ``groot_robocasa_zmq_eval.py``.  This client
keeps the existing RoboCasa rollout stack, but sends observations directly to
the N1.5 ``get_action`` endpoint and returns ``(action, info)`` for the N1.6
rollout helper API.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import io
import inspect
from pathlib import Path
import sys
import uuid
from typing import Any

import msgpack
import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src/policies/Isaac-GR00T"))
sys.path.insert(0, str(REPO_ROOT / "src/benchmarks/robocasa"))
sys.path.insert(0, str(REPO_ROOT / "src/benchmarks/robosuite"))

from gr00t.eval.rollout_policy import (  # noqa: E402
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    run_rollout_gymnasium_policy,
)


OBS_ALIASES = {
    "video.res256_image_side_0": "video.robot0_agentview_left",
    "video.res256_image_side_1": "video.robot0_agentview_right",
    "video.res256_image_wrist_0": "video.robot0_eye_in_hand",
    "annotation.human.action.task_description": "annotation.human.task_description",
}
REQUIRED_OBS_KEYS = {
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "annotation.human.task_description",
}


def supported_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(target):
        supported = {field.name for field in fields(target)}
    else:
        supported = set(inspect.signature(target).parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


class N15MsgSerializer:
    @staticmethod
    def to_bytes(data: dict[str, Any]) -> bytes:
        return msgpack.packb(data, default=N15MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=N15MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if "__ModalityConfig_class__" in obj:
            return obj.get("as_json", obj)
        return obj

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if isinstance(obj, np.generic):
            return obj.item()
        return obj


class N15PolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 120000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def call_endpoint(
        self, endpoint: str, data: dict[str, Any] | None = None, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        self.socket.send(N15MsgSerializer.to_bytes(request))
        response = N15MsgSerializer.from_bytes(self.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"N1.5 server error: {response['error']}")
        return response

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        aliased = dict(observation)
        for src, dst in OBS_ALIASES.items():
            if src in aliased and dst not in aliased:
                aliased[dst] = aliased[src]
        filtered = {key: aliased[key] for key in REQUIRED_OBS_KEYS if key in aliased}
        missing = sorted(REQUIRED_OBS_KEYS - filtered.keys())
        if missing:
            raise KeyError(f"Missing N1.5 PandaOmron observation keys: {missing}")
        return self.call_endpoint("get_action", filtered), {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T N1.5 RoboCasa ZMQ rollout")
    parser.add_argument("--policy-client-host", required=True)
    parser.add_argument("--policy-client-port", type=int, required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video-file-prefix", type=str, default=None)
    parser.add_argument("--no-video-outcome-suffix", action="store_true")
    parser.add_argument(
        "--one-episode-per-env",
        action="store_true",
        help="Count only the first episode from each vector env. Use this for exact seed pairs.",
    )
    parser.add_argument("--video-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir
    if video_dir is None:
        env_tag = args.env_name.replace("/", "_")
        video_dir = f"/tmp/sim_eval_videos_n15_{env_tag}_{uuid.uuid4()}"
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    video = VideoConfig(
        **supported_kwargs(
            VideoConfig,
            {
                "video_dir": video_dir,
                "max_episode_steps": args.max_episode_steps,
                "fps": args.video_fps,
                "steps_per_render": args.steps_per_render,
                "filename_prefix": args.video_file_prefix,
                "append_outcome_to_filename": not args.no_video_outcome_suffix,
                "record_first_episode_only": args.one_episode_per_env,
            },
        )
    )
    multistep = MultiStepConfig(
        **supported_kwargs(
            MultiStepConfig,
            {
                "n_action_steps": args.n_action_steps,
                "max_episode_steps": args.max_episode_steps,
                "terminate_on_success": True,
            },
        )
    )
    wrapper_configs = WrapperConfigs(
        **supported_kwargs(
            WrapperConfigs,
            {
                "video": video,
                "multistep": multistep,
                "seed": args.seed,
                "one_episode_per_env": args.one_episode_per_env,
            },
        )
    )
    policy = N15PolicyClient(args.policy_client_host, args.policy_client_port)
    rollout_kwargs = supported_kwargs(
        run_rollout_gymnasium_policy,
        {
            "env_name": args.env_name,
            "policy": policy,
            "wrapper_configs": wrapper_configs,
            "n_episodes": args.n_episodes,
            "n_envs": args.n_envs,
            "eval_seed": args.seed,
            "one_episode_per_env": args.one_episode_per_env,
        },
    )
    results = run_rollout_gymnasium_policy(**rollout_kwargs)
    print("Video saved to: ", video_dir)
    print("results: ", results)
    print("success rate: ", np.mean(results[1]) if results[1] else 0.0)


if __name__ == "__main__":
    main()
