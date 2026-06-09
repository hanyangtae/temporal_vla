#!/usr/bin/env python3
"""RoboCasa benchmark-style eval client for a GR00T N1.5 ZMQ server.

This mirrors ``robocasa-benchmark/Isaac-GR00T``'s ``scripts/run_eval.py`` client
path: ``gym.make("robocasa/<Task>", split=...)`` plus N1.5
``SimulationInferenceClient`` protocol. Keep this separate from
``groot_n15_robocasa_zmq_eval.py``, which uses this repo's N1.6 rollout helper.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import time
from typing import Any

import gymnasium as gym
import msgpack
import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
N15_ROOT = REPO_ROOT / "src" / "policies" / "Isaac-GR00T-N1.5"
for path in (
    N15_ROOT,
    REPO_ROOT / "src" / "benchmarks" / "robocasa",
    REPO_ROOT / "src" / "benchmarks" / "robosuite",
    REPO_ROOT,
):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

import robocasa  # noqa: E402,F401
import robosuite  # noqa: E402,F401
from robocasa.utils.dataset_registry_utils import get_task_horizon  # noqa: E402

from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper  # noqa: E402
from gr00t.eval.wrappers.video_recording_wrapper import (  # noqa: E402
    VideoRecorder,
    VideoRecordingWrapper,
)


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


class OfficialRoboCasaClient:
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

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        return self.call_endpoint("get_action", observations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--task", required=True, help="RoboCasa task name, e.g. OpenFridge")
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--steps-per-render", type=int, default=2)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> gym.Env:
    env = gym.make(f"robocasa/{args.task}", split=args.split, enable_render=True)
    if args.video_dir is not None:
        recorder = VideoRecorder.create_h264(fps=args.video_fps)
        env = VideoRecordingWrapper(
            env,
            recorder,
            video_dir=args.video_dir,
            steps_per_render=args.steps_per_render,
        )
    env = MultiStepWrapper(
        env,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps or get_task_horizon(args.task),
    )
    return env


def run() -> tuple[str, list[bool]]:
    args = parse_args()
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    client = OfficialRoboCasaClient(host=args.host, port=args.port)
    env = gym.vector.SyncVectorEnv([lambda: make_env(args)])

    completed = 0
    successes: list[bool] = []
    current_success = False
    start = time.time()
    obs, _info = env.reset()
    while completed < args.n_episodes:
        action_dict = client.get_action(obs)
        actions = action_dict.get("actions", action_dict)
        obs, _rewards, terminations, truncations, infos = env.step(actions)
        current_success |= bool(np.any(infos["success"][0]))
        if terminations[0] or truncations[0]:
            successes.append(current_success)
            print(
                f"EP {len(successes)} success: {current_success}; "
                f"Cumulative success rate: {float(np.mean(successes))}"
            )
            current_success = False
            completed += 1

    env.close()
    env_name = f"robocasa/{args.task}"
    print(f"Collecting {args.n_episodes} episodes took {time.time() - start:.2f} seconds")
    print("Video saved to: ", args.video_dir)
    print("results: ", (env_name, successes))
    print("success rate: ", float(np.mean(successes)) if successes else 0.0)
    return env_name, successes


if __name__ == "__main__":
    run()
