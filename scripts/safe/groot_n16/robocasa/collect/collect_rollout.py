#!/usr/bin/env python3
"""Collect GR00T N1.6 RoboCasa rollouts for SAFE.

The pkl stores one raw SAFE feature tensor per environment step. For GR00T N1.6
this is the flow-matching action-token feature before velocity projection, with
shape ``[K, H, D]``: denoising step, action horizon, feature dimension.
By default H is the embodiment's valid decoded action horizon, not the
checkpoint's model-level max action horizon.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import zmq


REPO_ROOT = Path(__file__).resolve().parents[5]
GROOT_ROOT = REPO_ROOT / "src/policies/Isaac-GR00T"
ROBOCASA365_ROOT = REPO_ROOT / "src/benchmarks/robocasa"
ROBOCASA365_ROBOSUITE_ROOT = REPO_ROOT / "src/benchmarks/robosuite"
ROBOCASA_V02_ROOT = GROOT_ROOT / "external_dependencies/robocasa"
ROBOCASA_ENV_SOURCE_ALIASES = {
    "robocasa_v02": "robocasa_v02",
    "robocasa365": "robocasa365",
}


def _normalize_robocasa_env_source(source: str) -> str:
    try:
        return ROBOCASA_ENV_SOURCE_ALIASES[source]
    except KeyError:
        raise ValueError(f"Unknown RoboCasa env source: {source!r}") from None


def _select_robocasa_env_source() -> str:
    source = os.environ.get("ROBOCASA_ENV_SOURCE")
    for idx, arg in enumerate(sys.argv):
        if arg == "--robocasa-env-source" and idx + 1 < len(sys.argv):
            source = sys.argv[idx + 1]
        elif arg.startswith("--robocasa-env-source="):
            source = arg.split("=", 1)[1]
    return _normalize_robocasa_env_source(source or "robocasa_v02")


ROBOCASA_ENV_SOURCE = _select_robocasa_env_source()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GROOT_ROOT))
if ROBOCASA_ENV_SOURCE == "robocasa365":
    sys.path.insert(0, str(ROBOCASA365_ROBOSUITE_ROOT))
    sys.path.insert(0, str(ROBOCASA365_ROOT))
else:
    sys.path.insert(0, str(ROBOCASA_V02_ROOT))

from gr00t.eval.rollout_policy import (  # noqa: E402
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    get_gym_env,
)
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper  # noqa: E402
from gr00t.eval.sim.wrapper.video_recording_wrapper import (  # noqa: E402
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.policy.server_client import MsgSerializer  # noqa: E402


OBS_ALIASES = {
    "video.robot0_agentview_left": "video.res256_image_side_0",
    "video.robot0_agentview_right": "video.res256_image_side_1",
    "video.robot0_eye_in_hand": "video.res256_image_wrist_0",
    "video.res256_image_side_0": "video.robot0_agentview_left",
    "video.res256_image_side_1": "video.robot0_agentview_right",
    "video.res256_image_wrist_0": "video.robot0_eye_in_hand",
    "annotation.human.task_description": "annotation.human.action.task_description",
    "annotation.human.action.task_description": "annotation.human.task_description",
}
REQUIRED_OBS_KEYS = {
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "annotation.human.action.task_description",
    "annotation.human.task_description",
}
SAFE_ACTION_COLUMNS = [
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
]
PRIMARY_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
]
GROOT_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]
VIDEO_RECORDING_KEYS = {
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
}


def _first_text(value: Any) -> str | None:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return _first_text(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _first_text(value[0])
    if value is None:
        return None
    return str(value)


def _without_action_prefix(key: str) -> str:
    return key[len("action.") :] if key.startswith("action.") else key


def _first_env_first_step(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0].reshape(-1)
    return arr[0, 0].reshape(-1)


def _extract_safe_action_vector(action: dict[str, Any]) -> np.ndarray:
    pieces = []
    by_normalized_key = {_without_action_prefix(k): v for k, v in action.items()}
    for key in PRIMARY_ACTION_KEYS:
        value = action.get(key)
        if value is None:
            value = by_normalized_key.get(_without_action_prefix(key))
        if value is not None:
            pieces.append(_first_env_first_step(value))

    if not pieces:
        for key in sorted(action):
            pieces.append(_first_env_first_step(action[key]))

    vector = np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[: min(7, vector.shape[0])] = vector[:7]
    return out


def _extract_groot_action_vector(action: dict[str, Any]) -> np.ndarray:
    pieces = []
    by_normalized_key = {_without_action_prefix(k): v for k, v in action.items()}
    for key in GROOT_ACTION_KEYS:
        value = action.get(key)
        if value is None:
            value = by_normalized_key.get(_without_action_prefix(key))
        if value is not None:
            pieces.append(_first_env_first_step(value))
    if not pieces:
        return _extract_safe_action_vector(action)
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def _to_pickleable_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_pickleable_numpy(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_pickleable_numpy(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_to_pickleable_numpy(val) for val in value)
    return value


def _prepare_observation(observation: dict[str, Any]) -> dict[str, Any]:
    aliased = dict(observation)
    for src, dst in OBS_ALIASES.items():
        if src in aliased and dst not in aliased:
            aliased[dst] = aliased[src]
    filtered = {key: aliased[key] for key in REQUIRED_OBS_KEYS if key in aliased}
    missing = sorted(REQUIRED_OBS_KEYS - filtered.keys())
    if missing:
        raise KeyError(f"Missing N1.6 RoboCasa observation keys: {missing}")
    return filtered


class N16SafeCollectingPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        endpoint: str = "get_action_with_features",
        timeout_ms: int = 120000,
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")
        self.endpoint = endpoint
        self.records: list[dict[str, Any]] = []
        self.task_description: str | None = None
        self.feature_kind: str | None = None
        self.feature_axes: list[str] | None = None
        self.feature_slice: str | None = None
        self.exported_action_token_count: int | None = None
        self.valid_action_horizon: int | None = None
        self.model_action_horizon: int | None = None
        self.num_inference_timesteps: int | None = None

    def call_endpoint(
        self, endpoint: str, data: dict[str, Any] | None = None, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        self.socket.send(MsgSerializer.to_bytes(request))
        response = MsgSerializer.from_bytes(self.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"N1.6 server error: {response['error']}")
        return response

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.records.clear()
        self.task_description = None
        self.feature_kind = None
        self.feature_axes = None
        self.feature_slice = None
        self.exported_action_token_count = None
        self.valid_action_horizon = None
        self.model_action_horizon = None
        self.num_inference_timesteps = None
        return {}

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        filtered = _prepare_observation(observation)
        if self.task_description is None:
            self.task_description = _first_text(
                filtered.get("annotation.human.action.task_description")
            )

        response = self.call_endpoint(self.endpoint, {"observation": filtered})
        action = response["action"]
        hidden_states = np.asarray(response["hidden_states"])
        if hidden_states.ndim == 4 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states[0]

        record: dict[str, Any] = {
            "hidden_state": torch.from_numpy(np.ascontiguousarray(hidden_states)),
            "action_vector": _extract_safe_action_vector(action),
            "groot_action_vector": _extract_groot_action_vector(action),
            "action": _to_pickleable_numpy(action),
        }
        self.feature_kind = response.get("feature_kind", self.feature_kind)
        self.feature_axes = response.get("feature_axes", self.feature_axes)
        self.feature_slice = response.get("feature_slice", self.feature_slice)
        self.exported_action_token_count = response.get(
            "exported_action_token_count", self.exported_action_token_count
        )
        self.valid_action_horizon = response.get("valid_action_horizon", self.valid_action_horizon)
        self.model_action_horizon = response.get("model_action_horizon", self.model_action_horizon)
        self.num_inference_timesteps = response.get(
            "num_inference_timesteps", self.num_inference_timesteps
        )
        self.records.append(record)
        return action, {}


def _coerce_success(value: Any) -> bool:
    if isinstance(value, list):
        return bool(np.any(value))
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, (bool, int, np.bool_, np.integer)):
        return bool(value)
    raise ValueError(f"Unknown success dtype: {type(value)}")


def _find_latest_upstream_video(wrapper_configs: WrapperConfigs) -> Path | None:
    video_dir = wrapper_configs.video.video_dir
    if video_dir is None:
        return None
    videos = sorted(Path(video_dir).glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    return videos[-1] if videos else None


class SafeVideoObservationFilter(gym.ObservationWrapper):
    """Keep the upstream recorder on the three canonical RoboCasa camera views."""

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


def _create_safe_eval_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    wrapper_configs: WrapperConfigs,
) -> gym.Env:
    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        env = SafeVideoObservationFilter(env)
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
            overlay_text=wrapper_configs.video.overlay_text,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def _run_single_rollout(
    env_name: str,
    policy: N16SafeCollectingPolicyClient,
    wrapper_configs: WrapperConfigs,
    seed: int | None,
) -> tuple[str, list[bool], dict[str, list[Any]], Path | None]:
    env = gym.vector.SyncVectorEnv(
        [
            lambda: _create_safe_eval_env(
                env_name=env_name,
                env_idx=0,
                total_n_envs=1,
                wrapper_configs=wrapper_configs,
            )
        ]
    )

    current_success = False
    current_length = 0
    episode_infos: dict[str, list[Any]] = defaultdict(list)
    upstream_video_path: Path | None = None

    try:
        observations, _ = env.reset(seed=seed)
        policy.reset()

        pbar = tqdm(total=1, desc="Episodes")
        while True:
            actions, _ = policy.get_action(observations)
            next_obs, _rewards, terminations, truncations, env_infos = env.step(actions)
            current_length += 1

            if "success" in env_infos:
                current_success |= _coerce_success(env_infos["success"][0])

            if "final_info" in env_infos and env_infos["final_info"][0] is not None:
                final_info = env_infos["final_info"][0]
                if "success" in final_info:
                    current_success |= _coerce_success(final_info["success"])

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
                upstream_video_path = _find_latest_upstream_video(wrapper_configs)
                if upstream_video_path is None and rendered_video is not None:
                    upstream_video_path = Path(rendered_video)
                break

            observations = next_obs

    finally:
        env.close()

    return env_name, [current_success], dict(episode_infos), upstream_video_path


def _write_safe_triplet(
    output_dir: Path,
    stem: str,
    policy: N16SafeCollectingPolicyClient,
    task_id: int,
    task_description: str,
    episode_idx: int,
    seed: int | None,
    episode_success: bool,
    env_name: str,
    upstream_video_path: Path | None,
) -> None:
    if not policy.records:
        raise RuntimeError("No feature records were collected during rollout")

    for old_path in output_dir.glob(f"task{task_id}--ep{episode_idx}--succ*.*"):
        old_path.unlink()

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAFE_ACTION_COLUMNS)
        writer.writeheader()
        for record in policy.records:
            values = record["action_vector"]
            writer.writerow({col: float(values[i]) for i, col in enumerate(SAFE_ACTION_COLUMNS)})

    pkl_path = output_dir / f"{stem}.pkl"
    payload = {
        "task_suite_name": "groot_n16_robocasa",
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "seed": seed,
        "episode_success": int(episode_success),
        "hidden_states": [record["hidden_state"] for record in policy.records],
        "actions": [record["action"] for record in policy.records],
        "action_vectors": np.stack(
            [record["groot_action_vector"] for record in policy.records], axis=0
        ),
        "action_keys": GROOT_ACTION_KEYS,
        "feature_kind": policy.feature_kind,
        "feature_axes": policy.feature_axes,
        "feature_slice": policy.feature_slice,
        "exported_action_token_count": policy.exported_action_token_count,
        "valid_action_horizon": policy.valid_action_horizon,
        "model_action_horizon": policy.model_action_horizon,
        "num_inference_timesteps": policy.num_inference_timesteps,
        "env_name": env_name,
        "robocasa_env_source": ROBOCASA_ENV_SOURCE,
        "video_source": "groot_upstream_video_recording_wrapper",
    }
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    if upstream_video_path is None or not upstream_video_path.exists():
        raise RuntimeError(f"GR00T upstream video was not written: {upstream_video_path}")
    shutil.move(str(upstream_video_path), str(output_dir / f"{stem}.mp4"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GR00T N1.6 RoboCasa SAFE rollouts")
    parser.add_argument("--policy-client-host", required=True)
    parser.add_argument("--policy-client-port", type=int, required=True)
    parser.add_argument("--feature-endpoint", default="get_action_with_features")
    parser.add_argument("--env-name", required=True)
    parser.add_argument(
        "--robocasa-env-source",
        choices=["robocasa_v02", "robocasa365"],
        default=ROBOCASA_ENV_SOURCE,
        help="RoboCasa environment source: robocasa_v02 or robocasa365.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--episode-start-idx", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for local_ep_idx in range(args.n_episodes):
        episode_idx = args.episode_start_idx + local_ep_idx
        episode_seed = None if args.seed is None else args.seed + local_ep_idx
        upstream_video_dir = output_dir / ".groot_video_tmp" / f"task{args.task_id}--ep{episode_idx}"
        if upstream_video_dir.exists():
            shutil.rmtree(upstream_video_dir)
        upstream_video_dir.mkdir(parents=True, exist_ok=True)
        wrapper_configs = WrapperConfigs(
            video=VideoConfig(video_dir=str(upstream_video_dir)),
            multistep=MultiStepConfig(
                terminate_on_success=True,
            ),
        )

        policy = N16SafeCollectingPolicyClient(
            args.policy_client_host,
            args.policy_client_port,
            endpoint=args.feature_endpoint,
        )
        results = _run_single_rollout(
            env_name=args.env_name,
            policy=policy,
            wrapper_configs=wrapper_configs,
            seed=episode_seed,
        )
        success = bool(results[1][0]) if results[1] else False
        task_description = args.task_description or policy.task_description or args.env_name
        stem = f"task{args.task_id}--ep{episode_idx}--succ{int(success)}"
        _write_safe_triplet(
            output_dir=output_dir,
            stem=stem,
            policy=policy,
            task_id=args.task_id,
            task_description=task_description,
            episode_idx=episode_idx,
            seed=episode_seed,
            episode_success=success,
            env_name=args.env_name,
            upstream_video_path=results[3],
        )
        shutil.rmtree(upstream_video_dir, ignore_errors=True)
        print(
            f"wrote {stem}: steps={len(policy.records)} success={int(success)} "
            f"seed={episode_seed} feature_kind={policy.feature_kind} "
            f"video_source=groot_upstream"
        )


if __name__ == "__main__":
    main()
