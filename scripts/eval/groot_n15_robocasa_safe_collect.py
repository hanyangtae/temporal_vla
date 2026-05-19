#!/usr/bin/env python3
"""Collect GR00T N1.5 RoboCasa rollouts for SAFE.

The output directory receives one ``task<ID>--ep<ID>--succ<0|1>`` triplet per
episode:

* ``.csv``: 7D action columns for backwards compatibility with SAFE's
  ``openvla`` loader
* ``.pkl``: GR00T metadata, token features, attention masks, native action dicts,
  and GR00T-concatenated action vectors
* ``.mp4``: matching rollout video, copied from the RoboCasa video wrapper
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any
import uuid

import msgpack
import numpy as np
import torch
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
        raise KeyError(f"Missing N1.5 PandaOmron observation keys: {missing}")
    return filtered


class N15SafeCollectingPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        endpoint: str = "get_action_with_features",
        feature_pool: str = "none",
        timeout_ms: int = 120000,
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")
        self.endpoint = endpoint
        self.feature_pool = feature_pool
        self.records: list[dict[str, Any]] = []
        self.task_description: str | None = None
        self.feature_kind: str | None = None

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
        self.records.clear()
        self.task_description = None
        self.feature_kind = None
        return {}

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        filtered = _prepare_observation(observation)
        if self.task_description is None:
            self.task_description = _first_text(filtered.get("annotation.human.task_description"))

        response = self.call_endpoint(self.endpoint, filtered)
        action = response["action"]
        hidden_states = np.asarray(response["hidden_states"])
        if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states[0]

        attention_mask = None
        if "attention_mask" in response:
            attention_mask = np.asarray(response["attention_mask"])
            if attention_mask.ndim == 2 and attention_mask.shape[0] == 1:
                attention_mask = attention_mask[0]

        hidden_states = _pool_hidden_states(hidden_states, attention_mask, self.feature_pool)

        record: dict[str, Any] = {
            "hidden_state": torch.from_numpy(np.ascontiguousarray(hidden_states)),
            "action_vector": _extract_safe_action_vector(action),
            "groot_action_vector": _extract_groot_action_vector(action),
            "action": _to_pickleable_numpy(action),
        }
        if attention_mask is not None:
            record["attention_mask"] = attention_mask

        self.feature_kind = response.get("feature_kind", self.feature_kind)
        self.records.append(record)
        return action, {}


def _find_new_mp4(video_dir: Path, before: set[Path]) -> Path | None:
    after = set(video_dir.glob("*.mp4"))
    created = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if created:
        return created[0]
    all_mp4 = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
    return all_mp4[0] if all_mp4 else None


def _pool_hidden_states(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray | None,
    feature_pool: str,
) -> np.ndarray:
    if feature_pool == "none" or hidden_states.ndim == 1:
        return hidden_states
    if hidden_states.ndim != 2:
        raise ValueError(f"Expected unbatched hidden states with rank 2, got {hidden_states.shape}")
    if feature_pool == "mean":
        return hidden_states.mean(axis=0).astype(np.float32, copy=False)
    if feature_pool == "masked_mean":
        if attention_mask is None:
            return hidden_states.mean(axis=0).astype(np.float32, copy=False)
        mask = attention_mask.astype(np.float32, copy=False)
        denom = max(float(mask.sum()), 1.0)
        return ((hidden_states.astype(np.float32) * mask[:, None]).sum(axis=0) / denom).astype(
            np.float32, copy=False
        )
    if feature_pool == "last_valid":
        if attention_mask is None:
            return hidden_states[-1].astype(np.float32, copy=False)
        idx = max(int(attention_mask.astype(np.int64).sum()) - 1, 0)
        return hidden_states[idx].astype(np.float32, copy=False)
    raise ValueError(f"Unknown feature pool: {feature_pool}")


def _write_safe_triplet(
    output_dir: Path,
    stem: str,
    policy: N15SafeCollectingPolicyClient,
    task_id: int,
    task_description: str,
    episode_idx: int,
    episode_success: bool,
    env_name: str,
    source_video: Path | None,
) -> None:
    if not policy.records:
        raise RuntimeError("No feature records were collected during rollout")

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAFE_ACTION_COLUMNS)
        writer.writeheader()
        for record in policy.records:
            values = record["action_vector"]
            writer.writerow({col: float(values[i]) for i, col in enumerate(SAFE_ACTION_COLUMNS)})

    pkl_path = output_dir / f"{stem}.pkl"
    payload = {
        "task_suite_name": "groot_n15_robocasa",
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "episode_success": int(episode_success),
        "hidden_states": [record["hidden_state"] for record in policy.records],
        "actions": [record["action"] for record in policy.records],
        "action_vectors": np.stack(
            [record["groot_action_vector"] for record in policy.records], axis=0
        ),
        "action_keys": GROOT_ACTION_KEYS,
        "attention_masks": [
            record["attention_mask"] for record in policy.records if "attention_mask" in record
        ],
        "feature_kind": policy.feature_kind,
        "env_name": env_name,
    }
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    if source_video is not None and source_video.exists():
        target_video = output_dir / f"{stem}.mp4"
        if source_video.resolve() == target_video.resolve():
            return
        if target_video.exists():
            target_video.unlink()
        shutil.move(str(source_video), str(target_video))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GR00T N1.5 RoboCasa SAFE rollouts")
    parser.add_argument("--policy-client-host", required=True)
    parser.add_argument("--policy-client-port", type=int, required=True)
    parser.add_argument("--feature-endpoint", default="get_action_with_features")
    parser.add_argument(
        "--feature-pool",
        choices=("none", "mean", "masked_mean", "last_valid"),
        default="none",
        help="Pool token features before saving. Use masked_mean for compact SAFE training data.",
    )
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--episode-start-idx", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video-file-prefix", type=str, default=None)
    parser.add_argument("--no-video-outcome-suffix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for local_ep_idx in range(args.n_episodes):
        episode_idx = args.episode_start_idx + local_ep_idx
        episode_seed = None if args.seed is None else args.seed + local_ep_idx
        video_prefix = args.video_file_prefix or f"safe_task{args.task_id}_ep{episode_idx}_{uuid.uuid4()}"
        before_mp4 = set(output_dir.glob("*.mp4"))

        wrapper_configs = WrapperConfigs(
            video=VideoConfig(
                video_dir=str(output_dir),
                max_episode_steps=args.max_episode_steps,
                fps=args.video_fps,
                steps_per_render=args.steps_per_render,
                filename_prefix=video_prefix,
                append_outcome_to_filename=not args.no_video_outcome_suffix,
                record_first_episode_only=True,
            ),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps,
                max_episode_steps=args.max_episode_steps,
                terminate_on_success=True,
            ),
            seed=episode_seed,
            one_episode_per_env=True,
        )

        policy = N15SafeCollectingPolicyClient(
            args.policy_client_host,
            args.policy_client_port,
            endpoint=args.feature_endpoint,
            feature_pool=args.feature_pool,
        )
        results = run_rollout_gymnasium_policy(
            env_name=args.env_name,
            policy=policy,
            wrapper_configs=wrapper_configs,
            n_episodes=1,
            n_envs=1,
        )
        success = bool(results[1][0]) if results[1] else False
        task_description = args.task_description or policy.task_description or args.env_name
        stem = f"task{args.task_id}--ep{episode_idx}--succ{int(success)}"
        source_video = _find_new_mp4(output_dir, before_mp4)
        _write_safe_triplet(
            output_dir=output_dir,
            stem=stem,
            policy=policy,
            task_id=args.task_id,
            task_description=task_description,
            episode_idx=episode_idx,
            episode_success=success,
            env_name=args.env_name,
            source_video=source_video,
        )
        print(
            f"wrote {stem}: steps={len(policy.records)} success={int(success)} "
            f"video={source_video}"
        )


if __name__ == "__main__":
    main()
