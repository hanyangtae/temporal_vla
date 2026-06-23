#!/usr/bin/env python3
"""Collect GR00T N1.5 LeRobot HTTP RoboCasa features as SAFE triplets.

The observation bridge follows the benchmark-style N1.5 LeRobot HTTP eval
client. The env/video/action-chunk loop and output contract follow the N1.6
SAFE collector:
``task{id}--ep{idx}--succ{0|1}.{pkl,csv,mp4}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
N15_EVAL_ROOT = REPO_ROOT / "scripts" / "safe" / "groot_n15" / "robocasa" / "eval"
N16_COLLECT_ROOT = REPO_ROOT / "scripts" / "safe" / "groot_n16" / "robocasa" / "collect"
N15_GROOT_ROOT = REPO_ROOT / "src" / "policies" / "Isaac-GR00T-N1.5"
N16_GROOT_ROOT = REPO_ROOT / "src" / "policies" / "Isaac-GR00T"
DEFAULT_MAX_EPISODE_STEPS = 720


def _prepend_path(path: Path) -> None:
    path_str = str(path)
    if not path.exists():
        return
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


for path in reversed(
    (
        N16_GROOT_ROOT,
        N15_GROOT_ROOT,
        N16_COLLECT_ROOT,
        N15_EVAL_ROOT,
        REPO_ROOT / "scripts" / "utils",
        REPO_ROOT / "src" / "benchmarks" / "robocasa",
        REPO_ROOT / "src" / "benchmarks" / "robosuite",
        REPO_ROOT,
    )
):
    _prepend_path(path)

from collect_artifacts import (  # noqa: E402
    write_collect_ep_meta_manifest,
    write_safe_triplet,
)
from collect_schema import (  # noqa: E402
    _extract_groot_action_vector,
    _extract_safe_action_vector,
    _to_pickleable_numpy,
)
from lerobot_http_eval import (  # noqa: E402
    official_obs_to_lerobot_inputs,
    step_success,
)
from src.policies.groot.robocasa.io import convert_http_actions_to_groot_chunk  # noqa: E402
from src.policies.groot.robocasa.scenario_replay import (  # noqa: E402
    ep_meta_manifest_path,
    get_robocasa_ep_meta,
    load_ep_meta_manifest,
    set_robocasa_ep_meta,
)
from src.policies.safe_metadata import normalize_feature_metadata  # noqa: E402
from vla_client import VLAClient  # noqa: E402


def _prefer_present(new: Any, old: Any) -> Any:
    return old if new is None else new


def _lerobot_action_to_official_chunk(action: dict[str, Any]) -> dict[str, np.ndarray]:
    chunk = convert_http_actions_to_groot_chunk(action)
    unbatched = {}
    for key, value in chunk.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        unbatched[key] = arr
    return unbatched


class N15LerobotHttpFeatureClient(VLAClient):
    """VLAClient-backed LeRobot GR00T N1.5 policy client with SAFE records."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 300.0,
        inference_seed: int | None = None,
    ):
        super().__init__(url, timeout=timeout)
        self.inference_seed = inference_seed
        self.records: list[dict[str, Any]] = []
        self.task_description: str | None = None
        self.feature_kind: str | None = None
        self.feature_axes: list[str] | None = None
        self.feature_slice: str | None = None
        self.exported_action_token_count: int | None = None
        self.feature_action_horizon: int | None = None
        self.valid_action_horizon: int | None = None
        self.model_action_horizon: int | None = None
        self.num_inference_timesteps: int | None = None
        self.capture_layers: list[int] | None = None
        self.layer_count: int | None = None
        self.token_count: int | None = None
        self.vl_feature_kind: str | None = None
        self.vl_feature_axes: list[str] | None = None
        self.vl_feature_dim: int | None = None

    def reset(self) -> None:
        self.records.clear()
        self.task_description = None
        self.feature_kind = None
        self.feature_axes = None
        self.feature_slice = None
        self.exported_action_token_count = None
        self.feature_action_horizon = None
        self.valid_action_horizon = None
        self.model_action_horizon = None
        self.num_inference_timesteps = None
        self.capture_layers = None
        self.layer_count = None
        self.token_count = None
        self.vl_feature_kind = None
        self.vl_feature_axes = None
        self.vl_feature_dim = None
        super().reset()

    def _update_metadata(self, features: dict[str, Any]) -> None:
        metadata = normalize_feature_metadata(features)
        self.feature_kind = _prefer_present(metadata.feature_kind, self.feature_kind)
        self.feature_axes = _prefer_present(metadata.feature_axes, self.feature_axes)
        self.feature_slice = _prefer_present(metadata.feature_slice, self.feature_slice)
        self.exported_action_token_count = _prefer_present(
            metadata.exported_action_token_count, self.exported_action_token_count
        )
        self.feature_action_horizon = _prefer_present(
            metadata.feature_action_horizon, self.feature_action_horizon
        )
        self.valid_action_horizon = _prefer_present(
            metadata.valid_action_horizon, self.valid_action_horizon
        )
        self.model_action_horizon = _prefer_present(
            metadata.model_action_horizon, self.model_action_horizon
        )
        self.num_inference_timesteps = _prefer_present(
            metadata.num_inference_timesteps, self.num_inference_timesteps
        )
        capture_layers = features.get("capture_layers", features.get("layer_indices"))
        self.capture_layers = _prefer_present(
            None if capture_layers is None else [int(layer) for layer in capture_layers],
            self.capture_layers,
        )
        self.layer_count = _prefer_present(features.get("layer_count"), self.layer_count)
        self.token_count = _prefer_present(features.get("token_count"), self.token_count)
        self.vl_feature_kind = _prefer_present(
            features.get("vl_feature_kind"), self.vl_feature_kind
        )
        self.vl_feature_axes = _prefer_present(
            features.get("vl_feature_axes"), self.vl_feature_axes
        )
        self.vl_feature_dim = _prefer_present(
            features.get("vl_feature_dim"), self.vl_feature_dim
        )

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        images, states, instruction = official_obs_to_lerobot_inputs(observation)
        if self.task_description is None:
            self.task_description = instruction
        call_inference_seed = (
            None if self.inference_seed is None else self.inference_seed + len(self.records)
        )
        actions, features, _latency_ms = self.predict_with_features(
            images,
            states,
            instruction,
            inference_seed=call_inference_seed,
        )
        if not isinstance(actions, dict):
            raise RuntimeError("LeRobot GR00T N1.5 /act_with_features must return sub-keyed actions")

        official_action = _lerobot_action_to_official_chunk(actions)
        if features is None:
            return official_action, {}

        hidden_states = np.asarray(features["hidden_states"])
        if hidden_states.ndim == 4 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states[0]
        record = {
            "hidden_state": torch.from_numpy(np.ascontiguousarray(hidden_states)),
            "action_vector": _extract_safe_action_vector(official_action),
            "groot_action_vector": _extract_groot_action_vector(official_action),
            "action": _to_pickleable_numpy(official_action),
            # Proprio state fed to this inference (paper expert-vs-VL state probe target).
            "state": _to_pickleable_numpy(states),
        }
        vl_hidden_states = features.get("vl_hidden_states")
        if vl_hidden_states is not None:
            record["vl_hidden_state"] = torch.from_numpy(
                np.ascontiguousarray(np.asarray(vl_hidden_states))
            )
        self._update_metadata(features)
        self.records.append(record)
        return official_action, {}


def _find_latest_video(video_dir: Path) -> Path | None:
    videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    return videos[-1] if videos else None


def _append_summary(summary_path: Path, task_id: int, task: str, episode_idx: int, seed: int | None, pkl_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        summary_path.write_text("task_id\ttask\tepisode_idx\tseed\texit_code\tpkl\n")
    with summary_path.open("a") as f:
        f.write(
            f"{task_id}\t{task}\t{episode_idx}\t"
            f"{'' if seed is None else seed}\t0\t{pkl_path}\n"
        )


def _resolve_task_dir(root_or_task_dir: Path, task: str) -> Path:
    """Return the task-specific directory used by the N1.6 collection layout."""

    return root_or_task_dir if root_or_task_dir.name == task else root_or_task_dir / task


def _resolve_cell_dir(root_or_cell_dir: Path, task: str, cell_id: str) -> Path:
    if root_or_cell_dir.name == cell_id and root_or_cell_dir.parent.name == task:
        return root_or_cell_dir
    if root_or_cell_dir.name == task:
        return root_or_cell_dir / cell_id
    return root_or_cell_dir / task / cell_id


def _normalize_instruction(value: str) -> str:
    return " ".join(value.strip().split())


def make_env(
    task: str,
    split: str,
    *,
    env_name: str,
    scenario_seed: int | None = None,
    video_dir: Path | None = None,
    video_fps: int = 20,
    steps_per_render: int = 2,
    overlay_text: bool = True,
    n_action_steps: int = 16,
    max_episode_steps: int = 720,
):
    del task, split
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401
    from gr00t.eval.rollout_policy import MultiStepConfig, VideoConfig, WrapperConfigs
    from src.policies.groot.robocasa.env_wrappers import wrap_groot_robocasa_eval_env

    env = gym.make(env_name, enable_render=True, seed=scenario_seed)
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=None if video_dir is None else str(video_dir),
            fps=video_fps,
            steps_per_render=steps_per_render,
            max_episode_steps=max_episode_steps,
            overlay_text=overlay_text,   # --no-overlay → 캡션 없는 클린 영상
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        )
    )
    return wrap_groot_robocasa_eval_env(env, wrapper_configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-server", default="http://127.0.0.1:8400")
    parser.add_argument("--task", default="OpenFridge")
    parser.add_argument(
        "--env-name",
        required=True,
        help=(
            "Explicit RoboCasa env id. Use the same env id as N1.6 collection, "
            "e.g. robocasa_panda_omron/CloseFridge_PandaOmron_Env."
        ),
    )
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--cell-id", default=None)
    parser.add_argument("--cell-index", type=int, default=None)
    parser.add_argument("--canonical-instruction", default=None)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--episode-start-idx", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument(
        "--n_action_steps",
        "--n-action-steps",
        dest="n_action_steps",
        type=int,
        default=16,
        help="Number of leading decoded actions to execute per policy inference.",
    )
    parser.add_argument(
        "--ep-meta-dir",
        type=Path,
        default=None,
        help="Optional directory for seed-keyed RoboCasa ep_meta JSON import/export.",
    )
    parser.add_argument(
        "--ep-meta-load-env-name",
        default=None,
        help=(
            "Optional env name used only when loading ep_meta manifests. This lets "
            "N1.5 replay manifests exported by the N1.6 env id."
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--wait-ready", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--no-overlay", action="store_true",
                        help="영상에 instruction/success 캡션을 안 그림(클린 mp4). "
                             "instruction은 ep_meta.lang 에 그대로 남음.")
    return parser.parse_args()


def run() -> dict[str, Any]:
    args = parse_args()
    if args.n_action_steps <= 0:
        raise ValueError(f"--n_action_steps must be positive: {args.n_action_steps}")
    if args.max_episode_steps <= 0:
        raise ValueError(f"--max-episode-steps must be positive: {args.max_episode_steps}")
    output_root = Path(args.output_dir)
    cell_id = getattr(args, "cell_id", None)
    cell_index = getattr(args, "cell_index", None)
    canonical_instruction = getattr(args, "canonical_instruction", None)
    if cell_id is not None and cell_index is None:
        raise ValueError("--cell-index is required when --cell-id is set")
    if cell_id is not None and canonical_instruction is None:
        raise ValueError("--canonical-instruction is required when --cell-id is set")
    effective_task_id = args.task_id if cell_index is None else cell_index

    output_dir = (
        _resolve_task_dir(output_root, args.task)
        if cell_id is None
        else _resolve_cell_dir(output_root, args.task, cell_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_root / "collection_summary.tsv"
        if cell_id is not None
        else output_dir.parent / "collection_summary.tsv"
    )
    ep_meta_dir = (
        None
        if args.ep_meta_dir is None
        else (
            _resolve_task_dir(Path(args.ep_meta_dir), args.task)
            if cell_id is None
            else _resolve_cell_dir(Path(args.ep_meta_dir), args.task, cell_id)
        )
    )
    if args.ep_meta_load_env_name is not None and ep_meta_dir is None:
        raise ValueError("--ep-meta-dir is required when --ep-meta-load-env-name is set")

    policy = N15LerobotHttpFeatureClient(
        args.vla_server,
        timeout=args.timeout,
        inference_seed=args.inference_seed,
    )
    if args.wait_ready:
        policy.wait_until_ready(max_wait=args.timeout)

    max_steps = args.max_episode_steps
    results: list[dict[str, Any]] = []
    env_name = args.env_name
    for local_ep_idx in range(args.n_episodes):
        episode_idx = args.episode_start_idx + local_ep_idx
        scenario_seed = None if args.seed is None else args.seed + local_ep_idx
        ep_meta_path = None
        replay_ep_meta = None
        if ep_meta_dir is not None and scenario_seed is not None:
            replay_requested = args.ep_meta_load_env_name is not None
            load_env_name = args.ep_meta_load_env_name if replay_requested else env_name
            ep_meta_path = ep_meta_manifest_path(ep_meta_dir, load_env_name, scenario_seed)
            if replay_requested and ep_meta_path.exists():
                replay_ep_meta = load_ep_meta_manifest(
                    ep_meta_path,
                    env_name=load_env_name,
                    scenario_seed=scenario_seed,
                )
            elif replay_requested:
                raise FileNotFoundError(
                    "--ep-meta-load-env-name requested but ep_meta manifest is missing: "
                    f"{ep_meta_path}"
                )

        upstream_video_dir = output_dir / ".groot_video_tmp" / f"task{effective_task_id}--ep{episode_idx}"
        if upstream_video_dir.exists():
            shutil.rmtree(upstream_video_dir)
        upstream_video_dir.mkdir(parents=True, exist_ok=True)
        env = make_env(
            args.task,
            args.split,
            env_name=args.env_name,
            scenario_seed=scenario_seed,
            video_dir=upstream_video_dir,
            video_fps=args.video_fps,
            steps_per_render=args.steps_per_render,
            overlay_text=not args.no_overlay,   # --no-overlay → 캡션 없는 클린 영상
            n_action_steps=args.n_action_steps,
            max_episode_steps=max_steps,
        )
        try:
            if replay_ep_meta is not None:
                set_robocasa_ep_meta(env, replay_ep_meta)
            obs, _info = env.reset(seed=scenario_seed)
            captured_ep_meta = replay_ep_meta or get_robocasa_ep_meta(env)
            if (
                ep_meta_dir is not None
                and scenario_seed is not None
                and replay_ep_meta is None
            ):
                write_collect_ep_meta_manifest(
                    ep_meta_manifest_path(ep_meta_dir, env_name, scenario_seed),
                    env_name=env_name,
                    scenario_seed=scenario_seed,
                    ep_meta=captured_ep_meta,
                    robocasa_env_source="robocasa365",
                )
            policy.reset()
            success = False
            first_success_step = None
            step_i = 0
            while step_i < max_steps:
                official_action, _ = policy.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(official_action)
                step_i += 1
                success_now = step_success(reward, info, env=env)
                if success_now and first_success_step is None:
                    first_success_step = step_i
                success = success or success_now
                if terminated or truncated or success:
                    break

            rendered_video = env.render()
            upstream_video_path = _find_latest_video(upstream_video_dir)
            if upstream_video_path is None and rendered_video is not None:
                upstream_video_path = Path(rendered_video)
            stem = f"task{effective_task_id}--ep{episode_idx}--succ{int(success)}"
            task_description = args.task_description or policy.task_description or args.task
            if canonical_instruction is not None and (
                _normalize_instruction(task_description)
                != _normalize_instruction(canonical_instruction)
            ):
                raise RuntimeError(
                    "Collected task description does not match canonical instruction: "
                    f"{task_description!r} != {canonical_instruction!r}"
                )
            extra_metadata = None
            if cell_id is not None:
                extra_metadata = {
                    "cell_id": cell_id,
                    "cell_index": effective_task_id,
                    "robocasa_task": args.task,
                    "canonical_instruction": canonical_instruction,
                }
            write_safe_triplet(
                output_dir=output_dir,
                stem=stem,
                policy=policy,
                task_id=effective_task_id,
                task_description=task_description,
                episode_idx=episode_idx,
                scenario_seed=scenario_seed,
                episode_success=success,
                env_name=env_name,
                upstream_video_path=upstream_video_path,
                ep_meta=captured_ep_meta,
                n_action_steps=args.n_action_steps,
                robocasa_env_source="robocasa365",
                max_episode_steps=max_steps,
                video_fps=args.video_fps,
                steps_per_render=args.steps_per_render,
                inference_seed=args.inference_seed,
                model_family="lerobot_groot_n15",
                policy_transport="http",
                task_suite_name="lerobot_groot_n15_robocasa",
                extra_metadata=extra_metadata,
            )
            pkl_path = output_dir / f"{stem}.pkl"
            _append_summary(summary_path, effective_task_id, args.task, episode_idx, scenario_seed, pkl_path)
            results.append(
                {
                    "episode_idx": episode_idx,
                    "success": success,
                    "first_success_step": first_success_step,
                    "steps": step_i,
                    "records": len(policy.records),
                    "pkl": str(pkl_path),
                }
            )
            print(
                f"wrote {stem}: steps={step_i} success={int(success)} "
                f"records={len(policy.records)} feature_kind={policy.feature_kind}"
            )
        finally:
            env.close()

    return {"episodes": results}


if __name__ == "__main__":
    run()
