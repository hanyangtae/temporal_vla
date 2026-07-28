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
import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
COLLECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
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
DEFAULT_MAX_EPISODE_STEPS = 720
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from path_setup import configure_repo_paths, prepend_path  # noqa: E402

configure_repo_paths(
    include_script_utils=True,
    include_groot=True,
    include_robocasa=ROBOCASA_ENV_SOURCE == "robocasa365",
    include_groot_robocasa_v02=ROBOCASA_ENV_SOURCE == "robocasa_v02",
)
prepend_path(COLLECT_ROOT)

from gr00t.eval.rollout_policy import (  # noqa: E402
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
)
from collect_artifacts import (  # noqa: E402
    write_collect_ep_meta_manifest as _write_ep_meta_manifest,
    write_safe_triplet as _write_safe_triplet,
)
from collect_env import (  # noqa: E402
    configure_headless_rendering as _configure_headless_rendering,
    run_single_rollout as _run_single_rollout,
)
from collect_policy_clients import (  # noqa: E402
    HttpN16SafeCollectingPolicyClient,
    N16SafeCollectingPolicyClient,
)
from src.policies.groot.robocasa.io import (  # noqa: E402
    REQUIRED_OBS_KEYS,
    prepare_groot_robocasa_observation as _prepare_observation,
)
from src.policies.groot.robocasa.scenario_replay import (  # noqa: E402
    ep_meta_manifest_path as _ep_meta_manifest_path,
    load_ep_meta_manifest as _load_ep_meta_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GR00T N1.6 RoboCasa SAFE rollouts")
    parser.add_argument("--policy-client-host", required=True)
    parser.add_argument("--policy-client-port", type=int, required=True)
    parser.add_argument(
        "--policy-transport",
        choices=["zmq", "http"],
        default="zmq",
        help="Feature policy transport. zmq keeps the upstream GR00T SAFE endpoint; "
        "http calls /act_with_features through the unified serving API.",
    )
    parser.add_argument("--feature-endpoint", default="get_action_with_features")
    parser.add_argument("--env-name", required=True)
    parser.add_argument(
        "--robocasa-env-source",
        choices=["robocasa_v02", "robocasa365"],
        default=ROBOCASA_ENV_SOURCE,
        help="RoboCasa environment source: robocasa_v02 or robocasa365.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-phases", action="store_true",
                        help="get_action 마다 event 라벨러 step() → per-record feature_phases "
                             "수집 (exp4-3 분리도 지도용; N1.5 http_feature_collect 규약).")
    parser.add_argument("--proximity-phases", action="store_true",
                        help="라벨러 proximity sub-phase 모드 (기본 4-phase). --label-phases 필요.")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--episode-start-idx", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="Number of leading decoded actions to execute per policy inference.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional smoke cap for wrapper max_episode_steps. Leave unset for full rollouts.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help=(
            "Optional per-call policy RNG seed base for HTTP/ZMQ. "
            "Each collected step adds its local index."
        ),
    )
    parser.add_argument(
        "--ep-meta-dir",
        type=Path,
        default=None,
        help="Optional directory for seed-keyed RoboCasa ep_meta JSON import/export.",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--no-overlay", action="store_true",
                        help="영상에 instruction/success 캡션을 안 그림(클린 mp4). "
                             "instruction은 ep_meta.lang 에 그대로 남음.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_action_steps <= 0:
        raise ValueError(f"--n_action_steps must be positive: {args.n_action_steps}")
    if args.video_fps <= 0:
        raise ValueError(f"--video-fps must be positive: {args.video_fps}")
    if args.steps_per_render <= 0:
        raise ValueError(f"--steps-per-render must be positive: {args.steps_per_render}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_episode_steps = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else DEFAULT_MAX_EPISODE_STEPS
    )

    for local_ep_idx in range(args.n_episodes):
        episode_idx = args.episode_start_idx + local_ep_idx
        scenario_seed = None if args.seed is None else args.seed + local_ep_idx
        ep_meta_manifest_path = None
        replay_ep_meta = None
        ep_meta_mode = "none"
        if args.ep_meta_dir is not None and scenario_seed is not None:
            ep_meta_manifest_path = _ep_meta_manifest_path(
                args.ep_meta_dir,
                args.env_name,
                scenario_seed,
            )
            if ep_meta_manifest_path.exists():
                replay_ep_meta = _load_ep_meta_manifest(
                    ep_meta_manifest_path,
                    env_name=args.env_name,
                    scenario_seed=scenario_seed,
                )
                ep_meta_mode = "imported"
            else:
                ep_meta_mode = "exported"
        upstream_video_dir = output_dir / ".groot_video_tmp" / f"task{args.task_id}--ep{episode_idx}"
        if upstream_video_dir.exists():
            shutil.rmtree(upstream_video_dir)
        upstream_video_dir.mkdir(parents=True, exist_ok=True)
        multistep_kwargs = {
            "n_action_steps": args.n_action_steps,
            "max_episode_steps": max_episode_steps,
            "terminate_on_success": True,
        }
        wrapper_configs = WrapperConfigs(
            video=VideoConfig(
                video_dir=str(upstream_video_dir),
                fps=args.video_fps,
                steps_per_render=args.steps_per_render,
                max_episode_steps=max_episode_steps,
                overlay_text=not args.no_overlay,   # --no-overlay → 캡션 없는 클린 영상
            ),
            multistep=MultiStepConfig(**multistep_kwargs),
        )

        if args.policy_transport == "http":
            policy = HttpN16SafeCollectingPolicyClient(
                args.policy_client_host,
                args.policy_client_port,
                inference_seed=args.inference_seed,
            )
        else:
            policy = N16SafeCollectingPolicyClient(
                args.policy_client_host,
                args.policy_client_port,
                endpoint=args.feature_endpoint,
                inference_seed=args.inference_seed,
            )
        results = _run_single_rollout(
            env_name=args.env_name,
            policy=policy,
            wrapper_configs=wrapper_configs,
            scenario_seed=scenario_seed,
            replay_ep_meta=replay_ep_meta,
            label_phases=args.label_phases,
            proximity_phases=args.proximity_phases,
        )
        feature_phases = results[5] if len(results) > 5 else []
        if ep_meta_manifest_path is not None and replay_ep_meta is None:
            _write_ep_meta_manifest(
                ep_meta_manifest_path,
                env_name=args.env_name,
                scenario_seed=scenario_seed,
                ep_meta=results[4],
                robocasa_env_source=ROBOCASA_ENV_SOURCE,
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
            scenario_seed=scenario_seed,
            episode_success=success,
            env_name=args.env_name,
            upstream_video_path=results[3],
            ep_meta=results[4],
            n_action_steps=args.n_action_steps,
            robocasa_env_source=ROBOCASA_ENV_SOURCE,
            max_episode_steps=max_episode_steps,
            video_fps=args.video_fps,
            steps_per_render=args.steps_per_render,
            inference_seed=args.inference_seed,
            model_family="groot_n16",
            policy_transport=args.policy_transport,
            task_suite_name="groot_n16_robocasa",
            extra_metadata=({"feature_phases": feature_phases} if feature_phases else None),
        )
        shutil.rmtree(upstream_video_dir, ignore_errors=True)
        print(
            f"wrote {stem}: steps={len(policy.records)} success={int(success)} "
            f"scenario_seed={scenario_seed} feature_kind={policy.feature_kind} "
            f"n_action_steps={args.n_action_steps} ep_meta={ep_meta_mode} "
            f"video_source=groot_upstream"
        )


if __name__ == "__main__":
    main()
