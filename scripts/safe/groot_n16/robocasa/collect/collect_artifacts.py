"""SAFE rollout artifact writing helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np

from collect_schema import GROOT_ACTION_KEYS, SAFE_ACTION_COLUMNS
from src.policies.groot.scenario_replay import json_safe, write_ep_meta_manifest


def write_collect_ep_meta_manifest(
    path: Path,
    *,
    env_name: str,
    scenario_seed: int,
    ep_meta: dict[str, Any],
    robocasa_env_source: str,
) -> None:
    write_ep_meta_manifest(
        path,
        env_name=env_name,
        scenario_seed=scenario_seed,
        ep_meta=ep_meta,
        robocasa_env_source=robocasa_env_source,
        sort_keys=True,
    )


def write_safe_triplet(
    output_dir: Path,
    stem: str,
    policy: Any,
    task_id: int,
    task_description: str,
    episode_idx: int,
    scenario_seed: int | None,
    episode_success: bool,
    env_name: str,
    upstream_video_path: Path | None,
    ep_meta: dict[str, Any],
    n_action_steps: int,
    robocasa_env_source: str,
) -> None:
    if not policy.records:
        raise RuntimeError("No feature records were collected during rollout")
    if policy.exported_action_token_count != n_action_steps:
        raise RuntimeError(
            "SAFE feature export horizon must match executed action steps: "
            f"exported_action_token_count={policy.exported_action_token_count}, "
            f"n_action_steps={n_action_steps}"
        )

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
        "seed": scenario_seed,
        "scenario_seed": scenario_seed,
        "episode_success": int(episode_success),
        "ep_meta": json_safe(ep_meta),
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
        "feature_action_horizon": policy.feature_action_horizon,
        "n_action_steps": n_action_steps,
        "valid_action_horizon": policy.valid_action_horizon,
        "model_action_horizon": policy.model_action_horizon,
        "num_inference_timesteps": policy.num_inference_timesteps,
        "env_name": env_name,
        "robocasa_env_source": robocasa_env_source,
        "video_source": "groot_upstream_video_recording_wrapper",
    }
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    if upstream_video_path is None or not upstream_video_path.exists():
        raise RuntimeError(f"GR00T upstream video was not written: {upstream_video_path}")
    shutil.move(str(upstream_video_path), str(output_dir / f"{stem}.mp4"))
