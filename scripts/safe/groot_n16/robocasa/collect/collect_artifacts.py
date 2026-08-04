"""SAFE rollout artifact writing helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np

# policy 계약(SafeFeaturePolicy/POLICY_*_ATTRS)은 collect_schema 가 단일 출처다.
from collect_schema import (
    GROOT_ACTION_KEYS,
    POLICY_REQUIRED_ATTRS,
    SAFE_ACTION_COLUMNS,
    SafeFeaturePolicy,
)
from src.policies.groot.robocasa.scenario_replay import json_safe, write_ep_meta_manifest


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


def _uses_action_token_horizon(policy: Any) -> bool:
    axes = list(getattr(policy, "feature_axes", None) or [])
    kind = getattr(policy, "feature_kind", None)
    if bool(kind) and str(kind).endswith("_multilayer"):
        return False
    if axes[:1] == ["layer"]:
        return False
    return True


def write_safe_triplet(
    output_dir: Path,
    stem: str,
    policy: SafeFeaturePolicy,
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
    max_episode_steps: int | None = None,
    video_fps: int | None = None,
    steps_per_render: int | None = None,
    inference_seed: int | None = None,
    model_family: str = "groot_n16",
    policy_transport: str = "zmq",
    task_suite_name: str = "groot_n16_robocasa",
    video_source: str = "groot_upstream_video_recording_wrapper",
    extra_metadata: dict[str, Any] | None = None,
    include_hidden_states: bool = True,
) -> None:
    missing = [a for a in POLICY_REQUIRED_ATTRS if not hasattr(policy, a)]
    if missing:
        raise RuntimeError(
            f"policy({type(policy).__name__}) 에 필수 속성 없음: {missing} — "
            "POLICY_REQUIRED_ATTRS 참조. 새 수집 클라이언트는 이 계약을 갖춰야 한다."
        )
    if not policy.records:
        raise RuntimeError("No feature records were collected during rollout")
    # Block-residual/multilayer features are not exported action-token chunks, so the
    # per-token export horizon invariant only applies to action-token SAFE features.
    # hidden_states 를 저장하지 않는 수집(--attn-only-records)에서는 무의미 — skip.
    if (
        include_hidden_states
        and _uses_action_token_horizon(policy)
        and policy.exported_action_token_count != n_action_steps
    ):
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
        "task_suite_name": task_suite_name,
        "model_family": model_family,
        "policy_transport": policy_transport,
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "seed": scenario_seed,
        "scenario_seed": scenario_seed,
        "episode_success": int(episode_success),
        "ep_meta": json_safe(ep_meta),
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
        "max_episode_steps": max_episode_steps,
        "video_fps": video_fps,
        "steps_per_render": steps_per_render,
        "inference_seed": inference_seed,
        "valid_action_horizon": policy.valid_action_horizon,
        "model_action_horizon": policy.model_action_horizon,
        "num_inference_timesteps": policy.num_inference_timesteps,
        "env_name": env_name,
        "robocasa_env_source": robocasa_env_source,
        "video_source": video_source,
    }
    if include_hidden_states:
        payload["hidden_states"] = [record["hidden_state"] for record in policy.records]
    else:
        # cam-attention 전용 수집(--attn-only-records): activation 텐서 미저장
        # (eval purge 규약) — cross_attn 요약만 아래에서 기록.
        payload["hidden_states_omitted"] = True
    if extra_metadata:
        payload.update(json_safe(extra_metadata))
    for key in (
        "capture_layers",
        "layer_indices",
        "layer_count",
        "token_count",
        "capture_token_mode",
    ):
        value = getattr(policy, key, None)
        if value is not None:
            payload[key] = json_safe(value)
    # DiT cross-attention 카메라 뷰별 mass (--capture-cross-attn). 모든 step 에 있을 때만.
    if all("cross_attn" in record for record in policy.records):
        # [n_records, n_cross_blocks, K, qgroup, kgroup] float32 (record 축이 맨 앞).
        payload["cross_attn"] = np.stack(
            [record["cross_attn"] for record in policy.records], axis=0
        )
        for key in (
            "cross_attn_axes",
            "cross_attn_blocks",
            "cross_attn_qgroups",
            "cross_attn_kgroups",
            "view_token_spans",
        ):
            value = getattr(policy, key, None)
            if value is not None:
                payload[key] = json_safe(value)
    # VL(goal) pathway feature (multilayer --capture-vl). 모든 step 에 있을 때만 기록.
    if include_hidden_states and all("vl_hidden_state" in record for record in policy.records):
        payload["vl_hidden_states"] = [record["vl_hidden_state"] for record in policy.records]
        payload["vl_feature_kind"] = getattr(policy, "vl_feature_kind", None)
        payload["vl_feature_axes"] = getattr(policy, "vl_feature_axes", None)
        payload["vl_feature_dim"] = getattr(policy, "vl_feature_dim", None)
    # Robot proprio state per inference (paper state-probe target). 모든 step 에 있을 때만 기록.
    if all("state" in record for record in policy.records):
        payload["states"] = [record["state"] for record in policy.records]
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    if upstream_video_path is None or not upstream_video_path.exists():
        raise RuntimeError(f"GR00T upstream video was not written: {upstream_video_path}")
    shutil.move(str(upstream_video_path), str(output_dir / f"{stem}.mp4"))
