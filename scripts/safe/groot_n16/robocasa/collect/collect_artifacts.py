"""SAFE rollout artifact writing helpers."""

from __future__ import annotations

import csv
import hashlib
import json
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
    grid_dir: Path | None = None,
) -> None:
    """SAFE rollout 산출물을 쓴다.

    ``grid_dir`` 를 주면 **docs/04 §3 좌표 레이아웃**으로 쓴다 —
    ``<grid_dir>/{rollout.pkl, traj.csv, video.mp4, meta.json}``. ``grid_dir`` 는
    ``<store>/grid/<plan_id>/<machine>/<instruction>/s<i>/n<j>/<arm>`` 이며 경로 조립은
    ``src/utils/collection_plan.py`` (``GridCell.rel_path`` · ``arm_dirname``)가 단일 출처다.

    주지 않으면 구 stem 레이아웃(``<output_dir>/<stem>.{pkl,csv,mp4}``)으로 쓰고 경고한다.
    규약 §8 은 좌표 없는 수집을 금지하지만, n16 경로(activation 폐기 예정)가 아직 구 배선이라
    즉시 실패시키지 않는다.
    """
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

    # 재수집 정책 — docs/04_data_storage_convention.md §2(쓰기 검사)·§8(덮어쓰기 금지) 준수.
    #
    # 구 배선은 여기서 `task{id}--ep{idx}--succ*.*` 를 전부 unlink 했다. 그 삭제는 발동할
    # 자리가 없거나(중간 사망은 이 함수가 완주 후에만 불려 pkl 자체가 없다) 발동하면 안 되는
    # 경우(succ 반전 = 라벨러·seed·캡처층·모델·채점 기준 변경 신호 → 비교 대상 소실)뿐이었다.
    #
    # succ 반전 파일은 덮어쓰기가 아니므로 지우지도 막지도 않고 경고만 남긴다.
    # 같은 stem 에 대한 처리는 아래 §2 쓰기 검사(pkl 직렬화 후)가 담당한다.
    if grid_dir is not None:
        # 좌표 레이아웃 — 한 칸 = 한 디렉토리, 파일명은 고정이라 succ 반전 충돌이 없다.
        dest = Path(grid_dir)
        dest.mkdir(parents=True, exist_ok=True)
        pkl_path, csv_path = dest / "rollout.pkl", dest / "traj.csv"
        mp4_path, meta_path = dest / "video.mp4", dest / "meta.json"
    else:
        print(
            "[collect][warn] grid_dir 없음 — 구 stem 레이아웃으로 쓴다. "
            "docs/04 §8 은 좌표 없는 수집을 금지한다(결손을 알 수 없다). "
            "collection_plan 의 GridCell.rel_path 로 경로를 만들어 넘길 것.",
            flush=True,
        )
        dest = output_dir
        other = sorted(
            p for p in output_dir.glob(f"task{task_id}--ep{episode_idx}--succ*.*")
            if not p.name.startswith(stem)
        )
        if other:
            print(
                f"[collect][warn] 같은 episode 의 다른 판정 산출물이 이미 있다 "
                f"(succ 반전 = 조건 변경 가능성): {[p.name for p in other]} — "
                f"지우지 않고 {stem}.* 로 새로 쓴다. 의도한 재수집이면 옛 파일을 직접 정리할 것.",
                flush=True,
            )
        pkl_path, csv_path = output_dir / f"{stem}.pkl", output_dir / f"{stem}.csv"
        mp4_path, meta_path = output_dir / f"{stem}.mp4", None

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
    # ── §2 쓰기 검사 ──────────────────────────────────────────────────────────
    # 기존 pkl 없음 → 그대로 쓴다 / 내용 동일 → skip / 내용 상이 → 에러 중단.
    # sig 레이아웃 이관 후에는 이 블록이 sig 디렉토리 단위 검사로 대체된다(규약 §2·§7).
    blob = pickle.dumps(payload)
    new_sig = hashlib.sha256(blob).hexdigest()
    if pkl_path.exists():
        old_sig = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
        if old_sig == new_sig:
            print(f"[collect] {pkl_path} 이미 동일 내용 (sig {new_sig[:16]}) — skip", flush=True)
            return
        raise RuntimeError(
            f"{pkl_path} 가 이미 있고 내용이 다르다 "
            f"(기존 {old_sig[:16]} != 신규 {new_sig[:16]}). "
            "덮어쓰기 금지(docs/04_data_storage_convention.md §8) — 조건을 바꾼 재수집이면 "
            "좌표(plan_id/machine)를 분리하고, 의도한 교체면 옛 파일을 직접 정리할 것."
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAFE_ACTION_COLUMNS)
        writer.writeheader()
        for record in policy.records:
            values = record["action_vector"]
            writer.writerow({col: float(values[i]) for i, col in enumerate(SAFE_ACTION_COLUMNS)})

    pkl_path.write_bytes(blob)

    if meta_path is not None:
        # docs/04 §6-3 — rollout 은 meta.json 을 함께 쓴다. sig 는 식별자가 아니라
        # 무결성 검증 열이므로(§3.1) 여기 기록한다. 캡처 밀도 5 열은 §4 인덱스가 요구.
        meta = {
            "sig": new_sig[:16],
            "pkl_sha256": new_sig,
            "model": model_family, "policy_transport": policy_transport,
            "task": payload.get("robocasa_task"), "task_id": task_id,
            "instruction": task_description,
            "env_seed": scenario_seed, "inference_seed": inference_seed,
            "episode_idx": episode_idx, "success": int(episode_success),
            "n_action_steps": n_action_steps, "env_name": env_name,
            "capture_token_mode": getattr(policy, "capture_token_mode", None),
            "feature_kind": policy.feature_kind,
            "feature_axes": policy.feature_axes,
            "capture_layers": getattr(policy, "capture_layers", None),
            "record_shape": (list(np.asarray(policy.records[0]["hidden_state"]).shape)
                             if include_hidden_states and policy.records else None),
        }
        for key in ("machine", "ckpt", "plan_id", "grid_instruction", "scene_idx",
                    "noise_idx", "armsig", "serve_gpu", "serve_boot_id"):
            if extra_metadata and key in extra_metadata:
                meta[key] = extra_metadata[key]
        meta_path.write_text(json.dumps(json_safe(meta), indent=2, ensure_ascii=False))

    if upstream_video_path is None or not upstream_video_path.exists():
        raise RuntimeError(f"GR00T upstream video was not written: {upstream_video_path}")
    shutil.move(str(upstream_video_path), str(mp4_path))
