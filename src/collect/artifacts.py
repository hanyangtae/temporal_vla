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

# policy 계약(SafeFeaturePolicy/POLICY_*_ATTRS)은 schema 가 단일 출처다.
from src.collect.schema import (
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
    grid_dir: Path | None = None,   # 필수 — None 이면 RuntimeError
    arm_config: dict[str, Any] | None = None,   # steered 수집(arm)의 진실 — config.json
    diag_unplanned: bool = False,   # 진단 캡처 — 수집 정본 불변식(export horizon) 완화
) -> None:
    """SAFE rollout 산출물을 쓴다.

    ``grid_dir`` 를 주면 **docs/04 §3 좌표 레이아웃**으로 쓴다 —
    ``<grid_dir>/{rollout.pkl, traj.csv, video.mp4, meta.json}``. ``grid_dir`` 는
    ``<store>/grid/<plan_id>/<machine>/<instruction>/s<i>/k<r>/n<j>/<arm>`` (지터 축이 없는
    legacy plan 은 ``k<r>`` 층 없음) 이며 경로 조립은 ``src/collect/plan.py``
    (``GridCell.rel_path`` · ``arm_dirname``)가 단일 출처다.

    **``grid_dir`` 은 필수다.** 없으면 RuntimeError — 규약 §8 이 좌표 없는 수집을 금지한다
    (좌표가 없으면 계획 대비 결손을 알 수 없다). ``output_dir``·``stem`` 은 구 호출부 호환과
    로그용으로만 남는다.
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
        and not diag_unplanned    # 진단 캡처: steering serve 는 전 토큰(16)을 내보냄 — 허용
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
    # 좌표 레이아웃에서는 파일명이 고정(rollout.pkl 등)이라 succ 반전 충돌 자체가 없다.
    # 같은 좌표 재실행은 아래 §2 쓰기 검사(pkl 직렬화 후)가 처리한다.
    if grid_dir is None:
        raise RuntimeError(
            "grid_dir 없이 수집할 수 없다 — docs/04 §8 은 좌표 없는 수집을 금지한다"
            "(좌표가 없으면 계획 대비 결손을 알 수 없다). "
            "collection_plan 의 add_grid_args/resolve_grid/grid_dir_for 로 좌표를 넘길 것."
        )
    # 좌표 레이아웃 — 한 칸 = 한 디렉토리, 파일명이 고정이라 succ 반전 충돌이 없다.
    dest = Path(grid_dir)
    dest.mkdir(parents=True, exist_ok=True)
    pkl_path, csv_path = dest / "rollout.pkl", dest / "traj.csv"
    mp4_path, meta_path = dest / "video.mp4", dest / "meta.json"

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
        payload["hidden_states_omitted"] = True  # (구 attn-only 흔적 — 현재 호출부는 항상 True)
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
        # 좌표 3축(docs/04 §3.1.1). v6: scene_idx = 주방(layout, style), jitter_idx = j
        # 층, jitter_reset_idx = 그 j 의 연속 reset 횟수, base_lat/base_back = base 오프셋,
        # init_robot_base_pos = 오프셋 **적용 후** 실제 값(재계산 대조용). v5 는 jitter_idx
        # 없이 jitter_reset_idx(k) 만, legacy 2축은 둘 다 없어 meta 에서도 빠진다.
        for key in ("machine", "ckpt", "plan_id", "grid_instruction", "scene_idx",
                    "jitter_reset_idx", "noise_idx", "armsig", "serve_gpu", "serve_boot_id",
                    "jitter_idx", "base_lat", "base_back", "side",
                    "layout_id", "style_id", "lang", "init_robot_base_pos"):
            if extra_metadata and key in extra_metadata:
                meta[key] = extra_metadata[key]
        meta_path.write_text(json.dumps(json_safe(meta), indent=2, ensure_ascii=False))

    # steered 수집(capture-ON arm): 개입 파라미터의 진실 기록 (docs/04 §3.3 — armsig 는
    # 해시라 복원 불가, config.json 이 사람이 읽는 정본). base 수집은 넘기지 않는다.
    if arm_config is not None:
        (dest / "config.json").write_text(
            json.dumps(json_safe(arm_config), indent=2, ensure_ascii=False)
        )

    if upstream_video_path is None or not upstream_video_path.exists():
        raise RuntimeError(f"GR00T upstream video was not written: {upstream_video_path}")
    shutil.move(str(upstream_video_path), str(mp4_path))


def write_eval_artifacts(
    grid_dir: Path,
    *,
    sidecar: dict[str, Any],
    upstream_video_path: Path | None,
    action_rows: list[Any] | None,
    arm_config: dict[str, Any] | None,
) -> None:
    """평가 rollout(캡처 OFF, pkl 無)을 좌표 레이아웃 arm 디렉토리에 쓴다 (docs/04 §3.3).

    ``<grid_dir>/{meta.json, traj.csv, video.mp4, config.json}`` — 수집(pkl 有)과 달리
    activation 이 없으므로 meta.json 이 판정 사이드카를 겸한다. ``config.json`` 은 arm 의
    진실(armsig 재료 파라미터 + serve steering 지문)이며 base 디렉토리에는 쓰지 않는다.

    §2 쓰기 검사: 같은 좌표에 meta.json 이 이미 있으면 내용 동일 → skip, 상이 → 에러.
    """
    dest = Path(grid_dir)
    dest.mkdir(parents=True, exist_ok=True)
    meta_path, config_path = dest / "meta.json", dest / "config.json"
    csv_path, mp4_path = dest / "traj.csv", dest / "video.mp4"

    body = json.dumps(json_safe(sidecar), indent=2, ensure_ascii=False)
    if meta_path.exists():
        if meta_path.read_text() == body:
            print(f"[eval] {meta_path} 이미 동일 내용 — skip", flush=True)
            return
        raise RuntimeError(
            f"{meta_path} 가 이미 있고 내용이 다르다 — 덮어쓰기 금지"
            "(docs/04_data_storage_convention.md §8). 조건이 다르면 armsig(=디렉토리)가 "
            "달라야 정상이다. 같은 arm 재실행이 결과가 달라졌다면 원인(seed·머신·serve "
            "지문)을 먼저 확인할 것."
        )
    meta_path.write_text(body)

    if arm_config is not None:
        config_path.write_text(json.dumps(json_safe(arm_config), indent=2, ensure_ascii=False))

    if action_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SAFE_ACTION_COLUMNS)
            writer.writeheader()
            for values in action_rows:
                writer.writerow(
                    {col: float(values[i]) for i, col in enumerate(SAFE_ACTION_COLUMNS)}
                )

    if upstream_video_path is not None and upstream_video_path.exists():
        shutil.move(str(upstream_video_path), str(mp4_path))
