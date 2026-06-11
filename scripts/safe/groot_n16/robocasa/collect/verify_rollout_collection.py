#!/usr/bin/env python3
"""Verify GR00T N1.6 RoboCasa rollout collection artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np


TARGET_ATOMIC_SEEN18 = (
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
)

TASK_SETS = {
    "target_atomic_seen18": TARGET_ATOMIC_SEEN18,
    "robocasa365_atomic_seen18": TARGET_ATOMIC_SEEN18,
}

PKL_RE = re.compile(r"^task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<succ>[01])\.pkl$")


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape: {text!r}") from exc


def parse_optional_int(text: str) -> int | None:
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid optional int: {text!r}") from exc


def parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_csv_ints(text: str) -> list[int]:
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {text!r}") from exc


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def load_instruction_cell_config(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = {row["cell_id"]: dict(row) for row in reader}
    for row in rows.values():
        row["cell_index"] = int(row["cell_index"])
    return rows


def load_collection_summary_seed_info(root: Path) -> dict[str, Any] | None:
    summary_path = root / "collection_summary.tsv"
    if not summary_path.is_file():
        return None

    seeds: list[int] = []
    rows = 0
    seed_base: int | None = None
    matches_episode_formula = True
    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows += 1
            try:
                seed = int(row["seed"])
                episode_idx = int(row["episode_idx"])
            except (KeyError, TypeError, ValueError):
                matches_episode_formula = False
                continue
            if seed_base is None:
                seed_base = seed - episode_idx
            seeds.append(seed)
            if seed - episode_idx != seed_base:
                matches_episode_formula = False

    if not seeds:
        return {"path": str(summary_path), "rows": rows, "seeds": None}

    return {
        "path": str(summary_path),
        "rows": rows,
        "seed_start": min(seeds),
        "seed_end": max(seeds),
        "unique_seeds": len(set(seeds)),
        "matches_seed_start_plus_episode_idx": matches_episode_formula,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="raw_rollouts directory to verify")
    parser.add_argument(
        "--layout",
        choices=["task", "task_instruction"],
        default="task",
        help="Artifact layout: task uses root/<task>/*.pkl; task_instruction uses root/<task>/<cell_id>/*.pkl.",
    )
    parser.add_argument(
        "--cell-config",
        type=Path,
        default=None,
        help="Instruction cell TSV used to validate task_instruction layout metadata.",
    )
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="target_atomic_seen18")
    parser.add_argument(
        "--tasks-override",
        nargs="+",
        default=None,
        help="Task names to verify instead of the named task set; mirrors collection TASKS_OVERRIDE.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--expected-env-source",
        "--expected-robocasa-env-source",
        dest="expected_env_source",
        default="robocasa365",
    )
    parser.add_argument("--expected-model-family", default=None)
    parser.add_argument("--expected-policy-transport", default=None)
    parser.add_argument("--expected-task-suite-name", default=None)
    parser.add_argument("--expected-video-source", default=None)
    parser.add_argument(
        "--expected-feature-kind",
        default="groot_n16_dit_valid_action_tokens_pre_velocity",
    )
    parser.add_argument("--expected-feature-axes", type=parse_csv_strings, default=None)
    parser.add_argument("--expected-hidden-shape", type=parse_shape, default=(4, 16, 1024))
    parser.add_argument("--expected-action-dim", type=int, default=12)
    parser.add_argument("--expected-model-horizon", type=parse_optional_int, default=50)
    parser.add_argument("--expected-valid-horizon", type=parse_optional_int, default=16)
    parser.add_argument("--expected-feature-action-horizon", type=parse_optional_int, default=None)
    parser.add_argument("--expected-n-action-steps", type=parse_optional_int, default=None)
    parser.add_argument("--expected-capture-layers", type=parse_csv_ints, default=None)
    parser.add_argument("--expected-layer-count", type=parse_optional_int, default=None)
    parser.add_argument("--expected-token-count", type=parse_optional_int, default=None)
    parser.add_argument("--expected-max-episode-steps", type=parse_optional_int, default=None)
    parser.add_argument("--expected-video-fps", type=parse_optional_int, default=None)
    parser.add_argument("--expected-steps-per-render", type=parse_optional_int, default=None)
    parser.add_argument("--expected-inference-seed", type=parse_optional_int, default=None)
    parser.add_argument(
        "--require-vl-hidden-states",
        action="store_true",
        help="Require VL pathway features in pkl. Default is DiT-only compatible.",
    )
    parser.add_argument("--expected-vl-hidden-shape", type=parse_shape, default=(2048,))
    parser.add_argument("--expected-vl-feature-kind", default=None)
    parser.add_argument("--expected-vl-feature-dim", type=parse_optional_int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    tasks = tuple(args.tasks_override) if args.tasks_override else TASK_SETS[args.task_set]
    task_filter = set(args.tasks_override or [])
    cell_config = (
        {}
        if args.cell_config is None
        else load_instruction_cell_config(args.cell_config)
    )
    expected_cells = {
        cell_id: row
        for cell_id, row in cell_config.items()
        if not task_filter or row["task"] in task_filter
    }
    expected_total = (
        len(expected_cells) * args.episodes_per_task
        if args.layout == "task_instruction"
        else len(tasks) * args.episodes_per_task
    )

    if not root.is_dir():
        raise SystemExit(f"ERROR: rollout root does not exist: {root}")

    rows: dict[tuple[Any, ...], Path] = {}
    errors: list[str] = []
    success_by_task: dict[Any, int] = defaultdict(int)
    count_by_task: dict[Any, int] = defaultdict(int)
    discovered_cells: dict[str, dict[str, Any]] = {}

    pkl_pattern = "*/*/*.pkl" if args.layout == "task_instruction" else "*/*.pkl"
    for path in sorted(root.glob(pkl_pattern)):
        match = PKL_RE.match(path.name)
        if match is None:
            errors.append(f"unexpected pkl filename: {path}")
            continue

        task_id = int(match.group("task_id"))
        episode_idx = int(match.group("episode_idx"))
        layout_cell_id = None
        layout_task_name = None
        expected_cell = None
        if args.layout == "task_instruction":
            layout_cell_id = path.parent.name
            layout_task_name = path.parent.parent.name
            key = ("cell", layout_cell_id, episode_idx)
            expected_cell = cell_config.get(layout_cell_id)
            discovered_cells.setdefault(
                layout_cell_id,
                {
                    "cell_id": layout_cell_id,
                    "task": layout_task_name,
                    "cell_index": task_id,
                },
            )
        else:
            key = ("task", task_id, episode_idx)
        if key in rows:
            errors.append(f"duplicate pkl for {key} episode={episode_idx}: {path}")
            continue
        rows[key] = path

        if args.layout == "task_instruction":
            if task_filter and layout_task_name not in task_filter:
                errors.append(f"unexpected task directory for {path}: {layout_task_name}")
                continue
            if expected_cell is not None:
                if layout_task_name != expected_cell["task"]:
                    errors.append(
                        f"task directory mismatch for {path}: expected {expected_cell['task']}"
                    )
                    continue
                if task_id != expected_cell["cell_index"]:
                    errors.append(
                        f"cell index mismatch for {path}: expected {expected_cell['cell_index']}"
                    )
                    continue
        else:
            if task_id >= len(tasks):
                errors.append(f"task id out of range in {path}")
                continue
            if path.parent.name != tasks[task_id]:
                errors.append(
                    f"task directory mismatch for {path}: expected {tasks[task_id]}"
                )
                continue

        csv_path = path.with_suffix(".csv")
        mp4_path = path.with_suffix(".mp4")
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            errors.append(f"missing or empty csv for {path}")
        if not mp4_path.is_file() or mp4_path.stat().st_size == 0:
            errors.append(f"missing or empty mp4 for {path}")

        try:
            payload = load_pickle(path)
        except Exception as exc:
            errors.append(f"failed to load {path}: {exc}")
            continue

        if args.layout == "task_instruction":
            assert layout_cell_id is not None
            assert layout_task_name is not None
            if payload.get("cell_id") != layout_cell_id:
                errors.append(f"cell_id mismatch for {path}: {payload.get('cell_id')}")
            if payload.get("cell_index") != task_id:
                errors.append(f"cell_index mismatch for {path}: {payload.get('cell_index')}")
            if payload.get("robocasa_task") != layout_task_name:
                errors.append(
                    f"robocasa_task mismatch for {path}: {payload.get('robocasa_task')}"
                )
            expected_instruction = None
            if expected_cell is not None:
                expected_instruction = expected_cell["canonical_instruction"]
            elif payload.get("canonical_instruction") is not None:
                expected_instruction = payload.get("canonical_instruction")
            if expected_instruction is None:
                errors.append(f"missing canonical_instruction for {path}")
            else:
                if payload.get("canonical_instruction") != expected_instruction:
                    errors.append(
                        "canonical_instruction mismatch for "
                        f"{path}: {payload.get('canonical_instruction')}"
                    )
                if payload.get("task_description") != expected_instruction:
                    errors.append(
                        f"task_description mismatch for {path}: "
                        f"{payload.get('task_description')}"
                    )
                ep_meta = payload.get("ep_meta")
                ep_lang = ep_meta.get("lang") if isinstance(ep_meta, dict) else None
                if ep_lang != expected_instruction:
                    errors.append(f"ep_meta lang mismatch for {path}: {ep_lang}")

        hidden_states = payload.get("hidden_states") or []
        action_vectors = np.asarray(payload.get("action_vectors", []))
        hidden_shape = tuple(np.asarray(hidden_states[0]).shape) if hidden_states else None
        if hidden_shape != args.expected_hidden_shape:
            errors.append(f"hidden shape mismatch for {path}: {hidden_shape}")
        if action_vectors.ndim != 2 or action_vectors.shape[1] != args.expected_action_dim:
            errors.append(f"action shape mismatch for {path}: {tuple(action_vectors.shape)}")
        if len(hidden_states) != action_vectors.shape[0]:
            errors.append(
                f"step count mismatch for {path}: hidden={len(hidden_states)} action={action_vectors.shape[0]}"
            )
        if payload.get("robocasa_env_source") != args.expected_env_source:
            errors.append(f"env source mismatch for {path}: {payload.get('robocasa_env_source')}")
        if (
            args.expected_model_family is not None
            and payload.get("model_family") != args.expected_model_family
        ):
            errors.append(f"model family mismatch for {path}: {payload.get('model_family')}")
        if (
            args.expected_policy_transport is not None
            and payload.get("policy_transport") != args.expected_policy_transport
        ):
            errors.append(
                f"policy transport mismatch for {path}: {payload.get('policy_transport')}"
            )
        if (
            args.expected_task_suite_name is not None
            and payload.get("task_suite_name") != args.expected_task_suite_name
        ):
            errors.append(
                f"task suite mismatch for {path}: {payload.get('task_suite_name')}"
            )
        if (
            args.expected_video_source is not None
            and payload.get("video_source") != args.expected_video_source
        ):
            errors.append(f"video source mismatch for {path}: {payload.get('video_source')}")
        if payload.get("feature_kind") != args.expected_feature_kind:
            errors.append(f"feature kind mismatch for {path}: {payload.get('feature_kind')}")
        if args.expected_feature_axes is not None:
            feature_axes = list(payload.get("feature_axes") or [])
            if feature_axes != args.expected_feature_axes:
                errors.append(f"feature axes mismatch for {path}: {feature_axes}")
        if args.expected_capture_layers is not None:
            capture_layers = list(payload.get("capture_layers") or [])
            if capture_layers != args.expected_capture_layers:
                errors.append(f"capture layers mismatch for {path}: {capture_layers}")
        if (
            args.expected_layer_count is not None
            and payload.get("layer_count") != args.expected_layer_count
        ):
            errors.append(f"layer_count mismatch for {path}: {payload.get('layer_count')}")
        if (
            args.expected_token_count is not None
            and payload.get("token_count") != args.expected_token_count
        ):
            errors.append(f"token_count mismatch for {path}: {payload.get('token_count')}")
        if payload.get("model_action_horizon") != args.expected_model_horizon:
            errors.append(
                f"model horizon mismatch for {path}: {payload.get('model_action_horizon')}"
            )
        if payload.get("valid_action_horizon") != args.expected_valid_horizon:
            errors.append(
                f"valid horizon mismatch for {path}: {payload.get('valid_action_horizon')}"
            )
        if args.expected_feature_action_horizon is not None:
            if payload.get("feature_action_horizon") != args.expected_feature_action_horizon:
                errors.append(
                    "feature action horizon mismatch for "
                    f"{path}: {payload.get('feature_action_horizon')}"
                )
            if (
                payload.get("exported_action_token_count")
                != args.expected_feature_action_horizon
            ):
                errors.append(
                    "exported action token count mismatch for "
                    f"{path}: {payload.get('exported_action_token_count')}"
                )
        if (
            args.expected_n_action_steps is not None
            and payload.get("n_action_steps") != args.expected_n_action_steps
        ):
            errors.append(f"n_action_steps mismatch for {path}: {payload.get('n_action_steps')}")
        if (
            args.expected_max_episode_steps is not None
            and payload.get("max_episode_steps") != args.expected_max_episode_steps
        ):
            errors.append(
                f"max_episode_steps mismatch for {path}: {payload.get('max_episode_steps')}"
            )
        if (
            args.expected_video_fps is not None
            and payload.get("video_fps") != args.expected_video_fps
        ):
            errors.append(f"video_fps mismatch for {path}: {payload.get('video_fps')}")
        if (
            args.expected_steps_per_render is not None
            and payload.get("steps_per_render") != args.expected_steps_per_render
        ):
            errors.append(
                f"steps_per_render mismatch for {path}: {payload.get('steps_per_render')}"
            )
        if (
            args.expected_inference_seed is not None
            and payload.get("inference_seed") != args.expected_inference_seed
        ):
            errors.append(f"inference_seed mismatch for {path}: {payload.get('inference_seed')}")
        vl_hidden_states = payload.get("vl_hidden_states") or []
        if args.require_vl_hidden_states and not vl_hidden_states:
            errors.append(f"missing vl_hidden_states for {path}")
        if vl_hidden_states:
            vl_shape = tuple(np.asarray(vl_hidden_states[0]).shape)
            if vl_shape != args.expected_vl_hidden_shape:
                errors.append(f"vl hidden shape mismatch for {path}: {vl_shape}")
            if len(vl_hidden_states) != len(hidden_states):
                errors.append(
                    f"vl step count mismatch for {path}: "
                    f"vl={len(vl_hidden_states)} hidden={len(hidden_states)}"
                )
            if (
                args.expected_vl_feature_kind is not None
                and payload.get("vl_feature_kind") != args.expected_vl_feature_kind
            ):
                errors.append(
                    f"vl feature kind mismatch for {path}: {payload.get('vl_feature_kind')}"
                )
            if (
                args.expected_vl_feature_dim is not None
                and payload.get("vl_feature_dim") != args.expected_vl_feature_dim
            ):
                errors.append(
                    f"vl feature dim mismatch for {path}: {payload.get('vl_feature_dim')}"
                )

        unit_id = layout_cell_id if args.layout == "task_instruction" else task_id
        count_by_task[unit_id] += 1
        success_by_task[unit_id] += int(payload.get("episode_success", 0))

    if args.layout == "task_instruction" and not expected_cells:
        expected_cells = discovered_cells
        expected_total = len(expected_cells) * args.episodes_per_task

    missing: list[tuple[Any, str, int]] = []
    if args.layout == "task_instruction":
        for cell_id, cell in expected_cells.items():
            for episode_idx in range(args.episodes_per_task):
                if ("cell", cell_id, episode_idx) not in rows:
                    missing.append((cell_id, cell["task"], episode_idx))
    else:
        for task_id, task in enumerate(tasks):
            for episode_idx in range(args.episodes_per_task):
                if ("task", task_id, episode_idx) not in rows:
                    missing.append((task_id, task, episode_idx))

    print(f"root={root}")
    if args.layout == "task_instruction":
        print(
            f"layout=task_instruction cells={len(expected_cells)} "
            f"episodes_per_cell={args.episodes_per_task}"
        )
    else:
        print(f"tasks={len(tasks)} episodes_per_task={args.episodes_per_task}")
    print(f"completed={len(rows)} expected={expected_total}")
    seed_info = load_collection_summary_seed_info(root)
    if seed_info is not None:
        if "seed_start" not in seed_info:
            print(f"summary_seeds=unavailable rows={seed_info['rows']}")
        else:
            message = (
                f"summary_seeds={seed_info['seed_start']}..{seed_info['seed_end']} "
                f"unique={seed_info['unique_seeds']} rows={seed_info['rows']}"
            )
            if seed_info["matches_seed_start_plus_episode_idx"]:
                message += " formula=seed_start+episode_idx"
            print(message)
    if args.layout == "task_instruction":
        for cell_id, cell in sorted(expected_cells.items()):
            count = count_by_task.get(cell_id, 0)
            succ = success_by_task.get(cell_id, 0)
            print(f"{cell_id}\t{cell['task']}\tcount={count}\tsuccess={succ}")
    else:
        for task_id, task in enumerate(tasks):
            count = count_by_task.get(task_id, 0)
            succ = success_by_task.get(task_id, 0)
            print(f"{task_id}\t{task}\tcount={count}\tsuccess={succ}")

    if missing:
        preview = ", ".join(f"{task}:ep{ep}" for _, task, ep in missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... +{len(missing) - 10}"
        print(f"missing={len(missing)} {preview}{suffix}")
        if not args.allow_partial:
            errors.append(f"missing {len(missing)} expected episodes")

    if errors:
        print("ERRORS:")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"- ... +{len(errors) - 50} more")
        raise SystemExit(1)

    print("status=ok" if not missing else "status=partial-ok")


if __name__ == "__main__":
    main()
