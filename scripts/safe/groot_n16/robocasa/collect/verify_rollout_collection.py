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


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


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
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="target_atomic_seen18")
    parser.add_argument(
        "--tasks-override",
        nargs="+",
        default=None,
        help="Task names to verify instead of the named task set; mirrors collection TASKS_OVERRIDE.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-env-source", default="robocasa365")
    parser.add_argument(
        "--expected-feature-kind",
        default="groot_n16_dit_valid_action_tokens_pre_velocity",
    )
    parser.add_argument("--expected-hidden-shape", type=parse_shape, default=(4, 16, 1024))
    parser.add_argument("--expected-action-dim", type=int, default=12)
    parser.add_argument("--expected-model-horizon", type=int, default=50)
    parser.add_argument("--expected-valid-horizon", type=int, default=16)
    parser.add_argument("--expected-feature-action-horizon", type=int, default=None)
    parser.add_argument("--expected-n-action-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    tasks = tuple(args.tasks_override) if args.tasks_override else TASK_SETS[args.task_set]
    expected_total = len(tasks) * args.episodes_per_task

    if not root.is_dir():
        raise SystemExit(f"ERROR: rollout root does not exist: {root}")

    rows: dict[tuple[int, int], Path] = {}
    errors: list[str] = []
    success_by_task: dict[int, int] = defaultdict(int)
    count_by_task: dict[int, int] = defaultdict(int)

    for path in sorted(root.glob("*/*.pkl")):
        match = PKL_RE.match(path.name)
        if match is None:
            errors.append(f"unexpected pkl filename: {path}")
            continue

        task_id = int(match.group("task_id"))
        episode_idx = int(match.group("episode_idx"))
        key = (task_id, episode_idx)
        if key in rows:
            errors.append(f"duplicate pkl for task={task_id} episode={episode_idx}: {path}")
            continue
        rows[key] = path

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
        if payload.get("feature_kind") != args.expected_feature_kind:
            errors.append(f"feature kind mismatch for {path}: {payload.get('feature_kind')}")
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

        count_by_task[task_id] += 1
        success_by_task[task_id] += int(payload.get("episode_success", 0))

    missing: list[tuple[int, str, int]] = []
    for task_id, task in enumerate(tasks):
        for episode_idx in range(args.episodes_per_task):
            if (task_id, episode_idx) not in rows:
                missing.append((task_id, task, episode_idx))

    print(f"root={root}")
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
