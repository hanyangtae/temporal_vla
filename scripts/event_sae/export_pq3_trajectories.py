"""Export trusted GR00T N1.5 PQ3 rollout PKLs to the common trajectory schema.

The exporter is standalone so it can run beside the remote rollout archive.
It loads one PKL at a time and writes only small, portable JSON artifacts:
``trajectory_records.jsonl`` for the existing AWE pipeline and
``trajectory_manifest.json`` for provenance and video timing.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CELL_COUNTS = {
    "OpenDrawer/pq3_drawer_left": 30,
    "OpenDrawer/pq3_drawer_right": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_beer": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_bread": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_pizza_cutter": 30,
}
EXPECTED_NUM_RECORDS = 12_041
MANIFEST_FORMAT = "groot_n15_pq3_trajectory_manifest_v1"
RECORDS_NAME = "trajectory_records.jsonl"
MANIFEST_NAME = "trajectory_manifest.json"
AUDIT_NAME = "trajectory_audit.json"
FILENAME_RE = re.compile(
    r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$"
)

STATE_FIELDS = {
    "eef_pos": ("observation.state.eef_pos_rel", 3),
    "eef_quat": ("observation.state.eef_quat_rel", 4),
    "gripper_qpos": ("observation.state.gripper_qpos", 2),
}
BASE_POSITION_KEY = "observation.state.base_position"
BASE_ROTATION_KEY = "observation.state.base_rotation"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _vector(value: Any, *, size: int, label: str) -> list[float]:
    array = _as_numpy(value).astype(np.float64, copy=False).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{label}: shape={array.shape}, expected=({size},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite value")
    return [float(item) for item in array]


def _rotate_vector_xyzw(
    vector: list[float],
    quaternion: list[float],
    *,
    label: str,
) -> np.ndarray:
    """Rotate a 3-vector by a Robosuite `(x, y, z, w)` quaternion."""
    vector_array = np.asarray(vector, dtype=np.float64)
    quaternion_array = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion_array))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label}: quaternion norm must be finite and non-zero")
    quaternion_array /= norm
    xyz = quaternion_array[:3]
    w = float(quaternion_array[3])
    cross = np.cross(xyz, vector_array)
    return vector_array + 2.0 * (w * cross + np.cross(xyz, cross))


def _absolute_eef_position(
    eef_pos_rel: list[float],
    base_position: list[float],
    base_rotation: list[float],
    *,
    label: str,
) -> list[float]:
    rotated = _rotate_vector_xyzw(
        eef_pos_rel,
        base_rotation,
        label=f"{label} base_rotation",
    )
    position = np.asarray(base_position, dtype=np.float64) + rotated
    if not np.isfinite(position).all():
        raise ValueError(f"{label}: reconstructed eef_pos_abs is non-finite")
    return [float(item) for item in position]


def _event_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _event_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _inventory(root: Path, allow_partial: bool) -> tuple[list[Path], dict[str, int]]:
    paths = sorted(path for path in root.rglob("*.pkl") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No PKLs under {root}")
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"PQ3 source inventory contains symlink PKL: {symlinks[0]}")
    counts = dict(
        sorted(Counter(path.relative_to(root).parent.as_posix() for path in paths).items())
    )
    if not allow_partial and counts != EXPECTED_CELL_COUNTS:
        raise ValueError(
            f"PQ3 source inventory mismatch: actual={counts}, expected={EXPECTED_CELL_COUNTS}"
        )
    return paths, counts


def _parse_filename(path: Path) -> tuple[int, int, bool]:
    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected PQ3 filename: {path.name}")
    return (
        int(match.group("task_id")),
        int(match.group("episode_idx")),
        bool(int(match.group("success"))),
    )


def _int_metadata(payload: dict, key: str, path: Path) -> int:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{path}: missing {key}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{path}: {key} must be positive, got {value}")
    return value


def export_trajectories(args: argparse.Namespace) -> None:
    if not args.trust_pkl:
        raise ValueError("Refusing pickle.load without explicit --trust-pkl")

    root = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    paths, cell_counts = _inventory(root, args.allow_partial_inventory)
    output.mkdir(parents=True)
    records_path = output / RECORDS_NAME
    manifest_path = output / MANIFEST_NAME
    episodes: list[dict[str, Any]] = []
    total_records = 0

    with records_path.open("w", encoding="utf-8") as records_file:
        for episode_num, path in enumerate(paths):
            relative_path = path.relative_to(root)
            filename_task_id, filename_episode_idx, filename_success = _parse_filename(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 -- trusted only with explicit flag.

            states = payload.get("states")
            actions = payload.get("actions")
            if not isinstance(states, (list, tuple)) or not states:
                raise ValueError(f"{path}: missing non-empty states")
            actions_len = len(actions) if isinstance(actions, (list, tuple)) else None
            if actions_len != len(states):
                raise ValueError(
                    f"{path}: actions/states length mismatch "
                    f"({actions_len} vs {len(states)})"
                )

            task_id = int(payload.get("task_id", filename_task_id))
            task_episode_idx = int(payload.get("episode_idx", filename_episode_idx))
            success = bool(int(payload.get("episode_success", filename_success)))
            if task_id != filename_task_id:
                raise ValueError(f"{path}: task_id={task_id}, filename={filename_task_id}")
            if task_episode_idx != filename_episode_idx:
                raise ValueError(
                    f"{path}: episode_idx={task_episode_idx}, filename={filename_episode_idx}"
                )
            if success != filename_success:
                raise ValueError(f"{path}: episode_success={success}, filename={filename_success}")

            cell_dir = relative_path.parent.as_posix()
            task_family = relative_path.parts[0]
            cell_id = str(payload.get("cell_id") or relative_path.parent.name)
            if cell_dir != f"{task_family}/{cell_id}":
                raise ValueError(f"{path}: cell_id={cell_id!r} disagrees with source path")
            robocasa_task = str(payload.get("robocasa_task") or task_family)
            if robocasa_task != task_family:
                raise ValueError(
                    f"{path}: robocasa_task={robocasa_task!r}, path family={task_family!r}"
                )

            task_description = str(payload.get("task_description") or "")
            prompt_task_description = str(
                payload.get("canonical_instruction") or task_description
            )
            if not task_description:
                raise ValueError(f"{path}: missing task_description")

            n_action_steps = _int_metadata(payload, "n_action_steps", path)
            steps_per_render = _int_metadata(payload, "steps_per_render", path)
            video_fps = float(payload.get("video_fps", 20.0))
            if not np.isfinite(video_fps) or video_fps <= 0:
                raise ValueError(f"{path}: invalid video_fps={video_fps}")

            for step_in_episode, state in enumerate(states):
                if not isinstance(state, dict):
                    raise ValueError(f"{path}: states[{step_in_episode}] is not a mapping")
                vectors = {
                    output_key: _vector(
                        state[source_key],
                        size=size,
                        label=f"{path}: states[{step_in_episode}][{source_key!r}]",
                    )
                    for output_key, (source_key, size) in STATE_FIELDS.items()
                    if source_key in state
                }
                missing = sorted(set(STATE_FIELDS) - set(vectors))
                if missing:
                    raise ValueError(f"{path}: states[{step_in_episode}] missing {missing}")
                base_position = (
                    _vector(
                        state[BASE_POSITION_KEY],
                        size=3,
                        label=(
                            f"{path}: states[{step_in_episode}]"
                            f"[{BASE_POSITION_KEY!r}]"
                        ),
                    )
                    if BASE_POSITION_KEY in state
                    else None
                )
                base_rotation = (
                    _vector(
                        state[BASE_ROTATION_KEY],
                        size=4,
                        label=(
                            f"{path}: states[{step_in_episode}]"
                            f"[{BASE_ROTATION_KEY!r}]"
                        ),
                    )
                    if BASE_ROTATION_KEY in state
                    else None
                )
                if base_position is None or base_rotation is None:
                    missing_base_fields = [
                        key
                        for key, value in (
                            (BASE_POSITION_KEY, base_position),
                            (BASE_ROTATION_KEY, base_rotation),
                        )
                        if value is None
                    ]
                    raise ValueError(
                        f"{path}: states[{step_in_episode}] missing "
                        f"absolute-frame inputs {missing_base_fields}"
                    )
                eef_pos_rel = vectors["eef_pos"]
                eef_pos_abs = _absolute_eef_position(
                    eef_pos_rel,
                    base_position,
                    base_rotation,
                    label=f"{path}: states[{step_in_episode}]",
                )
                record = {
                    "episode_num": episode_num,
                    "task_id": task_id,
                    "task_episode_idx": task_episode_idx,
                    "task_description": task_description,
                    "prompt_task_description": prompt_task_description,
                    "step_in_episode": step_in_episode,
                    **vectors,
                    "eef_pos_rel": eef_pos_rel,
                    "eef_pos_abs": eef_pos_abs,
                    "base_position": base_position,
                    "base_rotation": base_rotation,
                    "done": bool(success and step_in_episode == len(states) - 1),
                    "episode_success": success,
                    "task_family": task_family,
                    "cell_id": cell_id,
                    "source_file": relative_path.as_posix(),
                }
                records_file.write(json.dumps(record, separators=(",", ":")) + "\n")

            video_relative_path = relative_path.with_suffix(".mp4")
            csv_relative_path = relative_path.with_suffix(".csv")
            expected_video_frames = -(
                -(len(states) * n_action_steps) // steps_per_render
            )
            episodes.append(
                {
                    "episode_num": episode_num,
                    "task_id": task_id,
                    "task_episode_idx": task_episode_idx,
                    "cell_index": int(payload.get("cell_index", task_id)),
                    "cell_id": cell_id,
                    "task_family": task_family,
                    "robocasa_task": robocasa_task,
                    "task_description": task_description,
                    "prompt_task_description": prompt_task_description,
                    "success": success,
                    "source_file": relative_path.as_posix(),
                    "source_csv_relative_path": csv_relative_path.as_posix(),
                    "source_video_relative_path": video_relative_path.as_posix(),
                    "csv_present_at_export": (root / csv_relative_path).is_file(),
                    "video_present_at_export": (root / video_relative_path).is_file(),
                    "num_records": len(states),
                    "n_action_steps": n_action_steps,
                    "steps_per_render": steps_per_render,
                    "video_fps": video_fps,
                    "expected_video_frames": expected_video_frames,
                    "event_steps": _event_value(payload.get("event_steps", {})),
                    "grasp_steps": _event_value(payload.get("grasp_steps", [])),
                    "drop_steps": _event_value(payload.get("drop_steps", [])),
                }
            )
            total_records += len(states)
            if args.progress_every > 0 and (
                (episode_num + 1) % args.progress_every == 0
                or episode_num + 1 == len(paths)
            ):
                print(
                    f"[export] files={episode_num + 1}/{len(paths)} records={total_records}",
                    flush=True,
                )
            del payload, states, actions

    if not args.allow_partial_inventory and total_records != EXPECTED_NUM_RECORDS:
        raise ValueError(
            f"PQ3 record count mismatch: actual={total_records}, expected={EXPECTED_NUM_RECORDS}"
        )

    manifest = {
        "format": MANIFEST_FORMAT,
        "source_root": str(root),
        "trajectory_records_file": RECORDS_NAME,
        "num_files": len(paths),
        "num_records": total_records,
        "cell_counts": cell_counts,
        "expected_cell_counts": None if args.allow_partial_inventory else EXPECTED_CELL_COUNTS,
        "expected_num_records": None if args.allow_partial_inventory else EXPECTED_NUM_RECORDS,
        "inventory_verified": not args.allow_partial_inventory,
        "record_unit": "policy_inference_record",
        "state_frame": "robot_relative",
        "available_eef_position_frames": ["rel", "abs"],
        "state_fields": {
            "eef_pos": "legacy alias of eef_pos_rel[3]",
            "eef_pos_rel": "observation.state.eef_pos_rel[3]",
            "eef_pos_abs": (
                "observation.state.base_position[3] + "
                "R_xyzw(observation.state.base_rotation[4]) @ "
                "observation.state.eef_pos_rel[3]"
            ),
            "eef_quat": "observation.state.eef_quat_rel[4]",
            "gripper_qpos": "observation.state.gripper_qpos[2]",
            "base_position": "observation.state.base_position[3]",
            "base_rotation": "observation.state.base_rotation[4], xyzw",
        },
        "video_timeline": {
            "mapping": "frame f -> env_step f*steps_per_render -> record env_step//n_action_steps",
            "record_frame_start": "ceil(record*n_action_steps/steps_per_render)",
            "record_frame_stop_inclusive": "ceil((record+1)*n_action_steps/steps_per_render)-1",
        },
        "episodes": episodes,
        "video_inventory_at_export": {
            "present": sum(bool(episode["video_present_at_export"]) for episode in episodes),
            "missing": sum(not bool(episode["video_present_at_export"]) for episode in episodes),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[export] records={records_path}", flush=True)
    print(f"[export] manifest={manifest_path}", flush=True)


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')!r}")
    return manifest


def audit_trajectories(args: argparse.Namespace) -> dict:
    records_path = args.trajectory_records_path.resolve()
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    episode_manifest: dict[int, dict] = {}
    for episode in manifest["episodes"]:
        episode_num = int(episode["episode_num"])
        if episode_num in episode_manifest:
            raise ValueError(f"duplicate episode_num={episode_num} in manifest")
        episode_manifest[episode_num] = episode
    if len(episode_manifest) != int(manifest["num_files"]):
        raise ValueError("manifest num_files does not match episode inventory")
    seen_steps: dict[int, list[int]] = defaultdict(list)
    seen_done: dict[int, list[bool]] = defaultdict(list)
    seen_source: dict[int, set[str]] = defaultdict(set)
    record_count = 0
    dual_frame = set(manifest.get("available_eef_position_frames", [])) >= {
        "rel",
        "abs",
    }
    vector_fields = [
        ("eef_pos", 3),
        ("eef_quat", 4),
        ("gripper_qpos", 2),
    ]
    if dual_frame:
        vector_fields.extend(
            [("eef_pos_rel", 3), ("eef_pos_abs", 3), ("base_position", 3), ("base_rotation", 4)]
        )

    with records_path.open("r", encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, 1):
            record = json.loads(line)
            episode_num = int(record["episode_num"])
            if episode_num not in episode_manifest:
                raise ValueError(f"line {line_number}: unknown episode_num={episode_num}")
            episode = episode_manifest[episode_num]
            if int(record["task_id"]) != int(episode["task_id"]):
                raise ValueError(f"line {line_number}: task_id mismatch")
            for key, size in vector_fields:
                _vector(record[key], size=size, label=f"line {line_number} {key}")
            if dual_frame:
                expected_abs = _absolute_eef_position(
                    record["eef_pos_rel"],
                    record["base_position"],
                    record["base_rotation"],
                    label=f"line {line_number}",
                )
                if not np.allclose(
                    record["eef_pos"],
                    record["eef_pos_rel"],
                    rtol=1e-7,
                    atol=1e-9,
                ):
                    raise ValueError(f"line {line_number}: eef_pos alias mismatch")
                if not np.allclose(
                    record["eef_pos_abs"],
                    expected_abs,
                    rtol=1e-7,
                    atol=1e-9,
                ):
                    raise ValueError(f"line {line_number}: eef_pos_abs mismatch")
            seen_steps[episode_num].append(int(record["step_in_episode"]))
            seen_done[episode_num].append(bool(record["done"]))
            seen_source[episode_num].add(str(record["source_file"]))
            record_count += 1

    for episode_num, episode in episode_manifest.items():
        expected_steps = list(range(int(episode["num_records"])))
        if seen_steps.get(episode_num) != expected_steps:
            raise ValueError(f"episode {episode_num}: non-contiguous or missing steps")
        expected_done = [False] * len(expected_steps)
        if bool(episode["success"]):
            expected_done[-1] = True
        if seen_done[episode_num] != expected_done:
            raise ValueError(f"episode {episode_num}: done/success mismatch")
        if seen_source[episode_num] != {str(episode["source_file"])}:
            raise ValueError(f"episode {episode_num}: source_file mismatch")

    if record_count != int(manifest["num_records"]):
        raise ValueError(
            f"record count mismatch: records={record_count}, manifest={manifest['num_records']}"
        )

    video_root: Path | None = None
    if args.video_root is not None:
        video_root = args.video_root.resolve()
    else:
        source_root = Path(str(manifest["source_root"]))
        if source_root.is_dir():
            video_root = source_root

    missing_videos: list[str] = []
    present_videos = 0
    if video_root is not None:
        for episode in manifest["episodes"]:
            relative_video = str(episode["source_video_relative_path"])
            if (video_root / relative_video).is_file():
                present_videos += 1
            else:
                missing_videos.append(relative_video)
    elif args.require_complete_videos:
        raise FileNotFoundError(
            "Cannot verify videos: pass --video-root or audit where manifest source_root exists"
        )

    if args.require_complete_videos and missing_videos:
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} exact-stem videos under {video_root}; "
            f"first={missing_videos[0]}"
        )

    report = {
        "format": "groot_n15_pq3_trajectory_audit_v1",
        "trajectory_records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "num_episodes": len(episode_manifest),
        "num_records": record_count,
        "schema_finite_step_audit_passed": True,
        "video_root": str(video_root) if video_root is not None else None,
        "video_inventory_checked": video_root is not None,
        "num_videos_present": present_videos if video_root is not None else None,
        "num_videos_missing": len(missing_videos) if video_root is not None else None,
        "missing_video_relative_paths": missing_videos,
        "complete_video_inventory_required": bool(args.require_complete_videos),
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else records_path.parent / AUDIT_NAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[audit] episodes={len(episode_manifest)} records={record_count}", flush=True)
    if video_root is not None:
        print(
            f"[audit] videos present={present_videos} "
            f"missing={len(missing_videos)} root={video_root}",
            flush=True,
        )
    print(f"[audit] report={output_path}", flush=True)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export trusted PQ3 PKLs")
    export.add_argument("--input-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--trust-pkl", action="store_true")
    export.add_argument("--allow-partial-inventory", action="store_true")
    export.add_argument("--progress-every", type=int, default=10)
    export.set_defaults(func=export_trajectories)

    audit = subparsers.add_parser("audit", help="Audit exported JSONL and manifest")
    audit.add_argument("--trajectory-records-path", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--video-root", type=Path, default=None)
    audit.add_argument("--require-complete-videos", action="store_true")
    audit.add_argument("--output", type=Path, default=None)
    audit.set_defaults(func=audit_trajectories)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
