#!/usr/bin/env python3
"""Package GR00T waypoint frames into a portable Stage 3 media bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import imageio.v2 as imageio_v2
from PIL import Image, __version__ as pillow_version


BUNDLE_FORMAT = "event_sae_stage3_media_v4"
DEFAULT_PAPER_ENV_STEP_OFFSETS = [-4, -2, 0, 2, 4]
FIRST_VIDEO_FRAME_ENV_STEP = 1
VIEW_LAYOUT = {
    "side_0": {
        "view_index": 0,
        "source_camera": "robot0_agentview_left",
    },
    "side_1": {
        "view_index": 1,
        "source_camera": "robot0_agentview_right",
    },
    "wrist_0": {
        "view_index": 2,
        "source_camera": "robot0_eye_in_hand",
    },
}
DEFAULT_VIEW_NAME = "side_0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class VideoTimeline:
    num_records: int
    n_action_steps: int
    steps_per_render: int
    control_freq_hz: float
    encoded_video_fps: float

    def __post_init__(self) -> None:
        if self.num_records <= 0:
            raise ValueError("num_records must be positive")
        if self.n_action_steps <= 0 or self.steps_per_render <= 0:
            raise ValueError("timeline step counts must be positive")
        if self.n_action_steps < self.steps_per_render:
            raise ValueError("n_action_steps must be >= steps_per_render")
        if self.control_freq_hz <= 0 or self.encoded_video_fps <= 0:
            raise ValueError("timeline frequencies must be positive")

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return -(-numerator // denominator)

    @property
    def expected_num_frames(self) -> int:
        return self._ceil_div(
            self.num_records * self.n_action_steps,
            self.steps_per_render,
        )

    def record_env_step(self, record_index: int) -> int:
        if not 0 <= record_index < self.num_records:
            raise IndexError(
                f"record_index={record_index} outside [0,{self.num_records})"
            )
        return record_index * self.n_action_steps

    def frame_env_step(self, frame_index: int) -> int:
        if not 0 <= frame_index < self.expected_num_frames:
            raise IndexError(
                f"frame_index={frame_index} outside [0,{self.expected_num_frames})"
            )
        return FIRST_VIDEO_FRAME_ENV_STEP + frame_index * self.steps_per_render

    def nearest_frame_for_env_step(self, env_step: int) -> int:
        relative_step = env_step - FIRST_VIDEO_FRAME_ENV_STEP
        if relative_step <= 0:
            return 0
        frame_index, remainder = divmod(relative_step, self.steps_per_render)
        if remainder * 2 >= self.steps_per_render:
            frame_index += 1
        return min(frame_index, self.expected_num_frames - 1)

    def record_to_nearest_frame(self, record_index: int) -> int:
        return self.nearest_frame_for_env_step(self.record_env_step(record_index))

    def frame_offsets_for_env_step_offsets(
        self,
        env_step_offsets: list[int],
    ) -> list[int]:
        if any(offset % self.steps_per_render != 0 for offset in env_step_offsets):
            raise ValueError(
                "paper env-step offsets must be divisible by steps_per_render"
            )
        return [offset // self.steps_per_render for offset in env_step_offsets]

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "mapping": "pre_action_record_to_nearest_rendered_frame_v2",
            "num_records": self.num_records,
            "n_action_steps": self.n_action_steps,
            "steps_per_render": self.steps_per_render,
            "expected_num_frames": self.expected_num_frames,
            "first_video_frame_env_step": FIRST_VIDEO_FRAME_ENV_STEP,
            "control_freq_hz": self.control_freq_hz,
            "encoded_video_fps": self.encoded_video_fps,
            "sim_seconds_per_action_step": 1.0 / self.control_freq_hz,
            "sim_seconds_per_video_frame": (
                self.steps_per_render / self.control_freq_hz
            ),
            "playback_seconds_per_video_frame": 1.0 / self.encoded_video_fps,
        }


def fit_frame_window(
    anchor_frame_index: int,
    frame_offsets: list[int],
    num_frames: int,
) -> tuple[list[int], list[int], int]:
    requested = [anchor_frame_index + offset for offset in frame_offsets]
    if not requested:
        return [], [], 0
    if max(requested) - min(requested) >= num_frames:
        raise ValueError("frame window span exceeds episode length")
    shift = 0
    if max(requested) >= num_frames:
        shift -= max(requested) - (num_frames - 1)
    if min(requested) + shift < 0:
        shift += -(min(requested) + shift)
    selected = [index + shift for index in requested]
    if len(selected) != len(set(selected)):
        raise ValueError("frame window contains duplicate indices")
    if any(index < 0 or index >= num_frames for index in selected):
        raise ValueError("frame window is outside the episode")
    if anchor_frame_index not in selected:
        raise ValueError("frame window does not contain the keyframe anchor")
    return selected, requested, shift


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"video path must be relative and contained: {value}")
    return path


def save_frame_jpeg(
    frame: Any,
    path: Path,
    quality: int,
    scene_height_pixels: int,
    view_name: str,
    view_width_pixels: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.jpg")
    image = Image.fromarray(frame).convert("RGB")
    source_width, source_height = image.size
    if source_height < scene_height_pixels:
        raise ValueError(
            f"source frame height={source_height} is smaller than "
            f"scene_height_pixels={scene_height_pixels}"
        )
    crop_top_pixels = source_height - scene_height_pixels
    image = image.crop((0, crop_top_pixels, source_width, source_height))
    expected_source_width = view_width_pixels * len(VIEW_LAYOUT)
    if source_width != expected_source_width:
        raise ValueError(
            f"source frame width={source_width} does not match the expected "
            f"three-view montage width={expected_source_width}"
        )
    view = VIEW_LAYOUT[view_name]
    view_index = int(view["view_index"])
    crop_left_pixels = view_index * view_width_pixels
    crop_right_pixels = source_width - crop_left_pixels - view_width_pixels
    image = image.crop(
        (
            crop_left_pixels,
            0,
            crop_left_pixels + view_width_pixels,
            scene_height_pixels,
        )
    )
    image.save(temporary, format="JPEG", quality=quality, optimize=True)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width": image.width,
        "height": image.height,
        "source_width": source_width,
        "source_height": source_height,
        "crop_top_pixels": crop_top_pixels,
        "crop_left_pixels": crop_left_pixels,
        "crop_right_pixels": crop_right_pixels,
        "view_name": view_name,
        "view_index": view_index,
        "source_camera": str(view["source_camera"]),
        "view_width_pixels": view_width_pixels,
    }


def selected_episode_ids(values: list[int] | None) -> set[int] | None:
    if not values:
        return None
    if len(values) != len(set(values)):
        raise ValueError("--episode-num contains duplicates")
    return set(values)


def package_media(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.waypoint_summary.resolve()
    trajectory_manifest_path = args.trajectory_manifest.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)
    trajectory_manifest = load_json(trajectory_manifest_path)
    summary_episodes = {
        int(episode["episode_num"]): episode for episode in summary["episodes"]
    }
    manifest_episodes = {
        int(episode["episode_num"]): episode
        for episode in trajectory_manifest["episodes"]
    }
    if set(summary_episodes) != set(manifest_episodes):
        missing = sorted(set(summary_episodes) - set(manifest_episodes))
        extra = sorted(set(manifest_episodes) - set(summary_episodes))
        raise ValueError(
            f"summary/manifest episode mismatch: missing={missing}, extra={extra}"
        )

    episode_filter = selected_episode_ids(args.episode_num)
    if episode_filter is not None:
        unknown = sorted(episode_filter - set(summary_episodes))
        if unknown:
            raise ValueError(f"Unknown episode ids: {unknown}")

    samples_path = output_dir / "samples.jsonl"
    sample_ids: set[str] = set()
    num_frames_written = 0
    total_frame_bytes = 0
    source_video_cache: dict[Path, tuple[str, int]] = {}
    sample_count = 0
    resolved_frame_offsets: list[int] | None = None
    with samples_path.open("w", encoding="utf-8") as samples_file:
        for episode_num in sorted(summary_episodes):
            if episode_filter is not None and episode_num not in episode_filter:
                continue
            episode = summary_episodes[episode_num]
            provenance = manifest_episodes[episode_num]
            relative_video = validate_relative_path(
                str(provenance["source_video_relative_path"])
            )
            video_path = video_root / relative_video
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"episode {episode_num}: video not found: {video_path}"
                )

            timeline = VideoTimeline(
                num_records=int(provenance["num_records"]),
                n_action_steps=int(provenance.get("n_action_steps", 1)),
                steps_per_render=int(provenance.get("steps_per_render", 1)),
                control_freq_hz=float(args.control_freq_hz),
                encoded_video_fps=float(provenance.get("video_fps", 20.0)),
            )
            if int(episode["num_steps"]) != timeline.num_records:
                raise ValueError(f"episode {episode_num}: num_steps mismatch")
            frame_offsets = timeline.frame_offsets_for_env_step_offsets(
                args.paper_env_step_offsets
            )
            if resolved_frame_offsets is None:
                resolved_frame_offsets = frame_offsets
            elif resolved_frame_offsets != frame_offsets:
                raise ValueError("episodes resolve to different video-frame offsets")

            reader = imageio_v2.get_reader(video_path)
            try:
                actual_num_frames = int(reader.count_frames())
                if actual_num_frames != timeline.expected_num_frames:
                    raise ValueError(
                        f"episode {episode_num}: video frames={actual_num_frames}, "
                        f"expected={timeline.expected_num_frames}"
                    )
                if video_path not in source_video_cache:
                    source_video_cache[video_path] = (
                        sha256_file(video_path),
                        video_path.stat().st_size,
                    )
                source_video_sha256, source_video_bytes = source_video_cache[video_path]

                for waypoint_rank, raw_waypoint_index in enumerate(
                    episode["waypoint_indices"]
                ):
                    waypoint_index = int(raw_waypoint_index)
                    waypoint_env_step = timeline.record_env_step(waypoint_index)
                    anchor_frame_index = timeline.record_to_nearest_frame(
                        waypoint_index
                    )
                    frame_indices, requested_frame_indices, window_shift = fit_frame_window(
                        anchor_frame_index,
                        frame_offsets,
                        actual_num_frames,
                    )
                    sample_id = f"ep{episode_num:04d}_wp{waypoint_rank:03d}_r{waypoint_index:04d}"
                    if sample_id in sample_ids:
                        raise ValueError(f"Duplicate sample id: {sample_id}")
                    sample_ids.add(sample_id)

                    frame_records: list[dict[str, Any]] = []
                    for position, frame_index in enumerate(frame_indices):
                        frame_env_step = timeline.frame_env_step(frame_index)
                        frame = reader.get_data(frame_index)
                        relative_output = (
                            Path("frames")
                            / sample_id
                            / f"frame_{position:02d}_v{frame_index:04d}_s{frame_env_step:04d}.jpg"
                        )
                        frame_info = save_frame_jpeg(
                            frame,
                            output_dir / relative_output,
                            args.jpeg_quality,
                            args.scene_height_pixels,
                            args.view_name,
                            args.view_width_pixels,
                        )
                        frame_info["path"] = str(relative_output)
                        frame_info.update(
                            {
                                "position": position,
                                "requested_paper_env_step_offset": int(
                                    args.paper_env_step_offsets[position]
                                ),
                                "requested_video_frame_index": int(
                                    requested_frame_indices[position]
                                ),
                                "video_frame_index": frame_index,
                                "video_frame_delta_from_anchor": (
                                    frame_index - anchor_frame_index
                                ),
                                "video_frame_env_step": frame_env_step,
                                "env_step_delta_from_waypoint": (
                                    frame_env_step - waypoint_env_step
                                ),
                                "sim_time_delta_seconds": (
                                    (frame_env_step - waypoint_env_step)
                                    / timeline.control_freq_hz
                                ),
                            }
                        )
                        frame_records.append(frame_info)
                        num_frames_written += 1
                        total_frame_bytes += int(frame_info["bytes"])

                    anchor_positions = [
                        index for index, frame_index in enumerate(frame_indices)
                        if frame_index == anchor_frame_index
                    ]
                    if len(anchor_positions) != 1:
                        raise ValueError(
                            f"sample {sample_id}: anchor must appear exactly once"
                        )
                    anchor_frame_env_step = timeline.frame_env_step(anchor_frame_index)
                    sample = {
                        "format": BUNDLE_FORMAT,
                        "sample_id": sample_id,
                        "episode_num": episode_num,
                        "task_id": int(episode["task_id"]),
                        "task_episode_idx": int(episode["task_episode_idx"]),
                        "task_description": str(episode["task_description"]),
                        "prompt_task_description": str(
                            episode.get(
                                "prompt_task_description",
                                episode["task_description"],
                            )
                        ),
                        "cell_id": str(provenance.get("cell_id", "")),
                        "task_family": str(provenance.get("task_family", "")),
                        "robocasa_task": str(provenance.get("robocasa_task", "")),
                        "success": bool(episode["success"]),
                        "waypoint_rank": waypoint_rank,
                        "waypoint_index": waypoint_index,
                        "waypoint_env_step": waypoint_env_step,
                        "num_records": timeline.num_records,
                        "paper_env_step_offsets": list(args.paper_env_step_offsets),
                        "requested_video_frame_offsets": frame_offsets,
                        "requested_video_frame_indices": requested_frame_indices,
                        "video_frame_indices": frame_indices,
                        "video_window_shift": window_shift,
                        "boundary_shift_category": (
                            "interior"
                            if window_shift == 0
                            else "start_shifted"
                            if window_shift > 0
                            else "end_shifted"
                        ),
                        "anchor_alignment": "nearest_rendered_frame_ties_later",
                        "anchor_video_frame_index": anchor_frame_index,
                        "anchor_video_frame_env_step": anchor_frame_env_step,
                        "anchor_env_step_error": (
                            anchor_frame_env_step - waypoint_env_step
                        ),
                        "anchor_sim_time_error_seconds": (
                            (anchor_frame_env_step - waypoint_env_step)
                            / timeline.control_freq_hz
                        ),
                        "anchor_frame_position": anchor_positions[0],
                        "video_timeline": timeline.to_dict(),
                        "source_file": str(provenance.get("source_file", "")),
                        "source_video_relative_path": str(relative_video),
                        "source_video_sha256": source_video_sha256,
                        "source_video_bytes": source_video_bytes,
                        "frames": frame_records,
                    }
                    samples_file.write(json.dumps(sample, sort_keys=True) + "\n")
                    samples_file.flush()
                    sample_count += 1
            finally:
                reader.close()

    if args.expected_samples is not None and sample_count != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} samples, wrote {sample_count}"
        )
    if resolved_frame_offsets is None:
        raise ValueError("No media samples were selected")
    manifest = {
        "format": BUNDLE_FORMAT,
        "code_revision": git_revision(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "waypoint_summary_path": str(summary_path),
        "waypoint_summary_sha256": sha256_file(summary_path),
        "trajectory_manifest_path": str(trajectory_manifest_path),
        "trajectory_manifest_sha256": sha256_file(trajectory_manifest_path),
        "video_root": str(video_root),
        "output_dir": str(output_dir),
        "control_freq_hz": float(args.control_freq_hz),
        "paper_env_step_offsets": list(args.paper_env_step_offsets),
        "video_frame_offsets": resolved_frame_offsets,
        "anchor_alignment": "nearest_rendered_frame_ties_later",
        "image_format": "jpeg",
        "jpeg_quality": args.jpeg_quality,
        "scene_height_pixels": args.scene_height_pixels,
        "view_name": args.view_name,
        "view_index": int(VIEW_LAYOUT[args.view_name]["view_index"]),
        "source_camera": str(VIEW_LAYOUT[args.view_name]["source_camera"]),
        "view_width_pixels": args.view_width_pixels,
        "source_montage_layout": [
            {
                "view_name": view_name,
                "view_index": int(view["view_index"]),
                "source_camera": str(view["source_camera"]),
            }
            for view_name, view in VIEW_LAYOUT.items()
        ],
        "crop_policy": "remove_top_overlay_then_select_single_view",
        "num_samples": sample_count,
        "num_frames": num_frames_written,
        "total_frame_bytes": total_frame_bytes,
        "selected_episode_nums": (
            sorted(episode_filter) if episode_filter is not None else None
        ),
        "environment": {
            "python": platform.python_version(),
            "imageio": imageio.__version__,
            "pillow": pillow_version,
        },
    }
    write_json(output_dir / "media_manifest.json", manifest)
    audit = audit_bundle(output_dir, expected_samples=sample_count)
    print(
        f"[package] samples={sample_count} frames={num_frames_written} "
        f"bytes={total_frame_bytes} output={output_dir}",
        flush=True,
    )
    return {"manifest": manifest, "audit": audit}


def audit_bundle(bundle_dir: Path, expected_samples: int | None = None) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest = load_json(bundle_dir / "media_manifest.json")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("Unexpected media bundle format")
    samples_path = bundle_dir / "samples.jsonl"
    sample_ids: set[str] = set()
    frame_paths: set[str] = set()
    samples = 0
    frames = 0
    bytes_total = 0
    null_anchor_positions = 0
    boundary_counts: dict[str, int] = {}
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            sample = json.loads(line)
            sample_id = str(sample["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"line {line_number}: duplicate sample_id={sample_id}")
            sample_ids.add(sample_id)
            if len(sample["frames"]) != len(manifest["video_frame_offsets"]):
                raise ValueError(f"sample {sample_id}: wrong frame count")
            if sample.get("anchor_frame_position") is None:
                null_anchor_positions += 1
            anchor_indices = [
                int(frame["video_frame_index"])
                for frame in sample["frames"]
                if int(frame["video_frame_index"])
                == int(sample["anchor_video_frame_index"])
            ]
            if len(anchor_indices) != 1:
                raise ValueError(
                    f"sample {sample_id}: anchor must appear exactly once"
                )
            category = str(sample["boundary_shift_category"])
            boundary_counts[category] = boundary_counts.get(category, 0) + 1
            for frame in sample["frames"]:
                if int(frame["height"]) != int(manifest["scene_height_pixels"]):
                    raise ValueError(
                        f"frame {frame['path']}: unexpected output height"
                    )
                if (
                    int(frame["source_height"])
                    - int(frame["crop_top_pixels"])
                    != int(manifest["scene_height_pixels"])
                ):
                    raise ValueError(
                        f"frame {frame['path']}: invalid vertical crop provenance"
                    )
                if int(frame["width"]) != int(manifest["view_width_pixels"]):
                    raise ValueError(
                        f"frame {frame['path']}: unexpected output width"
                    )
                if int(frame["view_width_pixels"]) != int(
                    manifest["view_width_pixels"]
                ):
                    raise ValueError(f"frame {frame['path']}: view width mismatch")
                if str(frame["view_name"]) != str(manifest["view_name"]):
                    raise ValueError(f"frame {frame['path']}: view name mismatch")
                if str(frame["source_camera"]) != str(manifest["source_camera"]):
                    raise ValueError(f"frame {frame['path']}: source camera mismatch")
                if int(frame["view_index"]) != int(manifest["view_index"]):
                    raise ValueError(f"frame {frame['path']}: view index mismatch")
                if (
                    int(frame["crop_left_pixels"])
                    + int(frame["width"])
                    + int(frame["crop_right_pixels"])
                    != int(frame["source_width"])
                ):
                    raise ValueError(
                        f"frame {frame['path']}: invalid horizontal crop provenance"
                    )
                if int(frame["crop_left_pixels"]) != (
                    int(frame["view_index"]) * int(manifest["view_width_pixels"])
                ):
                    raise ValueError(
                        f"frame {frame['path']}: crop does not match view index"
                    )
                relative_path = validate_relative_path(str(frame["path"]))
                if str(relative_path) in frame_paths:
                    raise ValueError(f"duplicate frame path: {relative_path}")
                frame_paths.add(str(relative_path))
                path = bundle_dir / relative_path
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.stat().st_size != int(frame["bytes"]):
                    raise ValueError(f"frame size mismatch: {relative_path}")
                if sha256_file(path) != frame["sha256"]:
                    raise ValueError(f"frame hash mismatch: {relative_path}")
                bytes_total += path.stat().st_size
                frames += 1
            samples += 1

    required_samples = (
        expected_samples if expected_samples is not None else int(manifest["num_samples"])
    )
    if samples != required_samples:
        raise ValueError(f"samples={samples}, expected={required_samples}")
    if frames != int(manifest["num_frames"]):
        raise ValueError(f"frames={frames}, manifest={manifest['num_frames']}")
    if bytes_total != int(manifest["total_frame_bytes"]):
        raise ValueError(
            f"frame bytes={bytes_total}, manifest={manifest['total_frame_bytes']}"
        )
    if null_anchor_positions:
        raise ValueError(
            f"null anchor frame positions are not allowed: {null_anchor_positions}"
        )
    report = {
        "format": BUNDLE_FORMAT,
        "passed": True,
        "num_samples": samples,
        "num_frames": frames,
        "total_frame_bytes": bytes_total,
        "num_null_anchor_frame_positions": null_anchor_positions,
        "boundary_counts": dict(sorted(boundary_counts.items())),
    }
    write_json(bundle_dir / "packaging_audit.json", report)
    print(
        f"[audit] samples={samples} frames={frames} "
        f"null_anchor_positions={null_anchor_positions} passed=true",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package", help="Extract a media bundle")
    package.add_argument("--waypoint-summary", type=Path, required=True)
    package.add_argument("--trajectory-manifest", type=Path, required=True)
    package.add_argument("--video-root", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument(
        "--paper-env-step-offsets",
        type=int,
        nargs="+",
        default=DEFAULT_PAPER_ENV_STEP_OFFSETS,
        help="Paper rollout-step offsets; resolved to source video-frame offsets",
    )
    package.add_argument("--control-freq-hz", type=float, default=20.0)
    package.add_argument(
        "--scene-height-pixels",
        type=int,
        default=256,
        help="Keep this many bottom pixels and remove the top text overlay",
    )
    package.add_argument(
        "--view-name",
        choices=tuple(VIEW_LAYOUT),
        default=DEFAULT_VIEW_NAME,
        help="Select one view from the horizontal source montage",
    )
    package.add_argument(
        "--view-width-pixels",
        type=int,
        default=256,
        help="Width of each source montage view",
    )
    package.add_argument("--episode-num", type=int, nargs="+", default=None)
    package.add_argument("--expected-samples", type=int, default=None)
    package.add_argument("--jpeg-quality", type=int, default=95)

    audit = subparsers.add_parser("audit", help="Verify a media bundle")
    audit.add_argument("--bundle-dir", type=Path, required=True)
    audit.add_argument("--expected-samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "package":
        if not 1 <= args.jpeg_quality <= 100:
            raise ValueError("--jpeg-quality must be in [1,100]")
        if args.control_freq_hz <= 0:
            raise ValueError("--control-freq-hz must be positive")
        if args.scene_height_pixels <= 0:
            raise ValueError("--scene-height-pixels must be positive")
        if args.view_width_pixels <= 0:
            raise ValueError("--view-width-pixels must be positive")
        package_media(args)
    else:
        audit_bundle(args.bundle_dir, expected_samples=args.expected_samples)


if __name__ == "__main__":
    main()
