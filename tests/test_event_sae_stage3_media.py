import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "event_sae"
        / "stage3_media.py"
    )
    spec = importlib.util.spec_from_file_location("stage3_media", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MEDIA = _load_module()


def test_video_timeline_matches_groot_contract() -> None:
    timeline = MEDIA.VideoTimeline(
        num_records=144,
        n_action_steps=5,
        steps_per_render=2,
        control_freq_hz=20.0,
        encoded_video_fps=20.0,
    )

    assert timeline.expected_num_frames == 360
    assert timeline.record_env_step(1) == 5
    assert timeline.frame_env_step(2) == 5
    assert timeline.record_to_nearest_frame(1) == 2
    assert timeline.record_to_nearest_frame(2) == 5
    assert timeline.frame_env_step(5) == 11
    assert timeline.frame_offsets_for_env_step_offsets([-4, -2, 0, 2, 4]) == [
        -2,
        -1,
        0,
        1,
        2,
    ]


def test_fit_frame_window_shifts_at_start_and_preserves_anchor() -> None:
    selected, requested, shift = MEDIA.fit_frame_window(
        anchor_frame_index=0,
        frame_offsets=[-2, -1, 0, 1, 2],
        num_frames=10,
    )

    assert requested == [-2, -1, 0, 1, 2]
    assert selected == [0, 1, 2, 3, 4]
    assert shift == 2
    assert 0 in selected


def test_fit_frame_window_shifts_at_end_and_preserves_anchor() -> None:
    selected, _, shift = MEDIA.fit_frame_window(
        anchor_frame_index=9,
        frame_offsets=[-2, -1, 0, 1, 2],
        num_frames=10,
    )

    assert selected == [5, 6, 7, 8, 9]
    assert shift == -2
    assert 9 in selected


def test_fit_frame_window_rejects_too_short_episode() -> None:
    with pytest.raises(ValueError, match="span exceeds"):
        MEDIA.fit_frame_window(
            anchor_frame_index=1,
            frame_offsets=[-2, -1, 0, 1, 2],
            num_frames=4,
        )


def test_relative_video_path_rejects_escape() -> None:
    with pytest.raises(ValueError, match="relative"):
        MEDIA.validate_relative_path("../other/video.mp4")


def test_package_and_audit_synthetic_video(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_path = video_root / "cell" / "episode.mp4"
    video_path.parent.mkdir(parents=True)
    frames = []
    for index in range(10):
        banner = np.full((4, 96, 3), fill_value=255, dtype=np.uint8)
        side_0 = np.full((32, 32, 3), fill_value=index * 20, dtype=np.uint8)
        side_1 = np.full((32, 32, 3), fill_value=240, dtype=np.uint8)
        wrist_0 = np.full((32, 32, 3), fill_value=120, dtype=np.uint8)
        scene = np.concatenate([side_0, side_1, wrist_0], axis=1)
        frames.append(np.concatenate([banner, scene], axis=0))
    imageio.mimwrite(video_path, frames, fps=10, macro_block_size=1)

    waypoint_summary = {
        "episodes": [
            {
                "episode_num": 7,
                "task_id": 8,
                "task_episode_idx": 0,
                "task_description": "Open the left drawer",
                "success": True,
                "num_steps": 4,
                "waypoint_indices": [2],
            }
        ]
    }
    trajectory_manifest = {
        "episodes": [
            {
                "episode_num": 7,
                "num_records": 4,
                "n_action_steps": 5,
                "steps_per_render": 2,
                "source_video_relative_path": "cell/episode.mp4",
                "cell_id": "drawer_left",
                "task_family": "OpenDrawer",
                "robocasa_task": "PnPCounterToCab",
                "source_file": "episode.pkl",
            }
        ]
    }
    summary_path = tmp_path / "waypoint_summary.json"
    manifest_path = tmp_path / "trajectory_manifest.json"
    summary_path.write_text(json.dumps(waypoint_summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(trajectory_manifest), encoding="utf-8")
    output_dir = tmp_path / "bundle"

    result = MEDIA.package_media(
        Namespace(
            waypoint_summary=summary_path,
            trajectory_manifest=manifest_path,
            video_root=video_root,
            output_dir=output_dir,
            paper_env_step_offsets=[-4, -2, 0, 2, 4],
            control_freq_hz=20.0,
            scene_height_pixels=32,
            view_name="side_0",
            view_width_pixels=32,
            episode_num=None,
            expected_samples=1,
            jpeg_quality=95,
        )
    )

    assert result["audit"]["passed"] is True
    assert result["manifest"]["num_samples"] == 1
    assert result["manifest"]["num_frames"] == 5
    assert result["manifest"]["video_frame_offsets"] == [-2, -1, 0, 1, 2]
    sample = json.loads((output_dir / "samples.jsonl").read_text())
    assert sample["waypoint_env_step"] == 10
    assert sample["video_frame_indices"] == [3, 4, 5, 6, 7]
    assert sample["anchor_video_frame_index"] == 5
    assert sample["anchor_env_step_error"] == 1
    assert sample["anchor_frame_position"] == 2
    assert sample["frames"][0]["source_height"] == 36
    assert sample["frames"][0]["source_width"] == 96
    assert sample["frames"][0]["crop_top_pixels"] == 4
    assert sample["frames"][0]["crop_left_pixels"] == 0
    assert sample["frames"][0]["crop_right_pixels"] == 64
    assert sample["frames"][0]["view_name"] == "side_0"
    assert sample["frames"][0]["source_camera"] == "robot0_agentview_left"
    assert sample["frames"][0]["width"] == 32
    assert sample["frames"][0]["height"] == 32
    assert all((output_dir / frame["path"]).is_file() for frame in sample["frames"])

    audit = MEDIA.audit_bundle(output_dir, expected_samples=1)
    assert audit["num_frames"] == 5
