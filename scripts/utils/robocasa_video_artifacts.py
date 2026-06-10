"""RoboCasa rollout video artifact helpers.

Keep HTTP eval outputs aligned with the upstream GR00T/ZMQ convention:
episode-level mp4 files with ``_s{0|1}`` suffixes plus ``per_episode.tsv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import uuid
from typing import Iterable


@dataclass(frozen=True)
class EpisodeVideoRecord:
    episode_idx: int
    success: bool
    language: str = ""
    video_path: str | None = None


def episode_video_path(video_dir: str | Path, *, success: bool, uid: str | None = None) -> Path:
    token = uid or str(uuid.uuid4())
    return Path(video_dir) / f"{token}_s{int(bool(success))}.mp4"


def temp_episode_video_path(video_dir: str | Path, episode_idx: int) -> Path:
    return Path(video_dir) / f".episode_{episode_idx}.tmp.mp4"


def sanitize_tsv_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[\t\r\n]+", " ", text).strip()


def write_per_episode_tsv(
    video_dir: str | Path,
    records: Iterable[EpisodeVideoRecord],
) -> Path:
    out_dir = Path(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "per_episode.tsv"
    lines = ["episode_idx\tsuccess\tlanguage"]
    for record in records:
        lines.append(
            f"{int(record.episode_idx)}\t{int(bool(record.success))}\t"
            f"{sanitize_tsv_text(record.language)}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def save_frames_as_episode_video(
    frames: list,
    video_dir: str | Path,
    *,
    success: bool,
    fps: int,
) -> Path | None:
    if not frames:
        return None
    import imageio

    path = episode_video_path(video_dir, success=success)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path
