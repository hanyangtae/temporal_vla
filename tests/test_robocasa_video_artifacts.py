from __future__ import annotations

from pathlib import Path

from scripts.utils.robocasa_video_artifacts import (
    EpisodeVideoRecord,
    episode_video_path,
    write_per_episode_tsv,
)


def test_episode_video_path_matches_zmq_success_suffix(tmp_path: Path):
    assert episode_video_path(tmp_path, success=True, uid="abc").name == "abc_s1.mp4"
    assert episode_video_path(tmp_path, success=False, uid="def").name == "def_s0.mp4"


def test_per_episode_tsv_matches_zmq_columns(tmp_path: Path):
    write_per_episode_tsv(
        tmp_path,
        [
            EpisodeVideoRecord(episode_idx=0, success=True, language="open fridge"),
            EpisodeVideoRecord(episode_idx=1, success=False, language="close\tfridge"),
        ],
    )

    assert (tmp_path / "per_episode.tsv").read_text().splitlines() == [
        "episode_idx\tsuccess\tlanguage",
        "0\t1\topen fridge",
        "1\t0\tclose fridge",
    ]
