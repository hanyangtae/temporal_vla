"""RoboCasa LeRobot v2.1 dataset reader without the ``lerobot`` library.

Our ``lerobot`` submodule is v3.0-only and rejects v2.1 datasets at load time
(``BackwardCompatibilityError``). GR00T finetune expects v2.1 directly. This
helper reads parquet + per-camera mp4 + meta jsonl files using only pyarrow
and pyav, exposing just the metadata bits Phase 1 / Stage 0 need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


class RoboCasaV21Reader:
    """Single LeRobot v2.1 dataset root reader.

    Layout::

        {root}/meta/info.json            ← chunks_size, data_path, video_path
        {root}/meta/episodes.jsonl       ← per-episode length + tasks
        {root}/meta/tasks.jsonl          ← task_index → task_text
        {root}/data/chunk-{cc}/episode_{ee}.parquet
        {root}/videos/chunk-{cc}/{video_key}/episode_{ee}.mp4
    """

    def __init__(self, lerobot_root: Path | str):
        self.root = Path(lerobot_root)
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path_tpl = self.info["data_path"]
        self.video_path_tpl = self.info["video_path"]

        self.episodes: list[dict] = []
        with open(self.root / "meta" / "episodes.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.episodes.append(json.loads(line))
        self.episodes.sort(key=lambda e: int(e["episode_index"]))

        self.task_text_by_index: dict[int, str] = {}
        with open(self.root / "meta" / "tasks.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.task_text_by_index[int(d["task_index"])] = d["task"]

        # Cumulative ``dataset_from_index`` per episode — Phase1EpisodicDataset
        # 의 캐시 lookup 키와 일치시켜야 한다.
        self._ep_from: dict[int, int] = {}
        cum = 0
        for ep in self.episodes:
            self._ep_from[int(ep["episode_index"])] = cum
            cum += int(ep["length"])
        self.total_frames = cum
        self.total_episodes = len(self.episodes)

    # ── episode metadata ────────────────────────────────────────────────

    def episode_from_index(self, ep_idx: int) -> int:
        return self._ep_from[int(ep_idx)]

    def episode_length(self, ep_idx: int) -> int:
        return int(next(e for e in self.episodes if int(e["episode_index"]) == ep_idx)["length"])

    def episode_task_text(self, ep_idx: int) -> str:
        ep = next(e for e in self.episodes if int(e["episode_index"]) == ep_idx)
        tasks = ep.get("tasks") or []
        if tasks:
            return tasks[0]
        parquet, _ = self.episode_paths(ep_idx)
        table = pq.read_table(str(parquet), columns=["task_index"])
        ti = int(table.column("task_index")[0].as_py())
        return self.task_text_by_index.get(ti, "")

    def episode_paths(self, ep_idx: int) -> tuple[Path, dict[str, Path]]:
        chunk = ep_idx // self.chunks_size
        parquet = self.root / self.data_path_tpl.format(
            episode_chunk=chunk, episode_index=ep_idx,
        )
        video_dir = self.root / "videos" / f"chunk-{chunk:03d}"
        video_paths: dict[str, Path] = {}
        if video_dir.is_dir():
            for sub in video_dir.iterdir():
                if not sub.is_dir():
                    continue
                p = sub / f"episode_{ep_idx:06d}.mp4"
                if p.exists():
                    video_paths[sub.name] = p
        return parquet, video_paths

    # ── frame iteration (with images decoded via pyav) ─────────────────

    def iter_episode_frames(
        self, ep_idx: int,
    ) -> Iterator[tuple[int, int, str, dict[str, np.ndarray]]]:
        """Yield ``(abs_frame_idx, frame_in_episode, task_text, images_uint8_hwc)``
        for every frame of one episode.

        ``images_uint8_hwc`` keyed by the LeRobot column name (e.g.
        ``observation.images.robot0_agentview_left``).
        """
        import av  # 지연 import — 메타데이터만 필요할 때 (Phase 1) 는 av 가 없어도 OK.

        parquet, video_paths = self.episode_paths(ep_idx)
        if not parquet.exists():
            raise FileNotFoundError(f"Missing parquet: {parquet}")
        if not video_paths:
            raise FileNotFoundError(f"Missing videos for episode {ep_idx} under {self.root}")

        task_text = self.episode_task_text(ep_idx)
        table = pq.read_table(str(parquet), columns=["frame_index"])
        n_frames = table.num_rows
        ep_from = self._ep_from[ep_idx]

        cam_frames: dict[str, list[np.ndarray]] = {}
        for cam_key, mp4_path in video_paths.items():
            frames = self._decode_video_frames(mp4_path)
            if len(frames) < n_frames:
                raise RuntimeError(
                    f"Decoded {len(frames)} frames from {mp4_path}, expected ≥ {n_frames}"
                )
            cam_frames[cam_key] = frames

        for t in range(n_frames):
            images = {cam: cam_frames[cam][t] for cam in cam_frames}
            yield ep_from + t, t, task_text, images

    @staticmethod
    def _decode_video_frames(mp4_path: Path) -> list[np.ndarray]:
        import av
        container = av.open(str(mp4_path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            frames: list[np.ndarray] = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24")
                if arr.shape[:2] != (256, 256):
                    arr = np.array(Image.fromarray(arr).resize((256, 256)))
                frames.append(arr)
            return frames
        finally:
            container.close()

    # ── lerobot.meta-compatible dict view (Phase1 dataset 호환) ────────

    def lerobot_meta_episodes(self) -> dict[int, dict]:
        """Return ``{episode_index: {tasks, length, dataset_from_index, dataset_to_index}}``.

        lerobot v3 ``meta.episodes`` 와 대응되는 최소 필드만 노출.
        """
        out: dict[int, dict] = {}
        for ep in self.episodes:
            idx = int(ep["episode_index"])
            ep_from = self._ep_from[idx]
            length = int(ep["length"])
            out[idx] = {
                "episode_index": idx,
                "tasks": ep.get("tasks", []),
                "length": length,
                "dataset_from_index": ep_from,
                "dataset_to_index": ep_from + length,
            }
        return out
