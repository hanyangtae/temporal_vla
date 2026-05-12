"""Phase 1 ProgressPredictor 학습용 데이터셋 — RoboCasa LeRobot v2.1 호환 변형.

`Phase1EpisodicDataset` (`src/datasets/phase1_dataset.py`) 의 동등한 ``__getitem__``
계약을 제공하지만 ``lerobot.datasets.LeRobotDataset`` 을 import 하지 않는다.
대신 ``RoboCasaV21Reader`` 가 직접 parquet/jsonl 을 읽어 episode metadata 를 노출.

캐시 (Stage 0 산출물) 가 반드시 미리 존재해야 한다 — CLIP fallback 경로 없음.

샘플 형식 (Phase1EpisodicDataset 와 동일)::

    {
        "z_seq": Tensor[T, D],
        "targets": Tensor[T, 1],   # t / max(T-1, 1)
        "pred_mask": BoolTensor[T],  # dissimilarity sampling 으로 선택된 frame
        "ep_len": int,
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.datasets.robocasa_v21_reader import RoboCasaV21Reader


class Phase1V21EpisodicDataset(Dataset):
    """RoboCasa v2.1 native episodic dataset for Phase 1.

    Args:
        lerobot_root: ``data/robocasa/v1.0/.../<Task>/<date>/lerobot`` 의 한 task root.
        embed_cache_path: Stage 0 가 만든 ``embeddings.pt`` (``{abs_frame_idx: tensor[D]}``).
        episode_indices: 사용할 episode index 리스트. None 이면 cache 에 들어있는 episode 전부.
        window_size: dissimilarity sampling sub-trajectory 길이 (VITA=8).
        max_windows_per_episode: top-k window 수 (VITA=8).
        max_ep_len: episode 길이 상한 (padding 시 max length).
    """

    # RoboCasa atomic pretrain 의 episode 분포(p99=485, max=618)에 맞춰 default 조정.
    # VITA 원본(BridgeData) 은 p_max=120 였음 — 데이터셋에 따라 외부에서 override 권장.
    DEFAULT_MAX_EP_LEN = 485

    def __init__(
        self,
        lerobot_root: str | Path,
        embed_cache_path: str | Path,
        episode_indices: Optional[list[int]] = None,
        window_size: int = 8,
        max_windows_per_episode: int = 8,
        max_ep_len: int = DEFAULT_MAX_EP_LEN,
    ):
        self.lerobot_root = Path(lerobot_root)
        self.embed_cache_path = Path(embed_cache_path)
        self.window_size = window_size
        self.max_windows_per_episode = max_windows_per_episode
        self.max_ep_len = max_ep_len

        if not self.embed_cache_path.exists():
            raise FileNotFoundError(
                f"Eagle pre-LLM cache not found: {self.embed_cache_path}. "
                f"Run scripts/extract/extract_eagle_pre_llm_robocasa.py first."
            )

        self.reader = RoboCasaV21Reader(self.lerobot_root)

        # cache load — {abs_frame_idx: tensor[D]}
        self._embeddings: dict[int, torch.Tensor] = torch.load(
            str(self.embed_cache_path), map_location="cpu", weights_only=True,
        )
        cached_indices = set(int(k) for k in self._embeddings.keys())

        # episode_indices 결정
        all_eps = self.reader.lerobot_meta_episodes()
        if episode_indices is None:
            episode_indices = sorted(
                ep_idx for ep_idx, ep in all_eps.items()
                if ep["dataset_from_index"] in cached_indices
            )
        else:
            episode_indices = sorted(
                ep_idx for ep_idx in episode_indices
                if ep_idx in all_eps and all_eps[ep_idx]["dataset_from_index"] in cached_indices
            )

        # Episode boundary 수집: (ep_idx, ep_from, ep_to)
        self._episode_ranges: list[tuple[int, int, int]] = []
        for ep_idx in episode_indices:
            ep = all_eps[ep_idx]
            self._episode_ranges.append(
                (ep_idx, ep["dataset_from_index"], ep["dataset_to_index"])
            )

        # 에피소드별 (ep_idx, ep_from, ep_length, full_ep_length, pred_mask) 빌드.
        # full_ep_length 를 따로 두는 이유: targets = t / (full_ep_length - 1) 로 계산해야
        # truncate (ep_length < full_ep_length) 된 episode 의 frame 들도 올바른 progress 라벨
        # 을 받게 된다. ep_length 기준으로 계산하면 잘린 ep 의 마지막 frame 이 progress=1.0
        # 으로 잘못 라벨링됨 (실제로는 < 1.0).
        self._episode_data: list[tuple[int, int, int, int, torch.Tensor]] = []
        for ep_idx, ep_from, ep_to in self._episode_ranges:
            full_ep_length = ep_to - ep_from
            ep_length = min(full_ep_length, self.max_ep_len)
            if ep_length < self.window_size:
                continue
            pred_mask = self._compute_pred_mask(ep_from, ep_length)
            self._episode_data.append((ep_idx, ep_from, ep_length, full_ep_length, pred_mask))

    # ──────────────────────────────────────────────
    # Dissimilarity sampling (VITA Eq. 5) — 동일 알고리즘
    # ──────────────────────────────────────────────

    def _compute_pred_mask(self, ep_from: int, ep_length: int) -> torch.Tensor:
        stride = 1
        candidates = list(range(0, ep_length - self.window_size + 1, stride))
        if not candidates:
            return torch.zeros(ep_length, dtype=torch.bool)

        if len(candidates) <= self.max_windows_per_episode:
            selected_starts = candidates
        else:
            feats = torch.stack([
                torch.stack([self._embeddings[ep_from + i]
                             for i in range(s, s + self.window_size)]).mean(0)
                for s in candidates
            ])
            diff = feats.unsqueeze(0) - feats.unsqueeze(1)
            scores = (diff ** 2).sum(-1).sum(dim=1)
            k = min(self.max_windows_per_episode, len(candidates))
            selected_starts = [candidates[i] for i in scores.topk(k).indices.tolist()]

        pred_mask = torch.zeros(ep_length, dtype=torch.bool)
        for s in selected_starts:
            end = min(s + self.window_size, ep_length)
            pred_mask[s:end] = True
        return pred_mask

    # ──────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._episode_data)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, ep_from, ep_length, full_ep_length, pred_mask = self._episode_data[idx]
        z_seq = torch.stack([
            self._embeddings[ep_from + t] for t in range(ep_length)
        ])
        # progress label 은 full episode 기준 — truncate 되어도 frame t 의 라벨은
        # t / (full_ep_length - 1) (마지막 잘린 frame 도 정확한 progress < 1).
        targets = (
            torch.arange(ep_length, dtype=torch.float32) / max(full_ep_length - 1, 1)
        ).unsqueeze(-1)
        return {
            "z_seq": z_seq,
            "targets": targets,
            "pred_mask": pred_mask,
            "ep_len": ep_length,
        }
