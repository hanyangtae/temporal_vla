"""
[DEPRECATED] Phase 1 Window 기반 데이터셋.

episodic 방식(Phase1EpisodicDataset) 대비 val/unseen Spearman이 크게 낮아 deprecated.
결과 비교 (2026-04-09):
    episodic + mean: val Spearman 0.819, unseen 0.702
    window + mean:   val Spearman 0.343, unseen 0.411
    window + concat: val Spearman 0.495, unseen 0.469

현재 사용 코드: src/datasets/phase1_dataset.py (Phase1EpisodicDataset)
"""

import os
from typing import Optional, Literal
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _compute_window_feats(
    candidates: list,
    ep_from: int,
    window_size: int,
    embeddings: dict,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Dissimilarity scoring용 window feature 계산.

    Args:
        mode: "mean" — window mean embedding [N, D] (VITA 원본)
              "concat" — window frame concat [N, window_size*D]
    """
    if mode == "mean":
        return torch.stack([
            torch.stack([embeddings[ep_from + i] for i in range(s, s + window_size)]).mean(0)
            for s in candidates
        ])
    else:  # concat
        return torch.stack([
            torch.cat([embeddings[ep_from + i] for i in range(s, s + window_size)])
            for s in candidates
        ])


class Phase1WindowDataset(Dataset):
    """
    [DEPRECATED] Phase 1 ProgressPredictor 학습용 데이터셋 (window 방식).
    Spearman 성능이 episodic 방식 대비 낮아 deprecated.
    Phase1EpisodicDataset (full trajectory) 사용 권장.

    VITA 원본과 동일하게 dissimilarity sampling으로 선택된
    w_tr 크기의 sub-trajectory를 개별 샘플로 반환.
    각 window는 독립적으로 θ_0에서 시작 → MAML unroll depth = w_tr.

    학습: window 단위 (8 frames) → gradient vanishing 없음
    추론: full trajectory sequential TTT (eval 스크립트에서 처리)

    total samples = n_episodes * k_windows_per_episode
    """

    MAX_EP_LEN = 120

    def __init__(
        self,
        data_root: str,
        repo_id: str = "FedorX8/bridge_v2_lerobot",
        window_size: int = 8,
        max_windows_per_episode: int = 8,
        image_key: str = "observation.images.primary",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        embed_device: str = "cuda",
        embed_cache_path: Optional[str] = None,
        episode_indices: Optional[list] = None,
        start_episode: int = 0,
        max_episodes: Optional[int] = None,
        task_keywords: Optional[list] = None,
        dissimilarity_mode: Literal["mean", "concat"] = "mean",
    ):
        self.window_size = window_size
        self.max_windows_per_episode = max_windows_per_episode
        self.image_key = image_key
        self.dissimilarity_mode = dissimilarity_mode

        # LeRobotDataset 로드
        self.lerobot_ds = LeRobotDataset(repo_id=repo_id, root=data_root)
        meta = self.lerobot_ds.meta
        total_ep = meta.total_episodes

        if episode_indices is not None:
            indices = episode_indices
        else:
            end_ep = min(total_ep, start_episode + max_episodes) if max_episodes else total_ep
            indices = list(range(start_episode, end_ep))

        # task keyword 필터링
        if task_keywords is not None:
            filtered = []
            for ep_idx in indices:
                ep = meta.episodes[ep_idx]
                task_strs = ep.get("tasks", [])
                task_text = " ".join(task_strs).lower()
                if any(kw.lower() in task_text for kw in task_keywords):
                    filtered.append(ep_idx)
            print(f"Task filter {task_keywords}: {len(indices)} → {len(filtered)} episodes")
            indices = filtered

        self._n_episodes = len(indices)

        self._episode_ranges = []
        for ep_idx in indices:
            ep = meta.episodes[ep_idx]
            self._episode_ranges.append(
                (ep_idx, ep["dataset_from_index"], ep["dataset_to_index"])
            )

        # CLIP 임베딩 precompute 또는 캐시 로드
        self._embeddings = self._load_or_compute_embeddings(
            clip_model, clip_pretrained, embed_device, embed_cache_path
        )

        # Window 단위 샘플 구성
        self._windows = self._build_windows()
        print(f"Phase1WindowDataset: {self._n_episodes} episodes → {len(self._windows)} windows "
              f"(w_tr={window_size}, k={max_windows_per_episode}, diss={dissimilarity_mode})")

    def _load_or_compute_embeddings(self, clip_model_name, clip_pretrained, device, cache_path):
        """z_t dict {abs_frame_idx: tensor [1024]} 로드 또는 계산."""
        if cache_path and os.path.exists(cache_path):
            print(f"Loading embedding cache from {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        print(f"Computing CLIP embeddings for {self._n_episodes} episodes...")
        import open_clip
        clip, _, preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip = clip.to(device).eval()
        for p in clip.parameters():
            p.requires_grad_(False)

        embeddings = {}
        for ep_idx, ep_from, ep_to in tqdm(self._episode_ranges, desc="Encoding episodes"):
            first_item = self.lerobot_ds[ep_from]
            task_text = first_item["task"]
            tokens = tokenizer([task_text]).to(device)
            with torch.no_grad():
                text_emb = clip.encode_text(tokens).squeeze(0).cpu()

            for abs_idx in range(ep_from, ep_to):
                item = self.lerobot_ds[abs_idx]
                img = item[self.image_key]
                img_pil = to_pil_image(img.float().clamp(0, 1))
                clip_in = preprocess(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    vis_emb = clip.encode_image(clip_in).squeeze(0).cpu()
                z = F.normalize(torch.cat([vis_emb, text_emb]), dim=0)
                embeddings[abs_idx] = z

        if cache_path:
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            torch.save(embeddings, cache_path)
            print(f"Saved embedding cache to {cache_path}")

        return embeddings

    def _build_windows(self) -> list:
        windows = []
        for ep_idx, ep_from, ep_to in self._episode_ranges:
            ep_length = min(ep_to - ep_from, self.MAX_EP_LEN)
            if ep_length < self.window_size:
                continue
            selected_starts = self._select_diverse_windows(ep_from, ep_length)
            for start in selected_starts:
                windows.append((ep_from, start, ep_length))
        return windows

    def _select_diverse_windows(self, ep_from: int, ep_length: int) -> list:
        stride = 1
        candidates = list(range(0, ep_length - self.window_size + 1, stride))

        if len(candidates) <= self.max_windows_per_episode:
            return candidates

        feats = _compute_window_feats(
            candidates, ep_from, self.window_size,
            self._embeddings, self.dissimilarity_mode
        )

        diff = feats.unsqueeze(0) - feats.unsqueeze(1)
        scores = (diff ** 2).sum(-1).sum(dim=1)

        k = min(self.max_windows_per_episode, len(candidates))
        return [candidates[i] for i in scores.topk(k).indices.tolist()]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        ep_from, window_start, ep_length = self._windows[idx]

        z_seq = torch.stack([
            self._embeddings[ep_from + window_start + t]
            for t in range(self.window_size)
        ])

        targets = torch.tensor([
            (window_start + t) / max(ep_length - 1, 1)
            for t in range(self.window_size)
        ], dtype=torch.float32).unsqueeze(-1)

        pred_mask = torch.ones(self.window_size, dtype=torch.bool)

        return {
            "z_seq": z_seq,
            "targets": targets,
            "pred_mask": pred_mask,
            "ep_len": self.window_size,
        }
