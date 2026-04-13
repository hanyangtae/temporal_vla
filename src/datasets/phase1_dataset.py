"""
Phase 1 학습용 데이터셋 (VITA D.1 방식).

expert trajectory에서 (z_t sequence, progress targets t/T)를 생성.
z_t = L2-normalize([φ_v(o_t) ; φ_g(g)]) ∈ R^1024
    φ_v: OpenCLIP ViT-B/32 visual encoder (frozen)
    φ_g: OpenCLIP ViT-B/32 text encoder (frozen)

VITA D.1 학습 방식:
    - 에피소드 전체를 하나의 샘플로 반환 (padded to 120)
    - 학습 시 full trajectory 순차 TTT → train/inference mismatch 없음
    - ℓself: 전체 valid frame에서 계산 (meta-learning)
    - ℓpred: dissimilarity sampling으로 선택된 k=8 window frame에서만 계산 (shortcut 방지)
"""

import os
from typing import Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class Phase1EpisodicDataset(Dataset):
    """
    Phase 1 ProgressPredictor 학습용 데이터셋 (VITA D.1).

    에피소드 전체를 하나의 샘플로 반환.
    각 sample: z_seq [T, 1024], targets [T, 1], pred_mask [T], ep_len int.

    TTT inner update는 전체 T프레임에 대해 순차 수행 (θ_0 → θ_T).
    ℓpred는 dissimilarity sampling으로 선택된 k window에 속하는 frame에서만 계산.
    ℓself는 전체 valid frame에서 계산.

    Args:
        data_root: LeRobotDataset 로컬 경로
        repo_id: HuggingFace repo ID
        window_size: dissimilarity sampling용 sub-trajectory 길이 (VITA=8)
        max_windows_per_episode: dissimilarity sampling으로 선택할 window 수 k (VITA=8)
        image_key: 이미지 관측 키
        clip_model: OpenCLIP 모델 이름
        clip_pretrained: OpenCLIP pretrained weight
        embed_device: CLIP 연산 디바이스
        embed_cache_path: z_t 캐시 저장/로드 경로 (.pt). None이면 캐싱 안 함.
        episode_indices: 사용할 에피소드 인덱스 리스트
        start_episode: episode_indices가 없을 때 시작 인덱스
        max_episodes: episode_indices가 없을 때 최대 에피소드 수
        task_keywords: task description에 포함되어야 할 키워드 리스트 (OR 조건).
                       None이면 필터링 없음. 예: ["put", "place", "pick"]
    """

    MAX_EP_LEN = 120  # VITA D.1: pad all trajectories to max length 120

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
    ):
        self.window_size = window_size
        self.max_windows_per_episode = max_windows_per_episode
        self.image_key = image_key

        # LeRobotDataset 로드
        self.lerobot_ds = LeRobotDataset(repo_id=repo_id, root=data_root)
        meta = self.lerobot_ds.meta
        total_ep = meta.total_episodes

        # episode_indices가 직접 주어지면 우선 사용, 아니면 start/max로 범위 계산
        if episode_indices is not None:
            indices = episode_indices
        else:
            end_ep = min(total_ep, start_episode + max_episodes) if max_episodes else total_ep
            indices = list(range(start_episode, end_ep))

        # task keyword 필터링 (VITA: pick-and-place만 사용)
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

        # 에피소드 경계 수집: [(ep_idx, from, to), ...]
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

        # 에피소드별 데이터 구성: (ep_idx, ep_from, ep_length, pred_mask)
        self._episode_data = self._build_episode_data()

    # ──────────────────────────────────────────────
    # Embedding precompute
    # ──────────────────────────────────────────────

    def _load_or_compute_embeddings(
        self,
        clip_model_name: str,
        clip_pretrained: str,
        device: str,
        cache_path: Optional[str],
    ) -> dict:
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
            # 에피소드 첫 프레임에서 task text 가져오기 (에피소드 내 동일)
            first_item = self.lerobot_ds[ep_from]
            task_text = first_item["task"]
            tokens = tokenizer([task_text]).to(device)
            with torch.no_grad():
                text_emb = clip.encode_text(tokens).squeeze(0).cpu()  # [512]

            for abs_idx in range(ep_from, ep_to):
                item = self.lerobot_ds[abs_idx]
                img = item[self.image_key]  # float tensor [C, H, W] in [0, 1]
                img_pil = to_pil_image(img.float().clamp(0, 1))
                clip_in = preprocess(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    vis_emb = clip.encode_image(clip_in).squeeze(0).cpu()  # [512]
                z = F.normalize(torch.cat([vis_emb, text_emb]), dim=0)  # [1024]
                embeddings[abs_idx] = z

        if cache_path:
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            torch.save(embeddings, cache_path)
            print(f"Saved embedding cache to {cache_path}")

        return embeddings

    # ──────────────────────────────────────────────
    # Episode-level data construction
    # ──────────────────────────────────────────────

    def _build_episode_data(self) -> list:
        """
        에피소드별 (ep_idx, ep_from, ep_length, pred_mask) 구성.

        pred_mask: [ep_length] bool — dissimilarity sampling으로 선택된
                   window에 속하는 frame은 True (ℓpred 계산 대상).
                   ℓself는 전체 valid frame에서 별도로 계산됨.
        """
        episode_data = []
        for ep_idx, ep_from, ep_to in self._episode_ranges:
            ep_length = ep_to - ep_from
            if ep_length < self.window_size:
                continue
            ep_length = min(ep_length, self.MAX_EP_LEN)
            pred_mask = self._compute_pred_mask(ep_from, ep_length)
            episode_data.append((ep_idx, ep_from, ep_length, pred_mask))
        return episode_data

    def _compute_pred_mask(self, ep_from: int, ep_length: int) -> torch.Tensor:
        """
        VITA Eq.5 dissimilarity sampling → frame-level boolean mask.

        stride=1로 sliding window candidate 생성 후
        pairwise diversity score 기준 top-k window 선택.
        선택된 window에 속하는 모든 frame을 True로 마킹.

        ℓpred는 이 mask가 True인 frame에서만 계산되고,
        ℓself와 TTT inner update는 전체 frame에서 수행됨.

        Returns:
            pred_mask [ep_length] bool
        """
        stride = 1
        candidates = list(range(0, ep_length - self.window_size + 1, stride))

        if len(candidates) <= self.max_windows_per_episode:
            selected_starts = candidates
        else:
            # VITA 원본: window mean embedding 기준 dissimilarity scoring
            feats = torch.stack([
                torch.stack([self._embeddings[ep_from + i]
                             for i in range(s, s + self.window_size)]).mean(0)
                for s in candidates
            ])

            # VITA Eq.5: s(w) = Σ_{v∈W} ||w - v||²
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
        ep_idx, ep_from, ep_length, pred_mask = self._episode_data[idx]

        # 에피소드 전체 임베딩 시퀀스 [T, D]
        z_seq = torch.stack([
            self._embeddings[ep_from + t]
            for t in range(ep_length)
        ])

        # 진행률 label: t/T ∈ [0, 1]
        targets = (
            torch.arange(ep_length, dtype=torch.float32) / max(ep_length - 1, 1)
        ).unsqueeze(-1)  # [T, 1]

        return {
            "z_seq": z_seq,        # [T, D]
            "targets": targets,    # [T, 1]
            "pred_mask": pred_mask, # [T] bool — ℓpred 계산 대상 frame (선택된 window만 True)
            "ep_len": ep_length,
        }
