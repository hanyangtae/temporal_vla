"""
Pick-and-place 에피소드 전체에 대한 CLIP embedding 캐시 생성.

기존 캐시는 처음 ~3273개 에피소드만 포함.
이 스크립트는 전체 데이터셋에서 pick-and-place 에피소드만 필터하여
모든 프레임의 z_t = L2_norm([CLIP_vis(img) ; CLIP_text(goal)]) 를 계산.

실행:
    python scripts/build_clip_cache.py
    python scripts/build_clip_cache.py --save_path data/bridge_v2_pick_place_clip.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/bridge_v2_lerobot")
    parser.add_argument("--repo_id", type=str, default="FedorX8/bridge_v2_lerobot")
    parser.add_argument("--image_key", type=str, default="observation.images.primary")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--task_keywords", type=str, nargs="+",
                        default=["put", "place", "pick"])
    parser.add_argument("--save_path", type=str,
                        default="data/bridge_v2_pick_place_clip_embeddings.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="CLIP visual encoder 배치 크기")
    parser.add_argument("--max_episodes", type=int, default=5000,
                        help="인코딩할 최대 에피소드 수 (pick-and-place 필터 후)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 데이터셋 로드 + pick-and-place 에피소드 필터
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.data_root)
    meta = ds.meta

    episode_ranges = []
    for ep_idx in range(meta.total_episodes):
        ep = meta.episodes[ep_idx]
        task_text = " ".join(ep.get("tasks", [])).lower()
        if not any(kw.lower() in task_text for kw in args.task_keywords):
            continue
        episode_ranges.append((ep_idx, ep["dataset_from_index"], ep["dataset_to_index"]))

    print(f"Pick-and-place episodes: {len(episode_ranges)} / {meta.total_episodes}")
    if len(episode_ranges) > args.max_episodes:
        episode_ranges = episode_ranges[:args.max_episodes]
        print(f"Limiting to first {args.max_episodes} episodes")
    total_frames = sum(to - fr for _, fr, to in episode_ranges)
    print(f"Episodes to encode: {len(episode_ranges)}, frames: {total_frames}")

    # 2) CLIP 모델 로드
    import open_clip
    clip, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip = clip.to(device).eval()
    for p in clip.parameters():
        p.requires_grad_(False)

    # 3) 에피소드별 임베딩 계산
    embeddings = {}
    for ep_idx, ep_from, ep_to in tqdm(episode_ranges, desc="Encoding episodes"):
        # text embedding (에피소드 내 동일)
        first_item = ds[ep_from]
        task_text = first_item["task"]
        tokens = tokenizer([task_text]).to(device)
        with torch.no_grad():
            text_emb = clip.encode_text(tokens).squeeze(0).cpu()  # [512]

        # visual embedding (배치 처리)
        abs_indices = list(range(ep_from, ep_to))
        for batch_start in range(0, len(abs_indices), args.batch_size):
            batch_indices = abs_indices[batch_start:batch_start + args.batch_size]
            imgs = []
            for abs_idx in batch_indices:
                item = ds[abs_idx]
                img = item[args.image_key]
                img_pil = to_pil_image(img.float().clamp(0, 1))
                imgs.append(preprocess(img_pil))

            clip_in = torch.stack(imgs).to(device)
            with torch.no_grad():
                vis_embs = clip.encode_image(clip_in).cpu()  # [B, 512]

            for abs_idx, vis_emb in zip(batch_indices, vis_embs):
                z = F.normalize(torch.cat([vis_emb, text_emb]), dim=0)  # [1024]
                embeddings[abs_idx] = z

    # 4) 저장
    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    torch.save(embeddings, args.save_path)
    print(f"\nSaved {len(embeddings)} frame embeddings → {args.save_path}")
    print(f"Episodes: {len(episode_ranges)}, Frames: {len(embeddings)}")


if __name__ == "__main__":
    main()
