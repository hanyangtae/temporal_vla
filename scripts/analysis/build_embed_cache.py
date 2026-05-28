"""
Pick-and-place 에피소드 CLIP 임베딩 캐시 생성.

전체 BridgeData V2에서 pick-and-place 에피소드를 필터링하고,
최대 max_episodes개의 CLIP 임베딩을 캐시 파일로 저장.

사용법:
    python scripts/analysis/build_embed_cache.py \
        --max_episodes 5000 \
        --task_keywords put place pick \
        --save_path data/bridge_v2_pnp_5000_clip_embeddings.pt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.path_setup import DATA_ROOT
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT / "bridge_v2_lerobot"))
    parser.add_argument("--repo_id", type=str, default="FedorX8/bridge_v2_lerobot")
    parser.add_argument("--image_key", type=str, default="observation.images.primary")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task_keywords", type=str, nargs="+",
                        default=["put", "place", "pick"])
    parser.add_argument("--max_episodes", type=int, default=5000)
    parser.add_argument("--save_path", type=str,
                        default=str(DATA_ROOT / "bridge_v2_pnp_5000_clip_embeddings.pt"))
    args = parser.parse_args()

    # ── 데이터셋 로드 + 필터링 ──
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.data_root)
    meta = ds.meta
    total_ep = meta.total_episodes

    filtered = []
    for ep_idx in range(total_ep):
        ep = meta.episodes[ep_idx]
        task_text = " ".join(ep.get("tasks", [])).lower()
        if any(kw.lower() in task_text for kw in args.task_keywords):
            filtered.append(ep_idx)
    print(f"Task filter {args.task_keywords}: {total_ep} → {len(filtered)} episodes")

    selected = filtered[:args.max_episodes]
    print(f"Selected: {len(selected)} episodes (max {args.max_episodes})")

    # ── CLIP 모델 로드 ──
    import open_clip
    clip, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip = clip.to(args.device).eval()
    for p in clip.parameters():
        p.requires_grad_(False)

    # ── 임베딩 계산 ──
    embeddings = {}
    total_frames = 0

    for ep_idx in tqdm(selected, desc="Encoding episodes"):
        ep = meta.episodes[ep_idx]
        ep_from = ep["dataset_from_index"]
        ep_to = ep["dataset_to_index"]

        # task text (에피소드 내 동일)
        first_item = ds[ep_from]
        task_text = first_item["task"]
        tokens = tokenizer([task_text]).to(args.device)
        with torch.no_grad():
            text_emb = clip.encode_text(tokens).squeeze(0).cpu()

        for abs_idx in range(ep_from, ep_to):
            item = ds[abs_idx]
            img = item[args.image_key]
            img_pil = to_pil_image(img.float().clamp(0, 1))
            clip_in = preprocess(img_pil).unsqueeze(0).to(args.device)
            with torch.no_grad():
                vis_emb = clip.encode_image(clip_in).squeeze(0).cpu()
            z = F.normalize(torch.cat([vis_emb, text_emb]), dim=0)
            embeddings[abs_idx] = z
            total_frames += 1

    # ── 저장 ──
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(embeddings, args.save_path)
    print(f"Saved: {args.save_path}")
    print(f"  Episodes: {len(selected)}, Frames: {total_frames}")


if __name__ == "__main__":
    main()
