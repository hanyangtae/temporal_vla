"""Extract SAM features from RoboCasa LeRobot dataset.

Adapted from src/policies/dreamvla/data_process/sam_extractor.py for LeRobot format.
Saves per-frame SAM image encoder features as .pt files.

Usage (in dreamvla container):
    python scripts/extract/extract_sam_robocasa.py \
        --dataset_path /temporal_vla/src/benchmarks/robocasa/datasets/v1.0/.../lerobot \
        --save_path /temporal_vla/src/benchmarks/robocasa/datasets/features/sam \
        --checkpoint /temporal_vla/src/policies/dreamvla/segment-anything/ckpts/sam_vit_b_01ec64.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -- path setup --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "policies", "dreamvla"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "policies", "dreamvla", "segment-anything"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lerobot", "src"))

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class LeRobotFrameDataset(Dataset):
    """Wraps a LeRobotDataset to iterate single frames for feature extraction."""

    def __init__(self, lerobot_dataset, image_key, input_size):
        self.dataset = lerobot_dataset
        self.image_key = image_key
        self.transform = ResizeLongestSide(input_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # LeRobot returns (C, H, W) float [0, 1] tensor for single frame
        img_tensor = sample[self.image_key]
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]  # take first if temporal dim exists
        # Convert to PIL -> numpy -> SAM transform
        arr = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()
        img = Image.fromarray(arr).resize((224, 224))
        img = np.array(img)
        img = self.transform.apply_image(img)
        img_torch = torch.as_tensor(img).permute(2, 0, 1).contiguous()
        return {"img": img_torch, "idx": idx}


def extract_features(args, image_key, camera_name):
    """Extract SAM features for one camera."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    save_dir = os.path.join(args.save_path, camera_name, "training")
    os.makedirs(save_dir, exist_ok=True)

    # Load LeRobot dataset (single frame, no delta_timestamps)
    base_dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_path,
    )

    # SAM model
    model_type = "_".join(os.path.basename(args.checkpoint).split("_")[1:3])
    sam = sam_model_registry[model_type](checkpoint=args.checkpoint)
    sam.to("cuda")
    sam.eval()

    dataset = LeRobotFrameDataset(base_dataset, image_key, sam.image_encoder.img_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    print(f"Extracting SAM features for {camera_name} ({len(dataset)} frames)")

    for batch in tqdm(dataloader, desc=camera_name):
        imgs = batch["img"].to("cuda")
        indices = batch["idx"]

        with torch.no_grad():
            imgs = sam.preprocess(imgs)
            features = sam.image_encoder(imgs)  # [B, C, 64, 64]
            features = torch.nn.functional.avg_pool2d(features, kernel_size=4, stride=4, padding=0)
            features = features.flatten(start_dim=-2)  # [B, C, 256]

        for i in range(len(indices)):
            frame_idx = indices[i].item()
            out_path = os.path.join(save_dir, f"{frame_idx}.pt")
            if args.override or not os.path.exists(out_path):
                torch.save(features[i].to(torch.bfloat16).cpu(), out_path)


def main():
    parser = argparse.ArgumentParser(description="Extract SAM features from RoboCasa LeRobot dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--save_path", type=str, required=True, help="Output directory for SAM features")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM model checkpoint path")
    parser.add_argument("--repo_id", type=str, default="robocasa_pickplace", help="LeRobot repo_id")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument(
        "--static_key", type=str, default="observation.images.robot0_agentview_right"
    )
    parser.add_argument(
        "--wrist_key", type=str, default="observation.images.robot0_eye_in_hand"
    )
    args = parser.parse_args()

    extract_features(args, args.static_key, "rgb_static")
    extract_features(args, args.wrist_key, "rgb_gripper")

    print("SAM feature extraction complete.")


if __name__ == "__main__":
    main()
