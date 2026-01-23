#!/usr/bin/env python3
"""
Script to add 'time_to_go' features to a LeRobot dataset.
Usage:
    python scripts/add_time_features.py --repo_id lerobot/libero_goal_image --output_dir data/libero_goal_time
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import modify_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="Source repository ID (e.g. lerobot/libero_goal_image)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the modified dataset")
    parser.add_argument("--new_repo_id", type=str, default=None, help="New repository ID (optional)")
    parser.add_argument("--root", type=str, default=None, help="Root directory for datasets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Load the dataset
    logger.info(f"Loading dataset from {args.repo_id}...")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    
    # 2. Pre-compute episode lengths
    # dataset.meta.episodes is a HuggingFace Dataset containing metadata
    logger.info("Pre-computing episode lengths...")
    ep_lengths = {}
    
    # Iterate through the episodes metadata to build a map
    # Note: dataset.meta.episodes might be loaded lazily, so we access it
    if dataset.meta.episodes is None:
        dataset.meta.load_metadata()
        
    for ep in dataset.meta.episodes:
        # Depending on version, it might be a dict or row object
        ep_idx = ep["episode_index"]
        length = ep["length"]
        ep_lengths[ep_idx] = length

    fps = dataset.fps
    logger.info(f"Loaded lengths for {len(ep_lengths)} episodes. FPS: {fps}")

    # 3. Define the feature computation function
    def compute_task_time_to_go(row, ep_idx, frame_idx):
        """
        Calculates time-to-go for the current task (episode).
        Returns: (total_frames - frame_idx) / fps
        """
        total_frames = ep_lengths[ep_idx]
        # Calculate remaining time in seconds
        # frame_idx is 0-based.
        # If total is 100, frame 0 -> 100/fps remaining? 
        # Or (100 - 0) / fps.
        # At last frame (99), (100 - 99) / fps = 1/fps seconds remaining.
        remaining_frames = total_frames - frame_idx
        return np.float32(remaining_frames / fps)

    # 4. Define the new feature specification
    # We are adding a scalar float32 feature
    new_features = {
        "time_to_go_task": (
            compute_task_time_to_go,
            {
                "dtype": "float32",
                "shape": (1,),
                "names": None
            }
        )
    }

    # 5. Apply modification
    logger.info("Adding 'time_to_go_task' feature...")
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to a suffix if not provided
        output_dir = None 
        
    modify_features(
        dataset=dataset,
        add_features=new_features,
        output_dir=output_dir,
        repo_id=args.new_repo_id
    )
    
    logger.info("Done! Dataset modified.")

if __name__ == "__main__":
    main()
