
import argparse
import glob
import logging
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

def get_task_name_from_filename(filename):
    # Extract task name from filename like "put_the_cream_cheese_in_the_bowl_demo.hdf5"
    basename = os.path.basename(filename)
    task_name = basename.replace("_demo.hdf5", "").replace("_", " ")
    return task_name

def convert_libero_to_lerobot(
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    fps: int = 10,
    force_override: bool = False,
):
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory {output_dir} already exists. Use --force_override to overwrite.")
            return

    # Define features
    features = {
        "observation.images.agentview": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),  # 7 joints + 2 gripper
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"],
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        },
    }

    # Create LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="panda",  # Libero uses Panda
        use_videos=True,
    )

    # Find all HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))
    print(f"Found {len(hdf5_files)} HDF5 files.")

    total_episodes = 0
    
    for hdf5_path in tqdm(hdf5_files, desc="Processing files"):
        task_description = get_task_name_from_filename(hdf5_path)
        print(f"Processing task: {task_description}")

        try:
            with h5py.File(hdf5_path, 'r') as f:
                data_group = f['data']
                demo_keys = sorted(list(data_group.keys()))
                
                for demo_key in tqdm(demo_keys, desc=f"Episodes in {hdf5_path.name}", leave=False):
                    demo = data_group[demo_key]
                    
                    # Extract data
                    # Observations
                    agentview = demo['obs']['agentview_rgb'][:]
                    eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]
                    
                    # State: joint_states (7) + gripper_states (2)
                    joint_states = demo['obs']['joint_states'][:]
                    gripper_states = demo['obs']['gripper_states'][:]
                    robot_states = np.concatenate([joint_states, gripper_states], axis=1)
                    
                    actions = demo['actions'][:]
                    dones = demo['dones'][:]
                    
                    num_frames = len(actions)
                    
                    # Add frames
                    for i in range(num_frames):
                        # Flip images if necessary (Libero images are sometimes upside down, but usually correct in hdf5)
                        # Based on inspection, they seem correct. 
                        # Note: LeRobot expects (C, H, W) for images in add_frame if using pytorch, 
                        # or (H, W, C) if using numpy/PIL? 
                        # LeRobotDataset.add_frame saves images using PIL or image_writer.
                        # It expects numpy arrays to be (H, W, C) for images.
                        
                        frame = {
                            "observation.images.agentview": agentview[i], # (128, 128, 3)
                            "observation.images.wrist": eye_in_hand[i],   # (128, 128, 3)
                            "observation.state": robot_states[i].astype(np.float32),
                            "action": actions[i].astype(np.float32),
                            "next.done": np.array([dones[i]], dtype=bool),
                            "task": task_description
                        }
                        dataset.add_frame(frame)
                    
                    # Save episode
                    dataset.save_episode()
                    total_episodes += 1
                    
        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            continue

    dataset.finalize()
    print(f"Successfully converted {total_episodes} episodes to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing Libero HDF5 files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default="lerobot/libero_goal", help="Repository ID for the dataset")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the dataset")
    parser.add_argument("--force_override", action="store_true", help="Overwrite output directory if exists")
    
    args = parser.parse_args()
    
    init_logging()
    convert_libero_to_lerobot(
        args.input_dir,
        args.output_dir,
        args.repo_id,
        args.fps,
        args.force_override
    )
