
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
    """
    Convert LIBERO HDF5 dataset to LeRobot format.
    
    Data format:
    - observation.state: 7D [pos(3), axis-angle(3), extra(1)]
    - action: 7D [pos(3), axis-angle(3), gripper(1)]
    
    This matches the native LIBERO action format directly without conversion.
    """
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory {output_dir} already exists. Use --force_override to overwrite.")
            return

    print(f"Using 7D format: [pos(3), axis-angle(3), gripper(1)]")

    # Define features
    # Use "image" and "image2" to match LIBERO env camera_name_mapping:
    #   agentview_image -> image
    #   robot0_eye_in_hand_image -> image2
    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),  # 7D: [Pos(3), AxisAngle(3), Extra(1)]
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),  # 7D: [Pos(3), AxisAngle(3), Gripper(1)]
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
                    
                    # IMPORTANT: Flip main camera (agentview) 180° to match HuggingFaceVLA/libero convention
                    # X-VLA pretrained model expects flipped images for main camera
                    # This is equivalent to torch.flip(img, dims=[H, W])
                    # NOTE: Only flip 'image' (agentview), NOT 'image2' (wrist)
                    #       to match processor_xvla.py's LiberoProcessorStep behavior
                    agentview = agentview[:, ::-1, ::-1, :].copy()  # Flip H and W
                    # eye_in_hand is NOT flipped (matches eval env_preprocessor)
                    
                    # State: 7D [pos(3), axis-angle(3), extra(1)]
                    # Libero provides 'ee_pos' (3) and 'ee_ori' (3, axis-angle)
                    ee_pos = demo['obs']['ee_pos'][:]  # (N, 3)
                    ee_ori = demo['obs']['ee_ori'][:]  # (N, 3) - Axis Angle
                    
                    num_frames = len(ee_pos)
                    extra = np.zeros((num_frames, 1), dtype=np.float32)
                    
                    # Construct state vector: [pos3, axis-angle3, extra1] = 7D
                    robot_states = np.concatenate([ee_pos, ee_ori, extra], axis=-1)  # (N, 7)
                    
                    # Actions: Keep native 7D format [pos3, axis-angle3, gripper1]
                    # No conversion needed - LIBERO already uses this format
                    actions = demo['actions'][:]  # (N, 7): [pos3, axis-angle3, gripper1]
                    
                    dones = demo['dones'][:]
                    
                    # Add frames
                    for i in range(num_frames):
                        frame = {
                            "observation.images.image": agentview[i],    # (128, 128, 3) - main camera
                            "observation.images.image2": eye_in_hand[i], # (128, 128, 3) - wrist camera
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
            import traceback
            traceback.print_exc()
            continue

    dataset.finalize()
    print(f"Successfully converted {total_episodes} episodes to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing Libero HDF5 files", default="data/datasets/libero_goal")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for LeRobot dataset", default="data/lerobot_libero_goal_7d")
    parser.add_argument("--repo_id", type=str, default="lerobot_libero_goal_7d", help="Repository ID for the dataset")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the dataset")
    parser.add_argument("--force_override", action="store_true", help="Overwrite output directory if exists")
    
    args = parser.parse_args()
    
    init_logging()
    convert_libero_to_lerobot(
        args.input_dir,
        args.output_dir,
        args.repo_id,
        args.fps,
        args.force_override,
    )
