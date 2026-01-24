
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
from scipy.spatial.transform import Rotation as R

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

def get_task_name_from_filename(filename):
    # Extract task name from filename like "put_the_cream_cheese_in_the_bowl_demo.hdf5"
    basename = os.path.basename(filename)
    task_name = basename.replace("_demo.hdf5", "").replace("_", " ")
    return task_name

def mat_to_rot6d(rot_mats):
    """
    Convert rotation matrices (..., 3, 3) to 6D rotation representation (..., 6).
    """
    col1 = rot_mats[..., :3, 0]
    col2 = rot_mats[..., :3, 1]
    return np.concatenate([col1, col2], axis=-1)

def axis_angle_to_rot6d(axis_angle):
    """
    Convert axis-angle rotation (..., 3) to 6D rotation representation (..., 6).
    """
    rot = R.from_rotvec(axis_angle)
    rot_mats = rot.as_matrix()  # (..., 3, 3)
    return mat_to_rot6d(rot_mats)

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
            "shape": (20,),  # X-VLA expects 20-dim state (Pos(3) + Rot6D(6) + Padding)
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (20,),  # X-VLA ee6d format: [Pos(3), Rot6D(6), Gripper(1), Padding(10)]
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
                    
                    # State: X-VLA expects EEF state (Pos + Rot6D)
                    # Libero provides 'ee_pos' (3) and 'ee_ori' (3, axis-angle)
                    ee_pos = demo['obs']['ee_pos'][:]  # (N, 3)
                    ee_ori = demo['obs']['ee_ori'][:]  # (N, 3) - Axis Angle
                    
                    # Convert Axis-Angle to Rotation Matrix -> 6D Rotation
                    # Note: scipy Rotation handles batch conversion
                    rot = R.from_rotvec(ee_ori)
                    rot_mats = rot.as_matrix()  # (N, 3, 3)
                    ee_rot6d = mat_to_rot6d(rot_mats)  # (N, 6)
                    
                    # Construct 20-dim state vector
                    # [Pos(3), Rot6D(6), Zero(1), Padding(10)]
                    # The 'Zero(1)' is 'extra' in processor_xvla.py
                    # The 'Padding(10)' is because processor_xvla.py concatenates (proprio_state, zeros_like(proprio_state))
                    # proprio_state is (10,) -> total (20,)
                    
                    num_frames = len(ee_pos)
                    extra = np.zeros((num_frames, 1), dtype=np.float32)
                    proprio_state = np.concatenate([ee_pos, ee_rot6d, extra], axis=-1)  # (N, 10)
                    padding = np.zeros_like(proprio_state)
                    
                    robot_states = np.concatenate([proprio_state, padding], axis=-1)  # (N, 20)
                    
                    # Actions: Convert 7D (pos3 + axis-angle3 + gripper1) to 20D ee6d format
                    # ee6d format: [Pos(3), Rot6D(6), Gripper(1), Padding(10)]
                    # Note: xvla-libero uses ee6d action mode with this layout
                    raw_actions = demo['actions'][:]  # (N, 7): [pos3, axis-angle3, gripper1]
                    
                    action_pos = raw_actions[:, :3]  # (N, 3)
                    action_axis_angle = raw_actions[:, 3:6]  # (N, 3)
                    action_gripper = raw_actions[:, 6:7]  # (N, 1)
                    
                    # Convert axis-angle to 6D rotation
                    action_rot6d = axis_angle_to_rot6d(action_axis_angle)  # (N, 6)
                    
                    # Construct 20D action: [pos3, rot6d6, gripper1, padding10]
                    action_first10 = np.concatenate([action_pos, action_rot6d, action_gripper], axis=-1)  # (N, 10)
                    action_padding = np.zeros_like(action_first10)  # (N, 10)
                    actions = np.concatenate([action_first10, action_padding], axis=-1)  # (N, 20)
                    
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
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for LeRobot dataset", default="data/lerobot_libero_goal_flipped")
    parser.add_argument("--repo_id", type=str, default="lerobot_libero_goal_flipped", help="Repository ID for the dataset")
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
