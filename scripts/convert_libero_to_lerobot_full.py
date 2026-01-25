"""
LIBERO HDF5 → LeRobot Dataset 변환 스크립트 (전체 데이터 보존)

이 스크립트는 원본 HDF5의 모든 데이터를 LeRobot 형식으로 변환합니다.
action_mode="auto"와 함께 사용하도록 action을 10D로 변환합니다.

HDF5 원본 구조:
- obs/agentview_rgb: (N, 128, 128, 3) uint8
- obs/eye_in_hand_rgb: (N, 128, 128, 3) uint8
- obs/ee_pos: (N, 3) float64
- obs/ee_ori: (N, 3) float64 (axis-angle)
- obs/ee_states: (N, 6) float64
- obs/gripper_states: (N, 2) float64
- obs/joint_states: (N, 7) float64
- actions: (N, 7) float64 [delta_pos(3), delta_ori(3), gripper(1)]
- dones: (N,)

LeRobot 변환 후:
- observation.images.image: (128, 128, 3) - agentview (flipped for X-VLA)
- observation.images.image2: (128, 128, 3) - eye_in_hand
- observation.state: (20,) - X-VLA proprioception format:
    [ee_pos(3), ee_rot6d(6), gripper(2), joints(7), padding(2)]
- action: (10,) - [pos3, rot6d6, gripper1] for action_mode="auto" with real_dim=10
- next.done: (1,)
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle rotation (N, 3) to 6D rotation representation (N, 6).
    
    6D representation = first two columns of rotation matrix.
    """
    rot = R.from_rotvec(axis_angle)
    rot_mats = rot.as_matrix()  # (N, 3, 3)
    col1 = rot_mats[:, :3, 0]  # (N, 3)
    col2 = rot_mats[:, :3, 1]  # (N, 3)
    return np.concatenate([col1, col2], axis=-1)  # (N, 6)


def get_task_name_from_filename(filename):
    """Extract task name from filename like 'put_the_cream_cheese_in_the_bowl_demo.hdf5'"""
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

    # Define features - X-VLA 호환 형식
    # observation.state: 20D = [ee_pos(3), ee_rot6d(6), gripper(2), joints(7), padding(2)]
    features = {
        # Images
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
        # Robot state - X-VLA expects 20D proprioception vector
        "observation.state": {
            "dtype": "float32",
            "shape": (20,),
            "names": [
                "ee_x", "ee_y", "ee_z",  # ee_pos (3)
                "r1", "r2", "r3", "r4", "r5", "r6",  # ee_rot6d (6)
                "gripper_left", "gripper_right",  # gripper (2)
                "j1", "j2", "j3", "j4", "j5", "j6", "j7",  # joints (7)
                "pad1", "pad2",  # padding (2)
            ],
        },
        # Action - 10D [pos3, rot6d6, gripper1] for action_mode="auto" with real_dim=10
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": ["dx", "dy", "dz", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"],
        },
        # Done flag
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
        robot_type="panda",
        use_videos=True,
    )

    # Find all HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))
    print(f"Found {len(hdf5_files)} HDF5 files.")

    total_episodes = 0
    
    for hdf5_path in tqdm(hdf5_files, desc="Processing files"):
        task_description = get_task_name_from_filename(hdf5_path)
        print(f"\nProcessing task: {task_description}")

        try:
            with h5py.File(hdf5_path, 'r') as f:
                data_group = f['data']
                demo_keys = sorted(list(data_group.keys()))
                
                for demo_key in tqdm(demo_keys, desc=f"Episodes in {hdf5_path.name}", leave=False):
                    demo = data_group[demo_key]
                    
                    # Extract observations
                    agentview = demo['obs']['agentview_rgb'][:]
                    eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]
                    ee_pos = demo['obs']['ee_pos'][:]
                    ee_ori = demo['obs']['ee_ori'][:]
                    gripper_states = demo['obs']['gripper_states'][:]
                    joint_states = demo['obs']['joint_states'][:]
                    
                    # IMPORTANT: Flip main camera (agentview) 180° to match X-VLA convention
                    # X-VLA pretrained model expects flipped images for main camera
                    agentview = agentview[:, ::-1, ::-1, :].copy()
                    # eye_in_hand is NOT flipped (matches eval env_preprocessor)
                    
                    # Actions: Convert 7D [pos3, axis-angle3, gripper1] → 10D [pos3, rot6d6, gripper1]
                    raw_actions = demo['actions'][:]  # (N, 7)
                    action_pos = raw_actions[:, :3]  # (N, 3)
                    action_axis_angle = raw_actions[:, 3:6]  # (N, 3)
                    action_gripper = raw_actions[:, 6:7]  # (N, 1)
                    
                    # Convert axis-angle to 6D rotation
                    action_rot6d = axis_angle_to_rot6d(action_axis_angle)  # (N, 6)
                    
                    # Concatenate: [pos3, rot6d6, gripper1] = 10D
                    actions = np.concatenate([action_pos, action_rot6d, action_gripper], axis=-1)  # (N, 10)
                    
                    dones = demo['dones'][:]
                    
                    num_frames = len(ee_pos)
                    
                    # Add frames
                    for i in range(num_frames):
                        frame = {
                            # Images
                            "observation.images.image": agentview[i],
                            "observation.images.image2": eye_in_hand[i],
                            # Robot state
                            "observation.state.ee_pos": ee_pos[i].astype(np.float32),
                            "observation.state.ee_ori": ee_ori[i].astype(np.float32),
                            "observation.state.gripper": gripper_states[i].astype(np.float32),
                            "observation.state.joints": joint_states[i].astype(np.float32),
                            # Action
                            "action": actions[i].astype(np.float32),
                            # Done
                            "next.done": np.array([dones[i]], dtype=bool),
                            # Task description
                            "task": task_description,
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
    print(f"\n✅ Successfully converted {total_episodes} episodes to {output_dir}")
    print(f"Features saved: {list(features.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 to LeRobot format (full data)")
    parser.add_argument("--input_dir", type=Path, required=True, 
                        help="Directory containing Libero HDF5 files")
    parser.add_argument("--output_dir", type=Path, required=True, 
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default="lerobot_libero_goal_full", 
                        help="Repository ID for the dataset")
    parser.add_argument("--fps", type=int, default=10, 
                        help="FPS of the dataset")
    parser.add_argument("--force_override", action="store_true", 
                        help="Overwrite output directory if exists")
    
    args = parser.parse_args()
    
    init_logging()
    convert_libero_to_lerobot(
        args.input_dir,
        args.output_dir,
        args.repo_id,
        args.fps,
        args.force_override
    )
