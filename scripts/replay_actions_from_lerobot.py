"""
LeRobot 데이터셋의 action을 LIBERO 환경에서 replay하여 데이터 검증

이 스크립트는:
1. LeRobot 데이터셋에서 action 시퀀스를 읽어옴 (7D: pos3 + axis-angle3 + gripper1)
2. LIBERO 환경에서 해당 action을 순차적으로 실행
3. 결과를 비디오로 저장

이를 통해 데이터 변환이 올바른지 확인할 수 있습니다.
"""

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.libero import create_libero_envs
from lerobot.utils.io_utils import write_video

# LIBERO imports
from libero.libero import benchmark


def get_episode_actions(dataset: LeRobotDataset, episode_idx: int) -> np.ndarray:
    """
    Extract all actions from a specific episode.
    
    Args:
        dataset: LeRobot dataset
        episode_idx: Episode index
        
    Returns:
        actions: (N, 7) numpy array [pos3, axis-angle3, gripper1]
    """
    # Filter dataset by episode_index
    actions = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["episode_index"].item() == episode_idx:
            action = item["action"].numpy()
            actions.append(action)
    
    if not actions:
        raise ValueError(f"No frames found for episode {episode_idx}")
    
    return np.stack(actions)


def get_episode_task(dataset: LeRobotDataset, episode_idx: int) -> str:
    """Get the task description for an episode."""
    import pandas as pd
    
    # Load tasks parquet (task name is the index, task_index is the column)
    tasks_path = dataset.root / "meta" / "tasks.parquet"
    try:
        tasks_df = pd.read_parquet(tasks_path)
        # Create reverse mapping: task_index -> task_name
        task_index_to_name = {row["task_index"]: task_name for task_name, row in tasks_df.iterrows()}
    except Exception as e:
        print(f"Warning: Could not load tasks parquet: {e}")
        return "unknown"
    
    # Find first frame of the episode to get task_index
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["episode_index"].item() == episode_idx:
            task_idx = item.get("task_index", None)
            if task_idx is not None:
                task_idx_val = task_idx.item() if hasattr(task_idx, 'item') else task_idx
                task_name = task_index_to_name.get(task_idx_val, "unknown")
                return task_name
            return "unknown"
    return "unknown"


def replay_episode(
    env,
    actions: np.ndarray,
    max_steps: int = None,
    render: bool = True,
) -> tuple[list[np.ndarray], list[float], bool]:
    """
    Replay actions in the environment.
    
    Args:
        env: LIBERO environment
        actions: (N, 7) action sequence [pos3, axis-angle3, gripper1]
        max_steps: Maximum steps to execute
        render: Whether to collect rendered frames
        
    Returns:
        frames: List of rendered frames
        rewards: List of rewards
        success: Whether the task was successful
    """
    frames = []
    rewards = []
    success = False
    
    # Validate action dimension
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected action dim 7, got {actions.shape[-1]}")
    
    obs, info = env.reset()
    
    if render:
        frames.append(env.envs[0].render())
    
    n_steps = len(actions) if max_steps is None else min(len(actions), max_steps)
    
    for i in tqdm(range(n_steps), desc="Replaying", leave=False):
        action = actions[i:i+1]  # (1, 7) for vectorized env
        
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward[0])
        
        if render:
            frames.append(env.envs[0].render())
        
        # Check success
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "is_success" in final_info:
                if final_info["is_success"][0]:
                    success = True
                    break
        
        if terminated[0] or truncated[0]:
            break
    
    return frames, rewards, success


def main():
    parser = argparse.ArgumentParser(description="Replay LeRobot dataset actions in LIBERO environment")
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Episode index to replay (default: 0)")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/replay_videos"),
                        help="Output directory for videos")
    parser.add_argument("--suite", type=str, default="libero_goal",
                        help="LIBERO suite name")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video FPS")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum steps to replay")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load LeRobot dataset
    print(f"Loading dataset from {args.dataset_path}...")
    
    # Check if dataset exists locally
    if not args.dataset_path.exists():
        print(f"Error: Dataset not found at {args.dataset_path}")
        print("Please run convert_libero_to_lerobot.py first:")
        print(f"  python3 scripts/convert_libero_to_lerobot.py \\")
        print(f"    --input_dir data/datasets/libero_goal \\")
        print(f"    --output_dir {args.dataset_path} \\")
        print(f"    --repo_id {args.dataset_path.name}")
        return
    
    # Check for meta/info.json
    meta_path = args.dataset_path / "meta" / "info.json"
    if not meta_path.exists():
        print(f"Error: Dataset metadata not found at {meta_path}")
        print("The dataset may not be properly converted.")
        return
    
    # LeRobotDataset expects root to be the dataset directory itself
    dataset = LeRobotDataset(
        repo_id=args.dataset_path.name,
        root=args.dataset_path,  # Full path to dataset directory
        video_backend="pyav",  # Use pyav instead of torchcodec
    )
    
    print(f"Dataset info:")
    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Features: {list(dataset.features.keys())}")
    
    # Check action dimension
    sample = dataset[0]
    action_dim = sample["action"].shape[-1]
    print(f"  Action dimension: {action_dim}")
    
    if action_dim != 7:
        print(f"Warning: Expected action dimension 7, got {action_dim}")
        print("This dataset may have been converted with a different format.")
    
    if args.episode_idx >= dataset.num_episodes:
        print(f"Episode index {args.episode_idx} out of range (max: {dataset.num_episodes - 1})")
        return
    
    # Get task name for the episode
    task_description = get_episode_task(dataset, args.episode_idx)
    task_name = task_description.replace(" ", "_")
    print(f"\nTask: {task_description}")
    
    # Find task ID in benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.suite]()
    
    task_id = -1
    for i, task in enumerate(suite.tasks):
        if task.name == task_name:
            task_id = i
            break
    
    if task_id == -1:
        print(f"Task '{task_name}' not found in suite '{args.suite}'")
        print(f"Available tasks: {[t.name for t in suite.tasks]}")
        return
    
    print(f"Found task at ID: {task_id}")
    
    # Get actions for the episode
    actions = get_episode_actions(dataset, args.episode_idx)
    print(f"Loaded {len(actions)} actions from episode {args.episode_idx}")
    print(f"Action shape: {actions.shape}")
    print(f"Action sample (first): {actions[0]}")
    
    # Create environment
    print("\nCreating LIBERO environment...")
    envs = create_libero_envs(
        task=args.suite,
        n_envs=1,
        gym_kwargs={
            "task_ids": [task_id],
            "obs_type": "pixels_agent_pos",
        },
        camera_name="agentview_image",
        init_states=True,
        env_cls=gym.vector.SyncVectorEnv,
        control_mode="absolute",
        episode_length=800,
    )
    
    # Get the actual env
    env = list(list(envs.values())[0].values())[0]
    
    print(f"Environment created. Max steps: {env.call('_max_episode_steps')[0]}")
    
    # Replay actions
    print("\nReplaying actions...")
    frames, rewards, success = replay_episode(
        env,
        actions,
        max_steps=args.max_steps,
        render=True,
    )
    
    # Save video
    video_path = args.output_dir / f"{task_name}_ep{args.episode_idx}_replay.mp4"
    print(f"\nSaving video to {video_path}...")
    
    frames_array = np.stack(frames)
    write_video(str(video_path), frames_array, args.fps)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Replay Results:")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Task: {task_description}")
    print(f"  Episode: {args.episode_idx}")
    print(f"  Steps executed: {len(rewards)}")
    print(f"  Total reward: {sum(rewards):.4f}")
    print(f"  Success: {success}")
    print(f"  Video saved: {video_path}")
    print(f"{'='*50}")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
