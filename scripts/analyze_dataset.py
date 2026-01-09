import argparse
from pathlib import Path
from datasets import load_from_disk
import numpy as np
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze LeRobot datasets.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=Path("data/datasets/lerobot/libero_goal_image"),
        help="Path to the dataset (e.g. 'data/datasets/lerobot/libero_goal_image').",
    )
    
    return parser.parse_args()

def main():
    # 데이터셋 경로
    args = parse_arguments()
    dataset_path = Path(args.dataset_path)
    
    print(f"🚀 Loading dataset from: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    
    # Train split만 확인
    df = ds['train'].to_pandas() # pandas로 변환하면 분석이 쉬움
    
    # 1. Task 통계
    task_indices = df['task_index'].unique()
    print(f"\n📊 Task Statistics:")
    print(f"  - Total Tasks: {len(task_indices)}")
    print(f"  - Task Indices: {sorted(task_indices)}")
    
    # 2. Episode 통계
    episode_indices = df['episode_index'].unique()
    print(f"\n📊 Episode Statistics:")
    print(f"  - Total Episodes: {len(episode_indices)}")
    
    # Task 별 Episode 수 계산
    print("\n🔍 Episodes per Task:")
    for task_idx in sorted(task_indices):
        episodes_in_task = df[df['task_index'] == task_idx]['episode_index'].unique()
        print(f"  - Task {task_idx}: {len(episodes_in_task)} episodes")
        
    # 3. Episode 길이 분석
    print("\n📏 Episode Lengths:")
    lengths = df.groupby('episode_index').size()
    print(f"  - Min Length: {lengths.min()}")
    print(f"  - Max Length: {lengths.max()}")
    print(f"  - Mean Length: {lengths.mean():.2f}")

if __name__ == "__main__":
    main()