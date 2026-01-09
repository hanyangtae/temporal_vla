import argparse
import os
from pathlib import Path
from datasets import load_from_disk
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Add Time-to-Go labels to LeRobot dataset.")
    parser.add_argument("--dataset-path", type=str, default="data/datasets/lerobot/libero_goal_image", help="Path to input dataset.")
    parser.add_argument("--output-path", type=str, default="data/datasets/libero_goal_time_aware", help="Path to save processed dataset.")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS of the dataset.")
    args = parser.parse_args()

    input_path = Path(args.dataset_path)
    if not input_path.exists():
        print(f"❌ Input dataset not found at {input_path}")
        return

    # 1. Load Dataset
    print(f"🚀 Loading dataset from: {input_path}")
    ds = load_from_disk(str(input_path))
    
    # 2. Calculate Episode Lengths
    print("📏 Calculating episode lengths...")
    # episode_index를 기준으로 그룹화하여 최대 frame_index를 찾음
    # 메모리 효율을 위해 필요한 컬럼만 Pandas로 변환
    df = ds['train'].select_columns(['episode_index', 'frame_index']).to_pandas()
    
    # 각 에피소드의 총 프레임 수 = Max Frame Index + 1
    episode_lengths = df.groupby('episode_index')['frame_index'].max() + 1
    episode_lengths_dict = episode_lengths.to_dict()
    
    print(f"✅ Calculated lengths for {len(episode_lengths_dict)} episodes.")

    # 3. Add Time-to-Go Label
    print("⏳ Adding 'time_to_go' labels...")
    
    def add_time_label(example):
        ep_idx = example['episode_index']
        frame_idx = example['frame_index']
        
        total_len = episode_lengths_dict[ep_idx]
        # Time-to-Go = (Total - Current - 1) / FPS
        # 마지막 프레임(frame_idx == total_len - 1)은 0초가 됨
        steps_left = total_len - frame_idx - 1
        time_left = steps_left / args.fps
        
        example['time_to_go'] = float(time_left) # np.float32 대신 float 사용 (Arrow 호환성)
        return example

    # 병렬 처리 (num_proc)
    ds = ds.map(add_time_label, num_proc=os.cpu_count())
    
    # 4. Save
    print(f"💾 Saving processed dataset to: {args.output_path}")
    ds.save_to_disk(args.output_path)
    print("✅ Done!")

    # 검증 출력 (Episode 0)
    print("\n🔍 Verification (Episode 0 Sample):")
    # 필터링 대신 인덱스로 접근 (Episode 0이 맨 앞에 있다고 가정)
    # 정확하게 하려면 filter를 써야 하지만 느릴 수 있음.
    # 여기서는 앞부분 5개와 에피소드 길이 근처를 확인
    
    ep0_len = episode_lengths_dict[0]
    print(f"  - Episode 0 Length: {ep0_len} steps")
    
    indices_to_check = [0, ep0_len // 2, ep0_len - 1]
    for idx in indices_to_check:
        item = ds['train'][int(idx)]
        if item['episode_index'] == 0:
            print(f"  - Frame {item['frame_index']}: Time-to-Go = {item['time_to_go']:.2f}s")

if __name__ == "__main__":
    main()
