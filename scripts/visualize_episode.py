import argparse
import os
from pathlib import Path
from datasets import load_from_disk
import imageio
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Visualize a specific episode from LeRobot dataset.")
    parser.add_argument("episode_index", type=int, help="Index of the episode to visualize.")
    parser.add_argument("--dataset-path", type=str, default="data/datasets/lerobot/libero_goal_image", help="Path to the dataset.")
    parser.add_argument("--fps", type=int, default=10, help="FPS for the output video.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return

    print(f"🚀 Loading dataset from: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    
    # 데이터셋 정보 확인 (FPS)
    # ds['train'].info.description 등에 있을 수 있음.
    # 일단 인자로 받은 FPS 사용.

    # Pandas로 변환하여 해당 에피소드 필터링 (효율적이지 않지만 직관적)
    # 더 효율적인 방법: select나 filter 사용
    print(f"🔍 Extracting frames for Episode {args.episode_index}...")
    
    episode_ds = ds['train'].filter(lambda x: x['episode_index'] == args.episode_index)
    
    if len(episode_ds) == 0:
        print(f"❌ Episode {args.episode_index} not found!")
        return

    # Task 정보 확인
    first_frame = episode_ds[0]
    task_idx = first_frame.get('task_index', 'Unknown')
    print(f"🔹 Found {len(episode_ds)} frames.")
    print(f"🔹 Task Index: {task_idx}")
    
    # Instruction 정보 확인 (있다면)
    if 'language_instruction' in first_frame:
        print(f"📝 Instruction: {first_frame['language_instruction']}")
    
    frames = []
    for item in episode_ds:
        # 이미지 키 확인 (observation.images.image)
        if "observation.images.image" in item:
            img = item["observation.images.image"]
            # PIL Image면 numpy로 변환
            if isinstance(img, Image.Image):
                img = np.array(img)
            frames.append(img)
        else:
            print("❌ 'observation.images.image' key not found in dataset.")
            return

    # 비디오 저장
    output_filename = f"data/videos/episode_{args.episode_index}_{task_idx}.mp4"
    print(f"💾 Saving video to {output_filename} (FPS: {args.fps})...")
    imageio.mimsave(output_filename, frames, fps=args.fps)
    print("✅ Video saved!")

if __name__ == "__main__":
    main()
