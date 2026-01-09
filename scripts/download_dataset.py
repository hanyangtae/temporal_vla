import argparse
import sys
from pathlib import Path
from datasets import load_dataset

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download LeRobot datasets to local disk.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset (e.g. 'libero_goal_image'). The 'lerobot/' prefix is added automatically if missing.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Root directory to save the dataset (default: ./data/datasets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset if it already exists.",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Repo ID 구성
    dataset_name = args.dataset_name
    if not dataset_name.startswith("lerobot/"):
        repo_id = f"lerobot/{dataset_name}"
    else:
        repo_id = dataset_name
        
    print(f"🚀 Preparing to download dataset: {repo_id}")
    
    # 2. 저장 경로 설정
    # 예: data/lerobot/libero_goal_image
    # 전략: save_dir / repo_id (data/lerobot/libero_goal_image)
    save_path = args.save_dir / repo_id
    
    if save_path.exists() and not args.force:
        print(f"ℹ️ Dataset already exists at: {save_path}")
        print("   Use --force to overwrite.")
        return

    try:
        # 3. 데이터셋 로드 (HuggingFace Hub -> Cache)
        print(f"⬇️ Downloading from HuggingFace Hub ({repo_id})...")
        ds = load_dataset(repo_id)
        
        print("\n✅ Downloaded to cache successfully!")
        print(f"🔹 Dataset Info: {ds}")
        
        # 4. 로컬 저장
        print(f"💾 Saving to local disk: {save_path} ...")
        ds.save_to_disk(save_path)
        print(f"✅ Saved successfully to {save_path}")
        
        # 5. 샘플링 (검증용)
        if 'train' in ds:
            print("\n🧪 Sampling First Episode (Index 0)...")
            item = ds['train'][0]
            print("🔹 Sample Keys:", list(item.keys()))
            
            # 간단한 정보 출력
            if "action" in item:
                print(f"  - Action Shape: {getattr(item['action'], 'shape', 'Unknown')}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
