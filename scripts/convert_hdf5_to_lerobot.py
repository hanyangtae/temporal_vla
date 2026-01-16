#!/usr/bin/env python3
"""
LIBERO HDF5 데이터셋을 LeRobot 포맷 + time_to_go로 변환

Usage:
    python scripts/convert_hdf5_to_lerobot.py \
        --input data/datasets/libero_goal/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5 \
        --output data/datasets/lerobot_libero_goal_drawer \
        --task_name "open the middle drawer of the cabinet"
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_hdf5_to_lerobot(
    input_path: str,
    output_dir: str,
    task_name: str,
    fps: float = 10.0,
):
    """
    HDF5 데이터셋을 LeRobot 포맷으로 변환하고 time_to_go 추가
    
    LeRobot v2.0 로컬 포맷:
    - data/episode_XXXXXX/
        - frame_XXXXXX.png (이미지)
    - episodes.jsonl (메타데이터)
    - info.json (데이터셋 정보)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Input: {input_path}")
    print(f"📁 Output: {output_dir}")
    
    with h5py.File(input_path, 'r') as f:
        demos = sorted(
            [k for k in f['data'].keys() if k.startswith('demo')],
            key=lambda x: int(x.split('_')[1])
        )
        print(f"📊 Found {len(demos)} demos")
        
        all_episodes = []
        global_frame_idx = 0
        
        for ep_idx, demo_name in enumerate(tqdm(demos, desc="Converting")):
            demo = f['data'][demo_name]
            n_frames = demo['actions'].shape[0]
            
            # 에피소드 디렉토리 생성
            ep_dir = output_dir / f"episode_{ep_idx:06d}"
            ep_dir.mkdir(exist_ok=True)
            
            # 데이터 추출
            actions = demo['actions'][:]  # (N, 7)
            agentview = demo['obs']['agentview_rgb'][:]  # (N, 128, 128, 3)
            eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]  # (N, 128, 128, 3)
            
            # 로봇 상태 (ee_pos + ee_ori + gripper)
            ee_pos = demo['obs']['ee_pos'][:]  # (N, 3)
            ee_ori = demo['obs']['ee_ori'][:]  # (N, 3)
            gripper = demo['obs']['gripper_states'][:]  # (N, 2)
            robot_state = np.concatenate([ee_pos, ee_ori, gripper[:, :1]], axis=-1)  # (N, 7)
            
            episode_data = []
            
            for frame_idx in range(n_frames):
                # time_to_go 계산 (남은 시간)
                steps_remaining = n_frames - frame_idx - 1
                time_to_go = steps_remaining / fps
                
                # 이미지 저장
                img_agent = Image.fromarray(agentview[frame_idx])
                img_hand = Image.fromarray(eye_in_hand[frame_idx])
                img_agent.save(ep_dir / f"agentview_{frame_idx:06d}.png")
                img_hand.save(ep_dir / f"eye_in_hand_{frame_idx:06d}.png")
                
                # 프레임 데이터
                frame_data = {
                    "episode_index": ep_idx,
                    "frame_index": frame_idx,
                    "timestamp": frame_idx / fps,
                    "index": global_frame_idx,
                    
                    # Action (7-dim)
                    "action": actions[frame_idx].tolist(),
                    
                    # Robot state (7-dim: ee_pos + ee_ori + gripper)
                    "observation.state": robot_state[frame_idx].tolist(),
                    
                    # Images (경로만 저장)
                    "observation.images.agentview": f"episode_{ep_idx:06d}/agentview_{frame_idx:06d}.png",
                    "observation.images.eye_in_hand": f"episode_{ep_idx:06d}/eye_in_hand_{frame_idx:06d}.png",
                    
                    # Task description
                    "task": task_name,
                    
                    # Time-to-go (핵심!)
                    "time_to_go": time_to_go,
                }
                
                episode_data.append(frame_data)
                global_frame_idx += 1
            
            all_episodes.extend(episode_data)
    
    # episodes.jsonl 저장
    print(f"💾 Saving {len(all_episodes)} frames to episodes.jsonl...")
    with open(output_dir / "episodes.jsonl", "w") as f:
        for item in all_episodes:
            f.write(json.dumps(item) + "\n")
    
    # info.json 저장
    info = {
        "fps": fps,
        "num_episodes": len(demos),
        "num_frames": len(all_episodes),
        "task_name": task_name,
        "features": {
            "action": {"shape": [7], "dtype": "float64"},
            "observation.state": {"shape": [7], "dtype": "float64"},
            "observation.images.agentview": {"shape": [128, 128, 3], "dtype": "uint8"},
            "observation.images.eye_in_hand": {"shape": [128, 128, 3], "dtype": "uint8"},
            "time_to_go": {"shape": [], "dtype": "float64"},
        },
        "source": str(input_path),
    }
    
    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Conversion complete!")
    print(f"   Episodes: {len(demos)}")
    print(f"   Frames: {len(all_episodes)}")
    print(f"   Output: {output_dir}")
    
    # 검증
    print(f"\n🔍 Verification:")
    sample = all_episodes[0]
    print(f"   First frame time_to_go: {sample['time_to_go']:.2f}s")
    sample_last = [e for e in all_episodes if e['episode_index'] == 0][-1]
    print(f"   Last frame time_to_go: {sample_last['time_to_go']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 to LeRobot format")
    parser.add_argument("--input", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--task_name", type=str, default="open the middle drawer of the cabinet",
                        help="Task description for language conditioning")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS (default: 10)")
    args = parser.parse_args()
    
    convert_hdf5_to_lerobot(
        input_path=args.input,
        output_dir=args.output,
        task_name=args.task_name,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
