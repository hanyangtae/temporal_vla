#!/usr/bin/env python3
"""
LIBERO Subtask 데이터를 LeRobotDataset (Standard Format)으로 변환
"""
import argparse
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import torch

# LeRobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def load_subtask_entries(subtask_dir: Path, task_name: str) -> dict:
    """Subtask entries를 demo_idx별로 로드"""
    jsonl_path = subtask_dir / task_name / "subtask_entries.jsonl"
    if not jsonl_path.exists():
        return {}
    
    entries_by_demo = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            demo_idx = entry['demo_idx']
            if demo_idx not in entries_by_demo:
                entries_by_demo[demo_idx] = []
            entries_by_demo[demo_idx].append(entry)
    return entries_by_demo

def load_subtask_metadata(subtask_dir: Path, task_name: str) -> dict:
    """Subtask 메타데이터 로드"""
    meta_path = subtask_dir / task_name / "subtask_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Convert subtask data to LeRobotDataset")
    parser.add_argument("--hdf5_dir", type=str, default="data/datasets/libero_goal")
    parser.add_argument("--subtask_dir", type=str, default="data/datasets/libero_goal_subtasks")
    # repo_id는 로컬 저장시 폴더명 식별자로 사용됨
    parser.add_argument("--repo_id", type=str, default="libero/goal_subtask_time") 
    parser.add_argument("--output_dir", type=str, default="data/datasets/lerobot_libero_goal")
    parser.add_argument("--fps", type=int, default=10, help="Dataset FPS (must be integer)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    args = parser.parse_args()
    
    hdf5_dir = Path(args.hdf5_dir)
    subtask_dir = Path(args.subtask_dir)
    output_dir = Path(args.output_dir)
    
    # 1. 태스크 목록 확인
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted([
            p.name for p in subtask_dir.iterdir() 
            if p.is_dir() and not p.name.startswith('.')
        ])
    
    print(f"📊 Converting {len(tasks)} tasks to LeRobotDataset format...")

    # 2. Features 정의 (LeRobot 표준 + Custom Time Features)
    # LIBERO Image Sizes: AgentView(256x256), EyeInHand(128x128)
    features = {
        "observation.images.agentview": {
            "dtype": "video",
            "shape": (3, 128, 128),
            "names": ["c", "h", "w"],
        },
        "observation.images.eye_in_hand": {
            "dtype": "video",
            "shape": (3, 128, 128),
            "names": ["c", "h", "w"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "gripper"],
        },
        # Custom Time Features
        "time_to_go_subtask": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "time_to_go_full": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        # Debugging / Analysis info
        "subtask_idx": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
    }

    # 3. LeRobotDataset 생성
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(args.fps),  # Force int for PyAV compatibility
        features=features,
        root=output_dir,
        use_videos=True, # 비디오로 인코딩하여 저장 (용량 절약)
    )

    total_episodes = 0
    
    for task_name in tqdm(tasks, desc="Processing Tasks"):
        # HDF5 파일 확인
        hdf5_path = hdf5_dir / f"{task_name}_demo.hdf5"
        if not hdf5_path.exists():
            print(f"⚠️  HDF5 not found: {hdf5_path}")
            continue
        
        # Subtask 정보 로드
        subtask_entries = load_subtask_entries(subtask_dir, task_name)
        if not subtask_entries:
            continue
        
        metadata = load_subtask_metadata(subtask_dir, task_name)
        task_description = metadata['task'] if metadata else task_name.replace('_', ' ')
        
        # HDF5 열기
        with h5py.File(hdf5_path, 'r') as f:
            demos = sorted(
                [k for k in f['data'].keys() if k.startswith('demo')],
                key=lambda x: int(x.split('_')[1])
            )

            for demo_name in demos:
                demo_idx = int(demo_name.split('_')[1])
                
                # Subtask 정보가 있는 데모만 처리
                if demo_idx not in subtask_entries:
                    continue
                    
                entries = subtask_entries[demo_idx]
                n_frames = len(entries)
                
                # HDF5 데이터 로드
                demo_grp = f['data'][demo_name]
                if demo_grp['actions'].shape[0] != n_frames:
                    print(f"⚠️  Frame mismatch in {demo_name}: skipping")
                    continue

                # Bulk load for speed
                actions = demo_grp['actions'][:] # (N, 7)
                agentview = demo_grp['obs']['agentview_rgb'][:] # (N, 256, 256, 3)
                eye_in_hand = demo_grp['obs']['eye_in_hand_rgb'][:] # (N, 128, 128, 3)
                
                ee_pos = demo_grp['obs']['ee_pos'][:]
                ee_ori = demo_grp['obs']['ee_ori'][:]
                gripper = demo_grp['obs']['gripper_states'][:]
                states = np.concatenate([ee_pos, ee_ori, gripper[:, :1]], axis=-1) # (N, 7)
    
                # 프레임 단위로 추가
                for i in range(n_frames):
                    entry = entries[i]
                    
                    # Image Transpose: (H, W, C) -> (C, H, W)
                    agent_img = np.moveaxis(agentview[i], -1, 0)
                    hand_img = np.moveaxis(eye_in_hand[i], -1, 0)
                    
                    # Task String 구성: "Task: Subtask"
                    # XVLA 모델이 이 텍스트를 보고 현재 무엇을 해야 하는지, 얼마나 걸릴지 예측함
                    full_task_desc = f"{task_description}: {entry['subtask_name']}"

                    dataset.add_frame({
                        "observation.images.agentview": agent_img,
                        "observation.images.eye_in_hand": hand_img,
                        "observation.state": states[i].astype(np.float32),
                        "action": actions[i].astype(np.float32),
                        "time_to_go_subtask": np.array([entry['time_to_go_subtask']], dtype=np.float32),
                        "time_to_go_full": np.array([entry['time_to_go_full']], dtype=np.float32),
                        "subtask_idx": np.array([entry['subtask_idx']], dtype=np.int64),
                        "task": full_task_desc # LeRobotDataset이 자동으로 인덱싱 관리
                    })
    
                # 에피소드 저장 (비디오 인코딩 수행)
                dataset.save_episode()
                total_episodes += 1

    dataset.finalize()
    print(f"\n✅ Created LeRobotDataset at {output_dir / args.repo_id}")
    print(f"   Total Episodes: {total_episodes}")

if __name__ == "__main__":
    main()
