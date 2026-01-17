#!/usr/bin/env python3
"""
Subtask 기반 Baseline X-VLA Inference 테스트

목표: Subtask instruction으로 baseline 모델이 어떻게 반응하는지 확인
- 전체 task: "open the middle drawer of the cabinet"
- Subtask: "Move the gripper to the handle of the middle drawer"

LIBERO 환경 없이 **모델 action 출력만 비교**

Usage:
    python scripts/test_subtask_inference.py
"""

import json
import numpy as np
from pathlib import Path
import h5py

# LeRobot X-VLA imports
try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    HAS_XVLA = True
except ImportError as e:
    print(f"⚠️ X-VLA not available: {e}")
    HAS_XVLA = False

import torch
from transformers import AutoTokenizer


# ImageNet stats for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_xvla_model(model_path="lerobot/xvla-libero"):
    """X-VLA 모델 로드"""
    print(f"🤖 Loading X-VLA from {model_path}...")
    
    policy = XVLAPolicy.from_pretrained(model_path)
    policy.eval()
    
    if torch.cuda.is_available():
        policy = policy.cuda()
        print("   ✅ Using CUDA")
    else:
        print("   ⚠️ Using CPU")
    
    return policy


def load_tokenizer(tokenizer_name="facebook/bart-large"):
    """Tokenizer 로드"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def tokenize_instruction(tokenizer, instruction, max_length=64):
    """언어 인스트럭션 토큰화"""
    tokens = tokenizer(
        instruction,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    return tokens.input_ids


def load_demo_frame(hdf5_path, demo_idx=0, frame_idx=0):
    """HDF5에서 데모 프레임 로드"""
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][f'demo_{demo_idx}']
        
        # 이미지
        agentview = demo['obs']['agentview_rgb'][frame_idx]  # (128, 128, 3)
        eye_in_hand = demo['obs']['eye_in_hand_rgb'][frame_idx]  # (128, 128, 3)
        
        # 로봇 상태
        ee_pos = demo['obs']['ee_pos'][frame_idx]  # (3,)
        ee_ori = demo['obs']['ee_ori'][frame_idx]  # (3,)
        gripper = demo['obs']['gripper_states'][frame_idx]  # (2,)
        
        # Ground truth action
        action_gt = demo['actions'][frame_idx]  # (7,)
        
    return {
        "agentview": agentview,
        "eye_in_hand": eye_in_hand,
        "state": np.concatenate([ee_pos, ee_ori, gripper[:1]]),
        "action_gt": action_gt,
    }


def preprocess_image(img):
    """이미지 전처리 (HWC uint8 -> CHW float normalized)"""
    # HWC -> CHW
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # [0, 255] -> [0, 1]
    img = img / 255.0
    # ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img - mean) / std
    return img


def predict_action(policy, obs, instruction, tokenizer):
    """X-VLA로 action 예측"""
    # 이미지 전처리 (ImageNet normalization)
    agentview = preprocess_image(obs["agentview"])
    eye_in_hand = preprocess_image(obs["eye_in_hand"])
    
    # 배치 차원 추가
    agentview = agentview.unsqueeze(0)
    eye_in_hand = eye_in_hand.unsqueeze(0)
    
    # 로봇 상태 (7D -> 20D로 패딩)
    state_7d = torch.from_numpy(obs["state"]).float()
    # X-VLA expects 20D state: [proprio (10), zeros (10)]
    state_20d = torch.zeros(20)
    state_20d[:7] = state_7d
    state = state_20d.unsqueeze(0)
    
    # 언어 토큰화
    input_ids = tokenize_instruction(tokenizer, instruction)
    
    device = next(policy.parameters()).device
    agentview = agentview.to(device)
    eye_in_hand = eye_in_hand.to(device)
    state = state.to(device)
    input_ids = input_ids.to(device)
    
    # Inference
    with torch.no_grad():
        batch = {
            "observation.images.image": agentview,  # Main camera (agentview)
            "observation.images.image2": eye_in_hand,  # Secondary camera (wrist)
            "observation.state": state,
            "observation.language.tokens": input_ids,
        }
        
        action = policy.select_action(batch)
    
    return action.cpu().numpy()[0]


def main():
    print("=" * 70)
    print("🧪 Subtask-based Baseline X-VLA Inference Test")
    print("=" * 70)
    
    if not HAS_XVLA:
        print("❌ X-VLA not available")
        return
    
    # HDF5 데이터 경로
    hdf5_path = Path("/workspace/vla_tset/data/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5")
    
    if not hdf5_path.exists():
        print(f"❌ HDF5 file not found: {hdf5_path}")
        return
    
    # Subtask 메타데이터 로드
    subtask_file = Path("/workspace/vla_tset/data/datasets/libero_goal_subtasks/open_the_middle_drawer_of_the_cabinet/subtask_metadata.json")
    if subtask_file.exists():
        with open(subtask_file) as f:
            metadata = json.load(f)
        subtasks = metadata["subtasks"]
    else:
        subtasks = ["Move the gripper to the handle of the middle drawer"]
    
    full_task = "open the middle drawer of the cabinet"
    first_subtask = subtasks[0]
    
    print(f"\n📋 Full Task: {full_task}")
    print(f"📋 First Subtask: {first_subtask}")
    print(f"📋 All Subtasks: {subtasks}")
    
    # Tokenizer 로드
    print("\n📝 Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # 모델 로드
    policy = load_xvla_model()
    
    # 여러 프레임에서 테스트
    print("\n" + "=" * 70)
    print("🔬 Comparing Actions: Full Task vs Subtask Instruction")
    print("=" * 70)
    
    print(f"\n{'Frame':<6} {'GT Action (pos)':<25} {'Full Task':<25} {'Subtask':<25} {'Diff':<10}")
    print("-" * 91)
    
    diffs = []
    
    for frame_idx in [0, 10, 20, 30, 40]:
        try:
            obs = load_demo_frame(hdf5_path, demo_idx=0, frame_idx=frame_idx)
            
            # Full task instruction으로 예측
            action_full = predict_action(policy, obs, full_task, tokenizer)
            
            # Subtask instruction으로 예측
            action_subtask = predict_action(policy, obs, first_subtask, tokenizer)
            
            # Ground truth
            action_gt = obs["action_gt"]
            
            # Position 부분만 비교 (처음 3개)
            diff = np.linalg.norm(action_full[:3] - action_subtask[:3])
            diffs.append(diff)
            
            print(f"{frame_idx:<6} {str(action_gt[:3].round(3)):<25} {str(action_full[:3].round(3)):<25} {str(action_subtask[:3].round(3)):<25} {diff:.4f}")
            
        except Exception as e:
            import traceback
            print(f"Frame {frame_idx}: Error - {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    
    if diffs:
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        
        print(f"Average action difference: {avg_diff:.4f}")
        print(f"Max action difference: {max_diff:.4f}")
        
        if avg_diff < 0.01:
            print("\n✅ Very small difference! Subtask instruction produces similar actions.")
            print("   → Baseline model might generalize to subtask instructions!")
        elif avg_diff < 0.1:
            print("\n⚠️ Moderate difference. Some generalization possible.")
            print("   → Fine-tuning recommended for better subtask performance.")
        else:
            print("\n❌ Large difference! Baseline doesn't understand subtask instructions.")
            print("   → Fine-tuning required for subtask-based inference.")


if __name__ == "__main__":
    main()
