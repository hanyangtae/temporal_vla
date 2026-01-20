#!/usr/bin/env python3
"""
XVLA Time-Aware Fine-tuning 스크립트

PC2 (RTX 3060 12GB)에서 실행하기 위해 메모리 최적화:
- Batch size: 1
- Gradient accumulation: 8
- Mixed precision: fp16
- Gradient checkpointing: True

Usage:
    python scripts/finetune_xvla_time.py \
        --dataset_path data/datasets/lerobot_libero_goal \
        --output_dir outputs/train/xvla_time_drawer \
        --num_epochs 5
"""

import argparse
import os
from pathlib import Path
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

# LeRobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, XVLAModel
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def collate_fn(batch, tokenizer, max_length=64, image_size=224, target_time_key='time_to_go_subtask'):
    """배치 데이터 처리 - XVLAModel.forward 인터페이스에 맞춤"""
    from scipy.spatial.transform import Rotation
    import torchvision.transforms.functional as TF
    
    # XVLA 설정 상수
    max_proprio_dim = 20
    max_action_dim = 20
    
    # 이미지 처리 - XVLA는 224x224 입력 필요
    # LeRobotDataset은 video/image를 (C, H, W) Tensor로 반환함
    images = []
    for item in batch:
        # Agentview image (C, H, W) float32 [0, 1]
        img = item['observation.images.agentview']
        
        # Resize if needed (LIBERO is 128x128 or 256x256 -> XVLA needs 224x224)
        if img.shape[1] != image_size or img.shape[2] != image_size:
            img = TF.resize(img, [image_size, image_size], antialias=True)
            
        # ImageNet normalization
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images.append(img)
    
    # [B, C, H, W] -> [B, 1, C, H, W] (single image input)
    images = torch.stack(images).unsqueeze(1)
    
    # Image mask (all ones since we have valid images)
    image_mask = torch.ones(images.shape[0], 1, dtype=torch.bool)
    
    # 로봇 상태 (proprio) - [B, 7] -> [B, 20] with padding
    proprio_raw = torch.stack([item['observation.state'] for item in batch]) # [B, 7]
    proprio = torch.zeros(len(batch), max_proprio_dim, dtype=torch.float32)
    proprio[:, :proprio_raw.shape[1]] = proprio_raw
    
    # -------------------------------------------------------------------------
    # Action 처리: [B, Chunk, 7] (LIBERO) -> [B, Chunk, 20] (XVLA)
    # -------------------------------------------------------------------------
    
    # LeRobotDataset with delta_timestamps returns [B, Chunk, 7]
    actions_raw = torch.stack([item['action'] for item in batch]) # [B, Chunk, 7]
    batch_size, chunk_len, _ = actions_raw.shape
    
    # Flatten for batch processing: [B*Chunk, 7]
    flat_actions = actions_raw.reshape(-1, 7).numpy()
    
    # 1. Position [B*Chunk, 3]
    pos = flat_actions[:, :3]
    
    # 2. Rotation: Axis-Angle(3) -> Rotation Matrix(3x3) -> 6D Rotation(6)
    axis_angle = flat_actions[:, 3:6]
    rot_mat = Rotation.from_rotvec(axis_angle).as_matrix() # [N, 3, 3]
    rot_6d = rot_mat[:, :, :2].reshape(-1, 6) # [N, 6]
    
    # 3. Gripper: [-1, 1] -> [0, 1]
    gripper = flat_actions[:, 6:7]
    gripper = (gripper + 1.0) / 2.0
    
    # 4. Concatenate: [pos(3), rot6d(6), gripper(1)] -> 10-dim
    action_10d = np.concatenate([pos, rot_6d, gripper], axis=-1)
    
    # 5. Pad to 20-dim & Reshape
    actions = torch.zeros(batch_size, chunk_len, max_action_dim, dtype=torch.float32)
    action_tensor_10d = torch.from_numpy(action_10d).float().reshape(batch_size, chunk_len, 10)
    actions[:, :, :10] = action_tensor_10d
    
    # -------------------------------------------------------------------------

    # Task 텍스트 토큰화
    # LeRobotDataset conversion에서 이미 'task' 필드에 "Task: Subtask" 형태로 저장함
    tasks = [item['task'] for item in batch]
    
    tokens = tokenizer(
        tasks,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )
    
    # Domain ID (0 for LIBERO)
    domain_id = torch.zeros(len(batch), dtype=torch.long)
    
    # Time-to-go [B]
    time_to_go = torch.tensor([
        item[target_time_key].item() # Tensor(1) -> float
        for item in batch
    ], dtype=torch.float32)
    
    return {
        'input_ids': tokens['input_ids'],
        'image_input': images,
        'image_mask': image_mask,
        'domain_id': domain_id,
        'proprio': proprio,
        'action': actions,
        'time_to_go': time_to_go,
    }


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, gradient_accumulation_steps=8):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_time_loss = 0
    
    optimizer.zero_grad()
    
    # pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    # Log less frequently to reduce log file size
    print(f"Epoch {epoch} started...")
    
    for step, batch in enumerate(dataloader):
        # 데이터를 GPU로 이동
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids=batch['input_ids'],
                image_input=batch['image_input'],
                image_mask=batch['image_mask'],
                domain_id=batch['domain_id'],
                proprio=batch['proprio'],
                action=batch['action'],
                time_to_go=batch['time_to_go'],
            )
            
            # Loss 계산
            position_loss = outputs.get('position_loss', torch.tensor(0.0, device=device))
            rotate_loss = outputs.get('rotate6D_loss', torch.tensor(0.0, device=device))
            gripper_loss = outputs.get('gripper_loss', torch.tensor(0.0, device=device))
            action_loss = position_loss + rotate_loss + gripper_loss
            
            time_loss = outputs.get('time_loss', torch.tensor(0.0, device=device))
            loss = action_loss + time_loss
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient step
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_action_loss += action_loss.item() if isinstance(action_loss, torch.Tensor) else action_loss
        total_time_loss += time_loss.item() if isinstance(time_loss, torch.Tensor) else time_loss
        
        # Log every 1 steps (or more)
        if (step + 1) % 1 == 0:
            print(f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | "
                  f"Loss: {total_loss / (step + 1):.4f} | "
                  f"Action: {total_action_loss / (step + 1):.4f} | "
                  f"Time: {total_time_loss / (step + 1):.4f}")
    
    return {
        'loss': total_loss / len(dataloader),
        'action_loss': total_action_loss / len(dataloader),
        'time_loss': total_time_loss / len(dataloader),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XVLA with time prediction")
    parser.add_argument("--dataset_path", type=str, default="data/datasets/lerobot_libero_goal")
    parser.add_argument("--repo_id", type=str, default="libero/goal_subtask_time")
    parser.add_argument("--output_dir", type=str, default="outputs/train/xvla_time_drawer")
    parser.add_argument("--base_model", type=str, default="lerobot/xvla-libero")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_token_length", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ALL parameters except time/var decoders (Head Only)")
    parser.add_argument("--use_peft", action="store_true", help="Freeze VLM backbone only. Train SoftPrompt + Transformer + Time/Var Decoders.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--target_time_key", type=str, default="time_to_go_subtask", 
                        choices=["time_to_go_subtask", "time_to_go_full"],
                        help="Key for time target (subtask or full episode)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # 모델 로드 (Config 확인을 위해 먼저 로드)
    print(f"📦 Loading model from {args.base_model}...")
    if args.resume_from_checkpoint:
        policy = XVLAPolicy.from_pretrained(args.resume_from_checkpoint)
    else:
        policy = XVLAPolicy.from_pretrained(args.base_model)
    
    # Chunk Size 확인
    chunk_size = policy.config.chunk_size
    print(f"🧩 Using chunk size from config: {chunk_size}")

    # LeRobotDataset 로드
    # delta_timestamps를 사용하여 미래의 Action Chunk를 자동으로 로드
    # fps=10.0 가정
    print(f"📂 Loading LeRobotDataset from {args.dataset_path}...")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_path,
        video_backend="pyav",  # Force PyAV backend due to torchcodec issues
        delta_timestamps={
            "action": [i / 10.0 for i in range(chunk_size)]
        }
    )
    print(f"   Samples: {len(dataset)}")
    
    # 토크나이저 로드
    print(f"🔧 Loading tokenizer (facebook/bart-large)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    
    model = policy.model
    model = model.to(device)
    
    # Optimize speed with torch.compile
    # (Optional: PyTorch 2.0+ required)
    if hasattr(torch, "compile"):
        try:
            print("🚀 Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")

    if args.freeze_backbone:
        print("❄️ Freezing backbone (training only time/var decoders)...")
        for name, param in model.named_parameters():
            if 'time_decoder' not in name and 'var_decoder' not in name:
                param.requires_grad = False
    
    elif args.use_peft:
        print("❄️ Using PEFT (Freezing VLM, Training Policy & SoftPrompts)...")
        # VLM (Vision + Language)만 Freeze
        # XVLAModel.vlm ... -> Freeze
        # XVLAModel.transformer ... -> Train (Default)
        for name, param in model.named_parameters():
            # 정확히 VLM Backbone만 타겟팅 (startswith 사용)
            # "vlm" in name은 transformer.vlm_proj도 포함해버리므로 위험함
            if name.startswith("vlm."):
                param.requires_grad = False
            # 나머지는 기본적으로 requires_grad=True
            
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable_count:,} ({trainable_count / total_count * 100:.2f}%)")
    
    # DataLoader Optimization for High-RAM Environment (1TB)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_token_length, target_time_key=args.target_time_key),
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=8,       # Prefetch 8 batches per worker (32 workers * 8 * 512 batch ~ 130k samples buffer)
        persistent_workers=True, # Eliminate worker startup overhead
    )
    
    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n🚀 Starting training...")
    
    start_epoch = 1
    if args.resume_from_checkpoint:
        try:
            start_epoch = int(args.resume_from_checkpoint.split('-')[-1]) + 1
        except:
            pass

    # 학습 로그
    train_logs = []
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        metrics = train_epoch(
            model, dataloader, optimizer, scaler, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation
        )
        
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Loss: {metrics['loss']:.4f}")
        print(f"   Action Loss: {metrics['action_loss']:.4f}")
        print(f"   Time Loss: {metrics['time_loss']:.4f}")
        
        train_logs.append({
            'epoch': epoch,
            **metrics
        })
        
        if epoch % args.save_every == 0:
            checkpoint_dir = output_dir / f"checkpoint-{epoch}"
            checkpoint_dir.mkdir(exist_ok=True)
            policy.model = model
            policy.save_pretrained(str(checkpoint_dir))
            print(f"   💾 Saved checkpoint to {checkpoint_dir}")
    
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    policy.model = model
    policy.save_pretrained(str(final_dir))
    
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_logs, f, indent=2)
    
    print(f"\n✅ Training complete!")


if __name__ == "__main__":
    main()
