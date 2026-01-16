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
        --dataset_path data/datasets/hf_drawer_time \
        --output_dir outputs/train/xvla_time_drawer \
        --num_epochs 5
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# LeRobot imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, XVLAModel
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


def collate_fn(batch, tokenizer, max_length=64, image_size=224):
    """배치 데이터 처리 - XVLAModel.forward 인터페이스에 맞춤"""
    import numpy as np
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF
    
    # XVLA 설정
    chunk_size = 30
    max_proprio_dim = 20
    max_action_dim = 20
    
    # 이미지 처리 - XVLA는 224x224 입력 필요
    images = []
    for item in batch:
        img = item['observation.images.agentview']
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        # (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Resize to 224x224
        img = TF.resize(img, [image_size, image_size])
        # ImageNet normalization
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images.append(img)
    
    # [B, C, H, W] -> [B, 1, C, H, W] (single image input)
    images = torch.stack(images).unsqueeze(1)
    
    # Image mask (all ones since we have valid images)
    image_mask = torch.ones(images.shape[0], 1, dtype=torch.bool)
    
    # 로봇 상태 (proprio) - [B, 7] -> [B, 20] with padding
    proprio_raw = torch.tensor([item['observation.state'] for item in batch], dtype=torch.float32)
    proprio = torch.zeros(len(batch), max_proprio_dim, dtype=torch.float32)
    proprio[:, :proprio_raw.shape[1]] = proprio_raw
    
    # Action - [B, chunk_size, action_dim]
    # 7차원 action을 20차원으로 패딩
    actions_raw = torch.tensor([item['action'] for item in batch], dtype=torch.float32)  # [B, 7]
    actions = torch.zeros(len(batch), chunk_size, max_action_dim, dtype=torch.float32)
    actions[:, :, :actions_raw.shape[1]] = actions_raw.unsqueeze(1).repeat(1, chunk_size, 1)
    
    # Task 텍스트 토큰화
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
    time_to_go = torch.tensor([item['time_to_go'] for item in batch], dtype=torch.float32)
    
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
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
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
            
            # Loss 계산: action_loss + time_loss
            action_loss = outputs.get('loss', torch.tensor(0.0, device=device))
            time_loss = outputs.get('time_loss', torch.tensor(0.0, device=device))
            
            # Total loss = action_loss + time_loss
            loss = action_loss + time_loss
            
            # Gradient accumulation
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
        
        pbar.set_postfix({
            'loss': f"{total_loss / (step + 1):.4f}",
            'action': f"{total_action_loss / (step + 1):.4f}",
            'time': f"{total_time_loss / (step + 1):.4f}",
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'action_loss': total_action_loss / len(dataloader),
        'time_loss': total_time_loss / len(dataloader),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XVLA with time prediction")
    parser.add_argument("--dataset_path", type=str, default="data/datasets/hf_drawer_time")
    parser.add_argument("--output_dir", type=str, default="outputs/train/xvla_time_drawer")
    parser.add_argument("--base_model", type=str, default="lerobot/xvla-libero")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_token_length", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # 데이터셋 로드
    print(f"📂 Loading dataset from {args.dataset_path}...")
    dataset = DatasetDict.load_from_disk(args.dataset_path)
    train_data = dataset['train']
    print(f"   Samples: {len(train_data)}")
    
    # 토크나이저 로드 (XVLA는 facebook/bart-large 토크나이저 사용)
    print(f"🔧 Loading tokenizer (facebook/bart-large)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    
    # 모델 로드
    print(f"📦 Loading model from {args.base_model}...")
    policy = XVLAPolicy.from_pretrained(args.base_model)
    model = policy.model
    model = model.to(device)
    
    # 전체 파라미터 학습 (Server A100용)
    # PC2 (12GB VRAM)에서는 아래 코드로 time_decoder만 학습:
    # for name, param in model.named_parameters():
    #     if 'time_decoder' not in name and 'var_decoder' not in name:
    #         param.requires_grad = False
    
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable_count:,} ({trainable_count / total_count * 100:.2f}%)")
    
    # DataLoader
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_token_length),
        num_workers=0,  # 메모리 절약
    )
    
    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation}")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"   Learning rate: {args.learning_rate}")
    
    # 학습 로그
    train_logs = []
    
    for epoch in range(1, args.num_epochs + 1):
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
        
        # 체크포인트 저장
        if epoch % args.save_every == 0:
            checkpoint_dir = output_dir / f"checkpoint-{epoch}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # 모델 저장
            policy.model = model
            policy.save_pretrained(str(checkpoint_dir))
            
            print(f"   💾 Saved checkpoint to {checkpoint_dir}")
    
    # 최종 모델 저장
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    policy.model = model
    policy.save_pretrained(str(final_dir))
    
    # 학습 로그 저장
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_logs, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Final model: {final_dir}")
    print(f"   Train log: {output_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()
