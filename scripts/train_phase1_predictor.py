"""
Phase 1 ProgressPredictor 학습 스크립트.

VITA D.1 학습 방식:
    - 에피소드 전체(≤120프레임)를 배치로 구성
    - 매 timestep 순차 TTT 업데이트 → train/inference mismatch 없음
    - ℓpred: dissimilarity sampling으로 선택된 k=8 window frame에서만 계산
    - 5 epochs ≈ ~470 gradient steps

meta_forward()로 inner update를 통한 computational graph 유지 (MAML 방식).
ℓpred gradient가 TTT inner update 전체를 differentiating through 역전파됨.
"""

import argparse
import math
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ttt.predictor import ProgressPredictor
from src.datasets.phase1_dataset import Phase1EpisodicDataset

MAX_T = Phase1EpisodicDataset.MAX_EP_LEN  # 120


def collate_fn(batch):
    """
    에피소드를 배치 내 최대 길이(≤120)로 패딩.

    Returns:
        z_seq:     [B, T, D]
        targets:   [B, T, 1]
        pred_mask: [B, T] bool — ℓpred 계산 frame
        pad_mask:  [B, T] bool — valid (non-padded) frame
    """
    B = len(batch)
    D = batch[0]["z_seq"].shape[-1]
    max_T_batch = min(max(b["ep_len"] for b in batch), MAX_T)

    z_seqs    = torch.zeros(B, max_T_batch, D)
    targets   = torch.zeros(B, max_T_batch, 1)
    pred_masks = torch.zeros(B, max_T_batch, dtype=torch.bool)
    pad_masks  = torch.zeros(B, max_T_batch, dtype=torch.bool)

    for i, b in enumerate(batch):
        T = min(b["ep_len"], max_T_batch)
        z_seqs[i, :T]     = b["z_seq"][:T]
        targets[i, :T]    = b["targets"][:T]
        pred_masks[i, :T] = b["pred_mask"][:T]
        pad_masks[i, :T]  = True

    return {
        "z_seq":      z_seqs,
        "targets":    targets,
        "pred_mask":  pred_masks,
        "pad_mask":   pad_masks,
    }


def forward_pass(predictor, batch, device, lambda_self, create_graph=True):
    z_seq      = batch["z_seq"].to(device)      # [B, T, D]
    targets    = batch["targets"].to(device)    # [B, T, 1]
    pred_mask  = batch["pred_mask"].to(device)  # [B, T] bool
    pad_mask   = batch["pad_mask"].to(device)   # [B, T] bool

    z_seq_t    = z_seq.permute(1, 0, 2).contiguous()  # [T, B, D]
    valid_mask = pad_mask.permute(1, 0)                # [T, B] bool

    result = predictor.meta_forward(
        z_seq_t, create_graph=create_graph, valid_mask=valid_mask
    )

    progress_seq = result["progress_seq"]   # [T, B, 1]
    targets_t    = targets.permute(1, 0, 2) # [T, B, 1]
    pred_mask_t  = pred_mask.permute(1, 0)  # [T, B]

    # ℓpred: dissimilarity-selected window frame에서만 계산 (VITA Eq.5)
    pred_selected = progress_seq[pred_mask_t]   # [N, 1]
    tgt_selected  = targets_t[pred_mask_t]      # [N, 1]
    pred_loss = F.mse_loss(pred_selected, tgt_selected)

    # ℓself: 유효 frame에서의 SSL loss 평균 (패딩은 0으로 처리됨)
    ssl_loss = torch.stack(result["ssl_losses"]).mean()

    loss = pred_loss + lambda_self * ssl_loss
    return loss, pred_loss, ssl_loss


def validate(predictor, val_loader, device, lambda_self, val_steps=50):
    """val_loader에서 val_steps 배치만큼 loss를 측정."""
    predictor.eval()
    total_loss = total_pred = total_ssl = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= val_steps:
            break
        loss, pred_loss, ssl_loss = forward_pass(
            predictor, batch, device, lambda_self, create_graph=False
        )
        total_loss += loss.item()
        total_pred += pred_loss.item()
        total_ssl  += ssl_loss.item()
        n += 1
    predictor.train()
    return {
        "val/loss":      total_loss / max(n, 1),
        "val/pred_loss": total_pred / max(n, 1),
        "val/ssl_loss":  total_ssl  / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ProgressPredictor training")

    # Data
    parser.add_argument("--data_root", type=str, default="data/bridge_v2_lerobot")
    parser.add_argument("--repo_id", type=str, default="FedorX8/bridge_v2_lerobot")
    parser.add_argument("--image_key", type=str, default="observation.images.primary")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_windows_per_episode", type=int, default=8,
                        help="dissimilarity sampling k (VITA=8)")
    parser.add_argument("--train_episodes", type=int, default=2986)
    parser.add_argument("--val_episodes", type=int, default=287)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--embed_cache_path", type=str,
                        default="data/bridge_v2_lerobot_clip_embeddings.pt")

    # Model
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--inner_model_type", type=str, default="mlp",
                        choices=["linear", "mlp"])
    parser.add_argument("--head_hidden_dim", type=int, default=128)
    parser.add_argument("--eta_base", type=float, default=0.1)

    # Training
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="학습 epoch 수 (VITA=5)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_self", type=float, default=0.5)
    parser.add_argument("--val_steps", type=int, default=50,
                        help="validation 시 사용할 배치 수")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embed_device", type=str, default="cuda")

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--wandb_project", type=str, default="temporal-vla")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    args.save_dir = os.path.join(args.save_dir, date_str)
    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"phase1_window_w{args.window_size}_k{args.max_windows_per_episode}",
        config=vars(args),
    )

    # ──────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────
    dataset_kwargs = dict(
        data_root=args.data_root,
        repo_id=args.repo_id,
        window_size=args.window_size,
        max_windows_per_episode=args.max_windows_per_episode,
        image_key=args.image_key,
        embed_device=args.embed_device,
        embed_cache_path=args.embed_cache_path,
    )

    total_needed = args.train_episodes + args.val_episodes
    all_indices = list(range(total_needed))
    random.seed(args.split_seed)
    random.shuffle(all_indices)
    train_indices = all_indices[:args.train_episodes]
    val_indices   = all_indices[args.train_episodes:]

    print("Loading train dataset...")
    train_ds = Phase1EpisodicDataset(episode_indices=train_indices, **dataset_kwargs)
    print(f"Train: {len(train_ds)} episodes")

    print("Loading val dataset...")
    val_ds = Phase1EpisodicDataset(episode_indices=val_indices, **dataset_kwargs)
    print(f"Val:   {len(val_ds)} episodes")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    steps_per_epoch = len(train_loader)
    total_steps     = args.n_epochs * steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}, total: {total_steps}")

    # ──────────────────────────────────────────────
    # Model
    # ──────────────────────────────────────────────
    predictor = ProgressPredictor(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        inner_model_type=args.inner_model_type,
        eta_base=args.eta_base,
        learnable_eta=False,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"ProgressPredictor params: {n_params:,}")
    wandb.config.update({"n_params": n_params, "steps_per_epoch": steps_per_epoch})

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine LR schedule with 10% linear warmup (VITA Appendix D.1)
    warmup_steps = int(total_steps * 0.1)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch  = 0
    global_step  = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        predictor.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = start_epoch * steps_per_epoch
        print(f"Resumed from {args.resume_ckpt} (epoch {start_epoch})")

    # ──────────────────────────────────────────────
    # Training loop (epoch 기반, VITA D.1: 5 epochs)
    # ──────────────────────────────────────────────
    predictor.train()
    for epoch in range(start_epoch, args.n_epochs):
        for batch in train_loader:
            loss, pred_loss, ssl_loss = forward_pass(
                predictor, batch, device, args.lambda_self
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch+1}/{args.n_epochs} step {global_step}] "
                    f"loss={loss.item():.4f}  pred={pred_loss.item():.4f}  "
                    f"ssl={ssl_loss.item():.4f}  lr={current_lr:.2e}"
                )
                wandb.log({
                    "train/loss":      loss.item(),
                    "train/pred_loss": pred_loss.item(),
                    "train/ssl_loss":  ssl_loss.item(),
                    "train/lr":        current_lr,
                    "train/epoch":     epoch + 1,
                }, step=global_step)

        # ── Epoch 종료: validation + 체크포인트 저장 ──
        val_metrics = validate(predictor, val_loader, device, args.lambda_self, args.val_steps)
        wandb.log(val_metrics, step=global_step)
        print(
            f"[val  epoch {epoch+1}] "
            f"loss={val_metrics['val/loss']:.4f}  "
            f"pred={val_metrics['val/pred_loss']:.4f}  "
            f"ssl={val_metrics['val/ssl_loss']:.4f}"
        )

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pt")
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args":                 vars(args),
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

    # ── 최종 저장 ──
    val_metrics = validate(predictor, val_loader, device, args.lambda_self, args.val_steps)
    wandb.log(val_metrics, step=global_step)
    print(f"[final val] loss={val_metrics['val/loss']:.4f}")

    predictor.save_init()
    final_path = os.path.join(args.save_dir, "phase1_final.pt")
    torch.save(predictor.state_dict(), final_path)
    print(f"Phase 1 complete. Final: {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
