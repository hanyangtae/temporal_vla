"""
Phase 1 (GR00T × RoboCasa) — ProgressPredictor 사전 학습.

Stage 0 가 미리 추출한 Eagle pre-LLM 임베딩(`data/robocasa_eagle_pre_llm/{task}/embeddings.pt`)
을 입력으로, ``src/ttt/predictor.ProgressPredictor`` 를 학습한다. Stage 2 에서
DiT cross-attention KV 에 token 으로 직접 주입하기 위해 **input_dim/proj_dim=2048
고정** (Eagle Qwen3-1.7B hidden = DiT KV `backbone_embedding_dim`; 추가 projector 없음).

기존 `scripts/train/phase1_predictor.py` 의 학습 루프를 그대로 재사용하고,
dataset 구성만 task 별 ConcatDataset 으로 교체했다. dissimilarity 기반 frame
샘플링, meta_forward 시퀀셜 TTT, MSE(progress, t/T) + λ_self·SSL 손실 모두 동일.

Run inside the lerobot container (TTT는 GPU 만 있으면 됨, GR00T 의존성 불필요)::

    docker compose exec lerobot bash -lc \\
        'cd /temporal_vla && python scripts/train/phase1_groot_robocasa.py \\
            --tasks OpenDrawer'   # 1 task smoke test
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import wandb

# repo root 를 PYTHONPATH 에
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.path_setup import DATA_ROOT
from src.datasets.phase1_v21_dataset import Phase1V21EpisodicDataset
from src.ttt.predictor import ProgressPredictor


REPO_ROOT = Path("/temporal_vla")
DEFAULT_DATA_ROOT = DATA_ROOT / "robocasa/v1.0/pretrain/atomic"
DEFAULT_CACHE_ROOT = DATA_ROOT / "robocasa_eagle_pre_llm"
DEFAULT_SAVE_DIR = REPO_ROOT / "outputs/train/phase1_groot_robocasa"

DEFAULT_TASKS = [
    "OpenDrawer", "CloseDrawer",
    "OpenCabinet", "CloseCabinet",
    "OpenFridge", "CloseFridge",
    "OpenMicrowave", "CloseMicrowave",
    "PickPlaceCounterToStove", "PickPlaceCounterToSink",
]

DEFAULT_MAX_EP_LEN = Phase1V21EpisodicDataset.DEFAULT_MAX_EP_LEN  # 485 (RoboCasa atomic p99)


# ──────────────────────────────────────────────────────────────────────────
# Dataset 구성
# ──────────────────────────────────────────────────────────────────────────


def find_lerobot_root(data_root: Path, task: str) -> Path:
    matches = sorted((data_root / task).glob("*/lerobot"))
    if not matches:
        raise FileNotFoundError(f"No lerobot/ under {data_root / task}")
    return matches[0]  # 한 task 당 한 date 가 일반적


def build_per_task_dataset(
    task: str,
    data_root: Path,
    cache_root: Path,
    window_size: int,
    max_windows_per_episode: int,
    train_frac: float,
    split_seed: int,
    max_ep_len: int,
    max_episodes_per_task: int | None = None,
) -> tuple[Phase1V21EpisodicDataset, Phase1V21EpisodicDataset]:
    """task 한 개에 대해 (train_ds, val_ds) 를 구성. v2.1 호환 — RoboCasaV21Reader 사용."""
    cache_path = cache_root / task / "embeddings.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Eagle pre-LLM cache not found: {cache_path}. "
            f"Run scripts/extract/extract_eagle_pre_llm_robocasa.py first."
        )
    lerobot_root = find_lerobot_root(data_root, task)

    # cache 에 들어있는 frame 으로 eligible episode 추출 (Phase1V21EpisodicDataset 가
    # episode_indices=None 이면 자체적으로 같은 필터를 수행하므로, 여기서는 split 만 책임).
    cache = torch.load(str(cache_path), map_location="cpu", weights_only=True)
    cached_indices = set(int(k) for k in cache.keys())
    del cache

    from src.datasets.robocasa_v21_reader import RoboCasaV21Reader
    reader = RoboCasaV21Reader(lerobot_root)
    all_eps = reader.lerobot_meta_episodes()
    eligible = [
        ep_idx for ep_idx, ep in all_eps.items()
        if ep["dataset_from_index"] in cached_indices
    ]
    # episode index 기준 앞에서 N 개만 사용 (target 데이터셋이 500+ ep 라 cap 필요).
    if max_episodes_per_task is not None and max_episodes_per_task > 0:
        eligible = [e for e in eligible if e < max_episodes_per_task]
    rng = random.Random(split_seed)
    rng.shuffle(eligible)
    n_train = int(len(eligible) * train_frac)
    train_eps = sorted(eligible[:n_train])
    val_eps = sorted(eligible[n_train:])

    common = dict(
        lerobot_root=str(lerobot_root),
        embed_cache_path=str(cache_path),
        window_size=window_size,
        max_windows_per_episode=max_windows_per_episode,
        max_ep_len=max_ep_len,
    )
    train_ds = Phase1V21EpisodicDataset(episode_indices=train_eps, **common)
    val_ds = Phase1V21EpisodicDataset(episode_indices=val_eps, **common)
    print(f"[{task}] train={len(train_ds)} val={len(val_ds)} (cached_eps={len(eligible)})")
    return train_ds, val_ds


def build_concat_datasets(
    tasks: list[str],
    data_root: Path,
    cache_root: Path,
    window_size: int,
    max_windows_per_episode: int,
    train_frac: float,
    split_seed: int,
    max_ep_len: int,
    max_episodes_per_task: int | None = None,
) -> tuple[ConcatDataset, ConcatDataset]:
    train_parts, val_parts = [], []
    for task in tasks:
        tr, va = build_per_task_dataset(
            task, data_root, cache_root, window_size, max_windows_per_episode,
            train_frac, split_seed, max_ep_len,
            max_episodes_per_task=max_episodes_per_task,
        )
        train_parts.append(tr)
        val_parts.append(va)
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


# ──────────────────────────────────────────────────────────────────────────
# Train step (phase1_predictor 와 동일)
# ──────────────────────────────────────────────────────────────────────────


def make_collate_fn(max_ep_len: int):
    """max_ep_len 을 closure 로 받아 padding 길이 동적 결정."""
    def collate_fn(batch):
        B = len(batch)
        D = batch[0]["z_seq"].size(-1)

        z_seq = torch.zeros(B, max_ep_len, D)
        targets = torch.zeros(B, max_ep_len, 1)
        pred_mask = torch.zeros(B, max_ep_len, dtype=torch.bool)
        valid_mask = torch.zeros(B, max_ep_len, dtype=torch.bool)

        for i, b in enumerate(batch):
            T = b["ep_len"]
            z_seq[i, :T] = b["z_seq"]
            targets[i, :T] = b["targets"]
            pred_mask[i, :T] = b["pred_mask"]
            valid_mask[i, :T] = True

        return {
            "z_seq": z_seq, "targets": targets,
            "pred_mask": pred_mask, "valid_mask": valid_mask,
        }
    return collate_fn


def forward_pass(predictor, batch, device, lambda_self, create_graph=True):
    z_seq = batch["z_seq"].to(device)
    targets = batch["targets"].to(device)
    pred_mask = batch["pred_mask"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    z_seq_t = z_seq.permute(1, 0, 2).contiguous()
    valid_mask_t = valid_mask.permute(1, 0).contiguous()

    result = predictor.meta_forward(
        z_seq_t, create_graph=create_graph, valid_mask=valid_mask_t
    )

    progress_seq = result["progress_seq"]
    targets_t = targets.permute(1, 0, 2)
    pred_mask_t = pred_mask.permute(1, 0)

    pred_loss = F.mse_loss(
        progress_seq[pred_mask_t.unsqueeze(-1).expand_as(progress_seq)],
        targets_t[pred_mask_t.unsqueeze(-1).expand_as(targets_t)],
    )

    ssl_losses = torch.stack(result["ssl_losses"])
    n_valid_steps = valid_mask_t.any(dim=1).sum()
    ssl_loss = ssl_losses.sum() / n_valid_steps.clamp(min=1)

    loss = pred_loss + lambda_self * ssl_loss
    return loss, pred_loss, ssl_loss


def validate(predictor, val_loader, device, lambda_self, val_steps=50):
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
        total_ssl += ssl_loss.item()
        n += 1
    predictor.train()
    return {
        "val/loss": total_loss / max(n, 1),
        "val/pred_loss": total_pred / max(n, 1),
        "val/ssl_loss": total_ssl / max(n, 1),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 1 (GR00T × RoboCasa) ProgressPredictor 학습")

    # Data
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_windows_per_episode", type=int, default=8)
    parser.add_argument("--max_ep_len", type=int, default=DEFAULT_MAX_EP_LEN,
                        help="Episode padding 한도. RoboCasa atomic 은 p99=485, max=618. "
                             "VITA paper(BridgeData) 는 120 이었음.")
    parser.add_argument("--train_frac", type=float, default=0.9,
                        help="task 별 episode 의 train 비율 (나머지 val)")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_episodes_per_task", type=int, default=None,
                        help="task 별 사용할 episode index 상한 (ep_idx < N 만 사용). "
                             "target 데이터셋은 task 당 500+ episode 라 cap 권장.")

    # Model — proj_dim=2048 고정 (Stage 2 에서 DiT KV 직접 주입)
    parser.add_argument("--input_dim", type=int, default=2048,
                        help="Eagle pre-LLM hidden dim (= 2048, Qwen3-1.7B hidden_size)")
    parser.add_argument("--proj_dim", type=int, default=2048,
                        help="TTT inner dim. 2048 고정 — 학습된 표현이 곧바로 DiT KV(=backbone_embedding_dim) 토큰이 됨")
    parser.add_argument("--inner_model_type", type=str, default="linear",
                        choices=["linear", "mlp", "linear_preLN"])
    parser.add_argument("--head_hidden_dim", type=int, default=128)
    parser.add_argument("--eta_base", type=float, default=0.1)

    # Training
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_self", type=float, default=0.5)
    parser.add_argument("--val_steps", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--wandb_project", type=str, default="temporal-vla")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_suffix", type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.proj_dim != 2048:
        print(
            f"[warn] proj_dim={args.proj_dim} != 2048. Stage 2 wrapper 가 DiT KV 에 token "
            f"으로 직접 prepend 하므로 2048 (= Eagle/DiT KV dim) 권장."
        )

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    if args.save_suffix:
        date_str = f"{date_str}_{args.save_suffix}"
    save_dir = args.save_dir / date_str
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"phase1_groot_robocasa_{len(args.tasks)}tasks",
            config=vars(args),
        )

    # ──────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────
    print(f"[data] tasks={args.tasks}")
    train_ds, val_ds = build_concat_datasets(
        tasks=args.tasks,
        data_root=args.data_root,
        cache_root=args.cache_root,
        window_size=args.window_size,
        max_windows_per_episode=args.max_windows_per_episode,
        train_frac=args.train_frac,
        split_seed=args.split_seed,
        max_ep_len=args.max_ep_len,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    print(f"[data] total train={len(train_ds)} val={len(val_ds)} episodes "
          f"(max_ep_len={args.max_ep_len})")

    collate_fn = make_collate_fn(args.max_ep_len)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    steps_per_epoch = len(train_loader)
    total_steps = args.n_epochs * steps_per_epoch
    print(f"[steps] per_epoch={steps_per_epoch} total={total_steps} epochs={args.n_epochs}")

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
    print(f"[model] ProgressPredictor params: {n_params:,}")
    wandb.config.update({
        "n_params": n_params,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }, allow_val_change=True)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        predictor.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] from {args.resume_ckpt} (step {global_step})")

    # ──────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────
    predictor.train()
    for epoch in range(args.n_epochs):
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
                    f"[ep {epoch+1} step {global_step}/{total_steps}] "
                    f"loss={loss.item():.4f} pred={pred_loss.item():.4f} "
                    f"ssl={ssl_loss.item():.4f} lr={current_lr:.2e}"
                )
                wandb.log({
                    "train/loss": loss.item(),
                    "train/pred_loss": pred_loss.item(),
                    "train/ssl_loss": ssl_loss.item(),
                    "train/lr": current_lr,
                    "train/epoch": epoch + 1,
                }, step=global_step)

        # epoch 끝 — val + 체크포인트
        val_metrics = validate(predictor, val_loader, device, args.lambda_self, args.val_steps)
        wandb.log(val_metrics, step=global_step)
        print(
            f"[val ep {epoch+1}] "
            f"loss={val_metrics['val/loss']:.4f} "
            f"pred={val_metrics['val/pred_loss']:.4f} "
            f"ssl={val_metrics['val/ssl_loss']:.4f}"
        )
        predictor.train()

        # Phase 2 wrapper 가 plain state_dict 를 기대하므로 (groot_wrapper.py
        # `load_predictor_state` 참조) 매 epoch 저장도 같은 format 으로.
        # 비교 실험용 — 5ep 결과는 `epoch_05.pt`, 10ep 은 `epoch_10.pt`.
        ckpt_path = save_dir / f"epoch_{epoch+1:02d}.pt"
        torch.save(predictor.state_dict(), ckpt_path)
        print(f"[save] {ckpt_path}")

    # 최종 — predictor.save_init() 으로 inner state 초기값 보존 후 final.pt
    val_metrics = validate(predictor, val_loader, device, args.lambda_self, args.val_steps)
    wandb.log(val_metrics, step=global_step)
    print(f"[final val] loss={val_metrics['val/loss']:.4f}")

    predictor.save_init()
    final_path = save_dir / "phase1_final.pt"
    torch.save(predictor.state_dict(), final_path)
    print(f"[done] phase1 ({global_step} steps) → {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
