"""
Phase 1 ProgressPredictor 성능 평가 스크립트.

세 가지 split에서 각 100개 에피소드를 평가:
  - train:   학습에 사용된 에피소드
  - val:     validation에 사용된 에피소드
  - unseen:  어디에도 사용되지 않은 에피소드 (index >= train+val)

각 에피소드를 처음부터 끝까지 순차 처리 (inference 모드):
  predictor.reset() → for z_t: predictor.forward(z_t) → progress 예측

Metrics (에피소드별 계산 후 평균):
  - MSE:          MSE(predicted, t/T)
  - MAE:          MAE(predicted, t/T)
  - Pearson r:    predicted와 t/T의 상관계수
  - Mono rate:    단조증가 비율 = mean(v_{t+1} > v_t)
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.ttt.predictor import ProgressPredictor


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def mono_rate(pred: np.ndarray) -> float:
    """연속 프레임에서 progress가 증가하는 비율."""
    if len(pred) < 2:
        return 1.0
    return float(np.mean(np.diff(pred) >= 0))


def eval_episode(
    predictor: ProgressPredictor,
    embeddings: dict,
    ep_from: int,
    ep_to: int,
    device: torch.device,
) -> dict:
    """단일 에피소드 평가."""
    ep_len = ep_to - ep_from
    if ep_len < 2:
        return None

    predictor.reset()
    preds = []

    for abs_idx in range(ep_from, ep_to):
        if abs_idx not in embeddings:
            break
        z_t = embeddings[abs_idx].unsqueeze(0).to(device)  # [1, 1024]
        result = predictor.forward(z_t, update=True)
        preds.append(result["progress"].item())

    if len(preds) < 2:
        return None

    preds = np.array(preds)
    targets = np.linspace(0, 1, len(preds))

    return {
        "mse":       float(np.mean((preds - targets) ** 2)),
        "mae":       float(np.mean(np.abs(preds - targets))),
        "pearson_r": pearson_r(preds, targets),
        "mono_rate": mono_rate(preds),
        "preds":     preds,
        "targets":   targets,
        "ep_len":    len(preds),
    }


def save_episode_json(ep_idx: int, split: str, r: dict, save_path: str):
    """에피소드별 예측 결과를 JSON으로 저장."""
    records = [
        {"timestep": t, "progress_prediction": float(p), "gt_progress": float(g)}
        for t, (p, g) in enumerate(zip(r["preds"], r["targets"]))
    ]
    payload = {
        "episode_idx": ep_idx,
        "split": split,
        "ep_len": r["ep_len"],
        "metrics": {
            "mse":       r["mse"],
            "mae":       r["mae"],
            "pearson_r": r["pearson_r"],
            "mono_rate": r["mono_rate"],
        },
        "records": records,
    }
    with open(save_path, "w") as f:
        json.dump(payload, f)


def _render_progress_frame(
    preds: np.ndarray,
    targets: np.ndarray,
    current_t: int,
    title: str,
    width: int,
    height: int,
) -> np.ndarray:
    """타임스텝 t에서의 progress plot 프레임을 numpy 배열로 반환."""
    fig, ax = plt.subplots(figsize=(width / 120.0, height / 120.0), dpi=120)
    timesteps = np.arange(len(preds))
    ax.plot(timesteps, targets, "k--", linewidth=1.4, label="GT (t/T)", alpha=0.8)
    ax.plot(timesteps, preds,   color="#1F77B4", linewidth=1.4, label="Predicted")
    ax.axvline(current_t, color="#D62728", linewidth=1.5, alpha=0.9)
    ax.scatter([current_t], [float(preds[current_t])], color="#D62728", s=30, zorder=3)
    ax.text(
        0.02, 0.95,
        f"t={current_t}  pred={preds[current_t]:.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Progress")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def make_episode_video(
    ep_idx: int,
    r: dict,
    split_name: str,
    ds,
    ep_from: int,
    save_path: str,
    fps: int = 10,
    image_key: str = "observation.images.primary",
):
    """에피소드 원본 영상(왼쪽) + progress curve(오른쪽) 결합 MP4 생성."""
    import imageio

    T = len(r["preds"])
    preds   = r["preds"]
    targets = r["targets"]

    # 프레임 로드 (LeRobotDataset에서 직접)
    frames = []
    for abs_idx in range(ep_from, ep_from + T):
        item = ds[abs_idx]
        img = item[image_key]                          # [3, H, W] float32 [0,1]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        frames.append(img)

    if not frames:
        return

    video_h, video_w = frames[0].shape[:2]
    graph_w = max(video_w // 2, 480)
    title = (
        f"[{split_name}] ep {ep_idx}"
        f"  r={r['pearson_r']:.3f}  MSE={r['mse']:.4f}  mono={r['mono_rate']:.3f}"
    )

    writer = imageio.get_writer(save_path, fps=fps)
    try:
        for t, video_frame in enumerate(frames):
            graph = _render_progress_frame(preds, targets, t, title, graph_w, video_h)
            combined = np.concatenate([video_frame, graph], axis=1)
            writer.append_data(combined)
    finally:
        writer.close()

    print(f"Saved video: {save_path}")


def plot_single_episode(ep_idx: int, r: dict, split_name: str, save_path: str):
    """단일 에피소드 progress curve를 PNG로 저장."""
    timesteps = np.arange(len(r["targets"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timesteps, r["targets"], "k--", linewidth=1.5, label="GT (t/T)")
    ax.plot(timesteps, r["preds"],   "b-",  linewidth=1.5, label="Predicted")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Progress")
    ax.set_title(
        f"[{split_name}] Episode {ep_idx}"
        f"  |  r={r['pearson_r']:.3f}, MSE={r['mse']:.4f}, mono={r['mono_rate']:.3f}"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def eval_split(
    predictor: ProgressPredictor,
    embeddings: dict,
    episode_indices: list,
    meta,
    device: torch.device,
    n_samples: int = 100,
    seed: int = 42,
    split_name: str = "",
    n_examples: int = 5,
) -> dict:
    """n_samples개 에피소드를 랜덤 샘플해서 평가."""
    rng = random.Random(seed)
    sampled = rng.sample(episode_indices, min(n_samples, len(episode_indices)))

    results = []
    all_results = []     # 전체 결과 (JSON/개별 PNG 저장용)
    example_episodes = []  # 시각화용

    for ep_idx in tqdm(sampled, desc=f"Eval [{split_name}]"):
        ep = meta.episodes[ep_idx]
        ep_from = ep["dataset_from_index"]
        ep_to   = ep["dataset_to_index"]

        r = eval_episode(predictor, embeddings, ep_from, ep_to, device)
        if r is None:
            continue
        results.append(r)
        all_results.append((ep_idx, r))
        if len(example_episodes) < n_examples:
            example_episodes.append((ep_idx, r))

    if not results:
        return {}

    agg = {
        "mse":       np.mean([r["mse"]       for r in results]),
        "mae":       np.mean([r["mae"]        for r in results]),
        "pearson_r": np.mean([r["pearson_r"]  for r in results]),
        "mono_rate": np.mean([r["mono_rate"]  for r in results]),
        "n_episodes": len(results),
        "mean_ep_len": np.mean([r["ep_len"]   for r in results]),
    }
    agg["examples"] = example_episodes
    agg["all_results"] = all_results
    return agg


def plot_examples(examples: list, split_name: str, save_path: str):
    """예시 에피소드 progress curve 시각화."""
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (ep_idx, r) in zip(axes, examples):
        t = np.arange(len(r["targets"])) / max(len(r["targets"]) - 1, 1)
        ax.plot(t, r["targets"], "k--", label="GT (t/T)", linewidth=1.5)
        ax.plot(t, r["preds"],   "b-",  label="Predicted", linewidth=1.5)
        ax.set_title(f"ep {ep_idx}\nr={r['pearson_r']:.2f}, mono={r['mono_rate']:.2f}")
        ax.set_xlabel("Normalized time")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle(f"Progress curves — {split_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          type=str, default="checkpoints/phase1/phase1_final.pt")
    parser.add_argument("--data_root",     type=str, default="data/bridge_v2_lerobot")
    parser.add_argument("--repo_id",       type=str, default="FedorX8/bridge_v2_lerobot")
    parser.add_argument("--embed_cache",   type=str, default="data/bridge_v2_lerobot_clip_embeddings.pt")
    parser.add_argument("--train_episodes",type=int, default=2986)
    parser.add_argument("--val_episodes",  type=int, default=287)
    parser.add_argument("--split_seed",    type=int, default=42)
    parser.add_argument("--n_samples",     type=int, default=100)
    parser.add_argument("--device",        type=str, default="cuda")
    parser.add_argument("--save_dir",      type=str, default="eval_results/phase1")
    # Model architecture (학습 시와 동일해야 함)
    parser.add_argument("--input_dim",     type=int, default=1024)
    parser.add_argument("--proj_dim",      type=int, default=64)
    parser.add_argument("--inner_model_type", type=str, default="mlp")
    parser.add_argument("--head_hidden_dim",  type=int, default=128)
    parser.add_argument("--eta_base",      type=float, default=0.1)
    parser.add_argument("--n_examples",    type=int,   default=5,
                        help="grid PNG에 포함할 예시 에피소드 수")
    parser.add_argument("--save_json",     action="store_true",
                        help="에피소드별 예측 결과를 JSON으로 저장 (plot_phase1_progress.py 입력용)")
    parser.add_argument("--save_individual", action="store_true",
                        help="에피소드별 개별 PNG 저장")
    parser.add_argument("--save_video",    action="store_true",
                        help="에피소드별 원본영상+progress curve 결합 MP4 저장")
    parser.add_argument("--n_videos",      type=int, default=10,
                        help="저장할 최대 영상 수 (split당). eval 자체는 n_samples만큼 수행.")
    parser.add_argument("--fps",           type=int, default=10,
                        help="영상 FPS (기본: 10)")
    parser.add_argument("--image_key",     type=str, default="observation.images.primary",
                        help="LeRobotDataset에서 사용할 이미지 키")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 모델 로드 ──
    predictor = ProgressPredictor(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        inner_model_type=args.inner_model_type,
        eta_base=args.eta_base,
        learnable_eta=False,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    predictor.load_state_dict(state)
    predictor.save_init()   # θ_0 등록 (reset() 사용 위해)
    predictor.eval()
    print(f"Loaded: {args.ckpt}")

    # ── 임베딩 캐시 로드 ──
    print(f"Loading embedding cache...")
    embeddings = torch.load(args.embed_cache, map_location="cpu", weights_only=True)
    cached_indices = set(embeddings.keys())
    print(f"Cached frames: {len(cached_indices)}")

    # ── LeRobot meta 로드 (에피소드 경계용) ──
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.data_root)
    meta = ds.meta
    total_ep = meta.total_episodes

    # ── 학습 시와 동일한 train/val split 재현 ──
    total_used = args.train_episodes + args.val_episodes
    all_indices = list(range(total_used))
    random.seed(args.split_seed)
    random.shuffle(all_indices)
    train_idx = all_indices[:args.train_episodes]
    val_idx   = all_indices[args.train_episodes:]

    # unseen: 캐시에 있는 에피소드 중 train/val 에 없는 것
    used_set = set(all_indices)
    max_cached_ep = max(
        (ep_idx for ep_idx in range(total_ep)
         if any(i in cached_indices for i in [meta.episodes[ep_idx]["dataset_from_index"]])),
        default=total_used,
    )
    unseen_idx = [i for i in range(total_used, min(max_cached_ep + 1, total_ep))
                  if i not in used_set]
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, unseen pool: {len(unseen_idx)}")

    # ── 평가 ──
    splits = {
        "train":  train_idx,
        "val":    val_idx,
        "unseen": unseen_idx,
    }

    all_metrics = {}
    for split_name, indices in splits.items():
        if not indices:
            print(f"[{split_name}] 에피소드 없음, 스킵")
            continue
        metrics = eval_split(
            predictor, embeddings, indices, meta, device,
            n_samples=args.n_samples,
            seed=args.split_seed,
            split_name=split_name,
            n_examples=args.n_examples,
        )
        all_metrics[split_name] = metrics

        # 예시 grid 시각화
        if metrics.get("examples"):
            plot_examples(
                metrics["examples"],
                split_name=split_name,
                save_path=os.path.join(args.save_dir, f"progress_curves_{split_name}.png"),
            )

        # 에피소드별 JSON / 개별 PNG 저장
        if (args.save_json or args.save_individual) and metrics.get("all_results"):
            ep_dir = os.path.join(args.save_dir, "episodes", split_name)
            os.makedirs(ep_dir, exist_ok=True)
            for ep_idx, r in metrics["all_results"]:
                if args.save_json:
                    json_path = os.path.join(ep_dir, f"ep_{ep_idx:06d}.progress.json")
                    save_episode_json(ep_idx, split_name, r, json_path)
                if args.save_individual:
                    png_path = os.path.join(ep_dir, f"ep_{ep_idx:06d}.png")
                    plot_single_episode(ep_idx, r, split_name, png_path)
            print(f"[{split_name}] {len(metrics['all_results'])}개 에피소드 저장 → {ep_dir}")

        # 에피소드별 MP4 저장 (최대 n_videos개)
        if args.save_video and metrics.get("all_results"):
            ep_dir = os.path.join(args.save_dir, "episodes", split_name)
            os.makedirs(ep_dir, exist_ok=True)
            video_targets = metrics["all_results"][:args.n_videos]
            for ep_idx, r in tqdm(video_targets, desc=f"Video [{split_name}]"):
                ep_from = meta.episodes[ep_idx]["dataset_from_index"]
                video_path = os.path.join(ep_dir, f"ep_{ep_idx:06d}.mp4")
                make_episode_video(
                    ep_idx, r, split_name, ds, ep_from, video_path,
                    fps=args.fps, image_key=args.image_key,
                )

    # ── 결과 출력 ──
    print("\n" + "=" * 55)
    print(f"{'Split':<10} {'MSE':>8} {'MAE':>8} {'Pearson r':>10} {'Mono rate':>10} {'N':>6}")
    print("-" * 55)
    for split_name, m in all_metrics.items():
        if not m:
            continue
        print(
            f"{split_name:<10} "
            f"{m['mse']:>8.4f} "
            f"{m['mae']:>8.4f} "
            f"{m['pearson_r']:>10.4f} "
            f"{m['mono_rate']:>10.4f} "
            f"{m['n_episodes']:>6}"
        )
    print("=" * 55)

    # 결과 저장
    result_path = os.path.join(args.save_dir, "metrics.txt")
    with open(result_path, "w") as f:
        f.write(f"{'Split':<10} {'MSE':>8} {'MAE':>8} {'Pearson r':>10} {'Mono rate':>10} {'N':>6}\n")
        for split_name, m in all_metrics.items():
            if not m:
                continue
            f.write(
                f"{split_name:<10} "
                f"{m['mse']:>8.4f} "
                f"{m['mae']:>8.4f} "
                f"{m['pearson_r']:>10.4f} "
                f"{m['mono_rate']:>10.4f} "
                f"{m['n_episodes']:>6}\n"
            )
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
