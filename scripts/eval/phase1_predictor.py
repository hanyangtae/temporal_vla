"""
Phase 1 ProgressPredictor 성능 평가 스크립트 — RoboCasa atomic 10 task per-task mixture.

학습 (`scripts/train/phase1_groot_robocasa.py`) 과 동일한 dataset/split 재현:
  - per-task LeRobot v2.1 dataset 의 episodes 를 split_seed 로 train/val 분리.
  - Eagle pre-LLM cache (`data/robocasa_eagle_pre_llm/<Task>/embeddings.pt`) 사용.

여러 ckpt 동시 비교 가능 (`--ckpts ckpt_a.pt ckpt_b.pt`) — 같은 episode 들에 각각
적용해서 metric 나란히 출력. inner_model_type 이 ckpt 마다 다르면 `--inner_model_types`
로 지정.

각 에피소드를 처음부터 끝까지 순차 처리 (inference 모드):
  predictor.reset() → for z_t: predictor.forward(z_t) → progress 예측

Metrics (에피소드별 계산 후 평균):
  - MSE:           MSE(predicted, t/T)
  - MAE:           MAE(predicted, t/T)
  - Pearson r:     predicted 와 t/T 의 상관계수
  - Spearman ρ:    rank 기반 상관계수
  - Mono rate:     단조증가 비율 = mean(v_{t+1} >= v_t)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.path_setup import DATA_ROOT
from src.datasets.robocasa_v21_reader import RoboCasaV21Reader
from src.ttt.predictor import ProgressPredictor


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def spearman_rho(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return 0.0
    return float(spearmanr(pred, target).statistic)


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
        z_t = embeddings[abs_idx].unsqueeze(0).to(device)  # [1, D]
        result = predictor.forward(z_t, update=True)
        preds.append(result["progress"].item())

    if len(preds) < 2:
        return None

    preds = np.array(preds)
    targets = np.linspace(0, 1, len(preds))

    return {
        "mse":         float(np.mean((preds - targets) ** 2)),
        "mae":         float(np.mean(np.abs(preds - targets))),
        "pearson_r":   pearson_r(preds, targets),
        "spearman_rho": spearman_rho(preds, targets),
        "mono_rate":   mono_rate(preds),
        "preds":       preds,
        "targets":     targets,
        "ep_len":      len(preds),
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
            "mse":         r["mse"],
            "mae":         r["mae"],
            "pearson_r":   r["pearson_r"],
            "spearman_rho": r["spearman_rho"],
            "mono_rate":   r["mono_rate"],
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
        "mse":         np.mean([r["mse"]          for r in results]),
        "mae":         np.mean([r["mae"]           for r in results]),
        "pearson_r":   np.mean([r["pearson_r"]     for r in results]),
        "spearman_rho": np.mean([r["spearman_rho"] for r in results]),
        "mono_rate":   np.mean([r["mono_rate"]     for r in results]),
        "n_episodes":  len(results),
        "mean_ep_len": np.mean([r["ep_len"]        for r in results]),
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
# RoboCasa per-task dataset loading
# ──────────────────────────────────────────────

def _find_lerobot_root(data_root: Path, task: str) -> Path:
    matches = sorted((data_root / task).glob("*/lerobot"))
    if not matches:
        raise FileNotFoundError(f"No lerobot/ under {data_root / task}")
    return matches[0]


def load_robocasa_split(
    tasks: list[str],
    data_root: Path,
    cache_root: Path,
    train_frac: float,
    split_seed: int,
) -> tuple[dict, list[tuple], list[tuple]]:
    """학습 (phase1_groot_robocasa.py) 과 동일한 split 재현.

    Returns
    -------
    embeddings_by_task : dict[task -> {abs_idx: tensor}]
    train_pool         : list[(task, ep_idx, ep_from, ep_to)]
    val_pool           : list[(task, ep_idx, ep_from, ep_to)]
    """
    embeddings_by_task: dict[str, dict[int, torch.Tensor]] = {}
    train_pool: list[tuple] = []
    val_pool: list[tuple] = []

    for task in tasks:
        cache_path = cache_root / task / "embeddings.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Eagle cache 없음: {cache_path}")
        lerobot_root = _find_lerobot_root(data_root, task)

        cache = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        cached_indices = set(int(k) for k in cache.keys())
        embeddings_by_task[task] = {int(k): v for k, v in cache.items()}

        reader = RoboCasaV21Reader(lerobot_root)
        all_eps = reader.lerobot_meta_episodes()
        eligible = [
            ep_idx for ep_idx, ep in all_eps.items()
            if ep["dataset_from_index"] in cached_indices
        ]
        rng = random.Random(split_seed)
        rng.shuffle(eligible)
        n_train = int(len(eligible) * train_frac)
        train_eps = sorted(eligible[:n_train])
        val_eps = sorted(eligible[n_train:])

        for ep_idx in train_eps:
            ep = all_eps[ep_idx]
            train_pool.append((task, ep_idx, ep["dataset_from_index"], ep["dataset_to_index"]))
        for ep_idx in val_eps:
            ep = all_eps[ep_idx]
            val_pool.append((task, ep_idx, ep["dataset_from_index"], ep["dataset_to_index"]))

        print(f"[{task}] eligible={len(eligible)}  train={len(train_eps)}  val={len(val_eps)}")

    return embeddings_by_task, train_pool, val_pool


def eval_pool(
    predictor: ProgressPredictor,
    pool: list[tuple],
    embeddings_by_task: dict,
    device: torch.device,
    n_samples: int,
    seed: int,
    split_name: str,
    n_examples: int = 5,
    max_ep_len: int = 485,
) -> dict:
    """(task, ep_idx, ep_from, ep_to) pool 에서 n_samples 개 sample 평가."""
    rng = random.Random(seed)
    sampled = rng.sample(pool, min(n_samples, len(pool)))

    results = []
    all_results = []
    example_episodes = []

    for task, ep_idx, ep_from, ep_to in tqdm(sampled, desc=f"Eval [{split_name}]"):
        ep_len = ep_to - ep_from
        # 학습과 동일하게 max_ep_len 으로 truncate
        ep_to_eff = ep_from + min(ep_len, max_ep_len)
        embeddings = embeddings_by_task[task]
        r = eval_episode(predictor, embeddings, ep_from, ep_to_eff, device)
        if r is None:
            continue
        r["task"] = task
        results.append(r)
        all_results.append(((task, ep_idx), r))
        if len(example_episodes) < n_examples:
            example_episodes.append(((task, ep_idx), r))

    if not results:
        return {}

    return {
        "mse":          float(np.mean([r["mse"]          for r in results])),
        "mae":          float(np.mean([r["mae"]           for r in results])),
        "pearson_r":    float(np.mean([r["pearson_r"]     for r in results])),
        "spearman_rho": float(np.mean([r["spearman_rho"] for r in results])),
        "mono_rate":    float(np.mean([r["mono_rate"]     for r in results])),
        "n_episodes":   len(results),
        "mean_ep_len":  float(np.mean([r["ep_len"]        for r in results])),
        "examples":     example_episodes,
        "all_results":  all_results,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

DEFAULT_TASKS = [
    "OpenDrawer", "CloseDrawer",
    "OpenCabinet", "CloseCabinet",
    "OpenFridge", "CloseFridge",
    "OpenMicrowave", "CloseMicrowave",
    "PickPlaceCounterToStove", "PickPlaceCounterToSink",
]


def main():
    parser = argparse.ArgumentParser()
    # Multi-ckpt 비교
    parser.add_argument("--ckpts", type=str, nargs="+", required=True,
                        help="비교할 ckpt 경로들. 여러 개 가능.")
    parser.add_argument("--ckpt_labels", type=str, nargs="+", default=None,
                        help="ckpt 별 라벨 (출력용). 미지정 시 파일 stem.")
    parser.add_argument("--inner_model_types", type=str, nargs="+", default=None,
                        help="ckpt 별 inner_model_type (linear/mlp/linear_preLN). "
                             "미지정 시 모두 --inner_model_type 값 사용.")
    # Dataset (학습과 동일하게)
    parser.add_argument("--data_root", type=str,
                        default=str(DATA_ROOT / "robocasa/v1.0/pretrain/atomic"))
    parser.add_argument("--cache_root", type=str,
                        default=str(DATA_ROOT / "robocasa_eagle_pre_llm"))
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_ep_len", type=int, default=485,
                        help="episode 평가 truncate 길이 (학습과 동일).")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="split 당 평가할 random sample 수.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str,
                        default=str(REPO_ROOT / "outputs/eval/phase1_groot_robocasa"))
    # Model architecture (학습 시와 동일해야 함, ckpt 마다 다르면 --inner_model_types)
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--proj_dim", type=int, default=2048)
    parser.add_argument("--inner_model_type", type=str, default="linear",
                        choices=["linear", "mlp", "linear_preLN"])
    parser.add_argument("--head_hidden_dim", type=int, default=128)
    parser.add_argument("--eta_base", type=float, default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="학습 시 batch size. F.mse_loss(reduction='mean') 가 1/B 곱하기 때문에 "
                             "eval batch=1 path 가 학습보다 B 배 큰 inner-update step 을 받음. "
                             "이걸 보정해서 학습 trajectory 와 매칭하기 위해 "
                             "eta_eval = eta_base / train_batch_size 로 scale. 0 이면 보정 비활성.")
    # 출력
    parser.add_argument("--n_examples", type=int, default=5,
                        help="grid PNG 에 포함할 예시 에피소드 수")
    parser.add_argument("--save_json", action="store_true",
                        help="에피소드별 예측 결과를 JSON 으로 저장")
    parser.add_argument("--save_individual", action="store_true",
                        help="에피소드별 개별 PNG 저장")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ckpt 메타
    ckpts = args.ckpts
    labels = args.ckpt_labels or [Path(c).stem for c in ckpts]
    inner_types = args.inner_model_types or [args.inner_model_type] * len(ckpts)
    if not (len(labels) == len(inner_types) == len(ckpts)):
        raise ValueError(
            f"--ckpts({len(ckpts)}), --ckpt_labels({len(labels)}), "
            f"--inner_model_types({len(inner_types)}) 길이 불일치"
        )

    print(f"[ckpts] {len(ckpts)} 개 비교:")
    for c, l, t in zip(ckpts, labels, inner_types):
        print(f"  - {l:30s}  inner={t:14s}  path={c}")

    # ── Dataset 로딩 (모든 ckpt 가 같은 split 평가) ──
    print(f"\n[data] loading {len(args.tasks)} tasks...")
    embeddings_by_task, train_pool, val_pool = load_robocasa_split(
        tasks=args.tasks,
        data_root=Path(args.data_root),
        cache_root=Path(args.cache_root),
        train_frac=args.train_frac,
        split_seed=args.split_seed,
    )
    print(f"[data] total: train={len(train_pool)}  val={len(val_pool)} episodes\n")

    splits = {"train": train_pool, "val": val_pool}

    # ckpt → split → metrics
    all_metrics: dict[str, dict[str, dict]] = {}

    for ckpt_path, label, inner_type in zip(ckpts, labels, inner_types):
        print(f"\n=== Evaluating ckpt: {label} ({inner_type}) ===")
        predictor = ProgressPredictor(
            input_dim=args.input_dim,
            proj_dim=args.proj_dim,
            inner_model_type=inner_type,
            eta_base=args.eta_base,
            learnable_eta=False,
            head_hidden_dim=args.head_hidden_dim,
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        predictor.load_state_dict(state)
        # 학습/eval 의 batch-size mismatch 보정. ssl loss 의 1/B 분모 때문에 batch=1 eval 의
        # inner update step 이 batch=32 학습 대비 B 배 큼 → W 가 너무 빠르게 saturate 해서
        # progress 가 평평하게 나옴. eta 만 1/B 로 scale 해도 step 크기 일치.
        if args.train_batch_size and args.train_batch_size > 1:
            predictor.ttt.eta_base = predictor.ttt.eta_base / args.train_batch_size
            print(f"  eta_base scaled: {args.eta_base} → {predictor.ttt.eta_base} "
                  f"(/= train_batch_size={args.train_batch_size})")
        predictor.save_init()
        predictor.eval()
        print(f"  loaded: {ckpt_path}")

        ckpt_metrics = {}
        ckpt_save_dir = os.path.join(args.save_dir, label)
        os.makedirs(ckpt_save_dir, exist_ok=True)

        for split_name, pool in splits.items():
            if not pool:
                continue
            m = eval_pool(
                predictor=predictor,
                pool=pool,
                embeddings_by_task=embeddings_by_task,
                device=device,
                n_samples=args.n_samples,
                seed=args.split_seed,
                split_name=f"{label}/{split_name}",
                n_examples=args.n_examples,
                max_ep_len=args.max_ep_len,
            )
            ckpt_metrics[split_name] = m

            if m.get("examples"):
                # examples 의 ep_idx tag 가 (task, ep_idx) 튜플이므로 plot_examples 호환을 위해 str 화
                examples_str = [
                    (f"{task}/ep{ep}", r) for (task, ep), r in m["examples"]
                ]
                plot_examples(
                    examples_str,
                    split_name=f"{label} [{split_name}]",
                    save_path=os.path.join(ckpt_save_dir, f"progress_curves_{split_name}.png"),
                )

            if (args.save_json or args.save_individual) and m.get("all_results"):
                ep_dir = os.path.join(ckpt_save_dir, "episodes", split_name)
                os.makedirs(ep_dir, exist_ok=True)
                for (task, ep_idx), r in m["all_results"]:
                    tag = f"{task}_ep{ep_idx:06d}"
                    if args.save_json:
                        save_episode_json(tag, split_name, r, os.path.join(ep_dir, f"{tag}.progress.json"))
                    if args.save_individual:
                        plot_single_episode(tag, r, split_name, os.path.join(ep_dir, f"{tag}.png"))

        all_metrics[label] = ckpt_metrics

    # ── 결과 출력 (ckpt 별 나란히) ──
    header = (
        f"{'Ckpt':<24} {'Split':<6} {'MSE':>8} {'MAE':>8} "
        f"{'Pearson r':>10} {'Spearman':>10} {'Mono rate':>10} {'EpLen':>8} {'N':>6}"
    )
    sep_len = len(header) + 2
    print("\n" + "=" * sep_len)
    print(header)
    print("-" * sep_len)
    for label, ckpt_metrics in all_metrics.items():
        for split_name in ["train", "val"]:
            m = ckpt_metrics.get(split_name)
            if not m:
                continue
            print(
                f"{label:<24} {split_name:<6} "
                f"{m['mse']:>8.4f} {m['mae']:>8.4f} "
                f"{m['pearson_r']:>10.4f} {m['spearman_rho']:>10.4f} "
                f"{m['mono_rate']:>10.4f} {m['mean_ep_len']:>8.1f} {m['n_episodes']:>6}"
            )
    print("=" * sep_len)

    # metrics.txt
    result_path = os.path.join(args.save_dir, "metrics.txt")
    with open(result_path, "w") as f:
        f.write(header + "\n")
        for label, ckpt_metrics in all_metrics.items():
            for split_name in ["train", "val"]:
                m = ckpt_metrics.get(split_name)
                if not m:
                    continue
                f.write(
                    f"{label:<24} {split_name:<6} "
                    f"{m['mse']:>8.4f} {m['mae']:>8.4f} "
                    f"{m['pearson_r']:>10.4f} {m['spearman_rho']:>10.4f} "
                    f"{m['mono_rate']:>10.4f} {m['mean_ep_len']:>8.1f} {m['n_episodes']:>6}\n"
                )
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
