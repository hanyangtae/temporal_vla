#!/usr/bin/env python3
"""Produce four separate visualizations:
  - z_mean (full trajectory)
  - z_mean (truncated to t_d=17)
  - h_at   (full trajectory, = h_T)
  - h_at   (truncated to t_d=17)

Each: 2x2 panel (LDA / t-SNE Eu / t-SNE Maha / a-vs-b Maha).
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Ellipse
from scipy.linalg import cholesky, solve_triangular
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "safe_split_seen4_unseen2_openDrawer_pnpCab_100ep"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_train_logs"
    / "groot_n16-robocasa_seen4_unseen2_openDrawer_pnpCab_100ep-lstm-seed2_mean_mean"
    / "20260520/110644/model_final.ckpt"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep"
    / "rollout_level_four_separate"
)
LABEL_NAMES = {0: "failure", 1: "success"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--t-trunc", type=int, default=17)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--cov-reg", type=float, default=1e-3)
    return p.parse_args()


def tensor_to_numpy(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def load_manifest(split_root):
    with (split_root / "manifest.tsv").open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows.sort(key=lambda r: (int(r["task_id"]), int(r["episode_idx"])))
    return rows


def pkl_path(split_root, row):
    return (split_root / row["split"] / row["task"]
            / f"task{int(row['task_id'])}--ep{int(row['episode_idx'])}--succ{int(row['success'])}.pkl")


def pooled_hidden_states(record):
    feats = []
    for h in record["hidden_states"]:
        a = tensor_to_numpy(h).astype(np.float32, copy=False)
        feats.append(a.mean(axis=(0, 1)))
    return np.stack(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)


class LSTMDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)


def load_detector(ckpt_path: Path, hidden_dim: int, device: str) -> LSTMDetector:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    model = LSTMDetector(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def lstm_hidden_seq(model, X, device):
    x = torch.from_numpy(X).float().unsqueeze(0).to(device)
    out, _ = model.lstm(x)
    return out.squeeze(0).cpu().numpy()


def gather_per_rollout(split_root, ckpt, device, hidden_dim):
    model = load_detector(ckpt, hidden_dim, device)
    rows = load_manifest(split_root)
    records = []
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        X = pooled_hidden_states(rec)
        if X.shape[0] == 0:
            continue
        h_seq = lstm_hidden_seq(model, X, device)
        records.append({
            "task": r["task"], "split": r["split"],
            "success": int(r["success"]), "T": int(X.shape[0]),
            "z": X, "h": h_seq,
        })
    return records


def feature_vectors(records, mode, t_trunc):
    """mode in {'z_mean_full', 'z_mean_trunc', 'h_at_full', 'h_at_trunc'}"""
    out = []
    for r in records:
        if mode.endswith("_full"):
            t_eff = r["T"]
        else:
            t_eff = min(t_trunc, r["T"])
        if mode.startswith("z_mean"):
            out.append(r["z"][:t_eff].mean(axis=0))
        else:
            out.append(r["h"][t_eff - 1])
    return np.stack(out, axis=0).astype(np.float32)


def pooled_within_cov(X, labels, reg):
    X64 = X.astype(np.float64, copy=False)
    D = X64.shape[1]
    scatter = np.zeros((D, D), dtype=np.float64)
    dof = 0
    for c in np.unique(labels):
        Xc = X64[labels == c]
        if len(Xc) < 2:
            continue
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        scatter += Xc.T @ Xc
        dof += len(Xc) - 1
    cov = scatter / max(dof, 1)
    cov += reg * np.trace(cov) / D * np.eye(D)
    return cov


def whiten(X, cov):
    L = cholesky(cov, lower=True)
    return solve_triangular(L, X.astype(np.float64).T, lower=True).T.astype(np.float32)


def per_point_ab(X, labels):
    unique = np.sort(np.unique(labels))
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    own_idx = np.searchsorted(unique, labels)
    diff = X[:, None, :] - cents[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    own = dist[np.arange(len(labels)), own_idx]
    others = dist.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    return own, others.min(axis=1)


def lda_2d(X, labels, random_state):
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    Z = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X.astype(np.float64), labels)
    if Z.shape[1] == 1:
        rng = np.random.default_rng(random_state)
        Z = np.column_stack([Z[:, 0], rng.normal(scale=0.02 * (Z[:, 0].std() + 1e-9), size=Z.shape[0])])
    return Z


def label_colormap():
    return {0: (0.85, 0.32, 0.27, 1.0), 1: (0.27, 0.55, 0.85, 1.0)}


def scatter_with_centroids(ax, P2, labels, colors, title):
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(P2[m, 0], P2[m, 1], s=22, c=[colors[int(c)]], alpha=0.55,
                   label=LABEL_NAMES[int(c)], linewidths=0)
    for c in sorted(np.unique(labels)):
        m = labels == c
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        sx, sy = P2[m, 0].std(), P2[m, 1].std()
        ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, edgecolor=colors[int(c)],
                             facecolor="none", lw=1.4, alpha=0.85))
        ax.scatter([cx], [cy], s=220, marker="X", c=[colors[int(c)]],
                   edgecolors="black", linewidths=1.4, zorder=5)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9, markerscale=2, framealpha=0.85)


def _plot_ab(ax, own, other, labels, colors, xlabel, ylabel, title):
    for c in np.sort(np.unique(labels)):
        m = labels == c
        ax.scatter(own[m], other[m], s=22, c=[colors[int(c)]], alpha=0.55,
                   label=LABEL_NAMES[int(c)], linewidths=0)
    lo = float(min(own.min(), other.min()))
    hi = float(max(own.max(), other.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax.set_title(f"{title}\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


def silhouette_score_or_none(X, labels, metric, random_state):
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(silhouette_score(X, labels, metric=metric, sample_size=None, random_state=random_state))
    except ValueError:
        return None


def plot_one(X, labels, *, title, out_path, random_state, cov_reg):
    cov = pooled_within_cov(X, labels, reg=cov_reg)
    Xw = whiten(X, cov)
    Z_lda = lda_2d(X, labels, random_state=random_state)
    print(f"  t-SNE Eu...")
    Z_eu = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(X)
    print(f"  t-SNE Maha...")
    Z_mh = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(Xw)
    own, other = per_point_ab(Xw, labels)
    colors = label_colormap()

    sil_eu = silhouette_score_or_none(X, labels, "euclidean", random_state)
    sil_mh = silhouette_score_or_none(Xw, labels, "euclidean", random_state)

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    scatter_with_centroids(axes[0, 0], Z_lda, labels, colors,
                           f"LDA 2D (1D + jitter)\nEu silhouette = {sil_eu:.4f}")
    scatter_with_centroids(axes[0, 1], Z_eu, labels, colors,
                           f"t-SNE Euclidean")
    scatter_with_centroids(axes[1, 0], Z_mh, labels, colors,
                           f"t-SNE Mahalanobis\nMaha silhouette = {sil_mh:.4f}")
    _plot_ab(axes[1, 1], own, other, labels, colors,
             "a: maha dist to own", "b: maha dist to other",
             "per-point a vs b (Mahalanobis)")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5, w_pad=2.5)
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading detector + scoring on {args.device}...")
    records = gather_per_rollout(args.split_root, args.ckpt, args.device, args.hidden_dim)
    labels = np.asarray([r["success"] for r in records], dtype=np.int64)
    print(f"  n_rollouts={len(records)}")

    configs = [
        ("z_mean_full",  "z_mean | full trajectory",                       f"z_mean_full"),
        ("z_mean_trunc", f"z_mean | truncated to t_d={args.t_trunc}",      f"z_mean_t{args.t_trunc}"),
        ("h_at_full",    "h_at | full trajectory  (= h_T)",                f"h_at_full"),
        ("h_at_trunc",   f"h_at | truncated to t_d={args.t_trunc}",        f"h_at_t{args.t_trunc}"),
    ]
    for mode, title, fname in configs:
        print(f"\n=== {mode} ===")
        X = feature_vectors(records, mode, args.t_trunc)
        print(f"  feature shape={X.shape}")
        plot_one(X, labels, title=title,
                 out_path=args.out_dir / f"{fname}.png",
                 random_state=args.random_state, cov_reg=args.cov_reg)
    print(args.out_dir)


if __name__ == "__main__":
    main()
