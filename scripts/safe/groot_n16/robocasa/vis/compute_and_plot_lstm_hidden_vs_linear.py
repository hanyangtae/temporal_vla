#!/usr/bin/env python3
"""LSTM hidden state vs linear-aggregation baseline for success/fail separation.

Per rollout, compute:
  - h_T   : LSTM hidden state at last timestep (256-D, nonlinear+temporal)
  - z_mean: mean over time          (1024-D, linear+temporal)
  - z_last: z_T                     (1024-D, frame-level baseline)
  - z_disp: z_T - z_0               (1024-D, displacement)
  - z_vel : mean ||Δz||             (1-D scalar, dynamics magnitude)
  - z_path: sum ||Δz||              (1-D scalar, total traveled)
  - z_concat: concat(z_0, z_mid, z_T) (3072-D)

For each feature: silhouette by GT success/fail, centroid distance,
and ROC-AUC via logistic regression (train+val_seen → val_unseen).

Visualization: h_T and z_mean side by side (LDA / t-SNE / a-vs-b) by GT label.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler


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
    / "rollout_level_hT_vs_linear"
)
LABEL_NAMES = {0: "failure", 1: "success"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
def lstm_hidden_sequence(model, X, device):
    x = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, T, D)
    out, _ = model.lstm(x)  # (1, T, hidden)
    return out.squeeze(0).cpu().numpy()  # (T, hidden)


def compute_rollout_features(X, h_seq):
    T, D = X.shape
    diffs = np.diff(X, axis=0)
    vel = np.linalg.norm(diffs, axis=1) if len(diffs) > 0 else np.zeros(1, dtype=np.float32)
    mid = T // 2
    return {
        "h_T":     h_seq[-1].astype(np.float32),
        "z_mean":  X.mean(axis=0).astype(np.float32),
        "z_last":  X[-1].astype(np.float32),
        "z_disp":  (X[-1] - X[0]).astype(np.float32),
        "z_vel":   np.asarray([vel.mean()], dtype=np.float32),
        "z_path":  np.asarray([vel.sum()],  dtype=np.float32),
        "z_concat": np.concatenate([X[0], X[mid], X[-1]]).astype(np.float32),
    }


def gather_rollouts(split_root, ckpt, device, hidden_dim):
    model = load_detector(ckpt, hidden_dim, device)
    rows = load_manifest(split_root)
    feat_lists: dict[str, list[np.ndarray]] = {}
    meta = {"task": [], "split": [], "episode_idx": [], "success": [], "T": []}
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        X = pooled_hidden_states(rec)
        if X.shape[0] == 0:
            continue
        h_seq = lstm_hidden_sequence(model, X, device)
        feats = compute_rollout_features(X, h_seq)
        for k, v in feats.items():
            feat_lists.setdefault(k, []).append(v)
        meta["task"].append(r["task"])
        meta["split"].append(r["split"])
        meta["episode_idx"].append(int(r["episode_idx"]))
        meta["success"].append(int(r["success"]))
        meta["T"].append(X.shape[0])
    feats_mat = {k: np.stack(v, axis=0) for k, v in feat_lists.items()}
    meta_arrays = {
        "task": np.asarray(meta["task"]),
        "split": np.asarray(meta["split"]),
        "episode_idx": np.asarray(meta["episode_idx"], dtype=np.int64),
        "success": np.asarray(meta["success"], dtype=np.int64),
        "T": np.asarray(meta["T"], dtype=np.int64),
    }
    return feats_mat, meta_arrays


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


def l2_normalize(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def centroid_distance(X, labels, metric="euclidean"):
    unique = np.sort(np.unique(labels))
    if metric == "cosine":
        cents = []
        for c in unique:
            m = X[labels == c].mean(axis=0)
            cents.append(m / max(np.linalg.norm(m), 1e-12))
        cents = np.stack(cents, axis=0)
        sim = np.clip(cents @ cents.T, -1.0, 1.0)
        return float(1.0 - sim[0, 1])
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    return float(np.linalg.norm(cents[0] - cents[1]))


def silhouette_score_metric(X, labels, metric, random_state):
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(silhouette_score(X, labels, metric=metric, sample_size=None, random_state=random_state))
    except ValueError:
        return None


def ensure_2d(X):
    return X if X.ndim == 2 else X.reshape(-1, 1)


def rollout_level_metrics(X, labels, splits, random_state, cov_reg):
    X = ensure_2d(X.astype(np.float32))
    D = X.shape[1]
    if D <= 1:
        # 1-D feature: silhouette/cov etc still computable, but mahalanobis trivial
        Xw = X / max(X.std(), 1e-9)
        Xn = X  # cosine on 1D is degenerate; report as None
        eu = {
            "silhouette": silhouette_score_metric(X, labels, "euclidean", random_state),
            "centroid_distance": centroid_distance(X, labels, "euclidean"),
        }
        mh = {
            "silhouette": silhouette_score_metric(Xw, labels, "euclidean", random_state),
            "centroid_distance": centroid_distance(Xw, labels, "euclidean"),
        }
        co = {"silhouette": None, "centroid_distance": None}
    else:
        cov = pooled_within_cov(X, labels, reg=cov_reg)
        Xw = whiten(X, cov)
        Xn = l2_normalize(X)
        eu = {
            "silhouette": silhouette_score_metric(X, labels, "euclidean", random_state),
            "centroid_distance": centroid_distance(X, labels, "euclidean"),
        }
        mh = {
            "silhouette": silhouette_score_metric(Xw, labels, "euclidean", random_state),
            "centroid_distance": centroid_distance(Xw, labels, "euclidean"),
        }
        co = {
            "silhouette": silhouette_score_metric(Xn, labels, "cosine", random_state),
            "centroid_distance": centroid_distance(Xn, labels, "cosine"),
        }

    # ROC-AUC via logistic regression: train on (train + val_seen), test on val_unseen
    train_mask = np.isin(splits, ["train", "val_seen"])
    test_mask = splits == "val_unseen"
    if train_mask.sum() >= 4 and test_mask.sum() >= 2 and len(np.unique(labels[train_mask])) == 2 and len(np.unique(labels[test_mask])) == 2:
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])
        clf = LogisticRegression(max_iter=5000, C=1.0, random_state=random_state).fit(Xtr, labels[train_mask])
        proba = clf.predict_proba(Xte)[:, 1]
        roc_auc_unseen = float(roc_auc_score(labels[test_mask], proba))
    else:
        roc_auc_unseen = None

    return {
        "feature_dim": int(D),
        "n_rollouts": int(X.shape[0]),
        "euclidean": eu,
        "mahalanobis": mh,
        "cosine": co,
        "roc_auc_val_unseen": roc_auc_unseen,
    }


def label_colormap():
    return {0: (0.85, 0.32, 0.27, 1.0), 1: (0.27, 0.55, 0.85, 1.0)}


def scatter_with_centroids(ax, P2, labels, colors, title):
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(P2[m, 0], P2[m, 1], s=16, c=[colors[int(c)]], alpha=0.5,
                   label=LABEL_NAMES[int(c)], linewidths=0)
    for c in sorted(np.unique(labels)):
        m = labels == c
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        sx, sy = P2[m, 0].std(), P2[m, 1].std()
        ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, edgecolor=colors[int(c)],
                             facecolor="none", lw=1.2, alpha=0.85))
        ax.scatter([cx], [cy], s=180, marker="X", c=[colors[int(c)]],
                   edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


def _plot_ab(ax, own, other, labels, colors, xlabel, ylabel, title):
    for c in np.sort(np.unique(labels)):
        m = labels == c
        ax.scatter(own[m], other[m], s=16, c=[colors[int(c)]], alpha=0.55,
                   label=LABEL_NAMES[int(c)], linewidths=0)
    lo = float(min(own.min(), other.min()))
    hi = float(max(own.max(), other.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax.set_title(f"{title}\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)


def lda_2d(X, labels, random_state):
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    Z = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X.astype(np.float64), labels)
    if Z.shape[1] == 1:
        rng = np.random.default_rng(random_state)
        Z = np.column_stack([Z[:, 0], rng.normal(scale=0.02 * (Z[:, 0].std() + 1e-9), size=Z.shape[0])])
    return Z


def per_point_ab_eu(X, labels):
    unique = np.sort(np.unique(labels))
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    own_idx = np.searchsorted(unique, labels)
    diff = X[:, None, :] - cents[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    own = dist[np.arange(len(labels)), own_idx]
    others = dist.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    return own, others.min(axis=1)


def visualize_pair(h_T, z_mean, labels, out_path, random_state, cov_reg):
    """Compare h_T (LSTM hidden) vs z_mean (linear time average) side by side."""

    def prepare(X):
        cov = pooled_within_cov(X, labels, reg=cov_reg)
        Xw = whiten(X, cov)
        return X, Xw

    h, hw = prepare(h_T)
    z, zw = prepare(z_mean)

    Z_lda_h = lda_2d(h, labels, random_state=random_state)
    Z_lda_z = lda_2d(z, labels, random_state=random_state)

    # 600 rollouts is small; t-SNE all of them
    print("  t-SNE h_T (euclidean) ...")
    Z_tsne_h = TSNE(n_components=2, perplexity=30, metric="euclidean",
                    random_state=random_state, init="pca", learning_rate="auto").fit_transform(h)
    print("  t-SNE z_mean (euclidean) ...")
    Z_tsne_z = TSNE(n_components=2, perplexity=30, metric="euclidean",
                    random_state=random_state, init="pca", learning_rate="auto").fit_transform(z)
    print("  t-SNE h_T (mahalanobis) ...")
    Z_tsne_hw = TSNE(n_components=2, perplexity=30, metric="euclidean",
                     random_state=random_state, init="pca", learning_rate="auto").fit_transform(hw)
    print("  t-SNE z_mean (mahalanobis) ...")
    Z_tsne_zw = TSNE(n_components=2, perplexity=30, metric="euclidean",
                     random_state=random_state, init="pca", learning_rate="auto").fit_transform(zw)

    own_h, oth_h = per_point_ab_eu(h, labels)
    own_z, oth_z = per_point_ab_eu(z, labels)
    own_hw, oth_hw = per_point_ab_eu(hw, labels)
    own_zw, oth_zw = per_point_ab_eu(zw, labels)
    colors = label_colormap()

    fig, axes = plt.subplots(3, 4, figsize=(26, 18))
    # row 0: LDA
    scatter_with_centroids(axes[0, 0], Z_lda_h, labels, colors, "h_T | LDA (1D + jitter)")
    scatter_with_centroids(axes[0, 1], Z_lda_z, labels, colors, "z_mean | LDA (1D + jitter)")
    scatter_with_centroids(axes[0, 2], Z_lda_h, labels, colors, "h_T | LDA (duplicate)")
    scatter_with_centroids(axes[0, 3], Z_lda_z, labels, colors, "z_mean | LDA (duplicate)")
    # row 1: t-SNE Euclidean
    scatter_with_centroids(axes[1, 0], Z_tsne_h, labels, colors, "h_T | t-SNE Euclidean")
    scatter_with_centroids(axes[1, 1], Z_tsne_z, labels, colors, "z_mean | t-SNE Euclidean")
    scatter_with_centroids(axes[1, 2], Z_tsne_hw, labels, colors, "h_T | t-SNE Mahalanobis")
    scatter_with_centroids(axes[1, 3], Z_tsne_zw, labels, colors, "z_mean | t-SNE Mahalanobis")
    # row 2: a vs b
    _plot_ab(axes[2, 0], own_h, oth_h, labels, colors,
             "a: euclid to own", "b: to other", "h_T | a vs b (Eu)")
    _plot_ab(axes[2, 1], own_z, oth_z, labels, colors,
             "a: euclid to own", "b: to other", "z_mean | a vs b (Eu)")
    _plot_ab(axes[2, 2], own_hw, oth_hw, labels, colors,
             "a: maha to own", "b: to other", "h_T | a vs b (Maha)")
    _plot_ab(axes[2, 3], own_zw, oth_zw, labels, colors,
             "a: maha to own", "b: to other", "z_mean | a vs b (Maha)")

    fig.suptitle("Rollout-level: LSTM h_T vs linear z_mean | colored by GT success/failure", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def write_metrics_tsv(path, feature_to_metrics):
    fields = ["feature", "dim", "eu_silhouette", "eu_centroid_dist",
              "maha_silhouette", "maha_centroid_dist",
              "cos_silhouette", "cos_centroid_dist",
              "roc_auc_val_unseen", "n_rollouts"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for name, m in feature_to_metrics.items():
            writer.writerow({
                "feature": name,
                "dim": m["feature_dim"],
                "eu_silhouette": m["euclidean"]["silhouette"],
                "eu_centroid_dist": m["euclidean"]["centroid_distance"],
                "maha_silhouette": m["mahalanobis"]["silhouette"],
                "maha_centroid_dist": m["mahalanobis"]["centroid_distance"],
                "cos_silhouette": m["cosine"]["silhouette"],
                "cos_centroid_dist": m["cosine"]["centroid_distance"],
                "roc_auc_val_unseen": m["roc_auc_val_unseen"],
                "n_rollouts": m["n_rollouts"],
            })


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading detector + scoring rollouts on {args.device}...")
    feats, meta = gather_rollouts(args.split_root, args.ckpt, args.device, args.hidden_dim)
    labels = meta["success"]
    splits = meta["split"]
    print(f"  n_rollouts={len(labels)}, success={int(labels.sum())}, failure={int((1-labels).sum())}")
    for k, v in feats.items():
        print(f"  feature '{k}' shape={v.shape}")

    feature_to_metrics: dict[str, dict] = {}
    for name in ("h_T", "z_mean", "z_last", "z_disp", "z_vel", "z_path", "z_concat"):
        m = rollout_level_metrics(feats[name], labels, splits,
                                  random_state=args.random_state, cov_reg=args.cov_reg)
        feature_to_metrics[name] = m
        eu_s = m["euclidean"]["silhouette"]
        mh_s = m["mahalanobis"]["silhouette"]
        roc = m["roc_auc_val_unseen"]
        print(f"  [{name}] eu_sil={eu_s if eu_s is None else f'{eu_s:.4f}'}"
              f" maha_sil={mh_s if mh_s is None else f'{mh_s:.4f}'}"
              f" ROC_AUC_val_unseen={roc if roc is None else f'{roc:.4f}'}")

    with (args.out_dir / "rollout_hT_vs_linear.json").open("w") as f:
        json.dump({"features": feature_to_metrics}, f, indent=2, sort_keys=True)
        f.write("\n")
    write_metrics_tsv(args.out_dir / "rollout_hT_vs_linear.tsv", feature_to_metrics)

    visualize_pair(feats["h_T"], feats["z_mean"], labels,
                   out_path=args.out_dir / "hT_vs_zmean_separation.png",
                   random_state=args.random_state, cov_reg=args.cov_reg)
    print(args.out_dir)


if __name__ == "__main__":
    main()
