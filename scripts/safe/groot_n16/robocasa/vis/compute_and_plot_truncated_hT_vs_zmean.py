#!/usr/bin/env python3
"""Truncated rollout comparison: z_mean (linear) vs h_at (LSTM) at fixed t_d.

For each rollout we cap usage at t_eff = min(t_d, T).
  - z_mean: mean of z_1..z_{t_eff}   (1024-D, linear temporal aggregation)
  - h_at  : LSTM hidden at t_eff-1   (256-D, nonlinear+temporal)
For each (t_d, feature), report:
  - silhouette (Euclidean, Mahalanobis)
  - centroid distance
  - ROC-AUC on val_unseen (logistic regression trained on train+val_seen)
  - LDA 2D / t-SNE / a-vs-b visualization
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
    / "rollout_level_truncated"
)
LABEL_NAMES = {0: "failure", 1: "success"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--t-d-list", type=int, nargs="+", default=[17, 45])
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
            "episode_idx": int(r["episode_idx"]), "success": int(r["success"]),
            "T": int(X.shape[0]), "z": X, "h": h_seq,
        })
    return records


def features_at(records, t_d):
    z_mean_list, h_at_list, t_eff_list, truncated_list = [], [], [], []
    for r in records:
        t_eff = min(t_d, r["T"])
        z_mean_list.append(r["z"][:t_eff].mean(axis=0))
        h_at_list.append(r["h"][t_eff - 1])
        t_eff_list.append(t_eff)
        truncated_list.append(int(t_eff < t_d))
    return {
        "z_mean": np.stack(z_mean_list, axis=0).astype(np.float32),
        "h_at": np.stack(h_at_list, axis=0).astype(np.float32),
        "t_eff": np.asarray(t_eff_list, dtype=np.int64),
        "truncated": np.asarray(truncated_list, dtype=np.int64),
    }


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


def centroid_distance(X, labels, metric="euclidean"):
    unique = np.sort(np.unique(labels))
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    return float(np.linalg.norm(cents[0] - cents[1]))


def silhouette_score_metric(X, labels, metric, random_state):
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(silhouette_score(X, labels, metric=metric, sample_size=None, random_state=random_state))
    except ValueError:
        return None


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


def evaluate_feature(X, labels, splits, random_state, cov_reg):
    cov = pooled_within_cov(X, labels, reg=cov_reg)
    Xw = whiten(X, cov)
    eu_sil = silhouette_score_metric(X, labels, "euclidean", random_state)
    mh_sil = silhouette_score_metric(Xw, labels, "euclidean", random_state)
    eu_d = centroid_distance(X, labels, "euclidean")
    mh_d = centroid_distance(Xw, labels, "euclidean")

    train_mask = np.isin(splits, ["train", "val_seen"])
    test_mask = splits == "val_unseen"
    auc = None
    if (train_mask.sum() >= 4 and test_mask.sum() >= 2
            and len(np.unique(labels[train_mask])) == 2
            and len(np.unique(labels[test_mask])) == 2):
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])
        clf = LogisticRegression(max_iter=5000, C=1.0, random_state=random_state).fit(Xtr, labels[train_mask])
        proba = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(labels[test_mask], proba))

    return {
        "feature_dim": int(X.shape[1]),
        "n_rollouts": int(X.shape[0]),
        "euclidean": {"silhouette": eu_sil, "centroid_distance": eu_d},
        "mahalanobis": {"silhouette": mh_sil, "centroid_distance": mh_d},
        "roc_auc_val_unseen": auc,
    }


def label_colormap():
    return {0: (0.85, 0.32, 0.27, 1.0), 1: (0.27, 0.55, 0.85, 1.0)}


def scatter_with_centroids(ax, P2, labels, colors, title):
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(P2[m, 0], P2[m, 1], s=18, c=[colors[int(c)]], alpha=0.55,
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
        ax.scatter(own[m], other[m], s=18, c=[colors[int(c)]], alpha=0.55,
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


def plot_t_d(features_at_t, labels, t_d, out_path, random_state, cov_reg):
    """4×2 layout: rows = (LDA, t-SNE Eu, t-SNE Maha, a-vs-b);
    cols = (z_mean, h_at)."""
    z_mean = features_at_t["z_mean"]
    h_at = features_at_t["h_at"]

    def prep(X):
        cov = pooled_within_cov(X, labels, reg=cov_reg)
        return X, whiten(X, cov)

    z, zw = prep(z_mean)
    h, hw = prep(h_at)

    Z_lda_z = lda_2d(z, labels, random_state=random_state)
    Z_lda_h = lda_2d(h, labels, random_state=random_state)

    print(f"  [t_d={t_d}] t-SNE z_mean (Eu) ...")
    Z_eu_z = TSNE(n_components=2, perplexity=30, metric="euclidean",
                  random_state=random_state, init="pca", learning_rate="auto").fit_transform(z)
    print(f"  [t_d={t_d}] t-SNE z_mean (Maha) ...")
    Z_mh_z = TSNE(n_components=2, perplexity=30, metric="euclidean",
                  random_state=random_state, init="pca", learning_rate="auto").fit_transform(zw)
    print(f"  [t_d={t_d}] t-SNE h_at (Eu) ...")
    Z_eu_h = TSNE(n_components=2, perplexity=30, metric="euclidean",
                  random_state=random_state, init="pca", learning_rate="auto").fit_transform(h)
    print(f"  [t_d={t_d}] t-SNE h_at (Maha) ...")
    Z_mh_h = TSNE(n_components=2, perplexity=30, metric="euclidean",
                  random_state=random_state, init="pca", learning_rate="auto").fit_transform(hw)

    own_z, oth_z = per_point_ab(z, labels)
    own_zw, oth_zw = per_point_ab(zw, labels)
    own_h, oth_h = per_point_ab(h, labels)
    own_hw, oth_hw = per_point_ab(hw, labels)
    colors = label_colormap()

    fig, axes = plt.subplots(4, 2, figsize=(15, 26))
    # Col 0: z_mean
    scatter_with_centroids(axes[0, 0], Z_lda_z, labels, colors, f"t_d={t_d} | z_mean | LDA")
    scatter_with_centroids(axes[1, 0], Z_eu_z, labels, colors, f"t_d={t_d} | z_mean | t-SNE Eu")
    scatter_with_centroids(axes[2, 0], Z_mh_z, labels, colors, f"t_d={t_d} | z_mean | t-SNE Maha")
    _plot_ab(axes[3, 0], own_zw, oth_zw, labels, colors,
             "a: maha to own", "b: to other", f"t_d={t_d} | z_mean | a vs b (Maha)")
    # Col 1: h_at
    scatter_with_centroids(axes[0, 1], Z_lda_h, labels, colors, f"t_d={t_d} | h_at | LDA")
    scatter_with_centroids(axes[1, 1], Z_eu_h, labels, colors, f"t_d={t_d} | h_at | t-SNE Eu")
    scatter_with_centroids(axes[2, 1], Z_mh_h, labels, colors, f"t_d={t_d} | h_at | t-SNE Maha")
    _plot_ab(axes[3, 1], own_hw, oth_hw, labels, colors,
             "a: maha to own", "b: to other", f"t_d={t_d} | h_at | a vs b (Maha)")

    fig.suptitle(
        f"Rollout-level, truncated to t_d={t_d}  |  z_mean (linear) vs h_at (LSTM)  |  by GT success/failure",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.5, w_pad=2.0)
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def write_tsv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            for k in fields:
                r.setdefault(k, "")
            writer.writerow(r)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading detector + scoring on {args.device}...")
    records = gather_per_rollout(args.split_root, args.ckpt, args.device, args.hidden_dim)
    labels = np.asarray([r["success"] for r in records], dtype=np.int64)
    splits = np.asarray([r["split"] for r in records])
    T_arr = np.asarray([r["T"] for r in records])
    print(f"  n_rollouts={len(records)}, T mean(succ)={T_arr[labels==1].mean():.1f}, "
          f"T mean(fail)={T_arr[labels==0].mean():.1f}")

    all_summary_rows = []
    for t_d in args.t_d_list:
        feats = features_at(records, t_d)
        t_eff_succ = feats["t_eff"][labels == 1]
        t_eff_fail = feats["t_eff"][labels == 0]
        n_truncated = int(feats["truncated"].sum())
        print(f"\n=== t_d={t_d} ===")
        print(f"  t_eff(succ): mean={t_eff_succ.mean():.2f} median={np.median(t_eff_succ):.0f} "
              f"min={t_eff_succ.min()} max={t_eff_succ.max()}")
        print(f"  t_eff(fail): mean={t_eff_fail.mean():.2f} median={np.median(t_eff_fail):.0f} "
              f"min={t_eff_fail.min()} max={t_eff_fail.max()}")
        print(f"  truncated rollouts: {n_truncated} (these had T < t_d so used T)")

        for fname in ("z_mean", "h_at"):
            m = evaluate_feature(feats[fname], labels, splits, args.random_state, args.cov_reg)
            print(f"  [{fname}] eu_sil={m['euclidean']['silhouette']:.4f}  "
                  f"maha_sil={m['mahalanobis']['silhouette']:.4f}  "
                  f"AUC={m['roc_auc_val_unseen']:.4f}")
            all_summary_rows.append({
                "t_d": t_d, "feature": fname,
                "eu_silhouette": m["euclidean"]["silhouette"],
                "eu_centroid_dist": m["euclidean"]["centroid_distance"],
                "maha_silhouette": m["mahalanobis"]["silhouette"],
                "maha_centroid_dist": m["mahalanobis"]["centroid_distance"],
                "roc_auc_val_unseen": m["roc_auc_val_unseen"],
                "feature_dim": m["feature_dim"],
                "t_eff_succ_mean": float(t_eff_succ.mean()),
                "t_eff_fail_mean": float(t_eff_fail.mean()),
                "n_truncated_to_self_T": n_truncated,
            })

        plot_t_d(feats, labels, t_d,
                 out_path=args.out_dir / f"truncated_t{t_d}.png",
                 random_state=args.random_state, cov_reg=args.cov_reg)

    write_tsv(args.out_dir / "truncated_summary.tsv", all_summary_rows,
              ["t_d", "feature", "eu_silhouette", "eu_centroid_dist",
               "maha_silhouette", "maha_centroid_dist", "roc_auc_val_unseen",
               "feature_dim", "t_eff_succ_mean", "t_eff_fail_mean", "n_truncated_to_self_T"])

    with (args.out_dir / "truncated_summary.json").open("w") as f:
        json.dump({"rows": all_summary_rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(args.out_dir)


if __name__ == "__main__":
    main()
