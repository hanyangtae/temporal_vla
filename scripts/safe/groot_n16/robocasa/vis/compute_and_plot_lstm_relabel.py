#!/usr/bin/env python3
"""Relabel frames using SAFE-LSTM detector and re-analyze cluster structure.

For each rollout:
  - Compute per-frame failure prob p_t via SAFE-LSTM (seed2 ckpt).
  - First t* where p_t >= threshold → "predicted failure onset".
  - frames [0, t*-1] → label 0 (pre-failure)
  - frames [t*, T-1] → label 1 (post-failure)
  - If no crossing → all frames label 0.

Then compute centroid distance + silhouette (Eu/Maha/Cos) + visualize
the new 2-cluster structure.
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
    / "cluster_lstm_relabel"
)


# Threshold picked from val_seen best balanced-acc (model_perf_vs_det.csv).
DEFAULT_THRESHOLD = 0.9978993169999711
LABEL_NAMES = {0: "pre-failure", 1: "post-failure"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--scopes", nargs="+",
                   default=["all", "train", "val_seen", "val_unseen"])
    p.add_argument("--plot-scopes", nargs="+", default=["all", "val_unseen"])
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--tsne-sample", type=int, default=5000)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--cov-reg", type=float, default=1e-3)
    return p.parse_args()


def tensor_to_numpy(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def load_manifest(split_root, scope):
    with (split_root / "manifest.tsv").open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if scope != "all":
        rows = [r for r in rows if r["split"] == scope]
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

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)  # (B, T)


def load_detector(ckpt_path: Path, hidden_dim: int, device: str) -> LSTMDetector:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    model = LSTMDetector(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def score_rollout(model, X, device):
    x = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, T, D)
    return model(x).squeeze(0).cpu().numpy()  # (T,)


def load_scope_with_scores(split_root, scope, model, device):
    rows = load_manifest(split_root, scope)
    Xs, scores, gt_succ, rollout_idx = [], [], [], []
    score_per_rollout = []
    for ridx, r in enumerate(rows):
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        X = pooled_hidden_states(rec)
        n = X.shape[0]
        if n == 0:
            continue
        p = score_rollout(model, X, device)
        Xs.append(X)
        scores.append(p)
        gt_succ.append(np.full(n, int(r["success"]), dtype=np.int64))
        rollout_idx.append(np.full(n, ridx, dtype=np.int64))
        score_per_rollout.append({
            "rollout_idx": ridx, "task": r["task"], "split": r["split"],
            "episode_idx": int(r["episode_idx"]), "success": int(r["success"]),
            "T": int(n), "scores": p, "max_score": float(p.max()),
        })
    return {
        "features": np.concatenate(Xs, axis=0),
        "scores": np.concatenate(scores, axis=0),
        "ground_truth_success": np.concatenate(gt_succ),
        "rollout_idx": np.concatenate(rollout_idx),
        "per_rollout": score_per_rollout,
    }


def derive_relabel(per_rollout, scores, rollout_idx, threshold):
    """For each rollout, find first t where score >= threshold.

    Returns a same-length-as-scores label array (0 = pre, 1 = post)."""
    labels = np.zeros(scores.shape[0], dtype=np.int64)
    onset_records = []
    for rec in per_rollout:
        ridx = rec["rollout_idx"]
        mask = rollout_idx == ridx
        s = rec["scores"]
        cross = np.where(s >= threshold)[0]
        if len(cross) == 0:
            onset = -1  # never crossed
        else:
            onset = int(cross[0])
            # set frames at and after onset to label 1
            sub_labels = np.zeros(len(s), dtype=np.int64)
            sub_labels[onset:] = 1
            labels[mask] = sub_labels
        onset_records.append({
            **{k: v for k, v in rec.items() if k != "scores"},
            "onset_frame": int(onset),
            "fraction_post": float((labels[mask] == 1).sum() / len(s)) if len(s) > 0 else 0.0,
        })
    return labels, onset_records


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


def centroids(X, labels):
    unique = np.sort(np.unique(labels))
    return unique, np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)


def pairwise(C):
    diff = C[:, None, :] - C[None, :, :]
    return np.linalg.norm(diff.astype(np.float64), axis=-1)


def cosine_centroids(Xn, labels):
    unique = np.sort(np.unique(labels))
    cents = []
    for c in unique:
        m = Xn[labels == c].mean(axis=0)
        cents.append(m / max(np.linalg.norm(m), 1e-12))
    return unique, np.stack(cents, axis=0)


def cosine_pairwise(C):
    sim = np.clip(C @ C.T, -1.0, 1.0)
    return 1.0 - sim


def silhouette_info(X, labels, sample_size, random_state, metric="euclidean"):
    unique, counts = np.unique(labels, return_counts=True)
    info = {"n_points": int(X.shape[0]), "n_clusters": int(len(unique)),
            "cluster_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)}}
    if len(unique) < 2 or counts.min() < 2:
        info["score"] = None
        return info
    ss = sample_size if X.shape[0] > sample_size else None
    info["sample_size"] = int(ss or X.shape[0])
    info["score"] = float(silhouette_score(X, labels, metric=metric, sample_size=ss, random_state=random_state))
    return info


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


def per_point_ab_cos(Xn, labels):
    unique, cents = cosine_centroids(Xn, labels)
    own_idx = np.searchsorted(unique, labels)
    sim = np.clip(Xn @ cents.T, -1.0, 1.0)
    dist = 1.0 - sim
    own = dist[np.arange(len(labels)), own_idx]
    others = dist.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    return own, others.min(axis=1)


def analyse(X, labels, sample_size, random_state, cov_reg):
    Xn = l2_normalize(X)
    cov = pooled_within_cov(X, labels, reg=cov_reg)
    Xw = whiten(X, cov)
    _, eu_c = centroids(X, labels)
    _, mh_c = centroids(Xw, labels)
    _, co_c = cosine_centroids(Xn, labels)
    return {
        "feature_shape": list(X.shape),
        "label_names": ["pre-failure", "post-failure"],
        "euclidean": {
            "centroid_pairwise_distance": pairwise(eu_c).tolist(),
            "silhouette": silhouette_info(X, labels, sample_size, random_state, "euclidean"),
        },
        "mahalanobis": {
            "cov_regularization": cov_reg,
            "centroid_pairwise_distance": pairwise(mh_c).tolist(),
            "silhouette": silhouette_info(Xw, labels, sample_size, random_state, "euclidean"),
        },
        "cosine": {
            "centroid_pairwise_distance": cosine_pairwise(co_c).tolist(),
            "silhouette": silhouette_info(Xn, labels, sample_size, random_state, "cosine"),
        },
    }


def label_colormap():
    return {0: (0.27, 0.55, 0.85, 1.0), 1: (0.85, 0.32, 0.27, 1.0)}  # pre blue, post red


def scatter_with_centroids(ax, P2, labels, colors, title):
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(P2[m, 0], P2[m, 1], s=4, c=[colors[int(c)]], alpha=0.35,
                   label=LABEL_NAMES[int(c)], linewidths=0)
    for c in sorted(np.unique(labels)):
        m = labels == c
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        sx, sy = P2[m, 0].std(), P2[m, 1].std()
        ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, edgecolor=colors[int(c)],
                             facecolor="none", lw=1.2, alpha=0.85))
        ax.scatter([cx], [cy], s=160, marker="X", c=[colors[int(c)]],
                   edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


def _plot_ab(ax, own, other, labels, colors, xlabel, ylabel, title):
    for c in np.sort(np.unique(labels)):
        m = labels == c
        ax.scatter(own[m], other[m], s=4, c=[colors[int(c)]], alpha=0.35,
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


def plot_scope(X, labels, *, out_path, tsne_sample, random_state, cov_reg, scope_name, threshold):
    cov = pooled_within_cov(X, labels, reg=cov_reg)
    Xw = whiten(X, cov)
    Xn = l2_normalize(X)
    Z_lda_raw = lda_2d(X, labels, random_state=random_state)
    Z_lda_cos = lda_2d(Xn, labels, random_state=random_state)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=min(tsne_sample, X.shape[0]), replace=False)
    lab_s = labels[idx]
    X_s, Xw_s, Xn_s = X[idx], Xw[idx], Xn[idx]

    print(f"  [{scope_name}] t-SNE euclidean...")
    Z_eu = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(X_s)
    print(f"  [{scope_name}] t-SNE mahalanobis...")
    Z_mh = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(Xw_s)
    print(f"  [{scope_name}] t-SNE cosine...")
    Z_co = TSNE(n_components=2, perplexity=30, metric="cosine",
                random_state=random_state, init="random", learning_rate="auto").fit_transform(Xn_s)

    own_eu, other_eu = per_point_ab_eu(X, labels)
    own_mh, other_mh = per_point_ab_eu(Xw, labels)
    own_co, other_co = per_point_ab_cos(Xn, labels)
    colors = label_colormap()

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    scatter_with_centroids(axes[0, 0], Z_lda_raw, labels, colors, f"{scope_name} | Euclidean | LDA on raw")
    scatter_with_centroids(axes[0, 1], Z_eu, lab_s, colors, f"{scope_name} | Euclidean | t-SNE")
    _plot_ab(axes[0, 2], own_eu, other_eu, labels, colors,
             "a: euclid dist to own", "b: euclid dist to other", f"{scope_name} | Euclidean | a vs b")
    scatter_with_centroids(axes[1, 0], Z_lda_raw, labels, colors, f"{scope_name} | Mahalanobis | LDA (≡ Eu)")
    scatter_with_centroids(axes[1, 1], Z_mh, lab_s, colors, f"{scope_name} | Mahalanobis | t-SNE on whitened")
    _plot_ab(axes[1, 2], own_mh, other_mh, labels, colors,
             "a: maha dist to own", "b: maha dist to other", f"{scope_name} | Mahalanobis | a vs b")
    scatter_with_centroids(axes[2, 0], Z_lda_cos, labels, colors, f"{scope_name} | Cosine | LDA on L2-norm")
    scatter_with_centroids(axes[2, 1], Z_co, lab_s, colors, f"{scope_name} | Cosine | t-SNE")
    _plot_ab(axes[2, 2], own_co, other_co, labels, colors,
             "a: cosine dist to own", "b: cosine dist to other", f"{scope_name} | Cosine | a vs b")

    fig.suptitle(
        f"GR00T-N1.6 RoboCasa | SAFE-LSTM relabel (τ={threshold:.4f}) | scope={scope_name}",
        fontsize=13,
    )
    fig.tight_layout()
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

    print(f"loading detector from {args.ckpt} on {args.device}...")
    model = load_detector(args.ckpt, args.hidden_dim, args.device)

    results = {}
    onset_table_rows = []
    plot_cache = {}

    for scope in args.scopes:
        print(f"[{scope}] loading + scoring rollouts...")
        bundle = load_scope_with_scores(args.split_root, scope, model, args.device)
        X = bundle["features"]
        scores = bundle["scores"]
        rollout_idx = bundle["rollout_idx"]
        gt_succ = bundle["ground_truth_success"]
        per_rollout = bundle["per_rollout"]

        labels, onset_records = derive_relabel(per_rollout, scores, rollout_idx, args.threshold)

        n_pre, n_post = int((labels == 0).sum()), int((labels == 1).sum())
        crossed = sum(1 for r in onset_records if r["onset_frame"] >= 0)
        crossed_succ = sum(1 for r in onset_records if r["onset_frame"] >= 0 and r["success"] == 1)
        crossed_fail = sum(1 for r in onset_records if r["onset_frame"] >= 0 and r["success"] == 0)
        total_succ = sum(1 for r in onset_records if r["success"] == 1)
        total_fail = sum(1 for r in onset_records if r["success"] == 0)
        print(f"  X={X.shape}, pre={n_pre} post={n_post}, "
              f"rollouts crossed={crossed}/{len(onset_records)} "
              f"(succ {crossed_succ}/{total_succ}, fail {crossed_fail}/{total_fail})")

        scope_summary = {
            "n_points": int(X.shape[0]),
            "n_rollouts": len(onset_records),
            "rollouts_crossed": int(crossed),
            "rollouts_crossed_among_gt_success": int(crossed_succ),
            "rollouts_crossed_among_gt_failure": int(crossed_fail),
            "n_pre": n_pre, "n_post": n_post,
        }
        if min(n_pre, n_post) >= 2:
            scope_summary.update(analyse(X, labels,
                                         sample_size=args.sample_size,
                                         random_state=args.random_state,
                                         cov_reg=args.cov_reg))
        results[scope] = scope_summary

        for rec in onset_records:
            onset_table_rows.append({"scope": scope, **rec})

        if scope in args.plot_scopes and min(n_pre, n_post) >= 2:
            plot_cache[scope] = (X, labels)

    with (args.out_dir / "cluster_lstm_relabel.json").open("w") as f:
        json.dump({"threshold": args.threshold, "results": results}, f, indent=2, sort_keys=True)
        f.write("\n")

    silhouette_rows = []
    for scope, payload in results.items():
        for metric in ("euclidean", "mahalanobis", "cosine"):
            if metric not in payload:
                continue
            sil = payload[metric]["silhouette"]
            dist = float(np.asarray(payload[metric]["centroid_pairwise_distance"])[0, 1])
            silhouette_rows.append({
                "scope": scope, "metric": metric,
                "silhouette": sil.get("score"),
                "centroid_distance": dist,
                "n_pre": payload["n_pre"], "n_post": payload["n_post"],
                "n_points": sil.get("n_points"), "sample_size": sil.get("sample_size"),
            })
    write_tsv(args.out_dir / "cluster_lstm_relabel.tsv", silhouette_rows,
              ["scope", "metric", "silhouette", "centroid_distance",
               "n_pre", "n_post", "n_points", "sample_size"])
    write_tsv(args.out_dir / "onset_per_rollout.tsv", onset_table_rows,
              ["scope", "rollout_idx", "task", "split", "episode_idx", "success",
               "T", "max_score", "onset_frame", "fraction_post"])

    for scope, (X, labels) in plot_cache.items():
        plot_scope(X, labels,
                   out_path=args.out_dir / f"lstm_relabel_{scope}.png",
                   tsne_sample=args.tsne_sample,
                   random_state=args.random_state,
                   cov_reg=args.cov_reg,
                   scope_name=scope,
                   threshold=args.threshold)
    print(args.out_dir)


if __name__ == "__main__":
    main()
