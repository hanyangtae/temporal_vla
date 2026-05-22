#!/usr/bin/env python3
"""z_mean rollout-level visualization with labeling:
  - success rollouts → split by task (6 sub-clusters)
  - failure rollouts → single 'fail' cluster
Two figures: full trajectory and truncated to t_d=17.
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
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16"
    / "safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep"
    / "rollout_zmean_succByTask_vs_fail"
)


# label order: 6 task-successes then 'fail'
TASK_TO_SUCC_LABEL = {
    "CoffeeSetupMug": 0,
    "OpenSingleDoor": 1,
    "PnPCounterToCab": 2,
    "PnPSinkToCounter": 3,
    "PnPCounterToStove": 4,
    "OpenDrawer": 5,
}
FAIL_LABEL = 6
LABEL_NAMES = {
    0: "Coffee · succ",
    1: "OpenSingleDoor · succ",
    2: "PnPCounterToCab · succ",
    3: "PnPSinkToCounter · succ",
    4: "PnPCounterToStove · succ",
    5: "OpenDrawer · succ",
    6: "Failure (all tasks)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--t-trunc", type=int, default=17)
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


def gather(split_root):
    rows = load_manifest(split_root)
    records = []
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        X = pooled_hidden_states(rec)
        if X.shape[0] == 0:
            continue
        succ = int(r["success"])
        if succ == 1:
            lab = TASK_TO_SUCC_LABEL[r["task"]]
        else:
            lab = FAIL_LABEL
        records.append({"z": X, "T": int(X.shape[0]), "label": lab, "task": r["task"], "success": succ})
    return records


def z_mean_at(records, t_d_or_full):
    out, labels = [], []
    for r in records:
        if t_d_or_full == "full":
            t_eff = r["T"]
        else:
            t_eff = min(int(t_d_or_full), r["T"])
        out.append(r["z"][:t_eff].mean(axis=0))
        labels.append(r["label"])
    return np.stack(out, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


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
    cmap = plt.get_cmap("tab10")
    colors = {i: cmap(i) for i in range(6)}  # 6 task-successes
    colors[6] = (0.2, 0.2, 0.2, 1.0)  # fail = dark grey
    return colors


def scatter_with_centroids(ax, P2, labels, colors, title, marker_for_fail=True):
    for c in sorted(np.unique(labels)):
        m = labels == c
        marker = "o" if c != FAIL_LABEL else "s"
        alpha = 0.65 if c != FAIL_LABEL else 0.35
        ax.scatter(P2[m, 0], P2[m, 1], s=24, c=[colors[int(c)]], alpha=alpha,
                   marker=marker, label=LABEL_NAMES[int(c)], linewidths=0)
    for c in sorted(np.unique(labels)):
        m = labels == c
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        sx, sy = P2[m, 0].std(), P2[m, 1].std()
        ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, edgecolor=colors[int(c)],
                             facecolor="none", lw=1.3, alpha=0.85))
        ax.scatter([cx], [cy], s=220, marker="X", c=[colors[int(c)]],
                   edgecolors="black", linewidths=1.4, zorder=5)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


def _plot_ab(ax, own, other, labels, colors, xlabel, ylabel, title):
    for c in np.sort(np.unique(labels)):
        m = labels == c
        marker = "o" if c != FAIL_LABEL else "s"
        alpha = 0.65 if c != FAIL_LABEL else 0.35
        ax.scatter(own[m], other[m], s=24, c=[colors[int(c)]], alpha=alpha,
                   marker=marker, label=LABEL_NAMES[int(c)], linewidths=0)
    lo = float(min(own.min(), other.min()))
    hi = float(max(own.max(), other.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax.set_title(f"{title}\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)


def silhouette_safe(X, labels, metric, random_state):
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(silhouette_score(X, labels, metric=metric, sample_size=None, random_state=random_state))
    except ValueError:
        return None


def plot_z_mean_panel(X, labels, *, title, out_path, random_state, cov_reg):
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

    sil_eu = silhouette_safe(X, labels, "euclidean", random_state)
    sil_mh = silhouette_safe(Xw, labels, "euclidean", random_state)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    scatter_with_centroids(axes[0, 0], Z_lda, labels, colors,
                           f"LDA 2D\nEu silhouette = {sil_eu:.4f}")
    scatter_with_centroids(axes[0, 1], Z_eu, labels, colors, "t-SNE Euclidean")
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

    print("loading rollouts...")
    records = gather(args.split_root)
    print(f"  n_rollouts={len(records)}")
    # report label distribution
    all_labels = np.asarray([r["label"] for r in records])
    for k in sorted(np.unique(all_labels)):
        print(f"    label {k} = {LABEL_NAMES[int(k)]:30s}  n={(all_labels==k).sum()}")

    for mode_tag, mode_value, title_suffix in [
        ("full",        "full",            "z_mean | full trajectory"),
        (f"t{args.t_trunc}", args.t_trunc, f"z_mean | truncated to t_d={args.t_trunc}"),
    ]:
        print(f"\n=== {mode_tag} ===")
        X, labels = z_mean_at(records, mode_value)
        plot_z_mean_panel(X, labels,
                          title=f"{title_suffix}  |  success colored by task, failure pooled",
                          out_path=args.out_dir / f"zmean_succByTask_vs_fail_{mode_tag}.png",
                          random_state=args.random_state, cov_reg=args.cov_reg)
    print(args.out_dir)


if __name__ == "__main__":
    main()
