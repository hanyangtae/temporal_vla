#!/usr/bin/env python3
"""Euclidean counterpart of mahalanobis/cosine separation plots."""

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


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
    / "cluster_analysis_mean_mean"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--scopes", nargs="+", default=["all", "val_unseen"])
    p.add_argument("--tsne-sample", type=int, default=5000)
    p.add_argument("--random-state", type=int, default=0)
    return p.parse_args()


def tensor_to_numpy(v: Any) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def load_manifest(split_root: Path, scope: str) -> list[dict[str, str]]:
    with (split_root / "manifest.tsv").open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if scope != "all":
        rows = [r for r in rows if r["split"] == scope]
    rows.sort(key=lambda r: (int(r["task_id"]), int(r["episode_idx"])))
    return rows


def pkl_path(split_root: Path, row: dict[str, str]) -> Path:
    return (
        split_root
        / row["split"]
        / row["task"]
        / f"task{int(row['task_id'])}--ep{int(row['episode_idx'])}--succ{int(row['success'])}.pkl"
    )


def pooled_hidden_states(record: dict[str, Any]) -> np.ndarray:
    feats = []
    for h in record["hidden_states"]:
        a = tensor_to_numpy(h).astype(np.float32, copy=False)
        feats.append(a.mean(axis=(0, 1)))
    return np.stack(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)


def load_scope(split_root: Path, scope: str) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    rows = load_manifest(split_root, scope)
    Xs, tasks = [], []
    task_id_to_name: dict[int, str] = {}
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        f_arr = pooled_hidden_states(rec)
        n = f_arr.shape[0]
        if n == 0:
            continue
        tid = int(r["task_id"])
        task_id_to_name[tid] = r["task"]
        Xs.append(f_arr)
        tasks.append(np.full(n, tid, dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(tasks), task_id_to_name


def subsample(*arrays, n: int, rng: np.random.Generator):
    total = arrays[0].shape[0]
    if total <= n:
        return arrays
    idx = rng.choice(total, size=n, replace=False)
    return tuple(a[idx] for a in arrays)


def task_colormap(unique_tids: np.ndarray) -> dict[int, tuple]:
    cmap = plt.get_cmap("tab10")
    return {int(t): cmap(i % 10) for i, t in enumerate(unique_tids)}


def scatter_with_centroids(ax, P2, labels, names, colors, title) -> None:
    for tid in sorted(np.unique(labels)):
        m = labels == tid
        ax.scatter(P2[m, 0], P2[m, 1], s=4, c=[colors[int(tid)]], alpha=0.35, label=names[int(tid)], linewidths=0)
    for tid in sorted(np.unique(labels)):
        m = labels == tid
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        sx, sy = P2[m, 0].std(), P2[m, 1].std()
        ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, edgecolor=colors[int(tid)], facecolor="none", lw=1.2, alpha=0.8))
        ax.scatter([cx], [cy], s=140, marker="X", c=[colors[int(tid)]], edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)


def lda_2d(X, labels, random_state) -> np.ndarray:
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    Z = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X.astype(np.float64), labels)
    if Z.shape[1] == 1:
        rng = np.random.default_rng(random_state)
        Z = np.column_stack([Z[:, 0], rng.normal(scale=0.02 * (Z[:, 0].std() + 1e-9), size=Z.shape[0])])
    return Z


def per_point_a_b(X, labels):
    unique = np.sort(np.unique(labels))
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    own_idx = np.searchsorted(unique, labels)
    diff = X[:, None, :] - cents[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    own = dist[np.arange(len(labels)), own_idx]
    others = dist.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    near = others.min(axis=1)
    return own, near


def plot_scope(X, tasks, task_id_to_name, *, out_path, tsne_sample, random_state, scope_name) -> None:
    Z_lda = lda_2d(X, tasks, random_state=random_state)
    rng = np.random.default_rng(random_state)
    X_s, tasks_s = subsample(X, tasks, n=tsne_sample, rng=rng)
    print(f"  [{scope_name}] running euclidean t-SNE on {X_s.shape}...")
    Z_tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="euclidean",
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_s)

    own, other = per_point_a_b(X, tasks)
    unique_tids = np.sort(np.unique(tasks))
    colors = task_colormap(unique_tids)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    scatter_with_centroids(axes[0], Z_lda, tasks, task_id_to_name, colors, f"{scope_name} | LDA 2D on raw (task)")
    scatter_with_centroids(axes[1], Z_tsne, tasks_s, task_id_to_name, colors, f"{scope_name} | t-SNE (metric=euclidean)")
    ax3 = axes[2]
    for tid in unique_tids:
        m = tasks == tid
        ax3.scatter(own[m], other[m], s=4, c=[colors[int(tid)]], alpha=0.35, label=task_id_to_name[int(tid)], linewidths=0)
    lo = float(min(own.min(), other.min()))
    hi = float(max(own.max(), other.max()))
    ax3.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax3.set_xlabel("a: euclidean dist to own task centroid")
    ax3.set_ylabel("b: euclidean dist to nearest other task centroid")
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax3.set_title(f"{scope_name} | per-point a vs b (euclidean)\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)

    fig.suptitle(f"GR00T-N1.6 RoboCasa | euclidean task separation | scope={scope_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scope in args.scopes:
        print(f"[{scope}] loading...")
        X, tasks, task_id_to_name = load_scope(args.split_root, scope)
        print(f"  X={X.shape}")
        plot_scope(
            X, tasks, task_id_to_name,
            out_path=args.out_dir / f"euclidean_separation_{scope}.png",
            tsne_sample=args.tsne_sample,
            random_state=args.random_state,
            scope_name=scope,
        )


if __name__ == "__main__":
    main()
