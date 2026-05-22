#!/usr/bin/env python3
"""Visualize task separation under Euclidean vs Mahalanobis geometries.

Panels per scope:
  (1) LDA 2D projection (linear, exposes between/within ratio)
  (2) t-SNE on whitened features (Mahalanobis-equivalent local geometry)
  (3) Per-point scatter of a (own-centroid mahalanobis) vs b (nearest-other),
      with y=x line. Points above the line contribute positively to silhouette.
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
    p.add_argument("--cov-reg", type=float, default=1e-3)
    return p.parse_args()


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


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


def load_scope(split_root: Path, scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    rows = load_manifest(split_root, scope)
    Xs, tasks, succs = [], [], []
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
        succs.append(np.full(n, int(r["success"]), dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    return X, np.concatenate(tasks), np.concatenate(succs), task_id_to_name


def pooled_within_cov(X: np.ndarray, labels: np.ndarray, reg: float) -> np.ndarray:
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


def whiten(X: np.ndarray, cov: np.ndarray) -> np.ndarray:
    L = cholesky(cov, lower=True)
    return solve_triangular(L, X.astype(np.float64).T, lower=True).T.astype(np.float32)


def subsample(*arrays, n: int, rng: np.random.Generator):
    total = arrays[0].shape[0]
    if total <= n:
        return arrays
    idx = rng.choice(total, size=n, replace=False)
    return tuple(a[idx] for a in arrays)


def task_colormap(unique_tids: np.ndarray) -> dict[int, tuple]:
    cmap = plt.get_cmap("tab10")
    return {int(t): cmap(i % 10) for i, t in enumerate(unique_tids)}


def scatter_with_centroids(ax, P2: np.ndarray, labels: np.ndarray, names: dict[int, str], colors: dict[int, tuple], title: str) -> None:
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


def per_point_a_b(Xw: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.sort(np.unique(labels))
    cents = np.stack([Xw[labels == c].mean(axis=0) for c in unique], axis=0)
    own_idx = np.searchsorted(unique, labels)
    diff = Xw[:, None, :] - cents[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    own = dists[np.arange(len(labels)), own_idx]
    others = dists.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    nearest_other = others.min(axis=1)
    return own, nearest_other


def lda_2d(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    if n_comp <= 0:
        raise ValueError("LDA requires ≥2 classes")
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    Z = lda.fit_transform(X.astype(np.float64), labels)
    if Z.shape[1] == 1:
        rng = np.random.default_rng(0)
        Z = np.column_stack([Z[:, 0], rng.normal(scale=0.02 * (Z[:, 0].std() + 1e-9), size=Z.shape[0])])
    return Z


def plot_scope(
    X: np.ndarray,
    tasks: np.ndarray,
    task_id_to_name: dict[int, str],
    *,
    out_path: Path,
    tsne_sample: int,
    random_state: int,
    cov_reg: float,
    scope_name: str,
) -> None:
    cov = pooled_within_cov(X, tasks, reg=cov_reg)
    Xw = whiten(X, cov)

    Z_lda = lda_2d(X, tasks)
    rng = np.random.default_rng(random_state)
    Xw_s, tasks_s = subsample(Xw, tasks, n=tsne_sample, rng=rng)
    print(f"  [{scope_name}] running t-SNE on whitened {Xw_s.shape}...")
    Z_tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(Xw_s)

    own, other = per_point_a_b(Xw, tasks)

    unique_tids = np.sort(np.unique(tasks))
    colors = task_colormap(unique_tids)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    scatter_with_centroids(axes[0], Z_lda, tasks, task_id_to_name, colors, f"{scope_name} | LDA 2D (task)")
    scatter_with_centroids(axes[1], Z_tsne, tasks_s, task_id_to_name, colors, f"{scope_name} | t-SNE on whitened (Mahalanobis)")

    ax3 = axes[2]
    for tid in unique_tids:
        m = tasks == tid
        ax3.scatter(own[m], other[m], s=4, c=[colors[int(tid)]], alpha=0.35, label=task_id_to_name[int(tid)], linewidths=0)
    lo = min(own.min(), other.min())
    hi = max(own.max(), other.max())
    ax3.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax3.set_xlabel("a: distance to own task centroid (mahalanobis)")
    ax3.set_ylabel("b: distance to nearest other task centroid (mahalanobis)")
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax3.set_title(f"{scope_name} | per-point a vs b\n(mean (b−a)/max(a,b) = {sil_proxy:+.4f})")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)

    fig.suptitle(f"GR00T-N1.6 RoboCasa | task separation | scope={scope_name}", fontsize=12)
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
        X, tasks, _succ, task_id_to_name = load_scope(args.split_root, scope)
        print(f"  X={X.shape}, tasks={sorted(task_id_to_name.items())}")
        out_path = args.out_dir / f"mahalanobis_separation_{scope}.png"
        plot_scope(
            X,
            tasks,
            task_id_to_name,
            out_path=out_path,
            tsne_sample=args.tsne_sample,
            random_state=args.random_state,
            cov_reg=args.cov_reg,
            scope_name=scope,
        )


if __name__ == "__main__":
    main()
