#!/usr/bin/env python3
"""Coarse 3-label cluster analysis: Coffee / Open / PnP.

Labels:
  0 CoffeeSetupMug
  1 Open (OpenSingleDoor + OpenDrawer)
  2 PnP  (PnPCounterToCab + PnPSinkToCounter + PnPCounterToStove)

Outputs centroid distance + silhouette (Euclidean & Mahalanobis) for all
scopes, and LDA / Mahalanobis-t-SNE / a-vs-b visualization for scope=all.
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
    / "cluster_coarse_group"
)


COARSE_LABEL_NAMES = {0: "Coffee", 1: "Open", 2: "PnP"}

TASK_TO_COARSE = {
    "CoffeeSetupMug": 0,
    "OpenSingleDoor": 1,
    "OpenDrawer": 1,
    "PnPCounterToCab": 2,
    "PnPSinkToCounter": 2,
    "PnPCounterToStove": 2,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--scopes",
        nargs="+",
        default=["all", "train", "val_seen", "val_unseen"],
        choices=("all", "train", "val_seen", "val_unseen"),
    )
    p.add_argument("--plot-scopes", nargs="+", default=["all"])
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--tsne-sample", type=int, default=5000)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--cov-reg", type=float, default=1e-3)
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


def load_scope(split_root: Path, scope: str) -> tuple[np.ndarray, np.ndarray]:
    rows = load_manifest(split_root, scope)
    Xs, coarse = [], []
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        f_arr = pooled_hidden_states(rec)
        n = f_arr.shape[0]
        if n == 0:
            continue
        lab = TASK_TO_COARSE[r["task"]]
        Xs.append(f_arr)
        coarse.append(np.full(n, lab, dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(coarse)


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


def centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.sort(np.unique(labels))
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    return unique, cents


def pairwise(C: np.ndarray) -> np.ndarray:
    diff = C[:, None, :] - C[None, :, :]
    return np.linalg.norm(diff.astype(np.float64), axis=-1)


def silhouette_info(X, labels, sample_size, random_state):
    unique, counts = np.unique(labels, return_counts=True)
    info = {
        "n_points": int(X.shape[0]),
        "n_clusters": int(len(unique)),
        "cluster_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
    }
    if len(unique) < 2 or counts.min() < 2:
        info["score"] = None
        return info
    ss = sample_size if X.shape[0] > sample_size else None
    info["sample_size"] = int(ss or X.shape[0])
    info["score"] = float(silhouette_score(X, labels, metric="euclidean", sample_size=ss, random_state=random_state))
    return info


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


def analyse(X, coarse, *, sample_size, random_state, cov_reg):
    cov = pooled_within_cov(X, coarse, reg=cov_reg)
    Xw = whiten(X, cov)
    unique, eu_cents = centroids(X, coarse)
    _, mh_cents = centroids(Xw, coarse)
    names = [COARSE_LABEL_NAMES[int(k)] for k in unique]
    return {
        "feature_shape": list(X.shape),
        "labels_order": [int(k) for k in unique],
        "label_names": names,
        "euclidean": {
            "centroid_pairwise_distance": pairwise(eu_cents).tolist(),
            "silhouette": silhouette_info(X, coarse, sample_size, random_state),
        },
        "mahalanobis": {
            "cov_regularization": cov_reg,
            "centroid_pairwise_distance": pairwise(mh_cents).tolist(),
            "silhouette": silhouette_info(Xw, coarse, sample_size, random_state),
        },
    }


def flatten(results):
    rows = []
    for scope, payload in results.items():
        names = payload["label_names"]
        for metric in ("euclidean", "mahalanobis"):
            sil = payload[metric]["silhouette"]
            rows.append({"scope": scope, "metric": metric, "row_kind": "silhouette",
                         "i": "", "j": "", "name_i": "", "name_j": "",
                         "value": sil.get("score"),
                         "n_points": sil.get("n_points"),
                         "n_clusters": sil.get("n_clusters"),
                         "sample_size": sil.get("sample_size")})
            dist = np.asarray(payload[metric]["centroid_pairwise_distance"])
            n = dist.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    rows.append({"scope": scope, "metric": metric, "row_kind": "centroid_distance",
                                 "i": i, "j": j, "name_i": names[i], "name_j": names[j],
                                 "value": float(dist[i, j]),
                                 "n_points": "", "n_clusters": "", "sample_size": ""})
            for i in range(n):
                rows.append({"scope": scope, "metric": metric, "row_kind": "centroid_mean_dist_to_others",
                             "i": i, "j": "", "name_i": names[i], "name_j": "",
                             "value": float(dist[i].sum() / max(n - 1, 1)),
                             "n_points": "", "n_clusters": "", "sample_size": ""})
    return rows


def write_tsv(path, rows):
    fields = ["scope", "metric", "row_kind", "i", "j", "name_i", "name_j", "value", "n_points", "n_clusters", "sample_size"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def label_colormap(unique_labels: np.ndarray) -> dict[int, tuple]:
    cmap = plt.get_cmap("tab10")
    return {int(c): cmap(i % 10) for i, c in enumerate(unique_labels)}


def scatter_with_centroids(ax, P2, labels, colors, title):
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(P2[m, 0], P2[m, 1], s=4, c=[colors[int(c)]], alpha=0.35, label=COARSE_LABEL_NAMES[int(c)], linewidths=0)
    for c in sorted(np.unique(labels)):
        m = labels == c
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        sx, sy = P2[m, 0].std(), P2[m, 1].std()
        ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, edgecolor=colors[int(c)], facecolor="none", lw=1.2, alpha=0.8))
        ax.scatter([cx], [cy], s=160, marker="X", c=[colors[int(c)]], edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


def subsample(*arrays, n, rng):
    total = arrays[0].shape[0]
    if total <= n:
        return arrays
    idx = rng.choice(total, size=n, replace=False)
    return tuple(a[idx] for a in arrays)


def lda_2d(X, labels, random_state):
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    Z = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X.astype(np.float64), labels)
    if Z.shape[1] == 1:
        rng = np.random.default_rng(random_state)
        Z = np.column_stack([Z[:, 0], rng.normal(scale=0.02 * (Z[:, 0].std() + 1e-9), size=Z.shape[0])])
    return Z


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_a_b(Xn: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.sort(np.unique(labels))
    cents = []
    for c in unique:
        m = Xn[labels == c].mean(axis=0)
        cents.append(m / max(np.linalg.norm(m), 1e-12))
    cents = np.stack(cents, axis=0)
    own_idx = np.searchsorted(unique, labels)
    sim = np.clip(Xn @ cents.T, -1.0, 1.0)
    dist = 1.0 - sim
    own = dist[np.arange(len(labels)), own_idx]
    others = dist.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    near = others.min(axis=1)
    return own, near


def _plot_ab(ax, own, other, labels, colors, xlabel, ylabel, title):
    for c in np.sort(np.unique(labels)):
        m = labels == c
        ax.scatter(own[m], other[m], s=4, c=[colors[int(c)]], alpha=0.35,
                   label=COARSE_LABEL_NAMES[int(c)], linewidths=0)
    lo = float(min(own.min(), other.min()))
    hi = float(max(own.max(), other.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax.set_title(f"{title}\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)


def plot_scope(X, coarse, *, out_path, tsne_sample, random_state, cov_reg, scope_name):
    cov = pooled_within_cov(X, coarse, reg=cov_reg)
    Xw = whiten(X, cov)
    Xn = l2_normalize(X)

    Z_lda_raw = lda_2d(X, coarse, random_state=random_state)
    Z_lda_cos = lda_2d(Xn, coarse, random_state=random_state)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=min(tsne_sample, X.shape[0]), replace=False)
    coarse_s = coarse[idx]
    X_s = X[idx]
    Xw_s = Xw[idx]
    Xn_s = Xn[idx]

    print(f"  [{scope_name}] t-SNE euclidean...")
    Z_tsne_eu = TSNE(n_components=2, perplexity=30, metric="euclidean",
                     random_state=random_state, init="pca", learning_rate="auto").fit_transform(X_s)
    print(f"  [{scope_name}] t-SNE mahalanobis (whitened+euclidean)...")
    Z_tsne_mh = TSNE(n_components=2, perplexity=30, metric="euclidean",
                     random_state=random_state, init="pca", learning_rate="auto").fit_transform(Xw_s)
    print(f"  [{scope_name}] t-SNE cosine...")
    Z_tsne_co = TSNE(n_components=2, perplexity=30, metric="cosine",
                     random_state=random_state, init="random", learning_rate="auto").fit_transform(Xn_s)

    own_eu, other_eu = per_point_a_b(X, coarse)
    own_mh, other_mh = per_point_a_b(Xw, coarse)
    own_co, other_co = cosine_a_b(Xn, coarse)

    unique = np.sort(np.unique(coarse))
    colors = label_colormap(unique)

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    # row 0: Euclidean
    scatter_with_centroids(axes[0, 0], Z_lda_raw, coarse, colors, f"{scope_name} | Euclidean | LDA on raw")
    scatter_with_centroids(axes[0, 1], Z_tsne_eu, coarse_s, colors, f"{scope_name} | Euclidean | t-SNE (metric=eu)")
    _plot_ab(axes[0, 2], own_eu, other_eu, coarse, colors,
             "a: euclid dist to own centroid", "b: euclid dist to nearest other",
             f"{scope_name} | Euclidean | per-point a vs b")
    # row 1: Mahalanobis
    scatter_with_centroids(axes[1, 0], Z_lda_raw, coarse, colors, f"{scope_name} | Mahalanobis | LDA on raw (≡ Eu)")
    scatter_with_centroids(axes[1, 1], Z_tsne_mh, coarse_s, colors, f"{scope_name} | Mahalanobis | t-SNE on whitened")
    _plot_ab(axes[1, 2], own_mh, other_mh, coarse, colors,
             "a: maha dist to own centroid", "b: maha dist to nearest other",
             f"{scope_name} | Mahalanobis | per-point a vs b")
    # row 2: Cosine
    scatter_with_centroids(axes[2, 0], Z_lda_cos, coarse, colors, f"{scope_name} | Cosine | LDA on L2-norm")
    scatter_with_centroids(axes[2, 1], Z_tsne_co, coarse_s, colors, f"{scope_name} | Cosine | t-SNE (metric=cos)")
    _plot_ab(axes[2, 2], own_co, other_co, coarse, colors,
             "a: cosine dist to own centroid", "b: cosine dist to nearest other",
             f"{scope_name} | Cosine | per-point a vs b")

    fig.suptitle(f"GR00T-N1.6 RoboCasa | coarse Coffee/Open/PnP | scope={scope_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    plot_cache = {}
    for scope in args.scopes:
        print(f"[{scope}] loading...")
        X, coarse = load_scope(args.split_root, scope)
        unique, counts = np.unique(coarse, return_counts=True)
        print(f"  X={X.shape}, label counts: " + ", ".join(f"{COARSE_LABEL_NAMES[int(u)]}={c}" for u, c in zip(unique, counts)))
        results[scope] = analyse(X, coarse, sample_size=args.sample_size, random_state=args.random_state, cov_reg=args.cov_reg)
        if scope in args.plot_scopes:
            plot_cache[scope] = (X, coarse)

    with (args.out_dir / "cluster_coarse_group.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    write_tsv(args.out_dir / "cluster_coarse_group.tsv", flatten(results))

    for scope, (X, coarse) in plot_cache.items():
        plot_scope(X, coarse, out_path=args.out_dir / f"coarse_group_separation_{scope}.png",
                   tsne_sample=args.tsne_sample, random_state=args.random_state,
                   cov_reg=args.cov_reg, scope_name=scope)
    print(args.out_dir)


if __name__ == "__main__":
    main()
