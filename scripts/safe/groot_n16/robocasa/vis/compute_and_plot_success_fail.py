#!/usr/bin/env python3
"""Success vs Failure cluster analysis + 3-metric visualization."""

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
    / "cluster_success_fail"
)
LABEL_NAMES = {0: "failure", 1: "success"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--scopes", nargs="+", default=["all", "train", "val_seen", "val_unseen"])
    p.add_argument("--plot-scopes", nargs="+", default=["all", "val_unseen"])
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--tsne-sample", type=int, default=5000)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--cov-reg", type=float, default=1e-3)
    return p.parse_args()


def tensor_to_numpy(v: Any) -> np.ndarray:
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
    return (
        split_root / row["split"] / row["task"]
        / f"task{int(row['task_id'])}--ep{int(row['episode_idx'])}--succ{int(row['success'])}.pkl"
    )


def pooled_hidden_states(record):
    feats = []
    for h in record["hidden_states"]:
        a = tensor_to_numpy(h).astype(np.float32, copy=False)
        feats.append(a.mean(axis=(0, 1)))
    return np.stack(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)


def load_scope(split_root, scope):
    rows = load_manifest(split_root, scope)
    Xs, succ = [], []
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        f_arr = pooled_hidden_states(rec)
        n = f_arr.shape[0]
        if n == 0:
            continue
        Xs.append(f_arr)
        succ.append(np.full(n, int(r["success"]), dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(succ)


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


def cosine_pairwise(C):
    sim = np.clip(C @ C.T, -1.0, 1.0)
    return 1.0 - sim


def cosine_centroids(Xn, labels):
    unique = np.sort(np.unique(labels))
    cents = []
    for c in unique:
        m = Xn[labels == c].mean(axis=0)
        cents.append(m / max(np.linalg.norm(m), 1e-12))
    return unique, np.stack(cents, axis=0)


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


def analyse(X, succ, sample_size, random_state, cov_reg):
    Xn = l2_normalize(X)
    cov = pooled_within_cov(X, succ, reg=cov_reg)
    Xw = whiten(X, cov)
    _, eu_cents = centroids(X, succ)
    _, mh_cents = centroids(Xw, succ)
    _, co_cents = cosine_centroids(Xn, succ)
    return {
        "feature_shape": list(X.shape),
        "label_names": ["failure", "success"],
        "euclidean": {
            "centroid_pairwise_distance": pairwise(eu_cents).tolist(),
            "silhouette": silhouette_info(X, succ, sample_size, random_state, "euclidean"),
        },
        "mahalanobis": {
            "cov_regularization": cov_reg,
            "centroid_pairwise_distance": pairwise(mh_cents).tolist(),
            "silhouette": silhouette_info(Xw, succ, sample_size, random_state, "euclidean"),
        },
        "cosine": {
            "centroid_pairwise_distance": cosine_pairwise(co_cents).tolist(),
            "silhouette": silhouette_info(Xn, succ, sample_size, random_state, "cosine"),
        },
    }


def flatten(results):
    rows = []
    for scope, payload in results.items():
        for metric in ("euclidean", "mahalanobis", "cosine"):
            sil = payload[metric]["silhouette"]
            rows.append({"scope": scope, "metric": metric, "row_kind": "silhouette",
                         "value": sil.get("score"),
                         "n_points": sil.get("n_points"),
                         "n_clusters": sil.get("n_clusters"),
                         "sample_size": sil.get("sample_size"),
                         "name_pair": ""})
            dist = np.asarray(payload[metric]["centroid_pairwise_distance"])
            rows.append({"scope": scope, "metric": metric, "row_kind": "centroid_distance",
                         "value": float(dist[0, 1]), "n_points": "", "n_clusters": "",
                         "sample_size": "", "name_pair": "failure↔success"})
    return rows


def write_tsv(path, rows):
    fields = ["scope", "metric", "row_kind", "value", "n_points", "n_clusters", "sample_size", "name_pair"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def label_colormap(unique):
    return {0: (0.85, 0.32, 0.27, 1.0), 1: (0.27, 0.55, 0.85, 1.0)}  # failure red, success blue


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


def plot_scope(X, succ, *, out_path, tsne_sample, random_state, cov_reg, scope_name):
    cov = pooled_within_cov(X, succ, reg=cov_reg)
    Xw = whiten(X, cov)
    Xn = l2_normalize(X)
    Z_lda_raw = lda_2d(X, succ, random_state=random_state)
    Z_lda_cos = lda_2d(Xn, succ, random_state=random_state)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=min(tsne_sample, X.shape[0]), replace=False)
    succ_s = succ[idx]; X_s, Xw_s, Xn_s = X[idx], Xw[idx], Xn[idx]

    print(f"  [{scope_name}] t-SNE euclidean...")
    Z_eu = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(X_s)
    print(f"  [{scope_name}] t-SNE mahalanobis...")
    Z_mh = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(Xw_s)
    print(f"  [{scope_name}] t-SNE cosine...")
    Z_co = TSNE(n_components=2, perplexity=30, metric="cosine",
                random_state=random_state, init="random", learning_rate="auto").fit_transform(Xn_s)

    own_eu, other_eu = per_point_ab_eu(X, succ)
    own_mh, other_mh = per_point_ab_eu(Xw, succ)
    own_co, other_co = per_point_ab_cos(Xn, succ)
    colors = label_colormap(np.unique(succ))

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    scatter_with_centroids(axes[0, 0], Z_lda_raw, succ, colors, f"{scope_name} | Euclidean | LDA on raw")
    scatter_with_centroids(axes[0, 1], Z_eu, succ_s, colors, f"{scope_name} | Euclidean | t-SNE (metric=eu)")
    _plot_ab(axes[0, 2], own_eu, other_eu, succ, colors,
             "a: euclid dist to own", "b: euclid dist to other", f"{scope_name} | Euclidean | a vs b")
    scatter_with_centroids(axes[1, 0], Z_lda_raw, succ, colors, f"{scope_name} | Mahalanobis | LDA on raw (≡ Eu)")
    scatter_with_centroids(axes[1, 1], Z_mh, succ_s, colors, f"{scope_name} | Mahalanobis | t-SNE on whitened")
    _plot_ab(axes[1, 2], own_mh, other_mh, succ, colors,
             "a: maha dist to own", "b: maha dist to other", f"{scope_name} | Mahalanobis | a vs b")
    scatter_with_centroids(axes[2, 0], Z_lda_cos, succ, colors, f"{scope_name} | Cosine | LDA on L2-norm")
    scatter_with_centroids(axes[2, 1], Z_co, succ_s, colors, f"{scope_name} | Cosine | t-SNE (metric=cos)")
    _plot_ab(axes[2, 2], own_co, other_co, succ, colors,
             "a: cosine dist to own", "b: cosine dist to other", f"{scope_name} | Cosine | a vs b")

    fig.suptitle(f"GR00T-N1.6 RoboCasa | success vs failure | scope={scope_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, plot_cache = {}, {}
    for scope in args.scopes:
        print(f"[{scope}] loading...")
        X, succ = load_scope(args.split_root, scope)
        unique, counts = np.unique(succ, return_counts=True)
        print(f"  X={X.shape}, " + ", ".join(f"{LABEL_NAMES[int(u)]}={c}" for u, c in zip(unique, counts)))
        results[scope] = analyse(X, succ, sample_size=args.sample_size, random_state=args.random_state, cov_reg=args.cov_reg)
        if scope in args.plot_scopes:
            plot_cache[scope] = (X, succ)

    with (args.out_dir / "cluster_success_fail.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    write_tsv(args.out_dir / "cluster_success_fail.tsv", flatten(results))
    for scope, (X, succ) in plot_cache.items():
        plot_scope(X, succ, out_path=args.out_dir / f"success_fail_separation_{scope}.png",
                   tsne_sample=args.tsne_sample, random_state=args.random_state,
                   cov_reg=args.cov_reg, scope_name=scope)
    print(args.out_dir)


if __name__ == "__main__":
    main()
