#!/usr/bin/env python3
"""PnP-only instruction-level cluster analysis + 3-metric visualization.

Labels = unique task_description string (instruction).
Visualization colors by task (3 colors) so instruction structure within PnP
becomes visible. Instruction-level centroids are NOT drawn individually
(>100 of them); instead t-SNE/LDA shows the point distribution.
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
    / "cluster_pnp_instruction"
)
PNP_TASKS = {"PnPCounterToCab", "PnPSinkToCounter", "PnPCounterToStove"}
TASK_COLOR = {
    "PnPCounterToCab": (0.85, 0.32, 0.27, 1.0),     # red
    "PnPSinkToCounter": (0.27, 0.55, 0.85, 1.0),    # blue
    "PnPCounterToStove": (0.35, 0.70, 0.32, 1.0),   # green
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--sample-size", type=int, default=8000)
    p.add_argument("--tsne-sample", type=int, default=6000)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--cov-reg", type=float, default=1e-3)
    p.add_argument("--min-cluster-size", type=int, default=2,
                   help="instruction clusters with fewer points are dropped from silhouette")
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


def load_pnp(split_root):
    rows = load_manifest(split_root)
    Xs, task_names, instr = [], [], []
    for r in rows:
        if r["task"] not in PNP_TASKS:
            continue
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        f_arr = pooled_hidden_states(rec)
        n = f_arr.shape[0]
        if n == 0:
            continue
        Xs.append(f_arr)
        task_names.extend([r["task"]] * n)
        instr.extend([rec.get("task_description", "<missing>")] * n)
    X = np.concatenate(Xs, axis=0)
    return X, np.asarray(task_names), np.asarray(instr)


def encode_labels(strings):
    uniq, inv = np.unique(strings, return_inverse=True)
    return inv.astype(np.int64), uniq


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


def silhouette_metric(X, labels, metric, sample_size, random_state, min_cluster_size):
    unique, counts = np.unique(labels, return_counts=True)
    keep = counts >= min_cluster_size
    valid_clusters = unique[keep]
    if len(valid_clusters) < 2:
        return {"score": None, "n_clusters": int(len(unique)),
                "n_valid_clusters": int(len(valid_clusters)),
                "reason": "need ≥2 clusters with ≥min_cluster_size points"}
    mask = np.isin(labels, valid_clusters)
    Xv, yv = X[mask], labels[mask]
    ss = sample_size if Xv.shape[0] > sample_size else None
    return {
        "score": float(silhouette_score(Xv, yv, metric=metric, sample_size=ss, random_state=random_state)),
        "n_points": int(Xv.shape[0]),
        "n_clusters": int(len(unique)),
        "n_valid_clusters": int(len(valid_clusters)),
        "min_cluster_size": int(min_cluster_size),
        "sample_size": int(ss or Xv.shape[0]),
    }


def centroid_stats(X, labels):
    """Return mean within-task vs between-task instruction-centroid distance.

    `labels` is instruction id; the relation to PnP task is captured externally.
    Caller passes a task-id-per-instruction map separately.
    """
    raise NotImplementedError


def instruction_centroid_distance_stats(X, instr_ids, task_per_instr):
    unique = np.sort(np.unique(instr_ids))
    cents = np.stack([X[instr_ids == c].mean(axis=0) for c in unique], axis=0)
    K = len(unique)
    cent_dist = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
    same_task = task_per_instr[:, None] == task_per_instr[None, :]
    iu = np.triu_indices(K, k=1)
    pairs_same = same_task[iu]
    dists = cent_dist[iu]
    intra = dists[pairs_same]
    inter = dists[~pairs_same]
    return {
        "n_instructions": int(K),
        "intra_task_mean": float(intra.mean()) if len(intra) else None,
        "intra_task_median": float(np.median(intra)) if len(intra) else None,
        "inter_task_mean": float(inter.mean()) if len(inter) else None,
        "inter_task_median": float(np.median(inter)) if len(inter) else None,
        "intra_count": int(len(intra)),
        "inter_count": int(len(inter)),
        "ratio_inter_over_intra": float(inter.mean() / intra.mean()) if len(intra) and intra.mean() > 0 else None,
    }


def cosine_instruction_centroid_stats(Xn, instr_ids, task_per_instr):
    unique = np.sort(np.unique(instr_ids))
    cents = []
    for c in unique:
        m = Xn[instr_ids == c].mean(axis=0)
        cents.append(m / max(np.linalg.norm(m), 1e-12))
    cents = np.stack(cents, axis=0)
    K = len(unique)
    sim = np.clip(cents @ cents.T, -1.0, 1.0)
    cent_dist = 1.0 - sim
    same_task = task_per_instr[:, None] == task_per_instr[None, :]
    iu = np.triu_indices(K, k=1)
    pairs_same = same_task[iu]
    dists = cent_dist[iu]
    intra = dists[pairs_same]
    inter = dists[~pairs_same]
    return {
        "n_instructions": int(K),
        "intra_task_mean": float(intra.mean()) if len(intra) else None,
        "intra_task_median": float(np.median(intra)) if len(intra) else None,
        "inter_task_mean": float(inter.mean()) if len(inter) else None,
        "inter_task_median": float(np.median(inter)) if len(inter) else None,
        "intra_count": int(len(intra)),
        "inter_count": int(len(inter)),
        "ratio_inter_over_intra": float(inter.mean() / intra.mean()) if len(intra) and intra.mean() > 0 else None,
    }


def task_to_id(task_names):
    name_to_id = {name: i for i, name in enumerate(sorted(set(task_names)))}
    return np.asarray([name_to_id[n] for n in task_names], dtype=np.int64), name_to_id


def lda_2d(X, labels, random_state):
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    if n_comp < 2:
        rng = np.random.default_rng(random_state)
        Z1 = LinearDiscriminantAnalysis(n_components=1).fit_transform(X.astype(np.float64), labels)
        Z = np.column_stack([Z1[:, 0], rng.normal(scale=0.02 * (Z1[:, 0].std() + 1e-9), size=Z1.shape[0])])
        return Z
    return LinearDiscriminantAnalysis(n_components=2).fit_transform(X.astype(np.float64), labels)


def scatter_by_task(ax, P2, task_names_arr, title):
    for tname in sorted(set(task_names_arr.tolist())):
        m = task_names_arr == tname
        ax.scatter(P2[m, 0], P2[m, 1], s=4, c=[TASK_COLOR[tname]], alpha=0.35, label=tname, linewidths=0)
    for tname in sorted(set(task_names_arr.tolist())):
        m = task_names_arr == tname
        cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
        sx, sy = P2[m, 0].std(), P2[m, 1].std()
        ax.add_patch(Ellipse((cx, cy), width=2 * sx, height=2 * sy, edgecolor=TASK_COLOR[tname],
                             facecolor="none", lw=1.2, alpha=0.85))
        ax.scatter([cx], [cy], s=180, marker="X", c=[TASK_COLOR[tname]],
                   edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


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
    return own, others.min(axis=1)


def _plot_ab_by_task(ax, own, other, task_names_arr, xlabel, ylabel, title):
    for tname in sorted(set(task_names_arr.tolist())):
        m = task_names_arr == tname
        ax.scatter(own[m], other[m], s=4, c=[TASK_COLOR[tname]], alpha=0.35, label=tname, linewidths=0)
    lo = float(min(own.min(), other.min()))
    hi = float(max(own.max(), other.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax.set_title(f"{title}\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)


def plot_pnp(X, task_names_arr, instr_ids, *, out_path, tsne_sample, random_state, cov_reg):
    cov = pooled_within_cov(X, instr_ids, reg=cov_reg)
    Xw = whiten(X, cov)
    Xn = l2_normalize(X)

    # LDA uses instruction labels (max n_classes-1 dims)
    Z_lda_raw = lda_2d(X, instr_ids, random_state=random_state)
    Z_lda_cos = lda_2d(Xn, instr_ids, random_state=random_state)

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.choice(n, size=min(tsne_sample, n), replace=False)
    task_s = task_names_arr[idx]
    X_s, Xw_s, Xn_s = X[idx], Xw[idx], Xn[idx]

    print(f"  t-SNE euclidean...")
    Z_eu = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(X_s)
    print(f"  t-SNE mahalanobis...")
    Z_mh = TSNE(n_components=2, perplexity=30, metric="euclidean",
                random_state=random_state, init="pca", learning_rate="auto").fit_transform(Xw_s)
    print(f"  t-SNE cosine...")
    Z_co = TSNE(n_components=2, perplexity=30, metric="cosine",
                random_state=random_state, init="random", learning_rate="auto").fit_transform(Xn_s)

    own_eu, other_eu = per_point_ab_eu(X, instr_ids)
    own_mh, other_mh = per_point_ab_eu(Xw, instr_ids)
    own_co, other_co = per_point_ab_cos(Xn, instr_ids)

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    scatter_by_task(axes[0, 0], Z_lda_raw, task_names_arr, "Euclidean | LDA (instr labels, color=task)")
    scatter_by_task(axes[0, 1], Z_eu, task_s, "Euclidean | t-SNE (color=task)")
    _plot_ab_by_task(axes[0, 2], own_eu, other_eu, task_names_arr,
                     "a: euclid dist to own instr", "b: dist to nearest other instr",
                     "Euclidean | per-point a vs b (instr-level)")
    scatter_by_task(axes[1, 0], Z_lda_raw, task_names_arr, "Mahalanobis | LDA (≡ Eu)")
    scatter_by_task(axes[1, 1], Z_mh, task_s, "Mahalanobis | t-SNE on whitened (color=task)")
    _plot_ab_by_task(axes[1, 2], own_mh, other_mh, task_names_arr,
                     "a: maha dist to own instr", "b: dist to nearest other instr",
                     "Mahalanobis | per-point a vs b (instr-level)")
    scatter_by_task(axes[2, 0], Z_lda_cos, task_names_arr, "Cosine | LDA on L2-norm (color=task)")
    scatter_by_task(axes[2, 1], Z_co, task_s, "Cosine | t-SNE (color=task)")
    _plot_ab_by_task(axes[2, 2], own_co, other_co, task_names_arr,
                     "a: cosine dist to own instr", "b: dist to nearest other instr",
                     "Cosine | per-point a vs b (instr-level)")

    fig.suptitle("GR00T-N1.6 RoboCasa | PnP only | instruction-level cluster", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading PnP rollouts...")
    X, task_names_arr, instr_strs = load_pnp(args.split_root)
    instr_ids, instr_uniq = encode_labels(instr_strs)
    task_ids_per_point, task_name_to_id = task_to_id(task_names_arr)
    # task per instruction (use first occurrence)
    task_per_instr = np.empty(len(instr_uniq), dtype=np.int64)
    for k in range(len(instr_uniq)):
        first_idx = int(np.where(instr_ids == k)[0][0])
        task_per_instr[k] = task_ids_per_point[first_idx]
    print(f"  X={X.shape}, n_instructions={len(instr_uniq)}, "
          f"task counts: " + ", ".join(f"{t}={int((task_names_arr==t).sum())}" for t in sorted(set(task_names_arr.tolist()))))

    Xn = l2_normalize(X)
    cov = pooled_within_cov(X, instr_ids, reg=args.cov_reg)
    Xw = whiten(X, cov)

    results = {
        "feature_shape": list(X.shape),
        "n_instructions": int(len(instr_uniq)),
        "task_name_to_id": task_name_to_id,
        "silhouette_instruction_level": {
            "euclidean": silhouette_metric(X, instr_ids, "euclidean", args.sample_size, args.random_state, args.min_cluster_size),
            "mahalanobis": silhouette_metric(Xw, instr_ids, "euclidean", args.sample_size, args.random_state, args.min_cluster_size),
            "cosine": silhouette_metric(Xn, instr_ids, "cosine", args.sample_size, args.random_state, args.min_cluster_size),
        },
        "silhouette_task_level": {
            "euclidean": silhouette_metric(X, task_ids_per_point, "euclidean", args.sample_size, args.random_state, 2),
            "mahalanobis": silhouette_metric(Xw, task_ids_per_point, "euclidean", args.sample_size, args.random_state, 2),
            "cosine": silhouette_metric(Xn, task_ids_per_point, "cosine", args.sample_size, args.random_state, 2),
        },
        "instruction_centroid_distance": {
            "euclidean": instruction_centroid_distance_stats(X, instr_ids, task_per_instr),
            "mahalanobis": instruction_centroid_distance_stats(Xw, instr_ids, task_per_instr),
            "cosine": cosine_instruction_centroid_stats(Xn, instr_ids, task_per_instr),
        },
    }

    with (args.out_dir / "cluster_pnp_instruction.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    # flat TSV
    rows = []
    for metric in ("euclidean", "mahalanobis", "cosine"):
        s_instr = results["silhouette_instruction_level"][metric]
        s_task = results["silhouette_task_level"][metric]
        d_stats = results["instruction_centroid_distance"][metric]
        rows.append({"metric": metric, "row_kind": "silhouette_instr_level", "value": s_instr.get("score"),
                     "n_clusters": s_instr.get("n_clusters"), "n_valid_clusters": s_instr.get("n_valid_clusters"),
                     "n_points": s_instr.get("n_points")})
        rows.append({"metric": metric, "row_kind": "silhouette_task_level", "value": s_task.get("score"),
                     "n_clusters": s_task.get("n_clusters"), "n_valid_clusters": s_task.get("n_valid_clusters"),
                     "n_points": s_task.get("n_points")})
        rows.append({"metric": metric, "row_kind": "instr_centroid_intra_task_mean",
                     "value": d_stats["intra_task_mean"], "n_clusters": d_stats["n_instructions"]})
        rows.append({"metric": metric, "row_kind": "instr_centroid_inter_task_mean",
                     "value": d_stats["inter_task_mean"], "n_clusters": d_stats["n_instructions"]})
        rows.append({"metric": metric, "row_kind": "ratio_inter_over_intra",
                     "value": d_stats["ratio_inter_over_intra"], "n_clusters": d_stats["n_instructions"]})

    fields = ["metric", "row_kind", "value", "n_clusters", "n_valid_clusters", "n_points"]
    with (args.out_dir / "cluster_pnp_instruction.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            for k in fields:
                r.setdefault(k, "")
            writer.writerow(r)

    plot_pnp(X, task_names_arr, instr_ids,
             out_path=args.out_dir / "pnp_instruction_separation.png",
             tsne_sample=args.tsne_sample, random_state=args.random_state, cov_reg=args.cov_reg)
    print(args.out_dir)


if __name__ == "__main__":
    main()
