#!/usr/bin/env python3
"""Unified static cluster analysis for SAFE GR00T-N1.6 RoboCasa rollouts.

Consolidates the previous 11 compute_and_plot_* / plot_*_separation /
plot_zmean_* scripts behind a single CLI:

  --labeling {task, success_fail, task_failure, task7, fine_open,
              coarse_group, pnp_instruction}
  --feature  {z_inference, z_mean, z_last, z_disp, z_concat, z_path, h_T}
  --metrics  {euclidean, mahalanobis, cosine}        (multi)
  --scope    {all, train, val_seen, val_unseen, pnp_only}
  --truncate-t N            cap inferences for rollout-level features
  --ckpt PATH               required when feature=h_T

Outputs (under --out-dir):
  summary.tsv                              (appended; one row per metric)
  <labeling>__<feature>[_t<N>]__<scope>.json  full numerical results
  <labeling>__<feature>[_t<N>]__<scope>__<metric>.{png,pdf}  3-panel figure
    (LDA / t-SNE / per-point a-vs-b)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow `from core import ...` when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.aggregation import aggregate, broadcast_labels
from core.distance import l2_normalize, pooled_within_cov, whiten
from core.features import pooled_hidden_states
from core.io import load_manifest, load_pkl, pkl_path
from core.labeling import PNP_TASKS, apply as apply_labeling
from core.lstm import load_detector
from core.metrics import (
    centroid_distance,
    per_point_ab,
    silhouette_proxy_mean,
    silhouette_safe,
)
from core.plotting import (
    categorical_colormap,
    plot_ab,
    scatter_with_centroids,
)
from core.projection import lda_2d, run_tsne


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
    / "cluster_static"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--labeling",
        required=True,
        choices=("task", "success_fail", "task_failure", "task7",
                 "fine_open", "coarse_group", "pnp_instruction"),
    )
    p.add_argument(
        "--feature",
        required=True,
        choices=("z_inference", "z_mean", "z_last", "z_disp",
                 "z_concat", "z_path", "h_T"),
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["euclidean", "mahalanobis"],
        choices=("euclidean", "mahalanobis", "cosine"),
    )
    p.add_argument(
        "--scope",
        default="all",
        choices=("all", "train", "val_seen", "val_unseen", "pnp_only"),
    )
    p.add_argument("--truncate-t", type=int, default=None,
                   help="Cap inferences per rollout to first N (rollout-level features only)")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="LSTM detector checkpoint, required when --feature h_T")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tsne-perplexity", type=int, default=30)
    p.add_argument("--cov-reg", type=float, default=1e-3)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ---- data loading ----

def gather(split_root: Path, scope: str) -> list[dict]:
    rows = load_manifest(split_root)
    records = []
    for r in rows:
        if scope == "pnp_only":
            if r["task"] not in PNP_TASKS:
                continue
        elif scope != "all":
            if r["split"] != scope:
                continue
        rec = load_pkl(pkl_path(split_root, r))
        z = pooled_hidden_states(rec)
        if z.shape[0] == 0:
            continue
        records.append({
            "z": z,
            "task": r["task"],
            "split": r["split"],
            "success": int(r["success"]),
            "episode_idx": int(r["episode_idx"]),
            "task_description": rec.get("task_description", ""),
        })
    return records


# ---- per-metric analysis ----

def compute_for_metric(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    cov_reg: float,
    random_state: int,
) -> dict:
    """Compute silhouette, centroid distances, and a-vs-b in the given metric."""
    if metric == "euclidean":
        X_dist = X
        sil = silhouette_safe(X, labels, metric="euclidean", random_state=random_state)
        own, other = per_point_ab(X, labels, metric="euclidean")
        cd = centroid_distance(X, labels, metric="euclidean")
    elif metric == "mahalanobis":
        cov = pooled_within_cov(X, labels, reg=cov_reg)
        X_dist = whiten(X, cov)
        sil = silhouette_safe(X_dist, labels, metric="euclidean", random_state=random_state)
        own, other = per_point_ab(X_dist, labels, metric="euclidean")
        cd = centroid_distance(X_dist, labels, metric="euclidean")
    elif metric == "cosine":
        X_dist = l2_normalize(X)
        sil = silhouette_safe(X, labels, metric="cosine", random_state=random_state)
        own, other = per_point_ab(X, labels, metric="cosine")
        cd = centroid_distance(X, labels, metric="cosine")
    else:
        raise ValueError(f"unknown metric: {metric}")

    return {
        "silhouette": sil,
        "centroid_distance": cd,
        "own_dist": own,
        "other_dist": other,
        "proxy": silhouette_proxy_mean(own, other),
        "X_distance": X_dist,
    }


# ---- output tag helpers ----

def make_tag(labeling_scheme: str, feature: str, truncate_t: int | None, scope: str) -> str:
    tag = f"{labeling_scheme}__{feature}"
    if truncate_t is not None:
        tag += f"_t{truncate_t}"
    tag += f"__{scope}"
    return tag


# ---- visualization ----

def visualize(
    X: np.ndarray,
    labels: np.ndarray,
    res: dict,
    *,
    metric: str,
    labeling,
    feature: str,
    truncate_t: int | None,
    scope: str,
    tsne_perplexity: int,
    random_state: int,
    out_dir: Path,
) -> None:
    colors = categorical_colormap(np.unique(labels), fail_label=labeling.fail_label)

    # LDA input: for cosine use L2-normalized X; otherwise raw X (LDA is invariant to whitening)
    Z_lda_in = res["X_distance"] if metric == "cosine" else X
    Z_lda = lda_2d(Z_lda_in, labels, random_state=random_state)

    # t-SNE input: same space as the silhouette/centroid computation
    if metric == "cosine":
        Z_tsne_in, Z_tsne_metric = X, "cosine"
    elif metric == "mahalanobis":
        Z_tsne_in, Z_tsne_metric = res["X_distance"], "euclidean"
    else:
        Z_tsne_in, Z_tsne_metric = X, "euclidean"

    print(f"    t-SNE on shape {Z_tsne_in.shape} (metric={Z_tsne_metric})")
    Z_tsne = run_tsne(
        Z_tsne_in, metric=Z_tsne_metric,
        perplexity=tsne_perplexity, random_state=random_state,
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    if Z_lda is not None:
        scatter_with_centroids(
            axes[0], Z_lda, labels, colors, labeling.names,
            title=f"LDA 2D ({metric})", fail_label=labeling.fail_label,
        )
    else:
        axes[0].text(0.5, 0.5, "LDA unavailable\n(need ≥2 classes)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"LDA 2D ({metric})")

    scatter_with_centroids(
        axes[1], Z_tsne, labels, colors, labeling.names,
        title=f"t-SNE ({Z_tsne_metric})", fail_label=labeling.fail_label,
    )
    plot_ab(
        axes[2], res["own_dist"], res["other_dist"], labels, colors, labeling.names,
        metric_name=metric, fail_label=labeling.fail_label,
    )

    sil_s = f"{res['silhouette']:+.4f}" if res["silhouette"] is not None else "NA"
    suptitle = (
        f"labeling={labeling.scheme}  feature={feature}"
        + (f"  truncate-t={truncate_t}" if truncate_t is not None else "")
        + f"  scope={scope}  metric={metric}  silhouette={sil_s}"
    )
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)

    tag = make_tag(labeling.scheme, feature, truncate_t, scope)
    out_path = out_dir / f"{tag}__{metric}.png"
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"    saved {out_path}")


def write_outputs(
    results: dict[str, dict],
    *,
    labeling,
    feature: str,
    truncate_t: int | None,
    scope: str,
    n_rollouts: int,
    n_points: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_tsv = out_dir / "summary.tsv"
    fields = [
        "labeling", "feature", "truncate_t", "scope", "metric",
        "silhouette", "ab_proxy_mean", "n_clusters", "n_rollouts", "n_points",
    ]
    write_header = not summary_tsv.exists()
    with summary_tsv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if write_header:
            writer.writeheader()
        for metric, res in results.items():
            writer.writerow({
                "labeling": labeling.scheme,
                "feature": feature,
                "truncate_t": "" if truncate_t is None else truncate_t,
                "scope": scope,
                "metric": metric,
                "silhouette": res["silhouette"],
                "ab_proxy_mean": res["proxy"],
                "n_clusters": len(labeling.names),
                "n_rollouts": n_rollouts,
                "n_points": n_points,
            })

    full = {
        metric: {
            "silhouette": res["silhouette"],
            "centroid_distance": res["centroid_distance"].tolist(),
            "proxy": res["proxy"],
        }
        for metric, res in results.items()
    }
    tag = make_tag(labeling.scheme, feature, truncate_t, scope)
    json_path = out_dir / f"{tag}.json"
    with json_path.open("w") as f:
        json.dump({
            "config": {
                "labeling": labeling.scheme,
                "feature": feature,
                "truncate_t": truncate_t,
                "scope": scope,
                "n_rollouts": n_rollouts,
                "n_points": n_points,
            },
            "label_names": {str(k): v for k, v in labeling.names.items()},
            "results": full,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  appended summary -> {summary_tsv}")
    print(f"  wrote {json_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading rollouts (scope={args.scope}) from {args.split_root.name} ...")
    records = gather(args.split_root, args.scope)
    print(f"  n_rollouts={len(records)}")

    labeling = apply_labeling(args.labeling, records)
    print(f"  labeling={args.labeling}  n_clusters={len(labeling.names)}")
    unique, counts = np.unique(labeling.labels, return_counts=True)
    for k, n in zip(unique, counts):
        print(f"    {int(k):>3d} = {labeling.names[int(k)]:30s}  n_rollouts={int(n)}")

    lstm_model = None
    if args.feature == "h_T":
        if args.ckpt is None:
            raise SystemExit("--feature h_T requires --ckpt")
        print(f"  loading LSTM detector: {args.ckpt}")
        lstm_model = load_detector(args.ckpt, hidden_dim=args.hidden_dim, device=args.device)

    X, rollout_idx = aggregate(
        records,
        args.feature,
        truncate_t=args.truncate_t,
        lstm_model=lstm_model,
        device=args.device,
    )
    labels = broadcast_labels(labeling.labels, rollout_idx)
    print(f"  feature={args.feature}  X.shape={X.shape}  n_unique_labels={len(np.unique(labels))}")

    results: dict[str, dict] = {}
    for metric in args.metrics:
        print(f"\n--- metric: {metric} ---")
        res = compute_for_metric(
            X, labels,
            metric=metric, cov_reg=args.cov_reg, random_state=args.random_state,
        )
        sil_s = f"{res['silhouette']:+.4f}" if res["silhouette"] is not None else "NA"
        print(f"  silhouette = {sil_s}")
        print(f"  centroid_distance shape = {res['centroid_distance'].shape}")
        print(f"  a-vs-b proxy mean       = {res['proxy']:+.4f}")
        results[metric] = res
        if not args.no_plot:
            visualize(
                X, labels, res,
                metric=metric,
                labeling=labeling,
                feature=args.feature,
                truncate_t=args.truncate_t,
                scope=args.scope,
                tsne_perplexity=args.tsne_perplexity,
                random_state=args.random_state,
                out_dir=args.out_dir,
            )

    write_outputs(
        results,
        labeling=labeling, feature=args.feature, truncate_t=args.truncate_t,
        scope=args.scope, n_rollouts=len(records), n_points=int(X.shape[0]),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
