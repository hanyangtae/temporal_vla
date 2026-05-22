#!/usr/bin/env python3
"""Diagnose rollout-mean success/failure separability across GR00T feature aggregations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold


ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from run_config import ANALYSIS_ROOT, SPLIT_ROOT  # noqa: E402
from safe_feature_vectors import (  # noqa: E402
    load_manifest,
    parse_aggregation_command,
    pooled_hidden_states,
)


DEFAULT_SPLIT_ROOT = SPLIT_ROOT
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "rollout_mean_aggregation_separability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average each rollout over time, then compare success/failure separation "
            "with LDA and whitened Mahalanobis metrics."
        )
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--aggregation",
        action="append",
        default=None,
        metavar="HORIZON:DIFF",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        choices=("all", "train", "val_seen", "val_unseen"),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Optional task-specific scope inside val_unseen, e.g. OpenDrawer.",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-whiten-components",
        type=int,
        default=32,
        help="Number of PCA-whitened components used for Fisher/LDA diagnostic plots.",
    )
    args = parser.parse_args()
    if args.aggregation is None:
        args.aggregation = ["mean:mean", "mean:concat-2"]
    if args.scope is None:
        args.scope = ["val_unseen"]
    return args


def parse_aggregation(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Expected HORIZON:DIFF aggregation, got {value}")
    horizon, diff = value.split(":", maxsplit=1)
    return horizon, diff


def rollout_mean_features(
    rows: list[dict[str, str]],
    *,
    horizon_idx_rel: str,
    diff_idx_rel: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    horizon_command = parse_aggregation_command(horizon_idx_rel)
    diff_command = parse_aggregation_command(diff_idx_rel)

    features: list[np.ndarray] = []
    labels: list[int] = []
    task_ids: list[int] = []
    task_names: list[str] = []
    episode_indices: list[int] = []

    for row in rows:
        with Path(row["source_path"]).open("rb") as f:
            record = pickle.load(f)
        timestep_features = pooled_hidden_states(
            record,
            horizon_idx_rel=horizon_command,
            diff_idx_rel=diff_command,
        )
        if timestep_features.size == 0:
            continue
        features.append(timestep_features.mean(axis=0))
        labels.append(1 - int(row["success"]))
        task_ids.append(int(row["task_id"]))
        task_names.append(row["task"])
        episode_indices.append(int(row["episode_idx"]))

    if not features:
        raise ValueError("No rollout features were loaded.")

    meta = {
        "failure": np.asarray(labels, dtype=np.int64),
        "task_id": np.asarray(task_ids, dtype=np.int64),
        "task_name": np.asarray(task_names),
        "episode_idx": np.asarray(episode_indices, dtype=np.int64),
    }
    return np.stack(features, axis=0).astype(np.float32), meta


def centered(features: np.ndarray) -> np.ndarray:
    return features - features.mean(axis=0, keepdims=True)


def silhouette_or_none(features: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float | None:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return None
    return float(silhouette_score(features, labels, **kwargs))


def pca_whiten(features: np.ndarray) -> tuple[np.ndarray, PCA]:
    n_components = min(features.shape[0] - 1, features.shape[1])
    if n_components < 1:
        raise ValueError(f"Need at least two samples to whiten, got {features.shape[0]}")
    projector = PCA(n_components=n_components, whiten=True, svd_solver="full")
    return projector.fit_transform(centered(features)), projector


def pca_whiten_capped(features: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA]:
    actual = min(n_components, features.shape[0] - 1, features.shape[1])
    if actual < 1:
        raise ValueError(f"Need at least two samples to whiten, got {features.shape[0]}")
    projector = PCA(n_components=actual, whiten=True, svd_solver="full")
    return projector.fit_transform(centered(features)), projector


def fisher_projection(features: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 2 or counts.min() < 2:
        return None
    mu0 = features[labels == 0].mean(axis=0)
    mu1 = features[labels == 1].mean(axis=0)
    direction = mu1 - mu0
    norm = np.linalg.norm(direction)
    if norm <= 0:
        return None
    direction = direction / norm
    return (features - features.mean(axis=0, keepdims=True)) @ direction


def whitened_centroid_cv_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    cv_splits: int,
    seed: int,
) -> float | None:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 2 or counts.min() < 2:
        return None
    n_splits = min(cv_splits, int(counts.min()))
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty_like(labels)
    for train_idx, test_idx in cv.split(features, labels):
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]

        train_white, projector = pca_whiten(train_features)
        test_white = projector.transform(centered(test_features))
        mu0 = train_white[train_labels == 0].mean(axis=0)
        mu1 = train_white[train_labels == 1].mean(axis=0)
        dist0 = np.linalg.norm(test_white - mu0, axis=1)
        dist1 = np.linalg.norm(test_white - mu1, axis=1)
        pred[test_idx] = (dist1 < dist0).astype(labels.dtype)
    return float(balanced_accuracy_score(labels, pred))


def class_counts(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def score(features: np.ndarray, labels: np.ndarray, *, cv_splits: int, seed: int) -> dict[str, Any]:
    raw_sil = silhouette_or_none(features, labels, metric="euclidean")

    whitened_features, _ = pca_whiten(features)
    whitened_sil = silhouette_or_none(whitened_features, labels, metric="euclidean")

    mu0 = whitened_features[labels == 0].mean(axis=0)
    mu1 = whitened_features[labels == 1].mean(axis=0)
    diff = mu1 - mu0
    whitened_centroid = float(np.linalg.norm(diff))

    projected = fisher_projection(whitened_features, labels)
    fisher_sil = None
    fisher_effect = None
    if projected is not None:
        fisher_sil = silhouette_or_none(projected.reshape(-1, 1), labels, metric="euclidean")
        p0 = projected[labels == 0]
        p1 = projected[labels == 1]
        pooled_std = np.sqrt(0.5 * (p0.var(ddof=1) + p1.var(ddof=1)))
        if pooled_std > 0:
            fisher_effect = float(abs(p1.mean() - p0.mean()) / pooled_std)

    return {
        "n_rollouts": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "failure_counts": class_counts(labels),
        "raw_silhouette": raw_sil,
        "whitened_pca_silhouette": whitened_sil,
        "whitened_centroid_distance": whitened_centroid,
        "fisher_lda_silhouette": fisher_sil,
        "fisher_lda_effect_size": fisher_effect,
        "whitened_centroid_cv_balanced_accuracy": whitened_centroid_cv_balanced_accuracy(
            features,
            labels,
            cv_splits=cv_splits,
            seed=seed,
        ),
    }


def scope_rows(split_root: Path, scope: str, task: str | None = None) -> list[dict[str, str]]:
    rows = load_manifest(split_root, scope)
    if task is not None:
        rows = [row for row in rows if row["task"] == task]
    if not rows:
        raise ValueError(f"No rows for scope={scope}, task={task}")
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scope",
        "task",
        "aggregation",
        "horizon_idx_rel",
        "diff_idx_rel",
        "n_rollouts",
        "feature_dim",
        "n_success",
        "n_failure",
        "raw_silhouette",
        "whitened_pca_silhouette",
        "whitened_centroid_distance",
        "fisher_lda_silhouette",
        "fisher_lda_effect_size",
        "whitened_centroid_cv_balanced_accuracy",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Rollout-Mean Aggregation Separability",
        "",
        "Each rollout is first averaged over time into one feature vector.",
        "The table compares success/failure separation for candidate GR00T feature aggregations.",
        "",
        "Metrics:",
        "- `raw_silhouette`: Euclidean silhouette on rollout-mean features.",
        "- `whitened_pca_silhouette`: silhouette after PCA whitening of rollout-mean features.",
        "- `whitened_centroid_distance`: success/failure centroid distance in the whitened space.",
        "- `fisher_lda_silhouette`: silhouette after projecting whitened features onto the Fisher/LDA direction.",
        "- `fisher_lda_effect_size`: absolute projected class-mean gap divided by pooled standard deviation.",
        "- `whitened_centroid_cv_balanced_accuracy`: cross-validated nearest-centroid balanced accuracy after train-fold PCA whitening.",
        "",
        "| scope | task | aggregation | n | dim | success/failure | raw sil | white sil | white centroid | Fisher sil | Fisher effect | white-CV bal-acc |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        counts = f"{row['n_success']}/{row['n_failure']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scope"]),
                    str(row["task"]),
                    str(row["aggregation"]),
                    str(row["n_rollouts"]),
                    str(row["feature_dim"]),
                    counts,
                    f"{row['raw_silhouette']:.4f}",
                    f"{row['whitened_pca_silhouette']:.4f}",
                    f"{row['whitened_centroid_distance']:.4f}",
                    f"{row['fisher_lda_silhouette']:.4f}",
                    f"{row['fisher_lda_effect_size']:.4f}",
                    f"{row['whitened_centroid_cv_balanced_accuracy']:.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def save_rollout_mean_plots(
    out_dir: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    task_names: np.ndarray,
    scope: str,
    task: str,
    aggregation: str,
    n_components: int,
) -> dict[str, str]:
    plot_dir = out_dir / "plots" / f"{scope}_{task}_{aggregation.replace(':', '_').replace('-', '_')}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    white, _ = pca_whiten_capped(features, n_components)
    failure = labels == 1
    colors = np.where(failure, "#d62728", "#3b63d9")

    # Unsupervised whitened PCA view. This shows whether separation is visible without labels.
    pca_path = plot_dir / "rollout_mean_whitened_pca2_success_failure.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=180)
    ax.scatter(white[~failure, 0], white[~failure, 1], s=18, alpha=0.75, c="#3b63d9", label="success")
    ax.scatter(white[failure, 0], white[failure, 1], s=18, alpha=0.75, c="#d62728", label="failure")
    ax.set_title(f"{scope} {task}\n{aggregation} rollout-mean whitened PCA")
    ax.set_xlabel("whitened PC1")
    ax.set_ylabel("whitened PC2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pca_path, bbox_inches="tight")
    plt.close(fig)

    # Supervised Fisher/LDA direction in whitened space. This is diagnostic, not a t-SNE plot.
    projected = fisher_projection(white, labels)
    lda_path = plot_dir / "rollout_mean_whitened_fisher_lda_success_failure.png"
    if projected is not None:
        rng = np.random.default_rng(0)
        jitter = rng.normal(0.0, 0.035, size=projected.shape[0])
        fig, ax = plt.subplots(figsize=(6.0, 2.8), dpi=180)
        ax.scatter(projected[~failure], jitter[~failure], s=18, alpha=0.75, c="#3b63d9", label="success")
        ax.scatter(projected[failure], jitter[failure], s=18, alpha=0.75, c="#d62728", label="failure")
        ax.axvline(float(np.mean(projected[~failure])), color="#3b63d9", lw=1.5, alpha=0.8)
        ax.axvline(float(np.mean(projected[failure])), color="#d62728", lw=1.5, alpha=0.8)
        ax.set_yticks([])
        ax.set_xlabel("Fisher/LDA projection after PCA whitening")
        ax.set_title(f"{scope} {task}\n{aggregation} rollout-mean Fisher/LDA")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(lda_path, bbox_inches="tight")
        plt.close(fig)

    task_path = plot_dir / "rollout_mean_whitened_pca2_task_id.png"
    unique_tasks = np.unique(task_names)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=180)
    cmap = plt.get_cmap("tab10", max(10, len(unique_tasks)))
    for idx, task_name in enumerate(unique_tasks):
        mask = task_names == task_name
        ax.scatter(white[mask, 0], white[mask, 1], s=18, alpha=0.75, color=cmap(idx), label=str(task_name))
    ax.set_title(f"{scope} {task}\n{aggregation} rollout-mean task view")
    ax.set_xlabel("whitened PC1")
    ax.set_ylabel("whitened PC2")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(task_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "whitened_pca2": str(pca_path),
        "whitened_fisher_lda": str(lda_path),
        "whitened_task_pca2": str(task_path),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    targets: list[tuple[str, str | None]] = [(scope, None) for scope in args.scope]
    targets.extend(("val_unseen", task) for task in args.task)

    for scope, task in targets:
        rows = scope_rows(args.split_root, scope, task)
        for aggregation in args.aggregation:
            horizon, diff = parse_aggregation(aggregation)
            features, meta = rollout_mean_features(
                rows,
                horizon_idx_rel=horizon,
                diff_idx_rel=diff,
            )
            labels = meta["failure"]
            metrics = score(features, labels, cv_splits=args.cv_splits, seed=args.seed)
            plot_paths = save_rollout_mean_plots(
                args.out_dir,
                features=features,
                labels=labels,
                task_names=meta["task_name"],
                scope=scope,
                task=task or "combined",
                aggregation=aggregation,
                n_components=args.plot_whiten_components,
            )
            counts = metrics["failure_counts"]
            flat = {
                "scope": scope,
                "task": task or "combined",
                "aggregation": aggregation,
                "horizon_idx_rel": horizon,
                "diff_idx_rel": diff,
                "n_rollouts": metrics["n_rollouts"],
                "feature_dim": metrics["feature_dim"],
                "n_success": counts.get("0", 0),
                "n_failure": counts.get("1", 0),
                "raw_silhouette": metrics["raw_silhouette"],
                "whitened_pca_silhouette": metrics["whitened_pca_silhouette"],
                "whitened_centroid_distance": metrics["whitened_centroid_distance"],
                "fisher_lda_silhouette": metrics["fisher_lda_silhouette"],
                "fisher_lda_effect_size": metrics["fisher_lda_effect_size"],
                "whitened_centroid_cv_balanced_accuracy": metrics[
                    "whitened_centroid_cv_balanced_accuracy"
                ],
            }
            result_rows.append(flat)
            json_rows.append({**flat, "failure_counts": counts})
            json_rows[-1]["plots"] = plot_paths
            print(
                f"{flat['scope']}\t{flat['task']}\t{aggregation}\t"
                f"white_sil={flat['whitened_pca_silhouette']:.4f}\t"
                f"white_cv_bal_acc={flat['whitened_centroid_cv_balanced_accuracy']:.4f}"
            )

    write_tsv(args.out_dir / "rollout_mean_aggregation_separability.tsv", result_rows)
    write_summary(args.out_dir / "rollout_mean_aggregation_separability.md", result_rows)
    with (args.out_dir / "rollout_mean_aggregation_separability.json").open("w") as f:
        json.dump(json_rows, f, indent=2, sort_keys=True)
        f.write("\n")
    print(args.out_dir)


if __name__ == "__main__":
    main()
