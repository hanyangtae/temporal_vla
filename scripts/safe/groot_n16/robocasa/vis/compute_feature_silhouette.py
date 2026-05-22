#!/usr/bin/env python3
"""Compute silhouette scores for GR00T N1.6 SAFE rollout features."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_score


ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from run_config import (  # noqa: E402
    EXPERIMENT_FEATURE_SPACE_ROOT,
    FINAL_AGGREGATION_SLUG,
    FINAL_DIFF_IDX_REL,
    FINAL_HORIZON_IDX_REL,
    SPLIT_ROOT,
)
from safe_feature_vectors import (  # noqa: E402
    aggregation_slug,
    aggregation_space,
    load_manifest,
    load_scope_features,
    parse_aggregation_command,
)


DEFAULT_SPLIT_ROOT = SPLIT_ROOT
DEFAULT_OUT_DIR = EXPERIMENT_FEATURE_SPACE_ROOT / f"silhouette_{FINAL_AGGREGATION_SLUG}"
DEFAULT_VIS_ROOT = EXPERIMENT_FEATURE_SPACE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute success/failure, task, and task+failure silhouette scores."
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vis-root", type=Path, default=DEFAULT_VIS_ROOT)
    parser.add_argument("--horizon-idx-rel", default=FINAL_HORIZON_IDX_REL)
    parser.add_argument("--diff-idx-rel", default=FINAL_DIFF_IDX_REL)
    parser.add_argument("--projector", default="tsne")
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["all", "train", "val_seen", "val_unseen"],
        choices=("all", "train", "val_seen", "val_unseen"),
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument(
        "--covariance-sample-size",
        type=int,
        default=10000,
        help="Maximum number of points used to fit the shrinkage covariance for Mahalanobis distance.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def score_labels(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    sample_size: int,
    random_state: int,
    result_metric: str | None = None,
) -> dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    result: dict[str, Any] = {
        "n_points": int(features.shape[0]),
        "n_clusters": int(len(unique)),
        "cluster_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "metric": result_metric or metric,
    }
    if len(unique) < 2 or np.min(counts) < 2:
        result["score"] = None
        result["reason"] = "need at least two clusters with at least two points each"
        return result

    actual_sample_size = sample_size if features.shape[0] > sample_size else None
    result["sample_size"] = int(actual_sample_size or features.shape[0])
    result["score"] = float(
        silhouette_score(
            features,
            labels,
            metric=metric,
            sample_size=actual_sample_size,
            random_state=random_state,
        )
    )
    return result


def per_task_success_scores(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    metric: str,
    sample_size: int,
    random_state: int,
    result_metric: str | None = None,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for task_id in sorted(np.unique(labels["task"])):
        mask = labels["task"] == task_id
        task_names = labels["task_name"][mask]
        task_name = str(task_names[0]) if len(task_names) else str(task_id)
        scores[f"{int(task_id)}:{task_name}"] = score_labels(
            features[mask],
            labels["success"][mask],
            metric=metric,
            sample_size=sample_size,
            random_state=random_state,
            result_metric=result_metric,
        )
    return scores


def sample_rows(features: np.ndarray, max_rows: int, random_state: int) -> np.ndarray:
    if features.shape[0] <= max_rows:
        return features
    rng = np.random.default_rng(random_state)
    indices = rng.choice(features.shape[0], size=max_rows, replace=False)
    return features[indices]


def mahalanobis_whiten_features(
    features: np.ndarray,
    *,
    covariance_sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Whiten features so Euclidean distance equals shrinkage Mahalanobis distance."""

    fit_features = sample_rows(features, covariance_sample_size, random_state).astype(np.float64)
    estimator = LedoitWolf().fit(fit_features)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    precision = (precision + precision.T) * 0.5

    try:
        transform = np.linalg.cholesky(precision)
        decomposition = "cholesky"
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(precision)
        eigvals = np.clip(eigvals, 1e-12, None)
        transform = eigvecs @ np.diag(np.sqrt(eigvals))
        decomposition = "eigendecomposition"

    centered = features.astype(np.float64, copy=False) - estimator.location_
    whitened = centered @ transform
    metadata = {
        "covariance_estimator": "LedoitWolf",
        "covariance_fit_points": int(fit_features.shape[0]),
        "covariance_sample_size": int(covariance_sample_size),
        "decomposition": decomposition,
        "shrinkage": float(estimator.shrinkage_),
    }
    return whitened.astype(np.float32), metadata


def score_suite(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    metric: str,
    sample_size: int,
    random_state: int,
    result_metric: str | None = None,
) -> dict[str, Any]:
    return {
        "success_failure": score_labels(
            features,
            labels["success"],
            metric=metric,
            sample_size=sample_size,
            random_state=random_state,
            result_metric=result_metric,
        ),
        "task": score_labels(
            features,
            labels["task"],
            metric=metric,
            sample_size=sample_size,
            random_state=random_state,
            result_metric=result_metric,
        ),
        "task_plus_failure": score_labels(
            features,
            labels["task_failure"],
            metric=metric,
            sample_size=sample_size,
            random_state=random_state,
            result_metric=result_metric,
        ),
        "success_failure_within_each_task": per_task_success_scores(
            features,
            labels,
            metric=metric,
            sample_size=sample_size,
            random_state=random_state,
            result_metric=result_metric,
        ),
    }


def compute_original_scope(
    split_root: Path,
    scope: str,
    *,
    horizon_idx_rel: str,
    diff_idx_rel: str,
    sample_size: int,
    covariance_sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    features, labels = load_scope_features(
        split_root,
        scope,
        horizon_idx_rel=parse_aggregation_command(horizon_idx_rel),
        diff_idx_rel=parse_aggregation_command(diff_idx_rel),
    )
    metrics: dict[str, Any] = {
        "scope": scope,
        "space": aggregation_space(horizon_idx_rel, diff_idx_rel, features.shape[1]),
        "horizon_idx_rel": horizon_idx_rel,
        "diff_idx_rel": diff_idx_rel,
        "feature_shape": list(features.shape),
        "scores": {},
    }
    for metric in ("euclidean", "cosine"):
        metrics["scores"][metric] = score_suite(
            features,
            labels,
            metric=metric,
            sample_size=sample_size,
            random_state=random_state,
        )

    whitened, transform_metadata = mahalanobis_whiten_features(
        features,
        covariance_sample_size=covariance_sample_size,
        random_state=random_state,
    )
    metrics["mahalanobis_transform"] = transform_metadata
    metrics["scores"]["mahalanobis_shrinkage"] = score_suite(
        whitened,
        labels,
        metric="euclidean",
        sample_size=sample_size,
        random_state=random_state,
        result_metric="mahalanobis_shrinkage",
    )
    return metrics


def projection_scope_dir(
    vis_root: Path,
    scope: str,
    *,
    projector: str,
    horizon_idx_rel: str,
    diff_idx_rel: str,
) -> Path:
    return vis_root / scope / f"{projector}_{aggregation_slug(horizon_idx_rel, diff_idx_rel)}"


def load_projection_labels(split_root: Path, scope: str, rollout_indices: np.ndarray) -> dict[str, np.ndarray]:
    rows = load_manifest(split_root, scope)
    success_by_rollout = np.asarray([int(row["success"]) for row in rows], dtype=np.int64)
    task_by_rollout = np.asarray([int(row["task_id"]) for row in rows], dtype=np.int64)
    task_name_by_rollout = np.asarray([row["task"] for row in rows])
    rollout_indices = rollout_indices.astype(np.int64)
    success = success_by_rollout[rollout_indices]
    task = task_by_rollout[rollout_indices]
    failure = 1 - success
    return {
        "success": success,
        "failure": failure,
        "task": task,
        "task_failure": task * 2 + failure,
        "task_name": task_name_by_rollout[rollout_indices],
    }


def compute_projection_scope(
    split_root: Path,
    vis_root: Path,
    scope: str,
    *,
    projector: str,
    horizon_idx_rel: str,
    diff_idx_rel: str,
    sample_size: int,
    covariance_sample_size: int,
    random_state: int,
) -> dict[str, Any] | None:
    artifact_dir = projection_scope_dir(
        vis_root,
        scope,
        projector=projector,
        horizon_idx_rel=horizon_idx_rel,
        diff_idx_rel=diff_idx_rel,
    )
    projection_path = artifact_dir / "feats_projected_skip1.pkl"
    if not projection_path.exists():
        return None
    with projection_path.open("rb") as f:
        data = pickle.load(f)
    features = np.asarray(data["feats_projected"], dtype=np.float32)
    labels = load_projection_labels(split_root, scope, np.asarray(data["rollout_indices"]))
    metrics: dict[str, Any] = {
        "scope": scope,
        "space": f"{projector}_2d_{aggregation_slug(horizon_idx_rel, diff_idx_rel)}",
        "artifact_dir": str(artifact_dir),
        "horizon_idx_rel": horizon_idx_rel,
        "diff_idx_rel": diff_idx_rel,
        "feature_shape": list(features.shape),
        "scores": {},
    }
    metrics["scores"]["euclidean"] = score_suite(
        features,
        labels,
        metric="euclidean",
        sample_size=sample_size,
        random_state=random_state,
    )
    whitened, transform_metadata = mahalanobis_whiten_features(
        features,
        covariance_sample_size=covariance_sample_size,
        random_state=random_state,
    )
    metrics["mahalanobis_transform"] = transform_metadata
    metrics["scores"]["mahalanobis_shrinkage"] = score_suite(
        whitened,
        labels,
        metric="euclidean",
        sample_size=sample_size,
        random_state=random_state,
        result_metric="mahalanobis_shrinkage",
    )
    return metrics


def compact_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        for metric, scores in item["scores"].items():
            for label_kind in ("success_failure", "task", "task_plus_failure"):
                score_item = scores[label_kind]
                rows.append(
                    {
                        "scope": item["scope"],
                        "space": item["space"],
                        "metric": metric,
                        "label_kind": label_kind,
                        "score": score_item["score"],
                        "n_points": score_item["n_points"],
                        "n_clusters": score_item["n_clusters"],
                        "sample_size": score_item.get("sample_size"),
                    }
                )
            for task_name, score_item in scores["success_failure_within_each_task"].items():
                rows.append(
                    {
                        "scope": item["scope"],
                        "space": item["space"],
                        "metric": metric,
                        "label_kind": f"success_failure_within_task:{task_name}",
                        "score": score_item["score"],
                        "n_points": score_item["n_points"],
                        "n_clusters": score_item["n_clusters"],
                        "sample_size": score_item.get("sample_size"),
                    }
                )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["scope", "space", "metric", "label_kind", "score", "n_points", "n_clusters", "sample_size"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_key_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    key_rows = [
        row
        for row in rows
        if row["label_kind"] in {"success_failure", "task", "task_plus_failure"}
    ]
    write_tsv(path, key_rows)


def write_task_success_failure_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scope",
        "space",
        "metric",
        "task_id",
        "task",
        "score",
        "n_points",
        "n_clusters",
        "sample_size",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            prefix = "success_failure_within_task:"
            if not str(row["label_kind"]).startswith(prefix):
                continue
            task_spec = str(row["label_kind"])[len(prefix) :]
            task_id, task = task_spec.split(":", maxsplit=1)
            writer.writerow(
                {
                    "scope": row["scope"],
                    "space": row["space"],
                    "metric": row["metric"],
                    "task_id": task_id,
                    "task": task,
                    "score": row["score"],
                    "n_points": row["n_points"],
                    "n_clusters": row["n_clusters"],
                    "sample_size": row["sample_size"],
                }
            )


def lookup_score(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    space: str,
    metric: str,
    label_kind: str,
) -> float | None:
    for row in rows:
        if (
            row["scope"] == scope
            and row["space"] == space
            and row["metric"] == metric
            and row["label_kind"] == label_kind
        ):
            return row["score"]
    return None


def fmt_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    original_spaces = sorted({row["space"] for row in rows if str(row["space"]).startswith("original_")})
    projection_spaces = sorted({row["space"] for row in rows if not str(row["space"]).startswith("original_")})
    original_space = original_spaces[0] if original_spaces else "original"
    projection_space = projection_spaces[0] if projection_spaces else None
    lines = [
        "# Silhouette Score Summary",
        "",
        "This file is the human-readable summary of `silhouette_scores.tsv`.",
        "",
        "## How To Read",
        "",
        "- `score` is the silhouette score. It ranges from -1 to 1.",
        "- Around 0 means weak or no cluster separation.",
        "- Above 0.5 is usually a strong separation signal.",
        "- Negative values mean points may be closer to other clusters than to their assigned cluster.",
        f"- The most important rows are `{original_space}` with `mahalanobis_shrinkage`.",
        "",
        "Label meanings:",
        "",
        "- `success_failure`: every timestep is labeled by its rollout-level success or failure.",
        "- `task`: every timestep is labeled by task id.",
        "- `task_plus_failure`: task id and failure flag are combined into one label.",
        "- `success_failure_within_task:*`: success/failure silhouette computed inside one task only.",
        "",
        "## Key Scores",
        "",
        "| scope | space | metric | success/failure | task | task+failure |",
        "|---|---|---:|---:|---:|---:|",
    ]
    key_groups = [
        ("all", original_space, ("euclidean", "cosine", "mahalanobis_shrinkage")),
        ("val_unseen", original_space, ("euclidean", "cosine", "mahalanobis_shrinkage")),
    ]
    if projection_space is not None:
        key_groups.append(("val_unseen", projection_space, ("euclidean", "mahalanobis_shrinkage")))
    for scope, space, metrics in key_groups:
        for metric in metrics:
            lines.append(
                "| "
                + " | ".join(
                    [
                        scope,
                        space,
                        metric,
                        fmt_score(
                            lookup_score(
                                rows,
                                scope=scope,
                                space=space,
                                metric=metric,
                                label_kind="success_failure",
                            )
                        ),
                        fmt_score(
                            lookup_score(
                                rows,
                                scope=scope,
                                space=space,
                                metric=metric,
                                label_kind="task",
                            )
                        ),
                        fmt_score(
                            lookup_score(
                                rows,
                                scope=scope,
                                space=space,
                                metric=metric,
                                label_kind="task_plus_failure",
                            )
                        ),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The Mahalanobis score on original SAFE feature vectors is still near zero for success/failure.",
            "- `task_plus_failure` is negative in the important settings.",
            "- Therefore the latent space does not show a clear static success/failure cluster.",
            "- Detector quality should be interpreted with trajectory score dynamics and CP threshold crossing, not static cluster separation alone.",
            "",
            "## Files",
            "",
            "- Full machine-readable table: `silhouette_scores.tsv`",
            "- Full JSON with covariance metadata: `silhouette_scores.json`",
            "- Compact key table: `silhouette_key_scores.tsv`",
            "- Per-task success/failure table: `silhouette_task_success_failure.tsv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for scope in args.scopes:
        results.append(
            compute_original_scope(
                args.split_root,
                scope,
                horizon_idx_rel=args.horizon_idx_rel,
                diff_idx_rel=args.diff_idx_rel,
                sample_size=args.sample_size,
                covariance_sample_size=args.covariance_sample_size,
                random_state=args.random_state,
            )
        )
        projection = compute_projection_scope(
            args.split_root,
            args.vis_root,
            scope,
            projector=args.projector,
            horizon_idx_rel=args.horizon_idx_rel,
            diff_idx_rel=args.diff_idx_rel,
            sample_size=args.sample_size,
            covariance_sample_size=args.covariance_sample_size,
            random_state=args.random_state,
        )
        if projection is not None:
            results.append(projection)

    with (args.out_dir / "silhouette_scores.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    rows = compact_rows(results)
    write_tsv(args.out_dir / "silhouette_scores.tsv", rows)
    write_key_tsv(args.out_dir / "silhouette_key_scores.tsv", rows)
    write_task_success_failure_tsv(args.out_dir / "silhouette_task_success_failure.tsv", rows)
    write_markdown_summary(args.out_dir / "silhouette_summary.md", rows)
    print(args.out_dir)


if __name__ == "__main__":
    main()
