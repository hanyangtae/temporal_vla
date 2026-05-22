#!/usr/bin/env python3
"""Centroid distances + silhouette scores under Euclidean and Mahalanobis."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import cholesky, solve_triangular
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
    / "cluster_analysis_mean_mean"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["all", "train", "val_seen", "val_unseen"],
        choices=("all", "train", "val_seen", "val_unseen"),
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--cov-reg",
        type=float,
        default=1e-3,
        help="Ridge added to the pooled covariance diagonal before inversion.",
    )
    return parser.parse_args()


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_manifest(split_root: Path, scope: str) -> list[dict[str, str]]:
    path = split_root / "manifest.tsv"
    with path.open("r", newline="") as f:
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
    for hidden in record["hidden_states"]:
        h = tensor_to_numpy(hidden).astype(np.float32, copy=False)
        if h.ndim != 3:
            raise ValueError(f"Expected [K, H, D], got {h.shape}")
        feats.append(h.mean(axis=(0, 1)))
    if not feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(feats, axis=0)


def load_scope(split_root: Path, scope: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, str]]:
    rows = load_manifest(split_root, scope)
    chunks: list[np.ndarray] = []
    success, task, task_failure, ep_idx = [], [], [], []
    task_id_to_name: dict[int, str] = {}

    for row in rows:
        path = pkl_path(split_root, row)
        with path.open("rb") as f:
            record = pickle.load(f)
        feats = pooled_hidden_states(record)
        n = feats.shape[0]
        if n == 0:
            continue
        tid = int(row["task_id"])
        succ = int(row["success"])
        fail = 1 - succ
        task_id_to_name[tid] = row["task"]
        chunks.append(feats)
        success.append(np.full(n, succ, dtype=np.int64))
        task.append(np.full(n, tid, dtype=np.int64))
        task_failure.append(np.full(n, tid * 2 + fail, dtype=np.int64))
        ep_idx.append(np.full(n, int(row["episode_idx"]), dtype=np.int64))

    if not chunks:
        raise ValueError(f"No features for scope: {scope}")

    labels = {
        "success": np.concatenate(success),
        "failure": 1 - np.concatenate(success),
        "task": np.concatenate(task),
        "task_failure": np.concatenate(task_failure),
        "episode_idx": np.concatenate(ep_idx),
    }
    return np.concatenate(chunks, axis=0), labels, task_id_to_name


def pooled_within_cov(X: np.ndarray, labels: np.ndarray, reg: float) -> np.ndarray:
    X64 = X.astype(np.float64, copy=False)
    D = X64.shape[1]
    scatter = np.zeros((D, D), dtype=np.float64)
    dof = 0
    for c in np.unique(labels):
        Xc = X64[labels == c]
        if len(Xc) < 2:
            continue
        Xc_centered = Xc - Xc.mean(axis=0, keepdims=True)
        scatter += Xc_centered.T @ Xc_centered
        dof += len(Xc) - 1
    cov = scatter / max(dof, 1)
    cov += reg * np.trace(cov) / D * np.eye(D)
    return cov


def whiten(X: np.ndarray, cov: np.ndarray) -> np.ndarray:
    L = cholesky(cov, lower=True)
    return solve_triangular(L, X.astype(np.float64).T, lower=True).T.astype(np.float32)


def centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(labels)
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    return unique, cents


def pairwise_dist(C: np.ndarray) -> np.ndarray:
    diff = C[:, None, :] - C[None, :, :]
    return np.linalg.norm(diff.astype(np.float64), axis=-1)


def silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    info: dict[str, Any] = {
        "n_points": int(X.shape[0]),
        "n_clusters": int(len(unique)),
        "cluster_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
    }
    if len(unique) < 2 or counts.min() < 2:
        info["score"] = None
        info["reason"] = "need ≥2 clusters with ≥2 points each"
        return info
    ss = sample_size if X.shape[0] > sample_size else None
    info["sample_size"] = int(ss or X.shape[0])
    info["score"] = float(
        silhouette_score(X, labels, metric="euclidean", sample_size=ss, random_state=random_state)
    )
    return info


def label_key_pairs(label_kind: str, unique: np.ndarray, task_id_to_name: dict[int, str]) -> list[str]:
    if label_kind == "success_failure":
        return ["failure" if int(k) == 0 else "success" for k in unique]
    if label_kind == "task":
        return [task_id_to_name.get(int(k), str(int(k))) for k in unique]
    if label_kind == "task_failure":
        names = []
        for k in unique:
            tid = int(k) // 2
            fail = int(k) % 2
            names.append(f"{task_id_to_name.get(tid, str(tid))}|{'failure' if fail else 'success'}")
        return names
    return [str(int(k)) for k in unique]


LABEL_KINDS = ("success_failure", "task", "task_failure")


def analyse(
    X: np.ndarray,
    labels: dict[str, np.ndarray],
    task_id_to_name: dict[int, str],
    *,
    sample_size: int,
    random_state: int,
    cov_reg: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {"feature_shape": list(X.shape), "by_label_kind": {}}
    label_map = {
        "success_failure": labels["success"],
        "task": labels["task"],
        "task_failure": labels["task_failure"],
    }
    for kind in LABEL_KINDS:
        y = label_map[kind]
        cov = pooled_within_cov(X, y, reg=cov_reg)
        Xw = whiten(X, cov)
        eu_unique, eu_cents = centroids(X, y)
        mh_unique, mh_cents = centroids(Xw, y)
        names = label_key_pairs(kind, eu_unique, task_id_to_name)
        out["by_label_kind"][kind] = {
            "labels_order": [int(k) for k in eu_unique],
            "label_names": names,
            "euclidean": {
                "centroid_pairwise_distance": pairwise_dist(eu_cents).tolist(),
                "silhouette": silhouette(
                    X, y, sample_size=sample_size, random_state=random_state
                ),
            },
            "mahalanobis": {
                "cov_regularization": cov_reg,
                "centroid_pairwise_distance": pairwise_dist(mh_cents).tolist(),
                "silhouette": silhouette(
                    Xw, y, sample_size=sample_size, random_state=random_state
                ),
            },
        }
    return out


def flatten(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, payload in results.items():
        for kind, kind_data in payload["by_label_kind"].items():
            names = kind_data["label_names"]
            for metric in ("euclidean", "mahalanobis"):
                m = kind_data[metric]
                sil = m["silhouette"]
                rows.append(
                    {
                        "scope": scope,
                        "label_kind": kind,
                        "metric": metric,
                        "row_kind": "silhouette",
                        "i": "", "j": "",
                        "name_i": "", "name_j": "",
                        "value": sil.get("score"),
                        "n_points": sil.get("n_points"),
                        "n_clusters": sil.get("n_clusters"),
                        "sample_size": sil.get("sample_size"),
                    }
                )
                dist = np.asarray(m["centroid_pairwise_distance"])
                n = dist.shape[0]
                for i in range(n):
                    for j in range(i + 1, n):
                        rows.append(
                            {
                                "scope": scope,
                                "label_kind": kind,
                                "metric": metric,
                                "row_kind": "centroid_distance",
                                "i": i, "j": j,
                                "name_i": names[i], "name_j": names[j],
                                "value": float(dist[i, j]),
                                "n_points": "", "n_clusters": "", "sample_size": "",
                            }
                        )
                for i in range(n):
                    rows.append(
                        {
                            "scope": scope,
                            "label_kind": kind,
                            "metric": metric,
                            "row_kind": "centroid_mean_dist_to_others",
                            "i": i, "j": "",
                            "name_i": names[i], "name_j": "",
                            "value": float(dist[i].sum() / max(n - 1, 1)),
                            "n_points": "", "n_clusters": "", "sample_size": "",
                        }
                    )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scope", "label_kind", "metric", "row_kind",
        "i", "j", "name_i", "name_j",
        "value", "n_points", "n_clusters", "sample_size",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full: dict[str, Any] = {}
    for scope in args.scopes:
        X, labels, task_id_to_name = load_scope(args.split_root, scope)
        full[scope] = {
            "task_id_to_name": {str(k): v for k, v in sorted(task_id_to_name.items())},
            **analyse(
                X,
                labels,
                task_id_to_name,
                sample_size=args.sample_size,
                random_state=args.random_state,
                cov_reg=args.cov_reg,
            ),
        }
        print(f"[{scope}] X={X.shape} done")

    with (args.out_dir / "cluster_analysis.json").open("w") as f:
        json.dump(full, f, indent=2, sort_keys=True)
        f.write("\n")
    write_tsv(args.out_dir / "cluster_analysis.tsv", flatten(full))
    print(args.out_dir)


if __name__ == "__main__":
    main()
