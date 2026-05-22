#!/usr/bin/env python3
"""Cosine-based cluster analysis + visualization (magnitude-invariant).

Steps:
  - L2-normalize each feature row.
  - Centroid per cluster = mean direction (mean of unit vectors, then renormalized).
  - Centroid pairwise distance = cosine distance.
  - Silhouette via sklearn metric='cosine'.
  - Visualization: LDA 2D on normalized features, t-SNE (metric='cosine'),
    per-point a-vs-b cosine-distance scatter.
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
    / "cluster_cosine_mean_mean"
)


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
    p.add_argument("--plot-scopes", nargs="+", default=["all", "val_unseen"])
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--tsne-sample", type=int, default=5000)
    p.add_argument("--random-state", type=int, default=0)
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


def load_scope(split_root: Path, scope: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, str]]:
    rows = load_manifest(split_root, scope)
    Xs, succ, task, tf = [], [], [], []
    task_id_to_name: dict[int, str] = {}
    for r in rows:
        with pkl_path(split_root, r).open("rb") as f:
            rec = pickle.load(f)
        f_arr = pooled_hidden_states(rec)
        n = f_arr.shape[0]
        if n == 0:
            continue
        tid = int(r["task_id"])
        s = int(r["success"])
        task_id_to_name[tid] = r["task"]
        Xs.append(f_arr)
        succ.append(np.full(n, s, dtype=np.int64))
        task.append(np.full(n, tid, dtype=np.int64))
        tf.append(np.full(n, tid * 2 + (1 - s), dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    labels = {
        "success": np.concatenate(succ),
        "task": np.concatenate(task),
        "task_failure": np.concatenate(tf),
    }
    return X, labels, task_id_to_name


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_centroids(Xn: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.sort(np.unique(labels))
    cents = []
    for c in unique:
        m = Xn[labels == c].mean(axis=0)
        cents.append(m / max(np.linalg.norm(m), 1e-12))
    return unique, np.stack(cents, axis=0)


def cosine_pairwise(C: np.ndarray) -> np.ndarray:
    sim = C @ C.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def silhouette_cosine(X: np.ndarray, labels: np.ndarray, sample_size: int, random_state: int) -> dict[str, Any]:
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
        silhouette_score(X, labels, metric="cosine", sample_size=ss, random_state=random_state)
    )
    return info


def cosine_a_b(Xn: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, cents = cosine_centroids(Xn, labels)
    own_idx = np.searchsorted(unique, labels)
    sim = Xn @ cents.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    own = dist[np.arange(len(labels)), own_idx]
    others = dist.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    near = others.min(axis=1)
    return own, near


def label_names(label_kind: str, unique: np.ndarray, task_id_to_name: dict[int, str]) -> list[str]:
    if label_kind == "success_failure":
        return ["failure" if int(k) == 0 else "success" for k in unique]
    if label_kind == "task":
        return [task_id_to_name.get(int(k), str(int(k))) for k in unique]
    if label_kind == "task_failure":
        out = []
        for k in unique:
            tid = int(k) // 2
            fail = int(k) % 2
            out.append(f"{task_id_to_name.get(tid, str(tid))}|{'failure' if fail else 'success'}")
        return out
    return [str(int(k)) for k in unique]


LABEL_KINDS = ("success_failure", "task", "task_failure")


def analyse_scope(
    X: np.ndarray,
    labels: dict[str, np.ndarray],
    task_id_to_name: dict[int, str],
    *,
    sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    Xn = l2_normalize(X)
    label_map = {
        "success_failure": labels["success"],
        "task": labels["task"],
        "task_failure": labels["task_failure"],
    }
    out: dict[str, Any] = {"feature_shape": list(X.shape), "by_label_kind": {}}
    for kind in LABEL_KINDS:
        y = label_map[kind]
        unique, cents = cosine_centroids(Xn, y)
        names = label_names(kind, unique, task_id_to_name)
        out["by_label_kind"][kind] = {
            "labels_order": [int(k) for k in unique],
            "label_names": names,
            "centroid_pairwise_distance": cosine_pairwise(cents).tolist(),
            "silhouette": silhouette_cosine(
                Xn, y, sample_size=sample_size, random_state=random_state
            ),
        }
    return out


def flatten(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, payload in results.items():
        for kind, data in payload["by_label_kind"].items():
            names = data["label_names"]
            sil = data["silhouette"]
            rows.append(
                {
                    "scope": scope, "label_kind": kind, "metric": "cosine",
                    "row_kind": "silhouette",
                    "i": "", "j": "", "name_i": "", "name_j": "",
                    "value": sil.get("score"),
                    "n_points": sil.get("n_points"),
                    "n_clusters": sil.get("n_clusters"),
                    "sample_size": sil.get("sample_size"),
                }
            )
            dist = np.asarray(data["centroid_pairwise_distance"])
            n = dist.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    rows.append(
                        {
                            "scope": scope, "label_kind": kind, "metric": "cosine",
                            "row_kind": "centroid_distance",
                            "i": i, "j": j, "name_i": names[i], "name_j": names[j],
                            "value": float(dist[i, j]),
                            "n_points": "", "n_clusters": "", "sample_size": "",
                        }
                    )
            for i in range(n):
                rows.append(
                    {
                        "scope": scope, "label_kind": kind, "metric": "cosine",
                        "row_kind": "centroid_mean_dist_to_others",
                        "i": i, "j": "", "name_i": names[i], "name_j": "",
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


def subsample(*arrays, n: int, rng: np.random.Generator):
    total = arrays[0].shape[0]
    if total <= n:
        return arrays
    idx = rng.choice(total, size=n, replace=False)
    return tuple(a[idx] for a in arrays)


def lda_2d(Xn: np.ndarray, labels: np.ndarray, random_state: int) -> np.ndarray:
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    if n_comp <= 0:
        raise ValueError("LDA requires ≥2 classes")
    Z = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(Xn.astype(np.float64), labels)
    if Z.shape[1] == 1:
        rng = np.random.default_rng(random_state)
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
    scope_name: str,
) -> None:
    Xn = l2_normalize(X)
    Z_lda = lda_2d(Xn, tasks, random_state=random_state)

    rng = np.random.default_rng(random_state)
    Xn_s, tasks_s = subsample(Xn, tasks, n=tsne_sample, rng=rng)
    print(f"  [{scope_name}] running cosine t-SNE on {Xn_s.shape}...")
    Z_tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="cosine",
        random_state=random_state,
        init="random",
        learning_rate="auto",
    ).fit_transform(Xn_s)

    own, other = cosine_a_b(Xn, tasks)
    unique_tids = np.sort(np.unique(tasks))
    colors = task_colormap(unique_tids)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    scatter_with_centroids(axes[0], Z_lda, tasks, task_id_to_name, colors, f"{scope_name} | LDA 2D on L2-normalized (task)")
    scatter_with_centroids(axes[1], Z_tsne, tasks_s, task_id_to_name, colors, f"{scope_name} | t-SNE (metric=cosine)")

    ax3 = axes[2]
    for tid in unique_tids:
        m = tasks == tid
        ax3.scatter(own[m], other[m], s=4, c=[colors[int(tid)]], alpha=0.35, label=task_id_to_name[int(tid)], linewidths=0)
    lo = float(min(own.min(), other.min()))
    hi = float(max(own.max(), other.max()))
    ax3.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax3.set_xlabel("a: cosine dist to own task centroid")
    ax3.set_ylabel("b: cosine dist to nearest other task centroid")
    sil_proxy = float(np.mean((other - own) / np.maximum(own, other)))
    ax3.set_title(f"{scope_name} | per-point a vs b (cosine)\nmean (b−a)/max(a,b) = {sil_proxy:+.4f}")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.85)

    fig.suptitle(f"GR00T-N1.6 RoboCasa | cosine task separation | scope={scope_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    plot_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[int, str]]] = {}
    for scope in args.scopes:
        print(f"[{scope}] loading...")
        X, labels, task_id_to_name = load_scope(args.split_root, scope)
        print(f"  X={X.shape}")
        results[scope] = {
            "task_id_to_name": {str(k): v for k, v in sorted(task_id_to_name.items())},
            **analyse_scope(
                X,
                labels,
                task_id_to_name,
                sample_size=args.sample_size,
                random_state=args.random_state,
            ),
        }
        if scope in args.plot_scopes:
            plot_cache[scope] = (X, labels["task"], task_id_to_name)

    with (args.out_dir / "cluster_cosine.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    write_tsv(args.out_dir / "cluster_cosine.tsv", flatten(results))

    for scope, (X, tasks, task_id_to_name) in plot_cache.items():
        plot_scope(
            X, tasks, task_id_to_name,
            out_path=args.out_dir / f"cosine_separation_{scope}.png",
            tsne_sample=args.tsne_sample,
            random_state=args.random_state,
            scope_name=scope,
        )
    print(args.out_dir)


if __name__ == "__main__":
    main()
