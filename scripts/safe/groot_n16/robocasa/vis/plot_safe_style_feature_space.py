#!/usr/bin/env python3
"""Plot GR00T N1.6 rollout feature space with SAFE's original plotting scheme."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding

from compute_feature_silhouette import (
    load_manifest,
    parse_aggregation_command,
    pooled_hidden_states,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ID = "seen4_unseen2_openDrawer_pnpCab_100ep"
DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/split"
)
DEFAULT_OUT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/feature_space"
    / EXPERIMENT_ID
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror SAFE/scripts/visualize_features.py for GR00T N1.6 features."
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--scope",
        default="all",
        choices=("all", "train", "val_seen", "val_unseen"),
    )
    parser.add_argument("--task", help="Optional exact task name filter, e.g. OpenDrawer.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional exact task name filters for a family-level plot.",
    )
    parser.add_argument("--task-id", type=int, help="Optional exact task id filter.")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        type=int,
        help="Optional exact task id filters for a family-level plot.",
    )
    parser.add_argument("--name", help="Optional suffix for the output directory.")
    parser.add_argument("--horizon-idx-rel", default="mean")
    parser.add_argument("--diff-idx-rel", default="mean")
    parser.add_argument(
        "--projector",
        default="tsne",
        choices=("pca", "tsne", "umap", "isomap", "mds", "spectral"),
    )
    parser.add_argument("--custom-cmap", action="store_true")
    parser.add_argument("--feat-skip", type=int, default=1)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Use 0 to match SAFE exactly and project every timestep feature.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    return value.replace(".", "p").replace("-", "_").replace(":", "_")


def name_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def get_projector(name: str) -> Any:
    if name == "pca":
        return PCA(n_components=2)
    if name == "tsne":
        return TSNE(n_components=2)
    if name == "umap":
        import umap

        return umap.UMAP(n_components=2)
    if name == "isomap":
        return Isomap(n_components=2)
    if name == "mds":
        return MDS(n_components=2)
    if name == "spectral":
        return SpectralEmbedding(n_components=2)
    raise ValueError(f"Unknown projector {name}")


def sample_if_needed(
    feats: np.ndarray,
    labels: np.ndarray,
    rollout_indices: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_points <= 0 or len(feats) <= max_points:
        return feats, labels, rollout_indices
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(feats), size=max_points, replace=False))
    return feats[indices], labels[indices], rollout_indices[indices]


def task_cmap(custom_cmap: bool) -> mpl.colors.Colormap:
    if not custom_cmap:
        return mpl.cm.get_cmap("tab10", 10)

    custom_colors = [
        "#98df8a",
        "#c5b0d5",
        "#8c564b",
        "#ff7f0e",
        "#9467bd",
        "#bcbd22",
        "#7f7f7f",
        "#e377c2",
        "#2ca02c",
        "#c49c94",
    ]
    return mpl.colors.ListedColormap(custom_colors)


def main() -> None:
    args = parse_args()
    horizon_command = parse_aggregation_command(args.horizon_idx_rel)
    diff_command = parse_aggregation_command(args.diff_idx_rel)

    rollouts = load_manifest(args.split_root, args.scope)
    if args.task is not None:
        rollouts = [row for row in rollouts if row["task"] == args.task]
    if args.tasks is not None:
        tasks = set(args.tasks)
        rollouts = [row for row in rollouts if row["task"] in tasks]
    if args.task_id is not None:
        rollouts = [row for row in rollouts if int(row["task_id"]) == args.task_id]
    if args.task_ids is not None:
        task_ids = set(args.task_ids)
        rollouts = [row for row in rollouts if int(row["task_id"]) in task_ids]
    if not rollouts:
        raise ValueError(
            f"No rollouts matched scope={args.scope}, task={args.task}, task_id={args.task_id}"
        )
    print(f"Loaded {len(rollouts)} rollouts")

    feats = []
    labels = []
    rollout_indices = []
    for i, row in enumerate(rollouts):
        with Path(row["source_path"]).open("rb") as f:
            record = pickle.load(f)
        feat = pooled_hidden_states(
            record,
            horizon_idx_rel=horizon_command,
            diff_idx_rel=diff_command,
        )
        feat = feat[:: args.feat_skip]
        feats.append(feat)
        labels.append(np.ones(feat.shape[0]) * (1 - int(row["success"])))
        rollout_indices.append(np.ones(feat.shape[0]) * i)

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    rollout_indices = np.concatenate(rollout_indices, axis=0)
    feats, labels, rollout_indices = sample_if_needed(
        feats,
        labels,
        rollout_indices,
        max_points=args.max_points,
        seed=args.seed,
    )

    print(f"feats: {feats.shape} {feats.dtype}")
    print(f"labels: {labels.shape} {labels.dtype}")
    print(f"rollout_indices: {rollout_indices.shape} {rollout_indices.dtype}")

    folder_name = (
        f"{args.scope}_h{slug(args.horizon_idx_rel)}_d{slug(args.diff_idx_rel)}"
        f"-{args.projector}"
    )
    if args.task is not None:
        folder_name += f"-task_{name_slug(args.task)}"
    if args.tasks is not None:
        folder_name += "-tasks_" + "_".join(name_slug(task) for task in args.tasks)
    if args.task_id is not None:
        folder_name += f"-taskid_{args.task_id}"
    if args.task_ids is not None:
        folder_name += "-taskids_" + "_".join(str(task_id) for task_id in args.task_ids)
    if args.name is not None:
        folder_name += f"-{name_slug(args.name)}"
    if args.max_points > 0:
        folder_name += f"-sample{args.max_points}"
    save_folder = args.out_root / "safe_style_visualize_features" / folder_name
    save_folder.mkdir(parents=True, exist_ok=True)
    save_pkl_path = save_folder / f"feats_projected_skip{args.feat_skip}.pkl"

    if save_pkl_path.exists() and not args.force:
        with save_pkl_path.open("rb") as f:
            data = pickle.load(f)
        projector = data["projector"]
        feats_projected = data["feats_projected"]
        labels = data["labels"]
        rollout_indices = data["rollout_indices"]
    else:
        projector = get_projector(args.projector)
        feats_projected = projector.fit_transform(feats)
        print(f"feats_projected: {feats_projected.shape} {feats_projected.dtype}")
        with save_pkl_path.open("wb") as f:
            pickle.dump(
                {
                    "projector": projector,
                    "feats_projected": feats_projected,
                    "labels": labels,
                    "rollout_indices": rollout_indices,
                },
                f,
            )

    task_ids = []
    colors_by_success = []
    for i, row in enumerate(rollouts):
        feat_proj = feats_projected[rollout_indices == i]
        task_ids.append(np.ones(feat_proj.shape[0]) * int(row["task_id"]))
        if int(row["success"]) == 0:
            colors_by_success.append(np.linspace(0, 1, feat_proj.shape[0]))
        else:
            colors_by_success.append(np.zeros(feat_proj.shape[0]))
    colors_by_success = np.concatenate(colors_by_success, axis=0)
    task_ids = np.concatenate(task_ids, axis=0)

    plt.figure(dpi=200)
    plt.scatter(
        feats_projected[:, 0],
        feats_projected[:, 1],
        c=colors_by_success,
        cmap="coolwarm",
        s=0.5,
        alpha=0.5,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    succ_path = save_folder / f"feats_vis_skip{args.feat_skip}-succ.png"
    plt.savefig(succ_path, bbox_inches="tight")
    plt.close()

    plt.figure(dpi=200)
    plt.scatter(
        feats_projected[:, 0],
        feats_projected[:, 1],
        c=task_ids,
        cmap=task_cmap(args.custom_cmap),
        s=0.5,
        alpha=0.7,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    task_path = save_folder / f"feats_vis_skip{args.feat_skip}-taskid.png"
    plt.savefig(task_path, bbox_inches="tight")
    plt.close()

    print(succ_path)
    print(task_path)


if __name__ == "__main__":
    main()
