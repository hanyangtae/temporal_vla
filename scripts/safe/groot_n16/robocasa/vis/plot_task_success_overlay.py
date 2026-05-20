#!/usr/bin/env python3
"""Overlay task identity and rollout success on an existing SAFE feature plot.

This is a post-processing utility for artifacts created by
run_feature_visualization.py. It reuses the saved 2D projection and reconstructs
rollout metadata from the source split directory recorded in manifest.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROLLOUT_RE = re.compile(r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$")


@dataclass(frozen=True)
class RolloutMeta:
    path: Path
    task_name: str
    task_id: int
    episode_idx: int
    success: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw task-success overlays for an existing GR00T N1.6 SAFE t-SNE/PCA artifact."
    )
    parser.add_argument(
        "artifact_dir",
        type=Path,
        help="Directory containing manifest.json and feats_projected_skip1.pkl.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Scatter alpha.",
    )
    return parser.parse_args()


def load_projection(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pkl_path = artifact_dir / "feats_projected_skip1.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing projection file: {pkl_path}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    feats_projected = np.asarray(data["feats_projected"])
    rollout_indices = np.asarray(data["rollout_indices"], dtype=int)
    if feats_projected.ndim != 2 or feats_projected.shape[1] != 2:
        raise ValueError(f"Expected projected features shaped [N, 2], got {feats_projected.shape}")
    if len(feats_projected) != len(rollout_indices):
        raise ValueError(
            f"Projection/rollout index length mismatch: {len(feats_projected)} vs {len(rollout_indices)}"
        )
    return feats_projected, rollout_indices


def load_manifest(artifact_dir: Path) -> dict:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r") as f:
        return json.load(f)


def load_rollout_meta(source_data_path: Path) -> list[RolloutMeta]:
    paths = sorted(source_data_path.glob("**/*.pkl"))
    metas = []
    for path in paths:
        match = ROLLOUT_RE.match(path.name)
        if match is None:
            continue
        metas.append(
            RolloutMeta(
                path=path,
                task_name=path.parent.name,
                task_id=int(match.group("task_id")),
                episode_idx=int(match.group("episode_idx")),
                success=int(match.group("success")),
            )
        )

    # Match SAFE visualize_features.py ordering after load_rollouts().
    metas.sort(key=lambda item: (item.task_id, item.episode_idx))
    return metas


def build_point_meta(
    metas: list[RolloutMeta],
    rollout_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_indices = np.unique(rollout_indices)
    if len(unique_indices) != len(metas):
        raise ValueError(
            f"Rollout count mismatch: projection has {len(unique_indices)} rollout indices, "
            f"source has {len(metas)} pkl files"
        )

    task_ids = np.empty_like(rollout_indices, dtype=int)
    success = np.empty_like(rollout_indices, dtype=int)
    time_rel = np.zeros_like(rollout_indices, dtype=float)
    for idx in unique_indices:
        mask = rollout_indices == idx
        meta = metas[int(idx)]
        task_ids[mask] = meta.task_id
        success[mask] = meta.success
        n = int(mask.sum())
        time_rel[mask] = 0.0 if n <= 1 else np.linspace(0.0, 1.0, n)
    return task_ids, success, time_rel


def axis_limits(feats: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, y_min = feats.min(axis=0)
    x_max, y_max = feats.max(axis=0)
    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.04
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_taskid_failure_overlay(
    artifact_dir: Path,
    feats: np.ndarray,
    task_ids: np.ndarray,
    success: np.ndarray,
    metas: list[RolloutMeta],
    point_size: float,
    alpha: float,
) -> Path:
    task_order = sorted({meta.task_id for meta in metas})
    task_names = {meta.task_id: meta.task_name for meta in metas}
    cmap = plt.get_cmap("tab10", max(10, len(task_order)))
    color_by_task = {task_id: cmap(i) for i, task_id in enumerate(task_order)}

    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)
    for task_id in task_order:
        mask = task_ids == task_id
        ax.scatter(
            feats[mask, 0],
            feats[mask, 1],
            s=point_size,
            alpha=0.28,
            color=color_by_task[task_id],
            linewidths=0,
            label=f"task{task_id} {task_names[task_id]}",
        )

    fail_mask = success == 0
    ax.scatter(
        feats[fail_mask, 0],
        feats[fail_mask, 1],
        s=point_size * 1.8,
        alpha=alpha,
        facecolors="none",
        edgecolors="black",
        linewidths=0.25,
        label="failure rollout frames",
    )

    (x0, x1), (y0, y1) = axis_limits(feats)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Task clusters with failure rollout overlay")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=3)
    fig.tight_layout()

    save_path = artifact_dir / "feats_vis_skip1-taskid_failure_overlay.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_taskid_failred(
    artifact_dir: Path,
    feats: np.ndarray,
    task_ids: np.ndarray,
    success: np.ndarray,
    metas: list[RolloutMeta],
    point_size: float,
    alpha: float,
) -> Path:
    task_order = sorted({meta.task_id for meta in metas})
    task_names = {meta.task_id: meta.task_name for meta in metas}
    cmap = plt.get_cmap("tab10", max(10, len(task_order)))
    color_by_task = {task_id: cmap(i) for i, task_id in enumerate(task_order)}

    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)
    for task_id in task_order:
        mask = (task_ids == task_id) & (success == 1)
        ax.scatter(
            feats[mask, 0],
            feats[mask, 1],
            s=point_size,
            alpha=0.55,
            color=color_by_task[task_id],
            linewidths=0,
            label=f"task{task_id} {task_names[task_id]} success",
        )

    fail_mask = success == 0
    ax.scatter(
        feats[fail_mask, 0],
        feats[fail_mask, 1],
        s=point_size,
        alpha=alpha,
        color="#d62728",
        linewidths=0,
        label="failure rollout frames",
    )

    (x0, x1), (y0, y1) = axis_limits(feats)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Task clusters with failure frames in red")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=3)
    fig.tight_layout()

    save_path = artifact_dir / "feats_vis_skip1-taskid_failred.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_task_success_facets(
    artifact_dir: Path,
    feats: np.ndarray,
    task_ids: np.ndarray,
    success: np.ndarray,
    time_rel: np.ndarray,
    metas: list[RolloutMeta],
    point_size: float,
    alpha: float,
) -> Path:
    task_order = sorted({meta.task_id for meta in metas})
    task_names = {meta.task_id: meta.task_name for meta in metas}
    n_tasks = len(task_order)
    ncols = min(3, n_tasks)
    nrows = int(np.ceil(n_tasks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), dpi=220)
    axes_arr = np.asarray(axes).reshape(-1)
    (x0, x1), (y0, y1) = axis_limits(feats)

    for ax, task_id in zip(axes_arr, task_order):
        task_mask = task_ids == task_id
        succ_mask = task_mask & (success == 1)
        fail_mask = task_mask & (success == 0)

        ax.scatter(
            feats[succ_mask, 0],
            feats[succ_mask, 1],
            s=point_size,
            alpha=0.35,
            color="#3b63d9",
            linewidths=0,
            label="success rollout",
        )
        scatter = ax.scatter(
            feats[fail_mask, 0],
            feats[fail_mask, 1],
            s=point_size * 1.2,
            alpha=alpha,
            c=time_rel[fail_mask],
            cmap="Reds",
            vmin=0,
            vmax=1,
            linewidths=0,
            label="failure rollout time",
        )
        n_success = len({meta.episode_idx for meta in metas if meta.task_id == task_id and meta.success == 1})
        n_fail = len({meta.episode_idx for meta in metas if meta.task_id == task_id and meta.success == 0})
        ax.set_title(
            f"task{task_id} {task_names[task_id]}  succ={n_success} fail={n_fail}",
            fontsize=12,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_arr[n_tasks:]:
        ax.axis("off")

    fig.subplots_adjust(left=0.025, right=0.86, bottom=0.13, top=0.88, wspace=0.04, hspace=0.18)
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    cax = fig.add_axes([0.88, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("relative time within failed rollout")
    fig.suptitle("Success/failure distribution within each task cluster", y=0.965, fontsize=14)

    save_path = artifact_dir / "feats_vis_skip1-task_success_facets.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def update_manifest(artifact_dir: Path, outputs: list[Path]) -> None:
    manifest = load_manifest(artifact_dir)
    current = list(manifest.get("outputs", []))
    for output in outputs:
        if output.name not in current:
            current.append(output.name)
    manifest["outputs"] = current
    with (artifact_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    manifest = load_manifest(artifact_dir)
    source_data_path = Path(manifest["source_data_path"])

    feats, rollout_indices = load_projection(artifact_dir)
    metas = load_rollout_meta(source_data_path)
    task_ids, success, time_rel = build_point_meta(metas, rollout_indices)

    outputs = [
        plot_taskid_failred(
            artifact_dir, feats, task_ids, success, metas, args.point_size, args.alpha
        ),
        plot_taskid_failure_overlay(
            artifact_dir, feats, task_ids, success, metas, args.point_size, args.alpha
        ),
        plot_task_success_facets(
            artifact_dir, feats, task_ids, success, time_rel, metas, args.point_size, args.alpha
        ),
    ]
    update_manifest(artifact_dir, outputs)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
