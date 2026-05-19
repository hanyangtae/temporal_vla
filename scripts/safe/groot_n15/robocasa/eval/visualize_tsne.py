#!/usr/bin/env python3
"""Create t-SNE visualizations for a SAFE GR00T N1.5 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional visualization dependency
    go = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", default="/home/dongkyu/pdk_ws/SAFE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--holdout-data-path", default=None)
    parser.add_argument("--train-source-name", default="train_pool")
    parser.add_argument("--holdout-source-name", default="holdout")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=6000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--export-projector", action="store_true")
    parser.add_argument("--include-task-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-task-ids", type=int, nargs="*", default=None)
    return parser.parse_args()


def _load_rollouts_for_path(cfg, load_rollouts, data_path: str, source: str):
    cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg_local.dataset.data_path = data_path
    cfg_local.dataset.load_to_cuda = False
    rollouts = load_rollouts(cfg_local)
    return [(source, r) for r in rollouts]


def _lstm_hidden(model, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if not hasattr(model, "lstm"):
        return None

    x = batch["features"]
    bsz, timesteps, dim = x.shape
    n_history_steps = getattr(model, "n_history_steps", -1)

    if n_history_steps < 0:
        out, _ = model.lstm(x)
        return out

    x_padded = torch.nn.functional.pad(
        x, (0, 0, n_history_steps, 0), mode="constant", value=0
    )
    windows = []
    for timestep in range(timesteps):
        windows.append(x_padded[:, timestep : timestep + n_history_steps, :])
    x_seq = torch.stack(windows, dim=1).reshape(bsz * timesteps, n_history_steps, dim)
    out, _ = model.lstm(x_seq)
    out = out[:, -1, :].view(bsz, timesteps, -1)
    return out


def _fit_tsne(x: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    n_samples = x.shape[0]
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for t-SNE, got {n_samples}")

    x_scaled = StandardScaler().fit_transform(x)
    pca_dim = min(50, x_scaled.shape[1], n_samples - 1)
    if pca_dim >= 2 and x_scaled.shape[1] > pca_dim:
        x_scaled = PCA(n_components=pca_dim, random_state=random_state).fit_transform(x_scaled)

    safe_perplexity = min(float(perplexity), max(2.0, (n_samples - 1) / 3.0))
    projector = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return projector.fit_transform(x_scaled)


def _plot_tsne(
    xy: np.ndarray,
    meta: pd.DataFrame,
    title: str,
    output_path: Path,
    score_column: str = "score",
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=180)
    axes = axes.ravel()

    failure = meta["failure"].to_numpy().astype(bool)
    colors = np.where(failure, "#d62728", "#2ca02c")
    axes[0].scatter(xy[:, 0], xy[:, 1], c=colors, s=9, alpha=0.75, linewidths=0)
    axes[0].scatter([], [], c="#2ca02c", s=25, label="success")
    axes[0].scatter([], [], c="#d62728", s=25, label="failure")
    axes[0].legend(frameon=False, markerscale=1.5)
    axes[0].set_title("Success / failure")

    task_scatter = axes[1].scatter(
        xy[:, 0],
        xy[:, 1],
        c=meta["task_id"].to_numpy(),
        cmap="tab10",
        s=9,
        alpha=0.8,
        linewidths=0,
    )
    axes[1].set_title("Task id")
    fig.colorbar(task_scatter, ax=axes[1], fraction=0.046, pad=0.04)

    score_scatter = axes[2].scatter(
        xy[:, 0],
        xy[:, 1],
        c=meta[score_column].to_numpy(),
        cmap="magma",
        s=9,
        alpha=0.8,
        linewidths=0,
    )
    axes[2].set_title(score_column)
    fig.colorbar(score_scatter, ax=axes[2], fraction=0.046, pad=0.04)

    markers = ["o", "x", "^", "s", "D"]
    for source_idx, source in enumerate(sorted(meta["source"].unique())):
        marker = markers[source_idx % len(markers)]
        mask = meta["source"].to_numpy() == source
        if mask.any():
            axes[3].scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=colors[mask],
                marker=marker,
                s=12 if marker == "x" else 9,
                alpha=0.75,
                linewidths=0.6 if marker == "x" else 0,
                label=source,
            )
    axes[3].legend(frameon=False)
    axes[3].set_title("Source")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_paper_style_html(
    xy: np.ndarray,
    meta: pd.DataFrame,
    output_path: Path,
    title: str,
):
    if go is None:
        return

    fig = go.Figure()
    success_mask = meta["failure"].to_numpy() == 0
    fail_mask = ~success_mask

    hover_cols = [
        "source",
        "global_episode_idx",
        "original_rollout_index",
        "episode_uid",
        "rollout_index",
        "task_id",
        "task_description",
        "episode_idx",
        "frame_idx",
        "step",
        "norm_step",
        "score",
        "success",
        "failure",
    ]

    def hover_text(rows: pd.DataFrame) -> list[str]:
        texts = []
        for _, row in rows.iterrows():
            pieces = []
            for col in hover_cols:
                val = row[col]
                if isinstance(val, float):
                    val = f"{val:.4f}"
                pieces.append(f"{col}: {val}")
            texts.append("<br>".join(pieces))
        return texts

    if success_mask.any():
        success_rows = meta.loc[success_mask]
        fig.add_trace(
            go.Scattergl(
                x=xy[success_mask, 0],
                y=xy[success_mask, 1],
                mode="markers",
                name="successful rollout frames",
                marker=dict(size=5, color="#1f77b4", opacity=0.65),
                text=hover_text(success_rows),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if fail_mask.any():
        fail_rows = meta.loc[fail_mask]
        fig.add_trace(
            go.Scattergl(
                x=xy[fail_mask, 0],
                y=xy[fail_mask, 1],
                mode="markers",
                name="failed rollout frames by normalized time",
                marker=dict(
                    size=5,
                    color=fail_rows["norm_step"],
                    colorscale="Bluered",
                    cmin=0.0,
                    cmax=1.0,
                    opacity=0.75,
                    colorbar=dict(title="failed<br>norm step"),
                ),
                text=hover_text(fail_rows),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        width=1100,
        height=850,
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(output_path, include_plotlyjs=True)


def _write_score_html(
    xy: np.ndarray,
    meta: pd.DataFrame,
    output_path: Path,
    title: str,
):
    if go is None:
        return

    hover_cols = [
        "source",
        "global_episode_idx",
        "original_rollout_index",
        "episode_uid",
        "rollout_index",
        "task_id",
        "task_description",
        "episode_idx",
        "frame_idx",
        "step",
        "norm_step",
        "score",
        "success",
        "failure",
    ]
    texts = []
    for _, row in meta.iterrows():
        pieces = []
        for col in hover_cols:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.4f}"
            pieces.append(f"{col}: {val}")
        texts.append("<br>".join(pieces))

    fig = go.Figure(
        go.Scattergl(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=meta["score"],
                colorscale="Magma",
                cmin=float(meta["score"].min()),
                cmax=float(meta["score"].max()),
                opacity=0.75,
                colorbar=dict(title="SAFE<br>score"),
            ),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        width=1100,
        height=850,
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    fig.write_html(output_path, include_plotlyjs=True)


def _write_projector_files(
    vectors: np.ndarray,
    meta: pd.DataFrame,
    output_dir: Path,
    name: str,
):
    projector_dir = output_dir / f"projector_{name}"
    projector_dir.mkdir(parents=True, exist_ok=True)

    vector_path = projector_dir / "vectors.tsv"
    metadata_path = projector_dir / "metadata.tsv"
    config_path = projector_dir / "projector_config.pbtxt"

    np.savetxt(vector_path, vectors, delimiter="\t", fmt="%.7g")
    metadata_cols = [
        "source",
        "global_episode_idx",
        "original_rollout_index",
        "episode_uid",
        "rollout_index",
        "task_id",
        "task_description",
        "episode_idx",
        "frame_idx",
        "step",
        "norm_step",
        "score",
        "success",
        "failure",
    ]
    meta[metadata_cols].to_csv(metadata_path, sep="\t", index=False)
    config_path.write_text(
        "embeddings {\n"
        f'  tensor_path: "{vector_path.name}"\n'
        f'  metadata_path: "{metadata_path.name}"\n'
        "}\n"
    )
    return {
        "name": name,
        "directory": str(projector_dir),
        "vectors": str(vector_path),
        "metadata": str(metadata_path),
        "config": str(config_path),
    }


def main() -> None:
    args = _parse_args()
    safe_root = Path(args.safe_root)
    sys.path.insert(0, str(safe_root))

    from failure_prob.conf import process_cfg
    from failure_prob.data import load_rollouts
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model
    from failure_prob.utils.torch import move_to_device

    checkpoint = Path(args.checkpoint)
    config = Path(args.config) if args.config else checkpoint.parent / "config.yaml"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(config)
    cfg.dataset.load_to_cuda = False
    cfg.model.batch_size = args.batch_size
    cfg = process_cfg(cfg)

    source_rollouts = _load_rollouts_for_path(
        cfg, load_rollouts, args.train_data_path, args.train_source_name
    )
    if args.holdout_data_path:
        source_rollouts += _load_rollouts_for_path(
            cfg, load_rollouts, args.holdout_data_path, args.holdout_source_name
        )

    n_rollouts_before_filter = len(source_rollouts)
    indexed_source_rollouts = [
        (idx, source, rollout) for idx, (source, rollout) in enumerate(source_rollouts)
    ]
    include_task_ids = (
        None if args.include_task_ids is None else {int(x) for x in args.include_task_ids}
    )
    exclude_task_ids = {int(x) for x in (args.exclude_task_ids or [])}
    if include_task_ids is not None:
        indexed_source_rollouts = [
            item
            for item in indexed_source_rollouts
            if int(item[2].task_id) in include_task_ids
        ]
    if exclude_task_ids:
        indexed_source_rollouts = [
            item
            for item in indexed_source_rollouts
            if int(item[2].task_id) not in exclude_task_ids
        ]

    original_rollout_indices = [idx for idx, _, _ in indexed_source_rollouts]
    sources = [source for _, source, _ in indexed_source_rollouts]
    rollouts = [rollout for _, _, rollout in indexed_source_rollouts]
    if not rollouts:
        raise ValueError("No rollouts loaded")

    input_dim = rollouts[0].hidden_states.shape[-1]
    model = get_model(cfg, input_dim)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = RolloutDataset(cfg, rollouts, device="cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    point_raw = []
    point_hidden = []
    point_rows = []
    rollout_rows = []
    rollout_hidden_vectors = []

    rollout_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["features"].shape[0]
            batch_device = move_to_device(batch, device)
            scores = model(batch_device).squeeze(-1).detach().cpu().numpy()
            hidden = _lstm_hidden(model, batch_device)
            hidden_np = None if hidden is None else hidden.detach().cpu().numpy()
            features_np = batch["features"].detach().cpu().numpy()
            masks_np = batch["valid_masks"].detach().cpu().numpy()

            for local_idx in range(batch_size):
                rollout_idx = rollout_offset + local_idx
                rollout = rollouts[rollout_idx]
                source = sources[rollout_idx]
                original_rollout_index = original_rollout_indices[rollout_idx]
                length = int(masks_np[local_idx].sum())
                valid_scores = scores[local_idx, :length]
                valid_raw = features_np[local_idx, :length]
                valid_hidden = None if hidden_np is None else hidden_np[local_idx, :length]
                failure = int(1 - rollout.episode_success)
                task_min_step = int(rollout.task_min_step or length)
                early_end = min(max(task_min_step, 1), length)
                early_score = float(valid_scores[:early_end].max())
                end_score = float(valid_scores.max())
                episode_uid = (
                    f"{source}/task{int(rollout.task_id):02d}/"
                    f"episode{int(rollout.episode_idx):02d}"
                )

                rollout_rows.append(
                    {
                        "source": source,
                        "global_episode_idx": rollout_idx,
                        "original_rollout_index": original_rollout_index,
                        "episode_uid": episode_uid,
                        "rollout_index": rollout_idx,
                        "task_id": int(rollout.task_id),
                        "task_description": str(rollout.task_description),
                        "episode_idx": int(rollout.episode_idx),
                        "success": int(rollout.episode_success),
                        "failure": failure,
                        "length": length,
                        "task_min_step": task_min_step,
                        "early_score": early_score,
                        "end_score": end_score,
                        "last_score": float(valid_scores[-1]),
                    }
                )
                rollout_hidden_vectors.append(
                    valid_raw.mean(axis=0) if valid_hidden is None else valid_hidden[-1]
                )

                for step in range(length):
                    point_raw.append(valid_raw[step])
                    if valid_hidden is not None:
                        point_hidden.append(valid_hidden[step])
                    point_rows.append(
                        {
                            "source": source,
                            "global_episode_idx": rollout_idx,
                            "original_rollout_index": original_rollout_index,
                            "episode_uid": episode_uid,
                            "rollout_index": rollout_idx,
                            "task_id": int(rollout.task_id),
                            "task_description": str(rollout.task_description),
                            "episode_idx": int(rollout.episode_idx),
                            "success": int(rollout.episode_success),
                            "failure": failure,
                            "frame_idx": step,
                            "step": step,
                            "norm_step": float(step / max(length - 1, 1)),
                            "score": float(valid_scores[step]),
                        }
                    )
            rollout_offset += batch_size

    point_meta = pd.DataFrame(point_rows)
    rollout_meta = pd.DataFrame(rollout_rows)
    raw = np.asarray(point_raw, dtype=np.float32)
    hidden_arr = np.asarray(point_hidden, dtype=np.float32) if point_hidden else None
    rollout_vectors = np.asarray(rollout_hidden_vectors, dtype=np.float32)

    rng = np.random.default_rng(args.random_state)
    if args.max_points > 0 and len(point_meta) > args.max_points:
        sampled_indices = np.sort(rng.choice(len(point_meta), size=args.max_points, replace=False))
    else:
        sampled_indices = np.arange(len(point_meta))

    point_meta_sampled = point_meta.iloc[sampled_indices].reset_index(drop=True)
    raw_sampled = raw[sampled_indices]
    raw_xy = _fit_tsne(raw_sampled, args.perplexity, args.random_state)
    projector_outputs = []
    if args.export_projector:
        projector_outputs.append(
            _write_projector_files(raw_sampled, point_meta_sampled, output_dir, "raw_features")
        )

    _plot_tsne(
        raw_xy,
        point_meta_sampled,
        "GR00T feature t-SNE colored by SAFE checkpoint outputs",
        output_dir / "tsne_raw_features.png",
    )
    _write_paper_style_html(
        raw_xy,
        point_meta_sampled,
        output_dir / "tsne_raw_features_paper_style.html",
        "GR00T feature t-SNE: success frames vs failed-trajectory time",
    )
    _write_score_html(
        raw_xy,
        point_meta_sampled,
        output_dir / "tsne_raw_features_score.html",
        "GR00T feature t-SNE colored by SAFE score",
    )

    hidden_xy = None
    if hidden_arr is not None:
        hidden_sampled = hidden_arr[sampled_indices]
        if args.export_projector:
            projector_outputs.append(
                _write_projector_files(
                    hidden_sampled,
                    point_meta_sampled,
                    output_dir,
                    "lstm_hidden",
                )
            )
        hidden_xy = _fit_tsne(hidden_sampled, args.perplexity, args.random_state)
        _plot_tsne(
            hidden_xy,
            point_meta_sampled,
            "SAFE LSTM hidden t-SNE colored by checkpoint outputs",
            output_dir / "tsne_lstm_hidden.png",
        )
        _write_paper_style_html(
            hidden_xy,
            point_meta_sampled,
            output_dir / "tsne_lstm_hidden_paper_style.html",
            "SAFE LSTM hidden t-SNE: success frames vs failed-trajectory time",
        )
        _write_score_html(
            hidden_xy,
            point_meta_sampled,
            output_dir / "tsne_lstm_hidden_score.html",
            "SAFE LSTM hidden t-SNE colored by SAFE score",
        )

    rollout_xy = _fit_tsne(
        rollout_vectors,
        min(args.perplexity, max(2.0, (len(rollout_vectors) - 1) / 3.0)),
        args.random_state,
    )
    _plot_tsne(
        rollout_xy,
        rollout_meta,
        "Rollout-level t-SNE using final SAFE LSTM hidden state",
        output_dir / "tsne_rollout_lstm_last.png",
        score_column="end_score",
    )

    point_meta_sampled.to_csv(output_dir / "point_metadata_sampled.tsv", sep="\t", index=False)
    rollout_meta.to_csv(output_dir / "rollout_summary.tsv", sep="\t", index=False)
    np.savez_compressed(
        output_dir / "tsne_data.npz",
        sampled_indices=sampled_indices,
        raw_xy=raw_xy,
        hidden_xy=np.asarray([]) if hidden_xy is None else hidden_xy,
        rollout_xy=rollout_xy,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "config": str(config),
        "n_rollouts": int(len(rollout_meta)),
        "n_rollouts_before_filter": int(n_rollouts_before_filter),
        "n_points_total": int(len(point_meta)),
        "n_points_sampled": int(len(point_meta_sampled)),
        "threshold": args.threshold,
        "include_task_ids": None if include_task_ids is None else sorted(include_task_ids),
        "exclude_task_ids": sorted(exclude_task_ids),
        "outputs": [
            "tsne_raw_features.png",
            "tsne_lstm_hidden.png" if hidden_arr is not None else None,
            "tsne_rollout_lstm_last.png",
            "tsne_raw_features_paper_style.html" if go is not None else None,
            "tsne_lstm_hidden_paper_style.html" if hidden_arr is not None and go is not None else None,
            "tsne_raw_features_score.html" if go is not None else None,
            "tsne_lstm_hidden_score.html" if hidden_arr is not None and go is not None else None,
            "point_metadata_sampled.tsv",
            "rollout_summary.tsv",
            "tsne_data.npz",
        ],
        "projector_outputs": projector_outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
