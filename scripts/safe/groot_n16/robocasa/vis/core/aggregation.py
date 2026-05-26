"""Per-rollout / per-inference feature aggregation strategies.

A "datapoint" in our SAFE rollouts is 1 GR00T inference (= K·H mean-pooled
DiT action-token latent, 1024-D). Rollouts contain 11~45 such datapoints.

This module turns a list of rollout records into a feature matrix suitable
for cluster / classification analysis, under one of several aggregation
strategies. The result also indicates the mapping back to source rollouts
so the caller can broadcast rollout-level labels to per-row labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _truncated(z: np.ndarray, truncate_t: int | None) -> np.ndarray:
    if truncate_t is None or truncate_t >= z.shape[0]:
        return z
    return z[: max(1, truncate_t)]


def aggregate(
    records: list[dict],
    feature: str,
    *,
    truncate_t: int | None = None,
    lstm_model: Any = None,
    device: str = "cpu",
    n_concat_snapshots: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, rollout_idx).

    Each record must have `r["z"] : ndarray[n_inferences, D]` (the per-inference
    K·H-pooled latent).

    Output:
      - X : (N_rows, D_out)
      - rollout_idx : (N_rows,) — index into `records` for each row; for
        rollout-level features this is `np.arange(len(records))`, for
        `z_inference` it expands to one entry per inference.

    Supported features:
      - 'z_inference'      : per-inference; D_out = D
      - 'z_mean'           : rollout-level mean over inferences; D_out = D
      - 'z_last'           : last inference; D_out = D
      - 'z_disp'           : last - first inference; D_out = D
      - 'z_concat'         : concat n_concat_snapshots evenly-spaced inferences
      - 'z_path'           : sum of consecutive ||Δz||; D_out = 1
      - 'h_T'              : LSTM hidden at last (or truncated) inference;
                             requires lstm_model
    """
    if feature == "z_inference":
        chunks, rollout_idx = [], []
        for i, r in enumerate(records):
            z = _truncated(r["z"], truncate_t)
            chunks.append(z)
            rollout_idx.extend([i] * z.shape[0])
        if not chunks:
            raise ValueError("aggregate: empty records")
        return np.concatenate(chunks, axis=0), np.asarray(rollout_idx, dtype=np.int64)

    out = []
    for r in records:
        z = _truncated(r["z"], truncate_t)
        if feature == "z_mean":
            out.append(z.mean(axis=0))
        elif feature == "z_last":
            out.append(z[-1])
        elif feature == "z_disp":
            out.append(z[-1] - z[0])
        elif feature == "z_concat":
            T = z.shape[0]
            if n_concat_snapshots <= 1:
                out.append(z[-1].copy())
            else:
                idxs = np.linspace(0, T - 1, n_concat_snapshots).round().astype(np.int64)
                out.append(np.concatenate([z[i] for i in idxs]))
        elif feature == "z_path":
            if z.shape[0] < 2:
                out.append(np.zeros(1, dtype=np.float32))
            else:
                diffs = np.linalg.norm(np.diff(z, axis=0), axis=1)
                out.append(np.asarray([float(diffs.sum())], dtype=np.float32))
        elif feature == "h_T":
            if lstm_model is None:
                raise ValueError("aggregate(h_T): lstm_model is required")
            from .lstm import lstm_hidden_sequence
            h_seq = lstm_hidden_sequence(lstm_model, z, device)
            out.append(h_seq[-1])
        else:
            raise ValueError(f"aggregate: unknown feature {feature!r}")

    X = np.stack(out, axis=0).astype(np.float32)
    return X, np.arange(len(records), dtype=np.int64)


def broadcast_labels(rollout_labels: np.ndarray, rollout_idx: np.ndarray) -> np.ndarray:
    """Map per-rollout labels to per-row labels via rollout_idx mapping."""
    return rollout_labels[rollout_idx]
