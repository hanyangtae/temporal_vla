"""DiT action-token feature extraction and per-rollout temporal aggregations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .io import tensor_to_numpy


def pooled_hidden_states(record: dict[str, Any]) -> np.ndarray:
    """Mean-pool DiT action tokens over K (denoising) and H (action horizon).

    Input: record["hidden_states"] is a list of tensors with shape [K, H, D].
    Returns: float32 array [T, D] where T = number of env-steps in the rollout.
    """
    feats = []
    for h in record["hidden_states"]:
        a = tensor_to_numpy(h).astype(np.float32, copy=False)
        if a.ndim != 3:
            raise ValueError(f"Expected hidden state [K, H, D], got {a.shape}")
        feats.append(a.mean(axis=(0, 1)))
    return np.stack(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)


def frame_at(z: np.ndarray, p: float) -> np.ndarray:
    """Return the frame closest to progress fraction p in [0, 1]."""
    T = z.shape[0]
    idx = int(round(p * (T - 1)))
    idx = max(0, min(T - 1, idx))
    return z[idx]


def z_mean_over(z: np.ndarray, t_eff: int | None = None) -> np.ndarray:
    """Mean over the first t_eff frames (defaults to full trajectory)."""
    if t_eff is None or t_eff >= z.shape[0]:
        return z.mean(axis=0)
    return z[: max(1, t_eff)].mean(axis=0)


def z_last(z: np.ndarray) -> np.ndarray:
    return z[-1]


def z_displacement(z: np.ndarray) -> np.ndarray:
    return z[-1] - z[0]


def z_velocity_stats(z: np.ndarray) -> dict[str, float]:
    if z.shape[0] < 2:
        return {"mean": 0.0, "max": 0.0, "path": 0.0}
    diffs = np.linalg.norm(np.diff(z, axis=0), axis=1)
    return {"mean": float(diffs.mean()), "max": float(diffs.max()), "path": float(diffs.sum())}


def z_concat_snapshots(z: np.ndarray, n_snapshots: int = 3) -> np.ndarray:
    """Concatenate evenly-spaced snapshots (default: first / middle / last)."""
    T = z.shape[0]
    if n_snapshots <= 1:
        return z[-1].copy()
    idxs = np.linspace(0, T - 1, n_snapshots).round().astype(np.int64)
    return np.concatenate([z[i] for i in idxs])
