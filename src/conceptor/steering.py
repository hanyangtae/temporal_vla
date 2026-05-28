"""Multiplicative gate 기반 activation steering (COAST Sec. 3.2).

Contrastive conceptor C_steer 를 multiplicative gate 로 변환해 hidden state 에
적용한다. beta 로 steering 강도를 보간한다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_steering_matrix", "apply_steering"]


def build_steering_matrix(C_steer: np.ndarray, beta: float) -> np.ndarray:
    """Build multiplicative gate M = (1 - beta) * I + beta * C_steer.

    beta in [0, 1]: 0 -> identity (no steering), 1 -> full conceptor.

    Returns (d, d) matrix.
    """
    C_steer = np.asarray(C_steer, dtype=np.float64)
    if C_steer.ndim != 2 or C_steer.shape[0] != C_steer.shape[1]:
        raise ValueError(f"C_steer must be square (d, d), got {C_steer.shape}")
    d = C_steer.shape[0]
    return (1.0 - beta) * np.eye(d, dtype=np.float64) + beta * C_steer


def apply_steering(h: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply h' = h @ M.T to a batch of activations.

    Args:
        h: (B, S, d) or (B, d) hidden states
        M: (d, d) steering matrix

    Returns:
        same shape as h
    """
    h_arr = np.asarray(h)
    M = np.asarray(M, dtype=h_arr.dtype if np.issubdtype(h_arr.dtype, np.floating) else np.float64)
    if h_arr.shape[-1] != M.shape[0]:
        raise ValueError(
            f"last dim of h ({h_arr.shape[-1]}) must match M ({M.shape[0]})"
        )
    return h_arr @ M.T
