"""Pooled within-class covariance, Mahalanobis whitening, L2-normalize."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def pooled_within_cov(X: np.ndarray, labels: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    """Pooled within-class covariance with relative ridge regularization."""
    X64 = X.astype(np.float64, copy=False)
    D = X64.shape[1]
    scatter = np.zeros((D, D), dtype=np.float64)
    dof = 0
    for c in np.unique(labels):
        Xc = X64[labels == c]
        if len(Xc) < 2:
            continue
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        scatter += Xc.T @ Xc
        dof += len(Xc) - 1
    cov = scatter / max(dof, 1)
    cov += reg * np.trace(cov) / D * np.eye(D)
    return cov


def whiten(X: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Y = L^-1 X.T -> X.T transformed s.t. Euclidean(Y) == Mahalanobis(X, cov)."""
    L = cholesky(cov, lower=True)
    return solve_triangular(L, X.astype(np.float64).T, lower=True).T.astype(np.float32)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)
