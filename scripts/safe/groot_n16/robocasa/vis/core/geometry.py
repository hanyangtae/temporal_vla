"""Whitening + low-dim projection + ellipse/ellipsoid geometry helpers."""

from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import PCA

from .distance import pooled_within_cov, whiten


def whiten_by(X: np.ndarray, labels: np.ndarray, reg: float = 1e-2) -> np.ndarray:
    """Mahalanobis whitening of X by the pooled within-`labels` covariance (full-dim)."""
    return whiten(X, pooled_within_cov(X, labels, reg=reg))


def meandiff_axes(Xt: np.ndarray, succ: np.ndarray, seed: int = 0) -> np.ndarray:
    """3D coords for within-group succ/fail viewing:
      x = mean-difference direction (mu_fail - mu_succ), rank-1 / non-overfit,
      y,z = top-2 PCs of Xt orthogonal to x.
    succ is the binary (1=succ, 0=fail) vector. Returns [n, 3]."""
    mu_s, mu_f = Xt[succ == 1].mean(0), Xt[succ == 0].mean(0)
    d = mu_f - mu_s
    nd = np.linalg.norm(d)
    d = d / nd if nd > 0 else d
    ax = Xt @ d
    pcs = PCA(n_components=2, random_state=seed).fit_transform(Xt - np.outer(ax, d))
    return np.column_stack([ax - ax.mean(), pcs[:, 0], pcs[:, 1]])


def cov_ellipse_2d(pts: np.ndarray, n_sigma: float = 2.0):
    """(center[2], width, height, angle_deg) of the n-sigma covariance ellipse."""
    mu = pts.mean(0)
    cov = np.cov(pts.T)
    w, V = np.linalg.eigh(cov)
    order = w.argsort()[::-1]
    w, V = w[order], V[:, order]
    ang = math.degrees(math.atan2(V[1, 0], V[0, 0]))
    width, height = 2 * n_sigma * np.sqrt(np.maximum(w, 1e-12))
    return mu, float(width), float(height), float(ang)


def ellipsoid_3d(pts3: np.ndarray, n_sigma: float = 2.0, n: int = 18):
    """(X, Y, Z) n×n grids for an n-sigma covariance ellipsoid mesh surface."""
    mu = pts3.mean(0)
    cov = np.cov(pts3.T)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-9, None)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    sx = np.outer(np.cos(u), np.sin(v)).ravel()
    sy = np.outer(np.sin(u), np.sin(v)).ravel()
    sz = np.outer(np.ones_like(u), np.cos(v)).ravel()
    P = ((V @ np.diag(n_sigma * np.sqrt(w))) @ np.stack([sx, sy, sz])).T + mu
    return P[:, 0].reshape(n, n), P[:, 1].reshape(n, n), P[:, 2].reshape(n, n)
