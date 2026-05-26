"""Cluster metrics: silhouette, centroid distance, per-point a-vs-b, ROC-AUC."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .distance import l2_normalize


def silhouette_safe(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    sample_size: int | None = None,
    random_state: int = 0,
) -> float | None:
    """sklearn silhouette_score with safe fallbacks (None when undefined)."""
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return None
    ss = sample_size if (sample_size is not None and X.shape[0] > sample_size) else None
    try:
        return float(
            silhouette_score(X, labels, metric=metric, sample_size=ss, random_state=random_state)
        )
    except ValueError:
        return None


def centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.sort(np.unique(labels))
    cents = np.stack([X[labels == c].mean(axis=0) for c in unique], axis=0)
    return unique, cents


def cosine_centroids(Xn: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of unit vectors per class, renormalized. Xn must already be L2-normalized."""
    unique = np.sort(np.unique(labels))
    cents = []
    for c in unique:
        m = Xn[labels == c].mean(axis=0)
        cents.append(m / max(np.linalg.norm(m), 1e-12))
    return unique, np.stack(cents, axis=0)


def pairwise_euclidean(C: np.ndarray) -> np.ndarray:
    diff = C[:, None, :] - C[None, :, :]
    return np.linalg.norm(diff.astype(np.float64), axis=-1)


def pairwise_cosine(C: np.ndarray) -> np.ndarray:
    sim = np.clip(C @ C.T, -1.0, 1.0)
    return 1.0 - sim


def centroid_distance(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Return the pairwise distance matrix between centroids."""
    if metric == "cosine":
        Xn = l2_normalize(X)
        _, cents = cosine_centroids(Xn, labels)
        return pairwise_cosine(cents)
    _, cents = centroids(X, labels)
    return pairwise_euclidean(cents)


def per_point_ab(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray]:
    """For each point: (a) distance to own centroid, (b) to nearest other centroid.

    Distances are computed in `metric` space; for cosine the caller passes
    L2-normalized X is recommended (this helper also handles it internally).
    """
    if metric == "cosine":
        Xn = l2_normalize(X)
        unique, cents = cosine_centroids(Xn, labels)
        own_idx = np.searchsorted(unique, labels)
        sim = np.clip(Xn @ cents.T, -1.0, 1.0)
        dist = 1.0 - sim
    else:
        unique, cents = centroids(X, labels)
        own_idx = np.searchsorted(unique, labels)
        diff = X[:, None, :] - cents[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
    own = dist[np.arange(len(labels)), own_idx]
    others = dist.copy()
    others[np.arange(len(labels)), own_idx] = np.inf
    return own, others.min(axis=1)


def silhouette_proxy_mean(own: np.ndarray, other: np.ndarray) -> float:
    """Centroid-based silhouette proxy: mean((b - a) / max(a, b))."""
    return float(np.mean((other - own) / np.maximum(own, other)))


def logistic_auc(
    X: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    C: float = 1.0,
    random_state: int = 0,
) -> float | None:
    """Train logistic regression on `train_mask` rows, return ROC-AUC on `test_mask`."""
    if (
        train_mask.sum() < 4
        or test_mask.sum() < 2
        or len(np.unique(labels[train_mask])) != 2
        or len(np.unique(labels[test_mask])) != 2
    ):
        return None
    scaler = StandardScaler().fit(X[train_mask])
    Xtr = scaler.transform(X[train_mask])
    Xte = scaler.transform(X[test_mask])
    clf = LogisticRegression(max_iter=5000, C=C, random_state=random_state).fit(Xtr, labels[train_mask])
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(labels[test_mask], proba))
