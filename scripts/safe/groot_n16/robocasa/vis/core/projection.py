"""2D projection helpers: LDA and t-SNE."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


def lda_2d(X: np.ndarray, labels: np.ndarray, random_state: int = 0) -> np.ndarray | None:
    """Project to up to 2 LDA components; pad 1D output with tiny jitter."""
    n_classes = len(np.unique(labels))
    n_comp = min(2, n_classes - 1)
    if n_comp <= 0:
        return None
    Z = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(
        X.astype(np.float64), labels
    )
    if Z.shape[1] == 1:
        rng = np.random.default_rng(random_state)
        Z = np.column_stack(
            [Z[:, 0], rng.normal(scale=0.02 * (Z[:, 0].std() + 1e-9), size=Z.shape[0])]
        )
    return Z.astype(np.float32)


def run_tsne(
    X: np.ndarray,
    *,
    metric: str = "euclidean",
    perplexity: int = 30,
    random_state: int = 0,
    init: str = "pca",
) -> np.ndarray:
    """t-SNE 2D embedding. Use metric='euclidean' on raw or whitened data,
    or metric='cosine' on raw / L2-normalized data."""
    # init="pca" with cosine metric is invalid in sklearn; fall back to "random"
    if metric == "cosine" and init == "pca":
        init = "random"
    perp = min(perplexity, max(2, X.shape[0] // 3 - 1))
    return TSNE(
        n_components=2,
        perplexity=perp,
        metric=metric,
        random_state=random_state,
        init=init,
        learning_rate="auto",
    ).fit_transform(X).astype(np.float32)
