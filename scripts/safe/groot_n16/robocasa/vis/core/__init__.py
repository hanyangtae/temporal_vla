"""Shared helpers for GR00T-N1.6 SAFE rollout analysis & visualization.

Modules:
    io          -- manifest / pkl loading, rollout iteration
    features    -- DiT action-token feature extraction & temporal aggregations
    lstm        -- SAFE-LSTM detector loading / inference
    distance    -- pooled within-class covariance, whitening, L2-normalize
    metrics     -- silhouette / centroid distance / per-point a-vs-b / ROC-AUC
    projection  -- LDA 2D, t-SNE wrappers
    plotting    -- colormaps, scatter helpers, axis-bound helpers, video writer
"""
