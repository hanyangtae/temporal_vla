"""COAST conceptor algebra — contrastive conceptor activation steering.

공개 API:
    core      : compute_correlation, compute_conceptor, not/and/or_conceptor,
                contrastive_conceptor, as_float32
    steering  : build_steering_matrix, apply_steering
    analysis  : conceptor_quota, conceptor_overlap, eigenvalue_spectrum,
                failure_containment
"""

from __future__ import annotations

from src.conceptor.analysis import (
    conceptor_overlap,
    conceptor_quota,
    eigenvalue_spectrum,
    failure_containment,
)
from src.conceptor.core import (
    and_conceptor,
    as_float32,
    compute_conceptor,
    compute_correlation,
    contrastive_conceptor,
    not_conceptor,
    or_conceptor,
)
from src.conceptor.steering import apply_steering, build_steering_matrix

__all__ = [
    # core
    "compute_correlation",
    "compute_conceptor",
    "not_conceptor",
    "and_conceptor",
    "or_conceptor",
    "contrastive_conceptor",
    "as_float32",
    # steering
    "build_steering_matrix",
    "apply_steering",
    # analysis
    "conceptor_quota",
    "conceptor_overlap",
    "eigenvalue_spectrum",
    "failure_containment",
]
