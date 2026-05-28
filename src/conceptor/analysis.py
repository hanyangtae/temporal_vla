"""Conceptor 진단/선택 지표 (COAST App. A.10.2, Sec. 4.4).

레이어 선택 (quota), aperture 선택 / cross-task containment (overlap),
spectrum 시각화 (Fig. 3A), 실패 부분공간 전이 (containment) 에 사용.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "conceptor_quota",
    "conceptor_overlap",
    "eigenvalue_spectrum",
    "failure_containment",
]


def conceptor_quota(C: np.ndarray) -> float:
    """Quota q(C) = (1/d) * tr(C). COAST App. A.10.2 Stage 1: layer selection.

    C 가 부분공간을 얼마나 차지하는지(유효 rank 비율) 측정. [0, 1) 범위.
    """
    C = np.asarray(C, dtype=np.float64)
    d = C.shape[0]
    return float(np.trace(C) / d)


def conceptor_overlap(C_a: np.ndarray, C_b: np.ndarray) -> float:
    """Overlap sim(C_a, C_b) = tr(C_a @ C_b) / sqrt(tr(C_a^2) * tr(C_b^2)).

    Frobenius 내적 기반 cosine 유사도. aperture 선택(App. A.10.2 Stage 2)과
    cross-task containment(Sec. 4.4) 에 사용. sim(C, C) = 1.
    """
    C_a = np.asarray(C_a, dtype=np.float64)
    C_b = np.asarray(C_b, dtype=np.float64)
    num = np.sum(C_a * C_b)  # tr(C_a @ C_b) (둘 다 대칭)
    denom = np.sqrt(np.sum(C_a * C_a) * np.sum(C_b * C_b))
    if denom == 0.0:
        return 0.0
    return float(num / denom)


def eigenvalue_spectrum(C: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues (descending) of C. For Fig. 3A reproduction."""
    C = np.asarray(C, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(C)  # ascending, real (대칭 가정)
    return eigvals[::-1].copy()


def failure_containment(C_src: np.ndarray, C_tgt: np.ndarray) -> float:
    """Failure-subspace containment tr(C_src @ C_tgt) / tr(C_src^2).

    COAST Sec. 4.4 cross-task transfer: source 실패 부분공간이 target 실패
    부분공간에 얼마나 포함되는지 (1 에 가까울수록 강한 포함).
    """
    C_src = np.asarray(C_src, dtype=np.float64)
    C_tgt = np.asarray(C_tgt, dtype=np.float64)
    denom = np.sum(C_src * C_src)  # tr(C_src^2)
    if denom == 0.0:
        return 0.0
    return float(np.sum(C_src * C_tgt) / denom)
