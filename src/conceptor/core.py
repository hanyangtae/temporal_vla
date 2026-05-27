"""Conceptor 계산 + Boolean algebra (COAST Sec. 3.1 + App. A.9).

Conceptor C 는 활성화 분포의 주성분 부분공간을 ``soft-projection`` 으로 표현하는
PSD 행렬이다. 본 모듈은 활성화 행렬 X (N, d) 로부터 conceptor 를 fitting 하고,
Jaeger(2014) Boolean algebra (NOT / AND / OR) 및 COAST 의 contrastive conceptor
(Eq. 4) 를 구현한다.

Numerical precision (COAST App. A.9.4):
- 모든 correlation / matrix inverse 계산은 float64 로 수행한다.
- 입력이 float32 여도 내부적으로 float64 로 승격해 계산한 뒤 반환한다.
- float32 저장이 필요하면 ``as_float32`` 헬퍼로 export 한다.
- AND 는 singular case 안전을 위해 항상 Moore-Penrose pseudoinverse 를 사용한다.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_correlation",
    "compute_conceptor",
    "not_conceptor",
    "and_conceptor",
    "or_conceptor",
    "contrastive_conceptor",
    "as_float32",
]

# AND 연산 전 conceptor 고유값을 [eps, 1-eps] 로 clamp 하는 eps.
# C 자체(작은 aperture)와 NOT(C)=I-C(큰 aperture 라 C 가 1 에 saturate) 가 모두
# near-singular 가 될 수 있어, clamp 로 양쪽 inverse 의 조건수를 <= 1/eps 로 bound 한다.
_AND_EIG_CLAMP = 1e-3


def _validate_activation(X: np.ndarray) -> np.ndarray:
    """입력 활성화 행렬 검증 후 float64 (N, d) 로 반환."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, d), got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"X must have N >= 2 rows, got N={X.shape[0]}")
    return X.astype(np.float64, copy=False)


def compute_correlation(X: np.ndarray) -> np.ndarray:
    """Mean-center X (N, d) and return correlation R (d, d) in float64.

    R = X_tilde.T @ X_tilde / N,  where X_tilde = X - X.mean(axis=0).
    """
    X = _validate_activation(X)
    X_tilde = X - X.mean(axis=0, keepdims=True)
    n = X_tilde.shape[0]
    R = (X_tilde.T @ X_tilde) / n
    # 대칭성 보장 (부동소수점 오차 제거).
    return (R + R.T) * 0.5


def compute_conceptor(X: np.ndarray, alpha: float) -> np.ndarray:
    """Compute conceptor C = R(R + alpha^-2 I)^-1 from activation matrix X (N, d).

    R 의 고유값 sigma_i 에 대해 C 의 고유값은
    lambda_i = sigma_i / (sigma_i + alpha^-2) 이며 항상 [0, 1) 구간에 있다.

    Args:
        X: (N, d) activation matrix
        alpha: aperture, larger -> less regularized

    Returns:
        C: (d, d) conceptor matrix, PSD with eigenvalues in [0, 1)
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    R = compute_correlation(X)
    d = R.shape[0]
    M = R + (alpha ** -2) * np.eye(d, dtype=np.float64)
    # C = R @ inv(M). R 와 M=R+aI 는 commute 하므로 solve(M, R) 로 대칭 결과를 얻는다.
    C = np.linalg.solve(M, R)
    return (C + C.T) * 0.5


def not_conceptor(C: np.ndarray) -> np.ndarray:
    """NOT operation: I - C."""
    C = np.asarray(C, dtype=np.float64)
    return np.eye(C.shape[0], dtype=np.float64) - C


def _clamp_spectrum(C: np.ndarray, eps: float) -> np.ndarray:
    """conceptor 고유값을 [eps, 1-eps] 로 clip (대칭 PSD 가정).

    near-singular 방향(고유값 ~0 또는 ~1)을 capping 해 이어지는 inverse 의 조건수를
    bound 한다. 유효 aperture 를 위/아래로 제한하는 것과 동치다.
    """
    C = (C + C.T) * 0.5
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, 1.0 - eps)
    return (V * w) @ V.T


def and_conceptor(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """AND operation: (A^-1 + B^-1 - I)^-1 (Jaeger 2014).

    덧셈이 commutative 하므로 A AND B == B AND A.

    두 valid conceptor 의 AND 는 항상 valid conceptor(고유값 [0,1])여야 한다. 하지만
    naive pinv 는 near-singular 입력에서 0 근처 고유값을 거대값으로 역변환해 발산한다
    (작은 aperture 의 C, 또는 큰 aperture 라 1 에 saturate 한 C 의 NOT). 입력 고유값을
    ``_AND_EIG_CLAMP`` 로 [eps, 1-eps] clamp 하면 A, B 가 well-conditioned PD 가 되어
    plain inverse 로 안정적으로 계산되고 결과의 validity 가 보장된다(COAST App. A.9.4
    의 "near-singular conceptor 처리"를 우리 데이터 regime 에 맞춰 강건화).
    """
    A = _clamp_spectrum(np.asarray(A, dtype=np.float64), _AND_EIG_CLAMP)
    B = _clamp_spectrum(np.asarray(B, dtype=np.float64), _AND_EIG_CLAMP)
    d = A.shape[0]
    inner = np.linalg.inv(A) + np.linalg.inv(B) - np.eye(d, dtype=np.float64)
    C = np.linalg.inv(inner)
    return (C + C.T) * 0.5


def or_conceptor(R_A: np.ndarray, R_B: np.ndarray, alpha: float) -> np.ndarray:
    """OR operation: (R_A + R_B)(R_A + R_B + alpha^-2 I)^-1.

    correlation 행렬을 합쳐 다시 conceptor 로 fitting 하는 것과 동치이다.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    R_A = np.asarray(R_A, dtype=np.float64)
    R_B = np.asarray(R_B, dtype=np.float64)
    R_sum = R_A + R_B
    d = R_sum.shape[0]
    M = R_sum + (alpha ** -2) * np.eye(d, dtype=np.float64)
    C = np.linalg.solve(M, R_sum)
    return (C + C.T) * 0.5


def contrastive_conceptor(
    X_success: np.ndarray,
    X_failure: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (C_success, C_failure, C_steer = C_succ AND NOT C_fail).

    COAST Eq. 4:
        C_steer = C_s AND (NOT C_f) = (pinv(C_s) + pinv(I - C_f) - I)^-1.
    성공 부분공간 중 실패 부분공간과 겹치지 않는 방향만 남긴다.

    Returns:
        (C_s, C_f, C_steer) — all (d, d)
    """
    C_s = compute_conceptor(X_success, alpha)
    C_f = compute_conceptor(X_failure, alpha)
    C_steer = and_conceptor(C_s, not_conceptor(C_f))
    return C_s, C_f, C_steer


def as_float32(C: np.ndarray) -> np.ndarray:
    """float64 로 계산한 conceptor 를 저장/전송용 float32 로 export."""
    return np.asarray(C).astype(np.float32)
