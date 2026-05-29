import numpy as np
import pytest
from src.conceptor.core import (
    compute_conceptor, not_conceptor, and_conceptor,
    contrastive_conceptor,
)
from src.conceptor.steering import build_steering_matrix, apply_steering
from src.conceptor.analysis import conceptor_quota, conceptor_overlap, eigenvalue_spectrum


@pytest.fixture
def small_data():
    """N=20, d=64 — overcomplete easy case."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 64))

@pytest.fixture
def underdet_data():
    """N=15, d=1024 — COAST real setting (n<<d)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((15, 1024)).astype(np.float32)


# === core 테스트 ===

def test_conceptor_shape(small_data):
    C = compute_conceptor(small_data, alpha=1.0)
    assert C.shape == (64, 64)

def test_conceptor_psd(small_data):
    """C는 PSD여야 함 (eigenvalues >= -eps)."""
    C = compute_conceptor(small_data, alpha=1.0)
    eigvals = np.linalg.eigvalsh(C)
    assert eigvals.min() >= -1e-9

def test_conceptor_eigenvalues_in_unit_interval(small_data):
    """C eigenvalues는 [0, 1)에 있어야."""
    C = compute_conceptor(small_data, alpha=1.0)
    eigvals = np.linalg.eigvalsh(C)
    assert eigvals.max() < 1.0
    assert eigvals.min() >= -1e-9

def test_underdetermined_works(underdet_data):
    """n<<d 상황에서도 numerical error 없이 작동."""
    C = compute_conceptor(underdet_data, alpha=1.0)
    assert np.all(np.isfinite(C))
    assert C.shape == (1024, 1024)

def test_alpha_monotonicity(small_data):
    """Larger alpha => larger quota (less regularized, more variance retained)."""
    C_small = compute_conceptor(small_data, alpha=0.1)
    C_large = compute_conceptor(small_data, alpha=10.0)
    assert conceptor_quota(C_large) > conceptor_quota(C_small)


# === Boolean algebra ===

def test_not_idempotence(small_data):
    """NOT NOT C == C (within numerical precision)."""
    C = compute_conceptor(small_data, alpha=1.0)
    not_not_C = not_conceptor(not_conceptor(C))
    np.testing.assert_allclose(not_not_C, C, atol=1e-8)

def test_and_commutative(small_data):
    """A AND B == B AND A."""
    rng = np.random.default_rng(1)
    X_a = small_data
    X_b = rng.standard_normal(small_data.shape)
    C_a = compute_conceptor(X_a, alpha=1.0)
    C_b = compute_conceptor(X_b, alpha=1.0)
    A_and_B = and_conceptor(C_a, C_b)
    B_and_A = and_conceptor(C_b, C_a)
    np.testing.assert_allclose(A_and_B, B_and_A, atol=1e-6)


def test_and_near_singular_stays_valid_conceptor():
    """near-singular(작은 alpha) 입력에서도 AND 결과가 valid conceptor.

    n<<d 에 작은 aperture 면 conceptor 는 0 근처 고유값이 많아 default pinv 가
    발산한다(고유값이 1e5 단위로 폭발). 결과는 항상 고유값 [0,1] 이어야 한다.
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 256))
    C = compute_conceptor(X, alpha=0.1)  # 작은 aperture -> near-singular
    # C AND NOT C 는 의미상 빈 conceptor(교집합 없음) -> 고유값 ~0, 발산 금지
    C_and = and_conceptor(C, not_conceptor(C))
    eigvals = np.linalg.eigvalsh(C_and)
    assert eigvals.max() < 1.0 + 1e-6
    assert eigvals.min() > -1e-6


# === Contrastive ===

def test_contrastive_returns_triple(small_data):
    rng = np.random.default_rng(2)
    X_succ = small_data
    X_fail = rng.standard_normal(small_data.shape)
    C_s, C_f, C_steer = contrastive_conceptor(X_succ, X_fail, alpha=1.0)
    for M in (C_s, C_f, C_steer):
        assert M.shape == (64, 64)
        assert np.all(np.isfinite(M))


# === Steering ===

def test_steering_identity_at_beta_zero():
    C = np.random.default_rng(3).standard_normal((10, 10))
    C = C @ C.T  # PSD
    M = build_steering_matrix(C, beta=0.0)
    np.testing.assert_allclose(M, np.eye(10))

def test_apply_steering_shape():
    h = np.random.default_rng(4).standard_normal((2, 5, 16))  # (B, S, d)
    M = np.eye(16)
    h_out = apply_steering(h, M)
    assert h_out.shape == h.shape

def test_apply_steering_identity_preserves():
    """M=I 이면 h_out == h."""
    h = np.random.default_rng(5).standard_normal((2, 5, 16))
    h_out = apply_steering(h, np.eye(16))
    np.testing.assert_allclose(h_out, h)


# === Analysis ===

def test_quota_range(small_data):
    """quota는 [0, 1] 범위."""
    C = compute_conceptor(small_data, alpha=1.0)
    q = conceptor_quota(C)
    assert 0.0 <= q <= 1.0

def test_overlap_self_is_one(small_data):
    """sim(C, C) = 1."""
    C = compute_conceptor(small_data, alpha=1.0)
    s = conceptor_overlap(C, C)
    np.testing.assert_allclose(s, 1.0, atol=1e-8)

def test_eigenvalue_spectrum_sorted(small_data):
    """Eigenvalue spectrum이 descending order."""
    C = compute_conceptor(small_data, alpha=1.0)
    eigvals = eigenvalue_spectrum(C)
    assert np.all(np.diff(eigvals) <= 1e-9)
