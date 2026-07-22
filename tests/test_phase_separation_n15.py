"""N1.5 event-phase 분리도 모듈 순수함수 테스트 (rank-AUROC / pooling).

env: 아무 numpy 있는 python (torch 불필요 — 순수함수만). repo root 기준 실행.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_MOD = REPO / "scripts/safe/groot_n15/robocasa/analyze/phase_separation.py"
_spec = importlib.util.spec_from_file_location("phase_separation_n15", _MOD)
ps = importlib.util.module_from_spec(_spec)
sys.modules["phase_separation_n15"] = ps
_spec.loader.exec_module(ps)


# --- rank_auroc --------------------------------------------------------------
def test_rank_auroc_perfect():
    a = ps.rank_auroc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]))
    assert abs(a - 1.0) < 1e-9


def test_rank_auroc_reversed():
    a = ps.rank_auroc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 0]))
    assert abs(a - 0.0) < 1e-9


def test_rank_auroc_ties_half():
    # pos vals {1,2}, neg vals {1,2}: P(pos>neg) with ties = 0.5
    a = ps.rank_auroc(np.array([1.0, 1.0, 2.0, 2.0]), np.array([0, 1, 0, 1]))
    assert abs(a - 0.5) < 1e-9


def test_rank_auroc_single_class():
    assert ps.rank_auroc(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])) == 0.5


def test_rank_auroc_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(4, 20)
        s = rng.normal(size=n)
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        pos, neg = s[y == 1], s[y == 0]
        brute = np.mean([(1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg])
        assert abs(ps.rank_auroc(s, y) - brute) < 1e-9


# --- pool_denoise ------------------------------------------------------------
def test_pool_denoise_shape_and_value():
    rec = np.arange(7 * 4 * 5, dtype=np.float32).reshape(7, 4, 5)
    out = ps.pool_denoise(rec)
    assert out.shape == (7, 5)
    assert np.allclose(out, rec.mean(axis=1))


def test_pool_denoise_rejects_bad_ndim():
    try:
        ps.pool_denoise(np.zeros((7, 5)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-3D record")


# --- equal_budget_pool -------------------------------------------------------
def test_equal_budget_pool_uses_first_k():
    vecs = [np.full(3, i, dtype=np.float32) for i in range(5)]  # [0,0,0],[1..],...
    out = ps.equal_budget_pool(vecs, budget=2)
    assert np.allclose(out, [0.5, 0.5, 0.5])  # mean of first 2 = (0+1)/2


def test_equal_budget_pool_rejects_insufficient():
    try:
        ps.equal_budget_pool([np.zeros(3)], budget=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError when budget > available")


# --- loo_auroc separability sanity ------------------------------------------
def test_loo_auroc_separates_shifted_gaussians():
    rng = np.random.default_rng(1)
    X0 = rng.normal(0, 1, size=(10, 8))
    X1 = rng.normal(3, 1, size=(10, 8))  # clearly shifted → LOO AUROC high
    X = np.vstack([X0, X1])
    y = np.array([0] * 10 + [1] * 10)
    assert ps.loo_auroc(X, y) > 0.8


def test_loo_auroc_chance_on_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(20, 8))
    y = np.array([0] * 10 + [1] * 10)
    a = ps.loo_auroc(X, y)
    assert a is not None and abs(a - 0.5) < 0.35  # no real signal → near chance
