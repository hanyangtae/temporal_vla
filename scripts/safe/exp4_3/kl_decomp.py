#!/usr/bin/env python3
"""exp4-3 통합 진단 축: succ/fail 분리도를 **평균 성분 + 분산 성분**으로 분해.

동기 (2026-07-23 사용자): conceptor(분산)·setM(평균)이 각각 한 성분만 보므로 셀·layer·
phase 마다 "어느 성분이 얼마나 기여하는가"를 한 지표로 읽고 싶다. 가우시안 근사에서
KL 은 정확히 두 항으로 쪼개진다 (둘 다 ≥0):

  2·KL(fail ‖ succ) = (μs−μf)ᵀ Σs⁻¹ (μs−μf)              ← mean_term
                    + [ tr(Σs⁻¹Σf) − k + ln(det Σs/det Σf) ]  ← cov_term

**축소 부분공간 처리 (사용자 동의)**: D=1536 에서 record 수백~수천으로 Σ 를 추정하면
불안정하므로, 두 클래스 **train 표본을 합친 pooled PCA 의 top-k(기본 32)** 부분공간으로
사영한 뒤 그 안에서만 분해한다. 부분공간은 **train 에서만** 추정하고 held-out record 를
같은 기저로 사영해 평가 — 사영 기저가 라벨을 보지 않으므로(클래스 무관 pooled PCA)
순열 null 과 대칭이다. shrinkage(Ledoit-Wolf 형태, λ 자동)로 Σ 조건수를 안정화한다.

산출 (셀·layer·phase 당):
  kl_total, kl_mean_term, kl_cov_term, mean_frac(=mean/total),
  각 항의 episode-라벨 순열 null z (kl_z, mean_z_kl, cov_z_kl), k, lam_s, lam_f
"""
from __future__ import annotations

import numpy as np

TOP_K = 32          # 부분공간 차원 (WA-LQR k=64 보다 보수적 — 표본 수백 기준)
MIN_PER_CLASS = 40  # 클래스당 최소 train record (미달 시 None)
HELD_FRAC = 0.2
EPS = 1e-9


def _shrunk_cov(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 형태 shrinkage: Σ̂ = (1−λ)S + λ·(tr S/k)I. λ 는 표본 기반 추정."""
    n, k = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / max(1, n - 1)
    mu = float(np.trace(S) / k)
    d2 = float(((S - mu * np.eye(k)) ** 2).sum() / k)
    b2 = 0.0
    for i in range(n):
        z = Xc[i:i + 1]
        b2 += float((((z.T @ z) - S) ** 2).sum() / k)
    b2 = min(d2, b2 / (n ** 2)) if n > 1 else d2
    lam = float(b2 / d2) if d2 > EPS else 1.0
    lam = min(1.0, max(0.0, lam))
    return (1 - lam) * S + lam * mu * np.eye(k), lam


def _subspace(X_tr: np.ndarray, k: int) -> np.ndarray:
    """클래스 라벨 무관 pooled PCA 기저 [D,k] (train 표본만)."""
    Xc = X_tr - X_tr.mean(axis=0)
    # 표본이 D 보다 적으면 economy SVD 로 충분
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def kl_split(Xs_tr, Xf_tr, Xs_he, Xf_he, k: int = TOP_K) -> dict | None:
    """train 으로 부분공간·모수 추정 → held-out 표본으로 분해값 산출.

    held-out 은 **분포 추정을 갱신하는 데만** 쓰고(모수는 train), 실제 평가는 held-out
    표본 공분산/평균으로 계산해 in-sample 낙관을 피한다.
    """
    if min(len(Xs_tr), len(Xf_tr)) < MIN_PER_CLASS or min(len(Xs_he), len(Xf_he)) < 8:
        return None
    V = _subspace(np.concatenate([Xs_tr, Xf_tr], axis=0), k)
    ps, pf = Xs_he @ V, Xf_he @ V
    if min(len(ps), len(pf)) <= k:  # 표본 < 차원이면 Σ 특이 — k 축소
        k2 = max(4, min(len(ps), len(pf)) // 2)
        V = V[:, :k2]
        ps, pf = Xs_he @ V, Xf_he @ V
        k = k2
    Ss, lam_s = _shrunk_cov(ps)
    Sf, lam_f = _shrunk_cov(pf)
    dmu = ps.mean(axis=0) - pf.mean(axis=0)
    try:
        Ss_inv = np.linalg.inv(Ss)
        sign_s, logdet_s = np.linalg.slogdet(Ss)
        sign_f, logdet_f = np.linalg.slogdet(Sf)
    except np.linalg.LinAlgError:
        return None
    if sign_s <= 0 or sign_f <= 0:
        return None
    mean_term = float(dmu @ Ss_inv @ dmu)
    cov_term = float(np.trace(Ss_inv @ Sf) - k + (logdet_s - logdet_f))
    cov_term = max(0.0, cov_term)  # 이론상 ≥0, 수치오차 클리핑
    total = mean_term + cov_term
    return {"kl_total": 0.5 * total, "kl_mean_term": 0.5 * mean_term,
            "kl_cov_term": 0.5 * cov_term,
            "mean_frac": (mean_term / total) if total > EPS else None,
            "k": int(k), "lam_s": lam_s, "lam_f": lam_f}
