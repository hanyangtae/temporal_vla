"""사후 이산화 — 학습이 끝난 뒤 잠재를 클러스터링해 이산 상태를 만든다.

출처: task_classification@88543a2 `phase/clustering/posthoc.py`
      (https://github.com/robots-oh/task_classification)
이식 근거: docs/steering/29_sae_port_review.md §2.3(dead feature 처리), §5.A
이식 계획: docs/steering/30_sae_g1_port_handout.md §4 Phase A3 (G1엔 부수, G2에 필요)

[원본 대비 변경]
- gpu backend(`phase/clustering/gpu.py` GPUKMeans/GPUGMM) 경로 제거 — sklearn 만 남긴다
  (원격 CPU 노드 기준, 의존성 최소화). backend 인자도 삭제.
- scipy 의존 없음(원본도 없음). sklearn.cluster/mixture 만 사용.

--- 원본 모듈 설명 (보존) ---
이산 구조가 학습에 전혀 개입하지 않는다는 점이 핵심이다. 클러스터러는 **train 잠재로만**
적합해 val/test 누수를 막는다.

이산화 방법이 결과를 만들어내는지 보려고 등거리(k-means)와 공분산 인지(full-cov GMM)를
함께 돌린다. 둘이 크게 갈리면 그 구조는 방법의 산물이라고 봐야 한다.
"""
from __future__ import annotations

import numpy as np


def active_mask(z):
    """train 잠재 [N, d] → dead feature 를 뺀 bool 마스크 [d].

    (원본 `fit_clusters` 안에 있던 한 줄을 우리가 함수로 뽑았다 — top-k SAE 의
    dead-feature 비율은 G1 모델 선택 기준(<50%)이라 클러스터링과 무관하게 필요하다.)
    """
    z = np.asarray(z)
    return z.max(0) > 0


def dead_fraction(z):
    """dead feature 비율 (우리 추가 — 핸드아웃 §4 C1 선택 기준)."""
    return float(1.0 - active_mask(z).mean())


def fit_clusters(z, K, seed=0, drop_inactive=False):
    """train 잠재 [N, d] → (kmeans, gmm, active).

    [인자]
        drop_inactive  희소 코드 전용. train에서 한 번도 켜지지 않은 차원(dead feature)을
                       빼고 적합한다. 조밀 잠재에는 의미가 없으므로 기본 False.

    [반환] active는 [d] bool 마스크. assign에 반드시 같은 마스크를 넘겨야 한다
           (클러스터러가 그 부분공간에서만 적합됐으므로).
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    z = np.asarray(z)
    active = active_mask(z) if drop_inactive else np.ones(z.shape[1], bool)
    if not active.any():
        raise RuntimeError("활성 차원이 하나도 없습니다 — 인코더가 전부 죽었습니다")
    zf = z[:, active]

    km = KMeans(K, n_init=10, random_state=seed).fit(zf)
    gm = GaussianMixture(K, covariance_type="full", reg_covar=1e-4,
                         max_iter=200, random_state=seed).fit(zf)
    return km, gm, active


def assign(z, clf, active=None):
    """잠재 [N, d] → 상태열 [N] int64. 격자 복원이 없으므로 그대로 step 배열이다."""
    z = np.asarray(z)
    return clf.predict(z if active is None else z[:, active]).astype(np.int64)
