"""steering 판정 통계 라이브러리 — 종결 라운드 집계에서 증류 (2026-08-10 S7).

exp2~5 라운드 전용 집계 스크립트들을 archive 하면서, 차기 실험 판정이 그대로 다시 쓸
통계 패턴만 여기로 옮겼다. **구현은 검증된 라운드 코드의 이식**이며 출처를 함수마다
남긴다 (수치 재검산 시 git 이력의 원본과 대조 가능).

사용 계약 (판정 표준 — RESULTS·PITFALLS 의 교훈이 코드 계약이 된 것):
  - paired 비교는 같은 (좌표, 머신) 의 base·arm 만 (docs/04 §3.2).
  - fit-seed ↔ eval-seed 분리 필수 (in-sample rescue 방지).
  - 다중비교는 ``holm``, 판정은 위약(라벨 순열) 대조 동반.
  - 결과 판정 전 confound-audit 스킬 (길이·scene·instruction).
"""
from __future__ import annotations

import math
from itertools import permutations
from typing import Sequence

import numpy as np


# ── paired 판정 ────────────────────────────────────────────────────────────────
def mcnemar_exact(b: int, c: int) -> float:
    """양측 exact McNemar p (불일치쌍 b, c 만 사용). b+c==0 이면 p=1.

    출처: exp4_1/aggregate_rescue.py (oracle rescue 판정에 사용).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_tail)


def holm(pvals: dict) -> dict:
    """Holm step-down 보정 (key→adj_p). 출처: exp4_1/aggregate_rescue.py,
    exp3 6-Holm 판정(아카이브)과 동일 규칙."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run_max = {}, 0.0
    for i, (k, p) in enumerate(items):
        run_max = max(run_max, (m - i) * p)
        adj[k] = min(1.0, run_max)
    return adj


# ── Poisson-binomial (scene 혼재 성공수의 exact 검정) ─────────────────────────
def poisson_binomial_pmf(p: Sequence[float]) -> np.ndarray:
    """독립 Bernoulli(p_i) 합의 정확 pmf (DP). 출처: exp5_4/analysis/_sel_common.py."""
    dist = np.zeros(1)
    dist[0] = 1.0
    for pi in p:
        new = np.zeros(len(dist) + 1)
        new[:-1] += dist * (1 - pi)
        new[1:] += dist * pi
        dist = new
    return dist


def pb_pvalue(p: Sequence[float], h_obs: int) -> tuple[float, np.ndarray]:
    """P(H >= h_obs) — 관측 조건부 exact randomization p. 출처: _sel_common.py."""
    pmf = poisson_binomial_pmf(p)
    return float(pmf[int(h_obs):].sum()), pmf


def crit_and_power(p_null: Sequence[float], p_alt: Sequence[float], alpha: float):
    """(임계 h*, 검정력, 실제 α) — 사전 검정력 산출. 출처: exp5_4/analysis/selection_power.py.

    exp 라운드 다수의 null 이 '효과 없음' 아니라 '검출 불가'였던 교훈(eval-power-mde)
    → 실험 설계 단계에서 이 함수로 MDE 를 먼저 산출할 것.
    """
    pmf0 = poisson_binomial_pmf(p_null)
    tail0 = np.cumsum(pmf0[::-1])[::-1]
    h_star = int(np.argmax(tail0 <= alpha))
    if tail0[h_star] > alpha:
        return None, 0.0, float(tail0[h_star])
    pmf1 = poisson_binomial_pmf(p_alt)
    tail1 = np.cumsum(pmf1[::-1])[::-1]
    return h_star, float(tail1[h_star]), float(tail0[h_star])


# ── 순열 (위약 대조·seed 주효과 검정) ─────────────────────────────────────────
def seed_permutations(J: int, n_max: int | None, rng=None):
    """J! 전수(가능하면) 또는 무작위 부분표본. 출처: _sel_common.py
    (exp5-4 'seed 주효과' 판정 — column 순열 p 의 구현)."""
    total = math.factorial(J)
    if n_max is None or total <= n_max:
        return list(permutations(range(J))), True
    perms = set()
    while len(perms) < n_max:
        perms.add(tuple(rng.permutation(J).tolist()))
    return [np.array(p) for p in perms], False


def label_permutation_pvalue(delta_obs: float, deltas_perm: Sequence[float]) -> float:
    """위약(라벨 순열) 대조 p — 관측 Δ가 순열 Δ 분포에서 차지하는 우측 꼬리.

    라운드들의 위약 규약을 함수화: 위약 fit(라벨 셔플)으로 얻은 Δ 들과 비교.
    (+1)/(N+1) 보정 — 관측치 자신 포함.
    """
    arr = np.asarray(list(deltas_perm), dtype=float)
    return float((1 + (arr >= delta_obs).sum()) / (1 + len(arr)))
