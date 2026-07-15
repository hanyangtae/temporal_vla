#!/usr/bin/env python3
"""pq3 판정 규칙 — 사전 등록·동결 모듈 (계획서 v9 §F, Codex R2 #3).

이 파일은 Gate D(hash 동결) 시점에 내용이 확정되며, aggregate_pq3.py 가 본 파일의
sha256 을 summary 에 기록한다. **eval 시작 후 이 파일을 수정하면 판정 무효.**

Primary 6 가설 (Holm, 단측, α=0.05):
  {H1: cross_scene(perm) > base, H2: gated > base, H3: gated > perm(paired)}
  × {drawer(2 cell, n=60), ppcc(3 cell, n=90)}
검정: exact paired McNemar 단독 — discordant pair (b = A승·B패, c = A패·B승),
  p = P(X ≥ b), X ~ Binomial(b+c, 0.5). randomization 은 참고 병기(판정 미사용).
null 관문: |wins(null) − wins(base)| ≤ NULL_GATE_MARGIN_GAMES (task-pool) 이어야
  H1 해석 유효 (배선·방향 특이성 진단 전용 — "perm 무효" 단정 금지).
비재현 선언: paired Δ(SR) 단측 95% CI 상한 < +0.16 일 때만. 그 외 null 결과는 불확정.
"""

from __future__ import annotations

import math
import random

ALPHA = 0.05
NULL_GATE_MARGIN_GAMES = 4  # task-pool 판수 기준 (drawer n=60 → 6.7%p / ppcc n=90 → 4.4%p)
COAST_REF_DELTA = 0.16      # 비재현 CI 규칙의 참조 효과 (COAST GR00T RoboCasa +0.16)

# (가설, task) → (A arm, B arm) — arm 태그는 build_pq3_queue.ARM_ORDER 와 일치
HYPOTHESES = {
    "H1": ("ho_coast_cross_scene", "ho_base"),
    "H2": ("ho_gated_cross_scene", "ho_base"),
    "H3": ("ho_gated_cross_scene", "ho_coast_cross_scene"),
}
NULL_ARM = "ho_null_cross_scene"
BASE_ARM = "ho_base"


def exact_mcnemar_one_sided(b: int, c: int) -> float:
    """P(X ≥ b), X ~ Binomial(b+c, 0.5) — H: A > B (b = A승·B패 discordant)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down. 반환 {key: {p, p_adj, reject}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running_max = 0.0
    rejecting = True
    for rank, (key, p) in enumerate(items):
        p_adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, p_adj)
        p_adj = running_max  # 단조화
        if p_adj > alpha:
            rejecting = False
        out[key] = {"p": p, "p_adj": p_adj, "reject": rejecting and p_adj <= alpha}
    return out


def paired_delta_ci_upper_one_sided(b: int, c: int, n: int) -> float:
    """paired ΔSR = (b-c)/n 의 단측 95% CI 상한 (정규 근사, Var((b-c)/n) = (b+c-(b-c)^2/n)/n^2).

    비재현 선언 규칙 전용: 상한 < COAST_REF_DELTA 일 때만 "비재현", 그 외 불확정.
    """
    if n == 0:
        return float("inf")
    d = (b - c) / n
    var = max(0.0, (b + c - (b - c) ** 2 / n)) / (n ** 2)
    z = 1.6448536269514722  # Φ⁻¹(0.95)
    return d + z * math.sqrt(var)


def fwer_sim(episode_triples_by_task: dict[str, list[tuple[int, int, int]]],
             n_sim: int = 10000, seed: int = 20260715) -> float:
    """global-null FWER 시뮬레이션 (episode 단위 arm 라벨 교환 — 의존 구조 보존).

    episode_triples_by_task: task → [(base, perm, gated), ...] 실측 outcome.
    global null 하에서 arm 교환가능 → episode 마다 triple 을 무작위 순열로 재배정,
    6 가설 p → Holm → 하나라도 기각되는 비율 = FWER 추정.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_sim):
        pvals = {}
        for task, triples in episode_triples_by_task.items():
            perm_outcomes = []
            for t in triples:
                vals = list(t)
                rng.shuffle(vals)
                perm_outcomes.append(tuple(vals))
            arms = {"ho_base": [t[0] for t in perm_outcomes],
                    "ho_coast_cross_scene": [t[1] for t in perm_outcomes],
                    "ho_gated_cross_scene": [t[2] for t in perm_outcomes]}
            for h, (a, barm) in HYPOTHESES.items():
                av, bv = arms[a], arms[barm]
                b = sum(1 for x, y in zip(av, bv) if x == 1 and y == 0)
                c = sum(1 for x, y in zip(av, bv) if x == 0 and y == 1)
                pvals[f"{h}:{task}"] = exact_mcnemar_one_sided(b, c)
        if any(v["reject"] for v in holm(pvals).values()):
            hits += 1
    return hits / n_sim
