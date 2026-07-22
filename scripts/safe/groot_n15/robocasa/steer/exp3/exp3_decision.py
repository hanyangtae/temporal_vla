#!/usr/bin/env python3
"""exp3(구 pq3) 판정 규칙 — 사전 등록·동결 모듈 (계획서 v9 §F, Codex R2 #3).

이 파일은 Gate D(hash 동결) 시점에 내용이 확정되며, aggregate_exp3.py 가 본 파일의
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

# (가설, task) → (A arm, B arm) — arm 태그는 build_exp3_queue.ARM_ORDER 와 일치
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
    """paired ΔSR = (b-c)/n 의 단측 95% CI 상한 (정규 근사 — 참고 병기용).

    판정에는 아래 bootstrap 상한을 쓴다 (Gate 2 높음#12 — 근사 대신 사전 고정 방법).
    """
    if n == 0:
        return float("inf")
    d = (b - c) / n
    var = max(0.0, (b + c - (b - c) ** 2 / n)) / (n ** 2)
    z = 1.6448536269514722  # Φ⁻¹(0.95)
    return d + z * math.sqrt(var)


def paired_delta_ci_upper_bootstrap(
    pair_diffs: list[int], n_boot: int = 10000, seed: int = 20260715
) -> float:
    """episode-paired ΔSR 의 단측 95% CI 상한 — percentile bootstrap (사전 고정·동결).

    pair_diffs: episode 별 (A승-B승) ∈ {-1,0,1} 목록. 비재현 선언: 상한 < COAST_REF_DELTA.
    """
    n = len(pair_diffs)
    if n == 0:
        return float("inf")
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = sum(pair_diffs[rng.randrange(n)] for _ in range(n))
        stats.append(s / n)
    stats.sort()
    idx = min(n_boot - 1, max(0, int(math.ceil(0.95 * n_boot)) - 1))
    return stats[idx]


def final_status(holm_results: dict, null_gates: dict, ci_uppers: dict,
                 gated_na_tasks: tuple = ()) -> dict:
    """판정 최종 상태 (계획서 v9 §F 분기 — 수동 해석 개입 차단, Gate 2 높음#12).

    Args:
        holm_results: {"H1:drawer": {"reject": bool, ...}, ...}
            (기본 6개; gated 성립 게이트 미통과 task 는 H2/H3 가 사전 N/A — 그 수만큼 감소)
        null_gates:   {"drawer": bool(h1_interpretable), "ppcc": bool}
        ci_uppers:    {"H1:drawer": float(bootstrap 상한), ...}
        gated_na_tasks: gated 성립 게이트 미통과 task 목록 (H2/H3 제외 근거 기록)
    Returns: {"per_hypothesis": {...}, "overall": enum}
      enum: "positive->confirmatory" | "all_null_nonreplication" | "all_null_inconclusive"
            | "positive_but_null_gate_violated"
    """
    expected = 6 - 2 * len(gated_na_tasks)
    if len(holm_results) != expected:
        raise ValueError(
            f"primary 는 {expected}개여야 함 (gated N/A={list(gated_na_tasks)}): "
            f"{sorted(holm_results)}"
        )
    per = {}
    any_valid_reject = False
    any_gate_violated_reject = False
    for key, res in holm_results.items():
        h, task = key.split(":")
        interpretable = null_gates.get(task, False) if h == "H1" else True
        effective = bool(res["reject"]) and interpretable
        per[key] = {
            "reject": bool(res["reject"]),
            "null_gate_ok": interpretable,
            "effective_reject": effective,
        }
        if res["reject"] and not interpretable:
            any_gate_violated_reject = True
        if effective:
            any_valid_reject = True
    if any_valid_reject:
        overall = "positive->confirmatory"
    elif any_gate_violated_reject:
        overall = "positive_but_null_gate_violated"
    else:
        nonrep = all(u < COAST_REF_DELTA for u in ci_uppers.values())
        overall = "all_null_nonreplication" if nonrep else "all_null_inconclusive"
    return {"per_hypothesis": per, "overall": overall}


def fwer_sim(episode_triples_by_task: dict[str, list[tuple[int, int, int]]],
             n_sim: int = 10000, seed: int = 20260715,
             gated_na_tasks: tuple = ()) -> float:
    """global-null FWER 시뮬레이션 (episode 단위 arm 라벨 교환 — 의존 구조 보존).

    episode_triples_by_task: task → [(base, perm, gated), ...] 실측 outcome.
    gated N/A task 는 gated 원소를 무시하고 (base, perm) 교환·H1 만 family 에 포함.
    global null 하에서 arm 교환가능 → episode 마다 outcome 을 무작위 순열로 재배정,
    family 전체 p → Holm → 하나라도 기각되는 비율 = FWER 추정.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_sim):
        pvals = {}
        for task, triples in episode_triples_by_task.items():
            na = task in gated_na_tasks
            perm_outcomes = []
            for t in triples:
                vals = list(t[:2] if na else t)
                rng.shuffle(vals)
                perm_outcomes.append(tuple(vals))
            arms = {"ho_base": [t[0] for t in perm_outcomes],
                    "ho_coast_cross_scene": [t[1] for t in perm_outcomes]}
            if not na:
                arms["ho_gated_cross_scene"] = [t[2] for t in perm_outcomes]
            for h, (a, barm) in HYPOTHESES.items():
                if na and h in ("H2", "H3"):
                    continue
                av, bv = arms[a], arms[barm]
                b = sum(1 for x, y in zip(av, bv) if x == 1 and y == 0)
                c = sum(1 for x, y in zip(av, bv) if x == 0 and y == 1)
                pvals[f"{h}:{task}"] = exact_mcnemar_one_sided(b, c)
        if any(v["reject"] for v in holm(pvals).values()):
            hits += 1
    return hits / n_sim
