#!/usr/bin/env python3
"""exp5-4 선택(selection) 분석 공용 유틸.

원격 `~/exp53_analysis/selection_upper_bound.py` / `selection_generality.py` 의
load() / dir_from() / top-1 선택 로직과 **동일한 계약**을 유지하되, 20 scene × 8 seed
완전균형 설계를 [S,J,D] 행렬로 다뤄 순열 검정을 벡터화한다.

데이터 계약 (실측 확인, 2026-07-27)
  · npz `<cell>_sh*.npz`: X [n_rec,7,1536] (layer축 = [0,2,4,8,10,12,15]),
    bounds [n_ep+1], scene(=env seed), inf(=noise draw seed), label, ep, layers.
  · 3 cell 모두 scene 20개 × inf 8개 = 160판, (scene,inf) 쌍은 정확히 1판씩.
  · 8개 inf 값 {0,1e6,...,7e6} 은 **모든 scene 이 공유** → seed column 구조 성립.

방향 fit (변경 없음): 평가 scene 제외한 혼재 scene 각각 mean(fail)-mean(succ)
→ 평균 → L2 정규화. proj = V·w (클수록 실패 쪽) → 오름차순 top-1.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np


# ── 로딩 ────────────────────────────────────────────────────────────────
def load(npz_dir: Path, cell: str):
    """cell shard 들을 모아 episode 리스트 + layer 목록 반환 (ep 순 정렬)."""
    eps, layers = [], None
    for f in sorted(Path(npz_dir).glob(f"{cell}_sh*.npz")):
        z = np.load(f, allow_pickle=True)
        if layers is None:
            layers = [int(x) for x in z["layers"]]
        X, b = z["X"], z["bounds"]
        for i in range(len(z["label"])):
            eps.append(dict(X=X[b[i]:b[i + 1]], label=int(z["label"][i]),
                            scene=int(z["scene"][i]), inf=int(z["inf"][i]),
                            ep=int(z["ep"][i])))
    eps.sort(key=lambda e: e["ep"])
    return eps, layers


def to_matrix(eps, li: int, W: int = 1, rec: int | None = None):
    """[S,J,D] 활성 행렬 + [S,J] 라벨 + scene/seed 목록 + [S,J] episode index.

    rec=None : V = X[:W, li, :].mean(0)  (기본 W=1 → record 0 = t=0 활성, 누적창)
    rec=r    : V = X[r, li, :]           (단일 record — r×5 env-step 실행 후 inference)
    """
    scenes = sorted({e["scene"] for e in eps})
    seeds = sorted({e["inf"] for e in eps})
    si = {s: i for i, s in enumerate(scenes)}
    ji = {s: j for j, s in enumerate(seeds)}
    D = eps[0]["X"].shape[-1]
    A = np.full((len(scenes), len(seeds), D), np.nan)
    Y = np.full((len(scenes), len(seeds)), -1, dtype=int)
    E = np.full((len(scenes), len(seeds)), -1, dtype=int)
    for e in eps:
        i, j = si[e["scene"]], ji[e["inf"]]
        if rec is None:
            A[i, j] = e["X"][:W, li, :].mean(axis=0)
        else:
            if e["X"].shape[0] <= rec:
                raise ValueError(
                    f"ep {e['ep']} (scene {e['scene']}, inf {e['inf']}): "
                    f"n_rec={e['X'].shape[0]} ≤ rec={rec} — record 결측")
            A[i, j] = e["X"][rec, li, :]
        Y[i, j] = e["label"]
        E[i, j] = e["ep"]
    if np.isnan(A).any() or (Y < 0).any():
        raise ValueError("설계가 완전균형이 아님 — (scene,seed) 결측")
    return A.astype(np.float64), Y, scenes, seeds, E


def episode_index_matrix(eps):
    """episode 리스트 순서(index) 를 [S,J] 로 배치 (baseline 점수 정렬용)."""
    scenes = sorted({e["scene"] for e in eps})
    seeds = sorted({e["inf"] for e in eps})
    si = {s: i for i, s in enumerate(scenes)}
    ji = {s: j for j, s in enumerate(seeds)}
    I = np.full((len(scenes), len(seeds)), -1, dtype=int)
    for k, e in enumerate(eps):
        I[si[e["scene"]], ji[e["inf"]]] = k
    return I, scenes, seeds


# ── 선택 (벡터화 LOSO) ──────────────────────────────────────────────────
def loso_select(A, Y, fit_seeds=None, eval_seeds=None):
    """LOSO 방향으로 scene 별 top-1 / worst-1 선택.

    fit_seeds  : [J] bool — 방향 fit 에 쓸 seed column (기본 전체)
    eval_seeds : [J] bool — 후보(선택 대상) seed column (기본 전체)
    평가 scene 은 항상 fit 에서 제외(LOSO). fit_seeds ≠ eval_seeds 이면
    scene-out + seed-out 이중 제외(prospective) 가 된다.

    반환 dict: top1[S], worst1[S], valid[S], base[S](후보 평균 SR), chosen[S]
    """
    S, J, _ = A.shape
    fs = np.ones(J, bool) if fit_seeds is None else np.asarray(fit_seeds, bool)
    es = np.ones(J, bool) if eval_seeds is None else np.asarray(eval_seeds, bool)

    Ys = (Y == 1) & fs[None, :]
    Yf = (Y == 0) & fs[None, :]
    ns, nf = Ys.sum(1), Yf.sum(1)
    mixed = (ns > 0) & (nf > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_s = np.einsum("sj,sjd->sd", Ys.astype(np.float64), A) / np.maximum(ns, 1)[:, None]
        mu_f = np.einsum("sj,sjd->sd", Yf.astype(np.float64), A) / np.maximum(nf, 1)[:, None]
    Dm = (mu_f - mu_s) * mixed[:, None]
    tot, cnt = Dm.sum(0), int(mixed.sum())
    denom = cnt - mixed.astype(int)                      # 자기 scene 제외
    W = (tot[None, :] - Dm) / np.maximum(denom, 1)[:, None]
    nrm = np.linalg.norm(W, axis=1)
    valid = (denom > 0) & (nrm > 0)
    W = W / np.where(nrm > 0, nrm, 1)[:, None]

    proj = np.einsum("sjd,sd->sj", A, W)
    proj = np.where(es[None, :], proj, np.inf)
    chosen = np.argmin(proj, axis=1)
    projw = np.where(es[None, :], proj, -np.inf)
    worst = np.argmax(projw, axis=1)
    rows = np.arange(S)
    base = Y[:, es].mean(1)
    return dict(top1=Y[rows, chosen], worst1=Y[rows, worst], valid=valid,
                base=base, chosen=chosen)


def score_select(scores, Y, eval_seeds=None, ascending=True):
    """임의 점수 [S,J] 로 scene 내 top-1 (ascending=True → 작은 쪽 선택)."""
    S, J = Y.shape
    es = np.ones(J, bool) if eval_seeds is None else np.asarray(eval_seeds, bool)
    v = np.asarray(scores, float) * (1.0 if ascending else -1.0)
    ok = np.isfinite(v) & es[None, :]
    vv = np.where(ok, v, np.inf)
    chosen = np.argmin(vv, axis=1)
    worst = np.argmax(np.where(ok, v, -np.inf), axis=1)
    rows = np.arange(S)
    # 후보 중 하나라도 점수가 결측이면 그 scene 은 비교 불가 → 제외 (부분 후보 비교 금지)
    valid = np.isfinite(v[:, es]).all(1)
    return dict(top1=Y[rows, chosen], worst1=Y[rows, worst], valid=valid,
                base=Y[:, es].mean(1), chosen=chosen)


def delta_stat(res):
    """primary 통계량 Δ̂ = mean_i(y_top1,i − 후보 평균 SR_i) (전 scene ITT)."""
    v = res["valid"]
    top1 = res["top1"][v].astype(float)
    base = res["base"][v].astype(float)
    return dict(delta=float(np.mean(top1 - base)), hits=int(top1.sum()),
                n_scene=int(v.sum()), sr_top1=float(top1.mean()),
                sr_base=float(base.mean()),
                sr_worst1=float(res["worst1"][v].astype(float).mean()))


# ── Poisson-binomial exact ──────────────────────────────────────────────
def poisson_binomial_pmf(p):
    """독립 Bernoulli(p_i) 합의 정확 pmf (DP)."""
    dist = np.zeros(1)
    dist[0] = 1.0
    for pi in p:
        new = np.zeros(len(dist) + 1)
        new[:-1] += dist * (1 - pi)
        new[1:] += dist * pi
        dist = new
    return dist


def pb_pvalue(p, h_obs):
    """P(H >= h_obs) — 관측 m_i 조건부 exact randomization p."""
    pmf = poisson_binomial_pmf(p)
    return float(pmf[int(h_obs):].sum()), pmf


# ── seed column 공통 순열 ───────────────────────────────────────────────
def seed_permutations(J: int, n_max: int | None, rng=None):
    """8! 전수 (n_max 이상이면) 또는 무작위 부분표본."""
    from math import factorial
    total = factorial(J)
    if n_max is None or total <= n_max:
        return list(permutations(range(J))), True
    perms = set()
    while len(perms) < n_max:
        perms.add(tuple(rng.permutation(J).tolist()))
    return [np.array(p) for p in perms], False


# ── rollout csv 유틸 ────────────────────────────────────────────────────
def build_ep_index(root: Path):
    """rollout 루트를 한 번만 스캔해 {ep: csv path} 인덱스 구축."""
    idx = {}
    for p in Path(root).rglob("*--ep*--succ*.csv"):
        try:
            ep = int(p.name.split("--ep")[1].split("--")[0])
        except (IndexError, ValueError):
            continue
        idx[ep] = p
    return idx


def read_actions(csv_path: Path, n_rows: int):
    """csv 앞 n_rows 행 action (dx,dy,dz,droll,dpitch,dyaw,dgripper)."""
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for ln in f:
            if len(rows) >= n_rows:
                break
            parts = ln.strip().split(",")
            if len(parts) != len(header):
                continue
            rows.append([float(x) for x in parts])
    return (np.asarray(rows, dtype=np.float64) if rows else None), header
