#!/usr/bin/env python3
"""exp4-1: setM(setpoint형 mean-diff) + setM_pl(라벨순열 위약) fit — cell(instruction)별.

수학 (docs/steering/24a §4.1): r̂ = normalize(μ_fail − μ_succ) **비중심화** per-record fit,
s = μ_succ·r̂ (성공 평균의 r̂ 좌표 = setpoint). 적용은 serve SetpointSteering
(h' = h − β[(h·r̂)−s]r̂). conceptor 가 per-class 중심화로 지우는 평균차 항이 곧 신호.

exp3 fit30 수집 pkl(full-token, 승준 노드)을 소비하므로 **승준 노드에서 실행**
(~/anaconda3/bin/python, remote_compute.sh run-bg). 산출 NPZ 는 소용량 → pull-results.

Layer 선정 (2026-07-22 사용자 결정 — setM 은 activation 평균 분리도 기준):
  cell별 DiT capture layer 전부에 대해
    ① episode 5-fold CV: train fold 로 r̂ fit → held-out fold record 사영 h·r̂ 의 AUROC
       (truncation cap 내 record 만 — 길이 confound 통제)
    ② episode-level 라벨순열 N_PERM 개로 같은 CV → null 분포 → z = (AUROC−μ_null)/σ_null
    ③ episode-cluster bootstrap B 회 → r̂ 각도 안정성 (full-fit r̂ 과의 cos)
  선정 = z-score 최대 layer (동률 시 bootstrap cos 우선). SR·eval episode 무접촉.

산출 (out_root/<cell>/):
  setM/steer/dit_L{B}/conceptors.npz          — alpha0_v_steer [D] + alpha0_s (선정 layer만)
  setM_pl/steer/dit_L{B}/conceptors.npz       — dose-match 동결 순열 1개
  setM_loo/ep{E}/steer/dit_L{B}/conceptors.npz    — fit-풀 구제 대상별 leave-one-out
  setM_pl_loo/ep{E}/steer/dit_L{B}/conceptors.npz — 위 pairing 위약 (같은 순열 라벨)
  layer_sweep.json / metadata.json            — 전 진단 + 사전등록 기록

사용 (승준):
  python fit_mean_diff.py --manifest <task_PPCC_fit.tsv> --cell pq3_ppcc_bread \
    --targets <annotation_t0.tsv> --out-root outputs/eval/robocasa/groot_n15/exp4_1/npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # steer/ (fit_phase_conceptor_n15)

from fit_phase_conceptor_n15 import (  # noqa: E402
    FULLTOKEN_MODE,
    load_rollout_fulltoken,
)

# 토큰 세그먼트 (N1.5 DiT T=49: state 1 + future 32 + action 16) — fit_phase_conceptor 규약
SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))
MOVE_GAP_RATIO_MAX = 3.0   # 배포 게이트: |이동량| 중앙값 ≤ succ/fail 갭 × 3

N_PERM = 20        # layer-sweep null 분포용 (CV 비용 때문에 소수)
N_PERM_PL = 200    # 위약 후보 풀 배치 크기 (준직교 후보 확보용)
N_PERM_PL_MAX = 1000  # 준직교 후보를 찾을 때까지 순열 풀을 확장하는 상한
PL_SCALE_MIN, PL_SCALE_MAX = 0.5, 2.0  # 위약 dose-match 스케일 클립 (벗어나면 기록)
PL_COS_MAX = 0.3   # 위약-처치 |cos| 상한: 정렬(처치 희석)·반정렬(반처치) 모두 불공정
N_BOOT = 200
RNG_SEED = 424101  # exp4-1 고정 (재현)


# ---------------------------------------------------------------------------- fit 원자
def mean_diff(Xs: np.ndarray, Xf: np.ndarray) -> tuple[np.ndarray, float]:
    """비중심화 mean-diff: r̂=normalize(μ_fail−μ_succ), s=μ_succ·r̂."""
    if len(Xs) == 0 or len(Xf) == 0:
        raise ValueError("클래스 표본 0개 — mean-diff 불가")
    mu_s = Xs.mean(axis=0)
    mu_f = Xf.mean(axis=0)
    d = mu_f - mu_s
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise ValueError("μ_fail == μ_succ — mean-diff 방향 없음")
    r = d / n
    return r, float(mu_s @ r)


def seg_of_token(t: int) -> int:
    for i, (_n, lo, hi) in enumerate(SEGMENTS):
        if lo <= t < hi:
            return i
    raise ValueError(f"token {t} 세그먼트 밖")


def fit_seg_setpoint(Xs_tok: np.ndarray, Xf_tok: np.ndarray):
    """토큰 보존 표본 [N,T,D] 두 클래스 → (v_seg [S,D], s_tok [T], diag).

    방향은 **세그먼트별**(같은 종류 토큰끼리 pooled — 표본 안정), setpoint 는 **토큰 위치별**
    (s_t = 성공 평균의 그 위치 사영). 2026-07-23 진단: 세그먼트 방향 cos 이 drawer 에서
    ≈0(직교), ppcc 에서도 action 은 +0.2대 — pooled 방향 하나로는 어느 세그먼트도 못 겨냥.
    """
    T = Xs_tok.shape[1]
    v_seg, diag = [], []
    for name, lo, hi in SEGMENTS:
        a = Xs_tok[:, lo:hi, :].reshape(-1, Xs_tok.shape[2])
        b = Xf_tok[:, lo:hi, :].reshape(-1, Xf_tok.shape[2])
        v, _s = mean_diff(a, b)
        v_seg.append(v)
        gap = float((b.mean(0) - a.mean(0)) @ v)
        diag.append({"segment": name, "gap_fail_minus_succ": gap,
                     "n_rec_succ": int(len(a)), "n_rec_fail": int(len(b))})
    v_seg = np.stack(v_seg, axis=0)
    s_tok = np.asarray([float(Xs_tok[:, t, :].mean(axis=0) @ v_seg[seg_of_token(t)])
                        for t in range(T)])
    # 배포 게이트용: 토큰별 |현재 사영 − s_t| 중앙값 vs 세그먼트 갭
    move, gaps = [], []
    allX = np.concatenate([Xs_tok, Xf_tok], axis=0)
    for t in range(T):
        si = seg_of_token(t)
        proj = allX[:, t, :] @ v_seg[si]
        move.append(float(np.median(np.abs(proj - s_tok[t]))))
        gaps.append(abs(diag[si]["gap_fail_minus_succ"]))
    ratio = [m / (g + 1e-9) for m, g in zip(move, gaps)]
    return v_seg, s_tok, {"segments": diag,
                          "move_median_per_token": move,
                          "move_over_gap_median": float(np.median(ratio)),
                          "move_over_gap_max": float(np.max(ratio))}


def seg_from_means(mu_s_tok: np.ndarray, mu_f_tok: np.ndarray):
    """클래스 토큰평균 [T,D] 두 개 → (v_seg [S,D], s_tok [T]).

    fit_seg_setpoint 와 **정의 동일** (record 수가 토큰마다 같으므로 세그먼트 평균 =
    토큰평균의 세그먼트 내 평균). 순열 후보를 매번 concat 없이 평가하기 위한 경로.
    """
    v_seg = []
    for _n, lo, hi in SEGMENTS:
        d = mu_f_tok[lo:hi].mean(axis=0) - mu_s_tok[lo:hi].mean(axis=0)
        nrm = float(np.linalg.norm(d))
        if nrm == 0.0:
            raise ValueError("μ_fail == μ_succ — 세그먼트 방향 없음")
        v_seg.append(d / nrm)
    v_seg = np.stack(v_seg, axis=0)
    s_tok = np.asarray([float(mu_s_tok[t] @ v_seg[seg_of_token(t)])
                        for t in range(mu_s_tok.shape[0])])
    return v_seg, s_tok


def seg_dose(X_tok: np.ndarray, v_seg: np.ndarray, s_tok: np.ndarray) -> float:
    """세그먼트 연산자가 실제로 만드는 이동량 중앙값 median_t,n |h_t·r̂_seg(t) − s_t|."""
    v_tok = np.stack([v_seg[seg_of_token(t)] for t in range(X_tok.shape[1])])  # [T,D]
    proj = np.einsum("ntd,td->nt", X_tok.astype(np.float64), v_tok)
    return float(np.median(np.abs(proj - s_tok[None, :])))


def seg_dose_by_segment(X_tok: np.ndarray, v_seg: np.ndarray, s_tok: np.ndarray) -> np.ndarray:
    """세그먼트별 이동량 중앙값 [S] — 위약 dose-match 스케일 산출용."""
    v_tok = np.stack([v_seg[seg_of_token(t)] for t in range(X_tok.shape[1])])
    move = np.abs(np.einsum("ntd,td->nt", X_tok.astype(np.float64), v_tok) - s_tok[None, :])
    return np.asarray([float(np.median(move[:, lo:hi])) for _n, lo, hi in SEGMENTS])


SEG_EQUIV_V_TOL = 2e-3   # 방향 cos 이탈 허용 (float32 누적 오차)
SEG_EQUIV_S_RTOL = 2e-3  # setpoint 상대 오차 허용


def check_seg_equiv(v_a, s_a, v_b, s_b, tag: str) -> None:
    """같은 라벨에서 나온 두 경로(재fit=float32 누적 vs 선택=episode합 float64)의 일치 확인.

    배포본은 float64 경로(선택 단계 산출)를 쓴다 — 더 정확하기 때문. 여기서는 두 경로가
    수치오차 수준에서만 다른지 확인해 라벨/표본이 어긋난 사고를 잡는다 (완전 일치 요구는
    float32 평균 누적 때문에 성립하지 않는다 — 07-23 beer 에서 실측).
    """
    dv = float(np.abs(v_a - v_b).max())
    cosmin = float(np.min([v_a[i] @ v_b[i] for i in range(v_a.shape[0])]))
    ds = float(np.abs(s_a - s_b).max())
    scale = float(np.abs(s_b).max()) + 1e-9
    print(f"  [equiv {tag}] Δv={dv:.2e} min cos={cosmin:.6f} Δs={ds:.3f} "
          f"({ds / scale * 100:.3f}% of |s|max)", flush=True)
    if dv > SEG_EQUIV_V_TOL or ds / scale > SEG_EQUIV_S_RTOL:
        raise SystemExit(f"[{tag}] 세그먼트 연산자 불일치 — 라벨/표본 어긋남 의심 "
                         f"(Δv={dv:.2e} Δs/|s|={ds / scale:.2e})")


def select_placebo_seg(ep_tok, labels_arr, v_seg_ref, s_tok_ref, rng, tag=""):
    """위약 순열 선택 — **세그먼트 공간에서** 준직교 + dose-match (2026-07-23 교정).

    구 기준은 49토큰 pooled 방향/스칼라 s 로 골랐는데 배포 연산자는 세그먼트별 r̂_seg +
    토큰별 s_t 다. 그래서 pooled cos 가 통과해도 실제 개입 축에서는 정렬이 깨졌다
    (실측: drawer_left [-0.55,-0.46,+0.44], mixer [-0.43,-0.13,+0.35], bread action +0.41).

    기준 (2026-07-23 확정):
      ① 준직교 — max_s |cos(r̂_seg^pl, r̂_seg^treat)| ≤ PL_COS_MAX. 후보가 없으면 순열 풀을
         N_PERM_PL 씩 N_PERM_PL_MAX 까지 확장, 그래도 없으면 max|cos| 최소 폴백(기록).
      ② dose-match — 순열 방향은 실제 클래스 갭이 없어 이동량이 구조적으로 작다
         (밴드 ±25% 가 아예 도달 불가한 cell 존재: 구 기준 bread in_band=0). 그래서 밴드로
         거르는 대신 **세그먼트별 스케일** scale_s = dose_s^treat / dose_s^pl 를 연산자의
         seg_mask 에 실어(hook 은 mask 를 float 승수로 적용) 이동량을 처치와 **정확히** 맞춘다.
         후보는 필요한 스케일이 1에 가장 가까운 것(max_s |ln ratio| 최소)을 고르고,
         스케일은 [PL_SCALE_MIN, PL_SCALE_MAX] 로 클립(클립 시 기록).

    ep_tok: episode 별 토큰보존 record [n_i, T, D] 리스트 (labels_arr 와 같은 순서·길이).
    """
    keep = [i for i, X in enumerate(ep_tok) if len(X)]
    if not keep:
        raise ValueError("표본 있는 episode 0개")
    sums = {i: ep_tok[i].astype(np.float64).sum(axis=0) for i in keep}
    cnts = {i: len(ep_tok[i]) for i in keep}
    X_all = np.concatenate([ep_tok[i] for i in keep], axis=0)
    dose_ref = seg_dose(X_all, v_seg_ref, s_tok_ref)
    dose_ref_seg = seg_dose_by_segment(X_all, v_seg_ref, s_tok_ref)

    def build(pl):
        si = [i for i in keep if pl[i] == 1]
        fi = [i for i in keep if pl[i] == 0]
        if not si or not fi:
            raise ValueError("클래스 고갈")
        mu_s = sum(sums[i] for i in si) / sum(cnts[i] for i in si)
        mu_f = sum(sums[i] for i in fi) / sum(cnts[i] for i in fi)
        return seg_from_means(mu_s, mu_f)

    cands, n_drawn, ortho = [], 0, []
    while n_drawn < N_PERM_PL_MAX:
        for _ in range(N_PERM_PL):
            pl = labels_arr.copy()
            rng.shuffle(pl)
            pl = pl.tolist()
            n_drawn += 1
            try:
                v_c, s_c = build(pl)
            except ValueError:
                continue
            cos_seg = [float(v_c[s] @ v_seg_ref[s]) for s in range(len(SEGMENTS))]
            cands.append({"perm_id": n_drawn - 1, "labels": pl, "v_seg": v_c, "s_tok": s_c,
                          "cos_seg": cos_seg, "cos_max": max(abs(c) for c in cos_seg),
                          "dose": None})
        ortho = [c for c in cands if c["cos_max"] <= PL_COS_MAX]
        if ortho:
            break
    if not cands:
        raise ValueError("위약 후보 0 — 전 순열에서 클래스 고갈")
    fb = None
    if not ortho:  # 준직교 후보 없음 — max|cos| 최소 폴백 (기록)
        ortho = sorted(cands, key=lambda c: c["cos_max"])[:5]
        fb = f"no-ortho(min max|cos_seg|={ortho[0]['cos_max']:.2f}, n={n_drawn})"
    for c in ortho:  # dose 는 준직교 통과분만 계산 (비용 절감)
        c["dose_seg"] = seg_dose_by_segment(X_all, c["v_seg"], c["s_tok"])
        c["dose"] = seg_dose(X_all, c["v_seg"], c["s_tok"])
        c["scale_raw"] = dose_ref_seg / np.maximum(c["dose_seg"], 1e-12)
        c["scale_gap"] = float(np.max(np.abs(np.log(c["scale_raw"]))))
    ortho.sort(key=lambda c: c["scale_gap"])   # 스케일 보정이 가장 적게 필요한 후보
    sel = ortho[0]
    scale = np.clip(sel["scale_raw"], PL_SCALE_MIN, PL_SCALE_MAX)
    clipped = bool(np.any(scale != sel["scale_raw"]))
    if clipped:
        fb = (fb + " | " if fb else "") + \
             f"scale-clip({[round(x, 2) for x in sel['scale_raw'].tolist()]})"
    sel = {**sel, "dose_ref": dose_ref, "dose_ref_seg": dose_ref_seg.tolist(),
           "dose_seg": sel["dose_seg"].tolist(),
           "dose_ratio": sel["dose"] / dose_ref if dose_ref else None,
           "scale": scale.astype(np.float64), "scale_clipped": clipped,
           "fallback": fb, "n_drawn": n_drawn, "n_cand": len(cands),
           "n_ortho": len([c for c in cands if c["cos_max"] <= PL_COS_MAX])}
    print(f"  [placebo{tag}] perm={sel['perm_id']} cos_seg="
          f"{[round(c, 2) for c in sel['cos_seg']]} dose_seg="
          f"{[round(x, 1) for x in sel['dose_seg']]} (처치 "
          f"{[round(x, 1) for x in dose_ref_seg.tolist()]}) → scale="
          f"{[round(x, 2) for x in scale.tolist()]} 후보={sel['n_cand']} "
          f"ortho={sel['n_ortho']}" + (f" FALLBACK {fb}" if fb else ""), flush=True)
    return sel


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """rank AUROC (pos=fail 사영이 커야 1)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    r_pos = order[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def episode_records(roll, layer_idx: int, cap: int) -> np.ndarray:
    """rollout 의 layer 사영용 record 행렬 [n_cap, D] — truncation cap 적용 (per-record 유지,
    rollout pooling 금지 — memory feedback-no-rollout-pooling)."""
    X = roll["dit"][:, layer_idx, :]
    return X[:cap]


def episode_phase_records(roll, layer_idx: int, ph: str, dwell_cap: int) -> np.ndarray:
    """episode **전체 길이**에서 phase==ph 인 record 를 앞에서부터 dwell_cap 개까지.

    길이 통제는 phase 별 dwell cap 으로 한다 (2026-07-23 사용자 지적): episode 전역 cap 을
    먼저 걸면 cap 밖의 후반 phase(place·pull 등) record 가 통째로 소실. dwell cap 은
    성공 episode 들의 그 phase 체류 길이 스케일 — 실패의 timeout 체류 과대가중만 제어."""
    X = roll["dit"][:, layer_idx, :]
    idx = [i for i, p in enumerate(roll["phases"]) if p == ph][:dwell_cap]
    return X[idx]


def phase_dwell_caps(rolls, labels, phases) -> dict:
    """phase 별 dwell cap = 성공 episode 체류 길이(>0)의 ceil(μ+1σ). 성공 dwell 없는
    phase 는 미포함 (대조 불가 — 호출부에서 skip)."""
    caps = {}
    for ph in phases:
        dw = [sum(1 for p in r["phases"] if p == ph)
              for r, y in zip(rolls, labels) if y == 1]
        dw = [d for d in dw if d > 0]
        if dw:
            caps[ph] = int(np.ceil(np.mean(dw) + np.std(dw)))
    return caps


def gather_phase_tok(rolls, labels, layer_idx, cls, ph, dwell_cap):
    """phase dwell-cap 내 record 의 **토큰 보존** 표본 [N,T,D] (세그먼트 fit 용)."""
    out, n_eps = [], 0
    for r, y in zip(rolls, labels):
        if y != cls:
            continue
        idx = [i for i, p in enumerate(r["phases"]) if p == ph][:dwell_cap]
        if idx:
            out.append(r["tok"][idx, layer_idx])
            n_eps += 1
    return (np.concatenate(out, axis=0) if out else np.empty((0, 0, 0))), n_eps


def gather_phase(rolls, labels, layer_idx, cls, ph, dwell_cap):
    out, n_eps = [], 0
    for r, y in zip(rolls, labels):
        if y != cls:
            continue
        recs = episode_phase_records(r, layer_idx, ph, dwell_cap)
        if len(recs):
            out.append(recs)
            n_eps += 1
    return (np.concatenate(out, axis=0) if out else np.empty((0, 0))), n_eps


def gather(rolls, labels, layer_idx, cap, cls):
    out = []
    for r, y in zip(rolls, labels):
        if y == cls:
            recs = episode_records(r, layer_idx, cap)
            if len(recs):
                out.append(recs)
    return np.concatenate(out, axis=0) if out else np.empty((0, 0))


def cv_auroc(rolls, labels, layer_idx, cap, rng) -> float:
    """episode 5-fold CV record AUROC (fold 마다 train fit → held-out 사영)."""
    idx = np.arange(len(rolls))
    pos = idx[np.asarray(labels) == 0]  # fail
    neg = idx[np.asarray(labels) == 1]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [([], []) for _ in range(5)]
    for i, e in enumerate(pos):
        folds[i % 5][0].append(e)
    for i, e in enumerate(neg):
        folds[i % 5][1].append(e)
    scores_p, scores_n = [], []
    for k in range(5):
        test = set(folds[k][0]) | set(folds[k][1])
        tr = [i for i in idx if i not in test]
        Xs = gather([rolls[i] for i in tr], [labels[i] for i in tr], layer_idx, cap, 1)
        Xf = gather([rolls[i] for i in tr], [labels[i] for i in tr], layer_idx, cap, 0)
        if len(Xs) == 0 or len(Xf) == 0:
            continue
        r, _s = mean_diff(Xs, Xf)
        for i in test:
            proj = episode_records(rolls[i], layer_idx, cap) @ r
            (scores_p if labels[i] == 0 else scores_n).extend(proj.tolist())
    return auroc(np.asarray(scores_p), np.asarray(scores_n))


# ---------------------------------------------------------------------------- IO
def load_cell_rolls(manifest: Path, cell: str):
    rows = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if not p.is_absolute():
            p = REPO / p
        if p.parent.name != cell:
            continue
        rows.append({"pkl": p, "label": int(parts[1]), "scene": parts[2] if len(parts) > 2 else ""})
    if not rows:
        raise SystemExit(f"manifest 에 cell={cell} 행 없음: {manifest}")
    missing = [str(m["pkl"]) for m in rows if not m["pkl"].exists()]
    if missing:
        raise SystemExit(f"pkl 누락 {len(missing)}개: {missing[:3]}")
    rolls = []
    for m in rows:
        with open(m["pkl"], "rb") as f:
            d = pickle.load(f)
        if d.get("capture_token_mode") != FULLTOKEN_MODE:
            raise SystemExit(f"{m['pkl']}: full-token pkl 아님 ({d.get('capture_token_mode')})")
        r = load_rollout_fulltoken(d, m["pkl"], "mean")
        # 토큰 보존본 [n, T, D] (선정 layer 는 호출부에서 인덱싱) — 세그먼트 fit 용
        r["tok"] = np.stack(
            [np.asarray(rec, dtype=np.float32).mean(axis=1) for rec in d["hidden_states"]],
            axis=0)  # [n, L, T, D] → denoise mean
        r["success"] = m["label"]  # manifest 라벨 override (fit_phase_conceptor 관례)
        r["scene"] = m["scene"]
        r["episode_idx"] = int(d.get("episode_idx", -1))
        r["inference_seed"] = int(d.get("inference_seed", -1))
        r["scenario_seed"] = int(d.get("scenario_seed", -1))
        rolls.append(r)
    return rolls


def load_targets(targets_tsv: Path, cell: str):
    """annotation_t0.tsv 에서 이 cell 의 구제 대상 — pool별 (episode_idx, scenario_seed)."""
    lines = targets_tsv.read_text().splitlines()
    header = lines[0].split("\t")
    out = {"fit": [], "eval": []}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(header, ln.split("\t")))
        if r["cell"] != cell:
            continue
        out[r["pool"]].append({
            "episode_idx": int(r["episode_idx"]),
            "scenario_seed": int(r["scenario_seed"]),
            "inference_seed": int(r["inference_seed"]),
        })
    return out


def save_segment_npz(out_dir: Path, layer_blk: int, v_seg: np.ndarray, s_tok: np.ndarray,
                     meta: dict, subdir: str | None = "steer",
                     seg_mask: np.ndarray | None = None):
    """세그먼트별 방향 + 토큰별 setpoint NPZ (exp4-1 v2 규약, 2026-07-23).

    키: alpha0_v_seg [S,D] · alpha0_s_tok [T] · alpha0_seg_bounds [S,2] ·
        alpha0_seg_mask [S] (1=steer, 0=무개입 — future_only arm 용).
    serve(SetpointSteering)가 토큰 위치 → 세그먼트 → (r̂_seg, s_t) 로 적용.
    """
    d = (out_dir / subdir if subdir else out_dir) / f"dit_L{layer_blk}"
    d.mkdir(parents=True, exist_ok=True)
    if seg_mask is None:
        seg_mask = np.ones(len(SEGMENTS), dtype=np.float32)
    np.savez_compressed(
        d / "conceptors.npz",
        alpha0_v_seg=v_seg.astype(np.float32),
        alpha0_s_tok=s_tok.astype(np.float32),
        alpha0_seg_bounds=np.asarray([[lo, hi] for _n, lo, hi in SEGMENTS], dtype=np.int32),
        alpha0_seg_mask=np.asarray(seg_mask, dtype=np.float32),
    )
    (d / "metadata.json").write_text(json.dumps(
        {**meta, "selected_alpha": 0, "op": "setpoint_seg",
         "token_pool": "all_token_full", "segments": [s[0] for s in SEGMENTS],
         "seg_mask": [float(x) for x in seg_mask]}, indent=2, ensure_ascii=False))


def save_setpoint_npz(out_dir: Path, layer_blk: int, v: np.ndarray, s: float, meta: dict,
                      subdir: str | None = "steer"):
    """subdir='steer'=permanent 규약(<base>/steer/dit_L*), None=gated(phase 디렉토리가
    곧 phase 명 — serve 계약 <base>/<phase>/dit_L*, 추가 레벨 금지)."""
    d = (out_dir / subdir if subdir else out_dir) / f"dit_L{layer_blk}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "conceptors.npz",
                        alpha0_v_steer=v.astype(np.float32), alpha0_s=np.float32(s))
    (d / "metadata.json").write_text(
        json.dumps({**meta, "selected_alpha": 0}, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------- main
GATED_MIN_REC = 50     # phase 등록 최소 record/클래스 — 미달 phase 는 무개
GATED_MIN_EPS = 3
TERMINAL_PHASES = {"open-done", "insert-settle-done"}  # terminal 동치 phase 제외


def fit_gated(args, rolls, labels, out_cell: Path) -> None:
    """setM_gated / setM_gated_placebo — permanent fit 산출물(선정 layer·동결 순열) 전제.

    phase 별 (r̂_ph, s_ph) fit + 유의성 진단 (사용자: 유의성부터 판단).
    quota(record ≥GATED_MIN_REC/클래스·episode ≥GATED_MIN_EPS) 미달 phase 는 **NPZ 미생성 =
    serve 미등록 → identity(무개입)** — 실패 증거 없는 phase 에 global 방향을 씌우는 외삽
    금지 (2026-07-23 사용자 결정). 재실행 시 stale phase 디렉토리 오염 방지 위해
    setM_gated{,_placebo} 를 먼저 삭제.

    placebo (2026-07-23 리뷰 반영): permanent 동결 순열의 phase 재fit 은 phase 부분표본의
    지배축을 물려받아 준직교가 깨짐(실측 cos +0.62/−0.71) → **phase 별 독립 선택** —
    순열 후보 N_PERM_PL 에서 |cos(vs r̂_ph)|≤PL_COS_MAX 필터 후 phase-record dose-match.
    pairing 은 episode 단위(arm 간 동일 episode)라 phase 마다 순열이 달라도 유지. 위약 fit
    불가 phase 는 처치·위약 양쪽 제외(dose 대칭).
    """
    meta_perm = json.loads(
        next((out_cell / "setM_permanent" / "steer").glob("dit_L*/metadata.json")).read_text())
    blk = int(meta_perm["layer"])
    cap = int(meta_perm["cap_records"])
    cap_layers = [int(x) for x in rolls[0]["capture_layers"]]
    li = cap_layers.index(blk)
    labels_arr = np.asarray(labels)

    # 위약 순열은 phase 마다 select_placebo_seg 가 독립 추출 (같은 RNG 시드에서 시작하되
    # 필터가 phase 별 세그먼트 축 기준이라 선택 결과는 phase 마다 다르다)

    # cos 진단용 global 방향 (배포 폴백 아님 — 미달 phase 는 미등록=identity)
    z_g = np.load(next((out_cell / "setM_permanent" / "steer").glob("dit_L*/conceptors.npz")))
    # v2 permanent 는 세그먼트 방향 [S,D] — 진단용 기준축은 future 세그먼트(4/4 cell 최강)
    v_g_seg = z_g["alpha0_v_seg"].astype(np.float64)
    v_g = v_g_seg[1]
    # 재실행 stale phase 디렉토리 방지
    import shutil
    for d in ("setM_gated", "setM_gated_placebo",
              "setM_gated_future_only", "setM_gated_future_only_placebo"):
        shutil.rmtree(out_cell / d, ignore_errors=True)

    # phase 집합·dwell cap 은 episode 전체 길이 기준 (전역 cap 미적용 — 후반 phase 보존)
    phases = sorted({p for r in rolls for p in r["phases"]} - TERMINAL_PHASES)
    dwell_caps = phase_dwell_caps(rolls, labels, phases)
    # layer-sweep null 순열 (유의성 진단용, N_PERM 개)
    rng = np.random.default_rng(RNG_SEED + 777)
    null_perms = []
    for _ in range(N_PERM):
        pl = labels_arr.copy()
        rng.shuffle(pl)
        null_perms.append(pl.tolist())

    diag = []
    for ph in phases:
        if ph not in dwell_caps:
            diag.append({"phase": ph, "skipped_identity": True,
                         "skip_reason": "성공 episode dwell 없음 (대조 불가)"})
            print(f"  [{ph}] SKIP(성공 dwell 없음)", flush=True)
            continue
        dcap = dwell_caps[ph]
        Xs, ns_eps = gather_phase(rolls, labels, li, 1, ph, dcap)
        Xf, nf_eps = gather_phase(rolls, labels, li, 0, ph, dcap)
        entry = {"phase": ph, "dwell_cap": dcap,
                 "n_rec_succ": int(len(Xs)), "n_rec_fail": int(len(Xf)),
                 "n_eps_succ": ns_eps, "n_eps_fail": nf_eps}
        quota_ok = (len(Xs) >= GATED_MIN_REC and len(Xf) >= GATED_MIN_REC
                    and ns_eps >= GATED_MIN_EPS and nf_eps >= GATED_MIN_EPS)
        if quota_ok:
            v_ph, s_ph = mean_diff(Xs, Xf)
            proj_f = np.concatenate([episode_phase_records(r, li, ph, dcap) @ v_ph
                                     for r, y in zip(rolls, labels) if y == 0 and
                                     len(episode_phase_records(r, li, ph, dcap))])
            proj_s = np.concatenate([episode_phase_records(r, li, ph, dcap) @ v_ph
                                     for r, y in zip(rolls, labels) if y == 1 and
                                     len(episode_phase_records(r, li, ph, dcap))])
            a_fit = auroc(proj_f, proj_s)  # fit-표본 (참고)
            # 순열 null: 같은 phase 절차를 순열 라벨로
            null = []
            for pl in null_perms:
                try:
                    vp, _sp = mean_diff(gather_phase(rolls, pl, li, 1, ph, dcap)[0],
                                        gather_phase(rolls, pl, li, 0, ph, dcap)[0])
                except ValueError:
                    continue
                pf = np.concatenate([episode_phase_records(r, li, ph, dcap) @ vp
                                     for r, y in zip(rolls, pl) if y == 0 and
                                     len(episode_phase_records(r, li, ph, dcap))] or [np.empty(0)])
                ps_ = np.concatenate([episode_phase_records(r, li, ph, dcap) @ vp
                                      for r, y in zip(rolls, pl) if y == 1 and
                                      len(episode_phase_records(r, li, ph, dcap))] or [np.empty(0)])
                a_n = auroc(pf, ps_)
                if np.isfinite(a_n):
                    null.append(a_n)
            mu_n, sd_n = (float(np.mean(null)), float(np.std(null))) if null else (float("nan"),) * 2
            z = (a_fit - mu_n) / sd_n if null and sd_n > 1e-9 else float("nan")
            entry.update({"fit_auroc": a_fit, "null_mean": mu_n, "null_std": sd_n,
                          "perm_z": z, "cos_vs_global_future": float(v_ph @ v_g),
                          "s": s_ph, "fallback": False})
            # 배포 연산자(세그먼트)를 먼저 만들고, 위약은 **그 공간에서** 고른다
            # (pooled 기준으로 고르면 실제 개입 축의 준직교가 깨진다 — 07-23 교정)
            Xs_t = gather_phase_tok(rolls, labels, li, 1, ph, dcap)[0]
            Xf_t = gather_phase_tok(rolls, labels, li, 0, ph, dcap)[0]
            v_seg_ph, s_tok_ph, sd_ph = fit_seg_setpoint(Xs_t, Xf_t)
            ep_tok_ph = []
            for r in rolls:
                idx = [i for i, p in enumerate(r["phases"]) if p == ph][:dcap]
                ep_tok_ph.append(r["tok"][idx, li] if idx else np.empty((0, 0, 0)))
            try:
                sel = select_placebo_seg(ep_tok_ph, labels_arr, v_seg_ph, s_tok_ph,
                                         np.random.default_rng(RNG_SEED + 555), tag=f" {ph}")
            except ValueError as e:
                # 위약 성립 불가 → 처치도 이 phase 제외 (dose 대칭 — 미등록=identity)
                entry.update({"skipped_identity": True,
                              "skip_reason": f"placebo 후보 0 ({e})"})
                diag.append(entry)
                print(f"  [{ph}] SKIP(위약 불가 → 양쪽 무개입)", flush=True)
                continue
            entry.update({
                "placebo_perm_id": sel["perm_id"],
                "placebo_cos_seg_vs_treat": sel["cos_seg"],
                "placebo_cos_seg_max": sel["cos_max"],
                "placebo_dose_ratio": sel["dose_ratio"],
                "placebo_pool": {"n_drawn": sel["n_drawn"], "n_cand": sel["n_cand"],
                                 "n_ortho": sel["n_ortho"], "n_in_band": sel["n_in_band"],
                                 "fallback": sel["fallback"],
                                 "criterion": "segment-space: max_s|cos_seg|≤0.3 ∧ dose ±25%"},
            })
        else:
            # 미달 phase — NPZ 미생성 (미등록 → serve identity, 무개입. 사용자 결정 07-23)
            entry.update({"skipped_identity": True})
            diag.append(entry)
            print(f"  [{ph}] Nrec s/f={entry['n_rec_succ']}/{entry['n_rec_fail']} "
                  f"SKIP(quota 미달 → 무개입)", flush=True)
            continue
        # (세그먼트 연산자 v_seg_ph·s_tok_ph 는 위약 선택 직전에 산출됨)
        entry["seg_diag"] = sd_ph
        entry["cos_seg_vs_permanent"] = [float(v_seg_ph[i] @ v_g_seg[i])
                                         for i in range(len(SEGMENTS))]
        save_segment_npz(out_cell / "setM_gated" / ph, blk, v_seg_ph, s_tok_ph, subdir=None,
                         meta={"operator": "setM_gated", "cell": args.cell, "phase": ph,
                               "layer": blk, "dwell_cap": dcap,
                               **{k: entry[k] for k in entry if k != "phase"}})
        v_seg_pp, s_tok_pp = sel["v_seg"], sel["s_tok"]
        Xs_tp = gather_phase_tok(rolls, sel["labels"], li, 1, ph, dcap)[0]
        Xf_tp = gather_phase_tok(rolls, sel["labels"], li, 0, ph, dcap)[0]
        _v_chk, _s_chk, sd_pp = fit_seg_setpoint(Xs_tp, Xf_tp)
        check_seg_equiv(_v_chk, _s_chk, v_seg_pp, s_tok_pp, f"setM_gated_placebo[{ph}]")
        pl_scale_ph = sel["scale"]
        pl_meta_ph = {"perm_id": sel["perm_id"], "cos_seg_vs_treat": sel["cos_seg"],
                      "cos_seg_max": sel["cos_max"], "pl_cos_max": PL_COS_MAX,
                      "seg_dose": sel["dose_seg"], "seg_dose_treat": sel["dose_ref_seg"],
                      "dose_match_scale": pl_scale_ph.tolist(),
                      "scale_clipped": sel["scale_clipped"], "fallback": sel["fallback"]}
        save_segment_npz(out_cell / "setM_gated_placebo" / ph, blk, v_seg_pp, s_tok_pp,
                         subdir=None, seg_mask=pl_scale_ph,
                         meta={"operator": "setM_gated_placebo", "cell": args.cell,
                               "phase": ph, "layer": blk, **pl_meta_ph, "seg_diag": sd_pp})
        fmask = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        save_segment_npz(out_cell / "setM_gated_future_only" / ph, blk, v_seg_ph, s_tok_ph,
                         subdir=None, seg_mask=fmask,
                         meta={"operator": "setM_gated_future_only", "cell": args.cell,
                               "phase": ph, "layer": blk, "seg_diag": sd_ph})
        save_segment_npz(out_cell / "setM_gated_future_only_placebo" / ph, blk,
                         v_seg_pp, s_tok_pp, subdir=None,
                         seg_mask=np.asarray([0.0, pl_scale_ph[1], 0.0]),
                         meta={"operator": "setM_gated_future_only_placebo",
                               "cell": args.cell, "phase": ph, "layer": blk,
                               **pl_meta_ph, "seg_diag": sd_pp})
        diag.append(entry)
        print(f"  [{ph}] Nrec s/f={entry['n_rec_succ']}/{entry['n_rec_fail']} "
              f"auroc={entry['fit_auroc']:.3f} z={entry['perm_z']:.2f} "
              f"cos_gfut={entry['cos_vs_global_future']:+.2f}", flush=True)

    (out_cell / "setM_gated_diag.json").write_text(json.dumps({
        "cell": args.cell, "layer": blk, "cap_records": cap, "phases": diag,
        "quota": {"min_rec": GATED_MIN_REC, "min_eps": GATED_MIN_EPS},
        "note": "유의성(perm_z)은 fit-표본 AUROC 기준 진단 — phase 별 표본이 작아 CV 생략, "
                "배포 게이트는 quota(폴백)로만. placebo 는 permanent 동결 순열 재사용(pairing).",
    }, indent=2, ensure_ascii=False))
    print(f"[gated done] {out_cell}/setM_gated (+placebo) phases={len(phases)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="fit30 tsv (pkl\\tlabel\\tscene)")
    ap.add_argument("--cell", required=True, help="예: pq3_ppcc_bread (within-instruction fit)")
    ap.add_argument("--targets", type=Path, required=True,
                    help="annotation_t0.tsv — fit-풀 대상은 LOO, eval-풀 대상은 침범 검사")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--layers", default=None,
                    help="sweep 할 DiT 물리 layer 콤마목록 (기본: capture 전부)")
    ap.add_argument("--gated", action="store_true",
                    help="setM_gated/placebo 만 fit (permanent 산출물 전제 — 같은 layer·동결 순열)")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    rolls = load_cell_rolls(args.manifest, args.cell)
    labels = [r["success"] for r in rolls]
    if args.gated:
        fit_gated(args, rolls, labels, args.out_root / args.cell)
        return
    targets = load_targets(args.targets, args.cell)
    cap_layers = [int(x) for x in rolls[0]["capture_layers"]]
    layer_blks = ([int(x) for x in args.layers.split(",")] if args.layers else cap_layers)
    for b in layer_blks:
        if b not in cap_layers:
            raise SystemExit(f"layer {b} 는 capture {cap_layers} 에 없음")

    # ---- 침범 검사: rollout 정체성 = (scenario_seed, inference_seed). exp3 eval 의
    # seen 블록(ep0-29)은 fit 과 같은 scene·다른 inference_seed 계열(35xx000 vs 10xx000)
    # 이라 episode_idx 비교는 오탐 — 실측 대조로 확정(2026-07-22).
    fit_ids = {(r["scenario_seed"], r["inference_seed"]) for r in rolls}
    fit_scenes = {r["scenario_seed"] for r in rolls}
    bad = [t for t in targets["eval"]
           if (t["scenario_seed"], t["inference_seed"]) in fit_ids]
    if bad:
        raise SystemExit(f"eval-풀 대상이 fit 표본과 동일 rollout(설계 위반): {bad[:3]}")
    # seen(fit scene 재사용)/unseen 층화 플래그 — 집계에서 별도 보고 (scene-level leakage 표기)
    eval_targets_flagged = [
        {**t, "seen_scene": t["scenario_seed"] in fit_scenes} for t in targets["eval"]
    ]
    # fit-풀 대상은 반드시 fit rolls 안에 있어야 LOO 가 성립
    fit_pair_ids = {(r["episode_idx"], r["inference_seed"]) for r in rolls}
    missing_loo = [t for t in targets["fit"]
                   if (t["episode_idx"], t["inference_seed"]) not in fit_pair_ids]
    if missing_loo:
        raise SystemExit(f"fit-풀 대상이 fit 표본에 없음: {missing_loo[:3]}")

    # ---- truncation 표준 (memory truncation-length-standard): cap = ceil(μ+1σ) of succ 길이
    succ_lens = [r["length"] for r, y in zip(rolls, labels) if y == 1]
    if not succ_lens:
        raise SystemExit("성공 episode 0개 — fit 불가")
    cap = int(np.ceil(np.mean(succ_lens) + np.std(succ_lens)))
    cap_mean = int(np.ceil(np.mean(succ_lens)))

    n_s = sum(labels)
    print(f"[{args.cell}] rollouts={len(rolls)} succ={n_s} fail={len(rolls)-n_s} "
          f"cap={cap} (μ={cap_mean}) layers={layer_blks} "
          f"targets fit={len(targets['fit'])} eval={len(targets['eval'])}", flush=True)

    # ---- layer sweep: CV AUROC + permutation null z + bootstrap 각도 안정성
    sweep = []
    perm_labels_list = []
    labels_arr = np.asarray(labels)
    for pi in range(N_PERM):
        perm = labels_arr.copy()
        rng.shuffle(perm)  # episode-level 순열
        perm_labels_list.append(perm.tolist())
    for blk in layer_blks:
        li = cap_layers.index(blk)
        Xs = gather(rolls, labels, li, cap, 1)
        Xf = gather(rolls, labels, li, cap, 0)
        v_full, s_full = mean_diff(Xs, Xf)
        a_cv = cv_auroc(rolls, labels, li, cap, np.random.default_rng(RNG_SEED + blk))
        null = [
            cv_auroc(rolls, pl, li, cap, np.random.default_rng(RNG_SEED + blk + 7919 * (pi + 1)))
            for pi, pl in enumerate(perm_labels_list)
        ]
        null = [x for x in null if np.isfinite(x)]
        mu_n, sd_n = float(np.mean(null)), float(np.std(null))
        z = (a_cv - mu_n) / sd_n if sd_n > 1e-9 else float("nan")
        # bootstrap 각도 안정성 (episode-cluster 재표본 → r̂ 과 full-fit r̂ 의 cos)
        cos_list = []
        idx_s = [i for i in range(len(rolls)) if labels[i] == 1]
        idx_f = [i for i in range(len(rolls)) if labels[i] == 0]
        brng = np.random.default_rng(RNG_SEED + 100 + blk)
        for _ in range(N_BOOT):
            bs = list(brng.choice(idx_s, len(idx_s))) + list(brng.choice(idx_f, len(idx_f)))
            Xsb = gather([rolls[i] for i in bs], [labels[i] for i in bs], li, cap, 1)
            Xfb = gather([rolls[i] for i in bs], [labels[i] for i in bs], li, cap, 0)
            try:
                vb, _ = mean_diff(Xsb, Xfb)
                cos_list.append(float(vb @ v_full))
            except ValueError:
                continue
        sweep.append({
            "layer": blk, "cv_auroc": a_cv, "null_mean": mu_n, "null_std": sd_n,
            "perm_z": z, "boot_cos_mean": float(np.mean(cos_list)),
            "boot_cos_p05": float(np.percentile(cos_list, 5)),
            "s": s_full, "norm_mu_diff": float(np.linalg.norm(Xf.mean(0) - Xs.mean(0))),
            "n_rec_succ": int(len(Xs)), "n_rec_fail": int(len(Xf)),
        })
        print(f"  L{blk}: cv_auroc={a_cv:.3f} null={mu_n:.3f}±{sd_n:.3f} z={z:.2f} "
              f"boot_cos={np.mean(cos_list):.3f} s={s_full:.3f}", flush=True)

    # 선정: perm_z 최대 (동률 ±0.1 이내면 boot_cos 우선). SR 무접촉 — 사전등록 기록.
    best = max(sweep, key=lambda r: (round(r["perm_z"], 1), r["boot_cos_mean"]))
    blk = best["layer"]
    li = cap_layers.index(blk)
    print(f"[select] layer=L{blk} (perm_z={best['perm_z']:.2f})", flush=True)

    # ---- full-fit setM (eval-풀 대상용)
    Xs = gather(rolls, labels, li, cap, 1)
    Xf = gather(rolls, labels, li, cap, 0)
    v_full, s_full = mean_diff(Xs, Xf)

    # ---- 세그먼트 연산자 (v2 배포본, 2026-07-23): r̂_seg [S,D] + s_t [T]
    # 위약 선택보다 **먼저** 산출한다 — 위약의 준직교·dose 기준축이 바로 이 연산자이기 때문.
    Xs_tok = np.concatenate([r["tok"][:cap, li] for r, y in zip(rolls, labels) if y == 1], axis=0)
    Xf_tok = np.concatenate([r["tok"][:cap, li] for r, y in zip(rolls, labels) if y == 0], axis=0)
    v_seg, s_tok, seg_diag = fit_seg_setpoint(Xs_tok, Xf_tok)
    gaps_txt = " ".join(f"{d_['segment']}:{d_['gap_fail_minus_succ']:+.1f}"
                        for d_ in seg_diag["segments"])
    print(f"[seg] 갭 {gaps_txt} | 이동/갭 중앙 {seg_diag['move_over_gap_median']:.2f} "
          f"최대 {seg_diag['move_over_gap_max']:.2f}", flush=True)
    gate_ok = seg_diag["move_over_gap_median"] <= MOVE_GAP_RATIO_MAX

    # ---- setM_pl: 위약 순열 선택 — **세그먼트 공간** 준직교 ∧ dose-match
    # (구 pooled 기준의 실패는 select_placebo_seg docstring 참조)
    ep_tok = [r["tok"][:cap, li] for r in rolls]
    pl_sel = select_placebo_seg(ep_tok, labels_arr, v_seg, s_tok,
                                np.random.default_rng(RNG_SEED + 555))
    pl_labels = pl_sel["labels"]
    v_seg_p, s_tok_p = pl_sel["v_seg"], pl_sel["s_tok"]

    # 참고용 pooled(v1) 지표 — 배포 아님, 구 기준과의 대조 기록용
    all_rec = np.concatenate([episode_records(r, li, cap) for r in rolls], axis=0)
    dose_m = float(np.median(np.abs(all_rec @ v_full - s_full)))
    v_pool_p, s_pool_p = mean_diff(gather(rolls, pl_labels, li, cap, 1),
                                   gather(rolls, pl_labels, li, cap, 0))
    dose_pool_p = float(np.median(np.abs(all_rec @ v_pool_p - s_pool_p)))
    auroc_pool_p = auroc(np.asarray([x for r, y in zip(rolls, labels) if y == 0
                                     for x in episode_records(r, li, cap) @ v_pool_p]),
                         np.asarray([x for r, y in zip(rolls, labels) if y == 1
                                     for x in episode_records(r, li, cap) @ v_pool_p]))

    # ---- 저장 (full-fit)
    out_cell = args.out_root / args.cell
    import shutil as _sh
    for _d in ("setM_permanent_loo", "setM_permanent_placebo_loo",
               "setM_future_only_loo", "setM_future_only_placebo_loo"):
        _sh.rmtree(out_cell / _d, ignore_errors=True)   # 구 규약(steer/) 잔재 제거
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:12]
    targets_sha = hashlib.sha256(args.targets.read_bytes()).hexdigest()[:12]
    base_meta = {
        "operator": "setM_permanent", "cell": args.cell, "layer": blk, "cap_records": cap,
        "cap_mean_alt": cap_mean, "manifest_sha": manifest_sha, "targets_sha": targets_sha,
        "n_rollouts": len(rolls), "n_succ": int(n_s), "n_fail": int(len(rolls) - n_s),
        "s": s_full, "dose_median_fitset": dose_m,
        "layer_sweep_ref": "../layer_sweep.json", "rng_seed": RNG_SEED,
        "eval_targets": eval_targets_flagged,
        "n_eval_seen": sum(t["seen_scene"] for t in eval_targets_flagged),
        "n_eval_unseen": sum(not t["seen_scene"] for t in eval_targets_flagged),
        "note": "s≈0 이면 제거형(I−βr̂r̂ᵀ)과 동치 (24a §4.1); seen_scene 대상은 "
                "fit 이 같은 scene 의 다른 rollout 을 봄 — 집계에서 seen/unseen 층화",
    }
    seg_meta = {**base_meta, "operator": "setM_permanent", "seg_diag": seg_diag,
                "move_gate_ok": gate_ok, "move_gap_ratio_max": MOVE_GAP_RATIO_MAX}
    save_segment_npz(out_cell / "setM_permanent", blk, v_seg, s_tok, seg_meta)
    # future-only arm: 같은 연산자에서 state/action 세그먼트만 무개입 (mask)
    save_segment_npz(out_cell / "setM_future_only", blk, v_seg, s_tok,
                     {**seg_meta, "operator": "setM_future_only"},
                     seg_mask=np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    # 위약: 선택된 순열 라벨의 세그먼트 연산자 (v_seg_p·s_tok_p 는 선택 단계에서 산출됨).
    # 이동/갭 진단만 fit_seg_setpoint 로 다시 받는다 (수치 동일 — 검증 assert).
    Xs_tok_p = np.concatenate([r["tok"][:cap, li] for r, y in zip(rolls, pl_labels) if y == 1], axis=0)
    Xf_tok_p = np.concatenate([r["tok"][:cap, li] for r, y in zip(rolls, pl_labels) if y == 0], axis=0)
    v_chk, s_chk, seg_diag_p = fit_seg_setpoint(Xs_tok_p, Xf_tok_p)
    check_seg_equiv(v_chk, s_chk, v_seg_p, s_tok_p, "setM_permanent_placebo")
    pl_scale = pl_sel["scale"]                                   # [S] dose-match 승수
    pl_scale_fut = np.asarray([0.0, pl_scale[1], 0.0])           # future-only 위약
    pl_meta = {"perm_id": pl_sel["perm_id"], "cos_seg_vs_treat": pl_sel["cos_seg"],
               "cos_seg_max": pl_sel["cos_max"], "pl_cos_max": PL_COS_MAX,
               "seg_dose": pl_sel["dose_seg"], "seg_dose_treat": pl_sel["dose_ref_seg"],
               "dose_match_scale": pl_scale.tolist(), "scale_clipped": pl_sel["scale_clipped"],
               "pl_pool": {"n_drawn": pl_sel["n_drawn"], "n_cand": pl_sel["n_cand"],
                           "n_ortho": pl_sel["n_ortho"], "fallback": pl_sel["fallback"],
                           "criterion": "segment-space: max_s|cos_seg|≤0.3, dose 는 "
                                        "seg_mask 스케일로 처치와 정확 매칭"}}
    save_segment_npz(out_cell / "setM_permanent_placebo", blk, v_seg_p, s_tok_p,
                     {**seg_meta, "operator": "setM_permanent_placebo", **pl_meta,
                      "seg_diag": seg_diag_p}, seg_mask=pl_scale)
    save_segment_npz(out_cell / "setM_future_only_placebo", blk, v_seg_p, s_tok_p,
                     {**seg_meta, "operator": "setM_future_only_placebo", **pl_meta},
                     seg_mask=pl_scale_fut)

    # ---- 구(v1) pooled 산출물은 진단 보관용으로만 유지 (배포 금지 — 공간 불일치)
    save_setpoint_npz(out_cell / "_v1_pooled_setM", blk, v_full, s_full, base_meta)
    save_setpoint_npz(out_cell / "_v1_pooled_setM_placebo", blk, v_pool_p, s_pool_p, {
        **base_meta, "operator": "setM_permanent_placebo", "perm_id": pl_sel["perm_id"],
        "dose_median": dose_pool_p, "true_label_auroc": auroc_pool_p,
        "cos_setm": float(v_pool_p @ v_full), "pl_cos_max": PL_COS_MAX,
        "dose_ratio_vs_setm": dose_pool_p / dose_m if dose_m else None,
        "note": "pooled 지표는 진단 기록용 — 선택은 세그먼트 공간에서 수행됨",
    })

    # ---- fit-풀 대상 per-target LOO (setM + pairing 위약: 같은 순열 라벨에서 대상 제외)
    for t in targets["fit"]:
        keep = [i for i in range(len(rolls)) if rolls[i]["episode_idx"] != t["episode_idx"]]
        kl = [labels[i] for i in keep]
        kr = [rolls[i] for i in keep]
        Xs_t = np.concatenate([r["tok"][:cap, li] for r, y in zip(kr, kl) if y == 1], axis=0)
        Xf_t = np.concatenate([r["tok"][:cap, li] for r, y in zip(kr, kl) if y == 0], axis=0)
        v_seg_t, s_tok_t, sd_t = fit_seg_setpoint(Xs_t, Xf_t)
        # phase-registry 재활용: phase 이름 = ep{E} → serve 1회 기동으로 전 LOO 스위칭
        ep_ph = f"ep{t['episode_idx']}"
        save_segment_npz(out_cell / "setM_permanent_loo" / ep_ph, blk, v_seg_t, s_tok_t,
                         subdir=None,
                         meta={**seg_meta, "operator": "setM_permanent_loo",
                               "loo_episode_idx": t["episode_idx"], "seg_diag": sd_t})
        # future_only LOO 는 같은 연산자에 mask 만 다름 (추가 fit 비용 0)
        save_segment_npz(out_cell / "setM_future_only_loo" / ep_ph, blk, v_seg_t, s_tok_t,
                         subdir=None, seg_mask=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                         meta={**seg_meta, "operator": "setM_future_only_loo",
                               "loo_episode_idx": t["episode_idx"], "seg_diag": sd_t})
        # 위약도 **LOO 표본에서 독립 선택** — 전체fit 순열을 물려받아 재fit 하면 대상 1판을
        # 뺀 부분표본에서 방향이 회전해 준직교가 깨진다 (실측 max|cos_seg| bread 0.72 ·
        # drawer_right 0.47 · beer 0.30). gated 의 phase 별 독립 선택과 같은 이유.
        # pairing 은 episode 단위(arm 간 같은 대상)라 대상마다 순열이 달라도 유지된다.
        sel_t = select_placebo_seg([ep_tok[i] for i in keep], np.asarray(kl),
                                   v_seg_t, s_tok_t,
                                   np.random.default_rng(RNG_SEED + 555 + t["episode_idx"]),
                                   tag=f" LOO ep{t['episode_idx']}")
        v_seg_pt, s_tok_pt, pl_scale_t = sel_t["v_seg"], sel_t["s_tok"], sel_t["scale"]
        pl_l = sel_t["labels"]
        Xs_p = np.concatenate([r["tok"][:cap, li] for r, y in zip(kr, pl_l) if y == 1], axis=0)
        Xf_p = np.concatenate([r["tok"][:cap, li] for r, y in zip(kr, pl_l) if y == 0], axis=0)
        _vc, _sc, sd_pt = fit_seg_setpoint(Xs_p, Xf_p)
        check_seg_equiv(_vc, _sc, v_seg_pt, s_tok_pt, f"placebo_loo ep{t['episode_idx']}")
        loo_pl_meta = {"perm_id": sel_t["perm_id"], "loo_episode_idx": t["episode_idx"],
                       "dose_match_scale": pl_scale_t.tolist(),
                       "scale_clipped": sel_t["scale_clipped"],
                       "cos_seg_vs_treat": sel_t["cos_seg"], "cos_seg_max": sel_t["cos_max"],
                       "pl_pool": {"n_drawn": sel_t["n_drawn"], "n_ortho": sel_t["n_ortho"],
                                   "fallback": sel_t["fallback"]},
                       "seg_diag": sd_pt}
        save_segment_npz(out_cell / "setM_permanent_placebo_loo" / ep_ph, blk,
                         v_seg_pt, s_tok_pt, subdir=None, seg_mask=pl_scale_t,
                         meta={**seg_meta, "operator": "setM_permanent_placebo_loo",
                               **loo_pl_meta})
        save_segment_npz(out_cell / "setM_future_only_placebo_loo" / ep_ph, blk,
                         v_seg_pt, s_tok_pt, subdir=None,
                         seg_mask=np.asarray([0.0, pl_scale_t[1], 0.0]),
                         meta={**seg_meta, "operator": "setM_future_only_placebo_loo",
                               **loo_pl_meta})

    (out_cell / "layer_sweep.json").write_text(json.dumps({
        "cell": args.cell, "sweep": sweep, "selected_layer": blk,
        "selection_rule": "max perm_z (round .1 tie → boot_cos_mean)",
        "cap_records": cap, "n_perm": N_PERM, "n_boot": N_BOOT,
        "manifest_sha": manifest_sha, "rng_seed": RNG_SEED,
    }, indent=2, ensure_ascii=False))
    print(f"[done] {out_cell} — setM/setM_pl + LOO {len(targets['fit'])}쌍 "
          f"(layer L{blk})", flush=True)


if __name__ == "__main__":
    main()
