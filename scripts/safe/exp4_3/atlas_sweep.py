#!/usr/bin/env python3
"""exp4-3 분리도 지도: (model, cell) 당 layer × phase 셀의 분리도 통계 산출.

셀마다 3개 통계 (같은 데이터·같은 길이통제 위에서):
  var_z   분산분리 — C_steer=C_succ∧¬C_fail 의 held-out(성공 record) R-가중 이득 vs
          episode-라벨 순열 null z (exp4_1/conceptor_layer_sweep 재사용, α는 C_succ 이득 밴드)
  mean_z  평균분리 — mean-diff r̂ 사영의 held-out CV AUROC vs 순열 null z
          (exp4_1/fit_mean_diff 재사용)
  quota   COAST 참조 — in-sample tr(C_succ∧¬C_fail)/D, α=10, 통제 없이 phase pool
          (COAST A.10.2 Stage1·Fig.7A 와 직접 비교 가능한 형태)

길이통제: phase="__global__" 는 성공 길이 ceil(μ+1σ) cap, phase별은 dwell cap
(성공 episode 의 그 phase 체류 ceil(μ+1σ)) — per-record 유지, rollout pooling 금지.

phase축 분산분리(신규 로직)를 제외한 모든 수학은 exp4_1 에서 import.

사용 (승준 노드):
  python atlas_sweep.py --model n15 --cell pq3_ppcc_bread --manifest <fit tsv> \
      --out <...>/atlas/n15/pq3_ppcc_bread.json [--capture-layers 0,2,4,8,10,12,15]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/steer"))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/steer/exp4_1"))

from src.conceptor import and_conceptor, compute_conceptor, not_conceptor  # noqa: E402
from atlas_loader import load_cell_rolls  # noqa: E402
from kl_decomp import HELD_FRAC as KL_HELD_FRAC, kl_split  # noqa: E402
from conceptor_layer_sweep import (  # noqa: E402
    N_PERM as VAR_N_PERM,
    RNG_SEED as VAR_SEED,
    HELD_FRAC,
    c_steer,
    gain,
    pick_alpha,
    split_data,
    split_score,
)
from fit_mean_diff import (  # noqa: E402
    GATED_MIN_EPS,
    GATED_MIN_REC,
    TERMINAL_PHASES,
    auroc,
    cv_auroc,
    episode_records,
    episode_phase_records,
    gather,
    gather_phase,
    mean_diff,
    phase_dwell_caps,
)

GLOBAL_PHASE = "__global__"
QUOTA_ALPHA = 10.0  # COAST A.10.2 Stage1 고정값
MEAN_SEED = 424101  # fit_mean_diff.RNG_SEED 와 동일 스트림


# --------------------------------------------------------------- phase축 분산분리 (신규)
def _phase_split_data(rolls, labels, li, ph, dcap, rng):
    """episode 80/20 split → (Xs_tr, Xf_tr, X_held_succ) — phase 부분표본·dwell cap."""
    idx_s = [i for i in range(len(rolls)) if labels[i] == 1
             and len(episode_phase_records(rolls[i], li, ph, dcap))]
    idx_f = [i for i in range(len(rolls)) if labels[i] == 0
             and len(episode_phase_records(rolls[i], li, ph, dcap))]
    if len(idx_s) < 3 or len(idx_f) < 2:
        return None
    rng.shuffle(idx_s)
    rng.shuffle(idx_f)
    h_s = idx_s[: max(1, int(len(idx_s) * HELD_FRAC))]
    h_f = idx_f[: max(1, int(len(idx_f) * HELD_FRAC))]
    held = set(h_s) | set(h_f)
    tr = [i for i in range(len(rolls)) if i not in held]
    Xs, _ = gather_phase([rolls[i] for i in tr], [labels[i] for i in tr], li, 1, ph, dcap)
    Xf, _ = gather_phase([rolls[i] for i in tr], [labels[i] for i in tr], li, 0, ph, dcap)
    if len(Xs) == 0 or len(Xf) == 0:
        return None
    X_held = np.concatenate([episode_phase_records(rolls[i], li, ph, dcap) for i in h_s], axis=0)
    return Xs, Xf, X_held


def _phase_var_score(rolls, labels, li, ph, dcap, rng, alpha) -> float:
    got = _phase_split_data(rolls, labels, li, ph, dcap, rng)
    if got is None:
        return float("nan")
    Xs, Xf, X_held = got
    return gain(c_steer(Xs, Xf, alpha), X_held)


def var_sep(rolls, labels, li, blk, ph, dcap, perms) -> dict:
    """분산분리 z — phase=None(global)이면 exp4_1 원본 경로, 아니면 phase 래퍼."""
    if ph is None:
        got = split_data(rolls, labels, li, dcap, np.random.default_rng(VAR_SEED + blk))
        if got is None:
            return {"var_z": None, "var_gain": None, "alpha": None}
        Xs_tr, _Xf, X_held = got
        alpha, g_succ, sat = pick_alpha(Xs_tr, X_held)
        g = split_score(rolls, labels, li, dcap, np.random.default_rng(VAR_SEED + blk), alpha)
        null = [split_score(rolls, pl, li, dcap,
                            np.random.default_rng(VAR_SEED + blk + 31 * (pi + 1)), alpha)
                for pi, pl in enumerate(perms)]
    else:
        got = _phase_split_data(rolls, labels, li, ph, dcap, np.random.default_rng(VAR_SEED + blk))
        if got is None:
            return {"var_z": None, "var_gain": None, "alpha": None}
        Xs_tr, _Xf, X_held = got
        alpha, g_succ, sat = pick_alpha(Xs_tr, X_held)
        g = _phase_var_score(rolls, labels, li, ph, dcap,
                             np.random.default_rng(VAR_SEED + blk), alpha)
        null = [_phase_var_score(rolls, pl, li, ph, dcap,
                                 np.random.default_rng(VAR_SEED + blk + 31 * (pi + 1)), alpha)
                for pi, pl in enumerate(perms)]
    null = [x for x in null if np.isfinite(x)]
    if not null or not np.isfinite(g):
        return {"var_z": None, "var_gain": None, "alpha": alpha}
    mu, sd = float(np.mean(null)), float(np.std(null))
    return {"var_z": (g - mu) / sd if sd > 1e-12 else None, "var_gain": g,
            "var_null_mean": mu, "var_null_std": sd, "alpha": alpha,
            "alpha_saturated": bool(sat), "gain_succ_heldout": g_succ}


# ------------------------------------------------------------------------- 평균분리
def mean_sep(rolls, labels, li, blk, ph, dcap, perms) -> dict:
    if ph is None:
        a = cv_auroc(rolls, labels, li, dcap, np.random.default_rng(MEAN_SEED + blk))
        null = [cv_auroc(rolls, pl, li, dcap,
                         np.random.default_rng(MEAN_SEED + blk + 7919 * (pi + 1)))
                for pi, pl in enumerate(perms)]
    else:
        a = _phase_fit_auroc(rolls, labels, li, ph, dcap)
        null = [_phase_fit_auroc(rolls, pl, li, ph, dcap) for pl in perms]
    null = [x for x in null if np.isfinite(x)]
    if not null or not np.isfinite(a):
        return {"mean_z": None, "mean_auroc": a if np.isfinite(a) else None}
    mu, sd = float(np.mean(null)), float(np.std(null))
    return {"mean_z": (a - mu) / sd if sd > 1e-12 else None, "mean_auroc": a,
            "mean_null_mean": mu, "mean_null_std": sd}


def _phase_fit_auroc(rolls, labels, li, ph, dcap) -> float:
    """phase 부분표본 fit-표본 AUROC (phase당 episode 수가 적어 CV 대신 — fit_mean_diff --gated 규약)."""
    try:
        v, _s = mean_diff(gather_phase(rolls, labels, li, 1, ph, dcap)[0],
                          gather_phase(rolls, labels, li, 0, ph, dcap)[0])
    except ValueError:
        return float("nan")
    pf, ps = [], []
    for r, y in zip(rolls, labels):
        rec = episode_phase_records(r, li, ph, dcap)
        if len(rec):
            (pf if y == 0 else ps).append(rec @ v)
    if not pf or not ps:
        return float("nan")
    return auroc(np.concatenate(pf), np.concatenate(ps))


# ------------------------------------------------- 통합 진단: KL 분해 (평균+분산 성분)
def _kl_records(rolls, labels, li, ph, dcap, rng):
    """episode split → (Xs_tr, Xf_tr, Xs_he, Xf_he). phase=None 이면 global cap."""
    def recs(i):
        return (episode_records(rolls[i], li, dcap) if ph is None
                else episode_phase_records(rolls[i], li, ph, dcap))
    idx_s = [i for i in range(len(rolls)) if labels[i] == 1 and len(recs(i))]
    idx_f = [i for i in range(len(rolls)) if labels[i] == 0 and len(recs(i))]
    if len(idx_s) < 3 or len(idx_f) < 3:
        return None
    rng.shuffle(idx_s)
    rng.shuffle(idx_f)
    h_s = idx_s[: max(1, int(len(idx_s) * KL_HELD_FRAC))]
    h_f = idx_f[: max(1, int(len(idx_f) * KL_HELD_FRAC))]
    t_s = [i for i in idx_s if i not in set(h_s)]
    t_f = [i for i in idx_f if i not in set(h_f)]
    if not t_s or not t_f:
        return None
    cat = lambda ids: np.concatenate([recs(i) for i in ids], axis=0)  # noqa: E731
    return cat(t_s), cat(t_f), cat(h_s), cat(h_f)


def kl_diag(rolls, labels, li, blk, ph, dcap, perms) -> dict:
    """KL 총량·평균성분·분산성분 + 각각의 episode-라벨 순열 null z (통합 진단 축)."""
    got = _kl_records(rolls, labels, li, ph, dcap, np.random.default_rng(VAR_SEED + 5 + blk))
    if got is None:
        return {}
    obs = kl_split(*got)
    if obs is None:
        return {}
    nulls = {"kl_total": [], "kl_mean_term": [], "kl_cov_term": []}
    for pi, pl in enumerate(perms):
        g = _kl_records(rolls, pl, li, ph, dcap,
                        np.random.default_rng(VAR_SEED + 5 + blk + 131 * (pi + 1)))
        if g is None:
            continue
        r = kl_split(*g)
        if r is None:
            continue
        for k in nulls:
            nulls[k].append(r[k])
    out = {k: obs[k] for k in ("kl_total", "kl_mean_term", "kl_cov_term",
                               "mean_frac", "k", "lam_s", "lam_f")}
    for key, zname in (("kl_total", "kl_z"), ("kl_mean_term", "kl_mean_z"),
                       ("kl_cov_term", "kl_cov_z")):
        v = [x for x in nulls[key] if np.isfinite(x)]
        if len(v) >= 5:
            mu, sd = float(np.mean(v)), float(np.std(v))
            out[zname] = (obs[key] - mu) / sd if sd > 1e-12 else None
            out[zname + "_null"] = mu
        else:
            out[zname] = None
    return out


# --------------------------------------------------------------------- COAST quota
def coast_quota(rolls, labels, li, ph, dcap) -> float:
    """in-sample tr(C_steer)/D, α=10, **길이통제 없이** phase pool (COAST 원본 규약)."""
    if ph is None:
        Xs = np.concatenate([r["dit"][:, li, :] for r, y in zip(rolls, labels) if y == 1], axis=0)
        Xf = np.concatenate([r["dit"][:, li, :] for r, y in zip(rolls, labels) if y == 0], axis=0)
    else:
        big = 10**9  # phase 필터만, dwell cap 미적용 (COAST 는 통제 없음)
        Xs, _ = gather_phase(rolls, labels, li, 1, ph, big)
        Xf, _ = gather_phase(rolls, labels, li, 0, ph, big)
    if len(Xs) == 0 or len(Xf) == 0:
        return float("nan")
    C = and_conceptor(compute_conceptor(Xs, QUOTA_ALPHA),
                      not_conceptor(compute_conceptor(Xf, QUOTA_ALPHA)))
    return float(np.trace(C) / C.shape[0])


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="n15|n16|pi05|cosmos (라벨용)")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--capture-layers", default=None,
                    help="pkl 에 capture_layers 없을 때 명시 (콤마)")
    ap.add_argument("--layers", default=None, help="sweep 대상 물리 layer 제한 (콤마)")
    args = ap.parse_args()

    cl_override = ([int(x) for x in args.capture_layers.split(",")]
                   if args.capture_layers else None)
    rolls = load_cell_rolls(args.manifest, args.cell, cl_override)
    labels = [r["success"] for r in rolls]
    labels_arr = np.asarray(labels)
    cap_layers = rolls[0]["capture_layers"]
    blks = ([int(x) for x in args.layers.split(",")] if args.layers else cap_layers)

    succ_len = [r["length"] for r, y in zip(rolls, labels) if y == 1]
    if not succ_len:
        raise SystemExit(f"{args.cell}: 성공 episode 0 — atlas 산출 불가")
    glob_cap = int(np.ceil(np.mean(succ_len) + np.std(succ_len)))

    phases = sorted({p for r in rolls for p in r["phases"]} - TERMINAL_PHASES)
    dwell = phase_dwell_caps(rolls, labels, phases)

    rng = np.random.default_rng(VAR_SEED + 999)
    perms = []
    for _ in range(VAR_N_PERM):
        pl = labels_arr.copy()
        rng.shuffle(pl)
        perms.append(pl.tolist())

    n_s = int(sum(labels))
    print(f"[{args.model}/{args.cell}] rollouts={len(rolls)} succ={n_s} fail={len(rolls)-n_s} "
          f"D={rolls[0]['dit'].shape[2]} layers={blks} global_cap={glob_cap} "
          f"phases={ {p: dwell.get(p) for p in phases} }", flush=True)

    cells = []
    for blk in blks:
        li = cap_layers.index(blk)
        for ph in [None] + phases:
            ph_name = GLOBAL_PHASE if ph is None else ph
            dcap = glob_cap if ph is None else dwell.get(ph)
            if ph is not None and dcap is None:
                cells.append({"model": args.model, "cell": args.cell, "layer": blk,
                              "phase": ph_name, "skip_reason": "성공 dwell 없음"})
                continue
            if ph is None:
                ns = len(gather(rolls, labels, li, dcap, 1))
                nf = len(gather(rolls, labels, li, dcap, 0))
                eps_s = sum(1 for r, y in zip(rolls, labels) if y == 1)
                eps_f = len(rolls) - eps_s
            else:
                Xs_, eps_s = gather_phase(rolls, labels, li, 1, ph, dcap)
                Xf_, eps_f = gather_phase(rolls, labels, li, 0, ph, dcap)
                ns, nf = len(Xs_), len(Xf_)
            try:  # quota 도 공분산 기반 → N<2 phase 에서 크래시 가능(try 블록 밖이라 별도 격리)
                quota_val = coast_quota(rolls, labels, li, ph, dcap)
            except Exception:
                quota_val = None
            rec = {"model": args.model, "cell": args.cell, "layer": blk, "phase": ph_name,
                   "dwell_cap": dcap, "n_rec_s": int(ns), "n_rec_f": int(nf),
                   "n_eps_s": int(eps_s), "n_eps_f": int(eps_f),
                   "quota": quota_val}
            quota_ok = (ns >= GATED_MIN_REC and nf >= GATED_MIN_REC
                        and eps_s >= GATED_MIN_EPS and eps_f >= GATED_MIN_EPS)
            if ph is None or quota_ok:
                try:  # degenerate sub-split(held-out/permutation 에서 N<2) 이 한 phase 를
                    rec.update(var_sep(rolls, labels, li, blk, ph, dcap, perms))  # 죽여도 셀 전체는
                    rec.update(mean_sep(rolls, labels, li, blk, ph, dcap, perms))  # 완성되게 격리
                    rec.update(kl_diag(rolls, labels, li, blk, ph, dcap, perms))
                except Exception as e:
                    rec["skip_reason"] = f"metric 계산 실패(N 부족 등): {type(e).__name__}: {e}"
            else:
                rec["skip_reason"] = f"quota 미달 (rec {ns}/{nf}, eps {eps_s}/{eps_f})"
            cells.append(rec)
            vz, mz = rec.get("var_z"), rec.get("mean_z")
            kz, mf = rec.get("kl_z"), rec.get("mean_frac")
            print(f"  L{blk:<3} {ph_name:16s} var_z={vz if vz is None else round(vz,2)!s:>6} "
                  f"mean_z={mz if mz is None else round(mz,2)!s:>6} "
                  f"kl_z={kz if kz is None else round(kz,2)!s:>6} "
                  f"mfrac={mf if mf is None else round(mf,2)!s:>5} "
                  f"quota={'None' if rec['quota'] is None else format(rec['quota'], '.4f')} n={ns}/{nf}"
                  + (f"  [{rec['skip_reason']}]" if rec.get("skip_reason") else ""), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "cell": args.cell, "manifest": str(args.manifest),
        "capture_layers": cap_layers, "D": int(rolls[0]["dit"].shape[2]),
        "n_rollouts": len(rolls), "n_succ": n_s, "global_cap": glob_cap,
        "dwell_caps": dwell, "quota_alpha": QUOTA_ALPHA, "n_perm": VAR_N_PERM,
        "feature_kind": rolls[0].get("feature_kind"), "cells": cells,
    }, indent=2, ensure_ascii=False))
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
