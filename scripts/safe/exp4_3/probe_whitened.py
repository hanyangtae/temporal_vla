#!/usr/bin/env python3
"""exp4-3: raw mean-diff vs whitened mean-diff(LDA) vs QDA 판별력 — 24d 문서 연결·검증.

24d 핵심: succ/fail 판별 이득은 '차원'이 아니라 '방향의 질(whitening)'. rank-1 whitened
mean-diff(Σ⁻¹δ, LDA)가 raw mean-diff(setM)를 이기고, 다차원·비선형은 LDA 를 못 넘는다.

이 프로브는 셀 × layer 로 CV AUROC 3종을 낸다 (같은 top-k 부분공간·shrinkage·순열 null):
  raw    = normalize(μf−μs) 사영 (setM 방향)
  lda    = Σ_within⁻¹(μf−μs) 사영 (whitened rank-1) — 24d 1순위
  qda    = 2차 판별식 (평균차 + 공분산차 모두) — 공분산차가 판별에 실제 쓸모 있는지 검정
           (24d: 비선형≈선형이면 qda−lda≈0 이어야 함 = 공분산차는 non-discriminative)
whitening 이득 = lda − raw. qda 이득 = qda − lda.
연결 검증: lda 판별력이 atlas 의 kl_mean_z 와 같은 layer 에서 peak 인지.

사용 (승준): python probe_whitened.py --model n15 --cell <c> --manifest <tsv> --out <json>
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/steer/exp4_1"))

from atlas_loader import load_cell_rolls  # noqa: E402
from kl_decomp import TOP_K, _shrunk_cov, _subspace  # noqa: E402
from fit_mean_diff import (  # noqa: E402
    auroc, episode_records,
)

N_PERM = 30
RNG = 424103


def _dir_auroc(rolls, labels, li, cap, kind, k, rng):
    """episode 5-fold CV — kind ∈ {raw, lda, qda}. 부분공간 top-k 에서 fit."""
    idx = np.arange(len(rolls))
    pos = idx[np.asarray(labels) == 0]  # fail
    neg = idx[np.asarray(labels) == 1]  # succ
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [([], []) for _ in range(5)]
    for i, e in enumerate(pos):
        folds[i % 5][0].append(int(e))
    for i, e in enumerate(neg):
        folds[i % 5][1].append(int(e))
    sp, sn = [], []
    for kf in range(5):
        test = set(folds[kf][0]) | set(folds[kf][1])
        tr = [i for i in idx if i not in test]
        Xs = np.concatenate([episode_records(rolls[i], li, cap) for i in tr if labels[i] == 1], axis=0)
        Xf = np.concatenate([episode_records(rolls[i], li, cap) for i in tr if labels[i] == 0], axis=0)
        if len(Xs) < 20 or len(Xf) < 20:
            continue
        V = _subspace(np.concatenate([Xs, Xf], axis=0), k)
        ps, pf = Xs @ V, Xf @ V
        mus, muf = ps.mean(0), pf.mean(0)
        d = muf - mus
        if kind == "raw":
            w = d / (np.linalg.norm(d) + 1e-9)
            score = lambda X: (X @ V) @ w  # noqa: E731
        elif kind == "lda":
            Sw = 0.5 * (_shrunk_cov(ps)[0] + _shrunk_cov(pf)[0])  # pooled within
            w = np.linalg.solve(Sw, d)
            score = lambda X: (X @ V) @ w  # noqa: E731
        else:  # qda: log-lik ratio (fail − succ), 2차식
            Ss, _ = _shrunk_cov(ps)
            Sf, _ = _shrunk_cov(pf)
            Ss_i, Sf_i = np.linalg.inv(Ss), np.linalg.inv(Sf)
            _, lds = np.linalg.slogdet(Ss)
            _, ldf = np.linalg.slogdet(Sf)

            def score(X, Ss_i=Ss_i, Sf_i=Sf_i, mus=mus, muf=muf, lds=lds, ldf=ldf):
                P = X @ V
                qs = np.einsum("ni,ij,nj->n", P - mus, Ss_i, P - mus) + lds
                qf = np.einsum("ni,ij,nj->n", P - muf, Sf_i, P - muf) + ldf
                return qs - qf  # 클수록 fail 다움
        for i in test:
            proj = score(episode_records(rolls[i], li, cap))
            (sp if labels[i] == 0 else sn).extend(np.asarray(proj).ravel().tolist())
    return auroc(np.asarray(sp), np.asarray(sn)) if sp and sn else float("nan")


def _z(rolls, labels, li, cap, kind, k):
    a = _dir_auroc(rolls, labels, li, cap, kind, k, np.random.default_rng(RNG + li))
    if not np.isfinite(a):
        return None, None
    la = np.asarray(labels)
    null = []
    for pi in range(N_PERM):
        rr = np.random.default_rng(RNG + li + 977 * (pi + 1))
        pl = la.copy(); rr.shuffle(pl)
        v = _dir_auroc(rolls, pl.tolist(), li, cap, kind, k,
                       np.random.default_rng(RNG + li + 13 * (pi + 1)))
        if np.isfinite(v):
            null.append(v)
    if len(null) < 5:
        return a, None
    mu, sd = float(np.mean(null)), float(np.std(null))
    return a, ((a - mu) / sd if sd > 1e-12 else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--capture-layers", default=None)
    ap.add_argument("--layers", default=None, help="probe 대상 물리 layer 제한 (콤마)")
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()
    cl = [int(x) for x in args.capture_layers.split(",")] if args.capture_layers else None
    rolls = load_cell_rolls(args.manifest, args.cell, cl)
    labels = [r["success"] for r in rolls]
    cap_layers = rolls[0]["capture_layers"]
    blks = ([int(x) for x in args.layers.split(",")] if args.layers else cap_layers)
    succ_len = [r["length"] for r, y in zip(rolls, labels) if y == 1]
    cap = int(np.ceil(np.mean(succ_len) + np.std(succ_len)))
    print(f"[{args.model}/{args.cell}] rollouts={len(rolls)} succ={sum(labels)} "
          f"cap={cap} k={args.k} layers={cap_layers}", flush=True)
    cells = []
    for blk in blks:
        li = cap_layers.index(blk)
        row = {"model": args.model, "cell": args.cell, "layer": blk, "cap": cap, "k": args.k}
        for kind in ("raw", "lda", "qda"):
            a, z = _z(rolls, labels, li, cap, kind, args.k)
            row[f"{kind}_auroc"], row[f"{kind}_z"] = a, z
        row["whiten_gain"] = (row["lda_auroc"] - row["raw_auroc"]
                              if row["lda_auroc"] and row["raw_auroc"] else None)
        row["qda_gain"] = (row["qda_auroc"] - row["lda_auroc"]
                           if row["qda_auroc"] and row["lda_auroc"] else None)
        cells.append(row)
        print(f"  L{blk:<3} raw={row['raw_auroc']:.3f}(z{_fz(row['raw_z'])}) "
              f"lda={row['lda_auroc']:.3f}(z{_fz(row['lda_z'])}) "
              f"qda={row['qda_auroc']:.3f}(z{_fz(row['qda_z'])}) "
              f"whiten+{_fg(row['whiten_gain'])} qda+{_fg(row['qda_gain'])}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "cell": args.cell,
                                    "cap": cap, "k": args.k, "n_perm": N_PERM,
                                    "cells": cells}, indent=2))
    print(f"[done] {args.out}", flush=True)


def _fz(z):
    return "n/a" if z is None else f"{z:+.1f}"


def _fg(g):
    return "n/a" if g is None else f"{g:+.3f}"


if __name__ == "__main__":
    main()
