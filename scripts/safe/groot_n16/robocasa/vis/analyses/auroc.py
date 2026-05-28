"""auroc — 3 per-task×time AUROC tables (abs/mean/window) with permutation test.

cell = held-out CV logistic AUROC (fail-vs-success) on task-whitened FULL-DIM
features; */** = permutation-test p<.05/.01 vs chance; outlined = significant;
grey = <min-class minority. TSV/JSON carry null_mean/std/z/p + counts.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from core import geometry, io, metrics, schemes
from analyses._common import add_common

NAME = "auroc"
HELP = "per-task × time AUROC tables (abs/mean/window) + permutation significance"


def add_args(p):
    add_common(p)
    p.add_argument("--frames", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    p.add_argument("--windows", nargs="+", default=["0-4", "5-8", "9-12", "13-16"])
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--min-class", type=int, default=5)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--pca-dim", type=int, default=30)
    p.add_argument("--n-folds", type=int, default=5)


def _compute(rollouts, ids, mode, specs, args):
    sh = (len(ids), len(specs))
    A = np.full(sh, np.nan); NM = np.full(sh, np.nan); NS = np.full(sh, np.nan)
    Z = np.full(sh, np.nan); P = np.full(sh, np.nan)
    N = np.zeros((len(ids), len(specs), 2), dtype=int)
    for cj, sp in enumerate(specs):
        built = schemes.build_at(rollouts, mode, sp)
        if built is None:
            continue
        X, succ, task = built
        Xw = geometry.whiten_by(X, task, reg=args.cov_reg)
        for ti, tid in enumerate(ids):
            m = task == tid; st = succ[m]
            ns, nf = int((st == 1).sum()), int((st == 0).sum())
            N[ti, cj] = (ns, nf)
            if min(ns, nf) < args.min_class:
                continue
            real, null = metrics.cv_auroc(Xw[m], 1 - st, n_perm=args.n_perm,
                                          k=args.n_folds, pca_dim=args.pca_dim, seed=args.seed)
            if real is None or null is None or len(null) == 0:
                continue
            A[ti, cj] = real
            NM[ti, cj] = float(null.mean()); NS[ti, cj] = float(null.std())
            Z[ti, cj] = (real - null.mean()) / (null.std() + 1e-9)
            P[ti, cj] = (1 + int((null >= real).sum())) / (len(null) + 1)
        print(f"  {mode} {sp}: done")
    return A, NM, NS, Z, P, N


def _stars(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def _render(A, NM, NS, Z, P, N, ids, names, sr, specs, mode, col_labels, out_dir, n_perm):
    order = sorted(range(len(ids)), key=lambda i: sr[ids[i]])
    cmap = mpl.cm.get_cmap("viridis").copy(); cmap.set_bad("lightgrey")
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(specs), 0.42 * len(ids) + 1.4))
    ax.imshow(np.ma.masked_invalid(A[order]), cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    im = ax.images[0]
    ax.set_xticks(range(len(specs))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{names[ids[i]]} (SR{sr[ids[i]]:.2f})" for i in order], fontsize=8)
    ax.set_xlabel("진행도 (time index)")
    ax.set_title(f"Per-task CV AUROC — {mode}  (cell=AUROC; */**=perm-test p<.05/.01 vs chance; "
                 f"outlined=sig; grey=insufficient minority; n_perm={n_perm})")
    for ti, i in enumerate(order):
        for cj in range(len(specs)):
            if not np.isfinite(A[i, cj]):
                continue
            v = A[i, cj]
            ax.text(cj, ti, f"{v:.2f}{_stars(P[i, cj])}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < 0.78 else "black")
            if P[i, cj] < 0.05:
                ax.add_patch(Rectangle((cj - 0.5, ti - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.8))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CV AUROC")
    fig.tight_layout()
    fig.savefig(out_dir / f"auroc_table_{mode}.png", dpi=150); plt.close(fig)

    with (out_dir / f"auroc_table_{mode}.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        hdr = ["task", "SR"]
        for c in col_labels:
            hdr += [f"{c}_auroc", f"{c}_nullmean", f"{c}_nullstd", f"{c}_z", f"{c}_p", f"{c}_ns", f"{c}_nf"]
        w.writerow(hdr)
        for i in order:
            row = [names[ids[i]], f"{sr[ids[i]]:.2f}"]
            for cj in range(len(specs)):
                f3 = lambda M: "" if np.isnan(M[i, cj]) else f"{M[i, cj]:.3f}"
                row += [f3(A), f3(NM), f3(NS), f3(Z), f3(P), int(N[i, cj, 0]), int(N[i, cj, 1])]
            w.writerow(row)


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    ids = sorted({r["task"] for r in rollouts})
    sr = {t: float(np.mean([r["succ"] for r in rollouts if r["task"] == t])) for t in ids}
    out_dir = args.out_root / "auroc_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = [("abs", args.frames, [f"f{t}" for t in args.frames]),
             ("mean", args.frames, [f"≤{t}" for t in args.frames]),
             ("window", args.windows, [f"[{w}]" for w in args.windows])]
    all_json = {"n_perm": args.n_perm}
    for mode, specs, col_labels in plans:
        print(f"== {mode}  specs={specs}  (n_perm={args.n_perm}) ==")
        A, NM, NS, Z, P, N = _compute(rollouts, ids, mode, specs, args)
        _render(A, NM, NS, Z, P, N, ids, names, sr, specs, mode, col_labels, out_dir, args.n_perm)
        print(f"  wrote auroc_table_{mode}.png + .tsv  (sig p<.05: {int(np.nansum(P < 0.05))})")
        g = lambda M: {names[ids[i]]: [None if np.isnan(M[i, j]) else round(float(M[i, j]), 4)
                                       for j in range(len(specs))] for i in range(len(ids))}
        all_json[mode] = {"specs": list(map(str, specs)), "col_labels": col_labels,
                          "success_rate": {names[t]: sr[t] for t in ids},
                          "auroc": g(A), "null_mean": g(NM), "null_std": g(NS), "z": g(Z), "p": g(P),
                          "n_succ_alive": {names[ids[i]]: [int(N[i, j, 0]) for j in range(len(specs))] for i in range(len(ids))},
                          "n_fail_alive": {names[ids[i]]: [int(N[i, j, 1]) for j in range(len(specs))] for i in range(len(ids))}}
    json.dump(all_json, open(out_dir / "auroc_tables.json", "w"), indent=2)
    print(f"wrote {out_dir}/auroc_tables.json")
