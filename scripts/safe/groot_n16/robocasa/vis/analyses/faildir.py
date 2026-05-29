"""faildir — genuine pre-failure direction vs length/timestep reactivity.

Length-matched early-warning AUROC, failure-direction vs time-direction cosine,
dimensionality of the failure signal, and the projection histogram (4-panel).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core import io
from analyses._common import add_common

NAME = "faildir"
HELP = "genuine pre-failure direction vs length/time reactivity (length-matched)"


def add_args(p):
    add_common(p)
    p.add_argument("--ts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10, 12, 16])
    p.add_argument("--t0", type=int, default=4)
    p.add_argument("--min-class", type=int, default=6)
    p.add_argument("--folds", type=int, default=5)


def _cummean(z, t):
    return z[: t + 1].mean(0)


def _wt_auroc(X, yfail, task, k, seed, n_comp=None):
    aucs = []
    for t in np.unique(task):
        m = task == t; y = yfail[m]
        if y.sum() < k or (len(y) - y.sum()) < k:
            continue
        Xt = X[m]
        if n_comp:
            Xt = PCA(n_components=min(n_comp, Xt.shape[0] - 1, Xt.shape[1]), random_state=seed).fit_transform(Xt)
        fold = []
        for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(Xt, y):
            sc = StandardScaler().fit(Xt[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xt[tr]), y[tr])
            if len(np.unique(y[te])) > 1:
                fold.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(Xt[te]))[:, 1]))
        if fold:
            aucs.append(np.mean(fold))
    return (float(np.mean(aucs)), len(aucs)) if aucs else (None, 0)


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    Fs, Ts, fails, tasks = [], [], [], []
    for r in rollouts:
        z = r["z"]
        for t in range(len(z)):
            Fs.append(z[t]); Ts.append(t); fails.append(1 - r["succ"]); tasks.append(r["task"])
    F = np.asarray(Fs); Tabs = np.asarray(Ts, float)
    print(f"per-step points: {F.shape}")
    ridge = Ridge(alpha=1.0).fit(F, Tabs)
    u_time = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-12)
    time_r2 = float(ridge.score(F, Tabs))
    print(f"time-direction R^2: {time_r2:.3f}")

    curve = []
    for t in args.ts:
        alive = [r for r in rollouts if len(r["z"]) > t]
        if len(alive) < 50:
            continue
        task_a = np.array([r["task"] for r in alive]); yfail_a = np.array([1 - r["succ"] for r in alive])
        Xsingle = np.stack([r["z"][t] for r in alive]); Xcum = np.stack([_cummean(r["z"], t) for r in alive])
        a_single, nts = _wt_auroc(Xsingle, yfail_a, task_a, args.min_class, args.seed)
        a_cum, _ = _wt_auroc(Xcum, yfail_a, task_a, args.min_class, args.seed)
        a_timeproj, _ = _wt_auroc((Xsingle @ u_time).reshape(-1, 1), yfail_a, task_a, args.min_class, args.seed)
        curve.append({"t": t, "n_alive": len(alive), "frac_fail": float(yfail_a.mean()),
                      "auroc_single": a_single, "auroc_cum": a_cum, "auroc_timeproj": a_timeproj, "n_tasks": nts})
        print(f"  t={t:>2} alive={len(alive):>4} fail%={yfail_a.mean()*100:4.1f} single={a_single} cum={a_cum} timeproj={a_timeproj}")

    Xep = np.stack([r["z"].mean(0) for r in rollouts])
    yfail_ep = np.array([1 - r["succ"] for r in rollouts]); task_ep = np.array([r["task"] for r in rollouts])
    a_ep, _ = _wt_auroc(Xep, yfail_ep, task_ep, args.min_class, args.seed)
    Len = np.array([len(r["z"]) for r in rollouts], float)
    a_len, _ = _wt_auroc(Len.reshape(-1, 1), yfail_ep, task_ep, args.min_class, args.seed)
    print(f"naive episode-mean within-task AUROC={a_ep}; length-only AUROC={a_len}")

    t0 = args.t0
    alive0 = [r for r in rollouts if len(r["z"]) > t0]
    task0 = np.array([r["task"] for r in alive0]); yfail0 = np.array([1 - r["succ"] for r in alive0])
    X0 = np.stack([_cummean(r["z"], t0) for r in alive0])
    X0c = X0.copy()
    for t in np.unique(task0):
        X0c[task0 == t] -= X0c[task0 == t].mean(0)
    clf0 = LogisticRegression(max_iter=3000).fit(StandardScaler().fit_transform(X0c), yfail0)
    u_fail = clf0.coef_[0] / (np.linalg.norm(clf0.coef_[0]) + 1e-12)
    cos_ft = float(abs(np.dot(u_fail, u_time)))
    Xresid = X0 - np.outer(X0 @ u_time, u_time)
    a_full, _ = _wt_auroc(X0, yfail0, task0, args.min_class, args.seed)
    a_resid, _ = _wt_auroc(Xresid, yfail0, task0, args.min_class, args.seed)
    print(f"[t0={t0}] cos(u_fail,u_time)={cos_ft:.3f} AUROC full={a_full} resid={a_resid}")

    dims = [1, 2, 3, 5, 10, 20, 50]
    dim_auc = [_wt_auroc(X0, yfail0, task0, args.min_class, args.seed, n_comp=k)[0] for k in dims]
    proj0 = X0c @ u_fail

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ts = [c["t"] for c in curve]
    axes[0, 0].axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    axes[0, 0].plot(ts, [c["auroc_cum"] for c in curve], "o-", color="#1f77b4", label="cumulative z[0:t+1]")
    axes[0, 0].plot(ts, [c["auroc_single"] for c in curve], "s-", color="#2ca02c", label="single frame z[t]")
    axes[0, 0].plot(ts, [c["auroc_timeproj"] for c in curve], "^:", color="#d62728", label="time-direction proj only")
    if a_ep is not None:
        axes[0, 0].axhline(a_ep, color="purple", ls="-.", lw=1, label=f"naive episode-mean ({a_ep:.2f})")
    if a_len is not None:
        axes[0, 0].axhline(a_len, color="black", ls=":", lw=1, label=f"length-only ({a_len:.2f})")
    axes[0, 0].set_xlabel("fixed timestep t"); axes[0, 0].set_ylabel("within-task CV AUROC (eventual failure)")
    axes[0, 0].set_title("A. Length-matched early-warning"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylim(0.45, max(0.95, (a_ep or 0.6) + 0.05))
    tp0 = (curve[args.ts.index(t0)]["auroc_timeproj"] if t0 in args.ts else np.nan) or 0
    axes[0, 1].bar(["full", "residual\n(time removed)", "time-proj\nonly"], [a_full or 0, a_resid or 0, tp0],
                   color=["#1f77b4", "#2ca02c", "#d62728"])
    axes[0, 1].axhline(0.5, color="grey", ls="--", lw=1); axes[0, 1].set_ylim(0.45, 0.8)
    axes[0, 1].set_ylabel("within-task AUROC")
    axes[0, 1].set_title(f"B. Failure dir vs time dir @ t0={t0}\ncos={cos_ft:.3f} (time R^2={time_r2:.2f})")
    axes[1, 0].plot(dims, [d or np.nan for d in dim_auc], "o-", color="#ff7f0e")
    axes[1, 0].axhline(0.5, color="grey", ls="--", lw=1); axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("# PCA components (log)"); axes[1, 0].set_ylabel("within-task CV AUROC")
    axes[1, 0].set_title(f"C. Dimensionality of failure signal @ t0={t0}")
    axes[1, 1].hist(proj0[yfail0 == 0], bins=40, alpha=0.6, color="#2ca02c", density=True, label="eventual success")
    axes[1, 1].hist(proj0[yfail0 == 1], bins=40, alpha=0.6, color="#d62728", density=True, label="eventual failure")
    axes[1, 1].set_xlabel(f"projection on u_fail (task-centered, t0={t0})"); axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("D. Early failure direction separates outcomes"); axes[1, 1].legend(fontsize=8)
    fig.suptitle("Genuine pre-failure direction vs length/timestep reactivity (seen18 GR00T-N1.6)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir = args.out_root / "failure_direction"; out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "failure_direction.png", dpi=140); plt.close(fig)
    json.dump({"time_r2": time_r2, "naive_episode_mean_auroc": a_ep, "length_only_auroc": a_len, "t0": t0,
               "cos_ufail_utime": cos_ft, "auroc_full_t0": a_full, "auroc_resid_notime_t0": a_resid,
               "early_warning_curve": curve, "dim_auroc": dict(zip(dims, dim_auc))},
              open(out_dir / "failure_direction.json", "w"), indent=2)
    print(f"wrote {out_dir}/failure_direction.png + .json")
