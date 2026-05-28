"""coast — COAST conceptor analyses (length-controlled, baseline only).

--fig (any subset, default all):
  spectrum : Fig.3A conceptor eigenvalue spectra C+/C-/C_steer/C_all + aperture
             sweep + effective-rank bars (window-0 mean features).
  4a       : Fig.4A per-task top-2 eigvecs of C_steer scatter + 2σ ellipses
             (first t_d per-step activations).
  4b       : Fig.4B per-task v1(C_steer) over normalized trajectory time (mean±1σ).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

_REPO = Path(__file__).resolve().parents[6]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.conceptor import (  # noqa: E402
    and_conceptor, compute_conceptor, conceptor_overlap, conceptor_quota,
    eigenvalue_spectrum, not_conceptor,
)

from core import io
from analyses._common import add_common

NAME = "coast"
HELP = "COAST conceptor figures: spectrum (3A) / fig4a / fig4b (baseline)"


def add_args(p):
    add_common(p)
    p.add_argument("--fig", nargs="+", default=["spectrum", "4a", "4b"], choices=["spectrum", "4a", "4b"])
    p.add_argument("--t-d", type=int, default=16, help="first t_d steps (4a/4b)")
    p.add_argument("--window", type=int, default=5, help="W steps for window-0 (spectrum)")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1, 2, 5, 10, 20])
    p.add_argument("--min-class-pts", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=20)


def _pr(eig):
    eig = np.clip(eig, 0, None); s1, s2 = eig.sum(), (eig ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def _eff(C):
    eig = eigenvalue_spectrum(C)
    return {"quota": conceptor_quota(C), "n_eig_gt_0.5": int((eig > 0.5).sum()),
            "participation_ratio": _pr(eig), "d": int(C.shape[0])}


def _fig_spectrum(rollouts, base, args):
    X = np.stack([r["z"][:args.window].mean(0) for r in rollouts if len(r["z"]) >= args.window])
    succ = np.asarray([r["succ"] for r in rollouts if len(r["z"]) >= args.window])
    Xs, Xf = X[succ == 1], X[succ == 0]
    print(f"window-0 (W={args.window}): succ={len(Xs)} fail={len(Xf)} d={X.shape[1]}")
    sweep = []
    for a in args.alphas:
        Cs, Cf = compute_conceptor(Xs, a), compute_conceptor(Xf, a)
        Cst = and_conceptor(Cs, not_conceptor(Cf))
        sweep.append({"alpha": a, "overlap_succ_fail": conceptor_overlap(Cs, Cf),
                      "quota_steer": conceptor_quota(Cst), "pr_steer": _pr(eigenvalue_spectrum(Cst))})
    a = args.alpha
    Cs, Cf = compute_conceptor(Xs, a), compute_conceptor(Xf, a)
    Cst = and_conceptor(Cs, not_conceptor(Cf)); Call = compute_conceptor(X, a)
    eig = {k: eigenvalue_spectrum(C) for k, C in [("s", Cs), ("f", Cf), ("st", Cst), ("all", Call)]}
    summary = {"window": args.window, "n_succ": int(len(Xs)), "n_fail": int(len(Xf)), "main_alpha": a,
               "C_success": _eff(Cs), "C_failure": _eff(Cf), "C_steer": _eff(Cst), "C_all_overall_task": _eff(Call),
               "overlap_succ_fail": conceptor_overlap(Cs, Cf), "aperture_sweep": sweep}
    print(f"  C_steer eff dims = {summary['C_steer']['n_eig_gt_0.5']} ({100*summary['C_steer']['n_eig_gt_0.5']/X.shape[1]:.2f}% of d)")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    rank = np.arange(1, len(eig["s"]) + 1)
    axes[0].semilogy(rank, np.clip(eig["s"], 1e-6, None), "-", color="#2ca02c", label="C+ (success)")
    axes[0].semilogy(rank, np.clip(eig["f"], 1e-6, None), "--", color="#d62728", label="C- (failure)")
    axes[0].semilogy(rank, np.clip(eig["st"], 1e-6, None), ":", color="#1f77b4", lw=2, label="C_steer")
    axes[0].semilogy(rank, np.clip(eig["all"], 1e-6, None), "-", color="#999999", lw=1, label="C_all")
    axes[0].axhline(0.5, color="k", ls=":", lw=0.8); axes[0].set_xlim(0, min(200, len(eig["s"])))
    axes[0].set_xlabel("eigenvalue rank"); axes[0].set_ylabel("eigenvalue (log)")
    axes[0].set_title(f"(A) conceptor spectra (alpha={a})"); axes[0].legend(fontsize=9)
    axes[1].plot([r["alpha"] for r in sweep], [r["overlap_succ_fail"] for r in sweep], "o-", color="#9467bd", label="overlap(C+,C-)")
    axes[1].axhspan(0.85, 0.95, color="#9467bd", alpha=0.12, label="COAST sweet-spot")
    axes[1].set_xscale("log"); axes[1].set_xlabel("aperture alpha"); axes[1].set_ylabel("overlap")
    axes[1].set_title("(B) aperture sweep"); axes[1].legend(fontsize=9)
    keys = ["C_success", "C_failure", "C_steer", "C_all_overall_task"]
    vals = [summary[k]["n_eig_gt_0.5"] for k in keys]
    bars = axes[2].bar(["C+\nsuccess", "C-\nfailure", "C_steer\ndisc.", "C_all\noverall"], vals,
                       color=["#2ca02c", "#d62728", "#1f77b4", "#999999"])
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
    axes[2].set_ylabel("# dims eig>0.5"); axes[2].set_title(f"(C) effective dims (of d={X.shape[1]})")
    fig.suptitle(f"COAST Fig.3A — succ/fail subspace low-rank | window-0 (W={args.window}) | {base}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_dir = args.out_root / "coast_spectrum"; out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"spectrum_{base}.png", dpi=140); plt.close(fig)
    json.dump(summary, open(out_dir / f"spectrum_{base}.json", "w"), indent=2)
    print(f"[spectrum] wrote spectrum_{base}.png/.json")


def _cov_ellipse(ax, pts, color):
    mu = pts.mean(0); cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov); o = w.argsort()[::-1]; w, v = w[o], v[:, o]
    ang = math.degrees(math.atan2(v[1, 0], v[0, 0]))
    width, height = 2 * 2 * np.sqrt(np.maximum(w, 1e-12))
    ax.add_patch(Ellipse(mu, width, height, angle=ang, edgecolor=color, facecolor="none", lw=1.6, alpha=0.9))
    ax.scatter([mu[0]], [mu[1]], s=130, marker="X", c=[color], edgecolors="black", linewidths=1.2, zorder=5)
    return mu


def _fig_4a(rollouts, names, base, args):
    X = np.concatenate([r["z"][:args.t_d] for r in rollouts])
    succ = np.concatenate([np.full(len(r["z"][:args.t_d]), r["succ"]) for r in rollouts])
    task = np.concatenate([np.full(len(r["z"][:args.t_d]), r["task"]) for r in rollouts])
    ids = sorted(np.unique(task).tolist())
    ncol = 6; nrow = math.ceil(len(ids) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow)); axes = np.atleast_1d(axes).ravel()
    summary = []
    for ax, tid in zip(axes, ids):
        m = task == tid; Xt, st = X[m], succ[m]; Xs, Xf = Xt[st == 1], Xt[st == 0]
        if len(Xs) < args.min_class_pts or len(Xf) < args.min_class_pts:
            ax.set_title(f"{names[tid]}\n(insufficient s{len(Xs)}/f{len(Xf)})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); continue
        Cst = and_conceptor(compute_conceptor(Xs, args.alpha), not_conceptor(compute_conceptor(Xf, args.alpha)))
        w, V = np.linalg.eigh(Cst); v2 = V[:, w.argsort()[::-1][:2]]
        P = (Xt - Xt.mean(0)) @ v2; Ps, Pf = P[st == 1], P[st == 0]
        ax.scatter(Ps[:, 0], Ps[:, 1], s=4, alpha=0.25, c="#2ca02c", rasterized=True)
        ax.scatter(Pf[:, 0], Pf[:, 1], s=4, alpha=0.25, c="#d62728", rasterized=True)
        gap = float(np.linalg.norm(_cov_ellipse(ax, Ps, "#2ca02c") - _cov_ellipse(ax, Pf, "#d62728")))
        ax.set_title(f"{names[tid]}\nSR={st.mean():.2f} gap={gap:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        summary.append({"task": names[tid], "succ_rate": float(st.mean()), "centroid_gap_2d": gap})
    for ax in axes[len(ids):]:
        ax.axis("off")
    fig.suptitle(f"COAST Fig.4A (baseline) — top-2 eigvecs of C_steer | α={args.alpha}, first {args.t_d} steps | {base}\n"
                 f"(label-derived axis: in-sample; steered ellipses N/A)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_dir = args.out_root / "coast_fig4a"; out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fig4a_subspace_a{args.alpha}_td{args.t_d}_{base}.png", dpi=140); plt.close(fig)
    json.dump({"alpha": args.alpha, "t_d": args.t_d, "per_task": summary},
              open(out_dir / f"fig4a_subspace_a{args.alpha}_td{args.t_d}_{base}.json", "w"), indent=2)
    print(f"[4a] wrote fig4a_subspace_a{args.alpha}_td{args.t_d}_{base}.png/.json")


def _banded(ax, rolls, v1, mu, n_bins, color):
    edges = np.linspace(0, 1, n_bins + 1); centers = 0.5 * (edges[:-1] + edges[1:])
    buckets = [[] for _ in range(n_bins)]
    for r in rolls:
        T = r["z"].shape[0]; proj = (r["z"] - mu) @ v1
        bidx = np.clip(np.searchsorted(edges, np.arange(T) / max(T - 1, 1), side="right") - 1, 0, n_bins - 1)
        for b, p in zip(bidx, proj):
            buckets[b].append(p)
    mean = np.array([np.mean(b) if b else np.nan for b in buckets])
    std = np.array([np.std(b) if b else np.nan for b in buckets]); ok = ~np.isnan(mean)
    ax.plot(centers[ok], mean[ok], "-", color=color, lw=1.8)
    ax.fill_between(centers[ok], (mean - std)[ok], (mean + std)[ok], color=color, alpha=0.18)


def _fig_4b(rollouts, names, base, args):
    ids = sorted({r["task"] for r in rollouts})
    ncol = 6; nrow = math.ceil(len(ids) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.0 * nrow), sharex=True); axes = np.atleast_1d(axes).ravel()
    for ax, tid in zip(axes, ids):
        rolls = [r for r in rollouts if r["task"] == tid]
        succ_r = [r for r in rolls if r["succ"] == 1]; fail_r = [r for r in rolls if r["succ"] == 0]
        Xs = np.concatenate([r["z"][:args.t_d] for r in succ_r]) if succ_r else np.empty((0, 1))
        Xf = np.concatenate([r["z"][:args.t_d] for r in fail_r]) if fail_r else np.empty((0, 1))
        if len(Xs) < args.min_class_pts or len(Xf) < args.min_class_pts:
            ax.set_title(f"{names[tid]} (insufficient)", fontsize=8); continue
        w, V = np.linalg.eigh(and_conceptor(compute_conceptor(Xs, args.alpha), not_conceptor(compute_conceptor(Xf, args.alpha))))
        v1 = V[:, int(w.argmax())]; mu = np.concatenate([Xs, Xf]).mean(0)
        if ((Xs - mu) @ v1).mean() < ((Xf - mu) @ v1).mean():
            v1 = -v1
        _banded(ax, succ_r, v1, mu, args.n_bins, "#2ca02c"); _banded(ax, fail_r, v1, mu, args.n_bins, "#d62728")
        ax.set_title(f"{names[tid]} SR={len(succ_r)/len(rolls):.2f}", fontsize=8)
        ax.axhline(0, color="grey", lw=0.6, ls=":"); ax.tick_params(labelsize=6)
    for ax in axes[len(ids):]:
        ax.axis("off")
    fig.suptitle(f"COAST Fig.4B (baseline) — v1(C_steer) over norm. trajectory time (mean±1σ) | α={args.alpha} | {base}", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out_dir = args.out_root / "coast_fig4b"; out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fig4b_v1_traj_a{args.alpha}_td{args.t_d}_{base}.png", dpi=140); plt.close(fig)
    print(f"[4b] wrote fig4b_v1_traj_a{args.alpha}_td{args.t_d}_{base}.png")


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    base = Path(args.cache).stem.replace("pooled_all_", "")
    if "spectrum" in args.fig:
        _fig_spectrum(rollouts, base, args)
    if "4a" in args.fig:
        _fig_4a(rollouts, names, base, args)
    if "4b" in args.fig:
        _fig_4b(rollouts, names, base, args)
