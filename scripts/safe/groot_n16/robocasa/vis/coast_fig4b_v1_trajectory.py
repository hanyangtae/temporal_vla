"""COAST Figure 4B reproduction (baseline-only) for seen18 GR00T N1.6 RoboCasa.

COAST Fig.4B: "Projection onto v1(C_steer) over normalized trajectory time
(means ± 1σ)". For each task, v1 = top eigenvector of the contrastive conceptor
C_steer = C_success AND NOT C_failure (the single most success-vs-failure
discriminative direction). Every rollout's per-step activation is projected onto
v1 and plotted against normalized trajectory time; success and failure rollouts
are aggregated into mean ± 1σ bands.

Caveats (same as fig4a):
  * Baseline rollouts only -> only baseline-success/failure bands (no steered).
  * v1 is label-derived, so the two bands separate by construction; the
    informative part is the TEMPORAL shape — WHEN/HOW failure departs from the
    success band over the rollout.

C_steer (hence v1) is fit on the first t_d env steps (length-controlled), but the
projection is plotted over the FULL normalized trajectory of every rollout.

Outputs (analysis/visualizations/coast_fig4b/):
  fig4b_v1_traj_a{alpha}_td{t_d}_<base>.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
from src.conceptor import and_conceptor, compute_conceptor, not_conceptor  # noqa: E402


def parse_args():
    run_root = (
        REPO_ROOT / "outputs/eval/robocasa/groot_n16"
        / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache",
        type=Path,
        default=run_root / "analysis" / "feature_cache" / "pooled_all_hmean_dmean.npz",
    )
    p.add_argument("--out-root", type=Path, default=run_root / "analysis" / "visualizations")
    p.add_argument("--t-d", type=int, default=16, help="fit C_steer on first t_d steps")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--n-bins", type=int, default=20, help="normalized-time bins")
    p.add_argument("--min-class-pts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def reconstruct(cache):
    c = np.load(cache, allow_pickle=True)
    feats = c["feats"].astype(np.float64)
    step_idx, rollout_idx = c["step_idx"], c["rollout_idx"]
    ep_success, ep_task = c["ep_success"], c["ep_task_id"]
    names = {int(t): str(n) for t, n in zip(c["ep_task_id"], c["task_names"])}
    order = np.lexsort((step_idx, rollout_idx))
    rid = rollout_idx[order]
    uniq = np.unique(rollout_idx)
    starts = np.searchsorted(rid, uniq, side="left")
    ends = np.searchsorted(rid, uniq, side="right")
    rollouts = []
    for r, a, b in zip(uniq, starts, ends):
        z = feats[order[a:b]]  # [T, D] step-sorted
        rollouts.append({"z": z, "succ": int(ep_success[int(r)]), "task": int(ep_task[int(r)])})
    return rollouts, names


def banded(ax, rolls, v1, mu, n_bins, color, label):
    """Pool per-step v1-projections across rollouts into normalized-time bins; mean±1σ."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    buckets = [[] for _ in range(n_bins)]
    for r in rolls:
        T = r["z"].shape[0]
        proj = (r["z"] - mu) @ v1
        nt = np.arange(T) / max(T - 1, 1)
        bidx = np.clip(np.searchsorted(edges, nt, side="right") - 1, 0, n_bins - 1)
        for b, p in zip(bidx, proj):
            buckets[b].append(p)
    mean = np.array([np.mean(b) if b else np.nan for b in buckets])
    std = np.array([np.std(b) if b else np.nan for b in buckets])
    ok = ~np.isnan(mean)
    ax.plot(centers[ok], mean[ok], "-", color=color, lw=1.8, label=label)
    ax.fill_between(centers[ok], (mean - std)[ok], (mean + std)[ok], color=color, alpha=0.18)


def main():
    args = parse_args()
    base = args.cache.stem.replace("pooled_all_", "")
    rollouts, names = reconstruct(args.cache)
    ids = sorted({r["task"] for r in rollouts})

    ncol = 6
    nrow = math.ceil(len(ids) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.0 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, tid in zip(axes, ids):
        rolls = [r for r in rollouts if r["task"] == tid]
        succ_r = [r for r in rolls if r["succ"] == 1]
        fail_r = [r for r in rolls if r["succ"] == 0]
        # fit C_steer on first t_d activations
        Xs = np.concatenate([r["z"][: args.t_d] for r in succ_r]) if succ_r else np.empty((0, 1024))
        Xf = np.concatenate([r["z"][: args.t_d] for r in fail_r]) if fail_r else np.empty((0, 1024))
        if len(Xs) < args.min_class_pts or len(Xf) < args.min_class_pts:
            ax.set_title(f"{names[tid]} (insufficient)", fontsize=8)
            continue
        Cs = compute_conceptor(Xs, args.alpha)
        Cf = compute_conceptor(Xf, args.alpha)
        w, V = np.linalg.eigh(and_conceptor(Cs, not_conceptor(Cf)))
        v1 = V[:, int(w.argmax())]
        mu = np.concatenate([Xs, Xf]).mean(0)
        # orient so success mean projection is higher (readability)
        if ((Xs - mu) @ v1).mean() < ((Xf - mu) @ v1).mean():
            v1 = -v1
        banded(ax, succ_r, v1, mu, args.n_bins, "#2ca02c", "success")
        banded(ax, fail_r, v1, mu, args.n_bins, "#d62728", "failure")
        ax.set_title(f"{names[tid]}  SR={len(succ_r)/len(rolls):.2f}", fontsize=8)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.tick_params(labelsize=6)
    for ax in axes[len(ids):]:
        ax.axis("off")
    for ax in axes[(nrow - 1) * ncol:]:
        ax.set_xlabel("norm. trajectory time", fontsize=8)
    for k in range(0, len(axes), ncol):
        axes[k].set_ylabel("proj onto v1(C_steer)", fontsize=8)
    handles = [
        plt.Line2D([], [], color="#2ca02c", lw=2, label="baseline success (mean±1σ)"),
        plt.Line2D([], [], color="#d62728", lw=2, label="baseline failure (mean±1σ)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(
        f"COAST Fig.4B (baseline only) — v1(C_steer) over normalized trajectory time  |  "
        f"α={args.alpha}, C_steer fit on first {args.t_d} steps  |  {base}\n"
        f"(v1 is label-derived: bands separate by construction; read the TEMPORAL shape — when failure departs)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_dir = args.out_root / "coast_fig4b"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"fig4b_v1_traj_a{args.alpha}_td{args.t_d}_{base}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
