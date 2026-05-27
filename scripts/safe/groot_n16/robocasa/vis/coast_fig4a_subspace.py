"""COAST Figure 4A reproduction (baseline-only) for seen18 GR00T N1.6 RoboCasa.

Per task: fit the contrastive conceptor C_steer = C_success AND NOT C_failure on
per-step activations, take its top-2 eigenvectors (the subspace that maximally
separates success from failure), and scatter every per-step activation projected
onto those 2 directions, with a 2-sigma covariance ellipse + centroid for
baseline-success (green) and baseline-failure (red). One subplot per task.

IMPORTANT differences from the paper:
  * We have ONLY baseline rollouts, so only the SOLID (baseline) success/failure
    ellipses are drawn. The DASHED steered ellipses require actually running COAST
    steering and re-collecting rollouts (not done here).
  * The top-2 eigenvectors of C_steer are DERIVED from the success/failure labels,
    so this view is label-supervised and in-sample — separation is shown by
    construction (the "circular" caveat). The honest held-out separability is the
    CV AUROC (~0.6 length-controlled), not this picture.

Length control: only the first t_d env steps of each rollout contribute per-step
activations, so success/failure are sampled from the same early temporal support
(failures otherwise run to the 45-step timeout). Activation = mean over the K
denoising steps and H action-horizon tokens (the cached pre-velocity feature).

Outputs (analysis/visualizations/coast_fig4a/):
  fig4a_subspace_a{alpha}_td{t_d}_<base>.png
  fig4a_subspace_a{alpha}_td{t_d}_<base>.json   per-task 2D centroid gap + eig info
"""

from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    p.add_argument("--t-d", type=int, default=16, help="use first t_d env steps per rollout")
    p.add_argument("--alpha", type=float, default=10.0, help="conceptor aperture")
    p.add_argument("--min-class-pts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def reconstruct(cache, t_d):
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
    X, succ, task = [], [], []
    for r, a, b in zip(uniq, starts, ends):
        rows = order[a:b][:t_d]
        X.append(feats[rows])
        succ.append(np.full(len(rows), int(ep_success[int(r)])))
        task.append(np.full(len(rows), int(ep_task[int(r)])))
    return np.concatenate(X), np.concatenate(succ), np.concatenate(task), names


def cov_ellipse(ax, pts, color, ls, label):
    """2-sigma covariance ellipse + centroid marker for 2D points."""
    mu = pts.mean(0)
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    order = w.argsort()[::-1]
    w, v = w[order], v[:, order]
    angle = math.degrees(math.atan2(v[1, 0], v[0, 0]))
    width, height = 2 * 2 * np.sqrt(np.maximum(w, 1e-12))  # 2-sigma full axes
    ax.add_patch(Ellipse(mu, width, height, angle=angle, edgecolor=color,
                         facecolor="none", lw=1.6, ls=ls, alpha=0.9))
    ax.scatter([mu[0]], [mu[1]], s=130, marker="X", c=[color], edgecolors="black",
               linewidths=1.2, zorder=5, label=label)
    return mu


def main():
    args = parse_args()
    base = args.cache.stem.replace("pooled_all_", "")
    X, succ, task, names = reconstruct(args.cache, args.t_d)
    ids = sorted(np.unique(task).tolist())
    print(f"per-step activations (first {args.t_d} steps): {X.shape}, {len(ids)} tasks")

    ncol = 6
    nrow = math.ceil(len(ids) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    summary = []
    for ax, tid in zip(axes, ids):
        m = task == tid
        Xt, st = X[m], succ[m]
        Xs, Xf = Xt[st == 1], Xt[st == 0]
        if len(Xs) < args.min_class_pts or len(Xf) < args.min_class_pts:
            ax.set_title(f"{names[tid]}\n(insufficient: s{len(Xs)}/f{len(Xf)})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            summary.append({"task": names[tid], "skipped": True, "n_succ": int(len(Xs)), "n_fail": int(len(Xf))})
            continue
        Cs = compute_conceptor(Xs, args.alpha)
        Cf = compute_conceptor(Xf, args.alpha)
        Csteer = and_conceptor(Cs, not_conceptor(Cf))
        w, V = np.linalg.eigh(Csteer)
        v2 = V[:, w.argsort()[::-1][:2]]  # top-2 eigenvectors
        mu_all = Xt.mean(0)
        P = (Xt - mu_all) @ v2  # [n, 2]
        Ps, Pf = P[st == 1], P[st == 0]
        ax.scatter(Ps[:, 0], Ps[:, 1], s=4, alpha=0.25, c="#2ca02c", rasterized=True)
        ax.scatter(Pf[:, 0], Pf[:, 1], s=4, alpha=0.25, c="#d62728", rasterized=True)
        mu_s = cov_ellipse(ax, Ps, "#2ca02c", "-", "baseline success")
        mu_f = cov_ellipse(ax, Pf, "#d62728", "-", "baseline failure")
        gap = float(np.linalg.norm(mu_s - mu_f))
        ax.set_title(f"{names[tid]}\nSR={st.mean():.2f}  centroid gap={gap:.2f}", fontsize=8)
        ax.set_xlabel("Disc. PC1", fontsize=7); ax.set_ylabel("Disc. PC2", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        summary.append({"task": names[tid], "n_succ_pts": int(len(Xs)), "n_fail_pts": int(len(Xf)),
                        "succ_rate": float(st.mean()), "centroid_gap_2d": gap,
                        "top2_eigvals_Csteer": [float(x) for x in np.sort(w)[::-1][:2]]})
    for ax in axes[len(ids):]:
        ax.axis("off")
    # one shared legend
    handles = [
        plt.Line2D([], [], marker="X", color="w", markerfacecolor="#2ca02c", markeredgecolor="k", markersize=10, label="baseline success (solid 2σ)"),
        plt.Line2D([], [], marker="X", color="w", markerfacecolor="#d62728", markeredgecolor="k", markersize=10, label="baseline failure (solid 2σ)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(
        f"COAST Fig.4A (baseline only) — top-2 eigvecs of C_steer per task  |  α={args.alpha}, "
        f"first {args.t_d} steps  |  {base}\n"
        f"(label-derived axis: in-sample separation by construction; steered/dashed ellipses N/A — no steering run)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_dir = args.out_root / "coast_fig4a"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"fig4a_subspace_a{args.alpha}_td{args.t_d}_{base}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    json.dump({"alpha": args.alpha, "t_d": args.t_d, "per_task": summary},
              open(out_dir / f"fig4a_subspace_a{args.alpha}_td{args.t_d}_{base}.json", "w"), indent=2)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
