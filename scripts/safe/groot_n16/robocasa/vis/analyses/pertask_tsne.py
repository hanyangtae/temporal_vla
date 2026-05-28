"""pertask_tsne — within-task succ/fail t-SNE per task for tf/tp schemes.

For each scheme and a few time snapshots, whiten all rollouts by task (full-dim,
non-circular) then t-SNE EACH task's rollouts, colored succ(green)/fail(red) with
2σ ellipses. Answers: within a task, do succ/fail separate? (grid rows=task cols=time)
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from core import geometry, io
from core.projection import run_tsne
from core.schemes import SCHEME_LABEL, iter_scheme
from analyses._common import add_common

NAME = "pertask_tsne"
HELP = "within-task succ/fail t-SNE per task (tf/tp), grid rows=task cols=time"

SF = {1: "#2ca02c", 0: "#d62728"}


def add_args(p):
    add_common(p)
    p.add_argument("--schemes", nargs="+", default=["tf", "tp"])
    p.add_argument("--t-d", type=int, default=16)
    p.add_argument("--targets", type=float, nargs="+", default=[0.0, 0.33, 0.67, 1.0])
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--min-class", type=int, default=5)


def _ellipse(ax, pts, color):
    if len(pts) < 3:
        if len(pts):
            ax.scatter([pts[:, 0].mean()], [pts[:, 1].mean()], s=70, marker="X", c=[color],
                       edgecolors="black", linewidths=0.7, zorder=5)
        return
    mu, width, height, ang = geometry.cov_ellipse_2d(pts)
    ax.add_patch(Ellipse(mu, width, height, angle=ang, edgecolor=color, facecolor="none", lw=1.6, alpha=0.9, zorder=4))
    ax.scatter([mu[0]], [mu[1]], s=70, marker="X", c=[color], edgecolors="black", linewidths=0.7, zorder=5)


def _draw(ax, xy, succ, title):
    for o in (1, 0):
        m = succ == o
        ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.6, c=SF[o]); _ellipse(ax, xy[m], SF[o])
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def _pick(snaps, targets):
    xs = np.array([s["x_norm"] for s in snaps]); out, used = [], set()
    for t in targets:
        i = int(np.argmin(np.abs(xs - t)))
        if i not in used:
            out.append(snaps[i]); used.add(i)
    return out


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    ids = sorted({r["task"] for r in rollouts})
    sr = {t: float(np.mean([r["succ"] for r in rollouts if r["task"] == t])) for t in ids}
    order = sorted(ids, key=lambda t: sr[t])
    out_dir = args.out_root / "per_task_tsne"; out_dir.mkdir(parents=True, exist_ok=True)
    ncol = len(args.targets)
    for scheme in args.schemes:
        snaps = _pick(list(iter_scheme(rollouts, scheme, t_d=args.t_d)), args.targets)
        whitened = [geometry.whiten_by(s["X"], s["task"], reg=args.cov_reg) for s in snaps]
        (out_dir / scheme).mkdir(exist_ok=True)
        gfig, gaxes = plt.subplots(len(order), ncol, figsize=(2.9 * ncol, 2.7 * len(order)))
        gaxes = np.atleast_2d(gaxes)
        for ti, tid in enumerate(order):
            rfig, raxes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.3)); raxes = np.atleast_1d(raxes).ravel()
            for cj, (snap, Xw) in enumerate(zip(snaps, whitened)):
                mask = snap["task"] == tid; succ = snap["succ"][mask]
                ns, nf = int((succ == 1).sum()), int((succ == 0).sum())
                if min(ns, nf) < args.min_class:
                    for ax in (gaxes[ti, cj], raxes[cj]):
                        ax.set_title(f"t~{snap['x_norm']:.2f} s{ns}/f{nf} (skip)", fontsize=7)
                        ax.set_xticks([]); ax.set_yticks([])
                    continue
                xy = run_tsne(Xw[mask].astype(np.float32), metric="euclidean",
                              perplexity=min(30, max(5, mask.sum() // 3)), random_state=args.seed, init="pca")
                ttl = f"t~{snap['x_norm']:.2f} s{ns}/f{nf}"
                _draw(gaxes[ti, cj], xy, succ, f"{names[tid][:14]} {ttl}")
                _draw(raxes[cj], xy, succ, ttl)
            rfig.suptitle(f"{names[tid]} (SR{sr[tid]:.2f}) — {SCHEME_LABEL[scheme]} — within-task succ(green)/fail(red)", fontsize=10)
            rfig.tight_layout(rect=(0, 0, 1, 0.9))
            rfig.savefig(out_dir / scheme / f"task_{names[tid]}.png", dpi=130); plt.close(rfig)
        gfig.suptitle(f"Within-task succ/fail t-SNE — {SCHEME_LABEL[scheme]} (task-whiten, full-dim)\n"
                      f"rows=task (sorted by SR), cols=normalized time; 2σ ellipses", fontsize=12)
        gfig.tight_layout(rect=(0, 0, 1, 0.97))
        gp = out_dir / f"per_task_tsne_{scheme}_grid.png"
        gfig.savefig(gp, dpi=120); plt.close(gfig)
        print(f"wrote {gp}")
