"""snapshots — full-dim t-SNE Mahalanobis snapshots per temporal scheme.

rows=scheme, cols=time; colored task_succ_fail (18 task colors + black FAIL).
whitenTask (non-circular) and/or whitenTaskFail (fail-as-group). Optional 2σ
ellipses (per-task success + shared FAIL).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from core import geometry, io
from core.projection import run_tsne
from core.schemes import SCHEME_LABEL, SCHEMES, iter_scheme
from analyses._common import add_common

NAME = "snapshots"
HELP = "t-SNE Mahalanobis snapshots (scheme × time) colored task/succ/fail"

FAIL_LABEL = 18
WL_TAG = {"task": "whitenTask", "task_succ_fail": "whitenTaskFail"}


def add_args(p):
    add_common(p)
    p.add_argument("--t-d", type=int, default=16)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--targets", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--whiten-labels", nargs="+", default=["task", "task_succ_fail"],
                   choices=["task", "task_succ_fail"])
    p.add_argument("--ellipses", action="store_true")


def _ellipse(ax, pts, color, ls, lw=1.4):
    if len(pts) < 3:
        return
    mu, width, height, ang = geometry.cov_ellipse_2d(pts)
    ax.add_patch(Ellipse(mu, width, height, angle=ang, edgecolor=color, facecolor="none",
                         lw=lw, ls=ls, alpha=0.95, zorder=4))
    ax.scatter([mu[0]], [mu[1]], s=55, marker="X", c=[color], edgecolors="black", linewidths=0.7, zorder=5)


def _draw(ax, xy, task_sf, title, ellipses):
    cmap = mpl.cm.get_cmap("tab20", 20)
    fail = task_sf == FAIL_LABEL
    ax.scatter(xy[~fail, 0], xy[~fail, 1], s=5, alpha=0.55, c=task_sf[~fail], cmap=cmap,
               vmin=0, vmax=19, rasterized=True)
    ax.scatter(xy[fail, 0], xy[fail, 1], s=5, alpha=0.4, c="black", rasterized=True)
    if ellipses:
        for t in np.unique(task_sf[~fail]):
            _ellipse(ax, xy[task_sf == t], cmap(int(t) % 20), "-")
        _ellipse(ax, xy[fail], "black", "--", lw=1.8)
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def _pick(snaps, targets):
    xs = np.array([s["x_norm"] for s in snaps])
    out, used = [], set()
    for t in targets:
        i = int(np.argmin(np.abs(xs - t)))
        if i not in used:
            out.append(snaps[i]); used.add(i)
    return out


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    out_dir = args.out_root / "temporal_schemes" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ncol = len(args.targets); wls = args.whiten_labels
    esfx = "_ellipse" if args.ellipses else ""
    grids = {wl: plt.subplots(len(SCHEMES), ncol, figsize=(3.0 * ncol, 2.9 * len(SCHEMES))) for wl in wls}

    for si, scheme in enumerate(SCHEMES):
        picked = _pick(list(iter_scheme(rollouts, scheme, t_d=args.t_d)), args.targets)
        print(f"== {scheme}: {len(picked)} snapshots ==")
        rows = {wl: plt.subplots(1, ncol, figsize=(3.4 * ncol, 3.4)) for wl in wls}
        for j, snap in enumerate(picked):
            X, succ, task = snap["X"], snap["succ"], snap["task"]
            task_sf = np.where(succ == 1, task, FAIL_LABEL)
            for wl in wls:
                lab = task if wl == "task" else task_sf
                Xw = geometry.whiten_by(X, lab, reg=args.cov_reg)
                xy = run_tsne(Xw.astype(np.float32), metric="euclidean", perplexity=30,
                              random_state=args.seed, init="pca")
                ttl = f"t~{snap['x_norm']:.2f} s{snap['n_succ']}/f{snap['n_fail']}"
                _draw(np.atleast_1d(rows[wl][1]).ravel()[j], xy, task_sf, ttl, args.ellipses)
                _draw(grids[wl][1][si, j], xy, task_sf, f"{scheme} {ttl}", args.ellipses)
        for wl in wls:
            rfig, raxes = rows[wl]
            for ax in np.atleast_1d(raxes).ravel()[len(picked):]:
                ax.axis("off")
            rfig.suptitle(f"{SCHEME_LABEL[scheme]} — t-SNE Maha [{WL_TAG[wl]}, full-dim] (black=FAIL)", fontsize=11)
            rfig.tight_layout(rect=(0, 0, 1, 0.92))
            rfig.savefig(out_dir / f"snapshots_{scheme}_{WL_TAG[wl]}{esfx}.png", dpi=140)
            plt.close(rfig)

    for wl in wls:
        gfig, _ = grids[wl]
        gfig.suptitle(f"Temporal schemes — t-SNE Mahalanobis [{WL_TAG[wl]}, FULL-DIM] (task_succ_fail; black=FAIL)\n"
                      f"rows=scheme, cols=normalized time", fontsize=12)
        gfig.tight_layout(rect=(0, 0, 1, 0.96))
        gp = out_dir / f"snapshots_grid_{WL_TAG[wl]}{esfx}.png"
        gfig.savefig(gp, dpi=130); plt.close(gfig)
        print(f"wrote {gp}")
