"""html3d — interactive 3D HTML of within-task & all-task succ/fail geometry.

Per snapshot (feature mode × spec), self-contained plotly HTML with a task
dropdown: ALL TASKS (global PCA-3D, succ by task color / fail black + 2σ) and
per-task (x=mean-diff succ↔fail, y,z=PC⊥; 2σ ellipsoids). Labels show held-out
CV AUROC and a shuffle baseline. Modes: mean/concat/window/abs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from core import geometry, io, metrics, plot3d
from core.schemes import feat_at
from analyses._common import add_common

NAME = "html3d"
HELP = "interactive 3D HTML (rotatable) of succ/fail latent geometry per task"

SF = {1: "#2ca02c", 0: "#d62728"}
TAB20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728",
         "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2",
         "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]


def add_args(p):
    add_common(p)
    p.add_argument("--modes", nargs="+", default=["mean", "concat", "window"],
                   choices=["abs", "mean", "concat", "window"])
    p.add_argument("--cum-frames", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    p.add_argument("--abs-frames", type=int, nargs="+", default=[])
    p.add_argument("--windows", nargs="+", default=["0-4", "5-8", "9-12", "13-16"])
    p.add_argument("--concat-max-dim", type=int, default=10000)
    p.add_argument("--min-class", type=int, default=5)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--online", action="store_true")


def _fmt(v):
    return "NA" if v is None else f"{v:.2f}"


def _render(rollouts, names, ids, mode, spec, label_short, banner, out_path, args):
    sel, vecs = [], []
    for r in rollouts:
        v = feat_at(r["z"], mode, spec)
        if v is None:
            continue
        sel.append(r); vecs.append(v)
    if not sel:
        print(f"  {label_short}: no rollouts"); return
    X = np.stack(vecs).astype(np.float64)
    print(f"  {label_short}: N={len(sel)} D={X.shape[1]}")
    if mode == "concat" and X.shape[1] > args.concat_max_dim:
        print(f"    skip: D={X.shape[1]} > {args.concat_max_dim}"); return
    task = np.array([r["task"] for r in sel]); succ = np.array([r["succ"] for r in sel])
    Xw = geometry.whiten_by(X, task, reg=args.cov_reg)

    traces, ranges = [], []
    # ALL TASKS — global PCA-3D
    from sklearn.decomposition import PCA
    xyz_all = PCA(n_components=3, random_state=args.seed).fit_transform(Xw)
    sm, fm = succ == 1, succ == 0
    start = len(traces)
    traces.append(plot3d.scatter3d(xyz_all[sm], [TAB20[int(t) % 20] for t in task[sm]],
                                   "success (by task)", size=3, opacity=0.7))
    traces.append(plot3d.scatter3d(xyz_all[fm], "black", "fail (all tasks)", size=3, opacity=0.45))
    if fm.sum() >= 3:
        traces.append(plot3d.ellipsoid_surface(xyz_all[fm], "black", "fail 2σ"))
    ranges.append((f"ALL TASKS  (n={len(sel)}, s{int(sm.sum())}/f{int(fm.sum())})", start, len(traces)))

    for tid in ids:
        m = task == tid; st = succ[m]
        ns, nf = int((st == 1).sum()), int((st == 0).sum())
        if min(ns, nf) < args.min_class:
            continue
        xyz = geometry.meandiff_axes(Xw[m], st, seed=args.seed)
        real, null = metrics.cv_auroc(Xw[m], 1 - st, n_perm=1, seed=args.seed)
        shuf = float(null[0]) if (null is not None and len(null)) else None
        start = len(traces)
        for o in (1, 0):
            mm = st == o
            traces.append(plot3d.scatter3d(xyz[mm], SF[o], "succ" if o else "fail"))
            traces.append(plot3d.ellipsoid_surface(xyz[mm], SF[o], ("succ" if o else "fail") + " 2σ"))
        ranges.append((f"{names[tid]}  (AUROC {_fmt(real)} | shuf {_fmt(shuf)})  s{ns}/f{nf}",
                       start, len(traces)))

    n_all = len(traces)
    buttons = []
    for label, s, e in ranges:
        vis = [False] * n_all
        for k in range(s, e):
            vis[k] = True
        buttons.append({"label": label, "method": "update",
                        "args": [{"visible": vis}, {"title": f"{banner} — {label}"}]})
    for k in range(ranges[0][1], ranges[0][2]):
        traces[k]["visible"] = True

    layout = {"title": f"{banner} — {ranges[0][0]}",
              "scene": {"xaxis": {"title": "axis1"}, "yaxis": {"title": "axis2"}, "zaxis": {"title": "axis3"}},
              "updatemenus": [{"buttons": buttons, "direction": "down", "showactive": True,
                               "x": 0.0, "xanchor": "left", "y": 1.1, "yanchor": "top"}],
              "margin": {"l": 0, "r": 0, "t": 55, "b": 0}, "height": 800}
    intro = (f"<p style=\"font-family:sans-serif;font-size:13px\"><b>{banner}</b> — dropdown: ALL TASKS "
             f"(succ by task color, fail black+2σ) or per task (x=mean-diff, y,z=PC⊥). "
             f"Label: held-out CV AUROC | shuffle baseline.</p>")
    plot3d.write_html(out_path, traces, layout, online=args.online, intro=intro)
    print(f"    -> {out_path.relative_to(args.out_root)}  ({out_path.stat().st_size/1e6:.1f} MB)")


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    ids = sorted({r["task"] for r in rollouts})
    base = args.out_root / "succ_fail_3d"
    for mode in args.modes:
        if mode == "abs":
            for t in args.abs_frames:
                _render(rollouts, names, ids, "abs", t, f"absframe f{t}", f"absframe — frame {t}",
                        base / f"succ_fail_3d_absframe_f{t}.html", args)
        elif mode == "mean":
            print("== cumulative mean ==")
            for t in args.cum_frames:
                _render(rollouts, names, ids, "mean", t, f"cum_mean f{t}",
                        f"cumulative mean — up to frame {t}",
                        base / "mean" / f"succ_fail_3d_cum_mean_f{t}.html", args)
        elif mode == "concat":
            print("== cumulative concat ==")
            for t in args.cum_frames:
                _render(rollouts, names, ids, "concat", t, f"cum_concat f{t}",
                        f"cumulative concat — up to frame {t} (D={(t+1)*1024})",
                        base / "concat" / f"succ_fail_3d_cum_concat_f{t}.html", args)
        elif mode == "window":
            print("== window mean ==")
            for w in args.windows:
                _render(rollouts, names, ids, "window", w, f"window {w}",
                        f"window mean — frames [{w}]",
                        base / "window" / f"succ_fail_3d_window_w{w}.html", args)
