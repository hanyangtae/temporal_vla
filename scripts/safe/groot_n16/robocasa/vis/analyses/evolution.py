"""evolution — cluster-evolution frames + mp4 over PROGRESS or WINDOW bins.

Per temporal bin: one point/rollout, project (LDA / t-SNE Euclid / t-SNE Maha),
axis bounds locked across bins, frames -> mp4. Color: success(green/red) and
task_succ_fail. whiten-label task (non-circular) or task_succ_fail (fail-as-group).
"""

from __future__ import annotations

import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from core import io
from core.distance import pooled_within_cov, whiten
from core.features import frame_at
from core.metrics import silhouette_safe
from core.plotting import global_bounds_2d, make_video, scatter_with_centroids
from core.projection import lda_2d, run_tsne
from analyses._common import add_common

NAME = "evolution"
HELP = "cluster-evolution frames+mp4 over progress/window bins"

FAIL_LABEL = 18
SF_COLORS = {0: (0.85, 0.27, 0.24, 1.0), 1: (0.27, 0.6, 0.27, 1.0)}
SF_NAMES = {0: "fail", 1: "success"}


def add_args(p):
    add_common(p)
    p.add_argument("--mode", choices=("progress", "window"), default="progress")
    p.add_argument("--n-milestones", type=int, default=11)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--max-window-idx", type=int, default=8)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--whiten-label", choices=("task", "task_succ_fail"), default="task")
    p.add_argument("--full-dim-whiten", action="store_true")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--no-tsne", action="store_true")


def _load(args):
    rolls, names = io.reconstruct_rollouts(args.cache)
    z = {i: r["z"].astype(np.float32) for i, r in enumerate(rolls)}
    ep_success = np.array([r["succ"] for r in rolls])
    ep_task = np.array([r["task"] for r in rolls])
    c = io.load_feature_cache(args.cache)
    base = c["horizon_idx_rel"].item() if "horizon_idx_rel" in c.files else "h"
    return z, ep_success, ep_task, names, base


def _bins(args):
    if args.mode == "progress":
        return [("progress", float(p)) for p in np.linspace(0.0, 1.0, args.n_milestones)]
    return [("window", w) for w in range(args.max_window_idx + 1)]


def _build_bin(z, ep_success, ep_task, kind, val, W):
    X, succ, task = [], [], []
    for r, zr in z.items():
        if kind == "progress":
            X.append(frame_at(zr, val))
        else:
            lo, hi = val * W, (val + 1) * W
            if zr.shape[0] < hi:
                continue
            X.append(zr[lo:hi].mean(0))
        succ.append(int(ep_success[r])); task.append(int(ep_task[r]))
    if not X:
        return None
    return np.stack(X).astype(np.float64), np.asarray(succ), np.asarray(task)


def _palette(names):
    cmap = mpl.cm.get_cmap("tab20", 20)
    colors = {i: cmap(i % 20) for i in range(18)}
    colors[FAIL_LABEL] = (0.15, 0.15, 0.15, 1.0)
    nm = {i: names.get(i, str(i)) for i in range(18)}
    nm[FAIL_LABEL] = "FAIL"
    return colors, nm


def run(args):
    z, ep_success, ep_task, names, base = _load(args)
    sf_colors, sf_names = _palette(names)
    W = args.window
    mode_slug = f"window_W{W}" if args.mode == "window" else "progress"
    if args.whiten_label != "task":
        mode_slug += f"_wl-{args.whiten_label}" + ("_fulldim" if args.full_dim_whiten else "")
    out_dir = args.out_root / "evolution" / f"{mode_slug}_{base}"
    (out_dir / "success").mkdir(parents=True, exist_ok=True)
    (out_dir / "task_succ_fail").mkdir(parents=True, exist_ok=True)

    frames = []
    for kind, val in _bins(args):
        built = _build_bin(z, ep_success, ep_task, kind, val, W)
        if built is None:
            continue
        X, succ, task = built
        n_succ, n_fail = int((succ == 1).sum()), int((succ == 0).sum())
        task_sf = np.where(succ == 1, task, FAIL_LABEL)
        Xr = PCA(n_components=min(args.pca_dim, X.shape[0] - 1, X.shape[1]), random_state=args.seed).fit_transform(X)
        whiten_lab = task_sf if args.whiten_label == "task_succ_fail" else task
        Xfit = X if args.full_dim_whiten else Xr
        Xw = whiten(Xfit, pooled_within_cov(Xfit, whiten_lab, reg=args.cov_reg))
        rec = {"kind": kind, "val": val, "succ": succ, "task_sf": task_sf,
               "n_succ": n_succ, "n_fail": n_fail, "n": len(succ),
               "lda_succ": lda_2d(Xr, succ, random_state=args.seed) if n_succ >= 2 and n_fail >= 2 else None,
               "lda_tasksf": lda_2d(Xr, task_sf, random_state=args.seed) if len(np.unique(task_sf)) > 2 else None}
        if args.no_tsne:
            rec["tsne_eu"], rec["tsne_mh"] = Xr[:, :2], Xw[:, :2]
        else:
            rec["tsne_eu"] = run_tsne(Xr.astype(np.float32), metric="euclidean", perplexity=30, random_state=args.seed, init="pca")
            rec["tsne_mh"] = run_tsne(Xw.astype(np.float32), metric="euclidean", perplexity=30, random_state=args.seed, init="pca")
        rec["sil_eu"] = silhouette_safe(Xr.astype(np.float32), succ)
        rec["sil_mh"] = silhouette_safe(Xw, succ)
        rec["sil_task_mh"] = silhouette_safe(Xw, task)
        frames.append(rec)
        lbl = f"p={val:.2f}" if kind == "progress" else f"steps[{val*W}:{(val+1)*W})"
        print(f"  {lbl:<18} n={rec['n']:>4} (s{n_succ}/f{n_fail})  sil_eu={rec['sil_eu']}  sil_mh={rec['sil_mh']}")

    b_lda_s = global_bounds_2d([f["lda_succ"] for f in frames if f["lda_succ"] is not None])
    b_lda_t = global_bounds_2d([f["lda_tasksf"] for f in frames if f["lda_tasksf"] is not None])
    b_eu = global_bounds_2d([f["tsne_eu"] for f in frames])
    b_mh = global_bounds_2d([f["tsne_mh"] for f in frames])

    def render(color):
        paths = []
        for k, f in enumerate(frames):
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            if color == "success":
                colors, cnames, fail_label = SF_COLORS, SF_NAMES, 0
                labels, lda, b_lda = f["succ"], f["lda_succ"], b_lda_s
            else:
                colors, cnames, fail_label = sf_colors, sf_names, FAIL_LABEL
                labels, lda, b_lda = f["task_sf"], f["lda_tasksf"], b_lda_t
            if lda is not None:
                scatter_with_centroids(axes[0], lda, labels, colors, cnames, title="LDA 2D",
                                       fail_label=fail_label, xlim=b_lda[0], ylim=b_lda[1], show_ellipse=False)
            else:
                axes[0].set_title("LDA 2D (n/a)")
            scatter_with_centroids(axes[1], f["tsne_eu"], labels, colors, cnames, title="t-SNE Euclidean",
                                   fail_label=fail_label, xlim=b_eu[0], ylim=b_eu[1], show_ellipse=False)
            scatter_with_centroids(axes[2], f["tsne_mh"], labels, colors, cnames, title="t-SNE Mahalanobis",
                                   fail_label=fail_label, xlim=b_mh[0], ylim=b_mh[1], show_ellipse=False)
            lbl = f"progress p={f['val']:.2f}" if f["kind"] == "progress" else \
                  f"window steps[{int(f['val'])*W}:{(int(f['val'])+1)*W})"
            sil_eu = "NA" if f["sil_eu"] is None else f"{f['sil_eu']:+.3f}"
            sil_mh = "NA" if f["sil_mh"] is None else f"{f['sil_mh']:+.3f}"
            fig.suptitle(f"{args.mode} evolution [{color}]  |  {lbl}  |  n={f['n']} (s{f['n_succ']}/f{f['n_fail']})  "
                         f"|  succ/fail silhouette eu={sil_eu} maha={sil_mh}  |  {base}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fp = out_dir / color / f"frame_{k:03d}.png"
            fig.savefig(fp, dpi=120); plt.close(fig); paths.append(fp)
        if paths:
            make_video(paths, out_dir / f"evolution_{mode_slug}_{color}.mp4", args.fps)
            print(f"wrote evolution_{mode_slug}_{color}.mp4 ({len(paths)} frames)")

    render("success")
    render("task_succ_fail")

    with (out_dir / f"summary_{args.mode}.tsv").open("w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["bin", "label", "n", "n_succ", "n_fail", "sil_eu_sf", "sil_maha_sf", "sil_maha_task"])
        for k, f in enumerate(frames):
            lbl = f"{f['val']:.3f}" if f["kind"] == "progress" else f"{int(f['val'])*W}-{(int(f['val'])+1)*W}"
            wr.writerow([k, lbl, f["n"], f["n_succ"], f["n_fail"], f["sil_eu"], f["sil_mh"], f["sil_task_mh"]])
    print(f"wrote {out_dir}/summary_{args.mode}.tsv")
