"""windows — step-window separation: W∈{3,4,5} × {mean,concat}, CV AUROC/silhouette
vs window index (+ projection panels). window 0 = length-controlled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from core import geometry, io
from core.metrics import cv_auroc, silhouette_safe
from core.projection import run_tsne
from analyses._common import add_common

NAME = "windows"
HELP = "step-window succ/fail separation (W×agg) vs window index + projections"


def add_args(p):
    add_common(p)
    p.add_argument("--windows", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--aggs", nargs="+", default=["mean", "concat"])
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-class", type=int, default=8)
    p.add_argument("--max-window-idx", type=int, default=8)
    p.add_argument("--proj-window", type=int, default=2)
    p.add_argument("--proj-agg", default="concat")
    p.add_argument("--proj-win-idx", type=int, nargs="+", default=[0])


def _groups(rollout_idx, step_idx):
    order = np.lexsort((step_idx, rollout_idx))
    rid = rollout_idx[order]
    uniq = np.unique(rollout_idx)
    starts = np.searchsorted(rid, uniq, side="left"); ends = np.searchsorted(rid, uniq, side="right")
    return {int(r): order[a:b] for r, a, b in zip(uniq, starts, ends)}


def _build(feats, groups, ep_success, ep_task, W, w, agg):
    lo, hi = w * W, (w + 1) * W
    X, succ, task = [], [], []
    for r, idxs in groups.items():
        if len(idxs) < hi:
            continue
        block = feats[idxs[lo:hi]]
        X.append(block.mean(0) if agg == "mean" else block.reshape(-1))
        succ.append(int(ep_success[r])); task.append(int(ep_task[r]))
    if not X:
        return None
    return np.stack(X).astype(np.float64), np.asarray(succ), np.asarray(task)


def _reduce(X, dim, seed):
    return PCA(n_components=min(dim, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)


def run(args):
    c = io.load_feature_cache(args.cache)
    feats = c["feats"].astype(np.float32)
    ep_success, ep_task = c["ep_success"], c["ep_task_id"]
    groups = _groups(c["rollout_idx"], c["step_idx"])
    agg_base = Path(args.cache).stem.replace("pooled_all_", "")
    out_dir = args.out_root / "step_windows"; out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for W in args.windows:
        for agg in args.aggs:
            for w in range(args.max_window_idx + 1):
                built = _build(feats, groups, ep_success, ep_task, W, w, agg)
                if built is None:
                    continue
                X, succ, task = built
                ns, nf = int((succ == 1).sum()), int((succ == 0).sum())
                row = {"W": W, "agg": agg, "window": w, "steps": [w * W, (w + 1) * W], "n_succ": ns, "n_fail": nf, "n": len(succ)}
                if min(ns, nf) >= args.min_class:
                    Xr = _reduce(X, args.pca_dim, args.seed)
                    row["cv_auroc_sf"] = cv_auroc(Xr, (succ == 0).astype(int), k=args.n_folds, pca_dim=args.pca_dim, seed=args.seed)[0]
                    row["sil_eu_sf"] = silhouette_safe(Xr.astype(np.float32), succ)
                    Xw = geometry.whiten_by(Xr, task, reg=args.cov_reg)
                    row["sil_maha_sf"] = silhouette_safe(Xw, succ)
                    row["sil_task"] = silhouette_safe(Xr.astype(np.float32), task) if len(np.unique(task)) > 1 else None
                    row["sil_task_maha"] = silhouette_safe(Xw, task) if len(np.unique(task)) > 1 else None
                else:
                    row.update({k: None for k in ["cv_auroc_sf", "sil_eu_sf", "sil_maha_sf", "sil_task", "sil_task_maha"]})
                results.append(row)
                au = row.get("cv_auroc_sf")
                print(f"  W={W} {agg:<6} win{w} steps[{w*W:>2}:{(w+1)*W:>2}) n={len(succ):>4} (s{ns}/f{nf}) AUROC={au if au is None else round(au,3)}")

    win0 = [r for r in results if r["window"] == 0 and r.get("cv_auroc_sf") is not None]
    best = max(win0, key=lambda r: r["cv_auroc_sf"]) if win0 else None
    json.dump({"results": results, "best_window0": {k: best[k] for k in ("W", "agg", "cv_auroc_sf", "sil_maha_sf")} if best else None},
              open(out_dir / f"window_separation_{agg_base}.json", "w"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for W in args.windows:
        for agg in args.aggs:
            sub = [r for r in results if r["W"] == W and r["agg"] == agg and r.get("cv_auroc_sf") is not None]
            if not sub:
                continue
            xs = [r["window"] for r in sub]
            axes[0].plot(xs, [r["cv_auroc_sf"] for r in sub], "o-", label=f"W={W} {agg}")
            axes[1].plot(xs, [r["sil_maha_sf"] for r in sub], "o-", label=f"W={W} {agg}")
    axes[0].axhline(0.5, color="grey", ls="--", lw=1)
    axes[0].set_title("success/fail CV logistic AUROC vs window\n(window 0 = length-controlled)")
    axes[0].set_xlabel("window index"); axes[0].set_ylabel("CV AUROC"); axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="grey", ls="--", lw=1)
    axes[1].set_title("success/fail Mahalanobis silhouette vs window")
    axes[1].set_xlabel("window index"); axes[1].set_ylabel("silhouette (whitened)"); axes[1].legend(fontsize=8)
    ttl = f"Step-window separation  |  base={agg_base}"
    if best:
        ttl += f"  |  best@win0: W={best['W']} {best['agg']} AUROC={best['cv_auroc_sf']:.3f}"
    fig.suptitle(ttl, fontsize=13); fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"window_separation_{agg_base}.png", dpi=140); plt.close(fig)
    print(f"wrote window_separation_{agg_base}.png/.json")

    for w in args.proj_win_idx:
        built = _build(feats, groups, ep_success, ep_task, args.proj_window, w, args.proj_agg)
        if built is None:
            continue
        X, succ, task = built
        task_sf = np.where(succ == 1, task, 18)
        Xr = _reduce(X, args.pca_dim, args.seed)
        Xw = geometry.whiten_by(Xr, task, reg=args.cov_reg)
        proj = {}
        if len(np.unique(succ)) == 2:
            proj["LDA(succ/fail)"] = np.column_stack([
                LinearDiscriminantAnalysis(n_components=1).fit_transform(Xr, succ)[:, 0],
                np.random.default_rng(0).normal(0, 0.3, len(succ))])
        proj["PCA"] = Xr[:, :2]
        proj["t-SNE Maha"] = run_tsne(Xw.astype(np.float32), metric="euclidean", perplexity=30, random_state=args.seed, init="pca")
        ncol = len(proj)
        fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 11))
        cmap20 = mpl.cm.get_cmap("tab20", 20)
        for j, (nm, xy) in enumerate(proj.items()):
            fail = succ == 0
            axes[0, j].scatter(xy[~fail, 0], xy[~fail, 1], s=8, alpha=0.5, c="#2ca02c")
            axes[0, j].scatter(xy[fail, 0], xy[fail, 1], s=8, alpha=0.5, c="#d62728")
            axes[0, j].set_title(f"{nm} — success/fail"); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
            sc = axes[1, j].scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6, c=task_sf, cmap=cmap20, vmin=0, vmax=19)
            axes[1, j].set_title(f"{nm} — task_succ_fail (18=fail)"); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        fig.colorbar(sc, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)
        fig.suptitle(f"W={args.proj_window} {args.proj_agg} window {w} steps[{w*args.proj_window}:{(w+1)*args.proj_window}) "
                     f"n={len(succ)} (s{(succ==1).sum()}/f{(succ==0).sum()})  |  {agg_base}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"proj_W{args.proj_window}_{args.proj_agg}_win{w}_{agg_base}.png", dpi=140); plt.close(fig)
        print(f"wrote proj_W{args.proj_window}_{args.proj_agg}_win{w}_{agg_base}.png")
