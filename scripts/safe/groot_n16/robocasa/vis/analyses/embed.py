"""embed — SAFE-style feature-space scatter (PCA/t-SNE) colored succ/fail, task, progress.

Per-timestep or per-rollout level; silhouette scores for success_fail/task/task_failure.
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from core import io
from analyses._common import add_common

NAME = "embed"
HELP = "PCA/t-SNE feature-space scatter (succ/fail, task, progress) + silhouette"


def add_args(p):
    add_common(p)
    p.add_argument("--level", choices=("timestep", "rollout"), default="timestep")
    p.add_argument("--projector", choices=("pca", "tsne"), default="pca")
    p.add_argument("--max-points", type=int, default=20000)
    p.add_argument("--pca-pre", type=int, default=50)
    p.add_argument("--silhouette-sample", type=int, default=10000)


def _load_level(cache, level):
    d = io.load_feature_cache(cache)
    if level == "timestep":
        feats, success, task_id, progress = d["feats"], d["success"], d["task_id"], d["progress"]
    else:
        feats, success, task_id, progress = d["ep_feat_mean"], d["ep_success"], d["ep_task_id"], None
    name_by_id = {int(t): str(n) for t, n in zip(d["ep_task_id"], d["task_names"])}
    return feats, success, task_id, progress, name_by_id


def _project(feats, projector, pca_pre, seed):
    if projector == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(feats)
    x = feats
    if pca_pre and feats.shape[1] > pca_pre:
        x = PCA(n_components=pca_pre, random_state=seed).fit_transform(feats)
    return TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(x)


def _sil(feats, success, task_id, sample, seed):
    rng = np.random.default_rng(seed)
    n = len(feats)
    idx = rng.choice(n, size=min(sample, n), replace=False) if n > sample else np.arange(n)
    f = feats[idx]
    return {"success_fail": float(silhouette_score(f, success[idx])),
            "task": float(silhouette_score(f, task_id[idx])),
            "task_failure": float(silhouette_score(f, task_id[idx] * 2 + (1 - success[idx]))),
            "n_sampled": int(len(idx))}


def run(args):
    feats, success, task_id, progress, name_by_id = _load_level(args.cache, args.level)
    print(f"level={args.level} feats={feats.shape}")
    rng = np.random.default_rng(args.seed)
    if args.projector == "tsne" and len(feats) > args.max_points:
        sel = np.sort(rng.choice(len(feats), size=args.max_points, replace=False))
        feats_p, success_p, task_p = feats[sel], success[sel], task_id[sel]
        progress_p = progress[sel] if progress is not None else None
    else:
        feats_p, success_p, task_p, progress_p = feats, success, task_id, progress
    print(f"projecting {feats_p.shape} via {args.projector} ...")
    xy = _project(feats_p, args.projector, args.pca_pre, args.seed)
    sil = _sil(feats, success, task_id, args.silhouette_sample, args.seed)
    print("silhouette:", json.dumps(sil, indent=2))

    n_panels = 3 if progress_p is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    fail = success_p == 0
    axes[0].scatter(xy[~fail, 0], xy[~fail, 1], s=3, alpha=0.35, c="#2ca02c", label="success", rasterized=True)
    axes[0].scatter(xy[fail, 0], xy[fail, 1], s=3, alpha=0.35, c="#d62728", label="fail", rasterized=True)
    axes[0].set_title(f"success/fail  (silh={sil['success_fail']:.3f})"); axes[0].legend(markerscale=4)
    cmap = mpl.cm.get_cmap("tab20", 20)
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.5, c=task_p, cmap=cmap, vmin=0, vmax=19, rasterized=True)
    axes[1].set_title(f"task id  (silh={sil['task']:.3f})")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04).set_label("task_id")
    if progress_p is not None:
        sc = axes[2].scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.5, c=progress_p, cmap="viridis", vmin=0, vmax=1, rasterized=True)
        axes[2].set_title("progress 0->1"); fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    agg = Path(args.cache).stem.replace("pooled_all_", "")
    fig.suptitle(f"GR00T-N1.6 seen18 RoboCasa  |  {args.level}  |  {args.projector}  |  {agg}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir = args.out_root / "embedding_views"; out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.level}_{args.projector}_{agg}"
    fig.savefig(out_dir / f"{stem}.png", dpi=140); plt.close(fig)
    json.dump({"level": args.level, "projector": args.projector, "agg": agg, **sil},
              open(out_dir / f"{stem}.silhouette.json", "w"), indent=2)
    print(f"wrote {out_dir}/{stem}.png + .silhouette.json")
