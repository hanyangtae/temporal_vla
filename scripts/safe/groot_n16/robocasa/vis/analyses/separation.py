"""separation — Mahalanobis + LDA succ/fail separation (episode-mean).

Global LDA + t-SNE Maha (succ/fail, task) + per-task CV separability (Mahalanobis
& Euclidean centroid AUROC, logistic AUROC) honest cross-validated. whiten-label
task (honest) or task_failure (leaks outcome; viz only).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from core import geometry, io
from core.metrics import logistic_auc, silhouette_safe
from core.projection import lda_2d, run_tsne
from analyses._common import add_common

NAME = "separation"
HELP = "Mahalanobis+LDA succ/fail separation (episode-mean) + per-task CV AUROC"


def add_args(p):
    add_common(p)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--whiten-label", choices=("task", "task_failure"), default="task")
    p.add_argument("--pca-pre", type=int, default=50)
    p.add_argument("--n-folds", type=int, default=5)


def _cv_centroid_auroc(X, y, k, seed):
    if len(np.unique(y)) < 2 or np.bincount(y).min() < k:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores, truth = [], []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        cs, cf = X[tr][y[tr] == 1].mean(0), X[tr][y[tr] == 0].mean(0)
        scores.append(np.linalg.norm(X[te] - cs, axis=1) - np.linalg.norm(X[te] - cf, axis=1))
        truth.append((y[te] == 0).astype(int))
    s, t = np.concatenate(scores), np.concatenate(truth)
    return float(roc_auc_score(t, s)) if len(np.unique(t)) > 1 else None


def _cv_logistic_auroc(X, y, k, seed):
    if len(np.unique(y)) < 2 or np.bincount(y).min() < k:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs, n = [], len(y)
    for tr, te in skf.split(X, y):
        tm = np.zeros(n, bool); tm[tr] = True
        em = np.zeros(n, bool); em[te] = True
        a = logistic_auc(X, y, tm, em, random_state=seed)
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else None


def run(args):
    d = io.load_feature_cache(args.cache)
    X = d["ep_feat_mean"].astype(np.float64)
    succ = d["ep_success"].astype(np.int64); task = d["ep_task_id"].astype(np.int64)
    name_by_id = {int(t): str(n) for t, n in zip(d["ep_task_id"], d["task_names"])}
    agg = Path(args.cache).stem.replace("pooled_all_", "") + f"_whiten_{args.whiten_label}"
    out_dir = args.out_root / "separation_maha_lda"; out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = sorted(name_by_id)

    whiten_labels = task * 2 + (1 - succ) if args.whiten_label == "task_failure" else task
    Xw = geometry.whiten_by(X, whiten_labels, reg=args.cov_reg).astype(np.float64)
    sil = {"succ_fail_euclid": silhouette_safe(X.astype(np.float32), succ, sample_size=5000),
           "succ_fail_maha": silhouette_safe(Xw.astype(np.float32), succ, sample_size=5000),
           "task_euclid": silhouette_safe(X.astype(np.float32), task, sample_size=5000),
           "task_maha": silhouette_safe(Xw.astype(np.float32), task, sample_size=5000)}
    print("global silhouette:", json.dumps(sil, indent=2))

    pre = PCA(n_components=args.pca_pre, random_state=args.seed).fit_transform(X)
    Z_lda = lda_2d(pre, succ, random_state=args.seed)
    Z_mh = run_tsne(Xw.astype(np.float32), metric="euclidean", perplexity=30, random_state=args.seed, init="pca")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    fail = succ == 0
    axes[0].scatter(Z_lda[~fail, 0], Z_lda[~fail, 1], s=6, alpha=0.5, c="#2ca02c", label="success")
    axes[0].scatter(Z_lda[fail, 0], Z_lda[fail, 1], s=6, alpha=0.5, c="#d62728", label="fail")
    axes[0].set_title("LDA (succ/fail, PCA50->LDA) — in-sample viz"); axes[0].legend(markerscale=3)
    axes[1].scatter(Z_mh[~fail, 0], Z_mh[~fail, 1], s=6, alpha=0.4, c="#2ca02c", rasterized=True)
    axes[1].scatter(Z_mh[fail, 0], Z_mh[fail, 1], s=6, alpha=0.4, c="#d62728", rasterized=True)
    axes[1].set_title(f"t-SNE Mahalanobis (succ/fail)  silh_mh={sil['succ_fail_maha']:.3f}")
    cmap = mpl.cm.get_cmap("tab20", 20)
    sc = axes[2].scatter(Z_mh[:, 0], Z_mh[:, 1], s=6, alpha=0.5, c=task, cmap=cmap, vmin=0, vmax=19, rasterized=True)
    axes[2].set_title("t-SNE Mahalanobis (task id)"); fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Mahalanobis + LDA succ/fail separation (episode-mean)  |  {agg}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"global_maha_lda_{agg}.png", dpi=140); plt.close(fig)

    scores = []
    ncol = 6; nrow = math.ceil(len(task_ids) / ncol)
    figh, axesh = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    axesh = np.atleast_1d(axesh).ravel()
    for ax, tid in zip(axesh, task_ids):
        m = task == tid; yw, yr, s = Xw[m], X[m], succ[m]; srr = float(s.mean())
        auc_maha = _cv_centroid_auroc(yw, s, args.n_folds, args.seed)
        auc_eucl = _cv_centroid_auroc(yr, s, args.n_folds, args.seed)
        pre_t = PCA(n_components=min(args.pca_pre, m.sum() - 1), random_state=args.seed).fit_transform(yr)
        auc_logit = _cv_logistic_auroc(pre_t, s, args.n_folds, args.seed)
        scores.append({"task_id": int(tid), "task": name_by_id[tid], "n": int(m.sum()), "success_rate": srr,
                       "cv_auroc_maha_centroid": auc_maha, "cv_auroc_euclid_centroid": auc_eucl,
                       "cv_auroc_logistic_pca50": auc_logit,
                       "silhouette_maha_insample": silhouette_safe(yw.astype(np.float32), s)})
        if len(np.unique(s)) == 2:
            z1 = LinearDiscriminantAnalysis(n_components=1).fit_transform(pre_t, s)[:, 0]
            ax.hist(z1[s == 1], bins=20, alpha=0.6, color="#2ca02c"); ax.hist(z1[s == 0], bins=20, alpha=0.6, color="#d62728")
        au = f"{auc_maha:.2f}" if auc_maha is not None else "-"
        ax.set_title(f"{name_by_id[tid]}\nSR={srr:.2f} CVmaha={au}", fontsize=8); ax.set_yticks([])
    for ax in axesh[len(task_ids):]:
        ax.axis("off")
    figh.suptitle(f"Per-task LDA-1D succ/fail (in-sample viz); titles=CV Maha AUROC  |  {agg}", fontsize=12)
    figh.tight_layout(rect=(0, 0, 1, 0.95))
    figh.savefig(out_dir / f"per_task_grid_lda_{agg}.png", dpi=140); plt.close(figh)

    scores.sort(key=lambda r: (r["cv_auroc_maha_centroid"] or -1), reverse=True)
    json.dump({"global_silhouette": sil, "cov_reg": args.cov_reg, "n_folds": args.n_folds, "per_task": scores},
              open(out_dir / f"per_task_scores_{agg}.json", "w"), indent=2)
    macro = lambda key: float(np.mean([r[key] for r in scores if r[key] is not None]))
    print(f"macro CV AUROC maha={macro('cv_auroc_maha_centroid'):.3f} "
          f"eucl={macro('cv_auroc_euclid_centroid'):.3f} logit={macro('cv_auroc_logistic_pca50'):.3f}")
    print(f"wrote {out_dir}/(global_maha_lda|per_task_grid_lda|per_task_scores)_{agg}.*")
