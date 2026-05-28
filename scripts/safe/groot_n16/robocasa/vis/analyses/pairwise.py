"""pairwise — pairwise task-separability AUROC over time.

mode=progress    : per progress milestone, 1 point/rollout, pairwise CV AUROC.
mode=all_timesteps: all per-step points (truncated@t_d), GROUP-aware CV by rollout,
                    + dendrogram/heatmap.
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core import io
from core.features import frame_at
from analyses._common import add_common

NAME = "pairwise"
HELP = "pairwise task-separability AUROC (progress milestones or all-timesteps group-aware)"


def add_args(p):
    add_common(p)
    p.add_argument("--mode", choices=("progress", "all_timesteps"), default="progress")
    p.add_argument("--n-milestones", type=int, default=11)
    p.add_argument("--t-d", type=int, default=16)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--n-folds", type=int, default=5)


def _load(cache):
    rolls, names = io.reconstruct_rollouts(cache)
    z = {i: r["z"].astype(np.float32) for i, r in enumerate(rolls)}
    rollout_task = {i: r["task"] for i, r in enumerate(rolls)}
    return z, rollout_task, names


def _pairwise_matrix(Xr, label, ids, groups, k, seed):
    n = len(ids); S = np.full((n, n), np.nan)
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            m = (label == ids[i]) | (label == ids[j])
            X = Xr[m]; y = (label[m] == ids[i]).astype(int)
            g = groups[m] if groups is not None else None
            if np.bincount(y).min() < k:
                continue
            if g is not None:
                if min(len(np.unique(g[y == 0])), len(np.unique(g[y == 1]))) < k:
                    continue
                splitter = GroupKFold(n_splits=k).split(X, y, g)
            else:
                splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y)
            aucs = []
            for tr, te in splitter:
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
            if aucs:
                S[i, j] = S[j, i] = float(np.mean(aucs))
    return S


def _offdiag(S):
    n = S.shape[0]
    v = [S[i, j] for i in range(n) for j in range(i + 1, n) if np.isfinite(S[i, j])]
    return float(np.mean(v)), float(np.min(v)), float(np.median(v))


def run(args):
    base = Path(args.cache).stem.replace("pooled_all_", "")
    z, rollout_task, names = _load(args.cache)
    rollouts = sorted(z); ids = sorted(set(rollout_task.values())); labels = [names[t] for t in ids]
    out_dir = args.out_root / "pairwise_auroc"; out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "progress":
        ps = np.linspace(0.0, 1.0, args.n_milestones); curve = {}; Smats = {}
        cv = []
        for p in ps:
            X = np.stack([frame_at(z[r], float(p)) for r in rollouts]).astype(np.float64)
            task = np.array([rollout_task[r] for r in rollouts])
            Xr = PCA(n_components=min(args.pca_dim, X.shape[0] - 1, X.shape[1]), random_state=args.seed).fit_transform(X)
            S = _pairwise_matrix(Xr, task, ids, None, args.n_folds, args.seed)
            mean_a, min_a, med_a = _offdiag(S)
            n = len(ids); best = (np.inf, None, None)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.isfinite(S[i, j]) and S[i, j] < best[0]:
                        best = (S[i, j], labels[i], labels[j])
            cv.append({"p": float(p), "mean_auroc": mean_a, "min_auroc": min_a, "median_auroc": med_a,
                       "most_confusable": [best[1], best[2], best[0]]})
            Smats[f"{p:.2f}"] = np.round(np.nan_to_num(S, nan=-1), 4).tolist()
            print(f"  p={p:.2f} mean={mean_a:.4f} median={med_a:.4f} min={min_a:.3f} ({best[1]} vs {best[2]})")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot([c["p"] for c in cv], [c["mean_auroc"] for c in cv], "o-", label="mean over pairs")
        ax.plot([c["p"] for c in cv], [c["median_auroc"] for c in cv], "s--", label="median")
        ax.plot([c["p"] for c in cv], [c["min_auroc"] for c in cv], "^:", label="min (worst pair)")
        ax.axhline(0.5, color="grey", ls="--", lw=1); ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("progress p"); ax.set_ylabel("pairwise task AUROC")
        ax.set_title(f"Task separability vs progress (pairwise CV AUROC, 1 pt/rollout)  |  {base}"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / f"progress_pairwise_auroc_{base}.png", dpi=140); plt.close(fig)
        json.dump({"mode": "progress", "curve": cv, "auroc_matrices": Smats},
                  open(out_dir / f"progress_pairwise_auroc_{base}.json", "w"), indent=2)
        print(f"wrote progress_pairwise_auroc_{base}.png/.json")
    else:
        Xl, taskl, groupl = [], [], []
        for r in rollouts:
            zr = z[r][: args.t_d]
            Xl.append(zr); taskl.append(np.full(len(zr), rollout_task[r])); groupl.append(np.full(len(zr), r))
        X = np.concatenate(Xl).astype(np.float64); task = np.concatenate(taskl); groups = np.concatenate(groupl)
        print(f"all_timesteps@{args.t_d}: {X.shape[0]} points ({X.shape[0]/len(rollouts):.1f}/rollout), group-aware CV")
        Xr = PCA(n_components=min(args.pca_dim, X.shape[1]), random_state=args.seed).fit_transform(X)
        S = _pairwise_matrix(Xr, task, ids, groups, args.n_folds, args.seed)
        mean_a, min_a, med_a = _offdiag(S)
        print(f"  mean={mean_a:.4f} median={med_a:.4f} min={min_a:.3f}")
        D = 1.0 - np.nan_to_num(S, nan=0.5); np.fill_diagonal(D, 0.0)
        Z = linkage(squareform((D + D.T) / 2, checks=False), method="average")
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [1, 1.25]})
        dn = dendrogram(Z, labels=labels, ax=axes[0], orientation="left")
        axes[0].set_xlabel("1 − AUROC (smaller = more confusable)")
        axes[0].set_title(f"dendrogram (avg linkage, all timesteps truncated@{args.t_d})")
        order = dn["leaves"][::-1]; olabels = [labels[o] for o in order]
        im = axes[1].imshow(S[np.ix_(order, order)], cmap="RdYlGn", vmin=0.5, vmax=1.0)
        axes[1].set_xticks(range(len(ids))); axes[1].set_xticklabels(olabels, rotation=90, fontsize=7)
        axes[1].set_yticks(range(len(ids))); axes[1].set_yticklabels(olabels, fontsize=7)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="pairwise CV AUROC (group-aware)")
        axes[1].set_title("pairwise separability AUROC")
        fig.suptitle(f"Task separability — ALL timesteps truncated@{args.t_d}, group-aware CV  |  {base}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"all_timesteps_pairwise_auroc_td{args.t_d}_{base}.png", dpi=140); plt.close(fig)
        pairs = sorted([(labels[i], labels[j], S[i, j]) for i in range(len(ids)) for j in range(i + 1, len(ids))],
                       key=lambda x: (np.inf if np.isnan(x[2]) else x[2]))
        for a, b, s in pairs[:12]:
            print(f"  {a:<26} <-> {b:<26} AUROC={s:.3f}")
        json.dump({"mode": "all_timesteps", "t_d": args.t_d, "mean_auroc": mean_a, "min_auroc": min_a,
                   "median_auroc": med_a, "auroc_matrix": np.round(np.nan_to_num(S, nan=-1), 4).tolist(),
                   "labels": labels, "most_confusable_pairs": [[a, b, float(s)] for a, b, s in pairs[:20]]},
                  open(out_dir / f"all_timesteps_pairwise_auroc_td{args.t_d}_{base}.json", "w"), indent=2)
        print(f"wrote all_timesteps_pairwise_auroc_td{args.t_d}_{base}.png/.json")
