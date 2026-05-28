"""taskdist — task-only cluster geometry (length-controlled truncated z_mean).

--view (any subset, default all):
  distance : inter-task centroid-distance heatmaps (Eucl/Maha) + greedy
             least-distinguishable elimination order (JSON).
  table    : 3-criteria (Eucl/Maha centroid dist + Maha silhouette) per-task
             distinguishability ranking table.
  group    : pairwise CV-AUROC dendrogram + confusable groups (medoid reps).
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core import geometry, io
from core.metrics import centroid_distance
from analyses._common import add_common

NAME = "taskdist"
HELP = "task cluster distance / distinguishability table / grouping (truncated z_mean)"


def add_args(p):
    add_common(p)
    p.add_argument("--view", nargs="+", default=["distance", "table", "group"],
                   choices=["distance", "table", "group"])
    p.add_argument("--t-d", type=int, default=17)
    p.add_argument("--cov-reg", type=float, default=1e-2)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--taus", type=float, nargs="+", default=[0.8, 0.85, 0.9, 0.95])


def _truncated_zmean(cache, t_d):
    c = io.load_feature_cache(cache)
    feats = c["feats"].astype(np.float64)
    ep_task = c["ep_task_id"]
    names = {int(t): str(n) for t, n in zip(c["ep_task_id"], c["task_names"])}
    order = np.lexsort((c["step_idx"], c["rollout_idx"]))
    rid = c["rollout_idx"][order]; uniq = np.unique(c["rollout_idx"])
    starts = np.searchsorted(rid, uniq, side="left"); ends = np.searchsorted(rid, uniq, side="right")
    X, task = [], []
    for r, a, b in zip(uniq, starts, ends):
        X.append(feats[order[a:b][:t_d]].mean(0)); task.append(int(ep_task[int(r)]))
    return np.stack(X), np.asarray(task), names


def _greedy_elim(D, ids, names):
    idx_of = {t: i for i, t in enumerate(ids)}
    remaining = list(ids); removed = []
    while len(remaining) > 1:
        ri = [idx_of[t] for t in remaining]
        sub = D[np.ix_(ri, ri)]; np.fill_diagonal(sub, np.inf)
        nn_dist, nn_arg = sub.min(axis=1), sub.argmin(axis=1)
        finite = sub.copy(); finite[~np.isfinite(finite)] = np.nan
        mean_dist = np.nanmean(finite, axis=1)
        vl = int(np.lexsort((mean_dist, nn_dist))[0])
        victim = remaining[vl]
        removed.append({"rank_removed": len(removed) + 1, "task_id": victim, "task": names[victim],
                        "nearest_remaining": names[remaining[int(nn_arg[vl])]],
                        "nearest_dist": float(nn_dist[vl]), "mean_dist_to_remaining": float(mean_dist[vl]),
                        "n_remaining_before": len(remaining)})
        remaining.remove(victim)
    last = remaining[0]; ri = [idx_of[t] for t in ids if t != last]
    d_to_last = D[idx_of[last], ri]
    removed.append({"rank_removed": len(removed) + 1, "task_id": last, "task": names[last],
                    "nearest_remaining": None, "nearest_dist": float(d_to_last.min()) if len(d_to_last) else None,
                    "mean_dist_to_remaining": None, "n_remaining_before": 1, "note": "last survivor"})
    return removed


def _greedy_elim_silhouette(X, task, ids, names, cov_reg):
    remaining = list(ids); removed = []
    while len(remaining) >= 2:
        mask = np.isin(task, remaining); Xsub, tsub = X[mask], task[mask]
        Xw = geometry.whiten_by(Xsub, tsub, reg=cov_reg)
        sv = silhouette_samples(Xw, tsub)
        per = {t: float(sv[tsub == t].mean()) for t in remaining}
        victim = min(per, key=per.get)
        removed.append({"rank_removed": len(removed) + 1, "task_id": int(victim), "task": names[victim],
                        "silhouette": per[victim], "n_remaining_before": len(remaining)})
        remaining.remove(victim)
        if len(remaining) == 1:
            removed.append({"rank_removed": len(removed) + 1, "task_id": int(remaining[0]),
                            "task": names[remaining[0]], "silhouette": per[remaining[0]],
                            "n_remaining_before": 1, "note": "last survivor"})
            break
    return removed


def _heatmap(D, ids, names, metric, path, base, t_d):
    fig, ax = plt.subplots(figsize=(11, 9)); labels = [names[t] for t in ids]
    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ids))); ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{metric} centroid distance")
    ax.set_title(f"Inter-task centroid distance ({metric})  |  z_mean t_d={t_d}  |  {base}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _pair_auroc(Xa, Xb, pca_dim, k, seed):
    X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa), int), np.zeros(len(Xb), int)]
    Xr = PCA(n_components=min(pca_dim, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    if np.bincount(y).min() < k:
        return None
    aucs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xr, y):
        sc = StandardScaler().fit(Xr[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xr[tr]), y[tr])
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(Xr[te]))[:, 1]))
    return float(np.mean(aucs)) if aucs else None


def run(args):
    base = Path(args.cache).stem.replace("pooled_all_", "")
    X, task, names = _truncated_zmean(args.cache, args.t_d)
    ids = sorted(np.unique(task).tolist()); labels = [names[t] for t in ids]
    out_dir = args.out_root / "task_cluster_distance"; out_dir.mkdir(parents=True, exist_ok=True)
    D_eu = centroid_distance(X.astype(np.float32), task, metric="euclidean")
    D_mh = centroid_distance(geometry.whiten_by(X, task, reg=args.cov_reg), task, metric="euclidean")

    if "distance" in args.view:
        _heatmap(D_eu, ids, names, "euclidean", out_dir / f"task_distance_euclidean_{base}_td{args.t_d}.png", base, args.t_d)
        _heatmap(D_mh, ids, names, "mahalanobis", out_dir / f"task_distance_mahalanobis_{base}_td{args.t_d}.png", base, args.t_d)
        json.dump({"t_d": args.t_d, "cov_reg": args.cov_reg, "task_ids": ids,
                   "task_names": {t: names[t] for t in ids},
                   "elimination_order": {"euclidean": _greedy_elim(D_eu, ids, names),
                                         "mahalanobis": _greedy_elim(D_mh, ids, names)}},
                  open(out_dir / f"task_elimination_{base}_td{args.t_d}.json", "w"), indent=2)
        print(f"[distance] wrote task_distance_*.png + task_elimination_{base}_td{args.t_d}.json")

    if "table" in args.view:
        elim_eu, elim_mh = _greedy_elim(D_eu, ids, names), _greedy_elim(D_mh, ids, names)
        elim_si = _greedy_elim_silhouette(X, task, ids, names, args.cov_reg)
        ve = {e["task"]: e["nearest_dist"] for e in elim_eu}; re = {e["task"]: e["rank_removed"] for e in elim_eu}
        vm = {e["task"]: e["nearest_dist"] for e in elim_mh}; rm = {e["task"]: e["rank_removed"] for e in elim_mh}
        vs = {e["task"]: e["silhouette"] for e in elim_si}; rs = {e["task"]: e["rank_removed"] for e in elim_si}
        summary = sorted(({"task": names[t], "eu_dist": ve[names[t]], "eu_rank": re[names[t]],
                           "maha_dist": vm[names[t]], "maha_rank": rm[names[t]],
                           "silhouette": vs[names[t]], "sil_rank": rs[names[t]],
                           "mean_rank": (re[names[t]] + rm[names[t]] + rs[names[t]]) / 3.0} for t in ids),
                          key=lambda r: r["mean_rank"])
        json.dump({"t_d": args.t_d, "cov_reg": args.cov_reg, "order_euclidean": elim_eu,
                   "order_mahalanobis": elim_mh, "order_silhouette_maha": elim_si, "per_task_summary": summary},
                  open(out_dir / f"distinguishability_{base}_td{args.t_d}.json", "w"), indent=2)
        fig, ax = plt.subplots(figsize=(13, 0.42 * len(summary) + 1.5)); ax.axis("off")
        cols = ["task", "eu_dist", "eu_r", "maha_dist", "maha_r", "silh(maha)", "sil_r", "mean_r"]
        cell = [[r["task"], f"{r['eu_dist']:.3f}", r["eu_rank"], f"{r['maha_dist']:.3f}", r["maha_rank"],
                 f"{r['silhouette']:+.3f}", r["sil_rank"], f"{r['mean_rank']:.1f}"] for r in summary]
        tbl = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.35)
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#dddddd"); tbl[0, j].set_text_props(weight="bold")
        for i, r in enumerate(summary, start=1):
            color = plt.cm.RdYlGn((r["mean_rank"] - 1) / (len(summary) - 1))
            for j in range(len(cols)):
                tbl[i, j].set_facecolor((color[0], color[1], color[2], 0.35))
        ax.set_title(f"Task distinguishability — 3 criteria (rank 1 = least distinguishable)\nz_mean t_d={args.t_d} | {base}", fontsize=11, pad=14)
        fig.tight_layout(); fig.savefig(out_dir / f"distinguishability_{base}_td{args.t_d}.png", dpi=150); plt.close(fig)
        print(f"[table] wrote distinguishability_{base}_td{args.t_d}.png/.json")

    if "group" in args.view:
        n = len(ids); by_task = {t: X[task == t] for t in ids}
        S = np.full((n, n), np.nan)
        for i in range(n):
            S[i, i] = 1.0
            for j in range(i + 1, n):
                a = _pair_auroc(by_task[ids[i]], by_task[ids[j]], args.pca_dim, args.n_folds, args.seed)
                S[i, j] = S[j, i] = a if a is not None else np.nan
        mean_auroc = {labels[i]: float(np.nanmean(np.delete(S[i], i))) for i in range(n)}
        Dmat = np.nan_to_num(1.0 - S, nan=0.5); np.fill_diagonal(Dmat, 0.0)
        Z = linkage(squareform((Dmat + Dmat.T) / 2, checks=False), method="average")
        groups_by_tau = {}
        for tau in args.taus:
            cl = fcluster(Z, t=1.0 - tau, criterion="distance"); groups = {}
            for idx, c in enumerate(cl):
                groups.setdefault(int(c), []).append(idx)
            out = []
            for members in groups.values():
                ml = [labels[m] for m in members]
                if len(members) == 1:
                    out.append({"members": ml, "representative": ml[0], "size": 1})
                else:
                    sub = S[np.ix_(members, members)].copy(); np.fill_diagonal(sub, np.nan)
                    wm = np.nanmean(sub, axis=1)
                    out.append({"members": ml, "representative": ml[int(np.nanargmax(wm))], "size": len(members),
                                "within_group_mean_auroc": {ml[k]: float(wm[k]) for k in range(len(members))}})
            groups_by_tau[f"{tau}"] = sorted(out, key=lambda g: -g["size"])
        out_g = args.out_root / "task_grouping"; out_g.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [1, 1.25]})
        dn = dendrogram(Z, labels=labels, ax=axes[0], orientation="left", color_threshold=0.1)
        axes[0].set_xlabel("1 − AUROC (smaller = more confusable)"); axes[0].set_title("task dendrogram (avg linkage on 1−pairwise AUROC)")
        order = dn["leaves"][::-1]; olabels = [labels[o] for o in order]
        im = axes[1].imshow(S[np.ix_(order, order)], cmap="RdYlGn", vmin=0.5, vmax=1.0)
        axes[1].set_xticks(range(n)); axes[1].set_xticklabels(olabels, rotation=90, fontsize=7)
        axes[1].set_yticks(range(n)); axes[1].set_yticklabels(olabels, fontsize=7)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="pairwise CV AUROC")
        axes[1].set_title("pairwise separability AUROC (green=distinct, red=confusable)")
        fig.suptitle(f"Task grouping by pairwise separability | z_mean t_d={args.t_d} | {base}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_g / f"pairwise_auroc_{base}_td{args.t_d}.png", dpi=140); plt.close(fig)
        json.dump({"t_d": args.t_d, "labels": labels, "auroc_matrix": np.round(np.nan_to_num(S, nan=-1), 4).tolist(),
                   "mean_auroc_to_others": mean_auroc, "groups_by_tau": groups_by_tau},
                  open(out_g / f"task_groups_{base}_td{args.t_d}.json", "w"), indent=2)
        print(f"[group] wrote pairwise_auroc_{base}_td{args.t_d}.png + task_groups_{base}_td{args.t_d}.json")
