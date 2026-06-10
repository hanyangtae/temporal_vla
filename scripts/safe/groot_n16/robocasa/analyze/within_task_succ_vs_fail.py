"""Success-clustering control vs failure clustering (within-task), top-9 separable tasks.

Tests whether within-task FAILURE clusters are failure-specific modes or just
scene/initial-condition (or length) variation. Logic:
  - failure bimodal (high silhouette) but success unimodal -> failure-specific modes.
  - success ALSO bimodal with similar structure -> not failure-specific (scene/init/length).

Caveat tracked: successes have variable length, so the 'mean over first t_d steps'
feature partly encodes success length; we report each success cluster's mean
episode length so length-driven success clusters are visible (failures are all 45,
so failure clusters have no length variance).

Length control cutoffs: success mean (~18) and mean+1sigma (~26).

Outputs -> outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/within_task/
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from core.io import reconstruct_rollouts as reconstruct  # noqa: E402

DEFAULT_CACHE = (REPO / "outputs/eval/robocasa/groot_n16"
                 / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
                 / "analysis/feature_cache/pooled_all_hmean_dmean.npz")

OUT = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/within_task"
SEED = 0


def cv_auroc(X, y, k=5):
    if len(np.unique(y)) < 2 or np.bincount(y).min() < k:
        return None
    Xr = PCA(n_components=min(30, X.shape[1], X.shape[0] - 1), random_state=SEED).fit_transform(X)
    a = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(Xr, y):
        sc = StandardScaler().fit(Xr[tr])
        c = LogisticRegression(max_iter=2000).fit(sc.transform(Xr[tr]), y[tr])
        a.append(roc_auc_score(y[te], c.predict_proba(sc.transform(Xr[te]))[:, 1]))
    return float(np.mean(a))


def best_kmeans(Xr, kmax=6):
    best = (None, -1, None)
    for K in range(2, kmax + 1):
        if len(Xr) <= K:
            continue
        lab = KMeans(K, n_init=10, random_state=SEED).fit(Xr).labels_
        if len(set(lab)) < 2:
            continue
        sil = silhouette_score(Xr, lab)
        if sil > best[1]:
            best = (K, sil, lab)
    return best


def cluster_outcome(rollouts, t, names, outcome, t_d):
    rs = [r for r in rollouts if r["task"] == t and (r["succ"] == 1) == (outcome == "succ")]
    X = np.stack([r["z"][:t_d].mean(0) for r in rs])
    lens = np.array([len(r["z"]) for r in rs])
    d = min(10, len(X) - 1)
    Xr = PCA(n_components=d, random_state=SEED).fit_transform(X)
    K, sil, lab = best_kmeans(Xr)
    # per-cluster mean episode length (to expose length-driven clusters)
    clen = {}
    if lab is not None:
        for c in sorted(set(lab)):
            clen[int(c)] = round(float(lens[lab == c].mean()), 1)
    return {"n": len(X), "bestK": K, "sil": round(sil, 3) if K else None,
            "len_std": round(float(lens.std()), 1), "cluster_mean_len": clen,
            "X2": PCA(n_components=2, random_state=SEED).fit_transform(X), "lab": lab}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rollouts, names = reconstruct(DEFAULT_CACHE)
    ids = sorted(names)
    succ_len = np.array([len(r["z"]) for r in rollouts if r["succ"] == 1])
    t_mean, t_sig = int(round(succ_len.mean())), int(round(succ_len.mean() + succ_len.std()))

    # rank by separability (t_mean) -> top9 (same as failure-modes script)
    rank = []
    for t in ids:
        rs = [r for r in rollouts if r["task"] == t]
        X = np.stack([r["z"][:t_mean].mean(0) for r in rs])
        y = np.array([1 - r["succ"] for r in rs])
        rank.append((t, cv_auroc(X, y)))
    rank.sort(key=lambda r: (r[1] if r[1] is not None else -1), reverse=True)
    top9 = [t for t, _ in rank[:9]]

    results = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 14)); axes = axes.ravel()
    for ax, t in zip(axes, top9):
        F = cluster_outcome(rollouts, t, names, "fail", t_mean)
        S = cluster_outcome(rollouts, t, names, "succ", t_mean)
        Ft = cluster_outcome(rollouts, t, names, "fail", t_sig)
        St = cluster_outcome(rollouts, t, names, "succ", t_sig)
        verdict = ("failure-specific" if (F["sil"] or 0) > 0.45 and (S["sil"] or 0) < 0.35
                   else "both-bimodal(scene/len?)" if (F["sil"] or 0) > 0.45 and (S["sil"] or 0) >= 0.35
                   else "weak/none")
        results.append({"task": names[t],
                        "fail_n": F["n"], "fail_K": F["bestK"], "fail_sil": F["sil"],
                        "succ_n": S["n"], "succ_K": S["bestK"], "succ_sil": S["sil"],
                        "succ_len_std": S["len_std"], "succ_cluster_mean_len": S["cluster_mean_len"],
                        "fail_sil_tsig": Ft["sil"], "succ_sil_tsig": St["sil"], "verdict": verdict})
        # viz: SUCCESS colored by cluster
        if S["lab"] is not None:
            ax.scatter(S["X2"][:, 0], S["X2"][:, 1], s=14, c=S["lab"], cmap="tab10", alpha=0.85)
        ax.set_title(f"{names[t]} SUCCESS\nfail_sil={F['sil']} succ_sil={S['sil']} "
                     f"(succ K={S['bestK']}, lenstd={S['len_std']})\n{verdict}", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(top9):]:
        ax.axis("off")
    fig.suptitle(f"SUCCESS clustering control (top-9). PCA-2D of successes, color=cluster. t_mean={t_mean}\n"
                 f"compare succ_sil vs fail_sil: fail high & succ low => failure-specific modes", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "within_task_success_clusters_top9.png", dpi=140); plt.close(fig)

    json.dump({"t_mean": t_mean, "t_sig": t_sig, "results": results},
              open(OUT / "succ_vs_fail_clustering.json", "w"), indent=2)
    with (OUT / "succ_vs_fail_clustering.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "fail_n", "fail_K", "fail_sil", "succ_n", "succ_K", "succ_sil",
                    "succ_len_std", "fail_sil_tsig", "succ_sil_tsig", "verdict"])
        for r in results:
            w.writerow([r["task"], r["fail_n"], r["fail_K"], r["fail_sil"], r["succ_n"], r["succ_K"],
                        r["succ_sil"], r["succ_len_std"], r["fail_sil_tsig"], r["succ_sil_tsig"], r["verdict"]])

    print(f"t_mean={t_mean} t_sig={t_sig}\n")
    print(f"{'task':<28} {'fail_sil':>8} {'succ_sil':>8} {'succ_K':>6} {'succ_lenstd':>11}  verdict")
    for r in results:
        print(f"  {r['task']:<26} {str(r['fail_sil']):>8} {str(r['succ_sil']):>8} {str(r['succ_K']):>6} "
              f"{str(r['succ_len_std']):>11}  {r['verdict']}  succ_cl_len={r['succ_cluster_mean_len']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
