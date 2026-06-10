"""Within-task failure-mode clustering for the top-separable tasks (length-controlled).

Step 1: rank all 18 tasks by success/failure SEPARABILITY in latent — per-task CV
logistic AUROC (fail vs succ) on the length-controlled episode feature
(mean over first t_d steps). Take the top 9.

Step 2: for those 9, cluster the FAILURE rollouts into modes. Length control via
two cutoffs t_d = success-length mean (~18) and mean+1sigma (~26); failures are
all 45 steps so failure clustering has NO length variance (clusters = behavioral
modes, not length). PCA-reduce, KMeans K=2..6 (silhouette/Davies-Bouldin) + GMM
(BIC). Report best K and stability across the two cutoffs; visualize each task's
failures (PCA-2D) colored by cluster.

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
from sklearn.metrics import davies_bouldin_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from temporal_agg import DEFAULT_CACHE, reconstruct  # noqa: E402

OUT = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/within_task"
SEED = 0


def trunc_mean(z, t_d):
    return z[:t_d].mean(0)


def cv_auroc(X, y, seed=SEED, k=5):
    if len(np.unique(y)) < 2 or np.bincount(y).min() < k:
        return None
    Xr = PCA(n_components=min(30, X.shape[1], X.shape[0] - 1), random_state=seed).fit_transform(X)
    aucs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xr, y):
        sc = StandardScaler().fit(Xr[tr])
        c = LogisticRegression(max_iter=2000).fit(sc.transform(Xr[tr]), y[tr])
        p = c.predict_proba(sc.transform(Xr[te]))[:, 1]
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else None


def best_kmeans(Xr, kmax=6):
    best = (None, -1, None)  # K, silhouette, labels
    rows = {}
    for K in range(2, kmax + 1):
        if len(Xr) <= K:
            continue
        km = KMeans(K, n_init=10, random_state=SEED).fit(Xr)
        if len(set(km.labels_)) < 2:
            continue
        sil = silhouette_score(Xr, km.labels_)
        db = davies_bouldin_score(Xr, km.labels_)
        rows[K] = (round(float(sil), 3), round(float(db), 3))
        if sil > best[1]:
            best = (K, sil, km.labels_)
    return best, rows


def best_gmm_bic(Xr, kmax=6):
    best = (None, np.inf, None)
    for K in range(2, kmax + 1):
        if len(Xr) <= K:
            continue
        for ct in ("full", "diag"):
            try:
                g = GaussianMixture(K, covariance_type=ct, random_state=SEED, reg_covar=1e-4).fit(Xr)
                b = g.bic(Xr)
                if b < best[1]:
                    best = (K, float(b), ct)
            except Exception:
                pass
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rollouts, names = reconstruct(DEFAULT_CACHE)
    ids = sorted(names)
    succ_len = np.array([len(r["z"]) for r in rollouts if r["succ"] == 1])
    t_mean = int(round(succ_len.mean()))
    t_sig = int(round(succ_len.mean() + succ_len.std()))
    print(f"success len mean={succ_len.mean():.1f} std={succ_len.std():.1f} -> cutoffs t_mean={t_mean} t_mean1sig={t_sig}")

    # ---- step 1: rank by separability (at t_mean) ----
    rank = []
    for t in ids:
        rs = [r for r in rollouts if r["task"] == t]
        X = np.stack([trunc_mean(r["z"], t_mean) for r in rs])
        y = np.array([1 - r["succ"] for r in rs])
        auc = cv_auroc(X, y)
        rank.append((t, names[t], auc, int((y == 1).sum()), int((y == 0).sum())))
    rank.sort(key=lambda r: (r[2] if r[2] is not None else -1), reverse=True)
    print("\n=== task separability ranking (CV AUROC fail vs succ, t_mean) ===")
    for i, (t, nm, auc, nf, ns) in enumerate(rank):
        tag = " <== top9" if i < 9 else ""
        print(f"  {i+1:>2}. {nm:<28} AUROC={auc if auc is None else round(auc,3)}  f{nf}/s{ns}{tag}")
    top9 = [r[0] for r in rank[:9]]

    # ---- step 2: within-task failure clustering for top9 ----
    results = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    axes = axes.ravel()
    for ax, t in zip(axes, top9):
        rec = {"task": names[t]}
        fails = [r for r in rollouts if r["task"] == t and r["succ"] == 0]
        succs = [r for r in rollouts if r["task"] == t and r["succ"] == 1]
        for t_d, key in ((t_mean, "tmean"), (t_sig, "tsig")):
            Xf = np.stack([trunc_mean(r["z"], t_d) for r in fails])
            d = min(10, len(Xf) - 1)
            Xr = PCA(n_components=d, random_state=SEED).fit_transform(Xf)
            (bk, bsil, blab), krows = best_kmeans(Xr)
            gk = best_gmm_bic(Xr)
            rec[f"{key}_n_fail"] = len(Xf)
            rec[f"{key}_best_K_sil"] = bk
            rec[f"{key}_sil"] = round(bsil, 3) if bk else None
            rec[f"{key}_best_K_gmm_bic"] = gk[0]
            rec[f"{key}_km_silhouettes"] = {k: v[0] for k, v in krows.items()}
            if key == "tmean":
                blab_main = blab; Xr_main = Xr; n_succ = len(succs)
        # viz at t_mean: failures colored by cluster + success cloud (grey) for context
        Xf2 = PCA(n_components=2, random_state=SEED).fit_transform(
            np.stack([trunc_mean(r["z"], t_mean) for r in fails]))
        Xs2 = None
        if succs:
            # project successes into the SAME failure-PCA space for context
            pca = PCA(n_components=2, random_state=SEED).fit(np.stack([trunc_mean(r["z"], t_mean) for r in fails]))
            Xs2 = pca.transform(np.stack([trunc_mean(r["z"], t_mean) for r in succs]))
        if Xs2 is not None:
            ax.scatter(Xs2[:, 0], Xs2[:, 1], s=10, c="lightgrey", alpha=0.5, label="success")
        if blab_main is not None:
            ax.scatter(Xf2[:, 0], Xf2[:, 1], s=14, c=blab_main, cmap="tab10", alpha=0.8)
        else:
            ax.scatter(Xf2[:, 0], Xf2[:, 1], s=14, c="#d62728", alpha=0.8)
        ax.set_title(f"{names[t]}\nf{rec['tmean_n_fail']} bestK={rec['tmean_best_K_sil']} "
                     f"sil={rec['tmean_sil']} (tsig K={rec['tsig_best_K_sil']})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        results.append(rec)
    for ax in axes[len(top9):]:
        ax.axis("off")
    fig.suptitle(f"Within-task FAILURE clustering (top-9 separable tasks; PCA-2D of failures, color=cluster; "
                 f"grey=success). length-ctrl t_mean={t_mean}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "within_task_failure_clusters_top9.png", dpi=140); plt.close(fig)

    json.dump({"t_mean": t_mean, "t_mean1sig": t_sig,
               "separability_ranking": [{"task": nm, "auroc": a, "n_fail": nf, "n_succ": ns}
                                        for (t, nm, a, nf, ns) in rank],
               "top9_within_task": results}, open(OUT / "within_task_modes.json", "w"), indent=2)
    with (OUT / "within_task_modes.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "n_fail", "tmean_bestK", "tmean_sil", "tmean_gmmK",
                    "tsig_bestK", "tsig_sil", "tsig_gmmK"])
        for r in results:
            w.writerow([r["task"], r["tmean_n_fail"], r["tmean_best_K_sil"], r["tmean_sil"],
                        r["tmean_best_K_gmm_bic"], r["tsig_best_K_sil"], r["tsig_sil"],
                        r["tsig_best_K_gmm_bic"]])
    print("\n=== top9 within-task failure clustering ===")
    for r in results:
        print(f"  {r['task']:<28} f{r['tmean_n_fail']}  tmean: bestK={r['tmean_best_K_sil']} "
              f"sil={r['tmean_sil']} gmmK={r['tmean_best_K_gmm_bic']}  | "
              f"tsig: bestK={r['tsig_best_K_sil']} sil={r['tsig_sil']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
