"""A6: cross-task failure-mode sharing, WITH a success control.

Pool all FAILURE rollouts across tasks, cluster globally (KMeans, K sweep), and
ask whether clusters are SHARED across tasks (high task-entropy = a mode many
tasks fall into) or TASK-SPECIFIC (low entropy / task-pure). Then do the SAME on
SUCCESSES as a control: features here are task-identity-dominated, so if successes
show the same shared-vs-specific structure, the failure 'sharing' is not
failure-specific (it's scene/task structure), not a shared failure mode.

Length-clean: failures are all 45 steps; we use mean over first t_d=18 steps
(success mean) per rollout for both outcomes (consistent with within-task).

Per cluster c over 18 tasks: entropy H of the task distribution, normalized by
log(18). 'shared' if H/Hmax > 0.6; 'task-pure' if one task > 80%.

Outputs -> outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/mode_sharing/
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

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from core.io import reconstruct_rollouts as reconstruct  # noqa: E402

DEFAULT_CACHE = (REPO / "outputs/eval/robocasa/groot_n16"
                 / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
                 / "analysis/feature_cache/pooled_all_hmean_dmean.npz")
OUT = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/mode_sharing"
SEED = 0
T_D = 18
KS = [6, 8, 10, 12]


def features(rollouts, outcome, ids):
    X, tl = [], []
    for r in rollouts:
        if (r["succ"] == 1) != (outcome == "succ"):
            continue
        X.append(r["z"][:T_D].mean(0)); tl.append(r["task"])
    return np.stack(X).astype(np.float64), np.array(tl)


def cluster_stats(X, tl, ids, K):
    Xr = PCA(n_components=30, random_state=SEED).fit_transform(X)
    lab = KMeans(K, n_init=10, random_state=SEED).fit(Xr).labels_
    n_tasks = len(ids)
    Hmax = np.log(n_tasks)
    dist = np.zeros((K, n_tasks))
    ent = np.zeros(K)
    pure = np.zeros(K)
    for c in range(K):
        counts = np.array([np.sum(tl[lab == c] == t) for t in ids], float)
        dist[c] = counts
        p = counts / max(counts.sum(), 1)
        pp = p[p > 0]
        ent[c] = -(pp * np.log(pp)).sum()
        pure[c] = p.max()
    entn = ent / Hmax
    n_shared = int((entn > 0.6).sum())
    n_pure = int((pure > 0.8).sum())
    return dist, entn, pure, n_shared, n_pure, lab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rollouts, names = reconstruct(DEFAULT_CACHE)
    ids = sorted(names)
    lab_names = [names[t] for t in ids]

    summary = {"t_d": T_D, "by_K": {}}
    # cluster x task heatmaps at K=8
    figs = {}
    for outcome in ("fail", "succ"):
        X, tl = features(rollouts, outcome, ids)
        summary["by_K"].setdefault(outcome, {})
        for K in KS:
            dist, entn, pure, n_shared, n_pure, lab = cluster_stats(X, tl, ids, K)
            summary["by_K"][outcome][K] = {
                "n_shared_highentropy": n_shared, "n_task_pure": n_pure,
                "entropy_norm": [round(float(e), 3) for e in entn],
                "max_task_frac": [round(float(p), 3) for p in pure],
            }
            if K == 8:
                distn = dist / np.maximum(dist.sum(1, keepdims=True), 1)
                fig, ax = plt.subplots(figsize=(11, 5))
                im = ax.imshow(distn, cmap="magma", aspect="auto", vmin=0, vmax=distn.max())
                ax.set_xticks(range(len(ids))); ax.set_xticklabels(lab_names, rotation=90, fontsize=7)
                ax.set_yticks(range(K)); ax.set_yticklabels([f"cl{c} H/Hmax={entn[c]:.2f}" for c in range(K)], fontsize=7)
                ax.set_title(f"{outcome.upper()} global clusters x task (K=8, t_d={T_D})  "
                             f"shared(H/Hmax>.6)={n_shared}/{K}  task-pure={n_pure}/{K}")
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                fig.tight_layout(); fig.savefig(OUT / f"cluster_task_{outcome}_K8.png", dpi=140); plt.close(fig)

    # ---- comparison plot: shared-cluster count vs K, fail vs succ ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for outcome, col in (("fail", "#d62728"), ("succ", "#2ca02c")):
        sh = [summary["by_K"][outcome][K]["n_shared_highentropy"] for K in KS]
        pu = [summary["by_K"][outcome][K]["n_task_pure"] for K in KS]
        ax.plot(KS, sh, "o-", color=col, label=f"{outcome} shared (H/Hmax>0.6)")
        ax.plot(KS, pu, "s--", color=col, alpha=0.6, label=f"{outcome} task-pure (>80%)")
    ax.set_xlabel("K (global clusters)"); ax.set_ylabel("# clusters")
    ax.set_title("Cross-task mode sharing: failure vs success control\n"
                 "(fail shared >> succ shared => failure-specific cross-task modes)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mode_sharing_fail_vs_succ.png", dpi=140); plt.close(fig)

    json.dump(summary, open(OUT / "mode_sharing.json", "w"), indent=2)
    with (OUT / "mode_sharing.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["outcome", "K", "n_shared_highentropy", "n_task_pure"])
        for outcome in ("fail", "succ"):
            for K in KS:
                s = summary["by_K"][outcome][K]
                w.writerow([outcome, K, s["n_shared_highentropy"], s["n_task_pure"]])

    print(f"t_d={T_D}\n{'outcome':>7} {'K':>3} {'shared(H>.6)':>12} {'task-pure(>80%)':>16}")
    for outcome in ("fail", "succ"):
        for K in KS:
            s = summary["by_K"][outcome][K]
            print(f"{outcome:>7} {K:>3} {s['n_shared_highentropy']:>12} {s['n_task_pure']:>16}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
