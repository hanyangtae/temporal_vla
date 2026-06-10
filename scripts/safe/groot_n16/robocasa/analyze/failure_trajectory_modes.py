"""Failure trajectories THROUGH latent state-regions (time-resolved modes).

Per-step states (NOT rollout means) are clustered into 'regions'; each failure
rollout becomes a sequence of region labels over time, so we can see WITHIN-rollout
transitions (e.g. near-miss -> stuck). Regions defined on succ+fail per-step states
in TASK-WHITENED space (task-agnostic behavioral regions). The region where
SUCCESS final steps concentrate = 'goal region'; failures are classified by whether
they reach it and then leave.

Outputs -> outputs/eval/robocasa/groot_n16/cross_task_failure_analysis/trajectory_modes/
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
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
from sklearn.metrics import silhouette_score

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from core.distance import pooled_within_cov, whiten  # noqa: E402
from core.io import load_feature_cache, reconstruct_rollouts  # noqa: E402

RUN = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep"
CACHE = RUN / "analysis/feature_cache/pooled_all_hmean_dmean.npz"
RAW = RUN / "raw_rollouts"
OUT = RUN / "analysis/cross_task_failure_analysis/trajectory_modes"
SEED = 0
KS = [3, 4, 5, 6]


def build_states(rollouts):
    Xs = np.vstack([r["z"] for r in rollouts]).astype(np.float64)
    rid = np.concatenate([np.full(len(r["z"]), i) for i, r in enumerate(rollouts)])
    stp = np.concatenate([np.arange(len(r["z"])) for r in rollouts])
    Tl = np.concatenate([np.full(len(r["z"]), len(r["z"])) for r in rollouts])
    succ = np.concatenate([np.full(len(r["z"]), r["succ"]) for r in rollouts])
    task = np.concatenate([np.full(len(r["z"]), r["task"]) for r in rollouts])
    normt = stp / np.maximum(Tl - 1, 1)
    return Xs, rid, stp, Tl, succ, task, normt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = load_feature_cache(CACHE)
    rollouts, names = reconstruct_rollouts(cache)
    # rollouts[i] 는 np.unique(rollout_idx)[i] 에 해당하는 rollout (reconstruct_rollouts 와 동일 정렬).
    # ep_* 배열은 true rollout id 로 인덱싱되므로, 리스트 위치 i 가 아니라 rid_list[i] 로 접근해야
    # rollout_idx 가 0..N-1 비연속일 때도 task/episode/succ 가 어긋나지 않는다.
    rid_list = np.unique(cache["rollout_idx"])
    ep_task = cache["ep_task_id"]; ep_ep = cache["ep_episode_idx"]; ep_succ = cache["ep_success"]

    Xs, rid, stp, Tl, succ, task, normt = build_states(rollouts)
    print(f"per-step states: {Xs.shape}  (succ {int((succ==1).sum())}, fail {int((succ==0).sum())})")

    # ---- task-whiten (behavioral, task-agnostic) ----
    cov = pooled_within_cov(Xs, task, reg=1e-2)
    Xw = whiten(Xs, cov)
    Xr = PCA(n_components=30, random_state=SEED).fit_transform(Xw)  # for clustering
    X2 = PCA(n_components=2, random_state=SEED).fit_transform(Xw)   # canonical 2D for paths

    # ---- choose K by silhouette (subsample) ----
    rng = np.random.default_rng(SEED)
    sidx = rng.choice(len(Xr), size=min(6000, len(Xr)), replace=False)
    sils = {}
    for K in KS:
        lab = KMeans(K, n_init=5, random_state=SEED).fit(Xr).labels_
        sils[K] = round(float(silhouette_score(Xr[sidx], lab[sidx])), 3)
    K = max(sils, key=sils.get)
    print(f"region silhouettes {sils} -> K={K}")
    km = KMeans(K, n_init=10, random_state=SEED).fit(Xr)
    reg = km.labels_  # per-step region label

    # ---- goal region = where SUCCESS final steps concentrate ----
    is_final = stp == (Tl - 1)
    succ_final = is_final & (succ == 1)
    gc = Counter(reg[succ_final])
    goal_region = gc.most_common(1)[0][0]
    succ_final_dist = {int(c): int(n) for c, n in gc.items()}
    # region centroids (whitened) + distance to goal region centroid
    cent = np.stack([Xr[reg == c].mean(0) for c in range(K)])
    dist_to_goal = np.linalg.norm(cent - cent[goal_region], axis=1)
    # region task entropy + time occupancy
    Hmax = np.log(len(names))
    region_info = []
    for c in range(K):
        m = reg == c
        tcounts = np.array([np.sum(task[m] == t) for t in sorted(names)], float)
        p = tcounts / max(tcounts.sum(), 1); p = p[p > 0]
        H = float(-(p * np.log(p)).sum()) / Hmax
        region_info.append({"region": c, "n_steps": int(m.sum()),
                            "dist_to_goal": round(float(dist_to_goal[c]), 2),
                            "task_entropy": round(H, 3),
                            "mean_norm_time": round(float(normt[m].mean()), 3),
                            "frac_succ_steps": round(float((succ[m] == 1).mean()), 3)})
    print(f"goal_region={goal_region}  succ_final region dist={succ_final_dist}")
    for ri in region_info:
        print(f"  region {ri['region']}: n={ri['n_steps']} dist2goal={ri['dist_to_goal']} "
              f"H={ri['task_entropy']} meanT={ri['mean_norm_time']} fsucc={ri['frac_succ_steps']}")

    # ---- per-rollout region sequence + pattern classification ----
    def seq_of(i):
        m = rid == i
        order = np.argsort(stp[m])
        return reg[m][order]

    def classify(seqs, is_fail):
        rows = []
        for i in seqs:
            s = seqs[i]
            entered = bool(np.any(s == goal_region))
            last_reg = int(Counter(s[-5:]).most_common(1)[0][0])  # dominant of last 5
            ended_goal = last_reg == goal_region
            if not entered:
                pat = "never_goal(stuck)"
            elif ended_goal:
                pat = "reach_goal_stay"
            else:
                pat = "reach_goal_then_leave(near-miss->stuck)"
            rows.append((i, pat, last_reg, len(s)))
        return rows

    fail_ids = [i for i, r in enumerate(rollouts) if r["succ"] == 0]
    succ_ids = [i for i, r in enumerate(rollouts) if r["succ"] == 1]
    fail_seqs = {i: seq_of(i) for i in fail_ids}
    succ_seqs = {i: seq_of(i) for i in succ_ids}
    fail_rows = classify(fail_seqs, True)
    succ_rows = classify(succ_seqs, False)
    fail_pat = Counter(p for _, p, _, _ in fail_rows)
    succ_pat = Counter(p for _, p, _, _ in succ_rows)
    print(f"\nFAIL patterns: {dict(fail_pat)}")
    print(f"SUCC patterns: {dict(succ_pat)}")

    # ---- transition matrix (failures, consecutive steps) ----
    Tm = np.zeros((K, K))
    for i in fail_ids:
        s = fail_seqs[i]
        for a, b in zip(s[:-1], s[1:]):
            Tm[a, b] += 1
    Tmn = Tm / np.maximum(Tm.sum(1, keepdims=True), 1)

    # ---- viz ----
    # region scatter (PCA-2 of whitened), color=region; mark goal region centroid
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    sc = ax[0].scatter(X2[:, 0], X2[:, 1], s=2, c=reg, cmap="tab10", alpha=0.3, rasterized=True)
    for c in range(K):
        mu = X2[reg == c].mean(0)
        ax[0].scatter([mu[0]], [mu[1]], s=160, marker="*" if c == goal_region else "X",
                      c="black", zorder=5)
        ax[0].annotate(f"R{c}{'(GOAL)' if c==goal_region else ''}", mu, fontsize=8, weight="bold")
    ax[0].set_title(f"regions (PCA-2 whitened, K={K}); * = goal region")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    sc2 = ax[1].scatter(X2[:, 0], X2[:, 1], s=2, c=normt, cmap="viridis", alpha=0.3, rasterized=True)
    ax[1].set_title("same, colored by normalized time"); ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.colorbar(sc2, ax=ax[1], fraction=0.04)
    fig.tight_layout(); fig.savefig(OUT / f"regions_tsne_whiten_K{K}.png", dpi=130); plt.close(fig)

    # occupancy region x normalized-time (fail & succ)
    nb = 10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_, m, ttl in ((axes[0], succ == 0, "FAIL"), (axes[1], succ == 1, "SUCC")):
        occ = np.zeros((K, nb))
        tb = np.clip((normt[m] * nb).astype(int), 0, nb - 1)
        rr = reg[m]
        for b in range(nb):
            for c in range(K):
                occ[c, b] = np.sum((tb == b) & (rr == c))
        occ = occ / np.maximum(occ.sum(0, keepdims=True), 1)
        im = ax_.imshow(occ, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax_.set_yticks(range(K)); ax_.set_yticklabels([f"R{c}{'*' if c==goal_region else ''}" for c in range(K)])
        ax_.set_xlabel("normalized time bin"); ax_.set_title(f"{ttl} region occupancy over time")
        fig.colorbar(im, ax=ax_, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / "region_occupancy_time.png", dpi=130); plt.close(fig)

    # transition matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Tmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"R{c}{'*' if c==goal_region else ''}" for c in range(K)])
    ax.set_yticklabels([f"R{c}{'*' if c==goal_region else ''}" for c in range(K)])
    for a in range(K):
        for b in range(K):
            ax.text(b, a, f"{Tmn[a,b]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if Tmn[a, b] < 0.5 else "black")
    ax.set_xlabel("to region (t+1)"); ax.set_ylabel("from region (t)")
    ax.set_title("FAILURE region transition matrix (row-normalized)")
    fig.tight_layout(); fig.savefig(OUT / "region_transition_matrix.png", dpi=130); plt.close(fig)

    # example failure paths (a few per pattern) in PCA-2
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(X2[:, 0], X2[:, 1], s=1, c="lightgrey", alpha=0.3, rasterized=True)
    for c in range(K):
        mu = X2[reg == c].mean(0)
        ax.annotate(f"R{c}{'(GOAL)' if c==goal_region else ''}", mu, fontsize=8, weight="bold", color="navy")
    pat_examples = {}
    for pat in ["reach_goal_then_leave(near-miss->stuck)", "never_goal(stuck)", "reach_goal_stay"]:
        cand = [i for i, p, _, _ in fail_rows if p == pat]
        pat_examples[pat] = cand[:2]
    colmap = {"reach_goal_then_leave(near-miss->stuck)": "#d62728",
              "never_goal(stuck)": "#1f77b4", "reach_goal_stay": "#ff7f0e"}
    for pat, exs in pat_examples.items():
        for i in exs:
            m = rid == i; order = np.argsort(stp[m]); P = X2[m][order]
            ax.plot(P[:, 0], P[:, 1], "-", color=colmap[pat], alpha=0.7, lw=1.2)
            ax.scatter(P[0, 0], P[0, 1], c="green", s=30, zorder=6)  # start
            ax.scatter(P[-1, 0], P[-1, 1], c="black", s=30, marker="s", zorder=6)  # end
    ax.set_title("example FAILURE paths (green=start, black square=end)\n"
                 + " / ".join(f"{c.split('(')[0]}" for c in colmap))
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "example_failure_paths.png", dpi=130); plt.close(fig)

    # ---- mp4 representatives per pattern ----
    def mp4_of(i):
        rid_i = int(rid_list[i])
        tname = names[int(ep_task[rid_i])]
        return RAW / tname / f"task{int(ep_task[rid_i])}--ep{int(ep_ep[rid_i])}--succ{int(ep_succ[rid_i])}.mp4"

    with (OUT / "pattern_representatives.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pattern", "task", "episode", "region_seq(compressed)", "mp4", "mp4_exists"])
        for pat in colmap:
            cand = [i for i, p, _, _ in fail_rows if p == pat][:5]
            for i in cand:
                s = fail_seqs[i]
                comp = "→".join(str(x) for x in s[::3])  # every 3rd step
                p = mp4_of(i)
                w.writerow([pat, names[rollouts[i]["task"]], int(ep_ep[int(rid_list[i])]), comp, str(p), p.exists()])

    with (OUT / "trajectory_patterns.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rollout", "outcome", "task", "episode", "pattern", "last_region"])
        for i, pat, lr, _ in fail_rows:
            w.writerow([i, "fail", names[rollouts[i]["task"]], int(ep_ep[int(rid_list[i])]), pat, lr])

    json.dump({"K": K, "silhouettes": sils, "goal_region": int(goal_region),
               "succ_final_region_dist": succ_final_dist, "regions": region_info,
               "fail_patterns": dict(fail_pat), "succ_patterns": dict(succ_pat),
               "transition_matrix": np.round(Tmn, 3).tolist()},
              open(OUT / "trajectory_modes.json", "w"), indent=2)

    with (OUT / "summary.md").open("w") as f:
        f.write(f"# Failure trajectory modes (time-resolved)\n\nK={K} regions (silhouettes {sils}), "
                f"task-whitened per-step states. Goal region = R{goal_region} "
                f"(success-final concentration {succ_final_dist}).\n\n")
        f.write("## regions\n")
        for ri in region_info:
            f.write(f"- R{ri['region']}{'(GOAL)' if ri['region']==goal_region else ''}: "
                    f"n={ri['n_steps']} dist2goal={ri['dist_to_goal']} taskH={ri['task_entropy']} "
                    f"meanT={ri['mean_norm_time']} frac_succ={ri['frac_succ_steps']}\n")
        f.write(f"\n## failure trajectory patterns\n{dict(fail_pat)}\n")
        f.write(f"\n## success trajectory patterns (control)\n{dict(succ_pat)}\n")
        f.write("\nnear-miss->stuck = reach goal region then end elsewhere. Inspect "
                "pattern_representatives.tsv mp4s to confirm behaviorally.\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
