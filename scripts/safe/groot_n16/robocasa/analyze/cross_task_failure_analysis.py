"""Cross-task failure geometry analysis (COAST Sec 4.4) + failure-mode decomposition.

seen18 GR00T-N1.6 RoboCasa. Question: do failures share structure ACROSS tasks?

CRITICAL length confound: every failure runs to the 45-step timeout while successes
end early, so z_episode (full-episode mean) is length-confounded. We therefore run
the geometry analyses on BOTH z_episode and z_trunc16 (mean of first 16 steps,
length-controlled) and TRUST z_trunc16 for the verdict.

Reps: z_episode = z.mean(0); z_trunc16 = z[:16].mean(0); per rollout, D=1024.
Conceptor (relative ridge): R=corr(X); C = R @ inv(R + ridge*tr(R)/d*I).
Containment: tr(Ci·Cj)/tr(Ci^2).

Outputs -> outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/
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
from matplotlib.patches import Ellipse
from scipy.linalg import subspace_angles
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from src.conceptor import compute_correlation  # noqa: E402
from temporal_agg import DEFAULT_CACHE, reconstruct  # noqa: E402

OUT = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis"
SEED = 0
REPS = ("z_episode", "z_trunc16")


def rconceptor(X, ridge):
    R = compute_correlation(X)
    d = R.shape[0]
    C = R @ np.linalg.inv(R + ridge * np.trace(R) / d * np.eye(d))
    return 0.5 * (C + C.T)


def contain(Ci, Cj):
    return float(np.sum(Ci * Cj) / np.sum(Ci * Ci))


def offdiag_mean(M):
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return float(np.mean(M[iu]))


def load_reps():
    rollouts, names = reconstruct(DEFAULT_CACHE)
    data = {}
    for rep in REPS:
        per = {}
        for r in rollouts:
            z = r["z"]
            v = z.mean(0) if rep == "z_episode" else z[:16].mean(0)
            d = per.setdefault(r["task"], {"succ": [], "fail": []})
            d["succ" if r["succ"] == 1 else "fail"].append(v)
        data[rep] = {t: {k: np.asarray(v, float) for k, v in dd.items()} for t, dd in per.items()}
    return data, names, rollouts


def heatmap(M, labels, title, path, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(0.5 * len(labels) + 3, 0.5 * len(labels) + 2))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_matrix_tsv(M, labels, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([""] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + [f"{M[i, j]:.4f}" for j in range(len(labels))])


# ---------- Analysis 1: centroid distances ----------
def analysis1(data, ids, names, rep):
    lab = [names[t] for t in ids]
    fc = np.array([data[rep][t]["fail"].mean(0) for t in ids])
    sc = np.array([data[rep][t]["succ"].mean(0) for t in ids])
    Df = np.linalg.norm(fc[:, None] - fc[None], axis=-1)
    Ds = np.linalg.norm(sc[:, None] - sc[None], axis=-1)
    heatmap(Df, lab, f"failure centroid dist ({rep})", OUT / f"centroid_dist_failure_{rep}.png")
    heatmap(Ds, lab, f"success centroid dist ({rep})", OUT / f"centroid_dist_success_{rep}.png")
    write_matrix_tsv(Df, lab, OUT / f"centroid_distances_failure_{rep}.tsv")
    write_matrix_tsv(Ds, lab, OUT / f"centroid_distances_success_{rep}.tsv")
    return offdiag_mean(Df), offdiag_mean(Ds)


# ---------- Analysis 2: conceptor containment + principal angles ----------
def analysis2(data, ids, names, rep, ridge):
    lab = [names[t] for t in ids]
    Cf = [rconceptor(data[rep][t]["fail"], ridge) for t in ids]
    Cs = [rconceptor(data[rep][t]["succ"], ridge) for t in ids]
    n = len(ids)
    Mf = np.array([[contain(Cf[i], Cf[j]) for j in range(n)] for i in range(n)])
    Ms = np.array([[contain(Cs[i], Cs[j]) for j in range(n)] for i in range(n)])
    # principal angles on top-10 eigenvectors (mean angle in degrees; lower = more aligned)
    Vf = [np.linalg.eigh(C)[1][:, -10:] for C in Cf]
    Vs = [np.linalg.eigh(C)[1][:, -10:] for C in Cs]
    PAf = np.array([[np.degrees(subspace_angles(Vf[i], Vf[j])).mean() for j in range(n)] for i in range(n)])
    PAs = np.array([[np.degrees(subspace_angles(Vs[i], Vs[j])).mean() for j in range(n)] for i in range(n)])
    if abs(ridge - 1e-3) < 1e-12:  # save full artifacts only for the main ridge
        heatmap(Mf, lab, f"failure containment ({rep})", OUT / f"containment_failure_{rep}.png", 0, 1)
        heatmap(Ms, lab, f"success containment ({rep})", OUT / f"containment_success_{rep}.png", 0, 1)
        write_matrix_tsv(Mf, lab, OUT / f"containment_failure_{rep}.tsv")
        write_matrix_tsv(Ms, lab, OUT / f"containment_success_{rep}.tsv")
        write_matrix_tsv(PAf, lab, OUT / f"principal_angles_failure_{rep}.tsv")
        write_matrix_tsv(PAs, lab, OUT / f"principal_angles_success_{rep}.tsv")
    return offdiag_mean(Mf), offdiag_mean(Ms), offdiag_mean(PAf), offdiag_mean(PAs)


# ---------- Analysis 4: joint PCA + 2-sigma ellipses ----------
def joint_pca(data, ids, names, rep, outcome):
    lab = [names[t] for t in ids]
    allv = np.vstack([data[rep][t][outcome] for t in ids])
    p = PCA(n_components=2, random_state=SEED).fit(allv)
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = mpl.cm.get_cmap("tab20", 20)
    for k, t in enumerate(ids):
        Z = p.transform(data[rep][t][outcome])
        ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.25, color=cmap(k % 20))
        mu = Z.mean(0); cov = np.cov(Z.T)
        w, V = np.linalg.eigh(cov); o = w.argsort()[::-1]; w, V = w[o], V[:, o]
        ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        ax.add_patch(Ellipse(mu, 2 * 2 * np.sqrt(max(w[0], 1e-9)), 2 * 2 * np.sqrt(max(w[1], 1e-9)),
                             angle=ang, edgecolor=cmap(k % 20), facecolor="none", lw=1.4))
        ax.scatter([mu[0]], [mu[1]], s=80, marker="X", color=cmap(k % 20), edgecolors="k", linewidths=0.6)
    ax.set_title(f"Joint PCA of {outcome} ({rep}) — per-task 2σ ellipses (unsupervised)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(OUT / f"joint_pca_{outcome}_{rep}.png", dpi=140); plt.close(fig)


# ---------- Analysis 5: within-task failure modes ----------
def analysis5(data, ids, names, rep):
    rows = []
    for t in ids:
        X = data[rep][t]["fail"]
        n = len(X)
        d = min(10, n - 1)
        Xr = PCA(n_components=d, random_state=SEED).fit_transform(X)
        best_km = (None, -1)  # (K, silhouette)
        best_gmm = (None, np.inf, None)  # (K, BIC, covtype)
        per = {}
        for K in range(2, 7):
            if n <= K:
                continue
            km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xr)
            sil = silhouette_score(Xr, km.labels_) if len(set(km.labels_)) > 1 else -1
            db = davies_bouldin_score(Xr, km.labels_) if len(set(km.labels_)) > 1 else np.nan
            per[f"km{K}_sil"] = round(float(sil), 3); per[f"km{K}_db"] = round(float(db), 3)
            if sil > best_km[1]:
                best_km = (K, sil)
            for ct in ("full", "diag"):
                try:
                    g = GaussianMixture(K, covariance_type=ct, random_state=SEED, reg_covar=1e-4).fit(Xr)
                    bic = g.bic(Xr)
                    per[f"gmm{K}{ct[0]}_bic"] = round(float(bic), 1)
                    if bic < best_gmm[1]:
                        best_gmm = (K, bic, ct)
                except Exception:
                    pass
        rows.append({"task": names[t], "n_fail": n, "best_K_kmeans_sil": best_km[0],
                     "best_sil": round(best_km[1], 3),
                     "best_K_gmm_bic": best_gmm[0], "gmm_covtype": best_gmm[2], **per})
    with open(OUT / f"within_task_modes_{rep}.tsv", "w", newline="") as f:
        keys = ["task", "n_fail", "best_K_kmeans_sil", "best_sil", "best_K_gmm_bic", "gmm_covtype"]
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


# ---------- Analysis 6: cross-task mode sharing ----------
def analysis6(data, ids, names, rep, K=8):
    Xs, tlab = [], []
    for t in ids:
        Xs.append(data[rep][t]["fail"]); tlab += [t] * len(data[rep][t]["fail"])
    X = np.vstack(Xs); tlab = np.array(tlab)
    Xr = PCA(n_components=30, random_state=SEED).fit_transform(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xr)
    cl = km.labels_
    lab = [names[t] for t in ids]
    dist = np.zeros((K, len(ids)))
    ent = []
    for c in range(K):
        m = cl == c
        counts = np.array([np.sum(tlab[m] == t) for t in ids], float)
        dist[c] = counts
        p = counts / max(counts.sum(), 1)
        p = p[p > 0]
        ent.append(float(-(p * np.log(p)).sum()))
    ent = np.array(ent)
    ent_max = np.log(len(ids))
    n_shared = int(np.sum(ent / ent_max > 0.6))
    # heatmap of cluster x task (row-normalized)
    distn = dist / np.maximum(dist.sum(1, keepdims=True), 1)
    heatmap(distn, lab, f"global failure clusters x task ({rep}, K={K})  shared(H/Hmax>0.6)={n_shared}/{K}",
            OUT / f"cross_task_mode_sharing_{rep}.png", 0, distn.max(), "magma")
    with open(OUT / f"cross_task_mode_sharing_{rep}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cluster", "size", "entropy", "entropy_norm", "top_task"] + lab)
        for c in range(K):
            top = lab[int(dist[c].argmax())]
            w.writerow([c, int(dist[c].sum()), f"{ent[c]:.3f}", f"{ent[c]/ent_max:.3f}", top]
                       + [int(x) for x in dist[c]])
    return ent / ent_max, n_shared, K


# ---------- Bootstrap ----------
def bootstrap_a1(data, ids, rep, n_boot, rng):
    fF, sS = [], []
    for _ in range(n_boot):
        fc, sc = [], []
        for t in ids:
            Xf = data[rep][t]["fail"]; Xs = data[rep][t]["succ"]
            fc.append(Xf[rng.integers(0, len(Xf), len(Xf))].mean(0))
            sc.append(Xs[rng.integers(0, len(Xs), len(Xs))].mean(0))
        fc = np.array(fc); sc = np.array(sc)
        Df = np.linalg.norm(fc[:, None] - fc[None], axis=-1)
        Ds = np.linalg.norm(sc[:, None] - sc[None], axis=-1)
        fF.append(offdiag_mean(Df)); sS.append(offdiag_mean(Ds))
    return np.percentile(fF, [2.5, 97.5]), np.percentile(sS, [2.5, 97.5])


def bootstrap_a2(data, ids, rep, ridge, n_boot, rng):
    fF, sS = [], []
    for _ in range(n_boot):
        Cf, Cs = [], []
        for t in ids:
            Xf = data[rep][t]["fail"]; Xs = data[rep][t]["succ"]
            Cf.append(rconceptor(Xf[rng.integers(0, len(Xf), len(Xf))], ridge))
            Cs.append(rconceptor(Xs[rng.integers(0, len(Xs), len(Xs))], ridge))
        n = len(ids)
        Mf = np.array([[contain(Cf[i], Cf[j]) for j in range(n)] for i in range(n)])
        Ms = np.array([[contain(Cs[i], Cs[j]) for j in range(n)] for i in range(n)])
        fF.append(offdiag_mean(Mf)); sS.append(offdiag_mean(Ms))
    return np.percentile(fF, [2.5, 97.5]), np.percentile(sS, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, names, rollouts = load_reps()
    ids = sorted(names)
    # counts table
    with open(OUT / "task_counts.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["task", "n_succ", "n_fail", "SR"])
        for t in ids:
            ns = len(data["z_episode"][t]["succ"]); nf = len(data["z_episode"][t]["fail"])
            w.writerow([names[t], ns, nf, f"{ns/(ns+nf):.2f}"])
    excluded = [names[t] for t in ids if min(len(data["z_episode"][t]["succ"]),
                                             len(data["z_episode"][t]["fail"])) < 5]
    ids = [t for t in ids if min(len(data["z_episode"][t]["succ"]),
                                 len(data["z_episode"][t]["fail"])) >= 5]
    print(f"tasks included={len(ids)} excluded={excluded}")

    rng = np.random.default_rng(SEED)
    res = {}
    for rep in REPS:
        print(f"\n===== {rep} =====")
        fd, sd = analysis1(data, ids, names, rep)
        fc, sc, paf, pas = analysis2(data, ids, names, rep, 1e-3)
        # ridge sensitivity
        ridge_sweep = {}
        for rg in (1e-4, 1e-3, 1e-2):
            fcr, scr, _, _ = analysis2(data, ids, names, rep, rg)
            ridge_sweep[f"{rg}"] = {"fail_contain": round(fcr, 4), "succ_contain": round(scr, 4),
                                    "gap": round(fcr - scr, 4)}
            print(f"  ridge {rg}: fail_contain={fcr:.4f} succ_contain={scr:.4f} gap={fcr-scr:+.4f}")
        joint_pca(data, ids, names, rep, "fail")
        joint_pca(data, ids, names, rep, "succ")
        print("  bootstrapping A1 (1000) ...")
        f1ci, s1ci = bootstrap_a1(data, ids, rep, 1000, rng)
        print("  bootstrapping A2 (200) ...")
        f2ci, s2ci = bootstrap_a2(data, ids, rep, 1e-3, 200, rng)
        res[rep] = {
            "centroid_fail_mean": round(fd, 4), "centroid_succ_mean": round(sd, 4),
            "centroid_fail_ci": [round(x, 4) for x in f1ci], "centroid_succ_ci": [round(x, 4) for x in s1ci],
            "contain_fail_mean": round(fc, 4), "contain_succ_mean": round(sc, 4),
            "contain_fail_ci": [round(x, 4) for x in f2ci], "contain_succ_ci": [round(x, 4) for x in s2ci],
            "princ_angle_fail_deg": round(paf, 2), "princ_angle_succ_deg": round(pas, 2),
            "ridge_sweep": ridge_sweep,
            "verdict_location_fail_gt_succ": bool(fd > sd),
            "verdict_direction_fail_gt_succ": bool(fc > sc),
        }
        print(f"  centroid: fail={fd:.3f} succ={sd:.3f}  | containment: fail={fc:.3f} succ={sc:.3f}")

    # within-task + cross-task on z_trunc16 (primary) and z_episode
    modes = {rep: analysis5(data, ids, names, rep) for rep in REPS}
    sharing = {rep: analysis6(data, ids, names, rep, K=8) for rep in REPS}
    bestK_dist = {rep: {} for rep in REPS}
    for rep in REPS:
        from collections import Counter
        c = Counter(r["best_K_kmeans_sil"] for r in modes[rep])
        bestK_dist[rep] = dict(sorted(c.items()))

    json.dump({"included_tasks": [names[t] for t in ids], "excluded": excluded,
               "geometry": res,
               "within_task_bestK_kmeans": bestK_dist,
               "cross_task_shared_clusters": {rep: f"{sharing[rep][1]}/{sharing[rep][2]}" for rep in REPS}},
              open(OUT / "results.json", "w"), indent=2)

    # ---- summary.md ----
    def tbl(rep):
        r = res[rep]
        return (f"| Failure (pair) | {r['centroid_fail_mean']}  CI{r['centroid_fail_ci']} | "
                f"{r['contain_fail_mean']}  CI{r['contain_fail_ci']} |\n"
                f"| Success (pair) | {r['centroid_succ_mean']}  CI{r['centroid_succ_ci']} | "
                f"{r['contain_succ_mean']}  CI{r['contain_succ_ci']} |")
    with open(OUT / "summary.md", "w") as f:
        f.write("# Cross-task failure geometry — seen18 GR00T-N1.6\n\n")
        f.write(f"Tasks included: {len(ids)}/18. Excluded (S or F <5): {excluded or 'none'}.\n\n")
        f.write("**Length confound**: all failures=45-step timeout, successes early-stop. "
                "z_episode is length-confounded; **z_trunc16 (first-16 mean) is the trustworthy rep**.\n\n")
        for rep in REPS:
            r = res[rep]
            f.write(f"## 2x2 — {rep}\n\n")
            f.write("| | location (centroid dist) | direction (containment) |\n|---|---|---|\n")
            f.write(tbl(rep) + "\n\n")
            f.write(f"- location: failure>{'>' if r['verdict_location_fail_gt_succ'] else '<='}success "
                    f"({r['centroid_fail_mean']} vs {r['centroid_succ_mean']})\n")
            f.write(f"- direction: failure containment {'>' if r['verdict_direction_fail_gt_succ'] else '<='} "
                    f"success ({r['contain_fail_mean']} vs {r['contain_succ_mean']})\n")
            f.write(f"- principal angle (deg, lower=more aligned): fail {r['princ_angle_fail_deg']} / "
                    f"succ {r['princ_angle_succ_deg']}\n")
            f.write(f"- ridge sweep: {r['ridge_sweep']}\n\n")
        rt = res["z_trunc16"]
        coast = rt["verdict_location_fail_gt_succ"] and rt["verdict_direction_fail_gt_succ"]
        f.write(f"## COAST verdict (length-controlled z_trunc16)\n\n")
        f.write(f"'different location + shared direction' (failure spread in location AND higher "
                f"failure containment): **{'HOLDS' if coast else 'does NOT fully hold'}** "
                f"(location fail>succ={rt['verdict_location_fail_gt_succ']}, "
                f"direction fail>succ={rt['verdict_direction_fail_gt_succ']}).\n\n")
        f.write("## within-task failure modes (best K, kmeans silhouette)\n")
        for rep in REPS:
            f.write(f"- {rep}: K distribution {bestK_dist[rep]}\n")
        f.write("\n## cross-task mode sharing (global K=8 failure clusters)\n")
        for rep in REPS:
            f.write(f"- {rep}: shared (H/Hmax>0.6) = {sharing[rep][1]}/{sharing[rep][2]}\n")
        f.write("\n*Note*: within/cross-task clustering uses PCA(10/30)-reduced features "
                "(1024-d clustering on <100 points is degenerate).\n")
    print("\nwrote", OUT)
    print("verdict z_trunc16:", res["z_trunc16"]["verdict_location_fail_gt_succ"],
          res["z_trunc16"]["verdict_direction_fail_gt_succ"])


if __name__ == "__main__":
    main()
