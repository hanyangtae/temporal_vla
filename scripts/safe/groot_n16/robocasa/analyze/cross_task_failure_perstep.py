"""Cross-task failure containment — PER-STEP representation (COAST-faithful).

Companion to cross_task_failure_analysis.py (which used rollout-mean vectors).
COAST fits conceptors on PER-STEP activations, so here each task's failure (and
success) conceptor is fit on the STACK of per-inference-step vectors z_t across
that task's rollouts. Tests whether the COAST claim (failure containment >
success containment across tasks) replicates with the per-step representation.

Reps:
  perstep_all     : all env steps of each rollout
  perstep_first16 : first 16 env steps only (length-controlled; failures all run
                    to the 45-step timeout, so 'all' is dominated by late states)

Conceptor (relative ridge): R=corr(X); C = R @ inv(R + ridge*tr(R)/d*I).
Containment(Ci,Cj) = tr(Ci·Cj)/tr(Ci^2)  (failure_i contained in failure_j, etc).

Outputs -> outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/perstep/
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
from scipy.linalg import subspace_angles

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from src.conceptor import compute_correlation  # noqa: E402
from temporal_agg import DEFAULT_CACHE, reconstruct  # noqa: E402

OUT = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/perstep"
REPS = ("perstep_all", "perstep_first16")


def rconceptor(X, ridge):
    R = compute_correlation(X)
    d = R.shape[0]
    C = R @ np.linalg.inv(R + ridge * np.trace(R) / d * np.eye(d))
    return 0.5 * (C + C.T)


def contain(Ci, Cj):
    return float(np.sum(Ci * Cj) / np.sum(Ci * Ci))


def offdiag_mean(M):
    n = M.shape[0]
    return float(np.mean(M[np.triu_indices(n, 1)]))


def heatmap(M, labels, title, path, vmin=0, vmax=1):
    fig, ax = plt.subplots(figsize=(0.5 * len(labels) + 3, 0.5 * len(labels) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def pools(rollouts, rep):
    """task -> {'fail': [Xf stacked], 'succ': [Xs stacked]} of per-step vectors."""
    per = {}
    for r in rollouts:
        z = r["z"] if rep == "perstep_all" else r["z"][:16]
        d = per.setdefault(r["task"], {"succ": [], "fail": []})
        d["succ" if r["succ"] == 1 else "fail"].append(z)
    return {t: {k: np.vstack(v).astype(np.float64) for k, v in dd.items()} for t, dd in per.items()}


def analyze(rollouts, names, ids, rep, ridge):
    P = pools(rollouts, rep)
    lab = [names[t] for t in ids]
    Cf = [rconceptor(P[t]["fail"], ridge) for t in ids]
    Cs = [rconceptor(P[t]["succ"], ridge) for t in ids]
    n = len(ids)
    Mf = np.array([[contain(Cf[i], Cf[j]) for j in range(n)] for i in range(n)])
    Ms = np.array([[contain(Cs[i], Cs[j]) for j in range(n)] for i in range(n)])
    Vf = [np.linalg.eigh(C)[1][:, -10:] for C in Cf]
    Vs = [np.linalg.eigh(C)[1][:, -10:] for C in Cs]
    PAf = np.array([[np.degrees(subspace_angles(Vf[i], Vf[j])).mean() for j in range(n)] for i in range(n)])
    PAs = np.array([[np.degrees(subspace_angles(Vs[i], Vs[j])).mean() for j in range(n)] for i in range(n)])
    return {
        "fail_contain": offdiag_mean(Mf), "succ_contain": offdiag_mean(Ms),
        "fail_princ_angle": offdiag_mean(PAf), "succ_princ_angle": offdiag_mean(PAs),
        "Mf": Mf, "Ms": Ms, "lab": lab,
        "n_fail_steps": int(sum(P[t]["fail"].shape[0] for t in ids)),
        "n_succ_steps": int(sum(P[t]["succ"].shape[0] for t in ids)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rollouts, names = reconstruct(DEFAULT_CACHE)
    ids = sorted(names)
    summary = {}
    for rep in REPS:
        print(f"\n===== {rep} =====")
        main_res = analyze(rollouts, names, ids, rep, 1e-3)
        heatmap(main_res["Mf"], main_res["lab"], f"failure containment per-step ({rep})",
                OUT / f"containment_failure_{rep}.png")
        heatmap(main_res["Ms"], main_res["lab"], f"success containment per-step ({rep})",
                OUT / f"containment_success_{rep}.png")
        ridge_sweep = {}
        for rg in (1e-4, 1e-3, 1e-2):
            r = analyze(rollouts, names, ids, rep, rg)
            ridge_sweep[f"{rg}"] = {"fail": round(r["fail_contain"], 4), "succ": round(r["succ_contain"], 4),
                                    "gap_fail_minus_succ": round(r["fail_contain"] - r["succ_contain"], 4)}
            print(f"  ridge {rg}: fail={r['fail_contain']:.4f} succ={r['succ_contain']:.4f} "
                  f"gap={r['fail_contain']-r['succ_contain']:+.4f}")
        summary[rep] = {
            "fail_contain": round(main_res["fail_contain"], 4),
            "succ_contain": round(main_res["succ_contain"], 4),
            "gap_fail_minus_succ": round(main_res["fail_contain"] - main_res["succ_contain"], 4),
            "fail_princ_angle_deg": round(main_res["fail_princ_angle"], 2),
            "succ_princ_angle_deg": round(main_res["succ_princ_angle"], 2),
            "n_fail_steps": main_res["n_fail_steps"], "n_succ_steps": main_res["n_succ_steps"],
            "ridge_sweep": ridge_sweep,
            "COAST_replicates_failure_gt_success": bool(main_res["fail_contain"] > main_res["succ_contain"]),
        }
        print(f"  => fail_contain={summary[rep]['fail_contain']} succ_contain={summary[rep]['succ_contain']} "
              f"COAST replicate(fail>succ)={summary[rep]['COAST_replicates_failure_gt_success']}")
    json.dump(summary, open(OUT / "perstep_containment.json", "w"), indent=2)
    with (OUT / "perstep_containment.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rep", "fail_contain", "succ_contain", "gap(fail-succ)", "fail_angle", "succ_angle",
                    "n_fail_steps", "n_succ_steps", "COAST_replicates"])
        for rep in REPS:
            s = summary[rep]
            w.writerow([rep, s["fail_contain"], s["succ_contain"], s["gap_fail_minus_succ"],
                        s["fail_princ_angle_deg"], s["succ_princ_angle_deg"],
                        s["n_fail_steps"], s["n_succ_steps"], s["COAST_replicates_failure_gt_success"]])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
