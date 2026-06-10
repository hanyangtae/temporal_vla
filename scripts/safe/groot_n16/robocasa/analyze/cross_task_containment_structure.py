"""Cross-task failure containment STRUCTURE — what overlaps how (per-step).

Extends cross_task_failure_perstep.py: same per-step conceptor containment, but
(1) length control truncates at t_d = round(success_len_mean + 1*std) [data-driven,
    less aggressive than the fixed-16 cut], plus the confounded all-step rep for
    comparison;
(2) tasks are GROUPED BY FAMILY (close/open/pnp/turn/navigate/slide/other) so the
    18x18 containment matrices are readable as blocks; we report WITHIN-family vs
    CROSS-family mean containment and the top task pairs.

We include BOTH the confounded (perstep_all) and length-controlled (perstep_meansig)
results so we can see which overlaps are genuine vs timeout-driven.

Outputs -> outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/structure/
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

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from src.conceptor import compute_correlation  # noqa: E402
from temporal_agg import DEFAULT_CACHE, reconstruct  # noqa: E402

OUT = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/structure"


def family_of(name):
    n = name.lower()
    if n.startswith("pickplace") or n.startswith("pnp"):
        return "pnp"
    for p in ("close", "open", "turn", "navigate", "slide"):
        if n.startswith(p):
            return p
    return "other"


def rconceptor(X, ridge=1e-3):
    R = compute_correlation(X)
    d = R.shape[0]
    C = R @ np.linalg.inv(R + ridge * np.trace(R) / d * np.eye(d))
    return 0.5 * (C + C.T)


def contain(Ci, Cj):
    return float(np.sum(Ci * Cj) / np.sum(Ci * Ci))


def pools(rollouts, td):
    """task -> {'fail':X,'succ':X} stacked per-step; td=None means all steps."""
    per = {}
    for r in rollouts:
        z = r["z"] if td is None else r["z"][:td]
        d = per.setdefault(r["task"], {"succ": [], "fail": []})
        d["succ" if r["succ"] == 1 else "fail"].append(z)
    return {t: {k: np.vstack(v).astype(np.float64) for k, v in dd.items()} for t, dd in per.items()}


def containment_matrix(P, ids, outcome, ridge=1e-3):
    C = [rconceptor(P[t][outcome], ridge) for t in ids]
    n = len(ids)
    return np.array([[contain(C[i], C[j]) for j in range(n)] for i in range(n)])


def block_means(M, fams):
    """within-family mean (off-diag, same family) and cross-family mean."""
    n = M.shape[0]
    wi, cr = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (wi if fams[i] == fams[j] else cr).append(M[i, j])
    return float(np.mean(wi)) if wi else float("nan"), float(np.mean(cr)) if cr else float("nan")


def heatmap(M, labels, fams, title, path):
    fig, ax = plt.subplots(figsize=(0.55 * len(labels) + 3, 0.55 * len(labels) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=M.min(), vmax=M.max())
    tick = [f"[{fams[i]}] {labels[i]}" for i in range(len(labels))]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(tick, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(tick, fontsize=6)
    # family boundary lines
    bounds = [k for k in range(1, len(labels)) if fams[k] != fams[k - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=0.8); ax.axvline(b - 0.5, color="white", lw=0.8)
    ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def top_pairs(M, labels, fams, k=10):
    n = M.shape[0]
    pairs = [(i, j, 0.5 * (M[i, j] + M[j, i])) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda x: -x[2])
    return [(labels[i], labels[j], round(v, 3), "within" if fams[i] == fams[j] else "cross")
            for i, j, v in pairs[:k]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rollouts, names = reconstruct(DEFAULT_CACHE)
    ids_raw = sorted(names)
    fams_raw = {t: family_of(names[t]) for t in ids_raw}
    # order tasks by family then name for block structure
    famorder = ["close", "open", "pnp", "turn", "slide", "navigate", "other"]
    ids = sorted(ids_raw, key=lambda t: (famorder.index(fams_raw[t]), names[t]))
    labels = [names[t] for t in ids]
    fams = [fams_raw[t] for t in ids]

    # length control t_d = success mean + 1 std
    succ_len = np.array([len(r["z"]) for r in rollouts if r["succ"] == 1])
    td = int(round(succ_len.mean() + succ_len.std()))
    print(f"success length mean={succ_len.mean():.1f} std={succ_len.std():.1f} -> t_d(mean+1sigma)={td}")
    print(f"family order: {[f'{f}:{sum(1 for x in fams if x==f)}' for f in famorder if f in fams]}")

    reps = {"perstep_all": None, f"perstep_meansig(t{td})": td}
    summary = {}
    for rep, t_d in reps.items():
        P = pools(rollouts, t_d)
        Mf = containment_matrix(P, ids, "fail")
        Ms = containment_matrix(P, ids, "succ")
        heatmap(Mf, labels, fams, f"FAILURE containment ({rep}) — family blocks",
                OUT / f"containment_failure_{rep.split('(')[0]}.png")
        heatmap(Ms, labels, fams, f"SUCCESS containment ({rep}) — family blocks",
                OUT / f"containment_success_{rep.split('(')[0]}.png")
        fwi, fcr = block_means(Mf, fams)
        swi, scr = block_means(Ms, fams)
        tp_f = top_pairs(Mf, labels, fams)
        summary[rep] = {
            "fail_offdiag_mean": round(float(np.mean(Mf[np.triu_indices(len(ids), 1)])), 4),
            "succ_offdiag_mean": round(float(np.mean(Ms[np.triu_indices(len(ids), 1)])), 4),
            "fail_within_family": round(fwi, 4), "fail_cross_family": round(fcr, 4),
            "succ_within_family": round(swi, 4), "succ_cross_family": round(scr, 4),
            "fail_within_minus_cross": round(fwi - fcr, 4),
            "top_failure_pairs": tp_f,
        }
        print(f"\n== {rep} ==")
        print(f"  FAIL containment: within-family={fwi:.4f}  cross-family={fcr:.4f}  (gap {fwi-fcr:+.4f})")
        print(f"  SUCC containment: within-family={swi:.4f}  cross-family={scr:.4f}")
        print(f"  fail off-diag mean={summary[rep]['fail_offdiag_mean']} succ={summary[rep]['succ_offdiag_mean']}")
        print("  top failure-containment pairs:")
        for a, b, v, wc in tp_f:
            print(f"    {a:<26} <-> {b:<26} {v:.3f} ({wc})")

    json.dump({"t_d_mean_plus_1sigma": td, "summary": summary}, open(OUT / "structure.json", "w"), indent=2)
    with (OUT / "structure_summary.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rep", "fail_within_fam", "fail_cross_fam", "fail_within-cross",
                    "succ_within_fam", "succ_cross_fam", "fail_offdiag", "succ_offdiag"])
        for rep, s in summary.items():
            w.writerow([rep, s["fail_within_family"], s["fail_cross_family"], s["fail_within_minus_cross"],
                        s["succ_within_family"], s["succ_cross_family"],
                        s["fail_offdiag_mean"], s["succ_offdiag_mean"]])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
