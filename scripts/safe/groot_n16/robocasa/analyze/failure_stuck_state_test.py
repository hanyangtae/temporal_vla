"""Decisive test: is cross-task failure 'sharing' the shared LATE timeout/stuck state?

Slide a fixed window (W steps) along the trajectory and measure cross-task
containment of per-task failure conceptors built from THAT window only. All
failures are exactly 45 steps, so window-start = fraction of the failure
trajectory. Hypothesis (length/stuck artifact): failure containment RISES toward
late windows and becomes family-UNSTRUCTURED (within≈cross) — i.e. failures of all
tasks converge to a common late 'stuck/timeout' state. Successes are short (mean
~18), so late windows have few/no success rollouts (the asymmetry is itself the point).

Outputs -> outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/stuck_test/
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

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

OUT = REPO / "outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/cross_task_failure_analysis/stuck_test"
W = 5
STARTS = list(range(0, 45, 5))  # [0,5,...,40] -> 9 windows of 5 steps
RIDGE = 1e-3
MIN_ROLL = 5  # min rollouts (with full window) per task to include


def family_of(name):
    n = name.lower()
    if n.startswith("pickplace") or n.startswith("pnp"):
        return "pnp"
    for p in ("close", "open", "turn", "navigate", "slide"):
        if n.startswith(p):
            return p
    return "other"


def rconceptor(X, ridge=RIDGE):
    R = compute_correlation(X)
    d = R.shape[0]
    C = R @ np.linalg.inv(R + ridge * np.trace(R) / d * np.eye(d))
    return 0.5 * (C + C.T)


def contain(Ci, Cj):
    return float(np.sum(Ci * Cj) / np.sum(Ci * Ci))


def offdiag_within_cross(M, fams):
    n = M.shape[0]
    od, wi, cr = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            od.append(M[i, j])
            (wi if fams[i] == fams[j] else cr).append(M[i, j])
    return (float(np.mean(od)), float(np.mean(wi)) if wi else np.nan,
            float(np.mean(cr)) if cr else np.nan)


def window_pool(rollouts, ids, outcome, s, w):
    """task -> stacked per-step vectors in window [s:s+w); require full window."""
    per = {t: [] for t in ids}
    for r in rollouts:
        if (r["succ"] == 1) != (outcome == "succ"):
            continue
        if len(r["z"]) >= s + w:
            per[r["task"]].append(r["z"][s:s + w])
    return {t: (np.vstack(v) if v else None) for t, v in per.items()}, \
           {t: len(v) for t, v in per.items()}


def containment_curve(rollouts, ids, fams, outcome):
    rows = []
    for s in STARTS:
        pool, nroll = window_pool(rollouts, ids, outcome, s, W)
        keep = [i for i, t in enumerate(ids) if pool[t] is not None and nroll[t] >= MIN_ROLL]
        if len(keep) < 3:
            rows.append({"start": s, "n_tasks": len(keep), "offdiag": None,
                         "within": None, "cross": None, "n_roll_min": 0})
            continue
        kept_ids = [ids[i] for i in keep]; kept_fams = [fams[i] for i in keep]
        C = [rconceptor(pool[t]) for t in kept_ids]
        M = np.array([[contain(C[a], C[b]) for b in range(len(C))] for a in range(len(C))])
        od, wi, cr = offdiag_within_cross(M, kept_fams)
        rows.append({"start": s, "n_tasks": len(keep), "offdiag": round(od, 4),
                     "within": round(wi, 4) if wi == wi else None,
                     "cross": round(cr, 4) if cr == cr else None,
                     "n_roll_min": int(min(nroll[t] for t in kept_ids))})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rollouts, names = reconstruct(DEFAULT_CACHE)
    ids = sorted(names)
    fams = [family_of(names[t]) for t in ids]

    fail = containment_curve(rollouts, ids, fams, "fail")
    succ = containment_curve(rollouts, ids, fams, "succ")

    # ---- plot 1: containment vs window position ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fx = [r["start"] + W / 2 for r in fail if r["offdiag"] is not None]
    fy = [r["offdiag"] for r in fail if r["offdiag"] is not None]
    sx = [r["start"] + W / 2 for r in succ if r["offdiag"] is not None]
    sy = [r["offdiag"] for r in succ if r["offdiag"] is not None]
    axes[0].plot(fx, fy, "o-", color="#d62728", label="failure")
    axes[0].plot(sx, sy, "s-", color="#2ca02c", label="success (where ≥5 rollouts)")
    axes[0].set_xlabel("window center (env step) — failures are all 45 long")
    axes[0].set_ylabel("cross-task containment (off-diag mean)")
    axes[0].set_title("Cross-task containment along trajectory\n(rise toward late = shared stuck/timeout state)")
    axes[0].legend()
    # plot 2: within vs cross family for failures
    fwi = [r["within"] for r in fail if r["within"] is not None]
    fcr = [r["cross"] for r in fail if r["cross"] is not None]
    fwx = [r["start"] + W / 2 for r in fail if r["within"] is not None]
    axes[1].plot(fwx, fwi, "o-", color="#9467bd", label="failure within-family")
    axes[1].plot(fwx, fcr, "s--", color="#8c564b", label="failure cross-family")
    axes[1].set_xlabel("window center (env step)")
    axes[1].set_ylabel("failure containment")
    axes[1].set_title("Failure within vs cross family along trajectory\n(converging/equal late = task-agnostic stuck)")
    axes[1].legend()
    fig.suptitle("Stuck-state test: cross-task failure containment by trajectory window (W=5)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "stuck_state_containment_curve.png", dpi=140); plt.close(fig)

    json.dump({"W": W, "starts": STARTS, "failure": fail, "success": succ},
              open(OUT / "stuck_test.json", "w"), indent=2)
    with (OUT / "stuck_test.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["window", "outcome", "n_tasks", "offdiag_contain", "within_fam", "cross_fam", "n_roll_min"])
        for tag, rows in (("fail", fail), ("succ", succ)):
            for r in rows:
                w.writerow([f"{r['start']}-{r['start']+W}", tag, r["n_tasks"], r["offdiag"],
                            r["within"], r["cross"], r["n_roll_min"]])

    print("FAILURE containment by window (start-end : offdiag  within/cross  n_tasks):")
    for r in fail:
        print(f"  {r['start']:>2}-{r['start']+W:<2}  offdiag={r['offdiag']}  "
              f"within={r['within']} cross={r['cross']}  n_tasks={r['n_tasks']}")
    print("SUCCESS containment by window:")
    for r in succ:
        print(f"  {r['start']:>2}-{r['start']+W:<2}  offdiag={r['offdiag']}  n_tasks={r['n_tasks']} (min_roll={r['n_roll_min']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
