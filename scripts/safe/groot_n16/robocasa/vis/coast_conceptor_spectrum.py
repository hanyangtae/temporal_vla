"""COAST Figure 3A reproduction — how low-rank is the success/fail subspace?

COAST Fig. 3A: eigenvalue spectra of the success (C+) and failure (C-)
conceptors decay rapidly; the contrastive C_steer = C+ AND NOT C- isolates an
effective subspace of ~1% of the hidden dimension. This script reproduces that
for GR00T N1.6 RoboCasa features and quantifies "how many dimensions drive
success/fail vs task discrimination" via conceptor effective rank.

Features are LENGTH-CONTROLLED: per rollout we take the mean of the first W env
steps (window 0), so success and failure are sampled from the same early
temporal support (see diagnose_length_confound.py). Fitting C+/C- on
episode-mean features would conflate the success-vs-timeout length artifact.

Effective-rank measures per conceptor C (eigenvalues in [0,1)):
  quota  q  = tr(C)/d                      (COAST App. A.10.2 Stage 1)
  n_eig>0.5                                (hard count of "on" directions)
  participation ratio PR = (Σλ)^2 / Σλ^2   (soft effective rank)

Outputs (analysis/visualizations/coast_spectrum/):
  spectrum_<base>.png    (A) C+/C-/C_steer/C_all spectra  (B) aperture sweep
                         (C) effective-rank bars: succ/fail vs overall(task)
  spectrum_<base>.json   spectra summary + per-alpha overlap/quota + eff. ranks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
from src.conceptor import (  # noqa: E402
    and_conceptor,
    compute_conceptor,
    conceptor_overlap,
    conceptor_quota,
    eigenvalue_spectrum,
    not_conceptor,
)


def participation_ratio(eig: np.ndarray) -> float:
    eig = np.clip(eig, 0, None)
    s1 = eig.sum()
    s2 = (eig ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def eff_rank_summary(C: np.ndarray) -> dict:
    eig = eigenvalue_spectrum(C)
    return {
        "quota": conceptor_quota(C),
        "n_eig_gt_0.5": int((eig > 0.5).sum()),
        "participation_ratio": participation_ratio(eig),
        "d": int(C.shape[0]),
    }


def parse_args():
    run_root = (
        REPO_ROOT
        / "outputs/eval/robocasa/groot_n16"
        / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache",
        type=Path,
        default=run_root / "analysis" / "feature_cache" / "pooled_all_hmean_dmean.npz",
    )
    p.add_argument("--out-root", type=Path, default=run_root / "analysis" / "visualizations")
    p.add_argument("--window", type=int, default=5, help="W env steps for the length-controlled window 0")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1, 2, 5, 10, 20])
    p.add_argument("--main-alpha", type=float, default=10.0, help="alpha for the headline spectrum panel")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def window0_features(cache: Path, W: int):
    c = np.load(cache, allow_pickle=True)
    feats = c["feats"].astype(np.float64)
    step_idx = c["step_idx"]; rollout_idx = c["rollout_idx"]
    ep_success = c["ep_success"]
    order = np.lexsort((step_idx, rollout_idx))
    rid = rollout_idx[order]
    uniq = np.unique(rollout_idx)
    starts = np.searchsorted(rid, uniq, side="left")
    ends = np.searchsorted(rid, uniq, side="right")
    X, succ = [], []
    for r, a, b in zip(uniq, starts, ends):
        rows = order[a:b]  # step-sorted
        if len(rows) < W:
            continue
        X.append(feats[rows[:W]].mean(0))
        succ.append(int(ep_success[int(r)]))
    return np.stack(X), np.asarray(succ)


def main():
    args = parse_args()
    base = args.cache.stem.replace("pooled_all_", "")
    X, succ = window0_features(args.cache, args.window)
    Xs, Xf, Xa = X[succ == 1], X[succ == 0], X
    print(f"window-0 (W={args.window}) features: succ={len(Xs)} fail={len(Xf)} d={X.shape[1]}")

    # ---- aperture sweep (COAST Stage 2): overlap(C+,C-) + quota(C_steer) ----
    sweep = []
    for a in args.alphas:
        Cs = compute_conceptor(Xs, a)
        Cf = compute_conceptor(Xf, a)
        Csteer = and_conceptor(Cs, not_conceptor(Cf))
        sweep.append({
            "alpha": a,
            "overlap_succ_fail": conceptor_overlap(Cs, Cf),
            "quota_succ": conceptor_quota(Cs),
            "quota_fail": conceptor_quota(Cf),
            "quota_steer": conceptor_quota(Csteer),
            "pr_steer": participation_ratio(eigenvalue_spectrum(Csteer)),
        })
        print(f"  alpha={a:>5}: overlap={sweep[-1]['overlap_succ_fail']:.3f} "
              f"q_steer={sweep[-1]['quota_steer']:.4f} PR_steer={sweep[-1]['pr_steer']:.1f}")

    # ---- headline conceptors at main alpha ----
    a = args.main_alpha
    Cs = compute_conceptor(Xs, a)
    Cf = compute_conceptor(Xf, a)
    Csteer = and_conceptor(Cs, not_conceptor(Cf))
    Call = compute_conceptor(Xa, a)  # all rollouts -> total variance (task-dominated)
    eig_s = eigenvalue_spectrum(Cs)
    eig_f = eigenvalue_spectrum(Cf)
    eig_st = eigenvalue_spectrum(Csteer)
    eig_all = eigenvalue_spectrum(Call)

    summary = {
        "window": args.window, "n_succ": int(len(Xs)), "n_fail": int(len(Xf)),
        "main_alpha": a,
        "C_success": eff_rank_summary(Cs),
        "C_failure": eff_rank_summary(Cf),
        "C_steer": eff_rank_summary(Csteer),
        "C_all_overall_task": eff_rank_summary(Call),
        "overlap_succ_fail": conceptor_overlap(Cs, Cf),
        "aperture_sweep": sweep,
    }
    print("\nEffective rank (main alpha):")
    for k in ("C_success", "C_failure", "C_steer", "C_all_overall_task"):
        s = summary[k]
        print(f"  {k:<20} quota={s['quota']:.4f}  n_eig>0.5={s['n_eig_gt_0.5']:>4}  "
              f"PR={s['participation_ratio']:.1f}  (of d={s['d']})")
    print(f"  C_steer effective subspace = {summary['C_steer']['n_eig_gt_0.5']} dims "
          f"= {100*summary['C_steer']['n_eig_gt_0.5']/X.shape[1]:.2f}% of d  "
          f"(COAST: ~1%)")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # A: spectra
    rank = np.arange(1, len(eig_s) + 1)
    axes[0].semilogy(rank, np.clip(eig_s, 1e-6, None), "-", color="#2ca02c", label="C+ (success)")
    axes[0].semilogy(rank, np.clip(eig_f, 1e-6, None), "--", color="#d62728", label="C- (failure)")
    axes[0].semilogy(rank, np.clip(eig_st, 1e-6, None), ":", color="#1f77b4", lw=2, label="C_steer = C+ ∧ ¬C-")
    axes[0].semilogy(rank, np.clip(eig_all, 1e-6, None), "-", color="#999999", lw=1, label="C_all (overall/task)")
    axes[0].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("eigenvalue rank"); axes[0].set_ylabel("eigenvalue (log)")
    axes[0].set_title(f"(A) conceptor spectra (alpha={a}) — low-rank decay")
    axes[0].legend(fontsize=9); axes[0].set_xlim(0, min(200, len(eig_s)))
    # B: aperture sweep
    al = [r["alpha"] for r in sweep]
    axes[1].plot(al, [r["overlap_succ_fail"] for r in sweep], "o-", color="#9467bd", label="overlap(C+,C-)")
    axes[1].axhspan(0.85, 0.95, color="#9467bd", alpha=0.12, label="COAST sweet-spot band")
    axes[1].set_xscale("log"); axes[1].set_xlabel("aperture alpha"); axes[1].set_ylabel("overlap")
    axes[1].set_title("(B) aperture sweep: success/fail subspace overlap")
    axes[1].legend(fontsize=9)
    # C: effective-rank bars
    keys = ["C_success", "C_failure", "C_steer", "C_all_overall_task"]
    labels = ["C+\nsuccess", "C-\nfailure", "C_steer\nsucc/fail disc.", "C_all\noverall(task)"]
    vals = [summary[k]["n_eig_gt_0.5"] for k in keys]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#999999"]
    bars = axes[2].bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
    axes[2].set_ylabel("# dims with eigenvalue > 0.5")
    axes[2].set_title(f"(C) effective dimensions (of d={X.shape[1]})")
    fig.suptitle(f"COAST Fig.3A — success/fail subspace is low-rank  |  window-0 (W={args.window}, length-controlled)  |  {base}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = args.out_root / "coast_spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"spectrum_{base}.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    json_path = out_dir / f"spectrum_{base}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {fig_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
