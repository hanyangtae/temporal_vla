"""length — episode-length confound diagnostic + honest early-detection curve.

Documents that failures all run to the time limit (length predicts fail ~0.998),
so time-pooled separation is a length artifact; then the length-controlled signal:
at a FIXED absolute frame t, per-task CV-AUROC for eventual failure (macro).
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
from sklearn.metrics import roc_auc_score

from core import io, metrics, schemes
from analyses._common import add_common

NAME = "length"
HELP = "episode-length confound + length-controlled early-detection AUROC curve"


def add_args(p):
    add_common(p)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 16, 20])
    p.add_argument("--pca-dim", type=int, default=30)
    p.add_argument("--min-task-n", type=int, default=10)


def _base(cache: Path) -> str:
    return Path(cache).stem.replace("pooled_all_", "")


def run(args):
    rollouts, names = io.reconstruct_rollouts(args.cache)
    L = np.array([len(r["z"]) for r in rollouts])
    es = np.array([r["succ"] for r in rollouts])
    ids = sorted({r["task"] for r in rollouts})

    confound = {
        "succ_len_mean": float(L[es == 1].mean()), "succ_len_std": float(L[es == 1].std()),
        "fail_len_mean": float(L[es == 0].mean()), "fail_len_std": float(L[es == 0].std()),
        "max_len": int(L.max()),
        "frac_fail_at_maxlen": float((L[es == 0] == L.max()).mean()),
        "frac_succ_at_maxlen": float((L[es == 1] == L.max()).mean()),
        "auroc_length_only": float(roc_auc_score(1 - es, L)),
    }
    print("length confound:", json.dumps(confound, indent=2))

    curve = []
    for t in args.steps:
        built = schemes.build_at(rollouts, "abs", t)
        if built is None:
            continue
        X, succ, task = built
        aus = []
        for tid in ids:
            m = task == tid
            if m.sum() < args.min_task_n or len(np.unique(succ[m])) < 2:
                continue
            a, _ = metrics.cv_auroc(X[m], (succ[m] == 0).astype(int), pca_dim=args.pca_dim, seed=args.seed)
            if a is not None:
                aus.append(a)
        curve.append({"t": int(t), "n_alive": int(len(succ)),
                      "frac_fail_alive": float((succ == 0).mean()),
                      "macro_auroc": float(np.mean(aus)) if aus else None, "n_tasks": len(aus)})
        print(f"  t={t:>3} n_alive={len(succ):>5} %fail={np.mean(succ==0)*100:4.1f} "
              f"macroAUROC={curve[-1]['macro_auroc']}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(L[es == 1], bins=range(0, L.max() + 2), alpha=0.6, color="#2ca02c", label="success")
    axes[0].hist(L[es == 0], bins=range(0, L.max() + 2), alpha=0.6, color="#d62728", label="fail")
    axes[0].set_xlabel("episode length (steps)"); axes[0].set_ylabel("# rollouts")
    axes[0].set_title(f"length distribution (length-only AUROC={confound['auroc_length_only']:.3f})")
    axes[0].legend()
    ts = [d["t"] for d in curve if d["macro_auroc"] is not None]
    au = [d["macro_auroc"] for d in curve if d["macro_auroc"] is not None]
    axes[1].axhline(0.5, color="grey", ls="--", lw=1)
    axes[1].plot(ts, au, "o-", color="#1f77b4")
    axes[1].set_ylim(0.45, 1.0); axes[1].set_xlabel("fixed absolute timestep t (length-controlled)")
    axes[1].set_ylabel("per-task CV logistic AUROC (macro)")
    axes[1].set_title("causal early failure prediction\n(eventual failure from features at step t)")
    base = _base(args.cache)
    fig.suptitle(f"Length confound & honest early-detection signal  |  {base}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_dir = args.out_root / "length_confound"; out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"length_confound_{base}.png", dpi=140); plt.close(fig)
    json.dump({"confound": confound, "early_detection_curve": curve},
              open(out_dir / f"length_confound_{base}.json", "w"), indent=2)
    print(f"wrote {out_dir}/length_confound_{base}.{{png,json}}")
