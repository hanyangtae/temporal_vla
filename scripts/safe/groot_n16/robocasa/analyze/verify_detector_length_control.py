#!/usr/bin/env python3
"""Length-controlled verification of the seen18 SAFE-LSTM failure detector.

seen18 failures all run to the 45-step timeout while successes terminate early,
so any time-pooled / "by final end" separation is partly a length artifact
(rollout length alone gives AUROC ~0.998). The honest question is whether the
detector carries an *early* failure signal once length is controlled.

This reads ``per_rollout_scores.csv`` produced by ``finalize_lstm_detector.py``
and reports, per split (val_seen / val_unseen):

* detector AUROC at three eval times:
    - ``score_at_earliest_stop``  (single frame at task_min_step-1, length-ctrl)
    - ``score_by_earliest_stop``  (max-so-far capped at task_min_step, length-ctrl)
    - ``score_by_final_end``      (max-so-far over full rollout, length-CONFOUNDED)
* a **length-only baseline**: AUROC using rollout step-count as the score, which
  isolates how much the confounded number is explained by length alone;
* a **permutation null** (label shuffles) giving a p-value and a 95% null band,
  because with small per-task samples chance != 0.5.

It also reports per-task AUROC on ``val_unseen`` at ``by earliest stop`` — the
cleanest within-task, length-controlled number.

Verdict: the detector shows a genuine early-failure signal iff the length-
controlled AUROC on val_unseen significantly exceeds its permutation null
(and is not merely tracking length, which is neutralized under the per-task cap).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from run_config import FINAL_DETECTOR_DIR  # noqa: E402

EVAL_SCORE_COLS = {
    "at earliest stop (len-ctrl)": "score_at_earliest_stop",
    "by earliest stop (len-ctrl)": "score_by_earliest_stop",
    "by final end (confounded)": "score_by_final_end",
}


def auroc_with_null(
    y: np.ndarray, score: np.ndarray, n_perm: int, rng: np.random.Generator
) -> dict[str, float]:
    """AUROC of (higher score -> failure) with a label-permutation null."""
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "p": float("nan"), "null_lo": float("nan"), "null_hi": float("nan"), "n": len(y), "n_fail": int(y.sum())}
    obs = roc_auc_score(y, score)
    if n_perm <= 0:
        return {"auroc": float(obs), "p": float("nan"), "null_lo": float("nan"), "null_hi": float("nan"), "n": int(len(y)), "n_fail": int(y.sum())}
    null = np.empty(n_perm)
    y_perm = y.copy()
    for i in range(n_perm):
        rng.shuffle(y_perm)
        null[i] = roc_auc_score(y_perm, score)
    # two-sided-ish: how often shuffled AUROC is at least as extreme (away from 0.5)
    p = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (n_perm + 1)
    lo, hi = np.percentile(null, [2.5, 97.5])
    return {"auroc": float(obs), "p": float(p), "null_lo": float(lo), "null_hi": float(hi), "n": int(len(y)), "n_fail": int(y.sum())}


def fmt_row(label: str, r: dict[str, float]) -> str:
    sig = "*" if (r["p"] < 0.05 and not np.isnan(r["p"])) else " "
    return (
        f"| {label:<30} | {r['auroc']:.3f} | [{r['null_lo']:.3f}, {r['null_hi']:.3f}] "
        f"| {r['p']:.3f}{sig} | {r['n']:>4} | {r['n_fail']:>4} |"
    )


def split_block(df: pd.DataFrame, split: str, n_perm: int, rng: np.random.Generator) -> list[str]:
    sub = df[df["split"] == split]
    y = sub["label_failure"].to_numpy().astype(int)
    lines = [
        f"### {split}  (n={len(sub)}, fail={int(y.sum())}, succ={int((1 - y).sum())})",
        "",
        "| score | AUROC | null 95% band | perm p | n | n_fail |",
        "|---|---|---|---|---|---|",
    ]
    for label, col in EVAL_SCORE_COLS.items():
        lines.append(fmt_row(label, auroc_with_null(y, sub[col].to_numpy(), n_perm, rng)))
    # length-only baseline (the artifact)
    lines.append(fmt_row("length-only baseline", auroc_with_null(y, sub["length"].to_numpy().astype(float), n_perm, rng)))
    lines.append("")
    return lines


def per_task_block(df: pd.DataFrame, split: str, score_col: str, n_perm: int, rng: np.random.Generator) -> list[str]:
    sub = df[df["split"] == split]
    lines = [
        f"### per-task {split} @ {score_col} (length-controlled within task)",
        "",
        "| task_id | AUROC | null 95% band | perm p | n | n_fail | length-AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for task_id in sorted(sub["task_id"].unique()):
        ts = sub[sub["task_id"] == task_id]
        y = ts["label_failure"].to_numpy().astype(int)
        r = auroc_with_null(y, ts[score_col].to_numpy(), n_perm, rng)
        # within-task the cap makes length near-constant; report it to confirm neutralization
        len_auroc = auroc_with_null(y, ts["length"].to_numpy().astype(float), 0, rng)["auroc"] \
            if len(np.unique(y)) >= 2 else float("nan")
        sig = "*" if (r["p"] < 0.05 and not np.isnan(r["p"])) else " "
        lines.append(
            f"| {task_id} | {r['auroc']:.3f} | [{r['null_lo']:.3f}, {r['null_hi']:.3f}] "
            f"| {r['p']:.3f}{sig} | {r['n']:>4} | {r['n_fail']:>3} | {len_auroc:.3f} |"
        )
    lines.append("")
    return lines


def make_plot(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["val_seen", "val_unseen"]
    labels = list(EVAL_SCORE_COLS.keys()) + ["length baseline"]
    cols = list(EVAL_SCORE_COLS.values()) + ["length"]
    fig, axes = plt.subplots(1, len(splits), figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, splits):
        sub = df[df["split"] == split]
        y = sub["label_failure"].to_numpy().astype(int)
        vals = []
        for col in cols:
            vals.append(roc_auc_score(y, sub[col].to_numpy()) if len(np.unique(y)) >= 2 else np.nan)
        colors = ["#1f77b4", "#1f77b4", "#d62728", "#888888"]
        bars = ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{split} (n={len(sub)}, fail={int(y.sum())})")
        ax.set_ylim(0.0, 1.0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("AUROC (failure = positive)")
    fig.suptitle("seen18 SAFE-LSTM: length-controlled vs confounded failure AUROC")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final-dir", type=Path, default=FINAL_DETECTOR_DIR)
    p.add_argument("--scores-csv", type=Path, default=None, help="Defaults to <final-dir>/per_rollout_scores.csv")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scores_csv = args.scores_csv or (args.final_dir / "per_rollout_scores.csv")
    if not scores_csv.exists():
        raise FileNotFoundError(scores_csv)
    df = pd.read_csv(scores_csv)
    rng = np.random.default_rng(args.seed)

    lines = [
        "# seen18 SAFE-LSTM length-controlled verification",
        "",
        f"source: `{scores_csv}`",
        "",
        "AUROC orientation: higher detector score = predicted failure; `label_failure=1` for failures. "
        "`*` marks permutation p<0.05. Length-controlled eval times cap the max-so-far at each task's "
        "`task_min_step` (shortest rollout of that task), removing the failure-is-longer advantage.",
        "",
        "## Pooled per split",
        "",
    ]
    for split in ("val_seen", "val_unseen"):
        lines += split_block(df, split, args.n_perm, rng)

    lines += ["## Per-task (length-controlled)", ""]
    lines += per_task_block(df, "val_unseen", "score_by_earliest_stop", args.n_perm, rng)

    # verdict
    sub = df[df["split"] == "val_unseen"]
    y = sub["label_failure"].to_numpy().astype(int)
    a_ctrl = auroc_with_null(y, sub["score_by_earliest_stop"].to_numpy(), args.n_perm, rng)
    a_conf = roc_auc_score(y, sub["score_by_final_end"].to_numpy())
    a_len = roc_auc_score(y, sub["length"].to_numpy().astype(float))
    verdict = (
        "GENUINE early signal" if (a_ctrl["auroc"] > a_ctrl["null_hi"] and a_ctrl["p"] < 0.05)
        else "NOT distinguishable from length/chance"
    )
    lines += [
        "## Verdict (val_unseen, held-out tasks)",
        "",
        f"- length-controlled (by earliest stop) AUROC: **{a_ctrl['auroc']:.3f}** "
        f"(null 95% [{a_ctrl['null_lo']:.3f}, {a_ctrl['null_hi']:.3f}], p={a_ctrl['p']:.3f})",
        f"- confounded (by final end) AUROC: {a_conf:.3f}",
        f"- length-only baseline AUROC: {a_len:.3f}",
        "",
        f"**=> {verdict}.**",
        "",
    ]

    out_md = args.final_dir / "length_control_verification.md"
    out_png = args.final_dir / "length_control_verification.png"
    args.final_dir.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    make_plot(df, out_png)
    print("\n".join(lines))
    print(f"\nwrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
