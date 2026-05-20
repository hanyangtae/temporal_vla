#!/usr/bin/env python3
"""Summarize GR00T N1.6 SAFE-LSTM hyperparameter sweep runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from summarize_lstm_aggregation_ablation import _fmt, _mean_std, _read_threshold_metrics


REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_ROOT = REPO_ROOT / "outputs/eval/robocasa/groot_n16"
RUN_ROOT = OUT_ROOT / "safe_seen4_unseen2_100ep"
DEFAULT_LOG_ROOT = RUN_ROOT / "experiments/hparam_sweep/train_logs"
DEFAULT_JSON = RUN_ROOT / "experiments/hparam_sweep/reports/safe_lstm_hparam_sweep_summary.json"
DEFAULT_MD = RUN_ROOT / "experiments/hparam_sweep/reports/safe_lstm_hparam_sweep_summary.md"


def _match_value(text: str, key: str) -> str:
    match = re.search(rf"^\s+{re.escape(key)}:\s+(.+?)\s*$", text, re.MULTILINE)
    if not match:
        match = re.search(rf"^{re.escape(key)}:\s+(.+?)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {key}")
    return match.group(1).strip().strip("'\"")


def _parse_config(config_path: Path) -> dict[str, str | int | float]:
    text = config_path.read_text()
    return {
        "horizon_idx_rel": _match_value(text, "horizon_idx_rel"),
        "diff_idx_rel": _match_value(text, "diff_idx_rel"),
        "lr": float(_match_value(text, "lr")),
        "lambda_reg": float(_match_value(text, "lambda_reg")),
        "seed": int(_match_value(text, "seed")),
        "exp_suffix": _match_value(text, "exp_suffix"),
    }


def collect_runs(log_root: Path) -> list[dict]:
    runs = []
    for ckpt_path in sorted(log_root.glob("*/**/model_final.ckpt")):
        run_dir = ckpt_path.parent
        csv_path = run_dir / "model_perf_vs_det.csv"
        config_path = run_dir / "config.yaml"
        if not csv_path.exists() or not config_path.exists():
            continue
        cfg = _parse_config(config_path)
        runs.append(
            {
                **cfg,
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt_path),
                "metrics": _read_threshold_metrics(csv_path),
            }
        )
    return runs


def summarize_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(float(run["lr"]), float(run["lambda_reg"]))].append(run)

    summaries = []
    metric_names = (
        "best_bal_acc",
        "tdet_at_best",
        "tpr_at_best",
        "tnr_at_best",
        "f1_at_best",
        "roc_auc_maxsofar",
        "best_bal_acc_tdet_le_0p5",
    )
    split_names = ("train", "val_seen", "val_unseen")

    for (lr, lambda_reg), group_runs in grouped.items():
        summary = {
            "lr": lr,
            "lambda_reg": lambda_reg,
            "horizon_idx_rel": group_runs[0]["horizon_idx_rel"],
            "diff_idx_rel": group_runs[0]["diff_idx_rel"],
            "n_runs": len(group_runs),
            "seeds": sorted(int(r["seed"]) for r in group_runs),
            "splits": {},
        }
        for split in split_names:
            split_metrics = {}
            for metric in metric_names:
                values = [
                    r["metrics"][split][metric]
                    for r in group_runs
                    if split in r["metrics"] and metric in r["metrics"][split]
                ]
                split_metrics[metric] = _mean_std(values)
            summary["splits"][split] = split_metrics
        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda s: (
            s["splits"]["val_seen"]["best_bal_acc"]["mean"],
            s["splits"]["val_seen"]["roc_auc_maxsofar"]["mean"],
            -s["splits"]["val_seen"]["tdet_at_best"]["mean"],
        ),
        reverse=True,
    )


def write_markdown(path: Path, summaries: list[dict], runs: list[dict]) -> None:
    best = summaries[0]
    lines = [
        "# SAFE-LSTM Hyperparameter Sweep Summary",
        "",
        "Selection rule: choose the setting with the highest mean `val_seen` max balanced accuracy from SAFE's saved threshold sweep. Ties are broken by higher `val_seen` max-so-far ROC-AUC and then earlier mean detection time.",
        "",
        f"Runs included: {len(runs)} checkpoints.",
        f"Fixed aggregation: `horizon_idx_rel={best['horizon_idx_rel']}`, `diff_idx_rel={best['diff_idx_rel']}`.",
        f"Best hyperparameters: `lr={best['lr']:.0e}`, `lambda_reg={best['lambda_reg']:.0e}`.",
        "",
        "| rank | lr | lambda_reg | runs | val_seen bal-acc | val_seen T-det | val_seen ROC-AUC | val_unseen bal-acc | val_unseen T-det | val_unseen ROC-AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, summary in enumerate(summaries, start=1):
        val_seen = summary["splits"]["val_seen"]
        val_unseen = summary["splits"]["val_unseen"]
        lines.append(
            "| {rank} | `{lr:.0e}` | `{reg:.0e}` | {runs} | {vs_bal} | {vs_tdet} | {vs_auc} | {vu_bal} | {vu_tdet} | {vu_auc} |".format(
                rank=idx,
                lr=summary["lr"],
                reg=summary["lambda_reg"],
                runs=summary["n_runs"],
                vs_bal=_fmt(val_seen["best_bal_acc"]),
                vs_tdet=_fmt(val_seen["tdet_at_best"]),
                vs_auc=_fmt(val_seen["roc_auc_maxsofar"]),
                vu_bal=_fmt(val_unseen["best_bal_acc"]),
                vu_tdet=_fmt(val_unseen["tdet_at_best"]),
                vu_auc=_fmt(val_unseen["roc_auc_maxsofar"]),
            )
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- `T-det` is the mean relative first detection time on ground-truth failure rollouts at the threshold that maximizes balanced accuracy for that split.",
            "- `ROC-AUC` is computed from the saved max-so-far threshold curve.",
            "- This sweep does not change rollout features. It only retrains SAFE-LSTM on the selected aggregation.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    runs = collect_runs(args.log_root)
    summaries = summarize_runs(runs)
    if not summaries:
        raise SystemExit(f"No completed SAFE-LSTM hparam sweep runs found under {args.log_root}")
    payload = {
        "log_root": str(args.log_root),
        "selection_rule": "max mean val_seen best_bal_acc, then val_seen roc_auc_maxsofar, then earlier val_seen tdet_at_best",
        "runs": runs,
        "summaries": summaries,
        "best": summaries[0] if summaries else None,
    }
    args.json_out.write_text(json.dumps(payload, indent=2))
    write_markdown(args.md_out, summaries, runs)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    if summaries:
        best = summaries[0]
        print(
            "Best:",
            f"lr={best['lr']:.0e}",
            f"lambda_reg={best['lambda_reg']:.0e}",
            f"val_seen_bal_acc={_fmt(best['splits']['val_seen']['best_bal_acc'])}",
        )


if __name__ == "__main__":
    main()
