#!/usr/bin/env python3
"""Summarize GR00T N1.6 SAFE-LSTM aggregation ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_ROOT = REPO_ROOT / "outputs/eval/robocasa/groot_n16"
DEFAULT_LOG_ROOT = OUT_ROOT / "safe_train_logs_aggregation_ablation"
DEFAULT_JSON = OUT_ROOT / "safe_lstm_aggregation_ablation_summary.json"
DEFAULT_MD = OUT_ROOT / "safe_lstm_aggregation_ablation_summary.md"


def _parse_config(config_path: Path) -> dict[str, str | int]:
    text = config_path.read_text()

    def match_value(key: str) -> str:
        match = re.search(rf"^\s+{re.escape(key)}:\s+(.+?)\s*$", text, re.MULTILINE)
        if not match:
            match = re.search(rf"^{re.escape(key)}:\s+(.+?)\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing {key} in {config_path}")
        return match.group(1).strip().strip("'\"")

    seed_match = re.search(r"^\s+seed:\s+(\d+)\s*$", text, re.MULTILINE)
    if not seed_match:
        raise ValueError(f"Missing train.seed in {config_path}")

    return {
        "horizon_idx_rel": match_value("horizon_idx_rel"),
        "diff_idx_rel": match_value("diff_idx_rel"),
        "seed": int(seed_match.group(1)),
        "exp_suffix": match_value("exp_suffix"),
    }


def _auc_from_curve(points: list[tuple[float, float]]) -> float:
    by_fpr: dict[float, float] = {}
    for fpr, tpr in points:
        if math.isnan(fpr) or math.isnan(tpr):
            continue
        by_fpr[fpr] = max(by_fpr.get(fpr, 0.0), tpr)
    curve = sorted(by_fpr.items())
    if len(curve) < 2:
        return float("nan")
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc


def _read_threshold_metrics(csv_path: Path) -> dict[str, dict[str, float]]:
    rows_by_split: dict[str, list[dict[str, float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split_name"]
            rows_by_split[split].append(
                {
                    "threshold": float(row["threshold"]),
                    "avg_det_time": float(row["avg_det_time"]),
                    "tpr": float(row["tpr"]),
                    "tnr": float(row["tnr"]),
                    "acc": float(row["acc"]),
                    "bal_acc": float(row["bal_acc"]),
                    "f1": float(row["f1"]),
                }
            )

    metrics = {}
    for split, rows in rows_by_split.items():
        best = max(rows, key=lambda r: (r["bal_acc"], -r["avg_det_time"], r["f1"]))
        early_rows = [r for r in rows if r["avg_det_time"] <= 0.5]
        early_best = max(early_rows, key=lambda r: (r["bal_acc"], r["f1"])) if early_rows else None
        auc = _auc_from_curve([(1.0 - r["tnr"], r["tpr"]) for r in rows])
        metrics[split] = {
            "best_bal_acc": best["bal_acc"],
            "tdet_at_best": best["avg_det_time"],
            "tpr_at_best": best["tpr"],
            "tnr_at_best": best["tnr"],
            "f1_at_best": best["f1"],
            "threshold_at_best": best["threshold"],
            "roc_auc_maxsofar": auc,
            "best_bal_acc_tdet_le_0p5": early_best["bal_acc"] if early_best else float("nan"),
        }
    return metrics


def _mean_std(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    return {"mean": mean, "std": math.sqrt(var)}


def _fmt(stats: dict[str, float], digits: int = 3) -> str:
    mean = stats["mean"]
    std = stats["std"]
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _feature_dim(horizon: str, diff: str) -> int:
    factor = 1
    if horizon == "concat-2":
        factor *= 2
    if diff == "concat-2":
        factor *= 2
    return 1024 * factor


def collect_runs(log_root: Path) -> list[dict]:
    runs = []
    for ckpt_path in sorted(log_root.glob("*/**/model_final.ckpt")):
        run_dir = ckpt_path.parent
        csv_path = run_dir / "model_perf_vs_det.csv"
        config_path = run_dir / "config.yaml"
        if not csv_path.exists() or not config_path.exists():
            continue
        cfg = _parse_config(config_path)
        run = {
            **cfg,
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "metrics": _read_threshold_metrics(csv_path),
        }
        runs.append(run)
    return runs


def summarize_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["horizon_idx_rel"]), str(run["diff_idx_rel"]))].append(run)

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

    for (horizon, diff), group_runs in grouped.items():
        summary = {
            "horizon_idx_rel": horizon,
            "diff_idx_rel": diff,
            "feature_dim": _feature_dim(horizon, diff),
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
        "# SAFE-LSTM Aggregation Ablation Summary",
        "",
        "Selection rule: choose the aggregation with the highest mean `val_seen` max balanced accuracy from SAFE's saved threshold sweep. Ties are broken by higher `val_seen` max-so-far ROC-AUC and then earlier mean detection time.",
        "",
        f"Runs included: {len(runs)} checkpoints.",
        f"Best aggregation: `horizon_idx_rel={best['horizon_idx_rel']}`, `diff_idx_rel={best['diff_idx_rel']}`, feature dim `{best['feature_dim']}`.",
        "",
        "| rank | horizon | diff | dim | runs | val_seen bal-acc | val_seen T-det | val_seen ROC-AUC | val_unseen bal-acc | val_unseen T-det | val_unseen ROC-AUC |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, summary in enumerate(summaries, start=1):
        val_seen = summary["splits"]["val_seen"]
        val_unseen = summary["splits"]["val_unseen"]
        lines.append(
            "| {rank} | `{horizon}` | `{diff}` | {dim} | {runs} | {vs_bal} | {vs_tdet} | {vs_auc} | {vu_bal} | {vu_tdet} | {vu_auc} |".format(
                rank=idx,
                horizon=summary["horizon_idx_rel"],
                diff=summary["diff_idx_rel"],
                dim=summary["feature_dim"],
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
            "- `ROC-AUC` is computed from the saved max-so-far threshold curve, not from W&B's per-time `falert_early` / `falert_end` metrics.",
            "- All runs use the same SAFE-LSTM hyperparameters as the current reproduction run: `lr=3e-4`, `lambda_reg=1e-2`, `batch_size=64`, `n_epochs=1000`, seeds `0,1,2`.",
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
        raise SystemExit(f"No completed SAFE-LSTM aggregation runs found under {args.log_root}")
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
            f"horizon_idx_rel={best['horizon_idx_rel']}",
            f"diff_idx_rel={best['diff_idx_rel']}",
            f"feature_dim={best['feature_dim']}",
            f"val_seen_bal_acc={_fmt(best['splits']['val_seen']['best_bal_acc'])}",
        )


if __name__ == "__main__":
    main()
