#!/usr/bin/env python3
"""Plot SAFE-style conformal prediction curves for GR00T N1.6.

SAFE's original ``scripts/eval_conformal_figure.py`` reads WandB summary
tables. Our GR00T N1.6 final detector writes the same core quantities to local
CSV files, so this adapter keeps the paper-figure plotting semantics while
using the local ``final_detector`` artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_FINAL_DIR = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/conformal_figure"
)

METRIC_DISPLAY = {
    "bal_acc": "Bal-acc: Balanced Accuracy",
    "avg_det_time": "Average Detection Time (Normalized)",
    "f1": "F1 Score",
    "tpr": "TPR: True Positive Rate",
    "tnr": "TNR: True Negative Rate",
    "fpr": "FPR: False Positive Rate",
    "fnr": "FNR: False Negative Rate",
    "alpha": "Significance Level alpha",
}

SERIES_ORDER = (
    ("split_cp", "neg_success"),
    ("functional_cp", "neg_success"),
    ("split_cp", "pos_failure"),
)

SERIES_DISPLAY = {
    ("split_cp", "neg_success"): "split CP, calibrated on success",
    ("split_cp", "pos_failure"): "split CP, calibrated on failure",
    ("functional_cp", "neg_success"): "functional CP band, calibrated on success",
}

SERIES_STYLE = {
    ("split_cp", "neg_success"): {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    ("split_cp", "pos_failure"): {"color": "#d62728", "marker": "s", "linestyle": "-"},
    ("functional_cp", "neg_success"): {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--eval-time",
        default="by final end",
        choices=("at earliest stop", "by earliest stop", "by final end", "all"),
        help="Evaluation time slice to plot. Use 'all' to emit one subdirectory per slice.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--include-fixed-threshold",
        action="store_true",
        help="Overlay the val_seen-selected fixed-threshold baseline when available.",
    )
    return parser.parse_args()


def load_inputs(final_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, dict | None]:
    split_cp_path = final_dir / "split_cp_eval.csv"
    if not split_cp_path.exists():
        raise FileNotFoundError(split_cp_path)
    split_df = pd.read_csv(split_cp_path)
    split_df["method"] = "split_cp"
    split_df["fpr"] = 1.0 - split_df["tnr"]
    split_df["fnr"] = 1.0 - split_df["tpr"]

    dfs = [split_df]
    functional_cp_path = final_dir / "functional_cp_eval.csv"
    if functional_cp_path.exists():
        functional_df = pd.read_csv(functional_cp_path)
        functional_df["method"] = "functional_cp"
        if "fpr" not in functional_df.columns:
            functional_df["fpr"] = 1.0 - functional_df["tnr"]
        if "fnr" not in functional_df.columns:
            functional_df["fnr"] = 1.0 - functional_df["tpr"]
        dfs.append(functional_df)
    df = pd.concat(dfs, ignore_index=True, sort=False)

    fixed_path = final_dir / "fixed_threshold_eval.csv"
    fixed_df = pd.read_csv(fixed_path) if fixed_path.exists() else None

    final_op_path = final_dir / "final_operating_point.json"
    final_op = json.loads(final_op_path.read_text()) if final_op_path.exists() else None
    return df, fixed_df, final_op


def save_figure(fig: plt.Figure, path_no_suffix: Path, formats: list[str]) -> list[Path]:
    saved = []
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = path_no_suffix.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def plot_balacc_tdet(
    df: pd.DataFrame,
    fixed_df: pd.DataFrame | None,
    final_op: dict | None,
    *,
    eval_time: str,
    include_fixed_threshold: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 3.3), dpi=300)

    for method, calib_label in SERIES_ORDER:
        group = df[(df["method"] == method) & (df["calib_label"] == calib_label)].sort_values(
            "avg_det_time"
        )
        if group.empty:
            continue
        style = SERIES_STYLE[(method, calib_label)]
        ax.plot(
            group["avg_det_time"],
            group["bal_acc"],
            linewidth=1.5 if calib_label == "neg_success" else 1.0,
            markersize=4,
            alpha=0.85,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=SERIES_DISPLAY[(method, calib_label)],
        )

    if final_op is not None and final_op.get("eval_time") == eval_time:
        ax.scatter(
            [final_op["avg_det_time"]],
            [final_op["bal_acc"]],
            marker="*",
            s=95,
            color="black",
            linewidths=0.5,
            label=f"final op: alpha={final_op['alpha']}",
            zorder=5,
        )

    if include_fixed_threshold and fixed_df is not None:
        fixed_unseen = fixed_df[fixed_df["split"] == "val_unseen"]
        if not fixed_unseen.empty:
            row = fixed_unseen.iloc[0]
            ax.scatter(
                [row["avg_det_time"]],
                [row["bal_acc"]],
                marker="x",
                s=55,
                color="#555555",
                label="fixed threshold baseline",
                zorder=4,
            )

    ax.set_xlabel(METRIC_DISPLAY["avg_det_time"])
    ax.set_ylabel(METRIC_DISPLAY["bal_acc"])
    ax.set_title(f"GR00T N1.6 SAFE-LSTM CP ({eval_time})")
    max_x = max(0.7, float(df["avg_det_time"].max()) * 1.05)
    ax.set_xlim(0.0, max_x)
    ax.set_ylim(bottom=0.45, top=1.03)
    ax.grid(True, linewidth=0.35, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_metric_vs_alpha(df: pd.DataFrame, final_op: dict | None, *, metric: str, eval_time: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.3, 3.3), dpi=300)

    for method, calib_label in SERIES_ORDER:
        group = df[(df["method"] == method) & (df["calib_label"] == calib_label)].sort_values(
            "alpha"
        )
        if group.empty:
            continue
        style = SERIES_STYLE[(method, calib_label)]
        ax.plot(
            group["alpha"],
            group[metric],
            linewidth=1.5 if calib_label == "neg_success" else 1.0,
            markersize=4,
            alpha=0.85,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=SERIES_DISPLAY[(method, calib_label)],
        )

    if final_op is not None and final_op.get("eval_time") == eval_time and metric in final_op:
        ax.scatter(
            [final_op["alpha"]],
            [final_op[metric]],
            marker="*",
            s=90,
            color="black",
            linewidths=0.5,
            label="final op",
            zorder=5,
        )

    if metric == "tnr":
        ax.plot([0, 1], [1, 0], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    elif metric == "fpr":
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(METRIC_DISPLAY["alpha"])
    ax.set_ylabel(METRIC_DISPLAY[metric])
    ax.set_title(f"{metric} vs alpha ({eval_time})")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, linewidth=0.35, alpha=0.25)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return fig


def write_manifest(
    out_dir: Path,
    *,
    final_dir: Path,
    eval_time: str,
    formats: list[str],
    saved_paths: list[Path],
    final_op: dict | None,
) -> None:
    lines = [
        "# GR00T N1.6 SAFE Figure-8-Style CP Plots",
        "",
        f"- source final detector: `{final_dir}`",
        f"- evaluation time: `{eval_time}`",
        f"- output formats: `{', '.join(formats)}`",
        "",
        "These plots mirror SAFE's Figure 8 plotting target: balanced accuracy and",
        "error-rate curves over conformal prediction operating points. The input is",
        "the local `split_cp_eval.csv` and `functional_cp_eval.csv`, not WandB summary tables.",
        "",
    ]
    if final_op is not None:
        lines.extend(
            [
                "## Final Operating Point",
                "",
                f"- alpha: `{final_op['alpha']}`",
                f"- calibration label: `{final_op['calib_label']}`",
                f"- threshold: `{final_op['threshold']}`",
                f"- val_unseen bal-acc: `{final_op['bal_acc']}`",
                f"- val_unseen TPR/TNR: `{final_op['tpr']}` / `{final_op['tnr']}`",
                f"- val_unseen mean T-det: `{final_op['avg_det_time']}`",
                "",
            ]
        )
    lines.extend(["## Files", ""])
    lines.extend(f"- `{path.name}`" for path in saved_paths)
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def plot_one_eval_time(
    df_all: pd.DataFrame,
    fixed_df: pd.DataFrame | None,
    final_op: dict | None,
    *,
    final_dir: Path,
    out_dir: Path,
    eval_time: str,
    formats: list[str],
    include_fixed_threshold: bool,
) -> list[Path]:
    df = df_all[df_all["eval_time"] == eval_time].copy()
    if df.empty:
        raise ValueError(f"No rows for eval_time={eval_time}")

    eval_slug = eval_time.replace(" ", "_")
    save_dir = out_dir / eval_slug
    saved_paths: list[Path] = []

    fig = plot_balacc_tdet(
        df,
        fixed_df,
        final_op,
        eval_time=eval_time,
        include_fixed_threshold=include_fixed_threshold,
    )
    saved_paths.extend(save_figure(fig, save_dir / "cp_balacc_tdet", formats))

    for metric in ("fpr", "fnr", "tpr", "tnr", "bal_acc"):
        fig = plot_metric_vs_alpha(df, final_op, metric=metric, eval_time=eval_time)
        saved_paths.extend(save_figure(fig, save_dir / f"cp_alpha_{metric}", formats))

    write_manifest(
        save_dir,
        final_dir=final_dir,
        eval_time=eval_time,
        formats=formats,
        saved_paths=saved_paths,
        final_op=final_op if final_op and final_op.get("eval_time") == eval_time else None,
    )
    return saved_paths


def main() -> None:
    args = parse_args()
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("At least one output format is required")

    df, fixed_df, final_op = load_inputs(args.final_dir)
    eval_times = sorted(df["eval_time"].unique().tolist()) if args.eval_time == "all" else [args.eval_time]

    all_saved: list[Path] = []
    for eval_time in eval_times:
        all_saved.extend(
            plot_one_eval_time(
                df,
                fixed_df,
                final_op,
                final_dir=args.final_dir,
                out_dir=args.out_dir,
                eval_time=eval_time,
                formats=formats,
                include_fixed_threshold=args.include_fixed_threshold,
            )
        )

    print(f"Saved {len(all_saved)} figure files under {args.out_dir}")
    for path in all_saved:
        print(path)


if __name__ == "__main__":
    main()
