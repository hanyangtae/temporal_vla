#!/usr/bin/env python3
"""Pin the selected GR00T N1.6 SAFE-LSTM checkpoint and evaluate CP thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from run_config import (  # noqa: E402
    FINAL_DIFF_IDX_REL,
    FINAL_DETECTOR_DIR,
    FINAL_HORIZON_IDX_REL,
    FINAL_LAMBDA_REG,
    FINAL_LR,
    FINAL_LSTM_RUN_DIR,
    FINAL_SEED,
    SAFE_ROOT as CONFIG_SAFE_ROOT,
)


DEFAULT_SAFE_ROOT = Path(os.environ.get("SAFE_ROOT", str(CONFIG_SAFE_ROOT)))
DEFAULT_RUN_DIR = FINAL_LSTM_RUN_DIR
DEFAULT_FINAL_DIR = FINAL_DETECTOR_DIR


EVAL_TIMES = ("at earliest stop", "by earliest stop", "by final end")
ALPHAS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9)
FINAL_OPERATING_POINT = {
    "threshold_method": "split_cp",
    "eval_time": "by final end",
    "alpha": 0.2,
    "calib_label": "neg_success",
}


def _score_at_eval_time(scores: list[np.ndarray], rollouts: list[Any], eval_time: str) -> list[float]:
    if eval_time == "at earliest stop":
        return [float(s[r.task_min_step - 1]) for s, r in zip(scores, rollouts)]
    if eval_time == "by earliest stop":
        return [float(s[: r.task_min_step].max()) for s, r in zip(scores, rollouts)]
    if eval_time == "by final end":
        return [float(s[: len(r.hidden_states)].max()) for s, r in zip(scores, rollouts)]
    raise ValueError(f"Unknown eval_time: {eval_time}")


def _threshold_curve_rows(
    rollouts: list[Any],
    scores: list[np.ndarray],
    split_name: str,
    threshold: float,
) -> dict[str, float | str]:
    labels = [1 - r.episode_success for r in rollouts]
    test_scores = [float(s[: len(r.hidden_states)].max()) for s, r in zip(scores, rollouts)]
    result = eval_binary_classification(test_scores, labels, threshold)
    result["avg_det_time"] = eval_detection_time(scores, labels, threshold)
    return {
        "split": split_name,
        "threshold": float(threshold),
        "tpr": float(result["tpr"]),
        "tnr": float(result["tnr"]),
        "acc": float(result["acc"]),
        "bal_acc": float(result["bal_acc"]),
        "f1": float(result["f1"]),
        "avg_det_time": float(result["avg_det_time"]),
    }


def _best_balanced_threshold(rollouts: list[Any], scores: list[np.ndarray]) -> dict[str, float]:
    labels = np.asarray([1 - r.episode_success for r in rollouts])
    all_scores = np.concatenate(scores)
    thresholds = set(np.percentile(all_scores, np.linspace(0, 100, 101)).tolist())
    thresholds.add(float(all_scores.min()) - 1e-6)
    thresholds.add(float(all_scores.max()) + 1e-6)
    for score in scores:
        thresholds.add(float((score.min() + score.max()) / 2.0))

    rows = []
    for threshold in sorted(thresholds):
        rows.append(_threshold_curve_rows(rollouts, scores, "calibration", float(threshold)))
    best = max(rows, key=lambda row: (row["bal_acc"], row["tpr"], row["tnr"]))
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in best.items()}


def _build_cp_rows(
    cal_rollouts: list[Any],
    cal_scores_all: list[np.ndarray],
    test_rollouts: list[Any],
    test_scores_all: list[np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    cal_labels = [1 - r.episode_success for r in cal_rollouts]
    test_labels = [1 - r.episode_success for r in test_rollouts]

    for eval_time in EVAL_TIMES:
        cal_scores = _score_at_eval_time(cal_scores_all, cal_rollouts, eval_time)
        test_scores = _score_at_eval_time(test_scores_all, test_rollouts, eval_time)
        for alpha in ALPHAS:
            _, thresholds = split_conformal_binary(cal_scores, cal_labels, test_scores, alpha)
            for calib_label, threshold in (
                ("pos_failure", 1 - thresholds[1]),
                ("neg_success", thresholds[0]),
            ):
                result = eval_binary_classification(test_scores, test_labels, threshold)
                result["avg_det_time"] = eval_detection_time(test_scores_all, test_labels, threshold)
                rows.append(
                    {
                        "eval_time": eval_time,
                        "alpha": float(alpha),
                        "calib_label": calib_label,
                        "threshold": float(threshold),
                        "tpr": float(result["tpr"]),
                        "tnr": float(result["tnr"]),
                        "acc": float(result["acc"]),
                        "bal_acc": float(result["bal_acc"]),
                        "f1": float(result["f1"]),
                        "avg_det_time": float(result["avg_det_time"]),
                    }
                )
    return rows


def _alpha_key(alpha: float) -> str:
    return f"alpha_{str(alpha).replace('.', 'p')}"


def _copy_scores_by_split(scores_by_split: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    return {
        split: [np.asarray(scores, dtype=float).copy() for scores in split_scores]
        for split, split_scores in scores_by_split.items()
    }


def _build_functional_cp_rows(
    rollouts_by_split: dict[str, list[Any]],
    scores_by_split: dict[str, list[np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[float, np.ndarray]]:
    # SAFE's implementation randomly partitions successful calibration curves
    # into regression and modulation sets. Keep this deterministic for artifacts.
    np.random.seed(0)
    df, cp_bands_by_alpha = eval_functional_conformal(
        rollouts_by_split,
        _copy_scores_by_split(scores_by_split),
        method_name="lstm",
        alphas=list(ALPHAS),
        calib_split_names=["val_seen"],
        test_split_names=["val_unseen"],
        align_method="extend",
    )

    rows = []
    for row in df.to_dict("records"):
        alpha = float(row["alpha"])
        band = np.asarray(cp_bands_by_alpha[alpha]).reshape(-1)
        rows.append(
            {
                "eval_time": row["time"],
                "alpha": alpha,
                "calib_label": "neg_success" if row["calib on"] == "neg" else row["calib on"],
                "band_min": float(band.min()),
                "band_mean": float(band.mean()),
                "band_max": float(band.max()),
                "tpr": float(row["tpr"]),
                "tnr": float(row["tnr"]),
                "fpr": float(row["fpr"]),
                "fnr": float(row["fnr"]),
                "acc": float(row["acc"]),
                "bal_acc": float(row["bal_acc"]),
                "f1": float(row["f1"]),
                "avg_det_time": float(row["avg_det_time"]),
            }
        )
    return rows, {float(alpha): np.asarray(band).reshape(-1) for alpha, band in cp_bands_by_alpha.items()}


def _write_functional_bands(path: Path, cp_bands_by_alpha: dict[float, np.ndarray]) -> None:
    np.savez(
        path,
        **{_alpha_key(alpha): np.asarray(band).reshape(-1) for alpha, band in cp_bands_by_alpha.items()},
    )


def _find_functional_row(
    rows: list[dict[str, Any]],
    *,
    eval_time: str,
    alpha: float,
    calib_label: str = "neg_success",
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row["eval_time"] == eval_time
            and row["calib_label"] == calib_label
            and abs(float(row["alpha"]) - alpha) < 1e-9
        ):
            return row
    return None


def _best_functional_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["eval_time"] == "by final end" and row["calib_label"] == "neg_success"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["bal_acc"], -row["avg_det_time"], row["tpr"], row["tnr"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _validate_pinned_selection(cfg: Any) -> None:
    mismatches = []
    if str(cfg.dataset.horizon_idx_rel) != FINAL_HORIZON_IDX_REL:
        mismatches.append(
            f"horizon_idx_rel={cfg.dataset.horizon_idx_rel!r}, expected {FINAL_HORIZON_IDX_REL!r}"
        )
    if str(cfg.dataset.diff_idx_rel) != FINAL_DIFF_IDX_REL:
        mismatches.append(
            f"diff_idx_rel={cfg.dataset.diff_idx_rel!r}, expected {FINAL_DIFF_IDX_REL!r}"
        )
    if float(cfg.model.lr) != float(FINAL_LR):
        mismatches.append(f"lr={cfg.model.lr!r}, expected {FINAL_LR!r}")
    if float(cfg.model.lambda_reg) != float(FINAL_LAMBDA_REG):
        mismatches.append(f"lambda_reg={cfg.model.lambda_reg!r}, expected {FINAL_LAMBDA_REG!r}")
    if int(cfg.train.seed) != FINAL_SEED:
        mismatches.append(f"seed={cfg.train.seed!r}, expected {FINAL_SEED!r}")
    if mismatches:
        raise ValueError("Pinned final detector config mismatch: " + "; ".join(mismatches))


def _write_summary_md(
    path: Path,
    manifest: dict[str, Any],
    best_threshold: dict[str, Any],
    fixed_eval_rows: list[dict[str, Any]],
    cp_rows: list[dict[str, Any]],
    functional_cp_rows: list[dict[str, Any]],
) -> None:
    final_op = manifest["final_operating_point"]
    fixed_val_unseen = [row for row in fixed_eval_rows if row["split"] == "val_unseen"][0]
    functional_at_final_alpha = manifest.get("functional_cp_at_final_alpha")
    best_functional = manifest.get("best_functional_cp")
    lines = [
        "# GR00T N1.6 SAFE-LSTM Final Detector",
        "",
        "## Pinned Checkpoint",
        "",
        f"- source checkpoint: `{manifest['source_checkpoint']}`",
        f"- final checkpoint: `{manifest['final_checkpoint']}`",
        f"- aggregation: `horizon_idx_rel={manifest['horizon_idx_rel']}`, `diff_idx_rel={manifest['diff_idx_rel']}`",
        f"- hyperparameters: `lr={manifest['lr']}`, `lambda_reg={manifest['lambda_reg']}`, `batch_size={manifest['batch_size']}`, `seed={manifest['seed']}`",
        "",
        "## Fixed Threshold Baseline",
        "",
        "This baseline threshold is selected on `val_seen` by maximizing balanced accuracy on max-so-far rollout scores. It is reported for comparison, but the final operating point is the split-CP threshold below.",
        "",
        f"- selected threshold: `{_fmt(best_threshold['threshold'])}`",
        f"- val_seen bal-acc: `{_fmt(best_threshold['bal_acc'])}`",
        f"- val_seen TPR/TNR: `{_fmt(best_threshold['tpr'])}` / `{_fmt(best_threshold['tnr'])}`",
        f"- val_seen mean T-det: `{_fmt(best_threshold['avg_det_time'])}`",
        "",
        "## Held-Out Evaluation",
        "",
        f"- test split: `val_unseen`",
        f"- bal-acc: `{_fmt(fixed_val_unseen['bal_acc'])}`",
        f"- TPR/TNR: `{_fmt(fixed_val_unseen['tpr'])}` / `{_fmt(fixed_val_unseen['tnr'])}`",
        f"- acc/F1: `{_fmt(fixed_val_unseen['acc'])}` / `{_fmt(fixed_val_unseen['f1'])}`",
        f"- mean T-det: `{_fmt(fixed_val_unseen['avg_det_time'])}`",
        "",
        "## Final Operating Point",
        "",
        "The final detector operating point is fixed to split conformal prediction calibrated on successful `val_seen` rollouts. This gives the best practical balance observed here: strong failure recall while keeping false alarms controlled.",
        "",
        f"- method: `{final_op['threshold_method']}`",
        f"- alpha: `{final_op['alpha']}`",
        f"- eval time: `{final_op['eval_time']}`",
        f"- calibration label: `{final_op['calib_label']}`",
        f"- threshold: `{_fmt(final_op['threshold'])}`",
        f"- val_unseen bal-acc: `{_fmt(final_op['bal_acc'])}`",
        f"- val_unseen TPR/TNR: `{_fmt(final_op['tpr'])}` / `{_fmt(final_op['tnr'])}`",
        f"- val_unseen acc/F1: `{_fmt(final_op['acc'])}` / `{_fmt(final_op['f1'])}`",
        f"- val_unseen mean T-det: `{_fmt(final_op['avg_det_time'])}`",
        "",
        "## Functional CP",
        "",
        "Functional CP is computed from the same `val_seen` successful score curves using SAFE's `FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)` implementation. The final operating point remains split CP, but these rows reproduce the SAFE Figure-8-style functional-band evaluation.",
        "",
    ]
    if functional_at_final_alpha is not None:
        lines.extend(
            [
                f"- alpha `0.2`, by final end bal-acc: `{_fmt(functional_at_final_alpha['bal_acc'])}`",
                f"- alpha `0.2`, by final end TPR/TNR: `{_fmt(functional_at_final_alpha['tpr'])}` / `{_fmt(functional_at_final_alpha['tnr'])}`",
                f"- alpha `0.2`, by final end mean T-det: `{_fmt(functional_at_final_alpha['avg_det_time'])}`",
                "",
            ]
        )
    if best_functional is not None:
        lines.extend(
            [
                f"- best by-final-end functional alpha: `{best_functional['alpha']}`",
                f"- best by-final-end functional bal-acc: `{_fmt(best_functional['bal_acc'])}`",
                f"- best by-final-end functional TPR/TNR: `{_fmt(best_functional['tpr'])}` / `{_fmt(best_functional['tnr'])}`",
                f"- best by-final-end functional mean T-det: `{_fmt(best_functional['avg_det_time'])}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Files",
        "",
        "- `model_final.ckpt`: pinned detector checkpoint",
        "- `config.yaml`: training/evaluation config copied from the selected run",
        "- `manifest.json`: machine-readable checkpoint and selection metadata",
        "- `final_operating_point.json`: machine-readable final threshold and held-out metrics",
        "- `fixed_threshold_eval.csv`: val_seen-selected threshold applied to val_seen and val_unseen",
        "- `split_cp_eval.csv`: SAFE split-CP thresholds and held-out metrics for all alpha/eval-time settings",
        "- `functional_cp_eval.csv`: SAFE functional-CP band metrics for all alpha/eval-time settings",
        "- `functional_cp_bands.npz`: functional-CP timestep bands keyed by alpha",
        "- `per_rollout_scores.csv`: rollout-level max and earliest-stop scores",
        "",
        "Note: this run has physical `train`, `val_seen`, and `val_unseen` directories. There is no separate `test` directory; `val_unseen` is used here as the held-out test split.",
        "",
    ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.safe_root))

    global eval_binary_classification, eval_detection_time, eval_functional_conformal, split_conformal_binary
    from failure_prob.data import load_rollouts, split_rollouts
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model
    from failure_prob.utils.conformal.split_cp import split_conformal_binary
    from failure_prob.utils.metrics import (
        eval_binary_classification,
        eval_detection_time,
        eval_functional_conformal,
    )
    from failure_prob.utils.routines import model_forward_dataloader

    run_dir = args.run_dir
    final_dir = args.final_dir
    ckpt_path = run_dir / "model_final.ckpt"
    config_path = run_dir / "config.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    final_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = final_dir / "model_final.ckpt"
    final_config = final_dir / "config.yaml"
    shutil.copy2(ckpt_path, final_ckpt)
    shutil.copy2(config_path, final_config)

    cfg = OmegaConf.load(config_path)
    _validate_pinned_selection(cfg)
    cfg.dataset.load_to_cuda = False
    all_rollouts = load_rollouts(cfg)
    rollouts_by_split = split_rollouts(cfg, all_rollouts)

    train_rollouts = rollouts_by_split["train"]
    input_dim = train_rollouts[0].hidden_states.shape[-1]
    model = get_model(cfg, input_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    scores_by_split: dict[str, list[np.ndarray]] = {}
    for split, rollouts in rollouts_by_split.items():
        dataset = RolloutDataset(cfg, rollouts, device="cpu")
        dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
        scores_np = scores.detach().cpu().numpy()
        lengths = valid_masks.sum(dim=-1).cpu().numpy()
        scores_by_split[split] = [
            scores_np[i, : int(lengths[i])] for i in range(len(lengths))
        ]

    best_threshold = _best_balanced_threshold(
        rollouts_by_split["val_seen"], scores_by_split["val_seen"]
    )
    fixed_rows = [
        _threshold_curve_rows(
            rollouts_by_split[split],
            scores_by_split[split],
            split,
            best_threshold["threshold"],
        )
        for split in ("val_seen", "val_unseen")
    ]
    cp_rows = _build_cp_rows(
        rollouts_by_split["val_seen"],
        scores_by_split["val_seen"],
        rollouts_by_split["val_unseen"],
        scores_by_split["val_unseen"],
    )
    functional_cp_rows, functional_cp_bands = _build_functional_cp_rows(
        rollouts_by_split,
        scores_by_split,
    )
    final_operating_point = [
        row
        for row in cp_rows
        if (
            row["eval_time"] == FINAL_OPERATING_POINT["eval_time"]
            and row["alpha"] == FINAL_OPERATING_POINT["alpha"]
            and row["calib_label"] == FINAL_OPERATING_POINT["calib_label"]
        )
    ][0]
    final_operating_point = {
        **FINAL_OPERATING_POINT,
        **{
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in final_operating_point.items()
        },
        "calibration_split": "val_seen",
        "test_split": "val_unseen",
    }
    functional_cp_at_final_alpha = _find_functional_row(
        functional_cp_rows,
        eval_time=FINAL_OPERATING_POINT["eval_time"],
        alpha=FINAL_OPERATING_POINT["alpha"],
        calib_label=FINAL_OPERATING_POINT["calib_label"],
    )
    best_functional_cp = _best_functional_row(functional_cp_rows)

    per_rollout_rows = []
    for split, rollouts in rollouts_by_split.items():
        for rollout, scores in zip(rollouts, scores_by_split[split]):
            per_rollout_rows.append(
                {
                    "split": split,
                    "task_id": rollout.task_id,
                    "episode_idx": rollout.episode_idx,
                    "episode_success": rollout.episode_success,
                    "label_failure": 1 - rollout.episode_success,
                    "length": len(scores),
                    "task_min_step": rollout.task_min_step,
                    "score_at_earliest_stop": float(scores[rollout.task_min_step - 1]),
                    "score_by_earliest_stop": float(scores[: rollout.task_min_step].max()),
                    "score_by_final_end": float(scores[: len(scores)].max()),
                }
            )

    _write_csv(final_dir / "fixed_threshold_eval.csv", fixed_rows)
    _write_csv(final_dir / "split_cp_eval.csv", cp_rows)
    _write_csv(final_dir / "functional_cp_eval.csv", functional_cp_rows)
    _write_functional_bands(final_dir / "functional_cp_bands.npz", functional_cp_bands)
    _write_csv(final_dir / "per_rollout_scores.csv", per_rollout_rows)

    manifest = {
        "source_run_dir": str(run_dir),
        "source_checkpoint": str(ckpt_path),
        "final_checkpoint": str(final_ckpt),
        "config": str(final_config),
        "horizon_idx_rel": str(cfg.dataset.horizon_idx_rel),
        "diff_idx_rel": str(cfg.dataset.diff_idx_rel),
        "lr": float(cfg.model.lr),
        "lambda_reg": float(cfg.model.lambda_reg),
        "batch_size": int(cfg.model.batch_size),
        "seed": int(cfg.train.seed),
        "splits": {
            split: {
                "n": len(rollouts),
                "success": int(sum(r.episode_success for r in rollouts)),
                "failure": int(len(rollouts) - sum(r.episode_success for r in rollouts)),
            }
            for split, rollouts in rollouts_by_split.items()
        },
        "selected_threshold_source": "val_seen max-so-far balanced accuracy",
        "selected_threshold": best_threshold,
        "final_operating_point": final_operating_point,
        "functional_cp_at_final_alpha": functional_cp_at_final_alpha,
        "best_functional_cp": best_functional_cp,
    }
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (final_dir / "final_operating_point.json").write_text(
        json.dumps(final_operating_point, indent=2)
    )
    _write_summary_md(
        final_dir / "README.md",
        manifest,
        best_threshold,
        fixed_rows,
        cp_rows,
        functional_cp_rows,
    )

    print(f"finalized detector: {final_dir}")
    print(f"checkpoint: {final_ckpt}")
    print(f"selected threshold: {best_threshold['threshold']:.6f}")
    print(f"final operating threshold: {final_operating_point['threshold']:.6f}")


if __name__ == "__main__":
    main()
