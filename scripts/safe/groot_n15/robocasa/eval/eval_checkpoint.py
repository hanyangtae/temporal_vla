#!/usr/bin/env python3
"""Evaluate a trained SAFE checkpoint on GR00T N1.5 rollout PKLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", default="/home/dongkyu/pdk_ws/SAFE")
    parser.add_argument("--config", required=True, help="Training config.yaml saved next to the checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--cp-data-path", "--cp-cal-data-path", dest="cp_data_path", default=None)
    parser.add_argument("--cp-alpha", type=float, default=0.2)
    parser.add_argument("--functional-cp-seed", type=int, default=0)
    return parser.parse_args()


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "roc_auc": None,
        "ap": None,
    }
    if len(np.unique(labels)) < 2:
        return out
    out["roc_auc"] = float(roc_auc_score(labels, scores))
    out["ap"] = float(average_precision_score(labels, scores))
    return out


def _score_at_quantile(score: np.ndarray, task_min_step: int, q: float) -> float:
    idx = round((task_min_step - 1) * q)
    idx = min(max(idx, 0), len(score) - 1)
    return float(score[idx])


def _fixed_threshold_det_metrics(
    scores_by_rollout: list[np.ndarray],
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    class_scores = np.asarray([score.max() for score in scores_by_rollout])
    return _threshold_det_classification_metrics(class_scores, scores_by_rollout, labels, threshold)


def _threshold_det_classification_metrics(
    class_scores: np.ndarray,
    scores_by_rollout: list[np.ndarray],
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    lengths = np.asarray([len(score) for score in scores_by_rollout], dtype=np.float64)
    has_detection = class_scores >= threshold
    has_timeline_detection = np.asarray([bool(np.any(score >= threshold)) for score in scores_by_rollout])
    first_detection = np.asarray([
        int(np.argmax(score >= threshold)) if detected else int(len(score))
        for score, detected in zip(scores_by_rollout, has_timeline_detection)
    ])

    pos_mask = labels == 1
    neg_mask = ~pos_mask
    tp = int(np.sum(has_detection[pos_mask]))
    fn = int(np.sum(~has_detection[pos_mask]))
    fp = int(np.sum(has_detection[neg_mask]))
    tn = int(np.sum(~has_detection[neg_mask]))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    avg_det_time = float(np.mean((first_detection / lengths)[pos_mask])) if np.any(pos_mask) else 0.0
    return {
        "threshold": float(threshold),
        "avg_det_time": avg_det_time,
        "tpr": float(tpr),
        "tnr": float(tnr),
        "acc": float((tp + tn) / len(labels)),
        "bal_acc": float((tpr + tnr) / 2),
        "f1": float(f1),
    }


def _scores_for_eval_time(
    scores_by_rollout: list[np.ndarray],
    rollouts: list[object],
    eval_name: str,
) -> np.ndarray:
    if eval_name == "at_task_min":
        return np.asarray([
            score[min(max(rollout.task_min_step - 1, 0), len(score) - 1)]
            for score, rollout in zip(scores_by_rollout, rollouts)
        ])
    if eval_name == "max_to_task_min":
        return np.asarray([
            score[: rollout.task_min_step].max()
            for score, rollout in zip(scores_by_rollout, rollouts)
        ])
    if eval_name == "max_to_end":
        return np.asarray([score.max() for score in scores_by_rollout])
    raise ValueError(f"Unknown eval name: {eval_name}")


def main() -> None:
    args = _parse_args()
    safe_root = Path(args.safe_root)
    sys.path.insert(0, str(safe_root))

    from failure_prob.conf import process_cfg
    from failure_prob.data import load_rollouts
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model
    from failure_prob.utils.metrics import (
        eval_det_time_vs_classification,
        eval_functional_conformal,
        eval_split_conformal,
    )
    from failure_prob.utils.torch import move_to_device

    cfg = OmegaConf.load(args.config)
    cfg.dataset.data_path = args.data_path
    cfg.dataset.load_to_cuda = False
    cfg.model.batch_size = args.batch_size
    cfg = process_cfg(cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollouts = load_rollouts(cfg)
    if not rollouts:
        raise ValueError(f"No rollouts loaded from {args.data_path}")

    input_dim = rollouts[0].hidden_states.shape[-1]
    model = get_model(cfg, input_dim)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = RolloutDataset(cfg, rollouts, device="cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_scores = []
    all_masks = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            scores = model(batch).squeeze(-1).detach().cpu()
            masks = batch["valid_masks"].detach().cpu()
            all_scores.append(scores)
            all_masks.append(masks)

    score_tensor = torch.cat(all_scores, dim=0).numpy()
    mask_tensor = torch.cat(all_masks, dim=0).numpy()
    lengths = mask_tensor.sum(axis=1).astype(int)
    scores_by_rollout = [
        score_tensor[i, : lengths[i]].astype(np.float64)
        for i in range(len(rollouts))
    ]
    labels = np.asarray([1 - r.episode_success for r in rollouts], dtype=np.int64)

    summary_rows: list[dict[str, object]] = []
    for q in [0.25, 0.5, 0.75, 1.0]:
        q_scores = np.asarray([
            _score_at_quantile(score, rollout.task_min_step, q)
            for score, rollout in zip(scores_by_rollout, rollouts)
        ])
        metrics = _binary_metrics(labels, q_scores)
        summary_rows.append({
            "scope": "all",
            "eval": f"at_task_min_q{q}",
            "n": int(len(labels)),
            "n_failure": int(labels.sum()),
            "n_success": int((1 - labels).sum()),
            **metrics,
        })

    early_scores = np.asarray([
        score[: rollout.task_min_step].max()
        for score, rollout in zip(scores_by_rollout, rollouts)
    ])
    end_scores = np.asarray([score.max() for score in scores_by_rollout])
    for name, eval_scores in [("max_to_task_min", early_scores), ("max_to_end", end_scores)]:
        metrics = _binary_metrics(labels, eval_scores)
        summary_rows.append({
            "scope": "all",
            "eval": name,
            "n": int(len(labels)),
            "n_failure": int(labels.sum()),
            "n_success": int((1 - labels).sum()),
            **metrics,
        })

    for task_id in sorted({r.task_id for r in rollouts}):
        idx = np.asarray([i for i, r in enumerate(rollouts) if r.task_id == task_id])
        task_labels = labels[idx]
        task_name = next(r.task_description for r in rollouts if r.task_id == task_id)
        for name, eval_scores in [
            ("max_to_task_min", early_scores[idx]),
            ("max_to_end", end_scores[idx]),
        ]:
            metrics = _binary_metrics(task_labels, eval_scores)
            summary_rows.append({
                "scope": f"task{task_id}",
                "task": str(task_name),
                "eval": name,
                "n": int(len(task_labels)),
                "n_failure": int(task_labels.sum()),
                "n_success": int((1 - task_labels).sum()),
                **metrics,
            })

    det_rows = eval_det_time_vs_classification(rollouts, scores_by_rollout, labels)
    best_det = max(det_rows, key=lambda r: (r["bal_acc"], r["f1"], -r["avg_det_time"]))
    summary_rows.append({
        "scope": "all",
        "eval": "best_threshold_max_to_end",
        "n": int(len(labels)),
        "n_failure": int(labels.sum()),
        "n_success": int((1 - labels).sum()),
        **{k: float(v) for k, v in best_det.items()},
    })

    if args.fixed_threshold is not None:
        fixed_det = _fixed_threshold_det_metrics(scores_by_rollout, labels, args.fixed_threshold)
        summary_rows.append({
            "scope": "all",
            "eval": "fixed_threshold_max_to_end",
            "n": int(len(labels)),
            "n_failure": int(labels.sum()),
            "n_success": int((1 - labels).sum()),
            **fixed_det,
        })

    if args.cp_data_path is not None:
        cfg.dataset.data_path = args.cp_data_path
        cal_rollouts = load_rollouts(cfg)
        if not cal_rollouts:
            raise ValueError(f"No CP rollouts loaded from {args.cp_data_path}")

        cal_dataset = RolloutDataset(cfg, cal_rollouts, device="cpu")
        cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        cal_scores_all = []
        cal_masks_all = []
        with torch.no_grad():
            for batch in cal_loader:
                batch = move_to_device(batch, device)
                cal_scores_all.append(model(batch).squeeze(-1).detach().cpu())
                cal_masks_all.append(batch["valid_masks"].detach().cpu())
        cal_score_tensor = torch.cat(cal_scores_all, dim=0).numpy()
        cal_mask_tensor = torch.cat(cal_masks_all, dim=0).numpy()
        cal_lengths = cal_mask_tensor.sum(axis=1).astype(int)
        cal_scores_by_rollout = [
            cal_score_tensor[i, : cal_lengths[i]].astype(np.float64)
            for i in range(len(cal_rollouts))
        ]
        cal_labels = np.asarray([1 - r.episode_success for r in cal_rollouts], dtype=np.int64)

        cp_rows = eval_split_conformal(
            {"val_seen": cal_rollouts, "val_unseen": rollouts},
            {"val_seen": cal_scores_by_rollout, "val_unseen": scores_by_rollout},
            "model",
            alphas=[args.cp_alpha],
            calib_split_names=["val_seen"],
            test_split_names=["val_unseen"],
        )
        eval_name_by_safe_time = {
            "at earliest stop": "split_cp_at_task_min",
            "by earliest stop": "split_cp_max_to_task_min",
            "by final end": "split_cp_max_to_end",
        }
        calib_on_by_safe_name = {
            "neg": "neg_success",
            "pos": "pos_failure",
        }
        for row in cp_rows:
            summary_row = {
                "scope": "all",
                "eval": eval_name_by_safe_time[str(row["time"])],
                "calib_on": calib_on_by_safe_name[str(row["calib on"])],
                "alpha": float(row["alpha"]),
                "n": int(len(labels)),
                "n_failure": int(labels.sum()),
                "n_success": int((1 - labels).sum()),
                "n_cal": int(len(cal_labels)),
                "n_cal_failure": int(cal_labels.sum()),
                "n_cal_success": int((1 - cal_labels).sum()),
            }
            for key in [
                "threshold",
                "avg_det_time",
                "tpr",
                "tnr",
                "fpr",
                "fnr",
                "acc",
                "bal_acc",
                "f1",
                "weighted-acc",
                "roc_auc",
                "prc_auc",
            ]:
                summary_row[key] = float(row[key])
            summary_rows.append(summary_row)

        np.random.seed(args.functional_cp_seed)
        functional_df, cp_bands_by_alpha = eval_functional_conformal(
            {"val_seen": cal_rollouts, "val_unseen": rollouts},
            {"val_seen": list(cal_scores_by_rollout), "val_unseen": list(scores_by_rollout)},
            "model",
            calib_split_names=["val_seen"],
            test_split_names=["val_unseen"],
        )
        functional_path = output_dir / "functional_cp.tsv"
        functional_df.to_csv(functional_path, sep="\t", index=False)

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for eval_time, group in functional_df.groupby("time"):
            group = group.sort_values("avg_det_time")
            ax.plot(group["avg_det_time"], group["bal_acc"], marker="o", label=str(eval_time))
        ax.set_xlabel("T-det / avg_det_time")
        ax.set_ylabel("balanced accuracy")
        ax.set_title("Functional CP")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        curve_path = output_dir / "functional_cp_bal_acc_vs_tdet.png"
        fig.savefig(curve_path, dpi=160)
        plt.close(fig)

        np.savez_compressed(
            output_dir / "functional_cp_bands.npz",
            **{
                f"alpha_{str(alpha).replace('.', 'p')}": np.asarray(band)
                for alpha, band in cp_bands_by_alpha.items()
            },
        )

        for row in functional_df.to_dict("records"):
            summary_rows.append({
                "scope": "all",
                "eval": "functional_cp_by_final_end"
                if row["time"] == "by final end"
                else "functional_cp_by_earliest_stop",
                "calib_on": "neg_success",
                "alpha": float(row["alpha"]),
                "n": int(len(labels)),
                "n_failure": int(labels.sum()),
                "n_success": int((1 - labels).sum()),
                "n_cal": int(len(cal_labels)),
                "n_cal_failure": int(cal_labels.sum()),
                "n_cal_success": int((1 - cal_labels).sum()),
                "avg_det_time": float(row["avg_det_time"]),
                "tpr": float(row["tpr"]),
                "tnr": float(row["tnr"]),
                "fpr": float(row["fpr"]),
                "fnr": float(row["fnr"]),
                "acc": float(row["acc"]),
                "bal_acc": float(row["bal_acc"]),
                "f1": float(row["f1"]),
            })

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True))

    tsv_path = output_dir / "metrics_summary.tsv"
    keys = sorted({key for row in summary_rows for key in row.keys()})
    with tsv_path.open("w") as f:
        f.write("\t".join(keys) + "\n")
        for row in summary_rows:
            f.write("\t".join("" if row.get(k) is None else str(row.get(k)) for k in keys) + "\n")

    det_path = output_dir / "threshold_sweep.tsv"
    det_keys = sorted(det_rows[0].keys())
    with det_path.open("w") as f:
        f.write("\t".join(det_keys) + "\n")
        for row in det_rows:
            f.write("\t".join(str(row[k]) for k in det_keys) + "\n")

    np.savez_compressed(
        output_dir / "scores.npz",
        labels=labels,
        lengths=lengths,
        early_scores=early_scores,
        end_scores=end_scores,
    )

    print(f"Loaded rollouts: {len(rollouts)}")
    print(f"Failures: {int(labels.sum())}, successes: {int((1 - labels).sum())}")
    print(f"Best threshold: {best_det}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tsv_path}")
    print(f"Wrote {det_path}")
    if args.cp_data_path is not None:
        print(f"Wrote {output_dir / 'functional_cp.tsv'}")
        print(f"Wrote {output_dir / 'functional_cp_bands.npz'}")
        print(f"Wrote {output_dir / 'functional_cp_bal_acc_vs_tdet.png'}")


if __name__ == "__main__":
    main()
