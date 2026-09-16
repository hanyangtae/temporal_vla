#!/usr/bin/env python3
"""Train the temporal_vla SAFE LSTM and calibrate its one-sided CP band."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


ALPHA = 0.2
HORIZON = 150
CHUNK_STEPS = 4
FEATURE_DIM = 1024


def _observed_trajectory(values: np.ndarray, chunk_lengths: list[int], horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Expand chunk scores to environment steps without marking padding observed."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(chunk_lengths) != len(values) or not chunk_lengths:
        raise ValueError("trajectory chunk lengths do not match chunk values")
    out = np.zeros(horizon, dtype=np.float64)
    mask = np.zeros(horizon, dtype=bool)
    cursor = 0
    for value, chunk_length in zip(values, chunk_lengths):
        end = min(cursor + int(chunk_length), horizon)
        if end <= cursor:
            break
        out[cursor:end] = value
        mask[cursor:end] = True
        cursor = end
    return out, mask


def grouped_conformal_band(
    train_success_scores: np.ndarray,
    train_success_masks: np.ndarray,
    calibration_success_scores: np.ndarray,
    calibration_masks: np.ndarray,
    calibration_groups: list[str] | np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Return a grouped, one-sided upper CP band from observed success scores.

    The center and floored per-step standard deviation come only from successful
    training traces. Calibration residuals are maxima over observed steps, then
    maxima across policy seeds sharing a reset group. The order statistic follows
    ceil((G+1)*(1-alpha)); no finite band is returned when that statistic is
    unavailable.
    """
    import numpy as np
    train = np.asarray(train_success_scores, dtype=np.float64)
    train_mask = np.asarray(train_success_masks, dtype=bool)
    cal = np.asarray(calibration_success_scores, dtype=np.float64)
    cal_mask = np.asarray(calibration_masks, dtype=bool)
    if train.ndim != 2 or train_mask.shape != train.shape:
        raise ValueError("training scores/masks must have matching rank-2 shapes")
    if cal.ndim != 2 or cal_mask.shape != cal.shape or cal.shape[1] != train.shape[1]:
        raise ValueError("calibration scores/masks must match training horizon")
    if len(calibration_groups) != len(cal):
        raise ValueError("one reset group is required per calibration trajectory")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not train_mask.any() or not cal_mask.any():
        raise ValueError("training and calibration must contain observed success steps")

    count = train_mask.sum(axis=0)
    active = count > 0
    if not np.any(active):
        raise ValueError("training success traces have no observed steps")
    safe_count = np.maximum(count, 1)
    center = np.sum(np.where(train_mask, train, 0.0), axis=0) / safe_count
    variance = np.sum(np.where(train_mask, (train - center) ** 2, 0.0), axis=0)
    variance = variance / np.maximum(count - 1, 1)
    scale = np.maximum(np.sqrt(variance), 1e-6)

    group_residuals: dict[str, float] = {}
    for row, mask, group in zip(cal, cal_mask, calibration_groups):
        observed = np.flatnonzero(mask & active)
        if observed.size == 0:
            raise ValueError("calibration trajectory has no observed steps")
        residual = float(np.max((row[observed] - center[observed]) / scale[observed]))
        key = str(group)
        group_residuals[key] = max(group_residuals.get(key, -np.inf), residual)
    residuals = np.asarray(list(group_residuals.values()), dtype=np.float64)
    groups = len(residuals)
    rank = int(np.ceil((groups + 1) * (1 - alpha)))
    if groups == 0 or rank < 1 or rank > groups:
        raise ValueError(f"insufficient grouped calibration episodes: groups={groups}, rank={rank}")
    width = float(np.sort(residuals)[rank - 1])
    band = np.full_like(center, np.nextafter(1.0, np.inf))
    band[active] = center[active] + width * scale[active]
    return band


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records_for_split(manifest_path: Path, split: str) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload["episodes"] if row.get("split") == split]
    if not rows:
        raise ValueError(f"manifest has no {split} episodes")
    for row in rows:
        source = Path(row["path"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        row["path"] = str(source.resolve())
        row["sha256"] = _sha256(source)
        row["split"] = split
    return {"schema_version": payload.get("schema_version", 1), "episodes": rows}


def _episode_features(episodes: list[dict[str, Any]], require_both: bool = True) -> tuple[list[np.ndarray], np.ndarray, list[str], list[list[int]]]:
    import numpy as np
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    ids: list[str] = []
    chunk_lengths: list[list[int]] = []
    for episode in episodes:
        contexts = np.asarray([chunk["context"] for chunk in episode["chunks"]], dtype=np.float32)
        if contexts.ndim != 2 or contexts.shape[1] != FEATURE_DIM or not np.isfinite(contexts).all():
            raise ValueError(f"{episode.get('_path')}: contexts must be finite [T, 1024]")
        sequences.append(contexts)
        labels.append(float(episode["success"]))
        ids.append(str(episode["reset_id"]))
        chunk_lengths.append([len(chunk["executed_actions"]) for chunk in episode["chunks"]])
    if not sequences or (require_both and (not any(label == 1 for label in labels) or not any(label == 0 for label in labels))):
        raise ValueError("training split must contain both success and failure episodes")
    return sequences, np.asarray(labels, dtype=np.float32), ids, chunk_lengths


def _batch(sequences: list[np.ndarray], labels: np.ndarray, indices: np.ndarray, device: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    chosen = [sequences[int(i)] for i in indices]
    width = max(len(item) for item in chosen)
    features = np.zeros((len(chosen), width, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((len(chosen), width), dtype=np.float32)
    for row, item in enumerate(chosen):
        features[row, : len(item)] = item
        masks[row, : len(item)] = 1.0
    return {
        "features": torch.as_tensor(features, device=device),
        "valid_masks": torch.as_tensor(masks, device=device),
        "success_labels": torch.as_tensor(labels[indices], device=device),
    }


def _predict(model: Any, sequences: list[np.ndarray], chunk_lengths: list[list[int]], device: Any) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    import torch
    model.eval()
    width = max(len(item) for item in sequences)
    features = np.zeros((len(sequences), width, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((len(sequences), width), dtype=bool)
    for row, item in enumerate(sequences):
        features[row, : len(item)] = item
        masks[row, : len(item)] = True
    with torch.no_grad():
        output = model({"features": torch.as_tensor(features, device=device)}).squeeze(-1).cpu().numpy()
    expanded = []
    expanded_masks = []
    for row, item in enumerate(sequences):
        values, mask = _observed_trajectory(output[row, : len(item)], chunk_lengths[row])
        expanded.append(values)
        expanded_masks.append(mask)
    return np.stack(expanded), np.stack(expanded_masks)


def train(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    safe_root = args.safe_root.resolve()
    if not safe_root.is_dir():
        raise ValueError(f"SAFE data root does not exist: {safe_root}")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    manifest_path = args.manifest.resolve()

    # stage2 data loader performs hash, feature/action contract, and split checks.
    sys.path.insert(0, str(Path(__file__).parent))
    from data import load_manifest

    train_episodes = load_manifest(manifest_path, "train")
    calibration_episodes = load_manifest(manifest_path, "validation")
    train_sequences, train_labels, _, train_lengths = _episode_features(train_episodes)
    cal_sequences, cal_labels, cal_groups, cal_lengths = _episode_features(calibration_episodes, require_both=False)

    import torch
    import numpy as np
    if not np.any(cal_labels == 1):
        raise ValueError("validation split has no successful calibration episodes")
    from omegaconf import OmegaConf
    sys.path.insert(0, str(safe_root))
    from failure_prob.model.lstm import LstmModel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = OmegaConf.create({"model": {
        "name": "lstm", "n_layers": 1, "hidden_dim": 256, "n_history_steps": -1,
        "one_loss_per_seq": False, "lr": 3e-4, "lambda_reg": 1.0,
        "lambda_hard_heg": 0.0, "hard_neg_margin": 0.1, "hard_neg_beta": 50.0,
        "cumsum": False, "rmean": False, "use_time_weighting": False, "optimizer": "adam", "lr_step_size": 300,
        "lr_gamma": 1.0, "weight_decay": 1e-2, "warmup_steps": 0,
        "lambda_success": 1.0, "lambda_fail": 1.0, "init_weight_scale": 1.0,
        "grad_max_norm": None, "dropout": 0.0,
    }})
    model = LstmModel(cfg, FEATURE_DIM).to(device)
    optimizer, scheduler = model.get_optimizer()
    class_weights = [1.0 / ((np.sum(train_labels == 0) + 1) / len(train_labels)),
                     1.0 / ((np.sum(train_labels == 1) + 1) / len(train_labels))]
    for _ in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train_sequences))
        for start in range(0, len(order), args.batch_size):
            batch = _batch(train_sequences, train_labels, order[start:start + args.batch_size], device)
            loss, _ = model.forward_compute_loss(batch, class_weights)
            reg_loss, _ = model.compute_regularization_loss(cfg.model.lambda_reg)
            if not torch.isfinite(loss + reg_loss):
                raise FloatingPointError("nonfinite SAFE loss")
            optimizer.zero_grad()
            (loss + reg_loss).backward()
            optimizer.step()
        scheduler.step()

    train_success = [seq for seq, label in zip(train_sequences, train_labels) if label == 1]
    train_success_lengths = [lengths for lengths, label in zip(train_lengths, train_labels) if label == 1]
    cal_success = [seq for seq, label in zip(cal_sequences, cal_labels) if label == 1]
    cal_success_lengths = [lengths for lengths, label in zip(cal_lengths, cal_labels) if label == 1]
    cal_success_groups = [group for group, label in zip(cal_groups, cal_labels) if label == 1]
    train_scores, train_masks = _predict(model, train_success, train_success_lengths, device)
    cal_scores, cal_masks = _predict(model, cal_success, cal_success_lengths, device)
    band = grouped_conformal_band(train_scores, train_masks, cal_scores, cal_masks, cal_success_groups, ALPHA)

    torch.save(model.state_dict(), output / "model_final.ckpt")
    config = {"model": OmegaConf.to_container(cfg.model, resolve=True), "cp_band_time_unit": "environment_step", "feature_contract": "pi0_denoise0_action_mean_1024",
              "train": {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed, "device": str(device),
                        "regularization": float(cfg.model.lambda_reg), "class_weights": class_weights}}
    (output / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    np.save(output / "cp_band_by_alpha.npy", {ALPHA: band})

    training_manifest = _records_for_split(manifest_path, "train")
    calibration_manifest = _records_for_split(manifest_path, "validation")
    (output / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8")
    (output / "calibration_manifest.json").write_text(json.dumps(calibration_manifest, indent=2) + "\n", encoding="utf-8")
    artifacts = []
    for path in (output / "model_final.ckpt", output / "config.yaml", output / "cp_band_by_alpha.npy",
                 output / "training_manifest.json", output / "calibration_manifest.json"):
        artifacts.append({"path": path.name, "sha256": _sha256(path)})
    provenance = {"owner": "temporal_vla", "feature_contract": "pi0_denoise0_action_mean_1024",
                  "training_manifest": {"path": "training_manifest.json", "sha256": _sha256(output / "training_manifest.json")},
                  "calibration_manifest": {"path": "calibration_manifest.json", "sha256": _sha256(output / "calibration_manifest.json")},
                  "artifacts": artifacts}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"SAFE training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
