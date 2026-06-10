#!/usr/bin/env python3
"""Summarize SAFE rollout feature value scales.

This intentionally compares distributions, not tensor elements. GR00T N1.5
LeRobot HTTP and GR00T N1.6 HTTP/ZMQ can have different feature definitions and
dimensions, so value-scale checks should stay metadata-aware.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


def _shape_key(arr: np.ndarray) -> str:
    return "x".join(str(dim) for dim in arr.shape)


def _safe_float(value: float) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return _safe_float(numerator / denominator)


def _iter_rollout_pkls(root: Path):
    yield from sorted(path for path in root.rglob("*.pkl") if path.is_file())


def summarize_rollout_root(root: str | Path, *, label: str | None = None) -> dict[str, Any]:
    root = Path(root)
    values = []
    l2_values = []
    shape_counts: Counter[str] = Counter()
    model_families = set()
    policy_transports = set()
    feature_kinds = set()
    feature_axes = set()
    num_rollouts = 0
    num_feature_records = 0
    finite_total = 0
    total = 0

    for pkl_path in _iter_rollout_pkls(root):
        with pkl_path.open("rb") as f:
            payload = pickle.load(f)
        hidden_states = payload.get("hidden_states") or []
        if not hidden_states:
            continue
        num_rollouts += 1
        if payload.get("model_family") is not None:
            model_families.add(str(payload["model_family"]))
        if payload.get("policy_transport") is not None:
            policy_transports.add(str(payload["policy_transport"]))
        if payload.get("feature_kind") is not None:
            feature_kinds.add(str(payload["feature_kind"]))
        axes = payload.get("feature_axes")
        if axes is not None:
            feature_axes.add("/".join(str(axis) for axis in axes))

        for hidden in hidden_states:
            arr = np.asarray(hidden, dtype=np.float64)
            shape_counts[_shape_key(arr)] += 1
            finite = np.isfinite(arr)
            finite_total += int(finite.sum())
            total += int(arr.size)
            if not finite.any():
                continue
            flat = arr[finite]
            values.append(flat)
            l2_values.append(float(np.linalg.norm(flat)))
            num_feature_records += 1

    if values:
        concat = np.concatenate(values)
        mean = _safe_float(np.mean(concat))
        std = _safe_float(np.std(concat))
        abs_mean = _safe_float(np.mean(np.abs(concat)))
        min_value = _safe_float(np.min(concat))
        max_value = _safe_float(np.max(concat))
        l2_mean = _safe_float(np.mean(l2_values))
    else:
        mean = std = abs_mean = min_value = max_value = l2_mean = None

    return {
        "label": label or root.name,
        "root": str(root),
        "num_rollouts": num_rollouts,
        "num_feature_records": num_feature_records,
        "model_families": sorted(model_families),
        "policy_transports": sorted(policy_transports),
        "feature_kinds": sorted(feature_kinds),
        "feature_axes": sorted(feature_axes),
        "shape_counts": dict(sorted(shape_counts.items())),
        "finite_fraction": _safe_float(finite_total / total) if total else None,
        "mean": mean,
        "std": std,
        "abs_mean": abs_mean,
        "l2_mean": l2_mean,
        "min": min_value,
        "max": max_value,
    }


def compare_summaries(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "abs_mean_ratio_left_over_right": _ratio(left.get("abs_mean"), right.get("abs_mean")),
        "l2_mean_ratio_left_over_right": _ratio(left.get("l2_mean"), right.get("l2_mean")),
        "left_feature_kinds": left.get("feature_kinds", []),
        "right_feature_kinds": right.get("feature_kinds", []),
        "left_shape_counts": left.get("shape_counts", {}),
        "right_shape_counts": right.get("shape_counts", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", required=True)
    parser.add_argument("--right-root")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    left = summarize_rollout_root(args.left_root, label=args.left_label)
    result: dict[str, Any] = {"left": left}
    if args.right_root:
        right = summarize_rollout_root(args.right_root, label=args.right_label)
        result["right"] = right
        result["comparison"] = compare_summaries(left, right)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return result


if __name__ == "__main__":
    main()
