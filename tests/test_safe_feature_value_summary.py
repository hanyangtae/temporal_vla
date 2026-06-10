from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "safe" / "_common" / "feature_value_summary.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "safe_feature_value_summary_under_test",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_rollout_root_reports_shape_and_value_scale(tmp_path: Path):
    module = _load_module()
    task_dir = tmp_path / "OpenFridge"
    task_dir.mkdir()
    payload = {
        "model_family": "lerobot_groot_n15",
        "policy_transport": "http",
        "feature_kind": "groot_n15_dit_action_tokens_pre_decode",
        "feature_axes": ["denoising_step", "action_step", "feature_dim"],
        "hidden_states": [
            np.array([[[1.0, -1.0], [3.0, -3.0]]], dtype=np.float32),
            np.array([[[2.0, -2.0], [4.0, -4.0]]], dtype=np.float32),
        ],
    }
    with (task_dir / "task0--ep0--succ1.pkl").open("wb") as f:
        pickle.dump(payload, f)

    summary = module.summarize_rollout_root(tmp_path, label="n15")

    assert summary["label"] == "n15"
    assert summary["num_rollouts"] == 1
    assert summary["num_feature_records"] == 2
    assert summary["model_families"] == ["lerobot_groot_n15"]
    assert summary["feature_kinds"] == ["groot_n15_dit_action_tokens_pre_decode"]
    assert summary["shape_counts"] == {"1x2x2": 2}
    assert summary["finite_fraction"] == 1.0
    assert summary["abs_mean"] == 2.5
    assert summary["mean"] == 0.0
    assert summary["max"] == 4.0


def test_compare_summaries_reports_abs_mean_ratio():
    module = _load_module()

    comparison = module.compare_summaries(
        {"label": "n15", "abs_mean": 2.0, "l2_mean": 10.0},
        {"label": "n16", "abs_mean": 4.0, "l2_mean": 20.0},
    )

    assert comparison["left_label"] == "n15"
    assert comparison["right_label"] == "n16"
    assert comparison["abs_mean_ratio_left_over_right"] == 0.5
    assert comparison["l2_mean_ratio_left_over_right"] == 0.5
