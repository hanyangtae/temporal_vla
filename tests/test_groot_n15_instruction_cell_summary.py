from __future__ import annotations

import csv
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "analyze"
    / "summarize_instruction_cells.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_cell_summary_under_test",
        SUMMARY_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pkl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_summarize_instruction_cells_reports_sr_and_instruction_mismatches(tmp_path: Path):
    module = _load_module()
    canonical = "Open the right drawer."
    root = tmp_path / "raw_rollouts"
    cell_dir = root / "OpenDrawer" / "open_drawer_right"

    base_payload = {
        "task_id": 7,
        "cell_id": "open_drawer_right",
        "cell_index": 7,
        "robocasa_task": "OpenDrawer",
        "canonical_instruction": canonical,
        "task_description": canonical,
        "ep_meta": {"lang": canonical},
        "hidden_states": [np.zeros((4, 16, 2), dtype=np.float32)],
        "vl_hidden_states": [np.zeros((2048,), dtype=np.float32)],
    }
    _write_pkl(
        cell_dir / "task7--ep0--succ1.pkl",
        {**base_payload, "episode_idx": 0, "episode_success": 1, "scenario_seed": 100001},
    )
    _write_pkl(
        cell_dir / "task7--ep1--succ0.pkl",
        {
            **base_payload,
            "episode_idx": 1,
            "episode_success": 0,
            "scenario_seed": 100002,
            "task_description": "Open the left drawer.",
        },
    )

    rows = module.summarize_instruction_cells(root)

    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "OpenDrawer"
    assert row["cell_id"] == "open_drawer_right"
    assert row["cell_index"] == 7
    assert row["success"] == 1
    assert row["total"] == 2
    assert row["success_rate"] == 0.5
    assert row["seed_min"] == 100001
    assert row["seed_max"] == 100002
    assert row["instruction_mismatch"] == 1
    assert row["has_vl_hidden_states"] == 2


def test_write_summary_tsv_uses_stable_columns(tmp_path: Path):
    module = _load_module()
    output_path = tmp_path / "summary.tsv"
    rows = [
        {
            "task": "OpenDrawer",
            "cell_id": "open_drawer_right",
            "cell_index": 7,
            "canonical_instruction": "Open the right drawer.",
            "success": 1,
            "total": 2,
            "success_rate": 0.5,
            "seed_min": 100001,
            "seed_max": 100002,
            "instruction_mismatch": 0,
            "has_vl_hidden_states": 2,
        }
    ]

    module.write_summary_tsv(output_path, rows)

    with output_path.open(newline="") as f:
        written = list(csv.DictReader(f, delimiter="\t"))

    assert written == [
        {
            "task": "OpenDrawer",
            "cell_id": "open_drawer_right",
            "cell_index": "7",
            "canonical_instruction": "Open the right drawer.",
            "success": "1",
            "total": "2",
            "success_rate": "0.5000",
            "seed_min": "100001",
            "seed_max": "100002",
            "instruction_mismatch": "0",
            "has_vl_hidden_states": "2",
        }
    ]
