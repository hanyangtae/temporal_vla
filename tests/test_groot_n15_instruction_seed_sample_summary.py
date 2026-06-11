from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "analyze"
    / "summarize_instruction_seed_samples.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_seed_sample_summary_under_test",
        SUMMARY_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cells_config(path: Path) -> None:
    path.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tcanonical_instruction\t"
        "target_episodes\tseed_search_start\n"
        "0\tppcc_potato\tPickPlaceCounterToCabinet\tenv\t"
        "Pick the potato from the counter and place it in the cabinet.\t50\t100000\n"
        "1\tcoffee_setup_mug\tCoffeeSetupMug\tenv\t"
        "Pick the mug from the counter and place it under the coffee machine dispenser.\t50\t100000\n"
    )


def test_summarize_seed_samples_reports_top_instructions_by_cell(tmp_path: Path):
    module = _load_module()
    cells_config = tmp_path / "cells.tsv"
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_cells_config(cells_config)
    (shard_dir / "ppcc_potato_scan_samples.tsv").write_text(
        "cell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tis_match\n"
        "ppcc_potato\tPickPlaceCounterToCabinet\tenv\t100000\t"
        "Pick the steak from the counter and place it in the cabinet.\t"
        "Pick the potato from the counter and place it in the cabinet.\t0\n"
        "ppcc_potato\tPickPlaceCounterToCabinet\tenv\t100001\t"
        "Pick the steak from the counter and place it in the cabinet.\t"
        "Pick the potato from the counter and place it in the cabinet.\t0\n"
        "ppcc_potato\tPickPlaceCounterToCabinet\tenv\t100002\t"
        "Pick the potato from the counter and place it in the cabinet.\t"
        "Pick the potato from the counter and place it in the cabinet.\t1\n"
    )
    (shard_dir / "coffee_setup_mug_scan_samples.tsv").write_text(
        "cell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tis_match\n"
        "coffee_setup_mug\tCoffeeSetupMug\tenv\t100000\t"
        "Pick the mug from the counter and place it under the coffee machine dispenser.\t"
        "Pick the mug from the counter and place it under the coffee machine dispenser.\t1\n"
    )

    rows = module.summarize_seed_samples(
        shard_dir=shard_dir,
        cells_config=cells_config,
        top_n=2,
    )

    assert rows == [
        {
            "cell_index": 0,
            "cell_id": "ppcc_potato",
            "task": "PickPlaceCounterToCabinet",
            "canonical_instruction": "Pick the potato from the counter and place it in the cabinet.",
            "scanned": 3,
            "matches": 1,
            "match_rate": "0.333333",
            "unique_instructions": 2,
            "top_instructions": (
                "2::Pick the steak from the counter and place it in the cabinet. | "
                "1::Pick the potato from the counter and place it in the cabinet."
            ),
        },
        {
            "cell_index": 1,
            "cell_id": "coffee_setup_mug",
            "task": "CoffeeSetupMug",
            "canonical_instruction": (
                "Pick the mug from the counter and place it under the coffee machine dispenser."
            ),
            "scanned": 1,
            "matches": 1,
            "match_rate": "1.000000",
            "unique_instructions": 1,
            "top_instructions": (
                "1::Pick the mug from the counter and place it under the coffee machine dispenser."
            ),
        },
    ]


def test_write_sample_summary_tsv_uses_stable_columns(tmp_path: Path):
    module = _load_module()
    output_path = tmp_path / "seed_sample_summary.tsv"
    module.write_sample_summary_tsv(
        output_path,
        [
            {
                "cell_index": 0,
                "cell_id": "ppcc_potato",
                "task": "PickPlaceCounterToCabinet",
                "canonical_instruction": "Pick the potato from the counter and place it in the cabinet.",
                "scanned": 3,
                "matches": 1,
                "match_rate": "0.333333",
                "unique_instructions": 2,
                "top_instructions": "2::x | 1::y",
            }
        ],
    )

    with output_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    assert rows == [
        {
            "cell_index": "0",
            "cell_id": "ppcc_potato",
            "task": "PickPlaceCounterToCabinet",
            "canonical_instruction": "Pick the potato from the counter and place it in the cabinet.",
            "scanned": "3",
            "matches": "1",
            "match_rate": "0.333333",
            "unique_instructions": "2",
            "top_instructions": "2::x | 1::y",
        }
    ]
