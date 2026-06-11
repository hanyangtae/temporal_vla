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
    / "summarize_instruction_seed_scan.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_seed_scan_summary_under_test",
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
        "0\topen_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t"
        "Open the right drawer.\t2\t100000\n"
        "1\topen_drawer_left\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t"
        "Open the left drawer.\t2\t100000\n"
        "2\topen_cabinet_door\tOpenCabinet\t"
        "robocasa_panda_omron/OpenCabinet_PandaOmron_Env\t"
        "Open the cabinet door.\t2\t100000\n"
    )


def test_summarize_seed_scan_reports_cell_progress_and_integrity(tmp_path: Path):
    module = _load_module()
    cells_config = tmp_path / "cells.tsv"
    selected_tsv = tmp_path / "selected_instruction_seeds.tsv"
    progress_tsv = tmp_path / "selected_instruction_seeds_scan_progress.tsv"
    samples_tsv = tmp_path / "selected_instruction_seeds_scan_samples.tsv"
    ep0 = tmp_path / "ep0.json"
    ep1 = tmp_path / "ep1.json"
    ep0.write_text("{}")
    ep1.write_text("{}")
    _write_cells_config(cells_config)
    selected_tsv.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
        "0\topen_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100001\t"
        "Open the right drawer.\tOpen the right drawer.\tep0.json\n"
        "0\topen_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100002\t"
        "Open the right drawer.\tOpen the right drawer.\tep1.json\n"
        "1\topen_drawer_left\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100003\t"
        "Open the right drawer.\tOpen the left drawer.\tmissing.json\n"
    )
    progress_tsv.write_text(
        "cell_id\ttask\tenv_name\tlast_scanned_seed\tscanned\tfound\ttarget_count\n"
        "open_drawer_left\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100003\t4\t1\t2\n"
    )
    samples_tsv.write_text(
        "cell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tis_match\n"
        "open_drawer_left\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100003\t"
        "Open the right drawer.\tOpen the left drawer.\t0\n"
    )

    rows = module.summarize_seed_scan(
        cells_config=cells_config,
        selected_tsv=selected_tsv,
        path_base=tmp_path,
    )

    assert rows == [
        {
            "cell_index": 0,
            "cell_id": "open_drawer_right",
            "task": "OpenDrawer",
            "target_count": 2,
            "selected": 2,
            "instruction_mismatch": 0,
            "duplicate_seed_count": 0,
            "ep_meta_missing": 0,
            "scan_samples": 0,
            "last_scanned_seed": 100002,
            "progress_found": 2,
            "status": "complete",
        },
        {
            "cell_index": 1,
            "cell_id": "open_drawer_left",
            "task": "OpenDrawer",
            "target_count": 2,
            "selected": 1,
            "instruction_mismatch": 1,
            "duplicate_seed_count": 0,
            "ep_meta_missing": 1,
            "scan_samples": 1,
            "last_scanned_seed": 100003,
            "progress_found": 1,
            "status": "in_progress",
        },
        {
            "cell_index": 2,
            "cell_id": "open_cabinet_door",
            "task": "OpenCabinet",
            "target_count": 2,
            "selected": 0,
            "instruction_mismatch": 0,
            "duplicate_seed_count": 0,
            "ep_meta_missing": 0,
            "scan_samples": 0,
            "last_scanned_seed": "",
            "progress_found": 0,
            "status": "not_started",
        },
    ]


def test_write_summary_tsv_uses_stable_columns(tmp_path: Path):
    module = _load_module()
    output_path = tmp_path / "seed_scan_status.tsv"

    module.write_summary_tsv(
        output_path,
        [
            {
                "cell_index": 0,
                "cell_id": "open_drawer_right",
                "task": "OpenDrawer",
                "target_count": 2,
                "selected": 2,
                "instruction_mismatch": 0,
                "duplicate_seed_count": 0,
                "ep_meta_missing": 0,
                "scan_samples": 2,
                "last_scanned_seed": 100002,
                "progress_found": 2,
                "status": "complete",
            }
        ],
    )

    with output_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    assert rows == [
        {
            "cell_index": "0",
            "cell_id": "open_drawer_right",
            "task": "OpenDrawer",
            "target_count": "2",
            "selected": "2",
            "instruction_mismatch": "0",
            "duplicate_seed_count": "0",
            "ep_meta_missing": "0",
            "scan_samples": "2",
            "last_scanned_seed": "100002",
            "progress_found": "2",
            "status": "complete",
        }
    ]


def test_summarize_seed_scan_discovers_sharded_progress_and_samples(tmp_path: Path):
    module = _load_module()
    cells_config = tmp_path / "cells.tsv"
    manifest_dir = tmp_path / "manifests"
    shard_dir = manifest_dir / "shards"
    selected_tsv = manifest_dir / "selected_instruction_seeds.partial.tsv"
    shard_dir.mkdir(parents=True)
    ep0 = tmp_path / "ep0.json"
    ep0.write_text("{}")
    _write_cells_config(cells_config)
    selected_tsv.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
        "0\topen_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100001\t"
        "Open the right drawer.\tOpen the right drawer.\tep0.json\n"
    )
    (shard_dir / "open_drawer_right_scan_progress.tsv").write_text(
        "cell_id\ttask\tenv_name\tlast_scanned_seed\tscanned\tfound\ttarget_count\n"
        "open_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100006\t7\t1\t2\n"
    )
    (shard_dir / "open_drawer_right_scan_samples.tsv").write_text(
        "cell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tis_match\n"
        "open_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100000\t"
        "Open the left drawer.\tOpen the right drawer.\t0\n"
        "open_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100001\t"
        "Open the right drawer.\tOpen the right drawer.\t1\n"
    )

    rows = module.summarize_seed_scan(
        cells_config=cells_config,
        selected_tsv=selected_tsv,
        path_base=tmp_path,
    )

    right = next(row for row in rows if row["cell_id"] == "open_drawer_right")
    assert right["scan_samples"] == 2
    assert right["last_scanned_seed"] == 100006
    assert right["progress_found"] == 1


def test_summarize_seed_scan_aggregates_multiple_range_shards(tmp_path: Path):
    module = _load_module()
    cells_config = tmp_path / "cells.tsv"
    manifest_dir = tmp_path / "manifests"
    shard_dir = manifest_dir / "shards"
    selected_tsv = manifest_dir / "selected_instruction_seeds.partial.tsv"
    shard_dir.mkdir(parents=True)
    ep0 = tmp_path / "ep0.json"
    ep1 = tmp_path / "ep1.json"
    ep2 = tmp_path / "ep2.json"
    ep0.write_text("{}")
    ep1.write_text("{}")
    ep2.write_text("{}")
    _write_cells_config(cells_config)
    selected_tsv.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
        "0\topen_drawer_right\tOpenDrawer\tenv\t100001\t"
        "Open the right drawer.\tOpen the right drawer.\tep0.json\n"
        "0\topen_drawer_right\tOpenDrawer\tenv\t200001\t"
        "Open the right drawer.\tOpen the right drawer.\tep1.json\n"
        "0\topen_drawer_right\tOpenDrawer\tenv\t200002\t"
        "Open the right drawer.\tOpen the right drawer.\tep2.json\n"
    )
    (shard_dir / "open_drawer_right_scan_progress.tsv").write_text(
        "cell_id\ttask\tenv_name\tlast_scanned_seed\tscanned\tfound\ttarget_count\n"
        "open_drawer_right\tOpenDrawer\tenv\t100006\t7\t1\t2\n"
    )
    (shard_dir / "open_drawer_right_r1_scan_progress.tsv").write_text(
        "cell_id\ttask\tenv_name\tlast_scanned_seed\tscanned\tfound\ttarget_count\n"
        "open_drawer_right\tOpenDrawer\tenv\t200010\t11\t2\t2\n"
    )
    (shard_dir / "open_drawer_right_scan_samples.tsv").write_text(
        "cell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tis_match\n"
        "open_drawer_right\tOpenDrawer\tenv\t100001\t"
        "Open the right drawer.\tOpen the right drawer.\t1\n"
    )
    (shard_dir / "open_drawer_right_r1_scan_samples.tsv").write_text(
        "cell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tis_match\n"
        "open_drawer_right\tOpenDrawer\tenv\t200001\t"
        "Open the right drawer.\tOpen the right drawer.\t1\n"
        "open_drawer_right\tOpenDrawer\tenv\t200002\t"
        "Open the right drawer.\tOpen the right drawer.\t1\n"
    )

    rows = module.summarize_seed_scan(
        cells_config=cells_config,
        selected_tsv=selected_tsv,
        path_base=tmp_path,
    )

    right = next(row for row in rows if row["cell_id"] == "open_drawer_right")
    assert right["selected"] == 3
    assert right["scan_samples"] == 3
    assert right["last_scanned_seed"] == 200010
    assert right["progress_found"] == 3
