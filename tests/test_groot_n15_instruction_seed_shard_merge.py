from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "collect"
    / "merge_instruction_seed_shards.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_seed_shard_merge_under_test",
        MERGE_SCRIPT,
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
    )


def _write_selected_manifest(path: Path, rows: list[tuple[str, int, str]]) -> None:
    lines = [
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path"
    ]
    for cell_id, seed, instruction in rows:
        cell_index = "0" if cell_id == "open_drawer_right" else "1"
        lines.append(
            "\t".join(
                [
                    cell_index,
                    cell_id,
                    "OpenDrawer",
                    "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                    str(seed),
                    instruction,
                    instruction,
                    f"ep_meta/{cell_id}/{seed}.json",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def test_merge_seed_shards_sorts_and_truncates_by_cell_target(tmp_path: Path):
    module = _load_module()
    cells_config = tmp_path / "cells.tsv"
    shard_a = tmp_path / "right.tsv"
    shard_b = tmp_path / "left.tsv"
    _write_cells_config(cells_config)
    _write_selected_manifest(
        shard_a,
        [
            ("open_drawer_right", 100002, "Open the right drawer."),
            ("open_drawer_right", 100001, "Open the right drawer."),
            ("open_drawer_right", 100003, "Open the right drawer."),
        ],
    )
    _write_selected_manifest(
        shard_b,
        [
            ("open_drawer_left", 100011, "Open the left drawer."),
            ("open_drawer_left", 100010, "Open the left drawer."),
        ],
    )

    rows = module.merge_seed_shards(
        selected_seed_paths=[shard_b, shard_a],
        cells_config=cells_config,
        require_complete=True,
    )

    assert [(row["cell_id"], row["scenario_seed"]) for row in rows] == [
        ("open_drawer_right", 100001),
        ("open_drawer_right", 100002),
        ("open_drawer_left", 100010),
        ("open_drawer_left", 100011),
    ]


def test_merge_seed_shards_rejects_old_env_kwargs_schema(tmp_path: Path):
    module = _load_module()
    cells_config = tmp_path / "cells.tsv"
    shard = tmp_path / "old.tsv"
    _write_cells_config(cells_config)
    shard.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tenv_kwargs_json\tep_meta_path\n"
    )

    try:
        module.merge_seed_shards(
            selected_seed_paths=[shard],
            cells_config=cells_config,
        )
    except ValueError as exc:
        assert "env_kwargs_json is not supported" in str(exc)
    else:
        raise AssertionError("old env_kwargs_json shard should be rejected")


def test_merge_seed_shards_require_complete_reports_missing_cells(tmp_path: Path):
    module = _load_module()
    cells_config = tmp_path / "cells.tsv"
    shard = tmp_path / "right.tsv"
    _write_cells_config(cells_config)
    _write_selected_manifest(
        shard,
        [("open_drawer_right", 100001, "Open the right drawer.")],
    )

    try:
        module.merge_seed_shards(
            selected_seed_paths=[shard],
            cells_config=cells_config,
            require_complete=True,
        )
    except RuntimeError as exc:
        assert "open_drawer_right=1/2" in str(exc)
        assert "open_drawer_left=0/2" in str(exc)
    else:
        raise AssertionError("require_complete should reject incomplete shards")


def test_write_selected_seed_manifest_uses_seed_only_schema(tmp_path: Path):
    module = _load_module()
    output_path = tmp_path / "merged.tsv"
    module.write_selected_seed_manifest(
        output_path,
        [
            {
                "cell_index": 0,
                "cell_id": "open_drawer_right",
                "task": "OpenDrawer",
                "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                "scenario_seed": 100001,
                "instruction": "Open the right drawer.",
                "canonical_instruction": "Open the right drawer.",
                "ep_meta_path": "ep_meta/open_drawer_right/100001.json",
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
            "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
            "scenario_seed": "100001",
            "instruction": "Open the right drawer.",
            "canonical_instruction": "Open the right drawer.",
            "ep_meta_path": "ep_meta/open_drawer_right/100001.json",
        }
    ]
    assert "env_kwargs_json" not in rows[0]


def test_discover_seed_shard_paths_ignores_scan_audit_files(tmp_path: Path):
    module = _load_module()
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    selected_a = shard_dir / "open_drawer_right.tsv"
    selected_b = shard_dir / "open_drawer_left.tsv"
    progress = shard_dir / "open_drawer_right_scan_progress.tsv"
    samples = shard_dir / "open_drawer_right_scan_samples.tsv"
    selected_a.write_text("")
    selected_b.write_text("")
    progress.write_text("")
    samples.write_text("")

    assert module.discover_seed_shard_paths(shard_dir) == [selected_b, selected_a]
