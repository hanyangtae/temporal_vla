from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTOR_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "collect"
    / "select_instruction_seeds.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_seed_selector_under_test",
        SELECTOR_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cells_config(path: Path) -> None:
    path.write_text(
        "\t".join(
            [
                "cell_index",
                "cell_id",
                "task",
                "env_name",
                "canonical_instruction",
                "target_episodes",
                "seed_search_start",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "0",
                "open_drawer_right",
                "OpenDrawer",
                "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                "Open the right drawer.",
                "2",
                "100000",
            ]
        )
        + "\n"
    )


class FakeEnv:
    def __init__(self, lang: str):
        self.lang = lang
        self.closed = False

    def reset(self, seed=None):
        return {"annotation.human.task_description": self.lang}, {}

    def get_ep_meta(self):
        return {"lang": self.lang, "layout_id": 7}

    def close(self):
        self.closed = True


def test_selector_writes_matching_seed_rows_and_ep_meta_manifests(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    output_tsv = tmp_path / "selected.tsv"
    ep_meta_dir = tmp_path / "ep_meta"
    _write_cells_config(cells_path)

    langs_by_seed = {
        100000: "Open the left drawer.",
        100001: "Open the right drawer.",
        100002: "Open the right drawer.",
    }

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        assert env_name == "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
        assert task == "OpenDrawer"
        assert enable_render is False
        return FakeEnv(langs_by_seed[scenario_seed])

    rows = module.select_instruction_seeds(
        cells_config=cells_path,
        output_tsv=output_tsv,
        ep_meta_dir=ep_meta_dir,
        max_seeds_per_cell=3,
        target_per_cell=None,
        seed_start=None,
        cell_ids=None,
        enable_render=False,
        make_env_fn=fake_make_env,
    )

    assert [row["scenario_seed"] for row in rows] == [100001, 100002]
    assert output_tsv.is_file()

    with output_tsv.open(newline="") as f:
        written_rows = list(csv.DictReader(f, delimiter="\t"))

    assert [row["scenario_seed"] for row in written_rows] == ["100001", "100002"]
    assert all(row["cell_id"] == "open_drawer_right" for row in written_rows)
    assert all(row["instruction"] == "Open the right drawer." for row in written_rows)
    assert "env_kwargs_json" not in written_rows[0]

    for row in written_rows:
        manifest_path = Path(row["ep_meta_path"])
        assert manifest_path.is_file()
        payload = json.loads(manifest_path.read_text())
        assert payload["env_name"] == "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
        assert payload["scenario_seed"] == int(row["scenario_seed"])
        assert payload["ep_meta"]["lang"] == "Open the right drawer."
        assert payload["robocasa_env_source"] == "robocasa365"


def test_selector_fails_when_canonical_instruction_is_not_found(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    _write_cells_config(cells_path)

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        assert task == "OpenDrawer"
        return FakeEnv("Open the left drawer.")

    try:
        module.select_instruction_seeds(
            cells_config=cells_path,
            output_tsv=tmp_path / "selected.tsv",
            ep_meta_dir=tmp_path / "ep_meta",
            max_seeds_per_cell=2,
            target_per_cell=1,
            seed_start=None,
            cell_ids=None,
            enable_render=False,
            make_env_fn=fake_make_env,
        )
    except RuntimeError as exc:
        assert "open_drawer_right" in str(exc)
        assert "found 0/1" in str(exc)
    else:
        raise AssertionError("selector should fail when no matching seed exists")


def test_selector_resume_continues_after_existing_rows(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    output_tsv = tmp_path / "selected.tsv"
    ep_meta_dir = tmp_path / "ep_meta"
    _write_cells_config(cells_path)
    module.write_selected_seed_manifest(
        output_tsv,
        [
            {
                "cell_index": 0,
                "cell_id": "open_drawer_right",
                "task": "OpenDrawer",
                "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                "scenario_seed": 100001,
                "instruction": "Open the right drawer.",
                "canonical_instruction": "Open the right drawer.",
                "ep_meta_path": "/tmp/ep0.json",
            }
        ],
    )
    seen_seeds = []

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        assert task == "OpenDrawer"
        seen_seeds.append(scenario_seed)
        return FakeEnv("Open the right drawer.")

    rows = module.select_instruction_seeds(
        cells_config=cells_path,
        output_tsv=output_tsv,
        ep_meta_dir=ep_meta_dir,
        max_seeds_per_cell=5,
        target_per_cell=2,
        seed_start=None,
        cell_ids=None,
        enable_render=False,
        progress_interval=0,
        resume=True,
        make_env_fn=fake_make_env,
    )

    assert seen_seeds == [100002]
    assert [row["scenario_seed"] for row in rows] == [100001, 100002]


def test_selector_resume_continues_after_scan_progress(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    output_tsv = tmp_path / "selected.tsv"
    scan_progress_tsv = tmp_path / "scan_progress.tsv"
    _write_cells_config(cells_path)
    module.write_scan_progress(
        scan_progress_tsv,
        {
            "open_drawer_right": {
                "cell_id": "open_drawer_right",
                "task": "OpenDrawer",
                "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                "last_scanned_seed": 100005,
                "scanned": 6,
                "found": 0,
                "target_count": 1,
            }
        },
    )
    seen_seeds = []

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        assert task == "OpenDrawer"
        seen_seeds.append(scenario_seed)
        return FakeEnv("Open the right drawer.")

    rows = module.select_instruction_seeds(
        cells_config=cells_path,
        output_tsv=output_tsv,
        ep_meta_dir=tmp_path / "ep_meta",
        max_seeds_per_cell=5,
        target_per_cell=1,
        seed_start=None,
        cell_ids=None,
        enable_render=False,
        progress_interval=0,
        resume=True,
        scan_progress_tsv=scan_progress_tsv,
        make_env_fn=fake_make_env,
    )

    assert seen_seeds == [100006]
    assert [row["scenario_seed"] for row in rows] == [100006]


def test_selector_writes_scan_samples_for_matches_and_non_matches(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    output_tsv = tmp_path / "selected.tsv"
    scan_samples_tsv = tmp_path / "scan_samples.tsv"
    _write_cells_config(cells_path)
    langs_by_seed = {
        100000: "Open the left drawer.",
        100001: "Open the right drawer.",
    }

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        assert task == "OpenDrawer"
        return FakeEnv(langs_by_seed[scenario_seed])

    module.select_instruction_seeds(
        cells_config=cells_path,
        output_tsv=output_tsv,
        ep_meta_dir=tmp_path / "ep_meta",
        max_seeds_per_cell=5,
        target_per_cell=1,
        seed_start=None,
        cell_ids=None,
        enable_render=False,
        progress_interval=0,
        scan_samples_tsv=scan_samples_tsv,
        make_env_fn=fake_make_env,
    )

    with scan_samples_tsv.open(newline="") as f:
        samples = list(csv.DictReader(f, delimiter="\t"))

    assert [row["scenario_seed"] for row in samples] == ["100000", "100001"]
    assert [row["instruction"] for row in samples] == [
        "Open the left drawer.",
        "Open the right drawer.",
    ]
    assert [row["is_match"] for row in samples] == ["0", "1"]


def test_selector_skips_excluded_matching_seeds(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    output_tsv = tmp_path / "selected.tsv"
    excluded_tsv = tmp_path / "excluded.tsv"
    ep_meta_dir = tmp_path / "ep_meta"
    _write_cells_config(cells_path)
    module.write_selected_seed_manifest(
        excluded_tsv,
        [
            {
                "cell_index": 0,
                "cell_id": "open_drawer_right",
                "task": "OpenDrawer",
                "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                "scenario_seed": 100000,
                "instruction": "Open the right drawer.",
                "canonical_instruction": "Open the right drawer.",
                "ep_meta_path": "/tmp/excluded.json",
            }
        ],
    )
    langs_by_seed = {
        100000: "Open the right drawer.",
        100001: "Open the left drawer.",
        100002: "Open the right drawer.",
    }

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        assert task == "OpenDrawer"
        return FakeEnv(langs_by_seed[scenario_seed])

    rows = module.select_instruction_seeds(
        cells_config=cells_path,
        output_tsv=output_tsv,
        ep_meta_dir=ep_meta_dir,
        max_seeds_per_cell=3,
        target_per_cell=1,
        seed_start=None,
        cell_ids=None,
        exclude_selected_seeds=[excluded_tsv],
        enable_render=False,
        progress_interval=0,
        make_env_fn=fake_make_env,
    )

    assert [row["scenario_seed"] for row in rows] == [100002]
    assert not list((ep_meta_dir / "OpenDrawer" / "open_drawer_right").glob("*100000*.json"))


def test_selector_resume_fails_when_existing_row_overlaps_excluded_manifest(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    output_tsv = tmp_path / "selected.tsv"
    excluded_tsv = tmp_path / "excluded.tsv"
    _write_cells_config(cells_path)
    row = {
        "cell_index": 0,
        "cell_id": "open_drawer_right",
        "task": "OpenDrawer",
        "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
        "scenario_seed": 100000,
        "instruction": "Open the right drawer.",
        "canonical_instruction": "Open the right drawer.",
        "ep_meta_path": "/tmp/seed100000.json",
    }
    module.write_selected_seed_manifest(output_tsv, [row])
    module.write_selected_seed_manifest(excluded_tsv, [row])

    try:
        module.select_instruction_seeds(
            cells_config=cells_path,
            output_tsv=output_tsv,
            ep_meta_dir=tmp_path / "ep_meta",
            max_seeds_per_cell=3,
            target_per_cell=1,
            seed_start=None,
            cell_ids=None,
            exclude_selected_seeds=[excluded_tsv],
            enable_render=False,
            progress_interval=0,
            resume=True,
            make_env_fn=lambda *args, **kwargs: FakeEnv("Open the right drawer."),
        )
    except ValueError as exc:
        assert "overlap excluded manifests" in str(exc)
        assert "open_drawer_right/100000" in str(exc)
    else:
        raise AssertionError("selector should reject resumed held-out leakage")


def test_selector_resume_continues_after_scan_samples(tmp_path: Path):
    module = _load_module()
    cells_path = tmp_path / "cells.tsv"
    output_tsv = tmp_path / "selected.tsv"
    scan_samples_tsv = tmp_path / "scan_samples.tsv"
    _write_cells_config(cells_path)
    module.write_scan_samples(
        scan_samples_tsv,
        [
            {
                "cell_id": "open_drawer_right",
                "task": "OpenDrawer",
                "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                "scenario_seed": 100004,
                "instruction": "Open the left drawer.",
                "canonical_instruction": "Open the right drawer.",
                "is_match": 0,
            }
        ],
    )
    seen_seeds = []

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        assert task == "OpenDrawer"
        seen_seeds.append(scenario_seed)
        return FakeEnv("Open the right drawer.")

    rows = module.select_instruction_seeds(
        cells_config=cells_path,
        output_tsv=output_tsv,
        ep_meta_dir=tmp_path / "ep_meta",
        max_seeds_per_cell=5,
        target_per_cell=1,
        seed_start=None,
        cell_ids=None,
        enable_render=False,
        progress_interval=0,
        resume=True,
        scan_samples_tsv=scan_samples_tsv,
        make_env_fn=fake_make_env,
    )

    assert seen_seeds == [100005]
    assert [row["scenario_seed"] for row in rows] == [100005]
