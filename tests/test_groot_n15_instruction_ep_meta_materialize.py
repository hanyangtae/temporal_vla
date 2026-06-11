from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "collect"
    / "materialize_selected_ep_meta.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_ep_meta_materialize_under_test",
        MATERIALIZE_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_selected_manifest(path: Path, ep_meta_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "cell_index": "12",
            "cell_id": "navigate_fridge",
            "task": "NavigateKitchen",
            "env_name": "robocasa_panda_omron/NavigateKitchen_PandaOmron_Env",
            "scenario_seed": "100023",
            "instruction": "Navigate to the fridge.",
            "canonical_instruction": "Navigate to the fridge.",
            "ep_meta_path": str(ep_meta_root / "navigate_fridge_seed100023.json"),
        },
        {
            "cell_index": "13",
            "cell_id": "navigate_coffee_machine",
            "task": "NavigateKitchen",
            "env_name": "robocasa_panda_omron/NavigateKitchen_PandaOmron_Env",
            "scenario_seed": "100047",
            "instruction": "Navigate to the coffee machine.",
            "canonical_instruction": "Navigate to the coffee machine.",
            "ep_meta_path": str(ep_meta_root / "navigate_coffee_seed100047.json"),
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cell_index",
                "cell_id",
                "task",
                "env_name",
                "scenario_seed",
                "instruction",
                "canonical_instruction",
                "ep_meta_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


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


def test_materialize_selected_ep_meta_writes_missing_json_for_requested_cell(
    tmp_path: Path,
):
    module = _load_module()
    selected = tmp_path / "selected.tsv"
    ep_meta_root = tmp_path / "ep_meta"
    _write_selected_manifest(selected, ep_meta_root)
    seen = []

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        seen.append((env_name, task, scenario_seed, enable_render))
        return FakeEnv("Navigate to the fridge.")

    summary = module.materialize_selected_ep_meta(
        selected_seeds=selected,
        cell_ids=["navigate_fridge"],
        make_env_fn=fake_make_env,
    )

    assert summary["selected_rows"] == 1
    assert summary["written_json"] == 1
    assert seen == [
        (
            "robocasa_panda_omron/NavigateKitchen_PandaOmron_Env",
            "NavigateKitchen",
            100023,
            False,
        )
    ]
    target = ep_meta_root / "navigate_fridge_seed100023.json"
    with target.open() as f:
        payload = json.load(f)
    assert payload["env_name"] == "robocasa_panda_omron/NavigateKitchen_PandaOmron_Env"
    assert payload["scenario_seed"] == 100023
    assert payload["ep_meta"]["lang"] == "Navigate to the fridge."
    assert payload["robocasa_env_source"] == "robocasa365"


def test_materialize_selected_ep_meta_fails_on_instruction_mismatch(tmp_path: Path):
    module = _load_module()
    selected = tmp_path / "selected.tsv"
    _write_selected_manifest(selected, tmp_path / "ep_meta")

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        return FakeEnv("Navigate to the toaster.")

    try:
        module.materialize_selected_ep_meta(
            selected_seeds=selected,
            cell_ids=["navigate_fridge"],
            make_env_fn=fake_make_env,
        )
    except RuntimeError as exc:
        assert "instruction mismatch" in str(exc)
        assert "Navigate to the toaster." in str(exc)
    else:
        raise AssertionError("materialize should fail on instruction mismatch")


def test_materialize_selected_ep_meta_can_continue_on_instruction_mismatch(
    tmp_path: Path,
):
    module = _load_module()
    selected = tmp_path / "selected.tsv"
    _write_selected_manifest(selected, tmp_path / "ep_meta")

    langs_by_seed = {
        100023: "Navigate to the toaster.",
        100047: "Navigate to the coffee machine.",
    }

    def fake_make_env(env_name, *, task, scenario_seed, enable_render):
        return FakeEnv(langs_by_seed[scenario_seed])

    summary = module.materialize_selected_ep_meta(
        selected_seeds=selected,
        make_env_fn=fake_make_env,
        continue_on_mismatch=True,
    )

    assert summary["selected_rows"] == 2
    assert summary["instruction_mismatches"] == 1
    assert summary["skipped_mismatches"] == 1
    assert summary["written_json"] == 1
    assert not (tmp_path / "ep_meta" / "navigate_fridge_seed100023.json").exists()
    assert (tmp_path / "ep_meta" / "navigate_coffee_seed100047.json").exists()
