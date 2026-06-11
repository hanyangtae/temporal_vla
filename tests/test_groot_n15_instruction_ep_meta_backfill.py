from __future__ import annotations

import csv
import importlib.util
import json
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKFILL_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "collect"
    / "backfill_instruction_ep_meta.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_ep_meta_backfill_under_test",
        BACKFILL_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_selected_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
        "11\tturn_on_sink_faucet\tTurnOnSinkFaucet\t"
        "robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env\t100007\t"
        "Turn on the sink faucet.\tTurn on the sink faucet.\told/ep_meta.json\n"
    )


def _write_rollout_pkl(path: Path, *, scenario_seed: int = 100007) -> None:
    path.parent.mkdir(parents=True)
    payload = {
        "cell_id": "turn_on_sink_faucet",
        "cell_index": 11,
        "robocasa_task": "TurnOnSinkFaucet",
        "canonical_instruction": "Turn on the sink faucet.",
        "task_description": "Turn on the sink faucet.",
        "scenario_seed": scenario_seed,
        "env_name": "robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env",
        "robocasa_env_source": "robocasa365",
        "ep_meta": {"lang": "Turn on the sink faucet.", "layout_id": 7},
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_backfill_writes_ep_meta_json_and_rewrites_selected_manifest(tmp_path: Path):
    module = _load_module()
    raw_rollouts = tmp_path / "raw_rollouts"
    selected = tmp_path / "manifests" / "selected_instruction_seeds.tsv"
    ep_meta_root = tmp_path / "ep_meta"
    _write_selected_manifest(selected)
    _write_rollout_pkl(
        raw_rollouts
        / "TurnOnSinkFaucet"
        / "turn_on_sink_faucet"
        / "task11--ep7--succ0.pkl"
    )

    summary = module.backfill_ep_meta(
        raw_rollouts=raw_rollouts,
        selected_seeds=selected,
        ep_meta_root=ep_meta_root,
        rewrite_selected_seeds=True,
    )

    assert summary["processed_pkls"] == 1
    assert summary["written_json"] == 1
    assert summary["missing_manifest_rows"] == 0
    target_path = (
        ep_meta_root
        / "TurnOnSinkFaucet"
        / "turn_on_sink_faucet"
        / "robocasa_panda_omron_TurnOnSinkFaucet_PandaOmron_Env--seed100007.json"
    )
    assert target_path.exists()
    with target_path.open() as f:
        payload = json.load(f)
    assert payload["env_name"] == "robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env"
    assert payload["scenario_seed"] == 100007
    assert payload["robocasa_env_source"] == "robocasa365"
    assert payload["ep_meta"]["lang"] == "Turn on the sink faucet."

    with selected.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows[0]["ep_meta_path"] == str(target_path)


def test_backfill_reports_missing_manifest_rows(tmp_path: Path):
    module = _load_module()
    raw_rollouts = tmp_path / "raw_rollouts"
    selected = tmp_path / "manifests" / "selected_instruction_seeds.tsv"
    _write_selected_manifest(selected)
    _write_rollout_pkl(
        raw_rollouts
        / "TurnOnSinkFaucet"
        / "turn_on_sink_faucet"
        / "task11--ep7--succ0.pkl",
        scenario_seed=100008,
    )

    summary = module.backfill_ep_meta(
        raw_rollouts=raw_rollouts,
        selected_seeds=selected,
        ep_meta_root=tmp_path / "ep_meta",
        rewrite_selected_seeds=False,
    )

    assert summary["processed_pkls"] == 1
    assert summary["written_json"] == 1
    assert summary["missing_manifest_rows"] == 1
