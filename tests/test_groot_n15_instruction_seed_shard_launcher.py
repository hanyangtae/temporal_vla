from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "collect"
    / "launch_instruction_seed_shards.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_seed_shard_launcher_under_test",
        LAUNCH_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cell(cell_id: str, task: str = "OpenDrawer"):
    module = _load_module()
    return module.InstructionCell(cell_id=cell_id, task=task)


def test_session_name_uses_stable_short_slug_for_long_cells():
    module = _load_module()

    assert (
        module.session_name("ppdc_wooden_spoon", prefix="n15_seedscan_")
        == "n15_seedscan_ppdc_spoon"
    )
    assert (
        module.session_name("navigate_coffee_machine", prefix="n15_seedscan_")
        == "n15_seedscan_navigate_coffee"
    )
    assert (
        module.session_name("ppcc_potato", prefix="n15_seedscan_", shard_suffix="_r1")
        == "n15_seedscan_ppcc_potato_r1"
    )


def test_plan_launches_skips_active_sessions_and_caps_new_sessions(tmp_path: Path):
    module = _load_module()
    cells = [
        module.InstructionCell("ppcs_onion", "PickPlaceCounterToStove"),
        module.InstructionCell("ppcs_apple", "PickPlaceCounterToStove"),
        module.InstructionCell("ppcc_potato", "PickPlaceCounterToCabinet"),
    ]

    plans = module.plan_launches(
        cells,
        active_sessions={"n15_seedscan_ppcs_onion"},
        selected_cell_ids=None,
        session_prefix="n15_seedscan_",
        run_root=tmp_path,
        container_name="robocasa",
        repo_root=Path("/repo"),
        container_workdir="/temporal_vla",
        target_per_cell=50,
        max_seeds_per_cell=10000,
        progress_interval=25,
        max_new_sessions=1,
    )

    assert [(plan.cell_id, plan.reason) for plan in plans] == [
        ("ppcs_onion", "active"),
        ("ppcs_apple", "launch"),
        ("ppcc_potato", "max_new_sessions"),
    ]


def test_build_tmux_command_uses_seed_only_shard_and_log_paths(tmp_path: Path):
    module = _load_module()
    cell = module.InstructionCell("ppcc_potato", "PickPlaceCounterToCabinet")

    command = module.build_tmux_command(
        cell,
        session="n15_seedscan_ppcc_potato",
        run_root=tmp_path / "run",
        container_name="robocasa",
        repo_root=Path("/repo"),
        container_workdir="/temporal_vla",
        target_per_cell=50,
        max_seeds_per_cell=10000,
        progress_interval=25,
    )

    assert command[:5] == ["tmux", "new-session", "-d", "-s", "n15_seedscan_ppcc_potato"]
    inner = command[-1]
    assert "select_instruction_seeds.py" in inner
    assert "--cell-id ppcc_potato" in inner
    assert "--target-per-cell 50" in inner
    assert "--max-seeds-per-cell 10000" in inner
    assert f"--output-tsv {tmp_path / 'run' / 'manifests' / 'shards' / 'ppcc_potato.tsv'}" in inner
    assert f">> {tmp_path / 'run' / 'logs' / 'ppcc_potato.log'} 2>&1" in inner
    assert "--env-kwargs-json" not in inner


def test_build_tmux_command_can_launch_disjoint_seed_range_shard(tmp_path: Path):
    module = _load_module()
    cell = module.InstructionCell("ppcc_potato", "PickPlaceCounterToCabinet")

    command = module.build_tmux_command(
        cell,
        session="n15_seedscan_ppcc_potato_r1",
        run_root=tmp_path / "run",
        container_name="robocasa",
        repo_root=Path("/repo"),
        container_workdir="/temporal_vla",
        target_per_cell=50,
        max_seeds_per_cell=10000,
        progress_interval=25,
        shard_suffix="_r1",
        seed_start=200000,
    )

    inner = command[-1]
    assert "--seed-start 200000" in inner
    assert f"--output-tsv {tmp_path / 'run' / 'manifests' / 'shards' / 'ppcc_potato_r1.tsv'}" in inner
    assert f">> {tmp_path / 'run' / 'logs' / 'ppcc_potato_r1.log'} 2>&1" in inner


def test_plan_launches_skips_completed_shards(tmp_path: Path):
    module = _load_module()
    cells = [
        module.InstructionCell("ppcs_onion", "PickPlaceCounterToStove"),
        module.InstructionCell("ppcs_apple", "PickPlaceCounterToStove"),
    ]

    plans = module.plan_launches(
        cells,
        active_sessions=set(),
        completed_cell_ids={"ppcs_onion"},
        selected_cell_ids=None,
        session_prefix="n15_seedscan_",
        run_root=tmp_path,
        container_name="robocasa",
        repo_root=Path("/repo"),
        container_workdir="/temporal_vla",
        target_per_cell=50,
        max_seeds_per_cell=10000,
        progress_interval=25,
        max_new_sessions=None,
    )

    assert [(plan.cell_id, plan.reason) for plan in plans] == [
        ("ppcs_onion", "complete"),
        ("ppcs_apple", "launch"),
    ]


def test_completed_cell_ids_reads_seed_only_shards_against_cell_targets(tmp_path: Path):
    module = _load_module()
    cells = [
        module.InstructionCell("ppcs_onion", "PickPlaceCounterToStove", target_episodes=2),
        module.InstructionCell("ppcs_apple", "PickPlaceCounterToStove", target_episodes=2),
    ]
    shard_dir = tmp_path / "manifests" / "shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "ppcs_onion.tsv").write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
        "0\tppcs_onion\tPickPlaceCounterToStove\tenv\t100000\tinst\tinst\tep0.json\n"
        "0\tppcs_onion\tPickPlaceCounterToStove\tenv\t100001\tinst\tinst\tep1.json\n"
    )
    (shard_dir / "ppcs_apple.tsv").write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
        "1\tppcs_apple\tPickPlaceCounterToStove\tenv\t100050\tinst\tinst\tep0.json\n"
    )

    assert module.completed_cell_ids(cells, run_root=tmp_path) == {"ppcs_onion"}


def test_completed_cell_ids_sums_disjoint_range_shards(tmp_path: Path):
    module = _load_module()
    cells = [
        module.InstructionCell("ppcc_potato", "PickPlaceCounterToCabinet", target_episodes=3),
    ]
    shard_dir = tmp_path / "manifests" / "shards"
    shard_dir.mkdir(parents=True)
    header = (
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
    )
    (shard_dir / "ppcc_potato.tsv").write_text(
        header
        + "0\tppcc_potato\tPickPlaceCounterToCabinet\tenv\t100000\tinst\tinst\tep0.json\n"
    )
    (shard_dir / "ppcc_potato_r1.tsv").write_text(
        header
        + "0\tppcc_potato\tPickPlaceCounterToCabinet\tenv\t200000\tinst\tinst\tep1.json\n"
        + "0\tppcc_potato\tPickPlaceCounterToCabinet\tenv\t200001\tinst\tinst\tep2.json\n"
    )

    assert module.completed_cell_ids(cells, run_root=tmp_path) == {"ppcc_potato"}


def test_list_tmux_sessions_fails_on_permission_denied(monkeypatch):
    module = _load_module()

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="error connecting to /tmp/tmux-1000/default (Operation not permitted)\n",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    try:
        module.list_tmux_sessions()
    except RuntimeError as exc:
        assert "Operation not permitted" in str(exc)
    else:
        raise AssertionError("tmux permission denial should not be treated as no active sessions")
