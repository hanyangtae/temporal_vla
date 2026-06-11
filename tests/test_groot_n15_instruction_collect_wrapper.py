from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "collect"
    / "collect_instruction_fixed_http_features.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_collect_wrapper_under_test",
        WRAPPER_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_selected_manifest(path: Path) -> None:
    path.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\t"
        "canonical_instruction\tep_meta_path\n"
        "7\topen_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100001\t"
        "Open the right drawer.\tOpen the right drawer.\t/tmp/ep0.json\n"
        "7\topen_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100002\t"
        "Open the right drawer.\tOpen the right drawer.\t/tmp/ep1.json\n"
        "8\topen_drawer_left\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t100003\t"
        "Open the left drawer.\tOpen the left drawer.\t/tmp/ep2.json\n"
    )


def test_build_collect_commands_assigns_cell_local_episode_indices(tmp_path: Path):
    module = _load_module()
    manifest = tmp_path / "selected.tsv"
    output_root = tmp_path / "raw_rollouts"
    ep_meta_root = tmp_path / "ep_meta"
    _write_selected_manifest(manifest)

    rows = module.load_selected_seed_rows(manifest)
    commands = module.build_collect_commands(
        rows,
        output_dir=output_root,
        ep_meta_dir=ep_meta_root,
        vla_server="http://127.0.0.1:8400",
        n_action_steps=16,
        max_episode_steps=720,
        video_fps=20,
        steps_per_render=2,
        wait_ready=True,
        python_executable="/usr/bin/python",
    )

    assert len(commands) == 3
    assert commands[0].episode_idx == 0
    assert commands[1].episode_idx == 1
    assert commands[2].episode_idx == 0

    first = commands[0].argv
    assert first[:2] == ["/usr/bin/python", str(module.COLLECT_SCRIPT)]
    assert "--task" in first and first[first.index("--task") + 1] == "OpenDrawer"
    assert "--cell-id" in first and first[first.index("--cell-id") + 1] == "open_drawer_right"
    assert "--cell-index" in first and first[first.index("--cell-index") + 1] == "7"
    assert "--episode-start-idx" in first and first[first.index("--episode-start-idx") + 1] == "0"
    assert "--seed" in first and first[first.index("--seed") + 1] == "100001"
    assert "--inference-seed" in first and first[first.index("--inference-seed") + 1] == "100001"
    assert "--n-episodes" in first and first[first.index("--n-episodes") + 1] == "1"
    assert "--output-dir" in first and first[first.index("--output-dir") + 1] == str(output_root)
    assert "--ep-meta-dir" in first and first[first.index("--ep-meta-dir") + 1] == str(ep_meta_root)
    assert "--ep-meta-load-env-name" not in first
    assert "--env-kwargs-json" not in first
    assert "--wait-ready" in first
    assert "Open the right drawer." in first


def test_build_collect_commands_filters_cell_ids(tmp_path: Path):
    module = _load_module()
    manifest = tmp_path / "selected.tsv"
    _write_selected_manifest(manifest)

    rows = module.load_selected_seed_rows(manifest, cell_ids=["open_drawer_left"])
    commands = module.build_collect_commands(
        rows,
        output_dir=tmp_path / "raw_rollouts",
        ep_meta_dir=tmp_path / "ep_meta",
        vla_server="http://127.0.0.1:8400",
        n_action_steps=16,
        max_episode_steps=None,
        video_fps=20,
        steps_per_render=2,
        wait_ready=False,
        python_executable="/usr/bin/python",
    )

    assert len(commands) == 1
    assert commands[0].cell_id == "open_drawer_left"
    assert "--max-episode-steps" not in commands[0].argv
    assert "--wait-ready" not in commands[0].argv


def test_limit_rows_per_cell_keeps_manifest_order(tmp_path: Path):
    module = _load_module()
    manifest = tmp_path / "selected.tsv"
    _write_selected_manifest(manifest)

    rows = module.load_selected_seed_rows(manifest)
    limited = module.limit_rows_per_cell(rows, max_episodes_per_cell=1)

    assert [row.scenario_seed for row in limited] == [100001, 100003]


def test_parse_args_defaults_to_n16_aligned_episode_cap(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WRAPPER_SCRIPT),
            "--selected-seeds",
            "selected.tsv",
            "--output-dir",
            "raw_rollouts",
            "--ep-meta-dir",
            "ep_meta",
        ],
    )

    args = module.parse_args()

    assert args.max_episode_steps == 720


def test_parse_args_accepts_max_episodes_per_cell(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WRAPPER_SCRIPT),
            "--selected-seeds",
            "selected.tsv",
            "--output-dir",
            "raw_rollouts",
            "--ep-meta-dir",
            "ep_meta",
            "--max-episodes-per-cell",
            "1",
        ],
    )

    args = module.parse_args()

    assert args.max_episodes_per_cell == 1


def test_build_collect_commands_can_skip_existing_episode_pkls(tmp_path: Path):
    module = _load_module()
    manifest = tmp_path / "selected.tsv"
    output_root = tmp_path / "raw_rollouts"
    _write_selected_manifest(manifest)
    existing = output_root / "OpenDrawer" / "open_drawer_right" / "task7--ep0--succ1.pkl"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"pkl")

    rows = module.load_selected_seed_rows(manifest, cell_ids=["open_drawer_right"])
    commands = module.build_collect_commands(
        rows,
        output_dir=output_root,
        ep_meta_dir=tmp_path / "ep_meta",
        vla_server="http://127.0.0.1:8400",
        n_action_steps=16,
        max_episode_steps=720,
        video_fps=20,
        steps_per_render=2,
        wait_ready=False,
        skip_existing=True,
        python_executable="/usr/bin/python",
    )

    assert len(commands) == 1
    assert commands[0].episode_idx == 1
    assert "--seed" in commands[0].argv
    assert commands[0].argv[commands[0].argv.index("--seed") + 1] == "100002"


def test_parse_args_accepts_skip_existing(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WRAPPER_SCRIPT),
            "--selected-seeds",
            "selected.tsv",
            "--output-dir",
            "raw_rollouts",
            "--ep-meta-dir",
            "ep_meta",
            "--skip-existing",
        ],
    )

    args = module.parse_args()

    assert args.skip_existing is True



def test_build_collect_commands_can_enable_ep_meta_replay(tmp_path: Path):
    module = _load_module()
    manifest = tmp_path / "selected.tsv"
    ep_meta_root = tmp_path / "ep_meta"
    _write_selected_manifest(manifest)

    rows = module.load_selected_seed_rows(manifest, cell_ids=["open_drawer_right"])
    commands = module.build_collect_commands(
        rows[:1],
        output_dir=tmp_path / "raw_rollouts",
        ep_meta_dir=ep_meta_root,
        vla_server="http://127.0.0.1:8400",
        n_action_steps=16,
        max_episode_steps=None,
        video_fps=20,
        steps_per_render=2,
        wait_ready=False,
        replay_ep_meta=True,
        python_executable="/usr/bin/python",
    )

    argv = commands[0].argv
    assert "--ep-meta-dir" in argv
    assert argv[argv.index("--ep-meta-dir") + 1] == str(ep_meta_root)
    assert "--ep-meta-load-env-name" in argv
    assert (
        argv[argv.index("--ep-meta-load-env-name") + 1]
        == "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
    )
