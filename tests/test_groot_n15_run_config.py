from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_PATH = REPO_ROOT / "scripts/safe/groot_n15/robocasa/run_config.py"


def _load_run_config():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_run_config_under_test",
        RUN_CONFIG_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_paths_stay_under_groot_n15_output_root():
    with patch.dict(os.environ, {}, clear=True):
        run_config = _load_run_config()

    assert run_config.OUT_ROOT == REPO_ROOT / "outputs/eval/robocasa/groot_n15"
    assert (
        run_config.BASE_NEW_EMBODIMENT_CHECKPOINT
        == REPO_ROOT / "outputs/checkpoints/groot_n15_base_pandaomron_new_embodiment"
    )
    assert (
        run_config.SEEN5_SOURCE_ROOT
        == run_config.OUT_ROOT / "rollouts_seen5_100ep_per_task_subset100"
    )
    assert (
        run_config.SEEN5_SPLIT_ROOT
        == run_config.OUT_ROOT / "split_seen5_trainval75_cp15_test10_subset100"
    )


def test_env_overrides_drive_shared_paths():
    with patch.dict(
        os.environ,
        {
            "GROOT_N15_OUT_ROOT": "/tmp/groot_n15_outputs",
            "GROOT_N15_TARGET15_RUN_ID": "probe",
            "GROOT_N15_SEEN5_SOURCE_ROOT": "/tmp/source",
            "GROOT_N15_SEEN5_SPLIT_ROOT": "/tmp/split",
        },
        clear=True,
    ):
        run_config = _load_run_config()

    assert run_config.OUT_ROOT == Path("/tmp/groot_n15_outputs")
    assert run_config.TARGET15_RUN_ROOT == Path("/tmp/groot_n15_outputs/target15_seedpairs_probe")
    assert run_config.SEEN5_SOURCE_ROOT == Path("/tmp/source")
    assert run_config.SEEN5_SPLIT_ROOT == Path("/tmp/split")
