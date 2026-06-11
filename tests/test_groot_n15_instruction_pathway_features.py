from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "analyze"
    / "instruction_pathway_features.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_pathway_features_under_test",
        FEATURE_SCRIPT,
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


def test_load_instruction_feature_examples_pools_n15_dit_and_vl(tmp_path: Path):
    module = _load_module()
    root = tmp_path / "raw_rollouts"
    pkl_path = root / "OpenDrawer" / "open_drawer_right" / "task7--ep0--succ1.pkl"
    _write_pkl(
        pkl_path,
        {
            "task_id": 7,
            "cell_id": "open_drawer_right",
            "cell_index": 7,
            "robocasa_task": "OpenDrawer",
            "episode_idx": 0,
            "scenario_seed": 100001,
            "episode_success": 1,
            "hidden_states": [
                np.ones((4, 16, 1024), dtype=np.float16) * 2,
                np.ones((4, 16, 1024), dtype=np.float16) * 4,
            ],
            "vl_hidden_states": [
                np.ones((2048,), dtype=np.float16) * 3,
                np.ones((2048,), dtype=np.float16) * 5,
            ],
        },
    )

    examples = module.load_instruction_feature_examples(
        root,
        horizon_idx_rel="mean",
        diff_idx_rel="mean",
        require_vl=True,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.task == "OpenDrawer"
    assert example.cell_id == "open_drawer_right"
    assert example.cell_index == 7
    assert example.success == 1
    assert example.dit_features.shape == (2, 1024)
    assert example.dit_layer_features is None
    assert example.dit_capture_layers is None
    assert example.vl_features.shape == (2, 2048)
    np.testing.assert_allclose(example.dit_features[0], 2.0)
    np.testing.assert_allclose(example.dit_features[1], 4.0)
    np.testing.assert_allclose(example.vl_features[0], 3.0)
    np.testing.assert_allclose(example.vl_features[1], 5.0)


def test_flatten_examples_repeats_episode_metadata_per_step(tmp_path: Path):
    module = _load_module()
    example = module.InstructionFeatureExample(
        path=tmp_path / "x.pkl",
        task="OpenDrawer",
        cell_id="open_drawer_right",
        cell_index=7,
        episode_idx=2,
        scenario_seed=100123,
        success=0,
        dit_features=np.zeros((3, 4), dtype=np.float32),
        dit_layer_features=None,
        dit_capture_layers=None,
        vl_features=np.ones((3, 5), dtype=np.float32),
    )

    flattened = module.flatten_examples([example])

    assert flattened["dit"].shape == (3, 4)
    assert flattened["vl"].shape == (3, 5)
    assert flattened["success"].tolist() == [0, 0, 0]
    assert flattened["step_idx"].tolist() == [0, 1, 2]
    assert flattened["episode_idx"].tolist() == [2, 2, 2]
    assert flattened["scenario_seed"].tolist() == [100123, 100123, 100123]
    assert flattened["cell_id"].tolist() == [
        "open_drawer_right",
        "open_drawer_right",
        "open_drawer_right",
    ]


def test_preserve_dit_layers_exports_layer_arrays(tmp_path: Path):
    module = _load_module()
    root = tmp_path / "raw_rollouts"
    pkl_path = root / "OpenDrawer" / "open_drawer_right" / "task7--ep0--succ1.pkl"
    step0 = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    step1 = step0 + 100
    _write_pkl(
        pkl_path,
        {
            "task_id": 7,
            "cell_id": "open_drawer_right",
            "cell_index": 7,
            "robocasa_task": "OpenDrawer",
            "episode_idx": 0,
            "scenario_seed": 100001,
            "episode_success": 1,
            "feature_axes": ["layer", "model_token", "feature_dim"],
            "capture_layers": [0, 2],
            "hidden_states": [step0, step1],
            "vl_hidden_states": [
                np.ones((8,), dtype=np.float16),
                np.ones((8,), dtype=np.float16) * 2,
            ],
        },
    )

    examples = module.load_instruction_feature_examples(
        root,
        horizon_idx_rel="mean",
        diff_idx_rel="mean",
        require_vl=True,
        preserve_dit_layers=True,
    )
    flattened = module.flatten_examples(examples)

    assert examples[0].dit_features.shape == (2, 4)
    assert examples[0].dit_layer_features.shape == (2, 2, 4)
    assert examples[0].dit_capture_layers == (0, 2)
    assert flattened["dit"].shape == (2, 4)
    assert flattened["dit_layers"].shape == (2, 2, 4)
    assert flattened["dit_capture_layers"].tolist() == [0, 2]
    assert flattened["dit_layer0"].shape == (2, 4)
    assert flattened["dit_layer2"].shape == (2, 4)
    np.testing.assert_allclose(flattened["dit_layer0"][0], step0[0].astype(np.float32).mean(axis=0))
    np.testing.assert_allclose(flattened["dit_layer2"][1], step1[1].astype(np.float32).mean(axis=0))
