from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "steer"
    / "fit_instruction_conceptors.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_conceptor_fit_under_test",
        FIT_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cache(path: Path) -> None:
    rows = []
    for cell_id, cell_index, task, base in [
        ("open_drawer_right", 7, "OpenDrawer", 0.0),
        ("turn_on_microwave", 10, "TurnOnMicrowave", 10.0),
    ]:
        for episode_idx, success in [(0, 1), (1, 1), (2, 0), (3, 0)]:
            for step_idx in range(3):
                rows.append(
                    {
                        "cell_id": cell_id,
                        "cell_index": cell_index,
                        "task": task,
                        "episode_idx": episode_idx,
                        "scenario_seed": 100000 + episode_idx,
                        "step_idx": step_idx,
                        "success": success,
                        "dit": np.asarray(
                            [
                                base + success + step_idx,
                                base + episode_idx,
                                base + 2.0 * step_idx,
                            ],
                            dtype=np.float32,
                        ),
                        "vl": np.asarray(
                            [
                                base + success,
                                base + step_idx,
                                base + episode_idx,
                                base + 1.0,
                            ],
                            dtype=np.float32,
                        ),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dit=np.stack([row["dit"] for row in rows]),
        dit_layer0=np.stack([row["dit"] + 0.5 for row in rows]),
        vl=np.stack([row["vl"] for row in rows]),
        success=np.asarray([row["success"] for row in rows], dtype=np.int64),
        cell_id=np.asarray([row["cell_id"] for row in rows]),
        cell_index=np.asarray([row["cell_index"] for row in rows], dtype=np.int64),
        task=np.asarray([row["task"] for row in rows]),
        episode_idx=np.asarray([row["episode_idx"] for row in rows], dtype=np.int64),
        scenario_seed=np.asarray([row["scenario_seed"] for row in rows], dtype=np.int64),
        step_idx=np.asarray([row["step_idx"] for row in rows], dtype=np.int64),
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_dry_run_reports_cell_episode_balance(tmp_path: Path):
    module = _load_module()
    cache = tmp_path / "run" / "analysis" / "pathway_separation" / "features.npz"
    _write_cache(cache)
    out_dir = tmp_path / "conceptor"

    rows = module.fit_cells(
        cache_path=cache,
        pathway="dit",
        agg_mode="truncated",
        max_len=2,
        alphas=[0.1],
        overlap_band=(0.85, 0.95),
        out_dir=out_dir,
        cell_ids=["open_drawer_right"],
        min_episodes_per_class=2,
        dry_run=True,
    )

    assert rows[0]["status"] == "eligible"
    assert rows[0]["n_success_episodes"] == 2
    assert rows[0]["n_failure_episodes"] == 2
    assert rows[0]["n_success_samples"] == 4
    assert rows[0]["n_failure_samples"] == 4
    summary = _read_tsv(out_dir / "dit" / "truncated_w2" / "eligibility_summary.tsv")
    assert summary[0]["cell_id"] == "open_drawer_right"
    assert summary[0]["status"] == "eligible"
    latest = _read_tsv(out_dir / "eligibility_summary.tsv")
    assert latest[0]["pathway"] == "dit"


def test_fit_writes_cell_id_conceptor_artifacts(tmp_path: Path):
    module = _load_module()
    cache = tmp_path / "run" / "analysis" / "pathway_separation" / "features.npz"
    _write_cache(cache)
    out_dir = tmp_path / "conceptor"

    rows = module.fit_cells(
        cache_path=cache,
        pathway="vl",
        agg_mode="episode_mean",
        max_len=10,
        alphas=[0.1],
        overlap_band=(0.85, 0.95),
        out_dir=out_dir,
        cell_ids=["open_drawer_right"],
        min_episodes_per_class=2,
        dry_run=False,
    )

    cell_dir = out_dir / "vl" / "episode_mean" / "open_drawer_right"
    assert rows[0]["status"] == "fit"
    assert (cell_dir / "conceptors.npz").is_file()
    assert (cell_dir / "metadata.json").is_file()
    z = np.load(cell_dir / "conceptors.npz")
    assert z["alpha0.1_C_steer"].shape == (4, 4)
    metadata = json.loads((cell_dir / "metadata.json").read_text())
    assert metadata["group_key"] == "cell_id"
    assert metadata["cell_id"] == "open_drawer_right"
    assert metadata["pathway"] == "vl"
    assert metadata["agg_mode"] == "episode_mean"


def test_fit_accepts_aligned_layer_cache_key(tmp_path: Path):
    module = _load_module()
    cache = tmp_path / "run" / "analysis" / "pathway_separation" / "features.npz"
    _write_cache(cache)
    out_dir = tmp_path / "conceptor"

    rows = module.fit_cells(
        cache_path=cache,
        pathway="dit_layer0",
        agg_mode="truncated",
        max_len=2,
        alphas=[0.1],
        overlap_band=(0.85, 0.95),
        out_dir=out_dir,
        cell_ids=["open_drawer_right"],
        min_episodes_per_class=2,
        dry_run=True,
    )

    assert rows[0]["status"] == "eligible"
    assert rows[0]["feature_dim"] == 3
    summary = _read_tsv(out_dir / "dit_layer0" / "truncated_w2" / "eligibility_summary.tsv")
    assert summary[0]["pathway"] == "dit_layer0"


def test_fit_skips_cells_below_episode_balance(tmp_path: Path):
    module = _load_module()
    cache = tmp_path / "run" / "analysis" / "pathway_separation" / "features.npz"
    _write_cache(cache)
    out_dir = tmp_path / "conceptor"

    rows = module.fit_cells(
        cache_path=cache,
        pathway="dit",
        agg_mode="truncated",
        max_len=2,
        alphas=[0.1],
        overlap_band=(0.85, 0.95),
        out_dir=out_dir,
        cell_ids=["open_drawer_right"],
        min_episodes_per_class=3,
        dry_run=False,
    )

    assert rows[0]["status"] == "skipped"
    assert "below threshold 3" in rows[0]["reason"]
    assert not (out_dir / "dit" / "truncated_w2" / "open_drawer_right").exists()
