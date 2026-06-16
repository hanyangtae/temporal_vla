from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_SET_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "steer"
    / "fit_aligned_instruction_conceptors.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_aligned_conceptor_set_fit_under_test",
        FIT_SET_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_aligned_cache(path: Path) -> None:
    rows = []
    for episode_idx, success in [(0, 1), (1, 1), (2, 0), (3, 0)]:
        for step_idx in range(3):
            rows.append(
                {
                    "episode_idx": episode_idx,
                    "scenario_seed": 100000 + episode_idx,
                    "step_idx": step_idx,
                    "success": success,
                    "dit_layer0": np.asarray([success, step_idx], dtype=np.float32),
                    "dit_layer2": np.asarray([episode_idx, step_idx + 2], dtype=np.float32),
                    "vl": np.asarray([success, episode_idx, step_idx], dtype=np.float32),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dit_layer0=np.stack([row["dit_layer0"] for row in rows]),
        dit_layer2=np.stack([row["dit_layer2"] for row in rows]),
        vl=np.stack([row["vl"] for row in rows]),
        success=np.asarray([row["success"] for row in rows], dtype=np.int64),
        cell_id=np.asarray(["open_drawer_right"] * len(rows)),
        cell_index=np.asarray([7] * len(rows), dtype=np.int64),
        task=np.asarray(["OpenDrawer"] * len(rows)),
        episode_idx=np.asarray([row["episode_idx"] for row in rows], dtype=np.int64),
        scenario_seed=np.asarray([row["scenario_seed"] for row in rows], dtype=np.int64),
        step_idx=np.asarray([row["step_idx"] for row in rows], dtype=np.int64),
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_discovers_dit_layers_in_numeric_order_then_vl(tmp_path: Path):
    module = _load_module()
    cache = tmp_path / "run" / "analysis" / "pathway_separation" / "features.npz"
    _write_aligned_cache(cache)

    assert module.discover_aligned_pathways(cache) == ["dit_layer0", "dit_layer2", "vl"]
    assert module.discover_aligned_pathways(cache, include_vl=False) == [
        "dit_layer0",
        "dit_layer2",
    ]


def test_dry_run_writes_combined_aligned_summary(tmp_path: Path):
    module = _load_module()
    cache = tmp_path / "run" / "analysis" / "pathway_separation" / "features.npz"
    _write_aligned_cache(cache)
    out_dir = tmp_path / "conceptor"

    rows = module.fit_pathway_set(
        cache_path=cache,
        pathways=None,
        include_vl=True,
        agg_mode="truncated",
        max_len=2,
        alphas=[0.1],
        overlap_band=(0.85, 0.95),
        out_dir=out_dir,
        cell_ids=["open_drawer_right"],
        min_episodes_per_class=2,
        dry_run=True,
    )

    assert [row["pathway"] for row in rows] == ["dit_layer0", "dit_layer2", "vl"]
    assert all(row["status"] == "eligible" for row in rows)
    combined = _read_tsv(out_dir / "aligned_eligibility_summary.tsv")
    assert [row["pathway"] for row in combined] == ["dit_layer0", "dit_layer2", "vl"]
    latest = _read_tsv(out_dir / "eligibility_summary.tsv")
    assert [row["pathway"] for row in latest] == ["dit_layer0", "dit_layer2", "vl"]
