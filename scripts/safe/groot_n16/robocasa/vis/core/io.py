"""Manifest + pkl loading helpers for SAFE GR00T N1.6 rollouts."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def tensor_to_numpy(v: Any) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def load_manifest(split_root: Path, scope: str = "all") -> list[dict[str, str]]:
    """Load manifest.tsv ordered by (task_id, episode_idx). scope filter optional."""
    with (split_root / "manifest.tsv").open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if scope != "all":
        rows = [r for r in rows if r["split"] == scope]
    rows.sort(key=lambda r: (int(r["task_id"]), int(r["episode_idx"])))
    return rows


def pkl_path(split_root: Path, row: dict[str, str]) -> Path:
    return (
        split_root
        / row["split"]
        / row["task"]
        / f"task{int(row['task_id'])}--ep{int(row['episode_idx'])}--succ{int(row['success'])}.pkl"
    )


def load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)
