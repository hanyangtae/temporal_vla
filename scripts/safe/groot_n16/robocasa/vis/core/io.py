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


# ---- seen18 pooled-feature cache (npz from cache_pooled_features.py) ----

def load_feature_cache(path: Path):
    """np.load the pooled-feature npz (allow_pickle for object task_names)."""
    return np.load(path, allow_pickle=True)


def reconstruct_rollouts(cache) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Canonical per-rollout reconstruction from the pooled-feature cache.

    Accepts a path or an already-loaded npz. Returns:
      rollouts : list of {"z": ndarray[T, D] (step-sorted), "succ": int, "task": int}
      names    : {task_id: task_name}
    Replaces the per-script reconstruct() copies; output `r["z"]` is compatible
    with core.aggregation.aggregate().
    """
    if not hasattr(cache, "files"):
        cache = load_feature_cache(cache)
    feats = cache["feats"].astype(np.float64)
    step_idx, rollout_idx = cache["step_idx"], cache["rollout_idx"]
    ep_success, ep_task = cache["ep_success"], cache["ep_task_id"]
    names = {int(t): str(n) for t, n in zip(cache["ep_task_id"], cache["task_names"])}
    order = np.lexsort((step_idx, rollout_idx))
    rid = rollout_idx[order]
    uniq = np.unique(rollout_idx)
    starts = np.searchsorted(rid, uniq, side="left")
    ends = np.searchsorted(rid, uniq, side="right")
    rollouts = [
        {"z": feats[order[a:b]], "succ": int(ep_success[int(r)]), "task": int(ep_task[int(r)])}
        for r, a, b in zip(uniq, starts, ends)
    ]
    return rollouts, names
