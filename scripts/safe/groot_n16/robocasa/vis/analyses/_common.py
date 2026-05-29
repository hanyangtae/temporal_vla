"""Shared defaults/helpers for seen18 analysis subcommands."""

from __future__ import annotations

from pathlib import Path

# repo root: scripts/safe/groot_n16/robocasa/vis/analyses/_common.py -> parents[6]
RUN_ROOT = (
    Path(__file__).resolve().parents[6]
    / "outputs/eval/robocasa/groot_n16"
    / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
)
DEFAULT_CACHE = RUN_ROOT / "analysis" / "feature_cache" / "pooled_all_hmean_dmean.npz"
DEFAULT_OUT = RUN_ROOT / "analysis" / "visualizations"


def add_common(p) -> None:
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
