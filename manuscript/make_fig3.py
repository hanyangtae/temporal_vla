#!/usr/bin/env python3
"""Figure 3 for the KAI 2026 short paper (single panel, single column width).

Granularity sweep: how the *intrinsic* alignment of discovered states and the
*downstream* succ/fail stratification utility change with the number of
discovered states k.

Left axis  : intrinsic MI margin over a clock control (bits), per-task values
             summarised by the task median.
Right axis : downstream succ/fail separability, all-cell mean null-z.

Message: the intrinsic margin keeps growing with k (finer partitions carry more
phase information), but the downstream utility peaks at k=8 — i.e. a band that
is already finer than the 3-6 phases a human would annotate.

Data sources
------------
* intrinsic  : parsed at run time from
               ``docs/paper/kai2026/ref/align_pertask.json``        (k = 6, 8, 12)
               ``docs/paper/kai2026/ref/align_pertask_k16plus.json`` (k = 16, 24, 32)
               field ``by_k[<k>].per_shard[<task>].margin_bits``.
* downstream : hard-coded constants from ``docs/steering/41`` §8.3 (see below).

Re-running this script simply overwrites the outputs (idempotent).

Run with any python that has matplotlib, e.g.
    ~/miniconda3/envs/lerobot_safe/bin/python docs/paper/kai2026/make_fig3.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "ref"
OUT_DIR = HERE / "figs"
STEM = "fig3_granularity_utility"

KS = [6, 8, 12, 16, 24, 32]

# k -> ref json holding that sweep point
REF_FILES = {
    6: "align_pertask.json",
    8: "align_pertask.json",
    12: "align_pertask.json",
    16: "align_pertask_k16plus.json",
    24: "align_pertask_k16plus.json",
    32: "align_pertask_k16plus.json",
}

# PPCC_apple is excluded from the task median at every k: its clustering is
# degenerate (margin_bits is negative for k<=12, e.g. -0.63 at k=8), so it
# reports "worse than the clock control" rather than a granularity effect.
# Same exclusion as Figure 2 (see numbers.md, 주장 2).
EXCLUDE_TASKS = {"PPCC_apple"}

# --- downstream axis: docs/steering/41 §8.3, all-cell mean null-z of the
#     succ/fail stratification at each k (measured, not re-derived here). ------
DOWNSTREAM_Z = {6: 0.69, 8: 1.05, 12: 0.75, 16: 0.61, 24: -0.30, 32: 0.28}

# Number of phases a human annotator uses for these tasks (3-6).
GT_PHASE_BAND = (3, 6)

C_INTR = "#1f4e9c"      # intrinsic margin = dark blue (matches fig2 highlight)
C_DOWN = "#c1611f"      # downstream separability = orange
C_BAND = "#e8e8e8"      # human phase-count band
C_ZERO = "#999999"      # zero reference for the right axis

FS_LABEL = 8
FS_TICK = 7
FS_ANNOT = 6.5


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": FS_TICK,
            "axes.labelsize": FS_LABEL,
            "axes.titlesize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_ANNOT,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def load_intrinsic_medians() -> dict[int, float]:
    """Task-median margin_bits per k, with PPCC_apple excluded (see above)."""
    cache: dict[str, dict] = {}
    medians: dict[int, float] = {}
    for k in KS:
        fname = REF_FILES[k]
        if fname not in cache:
            with (REF_DIR / fname).open() as fh:
                cache[fname] = json.load(fh)
        per_shard = cache[fname]["by_k"][str(k)]["per_shard"]
        vals = [
            rec["margin_bits"]
            for task, rec in per_shard.items()
            if task not in EXCLUDE_TASKS
        ]
        medians[k] = statistics.median(vals)
    return medians


def build(ax: plt.Axes, medians: dict[int, float]) -> None:
    xs = KS
    y_intr = [medians[k] for k in xs]
    y_down = [DOWNSTREAM_Z[k] for k in xs]

    # human phase-count band (drawn first, behind everything)
    ax.axvspan(GT_PHASE_BAND[0], GT_PHASE_BAND[1], color=C_BAND, zorder=0, lw=0)
    ax.text(
        (GT_PHASE_BAND[0] + GT_PHASE_BAND[1]) / 2.0,
        0.028,
        "human phase count (3-6)",
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=FS_ANNOT,
        color="#777777",
        zorder=1,
    )

    ln_intr, = ax.plot(
        xs, y_intr, "-o", color=C_INTR, lw=1.1, ms=3.2, zorder=4,
        label="intrinsic MI margin (left)",
    )
    ax.set_xlabel("number of discovered states $k$", labelpad=2.0)
    ax.set_ylabel("intrinsic margin (bits, task median)", color=C_INTR,
                  labelpad=2.0)
    ax.tick_params(axis="y", colors=C_INTR)
    ax.spines["left"].set_color(C_INTR)
    ax.set_ylim(0.0, 0.72)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlim(2.0, 34.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in xs])
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#e4e4e4", zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.axhline(0.0, color=C_ZERO, lw=0.6, ls="--", zorder=2)
    ln_down, = ax2.plot(
        xs, y_down, "--s", color=C_DOWN, lw=1.1, ms=3.2, zorder=5,
        label="downstream succ/fail $z$ (right)",
    )
    # highlight the peak of the downstream curve
    ax2.plot([8], [DOWNSTREAM_Z[8]], "s", color=C_DOWN, ms=7.0,
             markerfacecolor="none", markeredgewidth=1.1, zorder=6)
    ax2.annotate(
        "peak utility\n$k=8$",
        xy=(8, DOWNSTREAM_Z[8]),
        xytext=(11.5, 1.20),
        fontsize=FS_ANNOT,
        color=C_DOWN,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=C_DOWN, lw=0.6,
                        shrinkA=0.0, shrinkB=3.0),
        zorder=7,
    )
    ax2.set_ylabel("downstream separability (mean null-$z$)", color=C_DOWN,
                   labelpad=3.0)
    ax2.tick_params(axis="y", colors=C_DOWN)
    ax2.spines["right"].set_color(C_DOWN)
    ax2.spines["left"].set_color(C_INTR)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(-0.55, 1.45)
    ax2.set_yticks([-0.5, 0.0, 0.5, 1.0])

    ax2.legend(
        handles=[ln_intr, ln_down],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.005, -0.02),
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.25,
        fontsize=FS_ANNOT,
    )


def main() -> None:
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    medians = load_intrinsic_medians()
    for k in KS:
        print(f"k={k:2d}  intrinsic task-median margin = {medians[k]:.4f} bits"
              f"   downstream mean null-z = {DOWNSTREAM_Z[k]:+.2f}")

    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6.0 / 2.54))
    build(ax, medians)
    fig.tight_layout(pad=0.35)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        path = OUT_DIR / f"{STEM}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
