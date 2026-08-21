#!/usr/bin/env python3
"""Figure 2 for the KAI 2026 short paper (2 panels, single column width).

Panel (a) segment granularity: GT phase segments vs discovered states.
Panel (b) phase information: MI margin over a clock control baseline.

All numbers are hard-coded constants taken from ``docs/paper/kai2026/numbers.md``
(the single source of truth); each constant carries the row it comes from.
Re-running this script simply overwrites the outputs (idempotent).

Run with any python that has matplotlib, e.g.
    ~/miniconda3/envs/lerobot_safe/bin/python docs/paper/kai2026/make_fig2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figs"
STEM = "fig2_granularity_margin"

# --- numbers.md > "주장 1 — 촘촘함" ------------------------------------------
# | 구간 길이 (pq3)            | GT 16.06 step vs 발견 4.3 step | 40 §5 부수 관찰 |
# | 구간 길이 (930판, k24 global) | GT 18.12 vs 6.00 step       | ref align_global.json |
SEG_LEN = {
    "A": {"gt": 16.06, "disc": 4.30},   # Dataset A = colleague pq3, 23 episodes
    "B": {"gt": 18.12, "disc": 6.00},   # Dataset B = grid 930 episodes, k=24 global
}

# --- numbers.md > "주장 2 — phase-순수성" ------------------------------------
# | margin (pq3, KMeans24)  | +1.665 bits (clock MI 0.51)                     | 40 §3 |
# | 자율 k 방식 열세         | HDBSCAN 최선 0.945, dendrogram gap 1.186 < 1.665 | 40 §3 |
MARGIN_A = [
    ("clock control", 0.000),
    ("HDBSCAN (best)", 0.945),
    ("dendrogram gap", 1.186),
    ("KMeans k=24", 1.665),
]

# --- numbers.md > "930판 per-task k8": margin +0.15~+0.51 (apple −0.63 퇴화 제외)
# raw values: docs/paper/kai2026/ref/align_pertask.json -> by_k["8"].per_shard[*].margin_bits
MARGIN_B_PERTASK = {
    "CoffeeSetupMug": 0.4175,
    "DishwasherRack_out": 0.2858,
    "OpenDrawer_left": 0.3947,
    "OpenDrawer_right": 0.3675,
    "OvenRack_out": 0.2507,
    "PPCC_bread": 0.1482,
    "PPCC_candle": 0.5137,
    "PPCC_jug": 0.2130,
    "PPCC_marshmallow": 0.3370,
    # PPCC_apple = -0.6307 excluded (degenerate clustering; see numbers.md 주장 2)
}

C_GT = "#8c8c8c"        # ground-truth phase = grey
C_DISC = "#3b6fb6"      # discovered states = blue
C_BASE = "#b8b8b8"      # clock control baseline
C_ALT = "#8fb3dc"       # non-highlighted discovery methods
C_HI = "#1f4e9c"        # highlighted (KMeans k=24)

FS_LABEL = 8
FS_TICK = 7
FS_TITLE = 8
FS_ANNOT = 6.5


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": FS_TICK,
            "axes.labelsize": FS_LABEL,
            "axes.titlesize": FS_TITLE,
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


def panel_a(ax: plt.Axes) -> None:
    groups = [("A\n23 ep", "A"), ("B\n930 ep", "B")]
    width = 0.34
    for i, (_, key) in enumerate(groups):
        gt = SEG_LEN[key]["gt"]
        disc = SEG_LEN[key]["disc"]
        b1 = ax.bar(i - width / 2, gt, width, color=C_GT,
                    label="GT phase" if i == 0 else None, zorder=3)
        b2 = ax.bar(i + width / 2, disc, width, color=C_DISC,
                    label="discovered" if i == 0 else None, zorder=3)
        for b, v in ((b1[0], gt), (b2[0], disc)):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=FS_ANNOT)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=FS_ANNOT)
    ax.set_xlabel("Dataset", fontsize=7.5, labelpad=1.5)
    ax.set_ylabel("segment length (steps)", fontsize=7.5, labelpad=1.5)
    ax.set_ylim(0, 30.0)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_title("(a) Segment granularity", pad=3)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=0.9,
              handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25,
              fontsize=FS_ANNOT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)


def panel_b(ax: plt.Axes) -> None:
    """Horizontal bars: long method names stay readable at column width."""
    # bottom-to-top: Dataset B scatter row, then Dataset A methods (best on top)
    xs = list(MARGIN_B_PERTASK.values())
    y_b = 0.0
    ax.scatter(xs, [y_b] * len(xs), s=8, facecolors="none",
               edgecolors=C_HI, linewidths=0.7, zorder=4)
    ax.vlines(sum(xs) / len(xs), y_b - 0.28, y_b + 0.28,
              color=C_HI, lw=1.0, zorder=5)

    labels = ["Dataset B, per-task k=8"]
    colors = [C_BASE, C_ALT, C_ALT, C_HI]
    for i, ((lbl, val), col) in enumerate(zip(MARGIN_A, colors)):
        y = i + 1.0
        ax.barh(y, val, 0.6, color=col, zorder=3)
        ax.text(val + 0.05, y, f"{val:.3f}", ha="left", va="center",
                fontsize=FS_ANNOT)
        labels.append(lbl)

    ax.axvline(0, color="#555555", lw=0.6, zorder=2)
    ax.axhline(0.5, color="#cccccc", lw=0.6, ls=":", zorder=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=FS_ANNOT)
    ax.set_ylim(-0.65, len(MARGIN_A) + 0.65)
    ax.set_xlim(-0.05, 2.05)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("MI margin over clock (bits)", fontsize=7.5, labelpad=1.5)
    ax.set_title("(b) Phase information", pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # single-column width ~8.5 cm, height ~5 cm
    fig, axes = plt.subplots(1, 2, figsize=(8.5 / 2.54, 5.0 / 2.54),
                             gridspec_kw={"width_ratios": [1.0, 1.55]})
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout(pad=0.3, w_pad=0.8)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        path = OUT_DIR / f"{STEM}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
