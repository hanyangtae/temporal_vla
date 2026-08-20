#!/usr/bin/env python
"""Figure: held-out phase readout accuracy (KAI-2026).

Panel (a): median accuracy per method (majority / clock / cluster / probe)
           with per-instruction points overlaid.
Panel (b): per-instruction scatter, clock (x) vs activation cluster (y),
           so "activation beats the time-only control" is visible per point.

Source (nothing hard-coded):
  outputs/analysis/grid_phase/phase_readout/readout.tsv

Run:
  ~/miniconda3/envs/lerobot_safe/bin/python manuscript/make_fig_readout.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE, HERE.parent, HERE.parents[1]]
REL = Path("outputs/analysis/grid_phase/phase_readout/readout.tsv")


def resolve() -> Path:
    for root in CANDIDATES:
        p = root / REL
        if p.is_file():
            return p
    raise FileNotFoundError(f"cannot locate {REL}")


CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

METHODS = [("major_acc", "majority\nclass", "#C4C4C4"),
           ("causal_time_acc", "time\nonly", "#A8A8A8"),
           ("action_probe_acc", "policy\naction", "#8C8C8C"),
           ("action_time_acc", "action\n+ time", "#6E6E6E"),
           ("cluster_acc", "activation\ncluster", "#4C72B0"),
           ("probe_acc", "activation\nprobe", "#1F4E79")]


def short(name: str) -> str:
    return name.replace("PPCC_", "PPCC ").replace("_", " ")


def main() -> None:
    src = resolve()
    rows = list(csv.DictReader(src.open(), delimiter="\t"))
    if not rows:
        raise SystemExit(f"empty: {src}")
    print(f"src : {src}  ({len(rows)} instructions)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.0 * CM, 5.8 * CM),
                               gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- (a) method comparison
    rng = np.random.default_rng(0)
    for i, (key, lab, col) in enumerate(METHODS):
        v = np.array([float(r[key]) for r in rows])
        ax1.bar(i, np.median(v), width=0.62, color=col, zorder=2)
        ax1.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=9,
                    facecolor="white", edgecolor="#333333", linewidth=0.6, zorder=3)
        ax1.text(i, 0.035, f"{np.median(v):.2f}", ha="center", fontsize=7,
                 color="white" if i >= 4 else "#222222")
    ax1.set_xticks(range(len(METHODS)))
    ax1.set_xticklabels([m[1] for m in METHODS])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("accuracy on unseen scenes")
    ax1.set_title("(a) phase readout on unseen scenes: internal vs external", pad=5)
    ax1.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # ---- (b) clock vs activation, per instruction
    x = np.array([float(r["action_time_acc"]) for r in rows])
    y = np.array([float(r["probe_acc"]) for r in rows])
    ax2.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", zorder=1)
    ax2.scatter(x, y, s=26, color="#4C72B0", edgecolor="black", linewidth=0.5,
                zorder=3)
    # 라벨이 몰리는 우상단을 피해 x 순서대로 위/아래 번갈아 배치
    for rank, idx in enumerate(np.argsort(x)):
        r, xi, yi = rows[idx], x[idx], y[idx]
        dx, dy = (5, 4) if rank % 2 == 0 else (5, -9)
        if xi > 0.75:                      # 오른쪽 끝은 왼쪽으로 뽑는다
            dx, dy = (-6, 6 if rank % 2 == 0 else -10)
        ax2.annotate(short(r["instruction"]), (xi, yi), fontsize=5.4,
                     textcoords="offset points", xytext=(dx, dy), color="#333333",
                     ha="right" if xi > 0.75 else "left")
    lo = min(x.min(), y.min()) - 0.06
    ax2.set_xlim(lo, 1.02)
    ax2.set_ylim(lo, 1.02)
    ax2.set_xlabel("best external baseline (action + time)")
    ax2.set_ylabel("activation probe accuracy")
    ax2.set_title("(b) same supervision, per instruction (9/10)", pad=4)
    ax2.grid(lw=0.4, alpha=0.35, zorder=0)
    ax2.set_axisbelow(True)

    out_dir = HERE / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_readout_accuracy"
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf / {stem}.png")
    print("median:", {m[0]: round(float(np.median([float(r[m[0]]) for r in rows])), 3)
                      for m in METHODS})
    print("activation > best external:", int((y > x).sum()), "/", len(rows))


if __name__ == "__main__":
    main()
