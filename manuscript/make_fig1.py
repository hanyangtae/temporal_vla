#!/usr/bin/env python
"""Figure F1 for the KAI-2026 domestic paper.

Panel (a): cluster x GT-phase contingency (row-normalised occupancy) for
           PPCC_candle at k=8, rows re-ordered so the block-diagonal shows.
Panel (b): paired raw -> residualised change of `margin` (survives) and
           `mi_scene` (collapses) over the 10 grid tasks.

All numbers are read from the canonical artefacts; nothing is hard-coded:
  outputs/analysis/grid_phase/paper_supp/contingency_pertask_k8.json
  outputs/analysis/grid_phase/paper_supp/resid_compare.tsv

Usage (idempotent, overwrites the outputs):
  python docs/paper/kai2026/make_fig1.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# The analysis artefacts live under the directory that hosts this paper tree.
CANDIDATE_ROOTS = [HERE, HERE / "template", HERE.parent, HERE.parents[2]]
REL_JSON = Path("outputs/analysis/grid_phase/ae_raw/contingency_pertask_k8_ae.json")
REL_TSV = Path("outputs/analysis/grid_phase/ae_raw/resid_compare_ae.tsv")


def _resolve(rel: Path) -> Path:
    for root in CANDIDATE_ROOTS:
        cand = root / rel
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"cannot locate {rel} under {CANDIDATE_ROOTS}")


JSON_PATH = _resolve(REL_JSON)
TSV_PATH = _resolve(REL_TSV)
OUT_DIR = HERE / "figs"
OUT_STEM = OUT_DIR / "fig1_purity_residual"

FOCUS_TASK = "PPCC_candle"
DISPLAY_TASK = "Pick and Place"   # 그림 표기용 이름 (내부 슬러그와 분리)
DEGENERATE_TASK = None  # raw+AE 에서는 전 instruction margin 양수 → 특례 없음

CM = 1.0 / 2.54
FIG_W = 17.5 * CM
FIG_H = 6.0 * CM

C_RAW = "#7f8c8d"      # raw endpoint (grey)
C_MARGIN = "#1f6fb4"   # margin, residualised endpoint (kept -> emphasised)
C_SCENE = "#c0504d"    # mi_scene, residualised endpoint (collapses)
C_DEGEN = "#b07aa1"    # PPCC_apple marker colour

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_contingency(task: str):
    with JSON_PATH.open() as fh:
        blob = json.load(fh)
    entry = blob["per_task"][task]
    counts = np.asarray(entry["counts"], dtype=float)
    row_sum = counts.sum(axis=1, keepdims=True)
    rownorm = np.asarray(entry["occupancy_rownorm"], dtype=float)
    if rownorm.shape != counts.shape:  # defensive: recompute
        rownorm = counts / np.maximum(row_sum, 1.0)
    return {
        "k": int(blob["k"]),
        "cluster_ids": list(entry["cluster_ids"]),
        "phase_names": list(entry["phase_names"]),
        "n_clusters": int(entry["n_clusters"]),
        "n_phases": int(entry["n_phases"]),
        "n_rec": int(entry["n_rec"]),
        "rownorm": rownorm,
        "row_n": row_sum.ravel(),
        "purity": float(entry["purity_phase"]),
    }


def load_resid():
    rows = []
    with TSV_PATH.open() as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            rows.append(
                {
                    "task": rec["task"],
                    "raw_margin": float(rec["raw_margin"]),
                    "resid_margin": float(rec["resid_margin"]),
                    "raw_mi_scene": float(rec["raw_mi_scene"]),
                    "resid_mi_scene": float(rec["resid_mi_scene"]),
                }
            )
    return rows


def short_phase(name: str) -> str:
    """Compact GT-phase label for the x axis."""
    repl = {
        "reach-to-object": "reach",
        "insert-settle": "insert",
        "transport": "transp.",
        "grasp-lift": "grasp",
        "align-insert": "align",
    }
    if name in repl:
        return repl[name]
    if len(name) > 8 and "-" in name:
        return name.partition("-")[0]
    return name[:8]


def short_task(name: str) -> str:
    return name.replace("PPCC_", "PPCC ").replace("_", " ")


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def panel_a(ax, cax, cont):
    rownorm = cont["rownorm"]
    top_phase = rownorm.argmax(axis=1)
    top_share = rownorm.max(axis=1)
    # block-diagonal ordering: group by dominant phase, then by purity
    order = sorted(
        range(rownorm.shape[0]),
        key=lambda i: (top_phase[i], -top_share[i]),
    )
    mat = rownorm[order]
    ylabels = [str(cont["cluster_ids"][i]) for i in order]

    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(cont["n_phases"]))
    ax.set_xticklabels([short_phase(p) for p in cont["phase_names"]], fontsize=7)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("annotated phase", labelpad=2)
    ax.set_ylabel("cluster (re-ordered)", labelpad=2)
    ax.set_title(
        f"(a) Cluster-phase contingency\n{DISPLAY_TASK}, k={cont['k']}",
        fontsize=8,
        pad=4,
    )
    # cell annotations, contrast-aware so text never disappears into the cell
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if v < 0.005:
                continue
            ax.text(
                c,
                r,
                f"{v:.2f}".lstrip("0"),
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if v > 0.55 else "#333333",
            )
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = plt.colorbar(im, cax=cax)
    # label placed *under* the bar: a rotated y-label would run into the
    # task names of panel (b)
    cb.ax.set_xlabel("row\nshare", fontsize=6, labelpad=11, linespacing=1.1)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([0, 0.5, 1.0])


def _dumbbell(ax, rows, key_raw, key_res, color_res, title, xlabel, zero_line):
    ypos = np.arange(len(rows))[::-1]
    for y, rec in zip(ypos, rows):
        a, b = rec[key_raw], rec[key_res]
        degen = rec["task"] == DEGENERATE_TASK
        ax.plot([a, b], [y, y], color="#bdbdbd", lw=0.9, zorder=1,
                solid_capstyle="round")
        ax.scatter([a], [y], s=13, facecolor="white", edgecolor=C_RAW,
                   linewidth=0.9, zorder=3)
        ax.scatter([b], [y], s=15, facecolor=C_DEGEN if degen else color_res,
                   edgecolor="none", zorder=4, marker="D" if degen else "o")
    if zero_line:
        ax.axvline(0.0, color="#444444", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(ypos)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.margins(x=0.08)
    ax.grid(axis="x", color="#e8e8e8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ypos


def panel_b(ax_m, ax_s, rows):
    rows = sorted(rows, key=lambda r: r["raw_margin"])
    ypos = _dumbbell(
        ax_m, rows, "raw_margin", "resid_margin", C_MARGIN,
        "(b) margin: survives", "phase margin (bits)", zero_line=True,
    )
    ax_m.set_yticklabels([short_task(r["task"]) for r in rows], fontsize=7)
    _dumbbell(
        ax_s, rows, "raw_mi_scene", "resid_mi_scene", C_SCENE,
        "(c) scene MI: collapses", "MI with scene id (bits)", zero_line=False,
    )
    ax_s.set_yticklabels([])
    ax_s.set_xlim(left=-0.02)

    n_pos = sum(
        1 for r in rows if r["task"] != DEGENERATE_TASK and r["resid_margin"] > 0
    )
    n_tot = sum(1 for r in rows if r["task"] != DEGENERATE_TASK)
    ax_m.text(
        0.98,
        0.97,
        f"{n_pos}/{n_tot} stay positive",
        transform=ax_m.transAxes,
        fontsize=6.5,
        color=C_MARGIN,
        ha="right",
        va="top",
    )
    return ypos


def build_legend(fig, rows):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor="white",
               markeredgecolor=C_RAW, markeredgewidth=0.9, markersize=3.6,
               label="raw"),
        Line2D([], [], marker="o", ls="none", color=C_MARGIN, markersize=3.8,
               label="residualised (margin)"),
        Line2D([], [], marker="o", ls="none", color=C_SCENE, markersize=3.8,
               label="residualised (scene MI)"),
    ]
    if DEGENERATE_TASK is not None:
        handles.append(Line2D([], [], marker="D", ls="none", color=C_DEGEN,
                              markersize=3.6,
                              label=f"{short_task(DEGENERATE_TASK)} (margin < 0)"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4 if DEGENERATE_TASK is not None else 3,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.55, -0.012),
        handletextpad=0.4,
        columnspacing=1.4,
    )


def main():
    cont = load_contingency(FOCUS_TASK)
    rows = load_resid()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # explicit rectangles: the long task names in (b) need a wide gutter that a
    # uniform GridSpec wspace cannot give without squashing (a).
    bottom, height = 0.30, 0.545
    ax_a = fig.add_axes([0.070, bottom, 0.185, height])
    ax_m = fig.add_axes([0.445, bottom, 0.255, height])
    ax_s = fig.add_axes([0.745, bottom, 0.235, height])
    divider = make_axes_locatable(ax_a)
    cax = divider.append_axes("right", size="5%", pad=0.09)

    panel_a(ax_a, cax, cont)
    panel_b(ax_m, ax_s, rows)
    build_legend(fig, rows)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT_STEM.with_suffix(f".{ext}"), **kw)
    plt.close(fig)

    print(f"json: {JSON_PATH}")
    print(f"tsv : {TSV_PATH}")
    print(
        f"{FOCUS_TASK}: k={cont['k']} clusters={cont['n_clusters']} "
        f"phases={cont['n_phases']} ({', '.join(cont['phase_names'])}) "
        f"purity={cont['purity']:.6f} n_rec={cont['n_rec']}"
    )
    print(f"wrote {OUT_STEM.with_suffix('.pdf')} / {OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
