#!/usr/bin/env python3
"""wrist cam attention 시계열 시각화 (cam_attn_records.csv → PNG).

1순위 산출물 = (a) per-episode 시계열: x=env step(5-step 해상도), 뷰별 attention
mass 라인 + wrist share of vision(uniform 1/3 점선) + phase 배경 band.
보조 = (b) cell 별 전 에피소드 wrist-share overlay (succ/fail 색), (c) phase 별
뷰 비중 bar (succ/fail 분리).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE_COLORS = {
    "reach": "#3aa0ff",
    "reach-to-object": "#3aa0ff",
    "reach-to-handle": "#3aa0ff",
    "grasp": "#9467bd",
    "grasp-handle": "#9467bd",
    "transport": "#ffb020",
    "place": "#e45756",
    "pull": "#e45756",
    "insert-settle": "#38c172",
    "release-settle": "#8ce39a",
    "terminal": "#999999",
}
_FALLBACK = ["#17becf", "#bcbd22", "#8c564b", "#e377c2"]
VIEW_STYLE = {
    "text": ("#bbbbbb", 1.0),
    "left": ("#3aa0ff", 1.2),
    "right": ("#ffb020", 1.2),
    "wrist": ("#e45756", 2.2),
}


def phase_color(phase: str, extra: dict) -> str:
    if phase in PHASE_COLORS:
        return PHASE_COLORS[phase]
    if phase not in extra:
        extra[phase] = _FALLBACK[len(extra) % len(_FALLBACK)]
    return extra[phase]


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("episode_idx", "success", "record_idx", "env_step"):
            row[key] = int(row[key])
        for key in ("mass_text", "mass_left", "mass_right", "mass_wrist", "wrist_share_vision"):
            row[key] = float(row[key])
    return rows


def draw_phase_bands(ax, steps, phases, nas: int, extra: dict) -> None:
    for step, phase in zip(steps, phases):
        ax.axvspan(step, step + nas, color=phase_color(phase, extra), alpha=0.14, lw=0)


def plot_episode(cell: str, ep_rows: list[dict], out_dir: Path) -> None:
    ep_rows = sorted(ep_rows, key=lambda row: row["record_idx"])
    steps = [row["env_step"] for row in ep_rows]
    phases = [row["phase"] for row in ep_rows]
    nas = steps[1] - steps[0] if len(steps) > 1 else 5
    succ = ep_rows[0]["success"]
    ep = ep_rows[0]["episode_idx"]
    extra: dict = {}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])
    for view in ("text", "left", "right", "wrist"):
        color, lw = VIEW_STYLE[view]
        ax1.plot(steps, [row[f"mass_{view}"] for row in ep_rows], color=color, lw=lw, label=view)
    draw_phase_bands(ax1, steps, phases, nas, extra)
    ax1.set_ylabel("attention mass fraction")
    ax1.set_title(
        f"{cell} ep{ep} succ={succ} — DiT cross-attn view mass "
        f"(action query, block/denoise mean, {nas}-step resolution)"
    )
    ax1.legend(loc="upper right", fontsize=8, ncol=4)

    ax2.plot(steps, [row["wrist_share_vision"] for row in ep_rows], color="#e45756", lw=2)
    ax2.axhline(1 / 3, color="k", ls="--", lw=0.8, label="uniform 1/3")
    draw_phase_bands(ax2, steps, phases, nas, extra)
    ax2.set_ylabel("wrist share of vision")
    ax2.set_xlabel("env step")
    ax2.legend(loc="upper right", fontsize=8)
    # phase 범례 (band 색)
    seen = []
    for phase in phases:
        if phase not in seen:
            seen.append(phase)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=phase_color(p, extra), alpha=0.4) for p in seen
    ]
    ax1.legend(
        handles + ax1.get_lines(),
        seen + [line.get_label() for line in ax1.get_lines()],
        loc="upper right", fontsize=7, ncol=2,
    )
    fig.tight_layout()
    out = out_dir / f"{cell}--ep{ep}--succ{succ}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_overlay(cell: str, cell_rows: list[dict], out_dir: Path) -> None:
    by_ep: dict[int, list[dict]] = defaultdict(list)
    for row in cell_rows:
        by_ep[row["episode_idx"]].append(row)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    curves = {0: [], 1: []}
    for ep, rows in sorted(by_ep.items()):
        rows = sorted(rows, key=lambda row: row["record_idx"])
        steps = [row["env_step"] for row in rows]
        vals = [row["wrist_share_vision"] for row in rows]
        succ = rows[0]["success"]
        ax.plot(steps, vals, color=("#2c7fb8" if succ else "#e45756"), alpha=0.3, lw=0.9)
        curves[succ].append((steps, vals))
    for succ, color, name in ((1, "#2c7fb8", "succ"), (0, "#e45756", "fail")):
        if not curves[succ]:
            continue
        max_len = max(len(s) for s, _ in curves[succ])
        grid = np.full((len(curves[succ]), max_len), np.nan)
        for i, (steps, vals) in enumerate(curves[succ]):
            grid[i, : len(vals)] = vals
        xs = np.arange(max_len) * (curves[succ][0][0][1] - curves[succ][0][0][0] if len(curves[succ][0][0]) > 1 else 5)
        ax.plot(xs, np.nanmean(grid, axis=0), color=color, lw=2.6, label=f"{name} mean (n={len(curves[succ])})")
    ax.axhline(1 / 3, color="k", ls="--", lw=0.8, label="uniform 1/3")
    ax.set_xlabel("env step")
    ax.set_ylabel("wrist share of vision")
    ax.set_title(f"{cell} — wrist attention share, all episodes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{cell}--overlay_wrist_share.png", dpi=110)
    plt.close(fig)


def plot_phase_bar(cell: str, cell_rows: list[dict], out_dir: Path) -> None:
    keys: list[tuple[str, int]] = []
    for row in cell_rows:
        key = (row["phase"], row["success"])
        if key not in keys:
            keys.append(key)
    # phase 등장 순서 유지, succ 먼저
    keys.sort(key=lambda k: ([row["phase"] for row in cell_rows].index(k[0]), -k[1]))
    views = ("text", "left", "right", "wrist")
    means = {
        key: [
            float(np.mean([row[f"mass_{view}"] for row in cell_rows if (row["phase"], row["success"]) == key]))
            for view in views
        ]
        for key in keys
    }
    x = np.arange(len(keys))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(keys)), 4.5))
    for i, view in enumerate(views):
        color, _ = VIEW_STYLE[view]
        ax.bar(x + (i - 1.5) * width, [means[key][i] for key in keys], width, label=view, color=color)
    ax.axhline(0.25, color="k", ls=":", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{'succ' if s else 'fail'}" for p, s in keys], fontsize=8)
    ax.set_ylabel("mean attention mass fraction")
    ax.set_title(f"{cell} — per-phase view attention mass (action query)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{cell}--phase_bar.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-csv",
        default="outputs/eval/robocasa/groot_n15/cam_attn/analysis/cam_attn_records.csv",
    )
    parser.add_argument("--out-dir", default="outputs/eval/robocasa/groot_n15/cam_attn/vis")
    parser.add_argument("--max-episode-plots", type=int, default=8, help="cell 당 per-episode 그림 수 (succ/fail 반반)")
    args = parser.parse_args()

    rows = load_rows(Path(args.records_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)
    for cell, cell_rows in sorted(by_cell.items()):
        by_ep: dict[int, list[dict]] = defaultdict(list)
        for row in cell_rows:
            by_ep[row["episode_idx"]].append(row)
        succ_eps = sorted(ep for ep, r in by_ep.items() if r[0]["success"])
        fail_eps = sorted(ep for ep, r in by_ep.items() if not r[0]["success"])
        half = args.max_episode_plots // 2
        chosen = succ_eps[:half] + fail_eps[:half]
        for ep in chosen:
            plot_episode(cell, by_ep[ep], out_dir)
        plot_overlay(cell, cell_rows, out_dir)
        plot_phase_bar(cell, cell_rows, out_dir)
        print(f"[cam_attn_vis] {cell}: {len(chosen)} episode plots + overlay + phase_bar")
    print(f"[cam_attn_vis] out: {out_dir}")


if __name__ == "__main__":
    main()
