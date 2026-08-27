#!/usr/bin/env python
"""Figure: unseen-scene phase readout accuracy — bar only (2-page version).

Panel (a): median accuracy per method with per-instruction points overlaid.
           Methods: majority / time (clock) / [Event-SAE, if results exist]
                    / activation cluster.
           (policy-action baselines were dropped from the paper; the external
           observer role is played by the published Event-SAE pipeline.)
Panel (b): per-instruction scatter — external baseline (x) vs activation
           cluster (y), same protocol (unsupervised discovery + train-scene
           majority mapping). x = Event-SAE when available, else clock.

Source (nothing hard-coded):
  outputs/analysis/grid_phase/phase_readout/readout.tsv
  outputs/analysis/grid_phase/phase_readout/esae.json   (optional)

Run:
  ~/miniconda3/envs/lerobot_safe/bin/python manuscript/make_fig_readout.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE, HERE.parent, HERE.parents[1]]
REL = Path("outputs/analysis/grid_phase/phase_readout/readout.tsv")
REL_ESAE = Path("outputs/analysis/grid_phase/phase_readout/esae.json")


def resolve(rel: Path, required: bool = True) -> Path | None:
    for root in CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    if required:
        raise FileNotFoundError(f"cannot locate {rel}")
    return None


CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def short(name: str) -> str:
    return name.replace("PPCC_", "PPCC ").replace("_", " ")


def main() -> None:
    src = resolve(REL)
    rows = list(csv.DictReader(src.open(), delimiter="\t"))
    if not rows:
        raise SystemExit(f"empty: {src}")
    print(f"src : {src}  ({len(rows)} instructions)")

    esae_path = resolve(REL_ESAE, required=False)
    esae = {}
    if esae_path:
        blob = json.loads(esae_path.read_text())
        esae = {k: v["esae_acc"] for k, v in blob["per_instruction"].items()}
        print(f"esae: {esae_path}  ({len(esae)} instructions)")

    # (키, 라벨, 색, 값 배열) — behavioral event 는 결과 파일이 있을 때만 들어간다
    methods = [("causal_time_acc", "time\n(clock)", "#A8A8A8",
                np.array([float(r["causal_time_acc"]) for r in rows]))]
    if esae:
        methods.append(("esae_acc", "behavioral\nevent", "#6E6E6E",
                        np.array([esae.get(r["instruction"], np.nan)
                                  for r in rows])))
    methods += [("cluster_acc", "activation\ncluster", "#4C72B0",
                 np.array([float(r["cluster_acc"]) for r in rows]))]

    fig, ax1 = plt.subplots(figsize=(8.0 * CM, 5.4 * CM))

    # ---- (a) method comparison
    rng = np.random.default_rng(0)
    for i, (key, lab, col, v) in enumerate(methods):
        vv = v[~np.isnan(v)]
        ax1.bar(i, np.median(vv), width=0.62, color=col, zorder=2)
        ax1.scatter(i + rng.uniform(-0.16, 0.16, len(vv)), vv, s=9,
                    facecolor="white", edgecolor="#333333", linewidth=0.6, zorder=3)
        ax1.text(i, 0.035, f"{np.median(vv):.2f}", ha="center", fontsize=7,
                 color="white" if key.startswith(("cluster", "probe")) else "#222222")
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m[1] for m in methods])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("accuracy on unseen scenes")
    ax1.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    out_dir = HERE / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_readout_bar"
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf / {stem}.png")
    print("median:", {m[0]: round(float(np.nanmedian(m[3])), 3) for m in methods})


if __name__ == "__main__":
    main()
