#!/usr/bin/env python
"""Figure: readout accuracy is robust to the number of clusters k.

Replaces the old Fig.2 (segment granularity + clustering-method comparison),
which belonged to the earlier "finer than human phases" framing.

Source: outputs/analysis/grid_phase/phase_readout/k_sweep.json
        {k: {"acc":…, "f1":…, "per":{instruction: acc}}}

Run:
  ~/miniconda3/envs/lerobot_safe/bin/python manuscript/make_fig_ksweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REL = Path("outputs/analysis/grid_phase/phase_readout/k_sweep.json")


def resolve() -> Path:
    for root in (HERE, HERE.parent, HERE.parents[1]):
        p = root / REL
        if p.is_file():
            return p
    raise FileNotFoundError(REL)


CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def main() -> None:
    src = resolve()
    blob = json.loads(src.read_text())
    ks = sorted(int(k) for k in blob)
    acc = [blob[str(k)]["acc"] for k in ks]
    f1 = [blob[str(k)]["f1"] for k in ks]
    names = sorted(blob[str(ks[0])]["per"])

    fig, ax = plt.subplots(figsize=(8.4 * CM, 5.6 * CM))
    # 사람 phase 수 대역 (3~6) 표시
    ax.axvspan(3, 6, color="#DDDDDD", alpha=0.7, zorder=0)
    ax.text(4.5, 0.06, "human\nphase count", fontsize=6, ha="center", color="#555555")

    for n in names:                       # instruction 별 궤적 (옅게)
        v = [blob[str(k)]["per"].get(n, np.nan) for k in ks]
        ax.plot(ks, v, color="#9DB8D6", lw=0.7, alpha=0.75, zorder=1)
    ax.plot(ks, acc, "-o", color="#1F4E79", lw=1.6, ms=4.5, zorder=3,
            label="accuracy (median)")
    ax.plot(ks, f1, "--s", color="#C44E52", lw=1.4, ms=4.0, zorder=3,
            label="macro-F1 (median)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("number of clusters $k$")
    ax.set_ylabel("held-out readout")
    ax.set_ylim(0, 1.02)
    ax.grid(lw=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("readout is robust once $k$ exceeds the phase count", pad=4)

    out = HERE / "figs" / "fig_k_sweep"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out}.pdf / {out}.png")
    print("k:", ks)
    print("acc:", [round(a, 3) for a in acc])
    print("f1 :", [round(a, 3) for a in f1])


if __name__ == "__main__":
    main()
