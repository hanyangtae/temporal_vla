#!/usr/bin/env python3
"""exp4-3: 분리도 지도 시각화 — layer×phase heatmap + COAST quota 곡선.

산출 3종:
  atlas_global.png   cell × layer 의 mean_z / var_z / quota (global 행만)
  atlas_phase.png    cell 패널마다 layer × phase heatmap (mean_z, var_z 각 1장)
  atlas_quota.png    COAST Fig.7A 대비 — x=layer, y=quota, 선=cell

사용: python atlas_heatmap.py --tsv <atlas_all.tsv> --out-dir <...>/figs
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

# 한글 라벨용 폰트: repo 동봉 ttf 를 직접 등록 (컨테이너엔 시스템 한글 폰트 없음·쓰기 불가)
_TTF = Path(__file__).resolve().parent / "assets" / "NanumGothic-Regular.ttf"
if _TTF.exists():
    font_manager.fontManager.addfont(str(_TTF))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_TTF)).get_name()
else:
    for _f in ("NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"):
        if any(_f == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
plt.rcParams["axes.unicode_minus"] = False

Z_LIM = 6.0  # 발산 컬러맵 대칭 상한 (|z|>=2 를 유의로 읽음)


def load(tsv: Path):
    rows = []
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            r["layer"] = int(r["layer"])
            for k in ("var_z", "mean_z", "quota", "var_gain", "mean_auroc",
                      "kl_z", "kl_mean_z", "kl_cov_z", "mean_frac"):
                r[k] = float(r[k]) if r.get(k) else float("nan")
            for k in ("n_rec_s", "n_rec_f"):
                r[k] = int(r[k]) if r.get(k) else 0
            rows.append(r)
    return rows


def _grid(ax, M, xt, yt, title, cmap, vmin, vmax, cbar_label, annot=True):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xt)), xt, fontsize=7)
    ax.set_yticks(range(len(yt)), yt, fontsize=7)
    ax.set_title(title, fontsize=9)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > (vmax - vmin) * 0.32 else "black")
    return im


def fig_global(rows, out: Path):
    g = [r for r in rows if r["phase"] == "__global__"]
    cells = sorted({(r["model"], r["cell"]) for r in g})
    layers = sorted({r["layer"] for r in g})
    by = {(r["model"], r["cell"], r["layer"]): r for r in g}
    panels = [
        ("mean_z", "RdBu_r", (-Z_LIM, Z_LIM), "평균분리 z (setM)"),
        ("var_z", "RdBu_r", (-Z_LIM, Z_LIM), "분산분리 z (conceptor)"),
        ("kl_z", "RdBu_r", (-Z_LIM, Z_LIM), "통합 KL z (총 분리도)"),
        ("mean_frac", "PuOr", (0.0, 1.0), "성분 비율 (1=평균형·0=분산형)"),
        ("quota", "viridis", None, "COAST quota tr(C)/D"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 0.42 * len(cells) + 2.6))
    ylab = [f"{m}/{c}".replace("pq3_", "") for m, c in cells]
    for ax, (key, cmap, rng, lab) in zip(axes, panels):
        M = np.full((len(cells), len(layers)), np.nan)
        for i, (m, c) in enumerate(cells):
            for j, l in enumerate(layers):
                r = by.get((m, c, l))
                if r:
                    M[i, j] = r[key]
        vmin, vmax = rng if rng else (np.nanmin(M), np.nanmax(M))
        im = _grid(ax, M, [f"L{l}" for l in layers], ylab, lab, cmap, vmin, vmax, lab)
        fig.colorbar(im, ax=ax, fraction=0.035)
        ax.set_xlabel("DiT layer")
    fig.suptitle("exp4-3 분리도 지도 — global (길이통제: 성공길이 mean+1sd cap)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[wrote] {out}")


def fig_phase(rows, out_prefix: Path):
    ph = [r for r in rows if r["phase"] != "__global__"]
    cells = sorted({(r["model"], r["cell"]) for r in ph})
    layers = sorted({r["layer"] for r in ph})
    for key, lab in (("mean_z", "평균분리 z"), ("var_z", "분산분리 z")):
        ncol = min(3, len(cells))
        nrow = int(np.ceil(len(cells) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.6 * nrow), squeeze=False)
        for idx, (m, c) in enumerate(cells):
            ax = axes[idx // ncol][idx % ncol]
            phs = sorted({r["phase"] for r in ph if r["model"] == m and r["cell"] == c})
            by = {(r["phase"], r["layer"]): r for r in ph
                  if r["model"] == m and r["cell"] == c}
            M = np.full((len(phs), len(layers)), np.nan)
            for i, p in enumerate(phs):
                for j, l in enumerate(layers):
                    r = by.get((p, l))
                    if r:
                        M[i, j] = r[key]
            im = _grid(ax, M, [f"L{l}" for l in layers], phs,
                       f"{m}/{c}".replace("pq3_", ""), "RdBu_r", -Z_LIM, Z_LIM, lab)
            fig.colorbar(im, ax=ax, fraction=0.04)
        for k in range(len(cells), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"exp4-3 분리도 지도 — phase × layer ({lab}, 길이통제: phase dwell cap; "
                     "빈칸=표본 미달 skip)", fontsize=11)
        fig.tight_layout()
        p = out_prefix.with_name(out_prefix.name + f"_{key}.png")
        fig.savefig(p, dpi=150)
        print(f"[wrote] {p}")


def fig_quota(rows, out: Path):
    g = [r for r in rows if r["phase"] == "__global__"]
    by = defaultdict(dict)
    for r in g:
        by[(r["model"], r["cell"])][r["layer"]] = r["quota"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (m, c), d in sorted(by.items()):
        ls = sorted(d)
        ax.plot(ls, [d[l] for l in ls], marker="o", ms=3.5,
                label=f"{m}/{c}".replace("pq3_", ""))
    ax.set_xlabel("DiT layer")
    ax.set_ylabel("quota  tr(C_steer)/D  (alpha=10, in-sample)")
    ax.set_title("COAST 참조 지표 (Fig.7A 대비): quota vs depth", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[wrote] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = load(args.tsv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_global(rows, args.out_dir / "atlas_global.png")
    fig_phase(rows, args.out_dir / "atlas_phase")
    fig_quota(rows, args.out_dir / "atlas_quota.png")


if __name__ == "__main__":
    main()
