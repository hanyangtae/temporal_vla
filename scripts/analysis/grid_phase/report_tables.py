#!/usr/bin/env python
"""grid phase separation — 표·그림 리포트.

`phase_sep_matrix.py` 가 낸 JSON 1개 이상을 받아
  (a) `matrix.tsv`  — 전 JSON 통합 평평한 표
  (b) task 별 heatmap PNG — 행=layer(+VL), 열=phase, 값=AUROC, 주석=null_z
      Tier B(49 토큰) JSON 이면 행=token(0..48, 세그먼트 경계선 표시)
를 출력 dir 에 쓴다. exploratory(혼재 scene 부족) task 는 제목에 표시한다.

colormap 은 0.5 중심 diverging (AUROC 는 0.5 가 chance) — TwoSlopeNorm.
축 라벨은 영문, 파일명에 instruction slug.

    python report_tables.py out/gridA.json out/gridB.json --out-dir out/report
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.colors import TwoSlopeNorm   # noqa: E402

TSV_COLS = ["source", "instruction", "slug", "tier", "layer", "seg", "token", "denoise",
            "phase_def", "phase", "phase_code", "auroc", "null_z", "p", "length_auroc",
            "budget_B", "n_succ", "n_fail", "n_ep", "n_mixed_scenes", "exploratory",
            "skipped", "skip_reason"]


def load_cells(paths: list[Path]) -> list[dict]:
    cells = []
    for p in paths:
        blob = json.loads(Path(p).read_text())
        for c in blob.get("cells", []):
            c = dict(c)
            c["source"] = Path(p).name
            cells.append(c)
    return cells


def write_matrix_tsv(cells: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(TSV_COLS)]
    for c in cells:
        lines.append("\t".join("" if c.get(k) is None else str(c.get(k)) for k in TSV_COLS))
    out.write_text("\n".join(lines) + "\n")


def _layer_sort_key(v: str):
    """숫자 layer 는 값 순, 'VL' 은 맨 뒤."""
    try:
        return (0, int(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _phase_order(cells: list[dict]) -> list[str]:
    seen = {}
    for c in cells:
        if c.get("phase") is None:
            continue
        seen.setdefault(str(c["phase"]), c.get("phase_code"))
    return sorted(seen, key=lambda k: (seen[k] is None, seen[k], k))


def heatmap(rows: list[str], cols: list[str], A: np.ndarray, Z: np.ndarray,
            title: str, ylabel: str, path: Path, hlines: list[float] | None = None) -> None:
    fin = A[np.isfinite(A)]
    lo = float(min(0.49, fin.min())) if fin.size else 0.0
    hi = float(max(0.51, fin.max())) if fin.size else 1.0
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.5, vmax=hi)
    h = max(2.2, 0.30 * len(rows) + 1.6)
    w = max(3.4, 1.30 * len(cols) + 2.0)
    fig, ax = plt.subplots(figsize=(w, h), dpi=140)
    im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7 if len(rows) > 20 else 9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    annot = len(rows) * len(cols) <= 260
    if annot:
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isfinite(A[i, j]):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=6, color="0.4")
                    continue
                z = Z[i, j]
                txt = f"{A[i, j]:.2f}" + ("" if not np.isfinite(z) else f"\nz{z:.1f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                        color="black" if abs(A[i, j] - 0.5) < 0.22 else "white")
    for y in (hlines or []):
        ax.axhline(y, color="0.2", lw=0.8, ls="--")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="LOSO AUROC")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def grid_from(cells: list[dict], row_key, rows: list, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    A = np.full((len(rows), len(cols)), np.nan)
    Z = np.full((len(rows), len(cols)), np.nan)
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: j for j, c in enumerate(cols)}
    for c in cells:
        r, p = row_key(c), str(c.get("phase"))
        if r not in ri or p not in ci:
            continue
        if c.get("auroc") is not None:
            A[ri[r], ci[p]] = float(c["auroc"])
        if c.get("null_z") is not None:
            Z[ri[r], ci[p]] = float(c["null_z"])
    return A, Z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="phase_sep_matrix.py 출력 JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--token-bounds", default="",
                    help="Tier B heatmap 세그먼트 경계 token index 콤마목록 (예: 1,17)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cells = load_cells([Path(p) for p in args.json])
    write_matrix_tsv(cells, out_dir / "matrix.tsv")
    print("[written]", out_dir / "matrix.tsv", f"({len(cells)} rows)", flush=True)

    live = [c for c in cells if not c.get("skipped")]
    bounds = [float(t.strip()) - 0.5 for t in args.token_bounds.split(",") if t.strip()]
    n_png = 0

    keys = sorted({(str(c.get("slug")), str(c.get("tier")), str(c.get("phase_def")),
                    str(c.get("denoise")), str(c.get("seg") or "")) for c in live})
    # Tier A 는 (slug, seg, denoise) 별 1장, Tier B 는 (slug, denoise) 별 1장.
    groups: dict[tuple, list[dict]] = {}
    vl_cells: list[dict] = []
    for c in live:
        if str(c.get("layer")) == "VL":
            vl_cells.append(c)            # VL 은 seg 축이 없다 → 모든 Tier A 그림에 행으로 붙인다
            continue
        if str(c.get("tier")) == "B":
            # Tier B 는 행이 token 이므로 layer 별로 그림을 나눈다 (한 장에 섞으면 덮어쓴다).
            g = (c["slug"], "B", c.get("phase_def"), c.get("denoise"), c.get("layer"))
        else:
            g = (c["slug"], "A", c.get("phase_def"), c.get("denoise"), c.get("seg"))
        groups.setdefault(g, []).append(c)
    for g in list(groups):
        if g[1] != "A":
            continue
        for c in vl_cells:
            if (c["slug"], c.get("phase_def"), c.get("denoise")) == (g[0], g[2], g[3]):
                groups[g].append(c)
    del keys

    for g, gc in sorted(groups.items(), key=lambda kv: str(kv[0])):
        slug, tier, pdef, dn, seg = g          # tier B 면 seg 자리는 layer
        cols = _phase_order(gc)
        if not cols:
            continue
        expl = any(bool(c.get("exploratory")) for c in gc)
        nmix = [c.get("n_mixed_scenes") for c in gc if c.get("n_mixed_scenes") is not None]
        tag = f"  [EXPLORATORY: mixed scenes={min(nmix) if nmix else '?'}]" if expl else ""
        if tier == "B":
            toks = sorted({int(c["token"]) for c in gc if c.get("token") is not None})
            # VL 행은 token 이 없다 → 별도 취급 없이 제외 (Tier B 는 token 축 전용).
            A, Z = grid_from(gc, lambda c: int(c["token"]) if c.get("token") is not None else -1,
                             toks, cols)
            title = f"{slug} | L{seg} tokens | phase-def={pdef} | denoise={dn}{tag}"
            path = out_dir / f"heatB__{slug}__L{seg}__dn-{dn}__{pdef.replace(':', '-')}.png"
            heatmap([str(t) for t in toks], cols, A, Z, title, "token", path, hlines=bounds)
        else:
            rws = sorted({str(c["layer"]) for c in gc if c.get("layer") is not None},
                         key=_layer_sort_key)
            A, Z = grid_from(gc, lambda c: str(c.get("layer")), rws, cols)
            title = f"{slug} | seg={seg} | phase-def={pdef} | denoise={dn}{tag}"
            path = out_dir / f"heat__{slug}__seg-{seg}__dn-{dn}__{pdef.replace(':', '-')}.png"
            heatmap(rws, cols, A, Z, title, "layer", path)
        n_png += 1
        print("[written]", path, flush=True)

    print(f"[report] {n_png} heatmap(s) → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
