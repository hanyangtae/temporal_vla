#!/usr/bin/env python3
"""exp4-3: 셀별 atlas JSON → 병합 TSV (+ COAST quota 곡선 요약).

사용: python atlas_emit.py --atlas-root <.../exp4_3/atlas> --out <.../atlas_all.tsv>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

COLS = ["model", "cell", "layer", "phase", "var_z", "var_gain", "mean_z", "mean_auroc",
        "quota", "n_rec_s", "n_rec_f", "n_eps_s", "n_eps_f", "dwell_cap", "alpha",
        "skip_reason"]


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else f"{v:.6g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows, meta = [], []
    for p in sorted(glob.glob(str(args.atlas_root / "*" / "*.json"))):
        a = json.loads(Path(p).read_text())
        meta.append({k: a.get(k) for k in
                     ("model", "cell", "D", "capture_layers", "n_rollouts", "n_succ",
                      "global_cap", "feature_kind")})
        for c in a["cells"]:
            rows.append([fmt(c.get(k)) for k in COLS])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\t".join(COLS) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[wrote] {args.out} rows={len(rows)} cells={len(meta)}")


if __name__ == "__main__":
    main()
