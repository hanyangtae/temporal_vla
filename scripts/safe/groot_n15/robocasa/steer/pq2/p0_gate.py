"""P0 scene 선정 게이트 (apple 완화판 — 2026-07-10 사용자 확정).

게이트: corrected 라벨 기준 succ ≥ 6 ∧ fail ≥ 6 (원안 15/15에서 완화 — apple scene 이
all-or-nothing 이부모델로 밴드[0.25,0.75] scene 부재 실증). fit30 은 fail ≥ 12 ∧ succ ≥ 12
일 때만 가능(30/30 split 후 half 하한 6) — 미달 scene 은 fit15 전용 + 천장/바닥 각주.
선발: 게이트 통과 scene 중 **seed 순서** (관측 SR 순위 금지 — Gate1 합의) 상위 N.

라벨 출처: 로컬 수집 cell = pkl 파일명 succ 플래그 / w2 수집 cell = manifest_w2.txt 파일명
(둘 다 corrected — robocasa fork 459c70f 이후 수집분).

usage: p0_gate.py --cells <cell>... [--select 3] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
SRC6 = REPO / "outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts/PickPlaceCounterToStove"

MIN_CLASS_GATE = 6     # 완화 게이트 (scene 전체, 양 클래스)
FIT30_MIN = 12         # fit30 가능 하한 (half 하한 6 × 2)


def scene_counts(cell: str):
    d = SRC6 / cell
    eps = {}
    for p in d.glob("task*--ep*--succ*.pkl"):
        m = re.match(r"task\d+--ep(\d+)--succ([01])\.pkl", p.name)
        if m:
            eps[int(m.group(1))] = int(m.group(2))
    source = "local_pkl"
    if not eps:
        mf = d / "manifest_w2.txt"
        if mf.exists():
            source = "manifest_w2"
            for line in mf.read_text().splitlines():
                m = re.search(r"task\d+--ep(\d+)--succ([01])\.pkl", line)
                if m:
                    eps[int(m.group(1))] = int(m.group(2))
    succ = sum(eps.values())
    return {"cell": cell, "n": len(eps), "succ": succ, "fail": len(eps) - succ, "source": source}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--select", type=int, default=3)
    ap.add_argument("--out", default="outputs/eval/robocasa/groot_n15/pq2_analysis/p0_gate_report.json")
    args = ap.parse_args()
    rows = []
    for cell in args.cells:
        r = scene_counts(cell)
        r["complete"] = r["n"] >= 60
        r["gate_pass"] = r["complete"] and r["succ"] >= MIN_CLASS_GATE and r["fail"] >= MIN_CLASS_GATE
        r["fit30_ok"] = r["gate_pass"] and r["succ"] >= FIT30_MIN and r["fail"] >= FIT30_MIN
        r["seed"] = int(re.search(r"s(\d+)$", cell).group(1)) if re.search(r"s(\d+)$", cell) else 0
        rows.append(r)
    passers = sorted([r for r in rows if r["gate_pass"]], key=lambda r: r["seed"])
    selected = [r["cell"] for r in passers[: args.select]]
    report = {"gate": {"min_class": MIN_CLASS_GATE, "fit30_min": FIT30_MIN,
                       "note": "완화판 (원안 15/15) — 2026-07-10 사용자 확정, 천장/바닥 각주 필수"},
              "rows": rows, "passers_seed_order": [r["cell"] for r in passers], "selected": selected}
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for r in sorted(rows, key=lambda r: r["seed"]):
        mark = "SELECT" if r["cell"] in selected else ("pass" if r["gate_pass"] else ("incomplete" if not r["complete"] else "fail"))
        print(f"{r['cell']:24s} {r['succ']:3d}/{r['n']:3d} succ  [{mark}]"
              + ("" if r["fit30_ok"] or not r["gate_pass"] else " (fit15 전용)"))
    print(f"selected({len(selected)}/{args.select}): {selected}")


if __name__ == "__main__":
    main()
