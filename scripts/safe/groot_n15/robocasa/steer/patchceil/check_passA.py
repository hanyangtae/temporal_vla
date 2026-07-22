"""patchceil pass A 결정론 게이트 — 재수집 성공 패턴 vs pq2 ho_base 기록(targets.tsv) 대조.

전 60판 × cell 에서 succ 플래그(파일명 스템)가 정확히 일치해야 한다. 하나라도 다르면
결정론 전제가 깨진 것 — 실험 중단·보고 (plan v2 §검증 1).

stdlib 전용. 사용: python3 check_passA.py [--cells ppcc_bread_s300033,ppcc_bread_s400020]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
GROOT = REPO / "outputs/eval/robocasa/groot_n15"
TASK = "PickPlaceCounterToCabinet"
CELL_INDEX = 5


def check_cell(cell: str) -> bool:
    targets = GROOT / "patchceil" / cell / "targets.tsv"
    rows = list(csv.DictReader(open(targets), delimiter="\t"))
    rdir = GROOT / "patchceil" / cell / "passA/raw_rollouts" / TASK / cell
    ok, missing, mismatch = 0, [], []
    for r in rows:
        ep = int(r["episode_idx"])
        expect = int(r["succ"])
        hits = sorted(rdir.glob(f"task{CELL_INDEX}--ep{ep}--succ*.pkl"))
        if not hits:
            missing.append(ep)
            continue
        got = 1 if hits[-1].name.endswith("succ1.pkl") else 0
        if got != expect:
            mismatch.append((ep, expect, got))
        else:
            ok += 1
    print(f"[{cell}] match={ok}/{len(rows)} missing={len(missing)} mismatch={len(mismatch)}")
    if missing:
        print(f"  missing ep: {missing}")
    if mismatch:
        print(f"  MISMATCH (ep, pq2, 재수집): {mismatch}")
        print("  → 결정론 전제 위반 — 원인 규명 전 patchceil 진행 금지")
    return not missing and not mismatch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells", default="ppcc_bread_s300033,ppcc_bread_s400020",
        help="콤마 구분 cell 목록",
    )
    args = ap.parse_args()
    results = [check_cell(c) for c in args.cells.split(",") if c]
    if all(results):
        print("PASS: 결정론 게이트 통과 — pass B/anchor 진행 가능")
        return 0
    print("FAIL: 결정론 게이트 불통과")
    return 1


if __name__ == "__main__":
    sys.exit(main())
