#!/usr/bin/env python3
"""pq3 fit 수집 게이트 (계획서 v9 §C) — pq2 p0_gate.py 파생, 경로 인자화.

게이트: fit 목표 판수 수집 완료 ∧ succ≥3 ∧ fail≥3 (env 원판정 — pq3 에 stove 없음,
succ_ever_th 미사용). 부족 시 collect_plan 순서(S16, S17, …)로 backfill EPLIST 를
출력하고 rc=2, plan 소진(≤S60) 시 rc=3 (cell 탈락 신호 — 3연속 탈락이면 Gate C
4-cell fallback 사용자 게이트).

exit: 0=게이트 통과 | 2=backfill 필요(stdout BACKFILL_EPLIST=...) | 3=plan 소진·탈락
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pq3_manifests import _read_plan, scan_collected  # noqa: E402

MIN_CLASS = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collected-dir", required=True, help="수집 결과 (succ 스템 디렉토리)")
    ap.add_argument("--manifest-dir", required=True, help="make_pq3_manifests --out-dir/<cell>")
    ap.add_argument("--fit-target", type=int, default=15)
    ap.add_argument("--min-class", type=int, default=MIN_CLASS)
    ap.add_argument("--backfill-step", type=int, default=1,
                    help="한 번에 지시할 backfill seed 수 (순서 준수)")
    args = ap.parse_args()

    plan = _read_plan(Path(args.manifest_dir))
    plan_idx = [r["s_idx"] for r in plan]
    collected = scan_collected(Path(args.collected_dir))
    n = len(collected)
    succ = sum(1 for r in collected.values() if r["succ"] == 1)
    fail = n - succ

    outside = sorted(set(collected) - set(plan_idx))
    if outside:
        print(f"[p0-pq3] 수집 ep 가 plan 밖: {outside} (seed 구획 위반)", file=sys.stderr)
        sys.exit(4)

    # 목표 판수: fit_target + 이미 수행된 backfill 유지 (수집된 전 판이 fit 재료)
    need_more = max(0, args.fit_target - n)
    class_ok = succ >= args.min_class and fail >= args.min_class
    report = {
        "n": n, "succ": succ, "fail": fail,
        "fit_target": args.fit_target, "min_class": args.min_class,
        "class_ok": class_ok,
        "collected_eps": sorted(collected),
    }

    remaining = [i for i in plan_idx if i not in collected]
    if need_more > 0 or not class_ok:
        take = need_more if need_more > 0 else args.backfill_step
        backfill = remaining[:take]
        report["deficient"] = (
            "count" if need_more > 0 else ("succ" if succ < args.min_class else "fail")
        )
        report["backfill"] = backfill
        print(json.dumps(report, ensure_ascii=False))
        if not backfill:
            print(f"[p0-pq3] plan 소진 (S{plan_idx[-1] + 1} 상한) — cell 탈락", file=sys.stderr)
            sys.exit(3)
        print("BACKFILL_EPLIST=" + " ".join(str(i) for i in backfill))
        sys.exit(2)

    print(json.dumps(report, ensure_ascii=False))
    print(f"[p0-pq3] GATE PASS n={n} succ={succ} fail={fail}")


if __name__ == "__main__":
    main()
