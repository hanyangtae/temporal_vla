"""patchceil — 추출 메타(JSON)로 t0 규칙 초안·donor 선정 재료 집계 (stdlib 전용).

출력: cell × succ 별 record 수 분포, phase 최초 진입 record(occurrence index),
실패의 도달 phase 분포(어디까지 갔나), wrong_grasp 비율.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
META = REPO / "outputs/eval/robocasa/groot_n15/patchceil/patchceil_meta"
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]
PHASES = ["reach-to-object", "grasp", "transport", "place", "insert-settle", "terminal"]


def first_record_of(phases: list[str], name: str) -> int | None:
    for i, p in enumerate(phases):
        if p == name:
            return i
    return None


def qtiles(xs: list[int]) -> str:
    if not xs:
        return "-"
    s = sorted(xs)
    n = len(s)
    return f"min{s[0]} q1:{s[n // 4]} med:{s[n // 2]} q3:{s[3 * n // 4]} max:{s[-1]}"


def main() -> None:
    for cell in CELLS:
        eps = []
        for f in sorted((META / cell).glob("ep*.json")):
            if "_actions" in f.name:
                continue
            eps.append(json.loads(f.read_text()))
        for succ in (1, 0):
            grp = [d for d in eps if int(d["episode_success"]) == succ]
            tag = "성공(donor풀)" if succ else "실패(target)"
            print(f"\n=== {cell} {tag} n={len(grp)} ===")
            print(f"  n_records: {qtiles([d['n_records'] for d in grp])}")
            for ph in PHASES:
                firsts = [
                    first_record_of(d["feature_phases"], ph)
                    for d in grp
                ]
                reached = [x for x in firsts if x is not None]
                print(
                    f"  {ph:16s} 도달 {len(reached)}/{len(grp)}"
                    + (f"  최초record {qtiles(reached)}" if reached else "")
                )
            wg = [d for d in grp if (d.get("wrong_grasp_steps") or [])]
            print(f"  wrong_grasp 발생: {len(wg)}/{len(grp)}")
            # 실패의 '마지막 phase' 분포 = 어디서 막혔나
            if not succ:
                last = Counter(d["feature_phases"][-1] for d in grp if d["feature_phases"])
                # 도달 최고 phase (PHASES 순서 기준)
                order = {p: i for i, p in enumerate(PHASES)}
                peak = Counter(
                    max((p for p in set(d["feature_phases"]) if p in order),
                        key=lambda p: order[p], default="?")
                    for d in grp
                )
                print(f"  timeout 시점 phase: {dict(last)}")
                print(f"  도달 최고 phase:    {dict(peak)}")


if __name__ == "__main__":
    main()
