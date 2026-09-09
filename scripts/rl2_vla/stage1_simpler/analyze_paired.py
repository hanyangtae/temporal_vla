#!/usr/bin/env python3
"""lane 간 per-episode 짝대응 비교 (resample 라운드용).

env reset seed는 episode_idx에 결정적(itertools.count(1000), cfg.seed 무관)이라
같은 (task, episode_idx) 셀은 arm 간 동일 초기조건 → 셀 단위 뒤집힘 판정 가능.

사용: python analyze_paired.py <lane_dir_A(기준, 예: vanilla)> <lane_dir_B> [...]
  lane_dir = .../stage1b_OOD_seed{S}/{arm} (logs/*.txt 보유)
B가 여러 개면 각각 A와 비교. 출력: task별·전체 SR, 구제(A실패→B성공)/파손(A성공→B실패) 수.
"""
import re
import sys
from pathlib import Path

TASK_ORDER = ["orange_juice", "spoon_on_towel_google", "tape_measure", "toy_dinosaur"]


def lane_successes(lane_dir):
    """{task_key: [bool, ...]} — 로그 txt의 'Success:' 줄 순서 = episode 순서."""
    out = {}
    for txt in sorted(Path(lane_dir, "logs").glob("*.txt")):
        m = re.search(r"simpler_(\w+?)-batch", txt.name)
        key = next((t for t in TASK_ORDER if m and t in m.group(1)), m.group(1) if m else txt.name)
        succ = [s == "True" for s in re.findall(r"^Success: (\w+)", txt.read_text(errors="ignore"), re.M)]
        out[key] = succ
    return out


def sr(v):
    return 100.0 * sum(v) / len(v) if v else float("nan")


def main():
    base_dir, *others = sys.argv[1:]
    base = lane_successes(base_dir)
    print(f"기준 lane: {base_dir}")
    for t, v in base.items():
        print(f"  {t:24} n={len(v):3} SR={sr(v):5.1f}")
    for other_dir in others:
        other = lane_successes(other_dir)
        print(f"\n비교 lane: {other_dir}")
        tot_r = tot_h = tot_n = 0
        for t in base:
            if t not in other:
                print(f"  {t:24} (비교 lane에 없음)")
                continue
            n = min(len(base[t]), len(other[t]))
            a, b = base[t][:n], other[t][:n]
            rescued = sum(1 for x, y in zip(a, b) if not x and y)
            harmed = sum(1 for x, y in zip(a, b) if x and not y)
            tot_r += rescued; tot_h += harmed; tot_n += n
            print(f"  {t:24} n={n:3} SR {sr(a):5.1f}→{sr(b):5.1f} ({sr(b)-sr(a):+5.1f})"
                  f"  구제 {rescued:2} / 파손 {harmed:2}")
        if tot_n:
            ab = [x for t in base if t in other for x in base[t][:min(len(base[t]), len(other[t]))]]
            bb = [x for t in base if t in other for x in other[t][:min(len(base[t]), len(other[t]))]]
            print(f"  {'전체':24} n={tot_n:3} SR {sr(ab):5.1f}→{sr(bb):5.1f} ({sr(bb)-sr(ab):+5.1f})"
                  f"  구제 {tot_r:2} / 파손 {tot_h:2}")


if __name__ == "__main__":
    main()
