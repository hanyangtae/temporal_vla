#!/usr/bin/env python3
"""RL2-VLA SIMPLER 재현 결과 집계 (Stage 1b/1c).

각 lane 로그에서 task별 "Task success rate: X" 를 순서대로 파싱해
arm × seed × (alpha) 표를 만들고, 논문 프로토콜(alpha 사후선택)과
alpha 평균 두 가지로 요약한다.

사용: python aggregate.py [experiments_dir]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

TASKS = ["orange_juice", "spoon_google", "tape_measure", "toy_dinosaur"]
# 논문 Fig 8 (막대 판독값)
PAPER = {"vanilla": [31.3, 46.0, 18.7, 48.0], "rephrase": [35.3, 44.6, 52.0, 49.3],
         "always": [41.3, 54.0, 44.6, 46.0], "adaptive": [43.3, 59.3, 59.3, 53.3]}

exp = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[3] / "RL2-VLA/experiments")


def parse(path):
    """로그에서 task별 SR(%) 리스트를 순서대로 뽑는다."""
    if not path.exists():
        return None
    srs = [float(m) * 100 for m in
           re.findall(r"^Task success rate: ([0-9.]+)", path.read_text(errors="ignore"), re.M)]
    return srs if len(srs) == len(TASKS) else (srs or None)


# lane 로그 수집: {(arm, seed, alpha): [task SR...]}
lanes = {}
for log in sorted(exp.glob("*.log")):
    name = log.stem
    if name.startswith("full_"):
        m = re.match(r"full_(\w+?)_s(\d+)(?:_a([\d.]+))?$", name)
        if not m:
            continue
        arm, seed, alpha = m.group(1), m.group(2), m.group(3)
    elif name.startswith("stage1b_launch_"):
        arm, seed, alpha = name.replace("stage1b_launch_", ""), "42", ("0.1" if "adaptive" in name else None)
    else:
        continue
    srs = parse(log)
    if srs:
        lanes[(arm, seed, alpha)] = srs

if not lanes:
    sys.exit("lane 로그를 찾지 못했습니다: " + str(exp))

done = {k: v for k, v in lanes.items() if len(v) == len(TASKS)}
partial = {k: v for k, v in lanes.items() if len(v) != len(TASKS)}
print(f"수집 lane {len(lanes)}개 (완료 {len(done)} / 진행중 {len(partial)})\n")
print(f"{'arm':10} {'seed':5} {'alpha':6} " + " ".join(f"{t:>13}" for t in TASKS) + f" {'평균':>7}")
for (arm, seed, alpha), srs in sorted(lanes.items()):
    tag = "" if len(srs) == len(TASKS) else "  ← 진행중"
    avg = sum(srs) / len(srs)
    print(f"{arm:10} {seed:5} {str(alpha or '-'):6} " +
          " ".join(f"{s:13.1f}" for s in srs) + f" {avg:7.1f}{tag}")
lanes = done  # 요약은 완료 lane 만

# ── 요약: arm별 (seed 평균) ─────────────────────────────────────────
print("\n" + "=" * 78)
print("arm 요약 — adaptive 는 두 프로토콜로 병기")
print("=" * 78)


def seed_avg(arm, pick):
    """pick: 'best'(seed별 최선 alpha=논문 프로토콜) | 'mean'(alpha 평균)
             | 'oracle'(task별 최선 alpha = 최대 관대 해석)"""
    per_seed = defaultdict(list)
    for (a, s, _), srs in lanes.items():
        if a == arm:
            per_seed[s].append(srs)
    out = []
    for _, runs in sorted(per_seed.items()):
        if pick == "oracle":
            out.append(sum(max(r[i] for r in runs) for i in range(len(TASKS))) / len(TASKS))
            continue
        avgs = [sum(r) / len(r) for r in runs]
        out.append(max(avgs) if pick == "best" else sum(avgs) / len(avgs))
    return out


rows = []
for arm in ["vanilla", "rephrase", "always"]:
    v = seed_avg(arm, "mean")
    if v:
        rows.append((arm, sum(v) / len(v), len(v), ""))
for pick, label in [("best", "adaptive (seed별 alpha 사후선택 = 논문 프로토콜)"),
                    ("oracle", "adaptive (task별 alpha oracle = 최대 관대)"),
                    ("mean", "adaptive (alpha 평균 = 사후선택 없음)")]:
    v = seed_avg("adaptive", pick)
    if v:
        rows.append((label, sum(v) / len(v), len(v), ""))

base = next((sr for a, sr, _, _ in rows if a == "rephrase"), None)
print(f"{'arm':42} {'SR':>6} {'ΔSR':>7} {'seed수':>6}   논문")
for arm, sr, n, _ in rows:
    key = arm.split()[0]
    paper = sum(PAPER[key]) / 4 if key in PAPER else None
    d = f"{sr - base:+.1f}" if base is not None and key != "rephrase" else "—"
    p = f"{paper:.1f}" if paper else ""
    print(f"{arm:42} {sr:6.1f} {d:>7} {n:6}   {p}")

print("\n논문 Fig 8 평균: vanilla 36.0 / rephrase 45.3 / always 46.5 (+1.2) / adaptive 53.7 (+8.4)")
