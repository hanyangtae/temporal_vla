"""CloseFridge smoke 요약: instruction × 성공/실패, phase 히스토그램, near-miss 진단.

사이드카(json, 캡처-OFF 수집)만 읽는다 — env/robocasa 불필요.
사용: python fridge_smoke_report.py --run-dir <.../fridge_smoke/raw_rollouts>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def compress(seq: list[str], limit: int = 24) -> str:
    out: list[list] = []
    for p in seq:
        if out and out[-1][0] == p:
            out[-1][1] += 1
        else:
            out.append([p, 1])
    txt = " ".join(f"{p}x{n}" for p, n in out[:limit])
    return txt + (" …" if len(out) > limit else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--transitions", action="store_true", help="에피소드별 phase 전이 출력")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.run_dir).rglob("task*--ep*--succ*.json")):
        d = json.loads(p.read_text())
        _open = str(d.get("task", "")).startswith("Open")
        esp = d.get("env_step_phases") or []
        ds = d.get("env_step_door_state") or []
        dw = d.get("env_step_door_worst") or []
        rows.append(
            {
                "cell": p.parent.name,
                "ep": int(p.stem.split("--ep")[1].split("--")[0]),
                "seed": d.get("scenario_seed"),
                "instr": d.get("task_description", ""),
                "succ": int(d.get("episode_success", 0)),
                "steps": int(d.get("steps", 0)),
                "phases": esp,
                "events": d.get("env_step_event_steps") or {},
                # 목표 방향으로 가장 멀리 간 값(=성공에 가장 근접했던 순간).
                # Open* 은 관절값이 커야 성공, Close* 은 작아야 성공 → 극값 방향이 반대.
                "door_best": (max(ds) if _open else min(ds)) if ds else None,
                "worst_best": (max(dw) if _open else min(dw)) if dw else None,
                "wg": len(d.get("env_step_wrong_grasp_steps") or []),
                "path": p,
            }
        )

    if not rows:
        print("no episodes")
        return

    print(f"{'cell':14s} {'ep':>3} {'seed':>7} {'succ':>4} {'steps':>5} "
          f"{'tgtBest':>7} {'allBest':>7} {'wg':>3}  instruction")
    for r in sorted(rows, key=lambda r: (r["cell"], r["ep"])):
        dm = f"{r['door_best']:.3f}" if r["door_best"] is not None else "-"
        wm = f"{r['worst_best']:.3f}" if r["worst_best"] is not None else "-"
        print(f"{r['cell']:14s} {r['ep']:>3} {r['seed']:>7} {r['succ']:>4} "
              f"{r['steps']:>5} {dm:>7} {wm:>7} {r['wg']:>3}  {r['instr']}")

    print("\n== instruction × 결과")
    tab: dict = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        tab[(r["instr"], r["cell"])][r["succ"]] += 1
    for k, (f0, s1) in sorted(tab.items()):
        print(f"  {k[0]:26s} {k[1]:14s} succ {s1}/{s1 + f0}  fail {f0}")

    print("\n== phase 점유 (env-step, 전체 합산)")
    cnt: collections.Counter = collections.Counter()
    for r in rows:
        cnt.update(r["phases"])
    tot = sum(cnt.values()) or 1
    for p, n in cnt.most_common():
        print(f"  {p:16s} {n:7d}  {n / tot:6.1%}")

    if args.transitions:
        print("\n== phase 전이")
        for r in sorted(rows, key=lambda r: (r["cell"], r["ep"])):
            print(f"  [{r['cell']} ep{r['ep']} succ{r['succ']}] {compress(r['phases'])}")
            print(f"      events: {r['events']}")


if __name__ == "__main__":
    main()
