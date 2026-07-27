#!/usr/bin/env python3
"""exp5-3 집계: within-scene setM steering (drawer_right) — arm×scene×seed 매트릭스.

입력:
  --baseline-stems : srv50 scene-matched 수집의 pkl/csv 스템 목록 (task7--ep{N}--succ{S})
                     — 무개입 baseline. ★cross-machine (srv50 수집 vs home eval) — 결정론은
                     머신-로컬이라 per-episode trajectory 짝은 성립 안 함. paired 통계는
                     "같은 (scene,seed) 조건의 독립 재표본" 해석 + 각주. home A0 앵커가
                     머신 이동 보정 참조점.
  --eval-root      : home eval 산출 루트 (exp5_3/<arm>/raw_rollouts/.../task7--ep*--succ*.csv)

ep ↔ (scene, seed) 매핑: epidx = scene_idx*8 + seed_idx (수집·eval 러너 공통 규약).
"""
import argparse
import json
import re
from pathlib import Path

SCENES = [100000, 100003, 100005, 100006, 100009, 100010, 100011, 100012, 100016,
          100018, 100020, 100022, 100023, 100025, 100026, 100031, 100033, 100034,
          100035, 100039]
MIXED_EXPECT_KEY = "mixed_scenes"  # fit_report.json 의 혼재 scene 목록과 대조
STEM_RE = re.compile(r"task7--ep(\d+)--succ([01])")


def stems_to_grid(lines):
    grid = {}
    for ln in lines:
        m = STEM_RE.search(ln)
        if not m:
            continue
        ep, s = int(m.group(1)), int(m.group(2))
        grid[(SCENES[ep // 8], ep % 8)] = s
    return grid


def mcnemar(b, c):
    """b = base실패→arm성공(구제), c = base성공→arm실패(해악). 정확 이항 p (양측)."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-stems", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--fit-report", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = stems_to_grid(Path(args.baseline_stems).read_text().splitlines())
    if len(base) != 160:
        print(f"[경고] baseline {len(base)}/160")
    mixed = None
    if args.fit_report:
        mixed = set(json.loads(Path(args.fit_report).read_text())[MIXED_EXPECT_KEY])

    out = dict(n_baseline=len(base),
               baseline_sr=round(sum(base.values()) / len(base), 4), arms={})
    for arm_dir in sorted(Path(args.eval_root).iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        grid = stems_to_grid([p.name for p in arm_dir.rglob("task7--ep*--succ*.json")])
        if not grid:
            continue
        common = sorted(set(grid) & set(base))
        b = sum(1 for k in common if base[k] == 0 and grid[k] == 1)  # 구제
        c = sum(1 for k in common if base[k] == 1 and grid[k] == 0)  # 해악
        rec = dict(
            n=len(grid), n_common=len(common),
            sr=round(sum(grid.values()) / len(grid), 4),
            base_sr_on_common=round(sum(base[k] for k in common) / max(len(common), 1), 4),
            delta_games=sum(grid[k] for k in common) - sum(base[k] for k in common),
            rescued_b=b, harmed_c=c,
            mcnemar_p_crossmachine_footnote=round(mcnemar(b, c), 4),
            per_seed_sr={k: round(sum(v for (s, kk), v in grid.items() if kk == k)
                                  / max(sum(1 for (s, kk) in grid if kk == k), 1), 3)
                         for k in range(8)},
            per_scene_delta={s: sum(grid[(s, k)] - base[(s, k)] for k in range(8)
                                    if (s, k) in grid and (s, k) in base)
                             for s in SCENES},
        )
        if mixed:
            mc = [k for k in common if k[0] in mixed]
            pc = [k for k in common if k[0] not in mixed]
            rec["sr_mixed_scenes"] = round(sum(grid[k] for k in mc) / max(len(mc), 1), 4)
            rec["sr_pure_scenes"] = round(sum(grid[k] for k in pc) / max(len(pc), 1), 4)
            rec["base_sr_mixed"] = round(sum(base[k] for k in mc) / max(len(mc), 1), 4)
            rec["base_sr_pure"] = round(sum(base[k] for k in pc) / max(len(pc), 1), 4)
        out["arms"][arm] = rec

    # A0 앵커로 머신 이동 진단: 앵커 (scene,seed) 부분집합에서 srv50 결과와 대조
    if "A0_anchor" in out["arms"]:
        a = out["arms"]["A0_anchor"]
        out["machine_shift_note"] = (
            f"home A0 앵커 SR {a['sr']} vs srv50 동일부분집합 {a['base_sr_on_common']} — "
            "차이가 크면 arm SR 해석에 머신 이동 보정 필요 (hardware-calibration 각주)")

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
