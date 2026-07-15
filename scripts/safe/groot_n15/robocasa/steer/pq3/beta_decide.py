#!/usr/bin/env python3
"""pq3 β 선택 판정 (계획서 v9 §D β sweep).

입력: task 의 cell 목록 × arm(perm/gated) × β{0.1,0.3} sweep 결과(사이드카 json 스템)
+ 각 cell sweep_manifest 의 base_label(수집 base 라벨 — paired 참조).

규칙 (task 별·arm 별 독립):
  - 하방 탈락: β 의 task-pool 승수 < base-2판 이면 그 β 탈락.
  - 승자: 생존 β 중 승수 최대, 동률 → 0.1.
  - red flag: 양 β 모두 task-pool base-4판 이하 → 해당 arm 중단 신호(rc=6, 사용자 결정).

exit: 0=선택 완료 | 6=red flag (arm 중단·사용자 보고) | 7=생존 β 없음 (사용자 결정)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pq3_manifests import STEM_RE  # noqa: E402

DOWNSIDE_GAMES = 2   # base 대비 -2판 → 해당 β 탈락
RED_FLAG_GAMES = 4   # 양 β 모두 base 대비 -4판 → arm 중단


def read_sweep_arm(sweep_root: Path, cell: str, task: str, arm_tag: str) -> dict[int, int]:
    d = sweep_root / cell / arm_tag / "raw_rollouts" / task / cell
    out = {}
    for p in sorted(d.glob("task*--ep*--succ*.json")):
        m = STEM_RE.match(p.name)
        if m:
            out[int(m.group("ep"))] = int(m.group("succ"))
    return out


def read_base_labels(manifest_dir: Path, cell: str) -> dict[int, int]:
    path = manifest_dir / cell / "sweep_manifest.tsv"
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("ep_idx"):
            continue
        ep, _envs, _noise, label = line.split("\t")[:4]
        out[int(ep)] = int(label)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", required=True, help="steer_eval_pq3/sweep 루트")
    ap.add_argument("--manifest-dir", required=True, help="make_pq3_manifests --out-dir")
    ap.add_argument("--task", required=True)
    ap.add_argument("--cells", required=True, help="task 의 cell 콤마 목록")
    ap.add_argument("--arms", default="perm,gated")
    ap.add_argument("--betas", default="0.1,0.3")
    ap.add_argument("--out", required=True, help="선택 결과 json 경로")
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    betas = [b.strip() for b in args.betas.split(",") if b.strip()]
    sweep_root = Path(args.sweep_root)
    manifest_dir = Path(args.manifest_dir)

    base_total = 0
    base_by_cell = {}
    for cell in cells:
        labels = read_base_labels(manifest_dir, cell)
        base_by_cell[cell] = labels
        base_total += sum(labels.values())

    result = {"task": args.task, "cells": cells, "base_wins": base_total,
              "downside_games": DOWNSIDE_GAMES, "red_flag_games": RED_FLAG_GAMES, "arms": {}}
    red_flag = False
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        wins_by_beta = {}
        detail = {}
        for beta in betas:
            arm_tag = f"sweep_{arm}_b{beta.replace('.', '')}"
            total, per_cell, missing = 0, {}, 0
            for cell in cells:
                got = read_sweep_arm(sweep_root, cell, args.task, arm_tag)
                expect = set(base_by_cell[cell])
                missing += len(expect - set(got))
                per_cell[cell] = sum(got.get(e, 0) for e in expect)
                total += per_cell[cell]
            if missing:
                raise SystemExit(f"[beta-decide] {arm} β={beta} 미완주 {missing}판 — sweep 완료 후 재실행")
            wins_by_beta[beta] = total
            detail[beta] = per_cell
        # 하방 탈락: base 대비 −2판이면 탈락 (경계 포함 — 계획 규칙 문언 그대로, Gate2 높음#11)
        survivors = [b for b in betas if wins_by_beta[b] > base_total - DOWNSIDE_GAMES]
        all_red = all(wins_by_beta[b] <= base_total - RED_FLAG_GAMES for b in betas)
        if all_red:
            sel, reason = None, "red_flag"
            red_flag = True
        elif not survivors:
            # 전 β 하방 탈락 (red flag 미달) — 임의 부활 금지, 사용자 결정 gate
            sel, reason = None, "no_survivor_user_gate"
        else:
            best = max(survivors, key=lambda b: wins_by_beta[b])
            ties = [b for b in survivors if wins_by_beta[b] == wins_by_beta[best]]
            sel = "0.1" if "0.1" in ties else best
            reason = "tie->0.1" if len(ties) > 1 else "argmax"
        result["arms"][arm] = {"wins": wins_by_beta, "per_cell": detail,
                               "survivors": survivors, "selected_beta": sel, "reason": reason}
        print(f"[beta-decide] {args.task}/{arm}: base={base_total} wins={wins_by_beta} "
              f"-> β={sel} ({reason})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if red_flag:
        print("[beta-decide] RED FLAG — 해당 arm 중단·사용자 결정 필요", file=sys.stderr)
        sys.exit(6)


if __name__ == "__main__":
    main()
