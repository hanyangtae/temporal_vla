#!/usr/bin/env python
"""detector 발화 기전 분해 — dwell-OOD vs 내용 신호.

phase-gt 절제 detector의 조기 발화가 (a) "현재 phase dwell이 학습 cap을 초과한 직후"
(= 학습 지지집합 이탈, dwell 신호 재활용)인지 (b) cap 이내 발화(= 내용 신호)인지 분류.

입력: failure_detector_sim.py 의 sim_detail.json (+ segA shard 의 phase_code 시퀀스).
발화한 실패 test episode 만 대상. dwell_at_fire = t_fire 까지의 현재-phase record 누적 수
(cap 정의와 동일하게 비연속 합산).

사용:
  python fire_phase_decomp.py --shard-dir <segA> --details run1/sim_detail.json run2/... \
      --arm pertask --model lstm --alpha 0.2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_phase_seqs(shard_dir: Path, slug: str):
    """slug -> {ep_id: phase_code 시퀀스(rec_idx 순)}"""
    z = np.load(shard_dir / f"{slug}.npz", allow_pickle=True)
    ep = z["ep_id"]
    rec = z["rec_idx"]
    ph = z["phase_code"]
    out = {}
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(rec[m])
        out[int(e)] = ph[m][order]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--details", nargs="+", required=True,
                    help="sim_detail.json 경로들 (라벨은 상위 디렉토리명: none/rollout/phase-gt)")
    ap.add_argument("--arm", default="pertask")
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    seq_cache: dict[str, dict] = {}
    # (mode, task) -> list of (dwell_at_fire, cap, over)
    agg = defaultdict(list)
    for dp in args.details:
        dp = Path(dp)
        mode = dp.parent.name
        d = json.loads(dp.read_text())
        caps_all = d["config"].get("truncate_caps", {})
        for r in d["episodes"]:
            if (r["arm"] != args.arm or r["model"] != args.model
                    or abs(r["alpha"] - args.alpha) > 1e-9):
                continue
            if r["succ"] == 1 or not r["fired"] or r["t_fire"] is None:
                continue
            task = r["task"]
            if task not in seq_cache:
                seq_cache[task] = load_phase_seqs(args.shard_dir, task)
            seq = seq_cache[task].get(r["ep_id"])
            if seq is None:
                continue
            t = int(r["t_fire"])
            if t >= len(seq):
                t = len(seq) - 1
            cur = int(seq[t])
            dwell = int(np.sum(seq[: t + 1] == cur))
            cap = caps_all.get(task, {}).get("phase_caps", {}).get(str(cur))
            # none run 도 caps 는 기록됨(항상 계산). cap 없는 phase(성공에 없음)는 별도 표기.
            agg[(mode, task)].append((dwell, cap, cur))

    modes = sorted({m for m, _ in agg})
    tasks = sorted({t for _, t in agg})
    print(f"arm={args.arm} model={args.model} alpha={args.alpha}")
    print(f"{'task':<20}" + "".join(f"{m:>34}" for m in modes))
    print(" " * 20 + "".join(f"{'n  overcap%  med(dwell/cap)':>34}" for _ in modes))
    for t in tasks:
        line = f"{t:<20}"
        for m in modes:
            v = agg.get((m, t), [])
            if not v:
                line += f"{'—':>34}"
                continue
            with_cap = [(dw, c) for dw, c, _ in v if c]
            nocap = len(v) - len(with_cap)
            over = [dw > c for dw, c in with_cap]
            ratio = [dw / c for dw, c in with_cap]
            cell = (f"n{len(v)}"
                    + (f" over{100 * np.mean(over):.0f}%" if over else "")
                    + (f" r{np.median(ratio):.2f}" if ratio else "")
                    + (f" nocap{nocap}" if nocap else ""))
            line += f"{cell:>34}"
        print(line)
    # 전체 합산
    print()
    for m in modes:
        v = [x for (mm, _), xs in agg.items() if mm == m for x in xs]
        with_cap = [(dw, c) for dw, c, _ in v if c]
        over = [dw > c for dw, c in with_cap]
        ratio = [dw / c for dw, c in with_cap]
        print(f"[{m}] fired-fail n={len(v)}  cap초과 발화 {100 * np.mean(over):.0f}%  "
              f"median dwell/cap {np.median(ratio):.2f}  cap없는 phase 발화 {len(v) - len(with_cap)}")


if __name__ == "__main__":
    main()
