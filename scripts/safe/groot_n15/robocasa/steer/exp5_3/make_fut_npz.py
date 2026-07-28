#!/usr/bin/env python3
"""permanent registry → future-only 사본: alpha0_seg_mask [1,1,1] → [0,1,0].
방향·setpoint 불변(재fit 불요) — state/action 세그먼트 delta 만 0 게인.
근거: β=1.0 full-mask 는 육안 떨림 + SR 붕괴; exp4-1 프리뷰에서 future_only 가
permanent 대비 Δjerk −0.14~−0.15 (더 부드러운 개입).
"""
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1])  # .../deploy/permanent
dst = Path(sys.argv[2])  # .../deploy/permanent_fut
n = 0
for npz in sorted(src.rglob("conceptors.npz")):
    z = dict(np.load(npz))
    z["alpha0_seg_mask"] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    out = dst / npz.relative_to(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **z)
    n += 1
print(f"[done] {n} NPZ → {dst} (mask [0,1,0])")
