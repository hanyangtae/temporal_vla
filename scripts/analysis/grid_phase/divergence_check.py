"""개입 arm vs base 영상 프레임 발산 분석 (replay 결정성 이용).

같은 (env_seed, inference_seed) 셀이므로 발산 = 개입의 인과 효과.
per-frame mean|Δpixel| 이 임계(TH) 넘는 첫 프레임 → record 로 환산해
trigger record·개입 ON record 와 대조한다.
"""
import glob
import json
import sys

import cv2
import numpy as np

BASE = "/temporal_vla/.claude/worktrees/grid-phase-sep/outputs/eval/robocasa/groot_n15"
ARM_DIR = sys.argv[1]          # 예: og_condg_d0_b10_merged/OpenDrawer_left/online
REF_DIR = sys.argv[2]          # 예: online_gated_replay_v2/OpenDrawer_left/base
N_EP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SPR, NAS, TH = 2, 5, 2.0       # steps_per_render, n_action_steps, 발산 임계(mean abs)


def vid(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f.astype(np.int16))
    cap.release()
    return fr


def find_mp4(root, ep):
    h = glob.glob(f"{root}/raw_rollouts/*/*/task0--ep{ep}--succ*.mp4")
    return h[0] if h else None


eps = sorted(int(f.split("--ep")[1].split("--")[0])
             for f in glob.glob(f"{BASE}/{ARM_DIR}/raw_rollouts/*/*/task0--ep*--succ0.json"))
print(f"# fail-결과 ep {len(eps)}개 중 앞 {N_EP}개 분석 (arm={ARM_DIR.split('/')[-1]})")
rows = []
for ep in eps[:N_EP]:
    a, b = find_mp4(f"{BASE}/{ARM_DIR}", ep), find_mp4(f"{BASE}/{REF_DIR}", ep)
    if not a or not b:
        continue
    js = glob.glob(f"{BASE}/{ARM_DIR}/raw_rollouts/*/*/task0--ep{ep}--succ*.json")[0]
    d = json.load(open(js))
    trig = d.get("trigger_step")
    g = d.get("phase_gated_flags") or []
    on = [i for i, x in enumerate(g) if x]
    fa, fb = vid(a), vid(b)
    n = min(len(fa), len(fb))
    diffs = np.array([np.abs(fa[i] - fb[i]).mean() for i in range(n)])
    div = np.nonzero(diffs > TH)[0]
    first = int(div[0]) if len(div) else None
    first_rec = None if first is None else (first * SPR) // NAS
    rows.append((ep, trig, (on[0] if on else None), len(on), first_rec,
                 float(diffs.max()), len(fa), len(fb)))
    o0 = on[0] if on else "-"
    print(f"ep{ep}: trig={trig} 개입ON첫={o0}(총{len(on)}) → 발산첫record={first_rec} "
          f"maxdiff={diffs.max():.1f} len(arm/base)={len(fa)}/{len(fb)}")
nd = [r for r in rows if r[4] is None]
print(f"\n요약: 분석 {len(rows)}판, 발산 없음 {len(nd)}판, "
      f"발산 있음 {len(rows)-len(nd)}판")
