#!/usr/bin/env python3
"""grid 아카이브 video.mp4 무결성 스캔 — 렌더 정지(freeze)·노이즈 프레임 검출.

배경(2026-09-03): CoffeeSetupMug s0/k1 5셀에서 영상이 ~4.7s 뒤 멈추고 일부 노이즈. 같은 셀의
pkl 에서 VL(goal) hidden 의 record 간 Δnorm 이 그 시점부터 상수로 고정 → 정책이 받은 관측
이미지도 깨진 것(영상만의 문제가 아님). 이 스크립트는 영상 지표로 전 셀을 1차 선별한다.

출력 TSV 열: rel_path frames fps first_freeze_s frozen_frac noisy_frac lap_med lap_max flag
  frozen_frac = 연속 프레임 평균 절대차 < 0.05 인 비율, first_freeze_s = 1초 이상 연속 정지 첫 시각
  noisy_frac = Laplacian 분산이 초반 20프레임 중앙값의 5배 초과인 비율
  flag = FREEZE(first_freeze_s 존재) | NOISE(noisy_frac>0) | OK
사용: scan_video_integrity.py --grid-root <root>/<plan_id> --out scan.tsv [--procs 8]
"""
from __future__ import annotations
import argparse, csv, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np

def analyze(args):
    root, path = args
    try:
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        prev = None; diffs = []; lap = []
        while True:
            ok, f = cap.read()
            if not ok: break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None: diffs.append(float(np.abs(g - prev).mean()))
            lap.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
            prev = g
        d = np.array(diffs); n = np.array(lap)
        if len(n) == 0:
            return dict(rel_path=str(path.parent.relative_to(root)), frames=0, flag="EMPTY")
        frozen = d < 0.05; first = None; w = max(int(fps), 1)
        for i in range(max(len(d) - w, 0)):
            if frozen[i:i + w].all(): first = round(i / fps, 2); break
        base = np.median(n[:20]) if len(n) >= 20 else np.median(n)
        noisy = float((n > 5 * base).mean()) if base > 0 else 0.0
        flag = "FREEZE" if first is not None else ("NOISE" if noisy > 0 else "OK")
        if first is not None and noisy > 0: flag = "FREEZE+NOISE"
        return dict(rel_path=str(path.parent.relative_to(root)), frames=len(n), fps=fps,
                    first_freeze_s=first, frozen_frac=round(float(frozen.mean()), 3),
                    noisy_frac=round(noisy, 3), lap_med=round(float(np.median(n)), 0),
                    lap_max=round(float(n.max()), 0), flag=flag)
    except Exception as e:  # noqa: BLE001
        return dict(rel_path=str(path.parent.relative_to(root)), frames=-1, flag=f"ERR:{e}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    vids = sorted(a.grid_root.rglob("video.mp4"))
    print(f"videos: {len(vids)}", flush=True)
    cols = ["rel_path", "frames", "fps", "first_freeze_s", "frozen_frac", "noisy_frac", "lap_med", "lap_max", "flag"]
    with Pool(a.procs) as pool, a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore"); w.writeheader()
        for i, r in enumerate(pool.imap_unordered(analyze, [(a.grid_root, v) for v in vids], chunksize=4), 1):
            w.writerow(r); f.flush()
            if i % 100 == 0: print(f"  {i}/{len(vids)}", flush=True)
    rows = list(csv.DictReader(a.out.open(), delimiter="\t"))
    from collections import Counter
    print("flag 분포:", dict(Counter(r["flag"] for r in rows)))
    bad = [r for r in rows if r["flag"] != "OK"]
    for r in bad[:40]: print("  ", r["rel_path"], r["flag"], "freeze@", r["first_freeze_s"], "frozen", r["frozen_frac"], "noisy", r["noisy_frac"])
    if len(bad) > 40: print("   ... 외", len(bad) - 40)

if __name__ == "__main__":
    main()
