#!/usr/bin/env python3
"""eval 영상 후처리 주석 — 프레임을 가리지 않도록 **상단 여백을 추가로 확장**해 그린다.

배치: 기존 instruction 배너(top) 아래에 상태 스트립을 삽입:
  왼쪽  = 실패 예측값 (failure_scores[record], 발화 시 FIRED 표시·빨강)
  오른쪽 = 현재 phase | 개입 상태 (OFF / ON: <op> β<beta> @<적용 phase> / identity)

프레임↔record 매핑: frame f → env_step = f*steps_per_render → record = env_step // n_action_steps.
sidecar(json)의 feature_phases / phase_gated_flags / failure_scores / trigger_step /
serve_steering / serve_failure_detector 를 사용. 필드 없는 arm(oracle 등)은 있는 것만 그림.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STRIP_H = 34
FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_one(mp4: Path, sidecar: Path, out: Path, steps_per_render: int,
                 banner_h: int = 62) -> None:
    d = json.loads(sidecar.read_text())
    nas = int(d.get("n_action_steps", 5))
    phases = d.get("feature_phases") or []
    gated = d.get("phase_gated_flags") or []
    scores = d.get("failure_scores") or []
    trig = d.get("trigger_step")
    steer = d.get("serve_steering") or {}
    op = steer.get("op", "")
    beta = steer.get("beta", "")
    reg = set(steer.get("phases") or [])

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out.parent.mkdir(parents=True, exist_ok=True)
    total_h = h + STRIP_H + (1 if (h + STRIP_H) % 2 else 0)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, total_h))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = (f * steps_per_render) // nas
        strip = np.zeros((total_h - h, w, 3), dtype=np.uint8)
        strip[:] = (28, 28, 28)
        # 왼쪽: 실패 예측값
        if rec < len(scores):
            fired = trig is not None and rec >= int(trig)
            col = (0, 0, 255) if fired else (0, 255, 128)
            txt = f"fail p={scores[rec]:.2f}" + ("  FIRED" if fired else "")
            cv2.putText(strip, txt, (6, STRIP_H - 12), FONT, 0.55, col, 1, cv2.LINE_AA)
        # 오른쪽: phase | 개입 상태
        ph = phases[rec] if rec < len(phases) else ""
        if rec < len(gated) and gated[rec]:
            mode = f"ON {op} b{beta} @{ph}"
            col = (0, 200, 255)
        elif trig is not None and rec >= int(trig):
            mode = f"latched ({ph}: {'identity' if ph not in reg else 'pending'})"
            col = (0, 200, 255)
        else:
            mode = "OFF"
            col = (180, 180, 180)
        txt = f"phase: {ph}  |  {mode}"
        size = cv2.getTextSize(txt, FONT, 0.55, 1)[0]
        cv2.putText(strip, txt, (max(6, w - size[0] - 6), STRIP_H - 12),
                    FONT, 0.55, col, 1, cv2.LINE_AA)
        # instruction 배너(top banner_h) 아래에 상태 스트립 삽입 — 장면 무손상
        vw.write(np.concatenate(
            [frame[:banner_h], strip, frame[banner_h:]], axis=0))
        f += 1
    vw.release()
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", type=Path, required=True, help="<out>/<slug>/<arm>")
    ap.add_argument("--out-sub", default="annotated", help="job-dir 하위 출력 디렉토리명")
    ap.add_argument("--steps-per-render", type=int, default=2)
    ap.add_argument("--banner-h", type=int, default=62,
                    help="기존 instruction 상단 배너 높이 (그 아래에 상태 스트립 삽입; "
                         "구버전 오버레이 영상은 0)")
    args = ap.parse_args()
    mp4s = sorted(args.job_dir.glob("raw_rollouts/*/*/task0--ep*--succ*.mp4"))
    if not mp4s:
        raise SystemExit(f"영상 없음: {args.job_dir}")
    n = 0
    for m in mp4s:
        sc = m.with_suffix(".json")
        if not sc.exists():
            print(f"[skip] sidecar 없음: {m.name}")
            continue
        annotate_one(m, sc, args.job_dir / args.out_sub / m.name, args.steps_per_render, args.banner_h)
        n += 1
    print(f"[annotate] {args.job_dir} → {args.out_sub}/ {n}개")


if __name__ == "__main__":
    main()
