#!/usr/bin/env python3
"""기존 eval/수집 mp4 에 instruction + step별 phase 상단 배너를 post-hoc 으로 입힌다.

eval 사이드카(json)의 ``phase_timeline``(env-step별, --proximity-phases 산출)과
``canonical_instruction``/``task_description`` 을 읽어, 프레임 i → env step
(i × steps_per_render) → phase 로 매핑해 상단 추가 배너(장면 무가림)에 그린다.
원본은 보존하고 ``<stem>--phase.mp4`` 를 새로 쓴다.

주의: 원격(48/50) 렌더분처럼 하단 캡션이 프레임에 burn-in 된 영상은 그 부분을
지울 수 없다 — 상단 배너만 추가된다. 진짜 클린 재생성은 episode 재실행 필요.

사용 (robocasa 컨테이너 — cv2/numpy 필요):
  python annotate_phase_video.py --sidecar <task..--epN--succX.json> [--steps-per-render 2]
  python annotate_phase_video.py --root <e1/cell/arm 트리> [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# phase → BGR (7-phase drawer + 6-phase ppcc 공통 팔레트, 미등록은 흰색)
PHASE_COLORS = {
    "reach-to-handle": (80, 200, 255), "grasp-handle": (0, 255, 128),
    "pull": (255, 180, 0), "disengage": (0, 80, 255), "push-back": (200, 0, 200),
    "wrong-grasp": (0, 0, 255), "open-done": (0, 255, 0),
    "reach-to-object": (80, 200, 255), "grasp": (0, 255, 128),
    "transport": (255, 180, 0), "place": (255, 120, 200),
    "insert-settle": (0, 255, 0),
}

# env-step GT 이벤트 → (라벨, BGR). 구 사이드카는 키 부재 → 빈 리스트로 no-op.
EVENT_STYLES = {
    "env_step_grasp_steps": ("GRASP", (0, 255, 0)),
    "env_step_drop_steps": ("DROP", (0, 0, 255)),
    "env_step_wrong_grasp_steps": ("WRONG", (0, 165, 255)),
}


def annotate(sidecar: Path, steps_per_render: int, overwrite: bool = False) -> Path | None:
    d = json.loads(sidecar.read_text())
    # env-step 해상도 GT(env_step_phases, 720+1) 우선 — phase_timeline 은 record(5 env-step)당 1개
    timeline = d.get("env_step_phases") or []
    step_div = 1
    if not timeline:
        timeline = d.get("phase_timeline") or []
        step_div = int(d.get("n_action_steps") or 5)  # frame→record 매핑
    instr = d.get("canonical_instruction") or d.get("task_description") or ""
    # 이벤트 마커 (env-step 축) — t0 주석 시 grasp/drop/wrong 통과 지점 시각화
    events = sorted(
        (int(e), label, color)
        for key, (label, color) in EVENT_STYLES.items()
        for e in (d.get(key) or [])
    )
    if not timeline:
        print(f"[skip] phase 라벨 없음: {sidecar.name}")
        return None
    mp4 = sidecar.with_suffix(".mp4")
    if not mp4.exists():
        print(f"[skip] mp4 없음: {mp4.name}")
        return None
    out = mp4.with_name(mp4.stem + "--phase.mp4")
    if out.exists() and not overwrite:
        print(f"[skip] 이미 존재: {out.name}")
        return out

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 6
    line_h = 26
    bar_h = 6  # 하단 이벤트 틱 진행바
    banner_h = 2 * line_h + 3 * pad + bar_h
    if (banner_h + H) % 2 == 1:  # h264/yuv420p 짝수 높이
        banner_h += 1
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + banner_h))

    i = 0
    succ = d.get("episode_success")
    # 진행바 x 좌표용 총 env-step 수 (record 해상도 fallback 시 근사)
    n_total = max(1, (len(timeline) if step_div == 1 else len(timeline) * step_div) - 1)
    flash: list | None = None  # [text, color, frames_left]
    flash_frames = max(6, int(round((cap.get(cv2.CAP_PROP_FPS) or 20.0) * 0.5)))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        env_step = i * steps_per_render
        step = min(env_step // step_div, len(timeline) - 1)
        ph = timeline[step]
        banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
        cv2.putText(banner, f"{instr} (succ={int(bool(succ))})"[:90], (pad, pad + line_h - 8),
                    font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, f"step {step:>3}  phase: {ph}", (pad, 2 * pad + 2 * line_h - 10),
                    font, 0.65, PHASE_COLORS.get(ph, (255, 255, 255)), 2, cv2.LINE_AA)
        # 렌더는 steps_per_render 간격 서브샘플 — [env_step, env_step+spr) 윈도우 매칭
        hits = [ev for ev in events if env_step <= ev[0] < env_step + steps_per_render]
        if hits:
            text = "  ".join(f"{label}@{e}" for e, label, _ in hits)
            flash = [text, hits[-1][2], flash_frames]
        if flash and flash[2] > 0:
            tw = cv2.getTextSize(flash[0], font, 0.6, 2)[0][0]
            cv2.putText(banner, flash[0], (max(pad, W - tw - pad), pad + line_h - 8),
                        font, 0.6, flash[1], 2, cv2.LINE_AA)
            flash[2] -= 1
        # 하단 진행바: 전체 타임라인 위 이벤트 고정 틱 + 현재 위치(흰색)
        y0 = banner_h - bar_h
        cv2.rectangle(banner, (0, y0), (W - 1, banner_h - 1), (60, 60, 60), -1)
        for e, _, color in events:
            x = int(min(e, n_total) / n_total * (W - 1))
            cv2.line(banner, (x, y0), (x, banner_h - 1), color, 2)
        x_now = int(min(env_step, n_total) / n_total * (W - 1))
        cv2.line(banner, (x_now, y0 - 1), (x_now, banner_h - 1), (255, 255, 255), 1)
        writer.write(np.concatenate([banner, frame], axis=0))
        i += 1
    cap.release()
    writer.release()
    print(f"[wrote] {out.name} frames={i} timeline={len(timeline)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None, help="트리 일괄 (succ*.json 스캔)")
    ap.add_argument("--steps-per-render", type=int, default=2, help="수집 시 --steps-per-render 값")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    targets = []
    if args.sidecar:
        targets = [args.sidecar]
    elif args.root:
        targets = sorted(args.root.rglob("*--succ*.json"))
        if args.limit:
            targets = targets[: args.limit]
    else:
        raise SystemExit("--sidecar 또는 --root 필요")
    n = sum(1 for t in targets if annotate(t, args.steps_per_render, args.overwrite))
    print(f"[done] {n}/{len(targets)}")


if __name__ == "__main__":
    main()
