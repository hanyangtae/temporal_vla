#!/usr/bin/env python
"""detector 발화 오버레이 영상 렌더 (docs/steering/43 후속 B-2).

`export_fire_scores.py` 의 JSON + 수집 영상(video.mp4) → 발화 시점을 눈으로 볼 수 있는
mp4. 상단 배너(task/instruction/scene·noise/실제 라벨/절제 mode), 좌측 패널에 실패
score 곡선 + CP 밴드 δ_t, 발화 이후 프레임 전체 빨간 테두리.

## ★ frame ↔ record 정렬 근거 (코드 실측)

수집 인자 (`scripts/safe/groot_n15/robocasa/collect/collect_grid.sh:255-256`):
`--n-action-steps 5 --video-fps 20 --steps-per-render 2`.

- MultiStepWrapper: `get_action` 1회 = feature record 1개 = env-step 5회 실행.
  → record r 은 0-based env-step [5r, 5r+5) 구간을 담당하고, 그 feature 는 env-step
  5r 직전 관측에서 뽑힌다.
- `VideoRecordingWrapper` (Isaac-GR00T `gr00t/eval/sim/wrapper/video_recording_wrapper.py`):
  `reset()` 에서 `step_count = 1`, `step()` 에서 **먼저** `step_count += 1` 한 뒤
  `step_count % steps_per_render == 0` 일 때 그 step 의 관측을 프레임으로 쓴다.
  steps_per_render=2 → step_count 짝수, 즉 1-based n번째 env-step 중 n 이 홀수일 때
  기록. 0-based 로는 env-step s = n-1 = 0, 2, 4, ... 이 프레임 0, 1, 2, ...
  → **frame f ↔ 0-based env-step s = 2f**.
- 두 식을 합치면 record r 의 담당 구간은 frame [ceil(5r/2), ceil(5(r+1)/2)) 이고,
  역방향은 **r(f) = floor(2f/5)** (= floor(s/5)).
- 자체 점검: 총 프레임 수 ≈ ceil(5T/2) (T = record 수). `--check-align` 이 실제
  프레임 수와 이 기대값의 차이를 출력하고, 허용 오차를 넘으면 fail-loud.

## 사용
    ~/miniconda3/envs/lerobot_safe/bin/python \
        scripts/analysis/grid_phase/render_fire_overlay.py \
        --json outputs/analysis/grid_phase/fire_videos/fire_scores.json \
        --video-root outputs/analysis/grid_phase/fire_videos/videos \
        --out-dir outputs/analysis/grid_phase/fire_videos
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

PANEL_W = 300          # 좌측 score 패널 폭
BANNER_H = 78          # 상단 배너 높이
BORDER = 14            # 발화 후 빨간 테두리 두께
FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_BG = (24, 24, 24)
COL_SCORE = (80, 220, 90)      # BGR: score = 초록
COL_BAND = (60, 160, 250)      # 밴드 δ_t = 주황
COL_FIRE = (60, 60, 235)       # 발화 = 빨강


def record_of_frame(f: int, T: int, steps_per_render: int, n_action_steps: int) -> int:
    """frame index → record index (상단 주석의 r(f) = floor(2f/5) 일반형)."""
    s = f * steps_per_render                     # 0-based env-step
    return min(T - 1, s // n_action_steps)


def expected_frames(T: int, steps_per_render: int, n_action_steps: int) -> int:
    return math.ceil(T * n_action_steps / steps_per_render)


def draw_panel(h: int, scores, band, t_now: int, t_fire, T: int) -> np.ndarray:
    """현재 시점까지의 score 곡선 + 밴드. 세로축 = score(0~1), 가로축 = record t."""
    pad_l, pad_r, pad_t, pad_b = 44, 12, 26, 30
    panel = np.full((h, PANEL_W, 3), COL_BG, np.uint8)
    x0, x1 = pad_l, PANEL_W - pad_r
    y0, y1 = pad_t, h - pad_b
    vmax = max(1e-6, float(max(max(scores), max(band))) * 1.08)

    cv2.rectangle(panel, (x0, y0), (x1, y1), (90, 90, 90), 1)
    for frac in (0.0, 0.5, 1.0):
        y = int(y1 - frac * (y1 - y0))
        cv2.line(panel, (x0, y), (x1, y), (55, 55, 55), 1)
        cv2.putText(panel, f"{frac*vmax:.2f}", (4, y + 4), FONT, 0.36, (170, 170, 170), 1,
                    cv2.LINE_AA)

    def pt(t, v):
        x = x0 + int((t / max(T - 1, 1)) * (x1 - x0))
        y = y1 - int(min(v / vmax, 1.0) * (y1 - y0))
        return x, y

    for t in range(1, T):                                   # 밴드는 전체 구간 표시
        cv2.line(panel, pt(t - 1, band[t - 1]), pt(t, band[t]), COL_BAND, 1, cv2.LINE_AA)
    for t in range(1, t_now + 1):                           # score 는 현재까지만
        cv2.line(panel, pt(t - 1, scores[t - 1]), pt(t, scores[t]), COL_SCORE, 2,
                 cv2.LINE_AA)
    cv2.circle(panel, pt(t_now, scores[t_now]), 3, (255, 255, 255), -1)
    if t_fire is not None:
        x, _ = pt(t_fire, scores[t_fire])
        cv2.line(panel, (x, y0), (x, y1), COL_FIRE, 1, cv2.LINE_AA)
        if t_now >= t_fire:
            cv2.circle(panel, pt(t_fire, scores[t_fire]), 5, COL_FIRE, 2)

    cv2.putText(panel, "failure score", (x0, 16), FONT, 0.42, COL_SCORE, 1, cv2.LINE_AA)
    cv2.putText(panel, "CP band", (x0 + 120, 16), FONT, 0.42, COL_BAND, 1, cv2.LINE_AA)
    cv2.putText(panel, f"t={t_now}/{T-1}  s={scores[t_now]:.3f}", (x0, h - 12), FONT,
                0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def render_one(rec: dict, mode: str, video_path: Path, out_path: Path, collect: dict,
               fps: int, check_align: bool, align_tol: int) -> dict:
    md = rec["modes"][mode]
    scores = [float(v) for v in md["scores"]]
    band = [float(v) for v in md["band"]]
    T = int(rec["T"])
    if len(scores) != T or len(band) != T:
        raise SystemExit(f"{rec['task']} ep{rec['ep_id']}: score/band 길이 != T")
    t_fire = md["t_fire"]
    spr = int(collect["steps_per_render"])
    nas = int(collect["n_action_steps"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    exp = expected_frames(T, spr, nas)
    align = {"n_frames": n_frames, "expected_frames": exp, "diff": n_frames - exp}
    if check_align and abs(n_frames - exp) > align_tol:
        raise SystemExit(
            f"{rec['task']} ep{rec['ep_id']}: 프레임 정렬 불일치 — 실제 {n_frames} vs "
            f"기대 {exp} (T={T}, spr={spr}, n_action_steps={nas})")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"프레임 0 읽기 실패: {video_path}")
    fh, fw = frame.shape[:2]
    out_w, out_h = fw + PANEL_W, fh + BANNER_H
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter 열기 실패: {out_path}")

    lab = "SUCCESS" if rec["succ"] else "FAILURE"
    phase_names = rec.get("phase_names", {})
    f_idx = 0
    while ok:
        t = record_of_frame(f_idx, T, spr, nas)
        canvas = np.full((out_h, out_w, 3), COL_BG, np.uint8)
        canvas[BANNER_H:, PANEL_W:] = frame
        canvas[BANNER_H:, :PANEL_W] = draw_panel(fh, scores, band, t, t_fire, T)

        fired = t_fire is not None and t >= t_fire
        if fired:
            cv2.rectangle(canvas, (0, 0), (out_w - 1, out_h - 1), COL_FIRE, BORDER)

        ph = phase_names.get(str(rec["phase_code"][t]), str(rec["phase_code"][t]))
        line1 = f"{rec['task']}  |  {rec['instruction']}  |  s{rec['scene']}n{rec['noise']}"
        line2 = (f"GT: {lab}   mode={mode}   t_fire="
                 f"{'-' if t_fire is None else t_fire}/{T-1}   t={t}  phase={ph}")
        cv2.putText(canvas, line1, (14, 28), FONT, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (14, 58), FONT, 0.56,
                    COL_FIRE if fired else (200, 200, 200), 1, cv2.LINE_AA)
        if fired:
            cv2.putText(canvas, "FIRED", (out_w - 110, 30), FONT, 0.8, COL_FIRE, 2,
                        cv2.LINE_AA)
        writer.write(canvas)
        ok, frame = cap.read()
        f_idx += 1

    writer.release()
    cap.release()
    align["written"] = f_idx
    return align


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--video-root", required=True,
                    help="JSON 의 video 상대경로 기준 루트 (grid 루트에서 pull 한 사본)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default=None, help="기본 = JSON config.modes 전부")
    ap.add_argument("--only", default=None,
                    help="task 필터 (콤마 구분, 부분 일치)")
    ap.add_argument("--fps", type=int, default=None, help="기본 = 수집 video_fps")
    ap.add_argument("--check-align", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--align-tol", type=int, default=2)
    args = ap.parse_args()

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    collect = payload["config"]["collect"]
    modes = ([m.strip() for m in args.modes.split(",")] if args.modes
             else list(payload["config"]["modes"]))
    fps = args.fps or int(collect["video_fps"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    made = []
    for rec in payload["episodes"]:
        if only and not any(o in rec["task"] for o in only):
            continue
        vpath = Path(args.video_root) / rec["video"]
        if not vpath.exists():
            print(f"[skip] 영상 없음: {vpath}")
            continue
        for mode in modes:
            tag = "succ" if rec["succ"] else "fail"
            name = (f"{rec['task']}_s{rec['scene']}n{rec['noise']}_{tag}_{mode}.mp4")
            outp = out_dir / name
            align = render_one(rec, mode, vpath, outp, collect, fps,
                               args.check_align, args.align_tol)
            tf = rec["modes"][mode]["t_fire"]
            print(f"[render] {name}  T={rec['T']} t_fire={tf} "
                  f"frames={align['written']} (기대 {align['expected_frames']}, "
                  f"diff {align['diff']})")
            made.append({"file": name, "task": rec["task"], "mode": mode,
                         "t_fire": tf, "T": rec["T"], "succ": rec["succ"], **align})
    (out_dir / "render_index.json").write_text(
        json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[render] {len(made)} 개 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
