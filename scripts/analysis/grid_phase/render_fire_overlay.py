#!/usr/bin/env python
"""detector 발화 오버레이 영상 렌더 (docs/steering/43 후속 B-2).

`export_fire_scores.py` 의 JSON + 수집 영상 → 발화 시점을 눈으로 볼 수 있는 mp4.

## ★ 레이아웃 원칙 (재제작 사유)

**장면 픽셀 위에는 어떤 텍스트·도형도 그리지 않는다.** 이전 판은 수집 영상 일부(kanu
머신분)가 캡션을 장면 하단에 구워 넣은 상태였고, 오버레이까지 겹쳐 로봇이 가렸다.
지금은 캔버스를 밖으로 확장한다:

    ┌───────────────────────────────────────┐ ← 여백(MARGIN) 안에만 빨간 테두리
    │ 배너 (task / instruction / GT / mode) │  ← 장면 밖 (위)
    ├──────────┬────────────────────────────┤
    │ score    │                            │
    │ 패널     │   장면 프레임 (무손상)     │
    │ (밖, 좌) │                            │
    └──────────┴────────────────────────────┘
    발화 이후 빨간 굵은 선은 **바깥 여백 안에서만** 그려진다. 테두리 두께 = 여백 폭이라
    장면·패널·배너 어느 것도 덮지 않는다.

입력 영상은 두 종류다 (`--source-mode auto` 가 높이로 판별):
- 높이 > 256: 프레임 **위**에 캡션 스트립이 덧붙은 판본 → 위 (h-256) 행을 잘라내
  클린 256px 장면만 사용. 장면 무손상.
- 높이 == 256: 캡션이 장면 하단에 구워진 판본 → crop 불가. `replay_clean_video.py`
  로 재생성한 클린 영상(`--clean-root/<stem>.mp4`)을 대신 쓴다. 없으면 skip.

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
        --clean-root outputs/analysis/grid_phase/fire_videos/clean_videos \
        --out-dir outputs/analysis/grid_phase/fire_videos
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SCENE_H = 256          # 수집 카메라 타일 높이 (768x256 = res256 3-view 가로 concat)
PANEL_W = 300          # 좌측 score 패널 폭 (장면 밖)
BANNER_H = 84          # 상단 배너 높이 (장면 밖)
BORDER = 12            # 발화 후 테두리 두께 (= 바깥 여백 폭, 장면을 덮지 않게)
MARGIN = BORDER + 2    # 캔버스 사방 여백 — 테두리 전용 영역 (테두리보다 넉넉하게)
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


def resolve_source(rec: dict, video_root: Path, clean_root: Path | None,
                   stem: str) -> tuple[Path | None, str, int, str]:
    """(영상 경로, 처리방식, crop_top, 사유). 장면을 가리는 소스는 절대 쓰지 않는다."""
    raw = video_root / rec["video"]
    if not raw.exists():
        return None, "missing", 0, f"원본 없음: {raw}"
    cap = cv2.VideoCapture(str(raw))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if h > SCENE_H:
        # 캡션이 장면 **위** 스트립에 있는 판본 → 스트립만 잘라내면 장면 무손상.
        return raw, "crop", h - SCENE_H, f"상단 스트립 {h - SCENE_H}px crop"
    # 높이 == 256: 캡션이 장면 하단에 구워짐 → replay 로 재생성한 클린 영상만 허용.
    if clean_root is not None:
        clean = clean_root / f"{stem}.mp4"
        if clean.exists():
            return clean, "replay", 0, "burn-in 원본 → replay 클린 영상 사용"
    return None, "blocked", 0, ("하단 burn-in 캡션 원본인데 클린 replay 영상이 없음 "
                                f"({clean_root}/{stem}.mp4)")


def render_one(rec: dict, mode: str, video_path: Path, crop_top: int, out_path: Path,
               collect: dict, fps: int, check_align: bool, align_tol: int) -> dict:
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
    frame = frame[crop_top:]
    fh, fw = frame.shape[:2]
    out_w, out_h = MARGIN * 2 + PANEL_W + fw, MARGIN * 2 + BANNER_H + fh
    sx, sy = MARGIN + PANEL_W, MARGIN + BANNER_H     # 장면 좌상단
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
        # 장면은 **그대로** 붙인다 (이후 어떤 그리기도 이 영역 밖에서만 일어난다).
        canvas[sy:sy + fh, sx:sx + fw] = frame
        canvas[sy:sy + fh, MARGIN:MARGIN + PANEL_W] = draw_panel(
            fh, scores, band, t, t_fire, T)

        fired = t_fire is not None and t >= t_fire
        ph = phase_names.get(str(rec["phase_code"][t]), str(rec["phase_code"][t]))
        line1 = f"{rec['task']}  |  {rec['instruction']}  |  s{rec['scene']}n{rec['noise']}"
        line2 = (f"GT: {lab}   mode={mode}   t_fire="
                 f"{'-' if t_fire is None else t_fire}/{T-1}   t={t}  phase={ph}")
        cv2.putText(canvas, line1, (MARGIN + 14, MARGIN + 30), FONT, 0.62,
                    (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (MARGIN + 14, MARGIN + 62), FONT, 0.56,
                    COL_FIRE if fired else (200, 200, 200), 1, cv2.LINE_AA)
        if fired:
            cv2.putText(canvas, "FIRED", (out_w - MARGIN - 118, MARGIN + 34), FONT, 0.8,
                        COL_FIRE, 2, cv2.LINE_AA)
            # 테두리는 여백(MARGIN) 안에서만 — 두께 BORDER, 중심선 BORDER/2 → 0..BORDER px.
            cv2.rectangle(canvas, (BORDER // 2, BORDER // 2),
                          (out_w - 1 - BORDER // 2, out_h - 1 - BORDER // 2),
                          COL_FIRE, BORDER)
        # 불변식: 장면 영역은 원본과 **픽셀 단위로 동일**해야 한다 (텍스트·테두리 침범 0).
        if not np.array_equal(canvas[sy:sy + fh, sx:sx + fw], frame):
            raise SystemExit(f"{out_path.name} frame {f_idx}: 장면 영역이 오염됨")
        writer.write(canvas)
        ok, frame = cap.read()
        if ok:
            frame = frame[crop_top:]
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
    ap.add_argument("--clean-root", default=None,
                    help="replay_clean_video.py 산출 디렉터리 (<stem>.mp4)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default=None, help="기본 = JSON config.modes 전부")
    ap.add_argument("--only", default=None,
                    help="task 필터 (콤마 구분, 부분 일치)")
    ap.add_argument("--stems", default=None, help="stem 필터 (콤마 구분, 완전 일치)")
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
    stems = [s.strip() for s in args.stems.split(",")] if args.stems else None
    clean_root = Path(args.clean_root) if args.clean_root else None

    made, sources = [], []
    for rec in payload["episodes"]:
        if only and not any(o in rec["task"] for o in only):
            continue
        stem = f"{rec['task']}_s{rec['scene']}n{rec['noise']}"
        if stems and stem not in stems:
            continue
        vpath, how, crop_top, why = resolve_source(rec, Path(args.video_root), clean_root,
                                                   stem)
        sources.append({"stem": stem, "source": how, "crop_top": crop_top, "why": why,
                        "path": None if vpath is None else vpath.name})
        if vpath is None:
            print(f"[skip] {stem}: {why}")
            continue
        print(f"[source] {stem}: {how} ({why})")
        for mode in modes:
            tag = "succ" if rec["succ"] else "fail"
            name = f"{stem}_{tag}_{mode}.mp4"
            outp = out_dir / name
            align = render_one(rec, mode, vpath, crop_top, outp, collect, fps,
                               args.check_align, args.align_tol)
            tf = rec["modes"][mode]["t_fire"]
            print(f"[render] {name}  T={rec['T']} t_fire={tf} "
                  f"frames={align['written']} (기대 {align['expected_frames']}, "
                  f"diff {align['diff']})")
            made.append({"file": name, "task": rec["task"], "mode": mode,
                         "t_fire": tf, "T": rec["T"], "succ": rec["succ"],
                         "source": how, "crop_top": crop_top, **align})
    (out_dir / "render_index.json").write_text(
        json.dumps({"videos": made, "sources": sources}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n[render] {len(made)} 개 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
