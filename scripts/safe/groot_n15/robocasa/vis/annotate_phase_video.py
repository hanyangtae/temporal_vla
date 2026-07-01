"""Rollout mp4 위에 상단 배너(instruction + 현재 action-phase) + 우측 step 카운터 오버레이.

용도: GR00T N1.5 RoboCasa event-phase 수집(`phase_event_aligned_4cell`)의 **실패 rollout**을
사람이 phase 진행과 함께 보게 만드는 주석 영상. 라벨러 검증(플랜 검증 §1)·정성 분석용.

프레임↔phase↔step 매핑 (수집 설정에 의해 결정, pkl 메타에서 읽음):
  video_fps=20, steps_per_render=2, n_action_steps=5.
  frame f  → env_step = f * steps_per_render
           → record r = env_step // n_action_steps   (get_action 호출 index)
           → phase   = feature_phases[clamp(r, 0, R-1)]
  상단 배너: instruction(1행) + "phase: <phase>"(2행, phase별 색). 우측: "step <env_step>".

의존성(시스템 ffmpeg 불필요): imageio + imageio-ffmpeg(ffmpeg 바이너리 번들) + pillow + numpy + torch(pkl unpickle).
  없으면: pip install --user imageio imageio-ffmpeg pillow
pkl은 torch 텐서를 담아 unpickle에 torch 필요(원격은 ~/anaconda3/bin/python).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

from PIL import Image, ImageDraw, ImageFont

# phase별 배너 색 (RGB). 미정의 phase는 흰색.
PHASE_COLORS = {
    "reach-to-object": (120, 200, 255),
    "reach-to-door": (120, 200, 255),
    "transport": (255, 210, 90),
    "insert-settle": (140, 255, 140),
    "release-settle": (140, 255, 140),
    "terminal": (200, 200, 200),
}


def _load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _read_meta(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return {
        "instruction": d.get("canonical_instruction") or d.get("task_description") or "",
        "feature_phases": list(d.get("feature_phases") or []),
        "steps_per_render": int(d.get("steps_per_render", 2)),
        "n_action_steps": int(d.get("n_action_steps", 5)),
        "success": int(d.get("episode_success", 0)),
        "cell_id": d.get("cell_id", ""),
    }


def annotate(pkl_path: Path, mp4_path: Path, out_path: Path, banner_frac: float = 0.16) -> None:
    meta = _read_meta(pkl_path)
    phases = meta["feature_phases"]
    R = len(phases)
    spr = meta["steps_per_render"]
    nas = meta["n_action_steps"]
    instr = meta["instruction"]

    reader = imageio.get_reader(str(mp4_path))
    fps = float(reader.get_meta_data().get("fps", 20) or 20)
    frames = [np.asarray(fr) for fr in reader]
    reader.close()
    if not frames:
        raise SystemExit(f"no frames in {mp4_path}")
    H, W = frames[0].shape[:2]
    banner_h = max(48, int(round(H * banner_frac)))
    f_main = _load_font(max(14, banner_h // 3))
    f_small = _load_font(max(12, banner_h // 4))

    writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=None)
    for f, frame in enumerate(frames):
        env_step = f * spr
        r = min(max(env_step // nas, 0), R - 1) if R else 0
        phase = phases[r] if R else "?"
        color = PHASE_COLORS.get(phase, (255, 255, 255))

        canvas = Image.new("RGB", (W, H + banner_h), (15, 15, 15))
        canvas.paste(Image.fromarray(frame[:, :, :3]), (0, banner_h))
        draw = ImageDraw.Draw(canvas)
        # instruction (1행, 필요시 폭에 맞게 축약)
        pad = 8
        instr_txt = instr
        while instr_txt and draw.textlength(instr_txt, font=f_small) > W - 140:
            instr_txt = instr_txt[:-2]
        if instr_txt != instr:
            instr_txt = instr_txt.rstrip() + "…"
        draw.text((pad, 4), instr_txt, fill=(235, 235, 235), font=f_small)
        # phase (2행, 색)
        draw.text((pad, 4 + banner_h // 3), f"phase: {phase}", fill=color, font=f_main)
        # step (우측)
        step_txt = f"step {env_step}"
        sw = draw.textlength(step_txt, font=f_main)
        draw.text((W - sw - pad, 4 + banner_h // 3), step_txt, fill=(235, 235, 235), font=f_main)

        writer.append_data(np.asarray(canvas))
    writer.close()
    print(f"[ok] {out_path.name}  frames={len(frames)} R={R} succ={meta['success']} cell={meta['cell_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="phase/step 오버레이 주석 영상")
    ap.add_argument("--run-dir", required=True, help="raw_rollouts dir (glob */*/*.pkl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-fail", action="store_true", help="succ0 만")
    ap.add_argument("--banner-frac", type=float, default=0.16)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkls = sorted(run_dir.glob("*/*/*.pkl"))
    n = 0
    for pkl_path in pkls:
        if args.only_fail and not pkl_path.name.endswith("succ0.pkl"):
            continue
        mp4_path = pkl_path.with_suffix(".mp4")
        if not mp4_path.exists():
            print(f"[skip] no mp4 for {pkl_path.name}")
            continue
        out_path = out_dir / f"{pkl_path.parent.name}--{pkl_path.stem}--annot.mp4"
        annotate(pkl_path, mp4_path, out_path, banner_frac=args.banner_frac)
        n += 1
    print(f"[done] {n} annotated -> {out_dir}")


if __name__ == "__main__":
    main()
