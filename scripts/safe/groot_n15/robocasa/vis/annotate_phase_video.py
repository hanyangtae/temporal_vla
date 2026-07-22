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
import json
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
    "reach-to-handle": (120, 200, 255),
    "grasp": (255, 120, 200),
    "grasp-handle": (255, 120, 200),
    "contact-door": (255, 120, 200),
    "push-close": (255, 210, 90),
    "swing-open": (255, 140, 60),
    "close-done": (140, 255, 140),
    "reach-to-head": (120, 200, 255),
    "contact-head": (255, 120, 200),
    "lift-open": (255, 210, 90),
    "push-down": (255, 140, 60),
    "wrong-grasp": (255, 70, 70),
    "disengage": (255, 100, 120),
    "transport": (255, 210, 90),
    "pull": (255, 210, 90),
    "push-back": (255, 140, 60),
    "place": (255, 140, 60),
    "insert-settle": (140, 255, 140),
    "release-settle": (140, 255, 140),
    "open-done": (140, 255, 140),
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
    # 캡처-ON 은 pkl, 캡처-OFF(--no-features) 는 동일 스템의 경량 json 사이드카.
    if pkl_path.suffix == ".json":
        d = json.loads(pkl_path.read_text())
    else:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    return {
        "instruction": d.get("canonical_instruction") or d.get("task_description") or "",
        "feature_phases": list(d.get("feature_phases") or []),
        # env-step 해상도 GT (있으면 우선; frame f → phase = env_step_phases[f*spr])
        "env_step_phases": list(d.get("env_step_phases") or []),
        "steps_per_render": int(d.get("steps_per_render", 2)),
        "n_action_steps": int(d.get("n_action_steps", 5)),
        "success": int(d.get("episode_success", 0)),
        "cell_id": d.get("cell_id", ""),
    }


def _phase_at(meta: dict, f: int) -> tuple[str, int, str]:
    """frame f 의 phase 를 env-step 해상도 우선으로 반환. (phase, env_step, 해상도라벨)."""
    spr = meta["steps_per_render"]
    nas = meta["n_action_steps"]
    env_step = f * spr
    esp = meta["env_step_phases"]
    if esp:  # env-step GT: frame f ↔ env_step 직접 대응
        return esp[min(env_step, len(esp) - 1)], env_step, "env-step"
    phases = meta["feature_phases"]
    R = len(phases)
    r = min(max(env_step // nas, 0), R - 1) if R else 0
    return (phases[r] if R else "?"), env_step, "record"


def annotate(pkl_path: Path, mp4_path: Path, out_path: Path, banner_frac: float = 0.24) -> None:
    meta = _read_meta(pkl_path)
    instr = meta["instruction"]
    res_label = "env-step" if meta["env_step_phases"] else "record"

    reader = imageio.get_reader(str(mp4_path))
    fps = float(reader.get_meta_data().get("fps", 20) or 20)
    frames = [np.asarray(fr) for fr in reader]
    reader.close()
    if not frames:
        raise SystemExit(f"no frames in {mp4_path}")
    H, W = frames[0].shape[:2]
    # 배너: instruction(위) / 여백 / phase(큰 글씨) 3구역 + 하단 phase 색 띠.
    banner_h = max(80, int(round(H * banner_frac)))
    bar_h = max(6, banner_h // 12)              # phase 색 띠 두께
    f_instr = _load_font(max(13, banner_h // 5))
    f_phase = _load_font(max(20, banner_h // 3))  # phase 는 크게

    writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=None)
    for f, frame in enumerate(frames):
        phase, env_step, _ = _phase_at(meta, f)
        color = PHASE_COLORS.get(phase, (255, 255, 255))

        canvas = Image.new("RGB", (W, H + banner_h), (15, 15, 15))
        canvas.paste(Image.fromarray(frame[:, :, :3]), (0, banner_h))
        draw = ImageDraw.Draw(canvas)
        pad = 10
        # instruction (맨 위 행, 폭 초과 시 축약)
        instr_txt = instr
        while instr_txt and draw.textlength(instr_txt, font=f_instr) > W - 2 * pad:
            instr_txt = instr_txt[:-2]
        if instr_txt != instr:
            instr_txt = instr_txt.rstrip() + "…"
        draw.text((pad, 6), instr_txt, fill=(210, 210, 210), font=f_instr)
        # phase (여백 아래, 큰 글씨·색) — instruction 과 분리된 구역
        phase_y = 6 + int(banner_h * 0.42)
        draw.text((pad, phase_y), f"phase: {phase}", fill=color, font=f_phase)
        # step + 해상도 (우측)
        info_txt = f"step {env_step}  [{res_label}]"
        sw = draw.textlength(info_txt, font=f_instr)
        draw.text((W - sw - pad, phase_y + 4), info_txt, fill=(200, 200, 200), font=f_instr)
        # phase 색 띠 (배너 하단 = 영상 바로 위): 현재 phase 색으로 꽉 채움
        draw.rectangle([0, banner_h - bar_h, W, banner_h], fill=color)

        writer.append_data(np.asarray(canvas))
    writer.close()
    print(f"[ok] {out_path.name}  frames={len(frames)} res={res_label} "
          f"succ={meta['success']} cell={meta['cell_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="phase/step 오버레이 주석 영상")
    ap.add_argument("--run-dir", required=True, help="raw_rollouts dir (glob */*/*.pkl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-fail", action="store_true", help="succ0 만")
    ap.add_argument("--max-ep", type=int, default=None, help="ep 번호 상한 (예: 29 → ep0-29만)")
    ap.add_argument("--banner-frac", type=float, default=0.24)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # cell 하위디렉토리 유무(= --cell-id 사용 여부)와 캡처 ON/OFF 를 모두 커버
    pkls: list[Path] = []
    for pat in ("*/*/*.pkl", "*/*/*.json", "*/*.pkl", "*/*.json"):
        pkls = sorted(run_dir.glob(pat))
        if pkls:
            break
    n = 0
    import re
    for pkl_path in pkls:
        if args.only_fail and not pkl_path.stem.endswith("succ0"):
            continue
        if args.max_ep is not None:
            m = re.search(r"--ep(\d+)--", pkl_path.name)
            if m and int(m.group(1)) > args.max_ep:
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
