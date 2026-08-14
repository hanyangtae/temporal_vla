#!/usr/bin/env python
"""수집 rollout 을 open-loop replay 해 **캡션 없는 클린 영상**을 재생성한다.

## 동기

grid 수집 영상 중 일부(kanu 머신분, 높이 256px)는 GR00T upstream
`VideoRecordingWrapper.overlay_text` 가 **장면 프레임 하단에 직접** 캡션을 구워 넣은
판본이라 로봇이 가려진다(`video_recording_wrapper.py` 의 `text_y = frame.shape[0] - 20`).
이후 판본은 프레임 위에 검은 스트립을 덧대는 방식이라 crop 으로 장면을 온전히 복원할 수
있지만, 하단 burn-in 판본은 복원 불가 → 저장된 action 을 env 에 다시 먹여 프레임을
재생성한다.

## 재현 규약

- `gym.make(env_name, enable_render=True, seed=scenario_seed)` + `reset(seed=scenario_seed)`.
  (선례: `scripts/safe/groot_n15/robocasa/eval/env_step_gt_batch.py` — 같은 pkl 트리를
  open-loop replay 해 env-step GT 를 소급 재구성했고 fidelity 검사를 통과했다.)
- record 당 chunk 앞 `n_action_steps` 개만 실행 (MultiStepWrapper 와 동일).
- **에피소드당 fresh 프로세스** (한 프로세스에서 gym.make 연속 생성 시 scene 오염).
  드라이버(`--bundle-dir`)가 워커를 하나씩 subprocess 로 띄운다.
- 프레임 cadence 는 `VideoRecordingWrapper` 와 동일: reset 후 `step_count=1`, step 마다
  먼저 +1 한 뒤 `step_count % steps_per_render == 0` 일 때 기록 → steps_per_render=2 면
  **0-based env-step s 가 짝수일 때** 프레임. 프레임 수 = ceil(n_action_steps*T/2).
- 카메라 구성도 동일: `src/policies/groot/robocasa/io.py` 의 `VIDEO_RECORDING_KEYS`
  (res256 side_0 / side_1 / wrist_0) 를 obs 순서대로 가로 concat → 768x256.
- 캡션을 그리는 video wrapper 는 **경유하지 않는다**. raw 카메라 프레임만 기록.

## 충실도 게이트

record r 의 관측(= env-step n_action_steps*r 직전)에서 얻은 eef 상대 위치를 수집 당시
저장값(`states[r]["observation.state.eef_pos_rel"]`)과 대조해 최대 편차를 보고한다.
`--eef-tol` (기본 1cm) 초과 시 `diverged=true` 로 표시하고 mp4 를 남기지 않는다
(틀린 장면 위에 detector score 를 얹는 것이 더 나쁘다).
참고 게이트로 원본 영상의 **오염되지 않은 상단 영역** 픽셀 차이도 같이 보고한다.

## 사용 (robocasa 컨테이너)

    docker exec -e MUJOCO_GL=egl robocasa python \
      /temporal_vla/.claude/worktrees/<wt>/scripts/analysis/grid_phase/replay_clean_video.py \
      --bundle-dir <dir of *.bundle.pkl> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

# src/policies/groot/robocasa/io.py 의 canonical 3-view 와 동일 (import 의존 없이 명시).
VIDEO_KEYS = (
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
)
EEF_STATE_KEY = "observation.state.eef_pos_rel"
OBS_EEF_KEY = "state.end_effector_position_relative"


def _concat_views(obs: dict) -> np.ndarray:
    frames = [np.asarray(obs[k]) for k in VIDEO_KEYS if k in obs]
    if not frames:
        raise SystemExit(f"video 키 없음: {sorted(k for k in obs if k.startswith('video.'))}")
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def _ref_top_diff(ref_video: Path, frames: list[np.ndarray], top_rows: int) -> dict:
    """원본 영상의 캡션 미오염 상단 영역과 replay 프레임의 픽셀 차이 (참고 게이트)."""
    import cv2

    cap = cv2.VideoCapture(str(ref_video))
    diffs = []
    i = 0
    while i < len(frames):
        ok, bgr = cap.read()
        if not ok:
            break
        ref = bgr[:top_rows].astype(np.int16)
        mine = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)[:top_rows].astype(np.int16)
        if ref.shape != mine.shape:
            cap.release()
            return {"error": f"shape {ref.shape} vs {mine.shape}"}
        diffs.append(float(np.abs(ref - mine).mean()))
        i += 1
    cap.release()
    if not diffs:
        return {"error": "no frames"}
    return {"n": len(diffs), "mean": round(float(np.mean(diffs)), 3),
            "max": round(float(np.max(diffs)), 3),
            "argmax": int(np.argmax(diffs))}


def replay_one(bundle_path: Path, out_dir: Path, eef_tol: float, ref_root: Path | None,
               top_rows: int) -> dict:
    import cv2
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401

    b = pickle.load(open(bundle_path, "rb"))
    nas = int(b.get("n_action_steps", 5))
    spr = int(b.get("steps_per_render", 2))
    fps = int(b.get("video_fps", 20))
    actions = b["actions"]
    states = b.get("states") or []
    stem = bundle_path.name.replace(".bundle.pkl", "")

    env = gym.make(b["env_name"], enable_render=True, seed=int(b["scenario_seed"]))
    obs, _ = env.reset(seed=int(b["scenario_seed"]))

    frames: list[np.ndarray] = []
    eef_dev: list[float] = []

    def check_eef(r: int, o: dict) -> None:
        if r < len(states) and EEF_STATE_KEY in states[r] and OBS_EEF_KEY in o:
            d = np.abs(np.asarray(o[OBS_EEF_KEY], np.float64)
                       - np.asarray(states[r][EEF_STATE_KEY], np.float64)).max()
            eef_dev.append(float(d))

    check_eef(0, obs)
    s = 0  # 0-based env-step
    for r, rec in enumerate(actions):
        for i in range(nas):
            a = {k: np.asarray(v[i]) for k, v in rec.items()}
            obs = env.step(a)[0]
            if s % spr == 0:                      # wrapper cadence 와 동일
                frames.append(_concat_views(obs).copy())
            s += 1
        check_eef(r + 1, obs)
    env.close()

    max_dev = max(eef_dev) if eef_dev else float("nan")
    diverged = bool(eef_dev) and max_dev > eef_tol
    info = {
        "bundle": bundle_path.name,
        "stem": stem,
        "env_name": b["env_name"],
        "scenario_seed": int(b["scenario_seed"]),
        "records": len(actions),
        "env_steps": s,
        "frames": len(frames),
        "expected_frames": int(np.ceil(len(actions) * nas / spr)),
        "eef_max_dev_m": None if not eef_dev else round(max_dev, 6),
        "eef_mean_dev_m": None if not eef_dev else round(float(np.mean(eef_dev)), 6),
        "eef_tol_m": eef_tol,
        "diverged": diverged,
        "pkl_sha256": b.get("pkl_sha256"),
    }

    if ref_root is not None:
        ref = ref_root / f"{stem}.mp4"
        if ref.exists():
            info["ref_top_diff"] = _ref_top_diff(ref, frames, top_rows)

    if diverged:
        info["note"] = "replay 발산 — mp4 미생성"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        outp = out_dir / f"{stem}.mp4"
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not wr.isOpened():
            raise SystemExit(f"VideoWriter 열기 실패: {outp}")
        for f in frames:
            wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        wr.release()
        info["out"] = str(outp.name)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=None, help="워커 모드(내부용): 단일 번들")
    ap.add_argument("--bundle-dir", default=None, help="드라이버 모드: *.bundle.pkl 디렉터리")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eef-tol", type=float, default=0.01, help="충실도 허용 편차 (m)")
    ap.add_argument("--ref-root", default=None,
                    help="원본 영상 디렉터리(<stem>.mp4) — 상단 픽셀 차이 참고 게이트")
    ap.add_argument("--ref-top-rows", type=int, default=170,
                    help="캡션에 오염되지 않은 상단 행 수 (256px 프레임 기준)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ref_root = Path(args.ref_root) if args.ref_root else None

    if args.bundle:
        info = replay_one(Path(args.bundle), out_dir, args.eef_tol, ref_root,
                          args.ref_top_rows)
        print("REPLAY_JSON " + json.dumps(info, ensure_ascii=False), flush=True)
        return 0

    if not args.bundle_dir:
        raise SystemExit("--bundle 또는 --bundle-dir 필요")
    bundles = sorted(Path(args.bundle_dir).glob("*.bundle.pkl"))
    print(f"[replay] {len(bundles)} 개 번들", flush=True)
    results = []
    for bp in bundles:
        # fresh 프로세스 강제 (재현성 규약)
        cmd = [sys.executable, __file__, "--bundle", str(bp), "--out-dir", str(out_dir),
               "--eef-tol", str(args.eef_tol), "--ref-top-rows", str(args.ref_top_rows)]
        if ref_root:
            cmd += ["--ref-root", str(ref_root)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        line = next((ln for ln in r.stdout.splitlines()
                     if ln.startswith("REPLAY_JSON ")), None)
        if line is None:
            print(f"[replay] FAIL {bp.name} rc={r.returncode}\n{r.stdout[-2000:]}\n"
                  f"{r.stderr[-2000:]}", flush=True)
            results.append({"bundle": bp.name, "error": f"rc={r.returncode}"})
            continue
        info = json.loads(line[len("REPLAY_JSON "):])
        results.append(info)
        print(f"[replay] {info['stem']} frames={info['frames']}/"
              f"{info['expected_frames']} eef_max={info['eef_max_dev_m']} "
              f"diverged={info['diverged']} ref={info.get('ref_top_diff')}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay_fidelity.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[replay] 완료 → {out_dir}/replay_fidelity.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
