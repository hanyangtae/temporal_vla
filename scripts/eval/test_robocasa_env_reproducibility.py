"""
RoboCasa GrootRoboCasaEnv 재현성 sanity test.

같은 seed 로 env 를 두 번 만들고 각각 reset 1회 → 첫 obs 의 카메라 3장 (left/right/wrist)
을 PNG 로 저장 + MD5 해시 비교.

사용:
  python scripts/eval/test_robocasa_env_reproducibility.py \
      --task OpenDrawer --seed 42 \
      --out-dir /temporal_vla/outputs/eval/robocasa/repro_test

도커 안 (robocasa) 에서 GPU 4번으로 돌릴 때:
  docker compose run --rm -e CUDA_VISIBLE_DEVICES=4 robocasa \
      bash -c 'cd /temporal_vla && python scripts/eval/test_robocasa_env_reproducibility.py'
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# 도커 안: MUJOCO_GL=egl 로 offscreen 렌더 (rollout_policy.py:89 도 동일).
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from path_setup import configure_repo_paths

configure_repo_paths(include_script_utils=True, include_robocasa=True)

import numpy as np
from PIL import Image

CAMERA_KEYS = [
    "video.res256_image_side_0",   # left agentview
    "video.res256_image_side_1",   # right agentview
    "video.res256_image_wrist_0",  # wrist
]


def run_once(env_id: str, seed: int, label: str, out_dir: Path) -> dict[str, str]:
    import gymnasium as gym
    import robocasa  # noqa: F401  (env 등록 trigger)
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401

    print(f"[{label}] gym.make({env_id}, seed={seed})", flush=True)
    env = gym.make(env_id, enable_render=True, seed=seed)
    obs, info = env.reset()

    hashes = {}
    for cam_key in CAMERA_KEYS:
        img = obs.get(cam_key)
        if img is None:
            print(f"[{label}] WARN: obs 에 '{cam_key}' 없음. keys={sorted(obs.keys())}")
            continue
        img = np.asarray(img, dtype=np.uint8)
        out_path = out_dir / f"{label}_{cam_key.split('.')[-1]}.png"
        Image.fromarray(img).save(out_path)
        md5 = hashlib.md5(img.tobytes()).hexdigest()
        hashes[cam_key] = md5
        print(f"[{label}] saved {out_path.name} shape={img.shape} md5={md5}", flush=True)

    # qpos / qvel 해시 — 물리 상태 재현성 검증 (렌더 무관)
    qpos = np.asarray(env.unwrapped.env.sim.data.qpos).copy()
    qvel = np.asarray(env.unwrapped.env.sim.data.qvel).copy()
    state_hash = hashlib.md5(np.concatenate([qpos, qvel]).tobytes()).hexdigest()
    hashes["__sim_state__"] = state_hash
    print(f"[{label}] qpos+qvel md5={state_hash} (len={qpos.size + qvel.size})", flush=True)

    # ep_meta 도 저장 (재현 메타 공유용 reference)
    try:
        ep_meta = env.unwrapped.env.get_ep_meta()
        import json
        with open(out_dir / f"{label}_ep_meta.json", "w") as f:
            json.dump(ep_meta, f, indent=2, default=str)
        print(f"[{label}] ep_meta saved (layout={ep_meta.get('layout_id')}, "
              f"style={ep_meta.get('style_id')}, "
              f"lang={ep_meta.get('lang', '')!r})", flush=True)
    except Exception as e:
        print(f"[{label}] ep_meta dump 실패: {e}", flush=True)

    env.close()
    return hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="OpenDrawer",
                   help="robocasa task class 이름 (예: OpenDrawer, TurnOnMicrowave)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str,
                   default="/temporal_vla/outputs/eval/robocasa/repro_test")
    args = p.parse_args()

    env_id = (
        args.task if "/" in args.task
        else f"robocasa_panda_omron/{args.task}_PandaOmron_Env"
    )
    out_dir = Path(args.out_dir) / f"{args.task}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== task: {args.task} | env_id: {env_id} | seed: {args.seed} ===\n", flush=True)
    print(f"out_dir: {out_dir}\n", flush=True)

    h1 = run_once(env_id, args.seed, "run1", out_dir)
    print()
    h2 = run_once(env_id, args.seed, "run2", out_dir)

    print("\n" + "=" * 70)
    print("재현성 검증 결과")
    print("=" * 70)
    all_match = True
    keys = sorted(set(h1) | set(h2))
    for k in keys:
        m1, m2 = h1.get(k, "—"), h2.get(k, "—")
        ok = m1 == m2 and m1 != "—"
        all_match = all_match and ok
        status = "OK" if ok else "MISMATCH"
        print(f"  [{status}] {k:35s} run1={m1[:12]}... run2={m2[:12]}...")
    print("-" * 70)
    print(f"  종합: {'재현 OK' if all_match else '재현 실패 (위 MISMATCH 항목 확인)'}")
    print("=" * 70)
    print(f"\n이미지 비교: {out_dir}/run1_*.png vs run2_*.png")


if __name__ == "__main__":
    main()
