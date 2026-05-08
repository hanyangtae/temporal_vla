"""VLA 평가 스크립트 (LIBERO 환경, 모델 무관).

통일 API 를 따르는 어떤 VLA 서버든 `--server-url` 만 바꾸면 평가 가능.
구조: scripts/eval/calvin.py 와 동일 패턴.

openvla_oft 컨테이너에서 실행 (LIBERO env 가 거기 있음):

  # 1) 모델 서버 백그라운드 기동
  python /temporal_vla/scripts/serve/openvla_oft.py \
      --profile /temporal_vla/configs/checkpoints/openvla_oft__moojink_libero_spatial.yaml &

  # 2) 평가 실행
  python /temporal_vla/scripts/eval/libero.py \
      --task-suite libero_spatial \
      --num-trials 50 \
      --max-steps 660 \
      --success-deadline 220 \
      --server-url http://localhost:8400 \
      --video-dir /temporal_vla/outputs/rollouts/libero/openvla_oft/<TS>

`success_deadline` 안에 env `done=True` 가 와야 진짜 성공. 그 이후
`max_steps` 까지 사이에 done 이 와도 실패로 마킹하고 즉시 종료.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# LIBERO 컨테이너의 openvla-oft 경로 (set_seed_everywhere 등 일부 유틸 재사용)
sys.path.insert(0, "/temporal_vla/scripts/utils")
sys.path.insert(0, "/temporal_vla/src/benchmarks/LIBERO")
sys.path.insert(0, "/temporal_vla/src")
sys.path.insert(0, "/temporal_vla/src/policies/openvla-oft")

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402

from processor.factory import make_libero_processors  # noqa: E402
from processor.types import TransitionKey  # noqa: E402
from vla_client import VLAClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


URL_MAP = {
    "http://localhost:8400": "openvla_oft",
    "http://localhost:8300": "upvla",
    "http://localhost:8200": "dreamvla",
    "http://localhost:8100": "xvla",
}

# LIBERO suite 별 표준 max_steps (openvla-oft 업스트림 기준).
# 영상 수집 시 max-steps 인자로 override (예: ×3).
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
NUM_STEPS_WAIT = 10  # objects stabilize 대기 (업스트림 default)


# ─── LIBERO 헬퍼 ─────────────────────────────────────────────────────────────


def _make_libero_env(task, resolution: int = 256) -> Tuple[OffScreenRenderEnv, str]:
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env, task.language


def _replay_image(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """env raw `agentview_image` 를 영상 저장용으로 180도 회전."""
    img = np.asarray(obs["agentview_image"])
    return img[::-1, ::-1].copy()


def _save_video(frames: List[np.ndarray], path: Path, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ─── /act 호출 (sub-key dict 또는 flat 양쪽 지원) ────────────────────────────


def _predict(
    vla_client: VLAClient, processed_obs: Dict, instruction: str
) -> Dict[str, np.ndarray]:
    """unified obs sub-keys → /act 호출 → 첫 step action sub-key dict.

    openvla_oft 서버는 내부 queue 로 chunk 를 분할해 한 번에 1 step (N=1) 만 반환.
    따라서 매 env step 마다 /act 를 1 회 호출.
    """
    images: Dict[str, np.ndarray] = {}
    states: Dict[str, np.ndarray] = {}
    for k, v in processed_obs.items():
        if k.startswith("observation.images."):
            cam = k.split("observation.images.")[-1]
            images[cam] = v
        elif k.startswith("observation.state"):
            states[k] = np.asarray(v)

    actions, _ = vla_client.predict(
        images=images, states=states or None, instruction=instruction
    )
    if isinstance(actions, dict):
        # 서버 N=1 → 각 키의 [0] 행만 사용
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in actions.items()}
    # flat ndarray fallback
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    return {"_flat": arr}


# ─── 단일 episode rollout ───────────────────────────────────────────────────


def _rollout(
    env,
    instruction: str,
    initial_state: np.ndarray,
    vla_client: VLAClient,
    obs_pipeline,
    action_pipeline,
    max_steps: int,
    success_deadline: int,
    record: bool = False,
) -> Tuple[bool, List[np.ndarray], int]:
    """단일 LIBERO episode rollout.

    Returns:
        (success, frames, terminated_step)
        success: deadline 안에 env `done=True` 가 왔는가.
        frames: replay 이미지 (record=True 일 때만 채움).
        terminated_step: 실제로 멈춘 step (effective; num_steps_wait 제외).
    """
    vla_client.reset()
    env.reset()
    obs = env.set_init_state(initial_state)

    frames: List[np.ndarray] = []
    success = False
    effective_t = 0

    # objects stabilize: dummy action 으로 NUM_STEPS_WAIT step 진행
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(DUMMY_ACTION.tolist())

    for t in range(max_steps):
        processed = obs_pipeline({TransitionKey.OBSERVATION: obs})
        processed_obs = processed[TransitionKey.OBSERVATION]
        action_pred = _predict(vla_client, processed_obs, instruction)
        # _flat fallback (서버가 sub-key 가 아니라 flat 으로 응답한 경우) 은 ndarray 직접
        action_in = (
            action_pred["_flat"] if "_flat" in action_pred else action_pred
        )

        processed_act = action_pipeline({TransitionKey.ACTION: action_in})
        action = processed_act[TransitionKey.ACTION]
        action_list = action.tolist() if isinstance(action, np.ndarray) else list(action)

        obs, _, done, _ = env.step(action_list)
        if record:
            frames.append(_replay_image(obs))

        effective_t = t + 1
        if done:
            success = effective_t < success_deadline
            break

    return success, frames, effective_t


# ─── 메인 evaluate ──────────────────────────────────────────────────────────


def evaluate(
    task_suite_name: str,
    server_url: str,
    num_trials: int,
    max_steps: int,
    success_deadline: int,
    video_dir: Path,
    output_path: Optional[Path],
    image_size: int = 256,
    seed: int = 7,
):
    np.random.seed(seed)

    vla_client = VLAClient(url=server_url)
    logger.info("VLA 서버 연결 대기: %s", server_url)
    server_info = vla_client.wait_until_ready()
    logger.info("VLA 서버 연결 완료: %s", server_info)

    obs_pipeline, action_pipeline = make_libero_processors(image_size=image_size)

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(
            f"unknown task_suite: {task_suite_name}. "
            f"available: {list(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[task_suite_name]()
    n_tasks = task_suite.n_tasks

    suite_video_dir = video_dir / task_suite_name
    suite_video_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    total_episodes = 0
    total_successes = 0
    t_start = time.time()

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        env, task_description = _make_libero_env(task, resolution=image_size)
        initial_states = task_suite.get_task_init_states(task_id)

        task_episodes = 0
        task_successes = 0
        for trial in range(num_trials):
            init_state = initial_states[trial % len(initial_states)]
            success, frames, eff_t = _rollout(
                env,
                task_description,
                init_state,
                vla_client,
                obs_pipeline,
                action_pipeline,
                max_steps=max_steps,
                success_deadline=success_deadline,
                record=True,
            )
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1
            else:
                # 실패 또는 deadline 초과 → 영상 저장
                desc = (
                    task_description.lower()
                    .replace(" ", "_")
                    .replace("\n", "_")
                    .replace(".", "_")[:60]
                )
                mp4 = (
                    suite_video_dir
                    / f"task{task_id:02d}_trial{trial:03d}--steps{eff_t}--{desc}.mp4"
                )
                _save_video(frames, mp4)
                logger.info("[FAIL] saved %s", mp4)

            results.append(
                {
                    "task_id": task_id,
                    "task": task_description,
                    "trial": trial,
                    "success": bool(success),
                    "terminated_step": int(eff_t),
                }
            )

            done_pct = total_successes / total_episodes * 100
            logger.info(
                "[%s] task=%d trial=%d success=%s eff_t=%d | overall %d/%d (%.1f%%)",
                task_suite_name,
                task_id,
                trial,
                success,
                eff_t,
                total_successes,
                total_episodes,
                done_pct,
            )

        env.close()
        logger.info(
            "Task %d done: %d/%d (task SR %.2f)",
            task_id,
            task_successes,
            task_episodes,
            task_successes / max(task_episodes, 1),
        )

    duration = time.time() - t_start
    summary = {
        "task_suite": task_suite_name,
        "server_url": server_url,
        "num_trials_per_task": num_trials,
        "n_tasks": n_tasks,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / max(total_episodes, 1),
        "max_steps": max_steps,
        "success_deadline": success_deadline,
        "duration_sec": duration,
        "results": results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary to %s", output_path)
    else:
        # video_dir 안에 자동 저장
        auto = video_dir / f"results_{task_suite_name}.json"
        auto.parent.mkdir(parents=True, exist_ok=True)
        with open(auto, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary to %s", auto)

    logger.info(
        "DONE [%s] success_rate=%.4f (%d/%d) in %.1fs",
        task_suite_name,
        summary["success_rate"],
        total_successes,
        total_episodes,
        duration,
    )


def main():
    parser = argparse.ArgumentParser(description="VLA LIBERO 평가 (모델 무관)")
    parser.add_argument(
        "--task-suite",
        type=str,
        required=True,
        choices=list(DEFAULT_MAX_STEPS.keys()),
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8400",
        help="VLA 서버 URL (통일 API)",
    )
    parser.add_argument("--num-trials", type=int, default=50, help="task 당 trial 수")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="rollout 최대 step (NUM_STEPS_WAIT 제외 effective). "
        "미지정 시 DEFAULT_MAX_STEPS[suite].",
    )
    parser.add_argument(
        "--success-deadline",
        type=int,
        default=None,
        help="이 step 안에 env done=True 가 와야 진짜 성공. 그 이후 done 은 실패. "
        "미지정 시 DEFAULT_MAX_STEPS[suite] (즉, max-steps 와 동일하면 deadline 효과 없음).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="실패 영상 저장 루트. 하위에 <task_suite>/ 가 자동 생성됨.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="summary JSON 경로 override."
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    max_steps = args.max_steps or DEFAULT_MAX_STEPS[args.task_suite]
    success_deadline = (
        args.success_deadline
        if args.success_deadline is not None
        else DEFAULT_MAX_STEPS[args.task_suite]
    )

    if success_deadline > max_steps:
        raise ValueError(
            f"success_deadline ({success_deadline}) > max_steps ({max_steps}) — "
            "deadline 이 더 길면 의미 없음."
        )

    evaluate(
        task_suite_name=args.task_suite,
        server_url=args.server_url,
        num_trials=args.num_trials,
        max_steps=max_steps,
        success_deadline=success_deadline,
        video_dir=Path(args.video_dir),
        output_path=Path(args.output) if args.output else None,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
