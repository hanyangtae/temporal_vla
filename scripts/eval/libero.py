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

# 이 파일 이름이 `libero.py` 라 자기 디렉토리(`scripts/eval`) 가 sys.path[0] 에 잡히면
# `from libero.libero import benchmark` 가 자기 자신을 import 해 ModuleNotFoundError 가 난다.
# LIBERO 패키지가 잡히도록 자기 디렉토리를 sys.path 에서 제거.
_self_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _self_dir]

# LIBERO 컨테이너의 openvla-oft 경로 (set_seed_everywhere 등 일부 유틸 재사용)
sys.path.insert(0, "/temporal_vla/scripts/utils")
sys.path.insert(0, "/temporal_vla/src/benchmarks/LIBERO")
sys.path.insert(0, "/temporal_vla/src")
sys.path.insert(0, "/temporal_vla/src/policies/openvla-oft")
# SAFE 수집 공통 writer (repo-relative: scripts/eval → scripts/safe/lerobot)
sys.path.insert(0, os.path.join(os.path.dirname(_self_dir), "safe", "lerobot"))
# repo root — src.collect.* (LIBERO BDDL sim-state phase 라벨러) import 용
sys.path.insert(0, os.path.dirname(os.path.dirname(_self_dir)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402

from processor.factory import make_libero_processors  # noqa: E402
from processor.types import TransitionKey  # noqa: E402
from vla_client import VLAClient  # noqa: E402
from collect_common import (  # noqa: E402
    SafeEpisodeCollector,
    episode_path,
    write_episode,
    flatten_subkeys,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


URL_MAP = {
    "http://localhost:8400": "openvla_oft",
    "http://localhost:8300": "upvla",
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


def _split_obs(processed_obs: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """unified obs sub-keys → (images, states) 분리."""
    images: Dict[str, np.ndarray] = {}
    states: Dict[str, np.ndarray] = {}
    for k, v in processed_obs.items():
        if k.startswith("observation.images."):
            images[k.split("observation.images.")[-1]] = v
        elif k.startswith("observation.state"):
            states[k] = np.asarray(v)
    return images, states


def _actions_to_dict(actions) -> Dict[str, np.ndarray]:
    """서버 응답(sub-key dict 또는 flat) → 첫 step action sub-key dict."""
    if isinstance(actions, dict):
        # 서버 N=1 → 각 키의 [0] 행만 사용
        return {k: np.asarray(v[0], dtype=np.float32) for k, v in actions.items()}
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    return {"_flat": arr}


def _predict(
    vla_client: VLAClient, processed_obs: Dict, instruction: str
) -> Dict[str, np.ndarray]:
    """unified obs sub-keys → /act 호출 → 첫 step action sub-key dict.

    openvla_oft 서버는 내부 queue 로 chunk 를 분할해 한 번에 1 step (N=1) 만 반환.
    따라서 매 env step 마다 /act 를 1 회 호출.
    """
    images, states = _split_obs(processed_obs)
    actions, _ = vla_client.predict(
        images=images, states=states or None, instruction=instruction
    )
    return _actions_to_dict(actions)


def _predict_with_features(
    vla_client: VLAClient, processed_obs: Dict, instruction: str
) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
    """SAFE 수집용: /act_with_features 호출 → (action sub-key dict, features|None).

    features 는 정책이 이번 step 에 새 추론을 돌렸을 때만 채워짐(그 외 None).
    """
    images, states = _split_obs(processed_obs)
    actions, features, _ = vla_client.predict_with_features(
        images=images, states=states or None, instruction=instruction
    )
    return _actions_to_dict(actions), features


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
    collector: Optional["SafeEpisodeCollector"] = None,
    task_name: str = "",
) -> Tuple[bool, List[np.ndarray], int, Dict]:
    """단일 LIBERO episode rollout.

    Returns:
        (success, frames, terminated_step, phase_info)
        success: deadline 안에 env `done=True` 가 왔는가.
        frames: replay 이미지 (record=True 일 때만 채움).
        terminated_step: 실제로 멈춘 step (effective; num_steps_wait 제외).
        phase_info: 수집 모드(collector!=None) + forced-order task 일 때 event-anchored
            phase 라벨(빈 dict=수집 아님/미등록 task). phase_scheme, phase_timeline(per
            env-step, list[str]), feature_phases(per latent, list[str]), event_steps,
            event_order.
    """
    vla_client.reset()
    env.reset()
    obs = env.set_init_state(initial_state)

    frames: List[np.ndarray] = []
    success = False
    effective_t = 0

    # 수집 모드일 때만 event-anchored phase 라벨러 부착(일반 eval=collector None→영향 없음).
    # forced-order task 만 event spec 보유 → 미등록 task(교환가능/단일)는 phase 라벨 생략(경고).
    # libero phase 라벨러는 S2 판정으로 archive 됨 (libero 축 폐기, 2026-08-07).
    # phase 라벨 없이 수집한다 — feature_phases 는 None 로 남는다.
    labeler = None
    feature_phases: List[str] = []  # 추론 발화 step마다 그 step의 phase label(latent와 정렬)

    # objects stabilize: dummy action 으로 NUM_STEPS_WAIT step 진행
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(DUMMY_ACTION.tolist())

    for t in range(max_steps):
        # 현재 obs(=예측에 쓰일 sim state)의 phase label 을 기록 — env.step 전 호출로 정렬 유지.
        cur_phase = labeler.step() if labeler is not None else None
        processed = obs_pipeline({TransitionKey.OBSERVATION: obs})
        processed_obs = processed[TransitionKey.OBSERVATION]
        if collector is not None:
            action_pred, features = _predict_with_features(
                vla_client, processed_obs, instruction
            )
            # 추론이 발화한 step(features 존재)만 SAFE latent record
            if features is not None and "_flat" not in action_pred:
                collector.record(
                    features,
                    action_vector=flatten_subkeys(action_pred, list(action_pred.keys())),
                    action={k: v.tolist() for k, v in action_pred.items()},
                )
                # 이 latent가 계산된 step의 phase label을 같은 순서로 적재(hidden_states와 1:1).
                if labeler is not None:
                    feature_phases.append(cur_phase)
        else:
            action_pred = _predict(vla_client, processed_obs, instruction)
        # _flat fallback (서버가 sub-key 가 아니라 flat 으로 응답한 경우) 은 ndarray 직접
        action_in = (
            action_pred["_flat"] if "_flat" in action_pred else action_pred
        )

        processed_act = action_pipeline({TransitionKey.ACTION: action_in})
        action = processed_act[TransitionKey.ACTION]
        action_list = action.tolist() if isinstance(action, np.ndarray) else list(action)

        try:
            obs, _, done, _ = env.step(action_list)
        except ValueError as e:
            # robosuite 가 "executing action in terminated episode" 를 raise 하는 경우.
            # long-horizon libero_10 에서 sim 이 unrecoverable terminated state 로 빠질 때 발생.
            # 해당 episode 는 실패로 마킹하고 다음 trial 로 진행 (전체 평가 중단 방지).
            logger.warning("env.step ValueError at t=%d: %s — marking as failure", t, e)
            effective_t = t + 1
            success = False
            break
        if record:
            frames.append(_replay_image(obs))

        effective_t = t + 1
        # LIBERO 진짜 성공 판정은 check_success() — env `done` 은 horizon(기본 1000) 기반이라
        # 짧은 rollout 에선 안 뜸. lerobot LiberoEnv 도 check_success() 를 사용.
        if env.check_success():
            success = effective_t <= success_deadline
            break
        if done:  # horizon 도달 (성공 못 함)
            break

    # off-by-one 보정: 루프 top 의 step() 은 각 예측 step 의 obs(=env.step 전) phase 만 본다.
    # success 는 마지막 env.step '직후' 판정되므로, 성공을 만든 터미널 이벤트(마지막 in/close 등)
    # 가 위 루프에서 평가되지 않는다 → 한 번 더 읽어 phase_timeline/max_phase 에 반영한다.
    # 이 step 은 latent 가 없으므로 feature_phases 에는 추가하지 않는다(정렬 유지).
    if labeler is not None:
        try:
            labeler.step()
        except Exception as e:  # 종료/오류 상태 env 에서 predicate 평가 실패 시 무시
            logger.warning("terminal labeler.step() skipped: %s", e)

    phase_info: Dict = {}
    if labeler is not None:
        phase_info = {
            "phase_scheme": "event_anchored",
            "phase_timeline": list(labeler.phase_timeline),  # list[str] per env-step
            "feature_phases": list(feature_phases),          # list[str] per recorded latent
            "event_steps": dict(labeler.event_steps),        # event key -> first-fire step
            "event_order": list(labeler.event_order_keys),
        }
    return success, frames, effective_t, phase_info


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
    safe_collect: bool = False,
    safe_output_dir: Optional[Path] = None,
    safe_run_id: str = "run",
    max_tasks: Optional[int] = None,
    task_names: Optional[List[str]] = None,
):
    np.random.seed(seed)

    # 첫 추론은 torch.compile warmup 으로 ~2분 걸릴 수 있음(cold serve) → timeout 넉넉히.
    vla_client = VLAClient(url=server_url, timeout=300)
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

    n_tasks_run = n_tasks if max_tasks is None else min(n_tasks, max_tasks)
    for task_id in range(n_tasks_run):
        task = task_suite.get_task(task_id)
        task_name = getattr(task, "name", f"task{task_id:02d}")
        # --task-names 필터: 지정 시 이름에 부분일치하는 task 만 평가(forced-order 타깃 등).
        if task_names and not any(tn in task_name for tn in task_names):
            continue
        env, task_description = _make_libero_env(task, resolution=image_size)
        initial_states = task_suite.get_task_init_states(task_id)

        task_episodes = 0
        task_successes = 0
        for trial in range(num_trials):
            init_state = initial_states[trial % len(initial_states)]
            collector = (
                SafeEpisodeCollector(
                    task_suite_name=task_suite_name,
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=trial,
                    env_name=f"libero/{task_suite_name}",
                )
                if safe_collect
                else None
            )
            success, frames, eff_t, phase_info = _rollout(
                env,
                task_description,
                init_state,
                vla_client,
                obs_pipeline,
                action_pipeline,
                max_steps=max_steps,
                success_deadline=success_deadline,
                record=True,
                collector=collector,
                task_name=task_name,
            )
            if collector is not None and len(collector) > 0:
                task_dir_name = getattr(task, "name", f"task{task_id:02d}")
                pkl_path = episode_path(
                    safe_output_dir, safe_run_id, task_dir_name, task_id, trial, success
                )
                # phase_info(BDDL sim-state phase 라벨)를 pkl payload에 동봉.
                write_episode(pkl_path, collector.payload(bool(success), **phase_info))
                # robocasa 번들 컨벤션: episode 영상을 pkl 과 동일 stem 으로 옆에 저장
                # (succ/fail 모두). split helper 가 {stem}.* 로 함께 복사.
                _save_video(frames, pkl_path.with_suffix(".mp4"))
                logger.info(
                    "[SAFE] wrote %s (%d latents) + mp4", pkl_path, len(collector)
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
    parser.add_argument(
        "--safe-collect",
        action="store_true",
        help="SAFE latent 수집 모드: /act_with_features 로 추론당 latent + episode pkl 저장.",
    )
    parser.add_argument(
        "--safe-output-dir",
        type=str,
        default=None,
        help="SAFE pkl 출력 base dir (예: outputs/eval/libero/pi05). "
        "하위에 rollouts_{run_id}/{task}/task{id}--ep{idx}--succ{0|1}.pkl 생성.",
    )
    parser.add_argument(
        "--safe-run-id", type=str, default="run", help="SAFE 수집 run 식별자."
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="suite 의 앞 N개 task 만 평가 (smoke/부분 수집용). 미지정 시 전체.",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        default=None,
        help="콤마구분 부분일치 필터: 이름에 매칭되는 task 만 평가 "
        "(예: 'KITCHEN_SCENE4,KITCHEN_SCENE6' 로 forced-order 만). 미지정 시 전체.",
    )
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

    if args.safe_collect and not args.safe_output_dir:
        raise ValueError("--safe-collect 사용 시 --safe-output-dir 필요")

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
        safe_collect=args.safe_collect,
        safe_output_dir=Path(args.safe_output_dir) if args.safe_output_dir else None,
        safe_run_id=args.safe_run_id,
        max_tasks=args.max_tasks,
        task_names=(
            [s.strip() for s in args.task_names.split(",") if s.strip()]
            if args.task_names
            else None
        ),
    )


if __name__ == "__main__":
    main()
