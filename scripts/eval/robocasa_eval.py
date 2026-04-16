"""
VLA closed-loop 평가 스크립트 (RoboCasa, 모델 무관).

통일 API를 따르는 어떤 VLA 서버든 --vla-server URL만 바꾸면 평가 가능.

사용법:
  # DreamVLA로 평가
  python scripts/eval/robocasa_eval.py \
    --task TurnOnMicrowave \
    --vla-server http://localhost:8200

  # X-VLA로 평가 (같은 스크립트, URL만 변경)
  python scripts/eval/robocasa_eval.py \
    --task TurnOnMicrowave \
    --vla-server http://localhost:8100

  # 태스크셋 평가
  python scripts/eval/robocasa_eval.py \
    --task-set pretrain50 \
    --vla-server http://localhost:8200 \
    --output-dir /temporal_vla/outputs/vla_eval

  # 태스크 리스트 확인
  python scripts/eval/robocasa_eval.py --list-tasks

출력 구조 (--output-dir 지정 시):
  {output_dir}/
    {TaskName}.json   ← 태스크 완료마다 즉시 저장
    summary.json      ← 전체 완료 후 종합 요약
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from path_setup import configure_repo_paths

configure_repo_paths(include_script_utils=True, include_robocasa=True)

import numpy as np
import robosuite
from robosuite import load_part_controller_config
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from tqdm import tqdm

from src.utils.common.logger import create_module_logger
from src.processor.factory import make_robocasa_processors
from src.processor.types import TransitionKey

logger = create_module_logger("robocasa_vla_eval")

STATIC_CAM = "robot0_agentview_left"
WRIST_CAM = "robot0_eye_in_hand"
CAMERA_NAMES = [STATIC_CAM, "robot0_agentview_right", WRIST_CAM]
IMAGE_SIZE = 224


def create_eval_env(
    env_name,
    robots="PandaMobile",
    controllers="OSC_POSE",
    camera_names=None,
    camera_widths=128,
    camera_heights=128,
    seed=None,
    obj_instance_split="pretrain",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    if camera_names is None:
        camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    controller_configs = load_part_controller_config(default_controller=controllers)
    return robosuite.make(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )


# ── 평가 ──────────────────────────────────────────────────────────────────

def _check_success(env) -> bool:
    result = env._check_success()
    return bool(result.get("task", False)) if isinstance(result, dict) else bool(result)


def run_vla_rollouts(
    env,
    vla_client,
    obs_pipeline,
    action_pipeline,
    num_rollouts: int,
    num_steps: int,
    video_path: str | None = None,
) -> dict:
    """robocasa run_random_rollouts 패턴을 따르되, 랜덤 액션 대신 VLA 예측을 사용."""
    video_writer = None
    if video_path is not None:
        try:
            import imageio
            video_writer = imageio.get_writer(video_path, fps=20)
        except ImportError:
            logger.error("영상 저장을 위해 imageio 패키지가 필요합니다. (pip install imageio[ffmpeg])")
            video_writer = None

    results = []
    num_success = 0

    for rollout_i in tqdm(range(num_rollouts), desc="롤아웃 평가"):
        obs = env.reset()
        if rollout_i == 0:
            logger.info(f"Available cameras in env: {env.sim.model.camera_names}")
        instruction = env.get_ep_meta().get("lang", "")
        vla_client.reset()

        latencies = []
        first_success_step = None

        for step_i in range(num_steps):
            # obs → processor → VLAClient 호출
            processed = obs_pipeline({TransitionKey.OBSERVATION: obs})
            processed_obs = processed[TransitionKey.OBSERVATION]

            images = {}
            states = {}
            for k, v in processed_obs.items():
                if k.startswith("observation.images."):
                    cam_name = k.split("observation.images.")[-1]
                    images[cam_name] = v
                elif k.startswith("observation.state"):
                    states[k] = v

            actions, latency_ms = vla_client.predict(images, states or None, instruction)
            # sub-keyed dict 또는 flat ndarray 양쪽 지원
            if isinstance(actions, dict):
                raw_action = {k: v[0] for k, v in actions.items()}
            else:
                raw_action = actions[0]
            latencies.append(latency_ms)

            # VLA 7D action → PandaMobile 12D (processor)
            processed_act = action_pipeline({TransitionKey.ACTION: raw_action})
            action = processed_act[TransitionKey.ACTION]

            obs, _, done, _ = env.step(action)

            if video_writer is not None:
                # 렌더링 시 카메라 이름이 환경에 따라 다를 수 있으므로 예외 처리
                # 기본적으로 robot0_frontview 또는 robot0_agentview_center 사용
                cam_name = "robot0_frontview" if "robot0_frontview" in env.sim.model.camera_names else "robot0_agentview_center"
                # 만약 위 카메라들이 없으면 첫 번째 카메라 사용
                if cam_name not in env.sim.model.camera_names and len(env.sim.model.camera_names) > 0:
                    cam_name = env.sim.model.camera_names[0]

                video_img = env.sim.render(height=512, width=512, camera_name=cam_name)[::-1]
                video_writer.append_data(video_img)

            if _check_success(env):
                if first_success_step is None:
                    first_success_step = step_i
                num_success += 1
                break

        results.append({
            "rollout": rollout_i,
            "success": first_success_step is not None,
            "first_success_step": first_success_step,
            "steps": step_i + 1,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "instruction": instruction,
        })

    if video_writer is not None:
        video_writer.close()
        logger.info("영상 저장: %s", video_path)

    return {
        "num_success": num_success,
        "num_rollouts": num_rollouts,
        "success_rate": num_success / num_rollouts if num_rollouts > 0 else 0.0,
        "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in results])) if results else 0.0,
        "rollouts": results,
    }


def evaluate_task(
    task_name: str,
    vla_client,
    obs_pipeline,
    action_pipeline,
    num_rollouts: int = 50,
    num_steps: int = 400,
    video_path: str | None = None,
    seed: int | None = None,
) -> dict:
    """단일 태스크 평가."""
    logger.info("태스크: %s (rollouts=%d, steps=%d)", task_name, num_rollouts, num_steps)
    env = create_eval_env(
        env_name=task_name,
        camera_names=CAMERA_NAMES,
        camera_widths=IMAGE_SIZE,
        camera_heights=IMAGE_SIZE,
        seed=seed,
    )
    try:
        info = run_vla_rollouts(
            env, vla_client, obs_pipeline, action_pipeline,
            num_rollouts, num_steps, video_path,
        )
    finally:
        env.close()

    return {
        "task": task_name,
        "mode": "vla",
        **info,
    }


# ── 결과 출력/저장 ────────────────────────────────────────────────────────

def log_summary(summary: dict):
    rate = summary["success_rate"]
    lines = [
        "=" * 50,
        f"태스크:        {summary['task']}",
        f"총 롤아웃:     {summary['num_rollouts']}",
        f"성공:          {summary['num_success']} / {summary['num_rollouts']}",
        f"성공률:        {rate:.1%}",
        f"평균 레이턴시: {summary['mean_latency_ms']:.1f} ms",
        "=" * 50,
    ]
    msg = "\n".join(lines)
    logger.info(msg) if rate > 0.5 else logger.warning(msg)


def log_all_summary(summaries: list[dict]):
    total_rollouts = sum(s["num_rollouts"] for s in summaries)
    total_ok = sum(s["num_success"] for s in summaries)
    overall = total_ok / total_rollouts if total_rollouts > 0 else 0.0

    lines = [
        "=" * 70,
        "  전체 평가 결과 요약",
        "=" * 70,
        f"{'태스크':<45} {'성공/전체':>12} {'성공률':>8}",
        "-" * 70,
    ]
    for s in summaries:
        lines.append(
            f"  {s['task']:<43} {s['num_success']}/{s['num_rollouts']:>8} {s['success_rate']:>7.1%}"
        )
    lines += [
        "-" * 70,
        f"  {'전체 합계':<43} {total_ok}/{total_rollouts:>8} {overall:>7.1%}",
        "=" * 70,
    ]
    msg = "\n".join(lines)
    logger.info(msg) if overall >= 0.5 else logger.warning(msg)


def save_task_result(summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{summary['task']}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("저장: %s", path)


def save_summary(all_summaries: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rollouts = sum(s["num_rollouts"] for s in all_summaries)
    total_ok = sum(s["num_success"] for s in all_summaries)
    result = {
        "total_rollouts": total_rollouts,
        "total_success": total_ok,
        "overall_success_rate": total_ok / max(total_rollouts, 1),
        "tasks": [
            {
                "task": s["task"],
                "num_rollouts": s["num_rollouts"],
                "num_success": s["num_success"],
                "success_rate": s["success_rate"],
                "mean_latency_ms": s.get("mean_latency_ms", 0.0),
            }
            for s in all_summaries
        ],
    }
    path = output_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("전체 요약 저장: %s", path)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VLA closed-loop 평가 (robocasa, 모델 무관)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--task", type=str, help="평가할 태스크 이름 (예: TurnOnMicrowave)")
    mode.add_argument("--task-set", type=str,
                       help=f"태스크셋 이름: {', '.join(TASK_SET_REGISTRY.keys())}")
    mode.add_argument("--list-tasks", action="store_true",
                       help="사용 가능한 태스크셋 목록 출력")

    p.add_argument("--vla-server", type=str, default="http://localhost:8200",
                   help="DreamVLA 서버 URL")
    p.add_argument("--num-rollouts", type=int, default=50,
                   help="태스크당 롤아웃 수 (기본: 50)")
    p.add_argument("--num-steps", type=int, default=400,
                   help="롤아웃당 최대 스텝 수 (기본: 400)")
    p.add_argument("--seed", type=int, default=None, help="환경 시드")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--video-dir", type=str, default=None,
                   help="롤아웃 영상 저장 디렉토리")
    p.add_argument("--resume", action="store_true",
                   help="이미 결과 JSON이 있는 태스크 건너뛰기")
    p.add_argument("--three-cameras", action="store_true",
                   help="3-camera 모드 (left+right agentview + wrist). pi0.5 등 3-camera 모델용")
    return p


def main():
    args = build_parser().parse_args()

    if args.list_tasks:
        for name, tasks in TASK_SET_REGISTRY.items():
            print(f"{name} ({len(tasks)} tasks): {', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''}")
        return

    if not args.task and not args.task_set:
        print("--task 또는 --task-set 중 하나를 지정하세요. --list-tasks로 목록 확인.")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    video_dir = Path(args.video_dir) if args.video_dir else None

    from vla_client import VLAClient
    vla_client = VLAClient(url=args.vla_server, timeout=300.0)
    logger.info("VLA 서버 연결 대기 중: %s", args.vla_server)
    server_info = vla_client.wait_until_ready()
    logger.info("VLA 서버 연결 완료: %s", server_info)

    # Processor pipeline (벤치마크별 obs/action 변환)
    obs_pipeline, action_pipeline = make_robocasa_processors(
        static_cam2="robot0_agentview_right" if args.three_cameras else None,
    )

    if args.task:
        tasks = [args.task]
    else:
        if args.task_set not in TASK_SET_REGISTRY:
            logger.error("알 수 없는 태스크셋: %s (--list-tasks로 확인)", args.task_set)
            sys.exit(1)
        tasks = TASK_SET_REGISTRY[args.task_set]

    if args.resume and output_dir and output_dir.exists():
        existing = {p.stem for p in output_dir.glob("*.json") if p.stem != "summary"}
        before = len(tasks)
        tasks = [t for t in tasks if t not in existing]
        logger.info("--resume: %d/%d개 태스크 남음", len(tasks), before)

    logger.info("평가 대상: %d개 태스크", len(tasks))

    all_summaries = []
    for i, task_name in enumerate(tasks):
        logger.info("[%d/%d] %s", i + 1, len(tasks), task_name)
        video_path = str(video_dir / f"{task_name}.mp4") if video_dir else None
        if video_dir:
            video_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = evaluate_task(
                task_name, vla_client,
                obs_pipeline, action_pipeline,
                num_rollouts=args.num_rollouts,
                num_steps=args.num_steps,
                video_path=video_path,
                seed=args.seed,
            )
            log_summary(summary)
            all_summaries.append(summary)
            if output_dir:
                save_task_result(summary, output_dir)
        except KeyboardInterrupt:
            logger.warning("평가 중단됨.")
            break
        except Exception:
            logger.error("실패:\n%s", traceback.format_exc())

    if len(all_summaries) > 1:
        log_all_summary(all_summaries)
    if output_dir and all_summaries:
        save_summary(all_summaries, output_dir)


if __name__ == "__main__":
    main()
