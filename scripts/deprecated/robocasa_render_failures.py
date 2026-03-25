"""
평가 결과 JSON에서 실패한 에피소드만 영상으로 렌더링하는 스크립트

사용법:
  # 평가 결과 디렉터리에서 실패 에피소드 렌더링 (태스크별 JSON 자동 탐색)
  python scripts/robocasa_render_failures.py \
    --result /temporal_vla/outputs/robocasa/eval \
    --output-dir /temporal_vla/outputs/failures

  # 단일 태스크 결과 JSON에서 실패 에피소드 렌더링
  python scripts/robocasa_render_failures.py \
    --result /temporal_vla/outputs/robocasa/eval/CloseBlenderLid.json \
    --output-dir /temporal_vla/outputs/failures

  # 데이터셋 직접 지정 + 에피소드 번호 수동 지정
  python scripts/robocasa_render_failures.py \
    --dataset /temporal_vla/data/datasets/v1.0/pretrain/atomic/TurnOnToaster/20250820/lerobot \
    --episodes 3,7,12,25 \
    --output-dir /temporal_vla/outputs/failures

  # 카메라 변경 (기본: robot0_agentview_center, robot0_eye_in_hand)
  python scripts/robocasa_render_failures.py \
    --result /temporal_vla/outputs/robocasa/eval \
    --cameras agentview robot0_eye_in_hand frontview

  # 에피소드 수 제한 (실패가 많을 때 앞에서 N개만)
  python scripts/robocasa_render_failures.py \
    --result /temporal_vla/outputs/robocasa/eval \
    --max-episodes 10

  # 병렬 실행 (태스크 단위 병렬화, worker 수 지정)
  python scripts/robocasa_render_failures.py \
    --result /temporal_vla/outputs/robocasa/eval \
    --output-dir /temporal_vla/outputs/failures \
    --workers 4
"""

import sys
from multiprocessing import get_context
from pathlib import Path

import argparse
import json

from path_setup import configure_repo_paths

configure_repo_paths(include_robocasa=True)

import imageio
import robocasa.utils.lerobot_utils as LU
import robosuite
from robocasa.scripts.dataset_scripts.playback_dataset import playback_trajectory_with_env
from tqdm import tqdm

from src.utils.common.logger import create_module_logger

logger = create_module_logger("robocasa_render_failures")


def _make_env_kwargs(dataset_path: Path) -> dict:
    meta = LU.get_env_metadata(dataset_path)
    kwargs = meta["env_kwargs"]
    kwargs["env_name"] = meta["env_name"]
    kwargs.update(
        has_renderer=False,
        renderer="mujoco",
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    return kwargs


def load_summaries(result_path: Path) -> list[dict]:
    """결과 파일 또는 디렉터리에서 에피소드 정보가 포함된 summary 목록을 로드."""
    if result_path.is_dir():
        summaries = []
        for f in sorted(result_path.glob("*.json")):
            if f.name.lower().endswith("summary.json"):
                continue
            with open(f) as fp:
                data = json.load(fp)
            if "episodes" in data:
                summaries.append(data)
        if not summaries:
            logger.error("디렉터리에 에피소드 정보가 포함된 JSON 파일 없음: %s", result_path)
            sys.exit(1)
        return summaries
    else:
        with open(result_path) as f:
            data = json.load(f)
        # 구 형식 (summaries 배열) 또는 단일 태스크 JSON
        return data.get("summaries") or [data]


def render_episode(
    env,
    dataset_path: Path,
    ep_num: int,
    output_path: Path,
    camera_names: list[str],
    video_skip: int = 1,
    camera_height: int = 512,
    camera_width: int = 512,
):
    """에피소드를 상태 재생으로 렌더링해서 mp4로 저장."""
    states = LU.get_episode_states(dataset_path, ep_num)
    initial_state = {
        "states": states[0],
        "model": LU.get_episode_model_xml(dataset_path, ep_num),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset_path, ep_num)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=20) as writer:
        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=None,
            render=False,
            video_writer=writer,
            video_skip=video_skip,
            camera_names=camera_names,
            camera_height=camera_height,
            camera_width=camera_width,
        )


def _render_task_worker(args: tuple):
    """병렬 실행을 위한 태스크 단위 워커 함수."""
    summary, output_dir, camera_names, max_episodes = args
    dataset_path = Path(summary["dataset"])
    task_name = summary["task"]
    failed_eps = [
        r["ep_num"] for r in summary.get("episodes", []) if not r["success"]
    ]

    if not failed_eps:
        return task_name, 0, 0

    if max_episodes:
        failed_eps = failed_eps[:max_episodes]

    env_kwargs = _make_env_kwargs(dataset_path)
    env = robosuite.make(**env_kwargs)

    task_dir = Path(output_dir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    rendered, errors = 0, 0
    for ep_num in failed_eps:
        output_path = task_dir / f"ep{ep_num:04d}.mp4"
        try:
            render_episode(env, dataset_path, ep_num, output_path, camera_names)
            rendered += 1
        except Exception as e:
            logger.error("[%s] ep %d 렌더링 실패: %s", task_name, ep_num, e)
            errors += 1

    env.close()
    logger.info("[%s] 완료: %d개 렌더링, %d개 에러", task_name, rendered, errors)
    return task_name, rendered, errors


def render_failures_from_summaries(
    summaries: list[dict],
    output_dir: Path,
    camera_names: list[str],
    max_episodes: int = None,
    workers: int = 1,
):
    """summary 목록에서 실패 에피소드들을 렌더링."""
    # 실패 에피소드가 있는 태스크만 필터링
    tasks_to_render = []
    for summary in summaries:
        failed_eps = [
            r["ep_num"] for r in summary.get("episodes", []) if not r["success"]
        ]
        if failed_eps:
            n = min(len(failed_eps), max_episodes) if max_episodes else len(failed_eps)
            logger.info("[%s] 실패 에피소드 %d개 예정", summary["task"], n)
            tasks_to_render.append(summary)
        else:
            logger.info("[%s] 실패 에피소드 없음, 건너뜀", summary["task"])

    if not tasks_to_render:
        logger.info("렌더링할 실패 에피소드가 없습니다.")
        return

    logger.info("총 %d개 태스크, workers=%d", len(tasks_to_render), workers)

    worker_args = [
        (s, str(output_dir), camera_names, max_episodes) for s in tasks_to_render
    ]

    if workers <= 1:
        for args in worker_args:
            _render_task_worker(args)
    else:
        ctx = get_context("spawn")
        total_rendered, total_errors = 0, 0
        with ctx.Pool(processes=workers) as pool:
            for task_name, rendered, errors in pool.imap_unordered(
                _render_task_worker, worker_args
            ):
                total_rendered += rendered
                total_errors += errors
                logger.info(
                    "[%s] 완료 (누적: %d개 렌더링, %d개 에러)",
                    task_name, total_rendered, total_errors,
                )
        logger.info("전체 완료: %d개 렌더링, %d개 에러", total_rendered, total_errors)


def render_from_episodes(
    dataset_path: Path,
    episodes: list[int],
    output_dir: Path,
    camera_names: list[str],
):
    """에피소드 번호를 직접 받아 렌더링."""
    env_kwargs = _make_env_kwargs(dataset_path)
    env = robosuite.make(**env_kwargs)

    task_dir = output_dir / env_kwargs["env_name"]
    task_dir.mkdir(parents=True, exist_ok=True)

    for ep_num in tqdm(episodes, desc="렌더링"):
        output_path = task_dir / f"ep{ep_num:04d}.mp4"
        try:
            render_episode(env, dataset_path, ep_num, output_path, camera_names)
            logger.info("저장: %s", output_path)
        except Exception as e:
            logger.error("ep %d 렌더링 실패: %s", ep_num, e)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="실패 에피소드 영상 렌더링")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--result", type=str,
                     help="평가 결과 JSON 파일 또는 태스크별 JSON이 있는 디렉터리 경로")
    src.add_argument("--dataset", type=str,
                     help="lerobot 데이터셋 경로 (--episodes와 함께 사용)")

    parser.add_argument("--episodes", type=str, default=None,
                        help="렌더링할 에피소드 번호 목록 (쉼표 구분, --dataset 모드 전용). 예: 3,7,12")
    parser.add_argument("--output-dir", type=str,
                        default="/temporal_vla/outputs/failures",
                        help="영상 저장 디렉토리 (기본: /temporal_vla/outputs/failures)")
    parser.add_argument("--cameras", nargs="+",
                        default=["robot0_agentview_center", "robot0_eye_in_hand"],
                        help="렌더링할 카메라 이름 목록 (기본: robot0_agentview_center robot0_eye_in_hand)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="태스크당 렌더링할 실패 에피소드 최대 수 (--result 모드 전용)")
    parser.add_argument("--height", type=int, default=512, help="영상 높이 픽셀")
    parser.add_argument("--width", type=int, default=512, help="영상 너비 픽셀")
    parser.add_argument("--workers", type=int, default=1,
                        help="병렬 워커 수 (태스크 단위 병렬화, 기본: 1)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.result:
        result_path = Path(args.result)
        if not result_path.exists():
            logger.error("결과 경로 없음: %s", result_path)
            sys.exit(1)
        summaries = load_summaries(result_path)
        logger.info("로드된 태스크 수: %d", len(summaries))
        render_failures_from_summaries(
            summaries, output_dir, args.cameras, args.max_episodes, args.workers
        )

    else:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error("데이터셋 경로 없음: %s", dataset_path)
            sys.exit(1)
        if not args.episodes:
            logger.error("--dataset 모드에서는 --episodes 가 필요합니다.")
            sys.exit(1)
        episodes = [int(e.strip()) for e in args.episodes.split(",")]
        render_from_episodes(dataset_path, episodes, output_dir, args.cameras)

    logger.info("완료. 저장 위치: %s", output_dir)


if __name__ == "__main__":
    main()
