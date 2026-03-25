"""fywang/calvin-task-ABCD-D-lerobot validation split action replay 검증.

데이터셋 action을 CALVIN 시뮬레이터에 그대로 재생해서:
  1. 시뮬레이터 궤적과 기록 궤적의 EE position 오차 측정
  2. task oracle로 태스크 성공 판정

실행 (calvin 컨테이너):
    docker compose exec calvin python /temporal_vla/scripts/calvin_replay_eval.py \\
        --data-dir /temporal_vla/data/calvin_lerobot/fywang_ABCD_D \\
        --env-path /temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset/validation \\
        --num-episodes 20 \\
        --seed 42

    # 특정 에피소드 지정:
    docker compose exec calvin python /temporal_vla/scripts/calvin_replay_eval.py \\
        --data-dir /temporal_vla/data/calvin_lerobot/fywang_ABCD_D \\
        --env-path /temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset/validation \\
        --episodes 22966 22980 23001

주요 설계 결정:
  - get_env(env_path): validation 경로 직접 전달 (scene= kwarg는 OmegaConf.merge 반환값
    미사용 버그로 동작 안 함, play_table_env.py L285)
  - task oracle: merged_config.yaml의 tasks 섹션에서 직접 초기화
    (별도 tasks yaml 불필요 — merged_config에 34개 task 정의 포함)
  - task_index: parquet에 per-frame 저장 → [0]으로 에피소드 값 추출
  - gripper action: 마지막 dim 이산화 (>0 → 1.0, ≤0 → -1.0)
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
from omegaconf import OmegaConf

# Calvin env
sys.path.insert(0, "/temporal_vla/src/benchmarks/calvin/calvin_env")
from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

# 로컬 유틸
sys.path.insert(0, "/temporal_vla/scripts/utils")
from calvin_parquet_reader import (
    list_available_episodes,
    load_task_mapping,
    read_episode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# validation 에피소드 범위 (fywang 데이터셋 기준)
VALIDATION_START = 22966
VALIDATION_END = 24052


def _setup_egl() -> None:
    try:
        import os

        egl_id = get_egl_device_id(0)
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    except EglDeviceNotFoundError:
        pass


def _gripper_discretize(action: np.ndarray) -> np.ndarray:
    """gripper 차원(마지막) 이산화: >0 → 1.0, ≤0 → -1.0."""
    action = action.copy()
    action[-1] = 1.0 if action[-1] > 0 else -1.0
    return action


def replay_episode(
    env,
    task_oracle,
    actions: np.ndarray,
    robot_obs: np.ndarray,
    scene_obs: np.ndarray,
) -> Dict:
    """단일 에피소드 리플레이.

    Returns:
        success:        bool
        max_ee_error:   float  (미터)
        mean_ee_error:  float  (미터)
    """
    env.reset(robot_obs=robot_obs[0], scene_obs=scene_obs[0])
    start_info = env.get_info()  # scene_info 포함 (get_env가 use_scene_info=True 전달)

    ee_errors: List[float] = []
    T = len(actions)

    for t in range(T):
        action = _gripper_discretize(actions[t])
        obs, _, _, _ = env.step(action)

        if t + 1 < len(robot_obs):
            recorded_ee = robot_obs[t + 1, :3]
            sim_ee = obs["robot_obs"][:3]
            ee_errors.append(float(np.linalg.norm(recorded_ee - sim_ee)))

    current_info = env.get_info()
    # task_name은 자연어(389종) → oracle key(34종)와 불일치.
    # get_task_info()로 34개 전체 체크. achieved 집합을 로깅해 교차 검증.
    achieved = task_oracle.get_task_info(start_info, current_info)
    success = len(achieved) > 0

    return {
        "success": success,
        "achieved": achieved,  # 달성된 oracle task 집합 (교차 검증용)
        "max_ee_error": float(np.max(ee_errors)) if ee_errors else 0.0,
        "mean_ee_error": float(np.mean(ee_errors)) if ee_errors else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calvin LeRobot dataset action replay 검증"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="fywang_ABCD_D/ 루트 (meta/, data/ 포함)",
    )
    parser.add_argument(
        "--env-path",
        required=True,
        type=Path,
        help="calvin_debug_dataset/validation 경로 (.hydra/ 포함)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="무작위 샘플링할 에피소드 수 (기본: 20)",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        help="검증할 에피소드 index 직접 지정 (지정 시 --num-episodes 무시)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="validation 에피소드 전체 순서대로 실행 (--num-episodes 무시)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir
    env_path = args.env_path

    # ── task 매핑 로드 + name 형식 확인 ─────────────────────────────
    logger.info("meta/tasks.jsonl 로드 중...")
    task_mapping = load_task_mapping(data_dir)
    logger.info(f"task 총 {len(task_mapping)}개")
    logger.info("처음 5개 task (CALVIN oracle 키와 일치 여부 확인):")
    for k, v in list(task_mapping.items())[:5]:
        logger.info(f"  {k}: '{v}'")

    # ── 에피소드 선택 ────────────────────────────────────────────────
    if args.episodes:
        episode_indices = sorted(args.episodes)
    else:
        available = [
            ep
            for ep in list_available_episodes(data_dir)
            if VALIDATION_START <= ep <= VALIDATION_END
        ]
        if not available:
            logger.error(
                f"validation 에피소드({VALIDATION_START}~{VALIDATION_END})가 로컬에 없습니다.\n"
                "chunk-022(나머지), chunk-023, chunk-024 다운로드 필요."
            )
            sys.exit(1)
        if args.all:
            episode_indices = available  # 이미 오름차순 정렬됨
            logger.info(f"validation 에피소드 전체 {len(episode_indices)}개 실행")
        else:
            rng = random.Random(args.seed)
            n = min(args.num_episodes, len(available))
            episode_indices = sorted(rng.sample(available, n))
            logger.info(
                f"validation 에피소드 {len(available)}개 중 {n}개 샘플링: {episode_indices}"
            )

    # ── env + task oracle 초기화 ─────────────────────────────────────
    _setup_egl()
    logger.info(f"env 로드: {env_path}")
    env = get_env(str(env_path), show_gui=False)

    # task oracle: merged_config.yaml의 tasks 섹션 사용
    # (34개 task 정의 포함, 별도 tasks yaml 불필요)
    render_conf = OmegaConf.load(env_path / ".hydra" / "merged_config.yaml")
    task_oracle = hydra.utils.instantiate(render_conf.tasks)
    logger.info("env + task oracle 초기화 완료")

    # ── 리플레이 실행 ────────────────────────────────────────────────
    results = []
    for ep_idx in episode_indices:
        data = read_episode(data_dir, ep_idx)
        task_name = task_mapping.get(
            data["task_index"], f"unknown_{data['task_index']}"
        )

        result = replay_episode(
            env,
            task_oracle,
            data["actions"],
            data["robot_obs"],
            data["scene_obs"],
        )
        result["episode_index"] = ep_idx
        result["task_name"] = task_name
        results.append(result)

        status = "SUCCESS" if result["success"] else "FAIL   "
        logger.info(
            f"Ep {ep_idx:05d} [{task_name:30s}]: {status}  "
            f"max_ee={result['max_ee_error']:.4f}m  "
            f"mean_ee={result['mean_ee_error']:.4f}m  "
            f"achieved={result['achieved']}"
        )

    # ── 결과 집계 ────────────────────────────────────────────────────
    if not results:
        logger.warning("결과 없음")
        return

    n_success = sum(r["success"] for r in results)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"Total: {n_success}/{n_total} SUCCESS ({100*n_success/n_total:.1f}%)")
    print(
        f"EE error  mean-of-max: {np.mean([r['max_ee_error'] for r in results]):.4f}m"
        f"  mean-of-mean: {np.mean([r['mean_ee_error'] for r in results]):.4f}m"
    )

    # 판단 기준
    rate = n_success / n_total
    if rate >= 0.9:
        verdict = "GOLD — 학습 진행 가능"
    elif rate >= 0.5:
        verdict = "PARTIAL — 패턴 분석 후 필터링"
    else:
        verdict = "FAIL — 좌표계/정규화 오류 의심, 변환 코드 재검토"
    print(f"판정: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
