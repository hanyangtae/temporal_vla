"""
RoboCasa 데이터셋 재생 + 성공률 평가 스크립트

사용법:
  # 단일 데이터셋 (빠른 모드: 최종 상태 체크)
  python scripts/robocasa_playback_eval.py \
    --dataset /temporal_vla/data/datasets/v1.0/pretrain/atomic/TurnOnToaster/20250820/lerobot

  # 액션 재생 모드 (open-loop, 궤적 발산 확인용)
  python scripts/robocasa_playback_eval.py --dataset <path> --use-actions

  # 전체 atomic 태스크 병렬 평가
  python scripts/robocasa_playback_eval.py --all --split pretrain --task-type atomic --workers 4

  # 결과를 JSON으로 저장
  python scripts/robocasa_playback_eval.py --all --split pretrain --output /temporal_vla/outputs/eval.json
"""

import sys
from pathlib import Path

import argparse
import json
import os
import traceback

from src.utils.common.logger import create_module_logger

# robocasa_eval 모듈은 logging.getLogger(__name__)을 쓰는데,
# sys.path 경유 임포트라 __name__ == "robocasa_eval" 이 된다.
create_module_logger("robocasa_eval")
logger = create_module_logger("robocasa_playback_eval")

from robocasa_eval import (
    evaluate_dataset,
    find_all_datasets,
    log_all_summary,
    log_memory_info,
    log_summary,
    run_all_datasets,
    safe_worker_count,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoboCasa 데이터셋 성공률 평가")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", type=str, help="평가할 lerobot 데이터셋 경로 (단일)")
    mode.add_argument("--all", action="store_true", help="지정 split의 모든 데이터셋 평가")

    p.add_argument("--base-path", default="/temporal_vla/data/datasets/v1.0",
                   help="데이터셋 루트 경로")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--task-type", default=None, choices=["atomic", "composite"])
    p.add_argument("--n", type=int, default=None, help="태스크당 평가 에피소드 수 (기본: 전체)")
    p.add_argument("--use-actions", action="store_true", help="open-loop 액션 재생 모드")
    p.add_argument("--output", type=str, default=None, help="결과 JSON 저장 경로")
    p.add_argument("--workers", type=int, default=1,
                   help="병렬 워커 수 (0=CPU 코어 수 자동, --all: 데이터셋 단위, 단일: 에피소드 단위)")
    p.add_argument("--reserve-gb", type=float, default=4.0,
                   help="OS용 예약 메모리 GB (기본: 4.0)")
    p.add_argument("--env-mem-gb", type=float, default=1.5,
                   help="env 1개당 예상 메모리 GB (기본: 1.5)")
    return p


def save_results(all_summaries: list[dict], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = (
        all_summaries[0] if len(all_summaries) == 1
        else {
            "summaries": all_summaries,
            "total_episodes": sum(s["total_episodes"] for s in all_summaries),
            "total_success": sum(s["n_success"] for s in all_summaries),
            "overall_success_rate": (
                sum(s["n_success"] for s in all_summaries)
                / max(sum(s["total_episodes"] for s in all_summaries), 1)
            ),
        }
    )
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("결과 저장됨: %s", path)


def main():
    args = build_parser().parse_args()

    workers = safe_worker_count(
        args.workers if args.workers > 0 else os.cpu_count(),
        env_mem_gb=args.env_mem_gb,
        reserve_gb=args.reserve_gb,
    )
    log_memory_info(workers)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error("경로가 존재하지 않음: %s", dataset_path)
            sys.exit(1)
        try:
            summary = evaluate_dataset(
                dataset_path, n=args.n, use_actions=args.use_actions, workers=workers
            )
        except KeyboardInterrupt:
            logger.warning("평가 중단됨.")
            sys.exit(0)
        except Exception:
            logger.error("평가 실패:\n%s", traceback.format_exc())
            sys.exit(1)
        log_summary(summary)
        all_summaries = [summary]

    else:
        base = Path(args.base_path) / args.split
        if not base.exists():
            logger.error("경로가 존재하지 않음: %s", base)
            sys.exit(1)
        datasets = find_all_datasets(base, args.task_type)
        if not datasets:
            logger.error("데이터셋 없음: %s", base)
            sys.exit(1)

        logger.info("총 %d개 데이터셋 평가 시작 (워커: %d)", len(datasets), workers)
        all_summaries, failed = run_all_datasets(
            datasets, n=args.n, use_actions=args.use_actions, workers=workers
        )

        if len(all_summaries) > 1:
            log_all_summary(all_summaries)
        if failed:
            logger.error("실패한 데이터셋 %d개:", len(failed))
            for f in failed:
                logger.error("  %s", f["dataset"])

    if args.output and all_summaries:
        save_results(all_summaries, args.output)


if __name__ == "__main__":
    main()
