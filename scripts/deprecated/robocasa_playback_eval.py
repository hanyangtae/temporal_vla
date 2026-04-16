"""
RoboCasa 데이터셋 재생 + 성공률 평가 스크립트

사용법:
  # 단일 데이터셋 (빠른 모드: 최종 상태 체크)
  python scripts/robocasa_playback_eval.py \
    --dataset /temporal_vla/data/datasets/v1.0/pretrain/atomic/TurnOnToaster/20250820/lerobot

  # 액션 재생 모드 (open-loop, 궤적 발산 확인용)
  python scripts/robocasa_playback_eval.py --dataset <path> --use-actions

  # 전체 atomic 태스크 병렬 평가 + 결과 저장
  python scripts/robocasa_playback_eval.py \
    --all --split pretrain --task-type atomic --workers 4 \
    --output-dir /temporal_vla/outputs/robocasa/eval

출력 구조 (--output-dir 지정 시):
  {output_dir}/
    {TaskName}.json   ← 태스크 완료마다 즉시 저장
    summary.json      ← 전체 완료 후 종합 요약
"""

import sys
from pathlib import Path

import argparse
import json
import os
import traceback

from path_setup import configure_repo_paths

configure_repo_paths(include_script_utils=True, include_robocasa=True)

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
from vla_client import DreamVLAClient


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
    p.add_argument("--vla-server", type=str, default=None,
                   help="DreamVLA 서버 URL (예: http://localhost:8200). 지정 시 closed-loop VLA 평가 모드")
    p.add_argument("--output-dir", type=str, default=None,
                   help="결과 저장 디렉터리 (태스크별 JSON + summary.json 생성)")
    p.add_argument("--workers", type=int, default=1,
                   help="병렬 워커 수 (0=CPU 코어 수 자동, --all: 데이터셋 단위, 단일: 에피소드 단위)")
    p.add_argument("--reserve-gb", type=float, default=4.0,
                   help="OS용 예약 메모리 GB (기본: 4.0)")
    p.add_argument("--env-mem-gb", type=float, default=1.5,
                   help="env 1개당 예상 메모리 GB (기본: 1.5)")
    p.add_argument("--resume", action="store_true",
                   help="output-dir에 이미 결과 JSON이 있는 태스크는 건너뛰고 이어서 평가")
    return p


def save_task_result(summary: dict, output_dir: Path):
    """태스크 하나의 결과를 {output_dir}/{TaskName}.json 으로 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{summary['task']}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("태스크 결과 저장: %s", path)


def save_summary(all_summaries: list[dict], output_dir: Path):
    """전체 요약을 {output_dir}/summary.json 으로 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_eps = sum(s["total_episodes"] for s in all_summaries)
    total_ok = sum(s["n_success"] for s in all_summaries)
    result = {
        "total_episodes": total_eps,
        "total_success": total_ok,
        "overall_success_rate": total_ok / max(total_eps, 1),
        "tasks": [
            {
                "task": s["task"],
                "dataset": s["dataset"],
                "total_episodes": s["total_episodes"],
                "n_success": s["n_success"],
                "success_rate": s["success_rate"],
            }
            for s in all_summaries
        ],
    }
    path = output_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("전체 요약 저장: %s", path)


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    vla_client = None
    if args.vla_server:
        vla_client = DreamVLAClient(url=args.vla_server)
        logger.info("DreamVLA 서버 연결 대기 중: %s", args.vla_server)
        vla_client.wait_until_ready()
        logger.info("DreamVLA 서버 연결 완료.")

    workers = safe_worker_count(
        args.workers if args.workers > 0 else os.cpu_count(),
        env_mem_gb=args.env_mem_gb,
        reserve_gb=args.reserve_gb,
    )
    if vla_client is not None:
        workers = 1  # VLA 평가는 서버 stateful → sequential 강제
    log_memory_info(workers)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error("경로가 존재하지 않음: %s", dataset_path)
            sys.exit(1)
        try:
            summary = evaluate_dataset(
                dataset_path, n=args.n, use_actions=args.use_actions,
                workers=workers, vla_client=vla_client,
            )
        except KeyboardInterrupt:
            logger.warning("평가 중단됨.")
            sys.exit(0)
        except Exception:
            logger.error("평가 실패:\n%s", traceback.format_exc())
            sys.exit(1)
        log_summary(summary)
        all_summaries = [summary]

        if output_dir:
            save_task_result(summary, output_dir)

    else:
        base = Path(args.base_path) / args.split
        if not base.exists():
            logger.error("경로가 존재하지 않음: %s", base)
            sys.exit(1)
        datasets = find_all_datasets(base, args.task_type)
        if not datasets:
            logger.error("데이터셋 없음: %s", base)
            sys.exit(1)

        # --resume: 이미 결과가 있는 태스크 건너뛰기
        if args.resume and output_dir and output_dir.exists():
            existing = {p.stem for p in output_dir.glob("*.json") if not p.stem.endswith("summary")}
            before = len(datasets)
            datasets = [d for d in datasets if d.parts[-3] not in existing]
            skipped = before - len(datasets)
            if skipped:
                logger.info("--resume: 이미 완료된 %d개 태스크 건너뜀 (%d개 남음)", skipped, len(datasets))
                # 기존 결과도 로드해서 최종 summary에 포함
                for json_path in sorted(output_dir.glob("*.json")):
                    if json_path.stem.endswith("summary"):
                        continue
                    if json_path.stem in existing:
                        with open(json_path) as fp:
                            existing_summary = json.load(fp)
                        # episodes 필드 제거해서 메모리 절약
                        existing_summary.pop("episodes", None)

            if not datasets:
                logger.info("모든 태스크가 이미 완료됨. summary만 갱신합니다.")
                # 기존 결과 로드 후 summary 재생성
                all_summaries = []
                for json_path in sorted(output_dir.glob("*.json")):
                    if json_path.stem.endswith("summary"):
                        continue
                    with open(json_path) as fp:
                        all_summaries.append(json.load(fp))
                if len(all_summaries) > 1:
                    log_all_summary(all_summaries)
                    save_summary(all_summaries, output_dir)
                return

        total = len(datasets)
        logger.info("총 %d개 데이터셋 평가 시작 (워커: %d)", total, workers)

        def on_task_done(summary: dict, completed: list[dict]):
            if output_dir is None:
                return
            save_task_result(summary, output_dir)
            logger.info("진행 [%d/%d]", len(completed), total)

        all_summaries, failed = run_all_datasets(
            datasets,
            n=args.n,
            use_actions=args.use_actions,
            workers=workers,
            task_callback=on_task_done,
            vla_client=vla_client,
        )

        # --resume 시 기존 결과를 병합해서 summary 생성
        if args.resume and output_dir and output_dir.exists():
            existing_summaries = []
            new_tasks = {s["task"] for s in all_summaries}
            for json_path in sorted(output_dir.glob("*.json")):
                if json_path.stem.endswith("summary") or json_path.stem in new_tasks:
                    continue
                with open(json_path) as fp:
                    existing_summaries.append(json.load(fp))
            if existing_summaries:
                logger.info("기존 결과 %d개 병합", len(existing_summaries))
                all_summaries = existing_summaries + all_summaries

        if len(all_summaries) > 1:
            log_all_summary(all_summaries)
        if failed:
            logger.error("실패한 데이터셋 %d개:", len(failed))
            for f in failed:
                logger.error("  %s", f["dataset"])

    if output_dir and len(all_summaries) > 1:
        save_summary(all_summaries, output_dir)


if __name__ == "__main__":
    main()
