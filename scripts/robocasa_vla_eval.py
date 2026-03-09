"""
DreamVLA closed-loop 평가 스크립트.

DreamVLA 서버(serve_dreamvla.py)에 매 스텝 env 관측값을 보내고,
반환된 action을 RoboCasa 환경에 실행해 성공률을 측정한다.

사용법:
  # 단일 데이터셋 (에피소드 초기 상태를 lerobot 데이터에서 로드)
  python scripts/robocasa_vla_eval.py \
    --dataset /temporal_vla/data/datasets/v1.0/pretrain/atomic/TurnOnToaster/20250820/lerobot \
    --vla-server http://localhost:8200

  # 전체 atomic 태스크
  python scripts/robocasa_vla_eval.py \
    --all --split pretrain --task-type atomic \
    --vla-server http://localhost:8200 \
    --output-dir /temporal_vla/outputs/vla_eval

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

import numpy as np
import robocasa.utils.lerobot_utils as LU
import robosuite
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from tqdm import tqdm

from src.utils.common.logger import create_module_logger
from robocasa_eval import find_all_datasets, log_all_summary

logger = create_module_logger("robocasa_vla_eval")


# ── 환경 설정 ──────────────────────────────────────────────────────────────

STATIC_CAM = "robot0_agentview_left"
WRIST_CAM = "robot0_eye_in_hand"


def _make_env(dataset_path: Path):
    meta = LU.get_env_metadata(dataset_path)
    kwargs = meta["env_kwargs"]
    kwargs["env_name"] = meta["env_name"]
    kwargs.update(
        has_renderer=False,
        renderer="mjviewer",
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[STATIC_CAM, WRIST_CAM],
        camera_heights=224,
        camera_widths=224,
    )
    return robosuite.make(**kwargs), meta["env_name"]


def _get_instruction(dataset_path: Path, ep_num: int) -> str:
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    task_map = {}
    for line in tasks_path.read_text().strip().split("\n"):
        entry = json.loads(line)
        task_map[entry["task_index"]] = entry["task"]

    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    chunk = ep_num // info.get("chunks_size", 1000)
    import pandas as pd
    df = pd.read_parquet(
        dataset_path / f"data/chunk-{chunk:03d}/episode_{ep_num:06d}.parquet",
        columns=["task_index"],
    )
    return task_map.get(int(df["task_index"].iloc[0]), "")


def _check_success(env) -> bool:
    result = env._check_success()
    return bool(result.get("task", False)) if isinstance(result, dict) else bool(result)


# ── 에피소드 평가 ──────────────────────────────────────────────────────────

def eval_episode(env, dataset_path: Path, ep_num: int, vla_client, max_steps: int = 400) -> dict:
    instruction = _get_instruction(dataset_path, ep_num)
    vla_client.reset()

    initial_state = {
        "states": LU.get_episode_states(dataset_path, ep_num)[0],
        "model": LU.get_episode_model_xml(dataset_path, ep_num),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset_path, ep_num)),
    }
    reset_to(env, initial_state)
    obs = env._get_observations()

    latencies = []
    first_success_step = None

    for t in range(max_steps):
        static_img = obs.get(f"{STATIC_CAM}_image", np.zeros((224, 224, 3), dtype=np.uint8))
        wrist_img = obs.get(f"{WRIST_CAM}_image", np.zeros((224, 224, 3), dtype=np.uint8))
        state = obs.get("robot0_proprio-state", np.zeros(32, dtype=np.float32))

        action, latency_ms = vla_client.predict(static_img, wrist_img, state, instruction)
        latencies.append(latency_ms)

        obs, _reward, done, _info = env.step(action)

        if _check_success(env) and first_success_step is None:
            first_success_step = t
        if done:
            break

    return {
        "success": _check_success(env),
        "first_success_step": first_success_step,
        "steps": t + 1,
        "mean_latency_ms": float(np.mean(latencies)),
        "instruction": instruction,
    }


# ── 데이터셋 평가 ──────────────────────────────────────────────────────────

def evaluate_dataset(dataset_path: Path, vla_client, n: int = None, max_steps: int = 400) -> dict:
    env, task_name = _make_env(dataset_path)
    logger.info("태스크: %s", task_name)

    episodes = LU.get_episodes(dataset_path)
    if n is not None:
        episodes = episodes[:n]
    total = len(episodes)

    results = []
    for ep_num in tqdm(range(total), desc="에피소드 평가"):
        try:
            r = eval_episode(env, dataset_path, ep_num, vla_client, max_steps)
            r["ep_num"] = ep_num
        except Exception as e:
            logger.error("ep %d 실패: %s", ep_num, e)
            r = {"ep_num": ep_num, "success": False, "error": str(e)}
        results.append(r)

    env.close()

    n_success = sum(r["success"] for r in results)
    lats = [r["mean_latency_ms"] for r in results if "mean_latency_ms" in r]
    return {
        "dataset": str(dataset_path),
        "task": task_name,
        "mode": "vla",
        "total_episodes": total,
        "n_success": n_success,
        "success_rate": n_success / total if total > 0 else 0.0,
        "mean_latency_ms": float(np.mean(lats)) if lats else 0.0,
        "episodes": results,
    }


def log_summary(summary: dict):
    rate = summary["success_rate"]
    lines = [
        "=" * 50,
        f"태스크:        {summary['task']}",
        f"데이터셋:      {summary['dataset']}",
        f"총 에피소드:   {summary['total_episodes']}",
        f"성공:          {summary['n_success']} / {summary['total_episodes']}",
        f"성공률:        {rate:.1%}",
        f"평균 레이턴시: {summary['mean_latency_ms']:.1f} ms",
        "=" * 50,
    ]
    msg = "\n".join(lines)
    logger.info(msg) if rate > 0.5 else logger.warning(msg)


# ── 저장 ───────────────────────────────────────────────────────────────────

def save_task_result(summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{summary['task']}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("저장: %s", path)


def save_summary(all_summaries: list[dict], output_dir: Path):
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
    p = argparse.ArgumentParser(description="DreamVLA closed-loop 평가")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", type=str, help="평가할 lerobot 데이터셋 경로 (단일)")
    mode.add_argument("--all", action="store_true", help="지정 split의 모든 데이터셋 평가")

    p.add_argument("--vla-server", type=str, default="http://localhost:8200",
                   help="DreamVLA 서버 URL")
    p.add_argument("--base-path", default="/temporal_vla/data/datasets/v1.0")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--task-type", default=None, choices=["atomic", "composite"])
    p.add_argument("--n", type=int, default=None, help="태스크당 평가 에피소드 수 (기본: 전체)")
    p.add_argument("--max-steps", type=int, default=400, help="에피소드당 최대 스텝 수")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="이미 결과 JSON이 있는 태스크 건너뛰기")
    return p


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    from vla_client import DreamVLAClient
    vla_client = DreamVLAClient(url=args.vla_server)
    logger.info("DreamVLA 서버 연결 대기 중: %s", args.vla_server)
    vla_client.wait_until_ready()
    logger.info("DreamVLA 서버 연결 완료.")

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error("경로가 존재하지 않음: %s", dataset_path)
            sys.exit(1)
        try:
            summary = evaluate_dataset(dataset_path, vla_client, n=args.n, max_steps=args.max_steps)
        except KeyboardInterrupt:
            logger.warning("평가 중단됨.")
            sys.exit(0)
        except Exception:
            logger.error("평가 실패:\n%s", traceback.format_exc())
            sys.exit(1)
        log_summary(summary)
        if output_dir:
            save_task_result(summary, output_dir)
        return

    base = Path(args.base_path) / args.split
    if not base.exists():
        logger.error("경로가 존재하지 않음: %s", base)
        sys.exit(1)

    datasets = find_all_datasets(base, args.task_type)
    if not datasets:
        logger.error("데이터셋 없음: %s", base)
        sys.exit(1)

    if args.resume and output_dir and output_dir.exists():
        existing = {p.stem for p in output_dir.glob("*.json") if not p.stem.endswith("summary")}
        datasets = [d for d in datasets if d.parts[-3] not in existing]
        logger.info("--resume: %d개 태스크 남음", len(datasets))

    all_summaries = []
    for i, ds_path in enumerate(datasets):
        task_name = ds_path.parts[-3]
        logger.info("[%d/%d] %s", i + 1, len(datasets), task_name)
        try:
            summary = evaluate_dataset(ds_path, vla_client, n=args.n, max_steps=args.max_steps)
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
    if output_dir and len(all_summaries) > 1:
        save_summary(all_summaries, output_dir)


if __name__ == "__main__":
    main()
