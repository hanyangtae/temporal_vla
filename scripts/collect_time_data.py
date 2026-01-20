#!/usr/bin/env python3
"""
작업 시간 데이터 수집 스크립트 (Baseline 전용)

lerobot-eval wrapper를 통해 baseline 모델의 실행 시간을 수집합니다.

Usage:
    python scripts/collect_time_data.py \
        --task libero_10 \
        --n_episodes 10 \
        --output data/time_data/baseline_times.json

Note:
    - Fine-tuned 모델 평가는 scripts/eval_subtask_based.py 사용
    - 이 스크립트는 시간 데이터 수집 용도로만 사용
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


def run_lerobot_eval(
    task: str,
    n_episodes: int,
    output_dir: str,
    seed: int = 42,
) -> dict:
    """
    lerobot-eval을 실행하고 결과를 반환합니다.
    
    Args:
        task: LIBERO 태스크 이름 (e.g., "libero_10")
        n_episodes: 에피소드 수
        output_dir: 결과 저장 디렉토리
        seed: 랜덤 시드
        
    Returns:
        eval_info.json 내용
    """
    cmd = [
        "lerobot-eval",
        "--policy.path=lerobot/xvla-libero",
        "--env.type=libero",
        f"--env.task={task}",
        "--env.control_mode=absolute",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        "--env.episode_length=800",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        raise RuntimeError("lerobot-eval failed")
    
    # 결과 파일 읽기
    eval_info_path = Path(output_dir) / "eval_info.json"
    if not eval_info_path.exists():
        raise FileNotFoundError(f"eval_info.json not found at {eval_info_path}")
    
    with open(eval_info_path) as f:
        return json.load(f)


def extract_time_data(eval_info: dict, fps: float = 30.0) -> list[dict]:
    """
    eval_info에서 시간 데이터를 추출합니다.
    
    lerobot-eval은 직접적인 스텝 수를 제공하지 않으므로,
    비디오 파일의 프레임 수에서 추정합니다.
    
    Args:
        eval_info: lerobot-eval의 eval_info.json 내용
        fps: 환경 FPS (기본 30)
        
    Returns:
        각 태스크별 시간 데이터 리스트
    """
    time_data = []
    
    for task_result in eval_info.get("per_task", []):
        task_group = task_result["task_group"]
        task_id = task_result["task_id"]
        metrics = task_result["metrics"]
        
        for ep_idx, (success, video_path) in enumerate(zip(
            metrics["successes"], 
            metrics.get("video_paths", [None] * len(metrics["successes"]))
        )):
            # 비디오에서 프레임 수 추출 (ffprobe 사용)
            steps = None
            if video_path:
                try:
                    result = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-count_frames", 
                         "-show_entries", "stream=nb_read_frames",
                         "-print_format", "json", video_path],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        probe_data = json.loads(result.stdout)
                        frames = int(probe_data["streams"][0]["nb_read_frames"])
                        steps = frames
                except Exception as e:
                    print(f"⚠️ Could not extract frames from {video_path}: {e}")
            
            time_data.append({
                "task_group": task_group,
                "task_id": task_id,
                "episode_idx": ep_idx,
                "success": success,
                "steps": steps,
                "time_seconds": steps / fps if steps else None,
                "video_path": video_path,
            })
    
    return time_data


def compute_statistics(time_data: list[dict]) -> dict:
    """
    시간 데이터의 통계를 계산합니다.
    """
    successful_times = [d["time_seconds"] for d in time_data 
                        if d["success"] and d["time_seconds"] is not None]
    all_times = [d["time_seconds"] for d in time_data 
                 if d["time_seconds"] is not None]
    
    stats = {
        "total_episodes": len(time_data),
        "successful_episodes": sum(1 for d in time_data if d["success"]),
        "success_rate": sum(1 for d in time_data if d["success"]) / len(time_data) * 100 if time_data else 0,
    }
    
    if successful_times:
        stats["successful_time_mean"] = float(np.mean(successful_times))
        stats["successful_time_std"] = float(np.std(successful_times))
        stats["successful_time_min"] = float(np.min(successful_times))
        stats["successful_time_max"] = float(np.max(successful_times))
    
    if all_times:
        stats["all_time_mean"] = float(np.mean(all_times))
        stats["all_time_std"] = float(np.std(all_times))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Collect time data from baseline XVLA evaluation (lerobot-eval wrapper)"
    )
    parser.add_argument("--task", type=str, default="libero_10", help="LIBERO task name")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes per task")
    parser.add_argument("--output", type=str, default="data/time_data/time_data.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=float, default=10.0, help="Environment FPS (LIBERO default: 10)")
    parser.add_argument("--eval_output_dir", type=str, default=None, help="Directory for lerobot-eval outputs")
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Collecting time data for task: {args.task}")
    print(f"📁 Output: {args.output}")
    print(f"🎯 Mode: Baseline model evaluation (lerobot-eval)")
    
    if args.eval_output_dir:
        eval_output_dir = args.eval_output_dir
    else:
        eval_output_dir = f"outputs/eval/time_collection_{args.task}"
    
    # lerobot-eval 실행
    eval_info = run_lerobot_eval(
        task=args.task,
        n_episodes=args.n_episodes,
        output_dir=eval_output_dir,
        seed=args.seed,
    )
    
    # 시간 데이터 추출
    time_data = extract_time_data(eval_info, fps=args.fps)
    
    model_info = {
        "type": "baseline",
        "path": "lerobot/xvla-libero",
    }
    
    # 통계 계산
    stats = compute_statistics(time_data)
    
    # 결과 저장
    result = {
        "metadata": {
            "task": args.task,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "fps": args.fps,
            "model": model_info,
            "collected_at": datetime.now().isoformat(),
        },
        "statistics": stats,
        "episodes": time_data,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Time data saved to: {output_path}")
    print(f"\n📈 Statistics:")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    if "successful_time_mean" in stats:
        print(f"   Successful Time: {stats['successful_time_mean']:.2f} ± {stats['successful_time_std']:.2f} sec")
        print(f"   Range: [{stats['successful_time_min']:.2f}, {stats['successful_time_max']:.2f}] sec")


if __name__ == "__main__":
    main()
