#!/usr/bin/env python3
"""
작업 시간 데이터 수집 및 시간 예측 평가 스크립트

=== Baseline 모델 (lerobot-eval wrapper) ===
    python scripts/collect_time_data.py \
        --task libero_10 \
        --n_episodes 10 \
        --output data/time_data/baseline_times.json

=== Fine-tuned 모델 (직접 inference + 시간 예측 평가) ===
    python scripts/collect_time_data.py \
        --task libero_10 \
        --n_episodes 10 \
        --model_path outputs/train/xvla_time_aware/best \
        --output data/time_data/finetuned_times.json
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

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
                    import subprocess
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
    
    # 시간 예측 통계 (Fine-tuned 모델인 경우)
    predicted_times = [d.get("predicted_time") for d in time_data 
                       if d.get("predicted_time") is not None]
    predicted_vars = [d.get("predicted_var") for d in time_data 
                      if d.get("predicted_var") is not None]
    
    if predicted_times and successful_times:
        # 성공한 에피소드만 비교 (실제 시간과 예측 시간)
        pairs = [(d["time_seconds"], d["predicted_time"]) 
                 for d in time_data 
                 if d["success"] and d.get("predicted_time") is not None and d["time_seconds"] is not None]
        
        if pairs:
            actual, predicted = zip(*pairs)
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            # MAE (Mean Absolute Error)
            stats["time_prediction_mae"] = float(np.mean(np.abs(actual - predicted)))
            # RMSE (Root Mean Squared Error)
            stats["time_prediction_rmse"] = float(np.sqrt(np.mean((actual - predicted) ** 2)))
            # 상관계수
            if len(pairs) > 1:
                stats["time_prediction_correlation"] = float(np.corrcoef(actual, predicted)[0, 1])
    
    if predicted_vars:
        stats["predicted_var_mean"] = float(np.mean(predicted_vars))
        stats["predicted_var_std"] = float(np.std(predicted_vars))
    
    return stats


def run_finetuned_eval(
    model_path: str,
    task: str,
    n_episodes: int,
    fps: float = 10.0,
    seed: int = 42,
) -> list[dict]:
    """
    Fine-tuned 모델을 직접 로드하여 평가하고 시간 예측 결과를 수집합니다.
    
    Args:
        model_path: Fine-tuned 모델 경로
        task: LIBERO 태스크 이름
        n_episodes: 에피소드 수
        fps: 환경 FPS
        seed: 랜덤 시드
        
    Returns:
        각 에피소드의 결과 (실제 시간 + 예측 시간 + 예측 불확실성)
    """
    import torch
    from lerobot.envs.libero import LiberoEnv
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from libero.libero import benchmark
    from transformers import AutoTokenizer
    
    print(f"🔧 Loading fine-tuned model from: {model_path}")
    
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = XVLAPolicy.from_pretrained(model_path)
    policy = policy.to(device)
    policy.eval()
    
    # 환경 설정
    # task_suite와 task_id 분리 (예: libero_goal_0 -> libero_goal, 0)
    if "_" in task and task.split("_")[-1].isdigit():
        task_suite_name = "_".join(task.split("_")[:-1])
        task_id = int(task.split("_")[-1])
    else:
        task_suite_name = task
        task_id = 0
    
    print(f"📦 Setting up environment: {task_suite_name}, task_id={task_id}")
    
    # TaskSuite 객체 로드
    suite_class = benchmark.get_benchmark_dict()[task_suite_name]
    task_suite = suite_class()
    task_info = task_suite.tasks[task_id]
    task_description = task_info.language
    print(f"   Task: {task_info.name}")
    print(f"   Description: {task_description}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    language_tokens = tokenizer(
        task_description,
        padding='max_length',
        max_length=64,
        truncation=True,
        return_tensors='pt'
    )['input_ids'].to(device)
    
    env = LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        obs_type="pixels_agent_pos",
        control_mode="absolute",
        episode_length=800,
    )
    
    time_data = []
    np.random.seed(seed)
    
    print(f"🎮 Running {n_episodes} episodes...")
    
    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        max_steps = 800
        
        # 첫 스텝에서 시간 예측 저장
        first_step_time_pred = None
        first_step_var_pred = None
        
        while not done and step_count < max_steps:
            # 관측 준비 (LiberoEnv는 'image', 'image2' 키 사용)
            # robot_state에서 필요한 정보 추출
            rs = obs["robot_state"]
            robot_state = np.concatenate([
                rs["eef"]["pos"],  # 3
                rs["eef"]["quat"],  # 4
                rs["gripper"]["qpos"],  # 2
            ])  # Total: 9
            
            # 20차원으로 패딩 (XVLA 요구사항)
            robot_state_padded = np.zeros(20, dtype=np.float32)
            robot_state_padded[:len(robot_state)] = robot_state
            
            # 이미지 전처리: (H, W, C) -> (C, H, W), uint8 -> float32
            img = torch.from_numpy(obs["pixels"]["image"]).permute(2, 0, 1).float() / 255.0
            
            obs_dict = {
                "observation.images.image": img.unsqueeze(0).to(device),
                "observation.state": torch.from_numpy(robot_state_padded).unsqueeze(0).to(device),
                "observation.language.tokens": language_tokens,
            }
            
            # 이미지 전처리 (BHWC -> BCHW)
            if obs_dict["observation.images.image"].dim() == 4:
                obs_dict["observation.images.image"] = obs_dict["observation.images.image"].permute(0, 3, 1, 2)
            
            # 언어 토큰 준비 (태스크 설명)
            task_description = info.get("task_description", f"Complete task {task_id}")
            # TODO: 토크나이저로 언어 토큰 생성 필요
            
            with torch.no_grad():
                # action 선택 (시간 예측 포함)
                action = policy.select_action(obs_dict)
                
                # 첫 스텝에서 시간 예측 저장
                if step_count == 0:
                    time_pred, var_pred = policy.get_last_time_prediction()
                    if time_pred is not None:
                        first_step_time_pred = float(time_pred.cpu().numpy())
                    if var_pred is not None:
                        first_step_var_pred = float(var_pred.cpu().numpy())
            
            # 환경 스텝
            action_np = action.cpu().numpy().squeeze()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step_count += 1
        
        # 에피소드 결과 저장
        success = info.get("success", False)
        actual_time = step_count / fps
        
        episode_result = {
            "task_group": task_suite,
            "task_id": task_id,
            "episode_idx": ep_idx,
            "success": success,
            "steps": step_count,
            "time_seconds": actual_time,
            "predicted_time": first_step_time_pred,
            "predicted_var": first_step_var_pred,
        }
        
        time_data.append(episode_result)
        
        status = "✅" if success else "❌"
        pred_str = f", pred={first_step_time_pred:.2f}s" if first_step_time_pred else ""
        print(f"  Episode {ep_idx + 1}/{n_episodes}: {status} steps={step_count}, time={actual_time:.2f}s{pred_str}")
    
    env.close()
    return time_data


def main():
    parser = argparse.ArgumentParser(
        description="Collect time data from XVLA evaluation (baseline or fine-tuned)"
    )
    parser.add_argument("--task", type=str, default="libero_10", help="LIBERO task name")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes per task")
    parser.add_argument("--output", type=str, default="data/time_data/time_data.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=float, default=10.0, help="Environment FPS (LIBERO default: 10)")
    parser.add_argument("--eval_output_dir", type=str, default=None, help="Directory for lerobot-eval outputs")
    
    # Fine-tuned 모델 평가용 옵션
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
        help="Path to fine-tuned model. If not provided, uses baseline (lerobot-eval wrapper)"
    )
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Collecting time data for task: {args.task}")
    print(f"📁 Output: {args.output}")
    
    # === Fine-tuned 모델 평가 ===
    if args.model_path:
        print(f"🎯 Mode: Fine-tuned model evaluation")
        print(f"   Model: {args.model_path}")
        
        time_data = run_finetuned_eval(
            model_path=args.model_path,
            task=args.task,
            n_episodes=args.n_episodes,
            fps=args.fps,
            seed=args.seed,
        )
        
        model_info = {
            "type": "finetuned",
            "path": args.model_path,
        }
    
    # === Baseline 모델 평가 (lerobot-eval wrapper) ===
    else:
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
    
    # Fine-tuned 모델 시간 예측 통계
    if args.model_path:
        print(f"\n📐 Time Prediction Metrics:")
        if "time_prediction_mae" in stats:
            print(f"   MAE: {stats['time_prediction_mae']:.3f} sec")
            print(f"   RMSE: {stats['time_prediction_rmse']:.3f} sec")
        if "time_prediction_correlation" in stats:
            print(f"   Correlation: {stats['time_prediction_correlation']:.3f}")
        if "predicted_var_mean" in stats:
            print(f"   Predicted Variance: {stats['predicted_var_mean']:.3f} ± {stats['predicted_var_std']:.3f}")


if __name__ == "__main__":
    main()
