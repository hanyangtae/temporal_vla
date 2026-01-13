#!/usr/bin/env python3
"""
Time-Aware X-VLA Evaluation Script

Evaluates the fine-tuned X-VLA model on LIBERO benchmark, measuring:
1. Task Success Rate (%)
2. Actual completion time (mean, std)
3. First-step time prediction vs actual (for fine-tuned models)

Usage:
    # Inside Docker container
    # Baseline evaluation
    python scripts/eval_time_aware.py \
        --checkpoint lerobot/xvla-libero \
        --env libero_goal \
        --n_episodes 50 \
        --baseline

    # Fine-tuned evaluation
    python scripts/eval_time_aware.py \
        --checkpoint outputs/train/xvla_time_aware/best \
        --env libero_goal \
        --n_episodes 50
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    # Task performance
    n_episodes: int = 0
    n_successes: int = 0
    success_rate: float = 0.0
    
    # Actual completion time (both baseline & fine-tuned)
    actual_times: list = field(default_factory=list)  # Time to complete each episode (seconds)
    actual_time_mean: float = 0.0
    actual_time_std: float = 0.0
    
    # First-step predictions (fine-tuned only)
    pred_times_step0: list = field(default_factory=list)  # Predicted time at step 0
    pred_stds_step0: list = field(default_factory=list)   # Predicted std at step 0
    pred_time_mean: float = 0.0      # Mean of first-step predictions
    pred_std_mean: float = 0.0       # Mean of predicted std
    
    # Comparison metrics
    time_error_mean: float = 0.0     # Mean(|pred - actual|)
    time_error_std: float = 0.0      # Std of errors
    
    def compute_aggregates(self):
        """Compute aggregate metrics from collected data."""
        if self.n_episodes > 0:
            self.success_rate = self.n_successes / self.n_episodes * 100
        
        # Actual time statistics
        if self.actual_times:
            self.actual_time_mean = float(np.mean(self.actual_times))
            self.actual_time_std = float(np.std(self.actual_times))
        
        # Prediction statistics (fine-tuned only)
        if self.pred_times_step0:
            self.pred_time_mean = float(np.mean(self.pred_times_step0))
        
        if self.pred_stds_step0:
            self.pred_std_mean = float(np.mean(self.pred_stds_step0))
        
        # Error statistics
        if self.pred_times_step0 and self.actual_times:
            errors = np.array(self.pred_times_step0) - np.array(self.actual_times)
            self.time_error_mean = float(np.mean(np.abs(errors)))
            self.time_error_std = float(np.std(errors))
    
    def to_dict(self, is_baseline: bool = False) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "n_episodes": self.n_episodes,
            "n_successes": self.n_successes,
            "success_rate": round(self.success_rate, 2),
            "actual_time_mean": round(self.actual_time_mean, 3),
            "actual_time_std": round(self.actual_time_std, 3),
        }
        
        if not is_baseline:
            result.update({
                "pred_time_mean": round(self.pred_time_mean, 3),
                "pred_std_mean": round(self.pred_std_mean, 3),
                "time_error_mean": round(self.time_error_mean, 3),
                "time_error_std": round(self.time_error_std, 3),
            })
        
        return result


def run_evaluation(
    checkpoint_path: str,
    env_type: str,
    n_episodes: int,
    device: str = "cuda",
    output_dir: str = "outputs/eval",
    seed: int = 42,
    is_baseline: bool = False,
    fps: float = 10.0,  # Environment FPS for time calculation
) -> EvalMetrics:
    """
    Run full evaluation on the specified environment.
    
    Args:
        is_baseline: If True, skip time prediction metrics (for pre-trained models
                     without time_decoder/var_decoder weights)
        fps: Frames per second of the environment (for converting steps to seconds)
    """
    logger.info(f"Loading policy from: {checkpoint_path}")
    logger.info(f"Environment: {env_type}")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Baseline mode: {is_baseline}")
    logger.info(f"FPS: {fps}")
    
    if is_baseline:
        logger.warning("⚠️ Baseline mode: Time prediction metrics will be SKIPPED")
        logger.warning("   (Pre-trained model doesn't have time_decoder/var_decoder weights)")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    metrics = EvalMetrics()
    
    # ================================================================
    # TODO: Replace this mock evaluation with actual environment code
    # ================================================================
    logger.warning("⚠️ Using MOCK evaluation - replace with actual environment code")
    
    for ep in tqdm(range(n_episodes), desc="Evaluating episodes"):
        # --- Mock episode simulation ---
        # In real implementation:
        # 1. Reset environment
        # 2. At step 0, call policy.predict_action_with_time() to get initial prediction
        # 3. Run episode until done
        # 4. Record success and actual completion time
        
        # Mock values (replace with real data)
        mock_success = np.random.random() > 0.3
        mock_actual_steps = np.random.randint(50, 200)  # Steps to complete
        mock_actual_time = mock_actual_steps / fps      # Convert to seconds
        
        metrics.n_episodes += 1
        if mock_success:
            metrics.n_successes += 1
        
        # Record actual completion time (both baseline & fine-tuned)
        metrics.actual_times.append(mock_actual_time)
        
        # Record first-step predictions (fine-tuned only)
        if not is_baseline:
            # Mock prediction at step 0
            mock_pred_time = mock_actual_time + np.random.randn() * 3.0  # Add noise
            mock_pred_std = abs(np.random.randn() * 2.0) + 1.0           # Always positive
            
            metrics.pred_times_step0.append(mock_pred_time)
            metrics.pred_stds_step0.append(mock_pred_std)
    
    # ================================================================
    
    metrics.compute_aggregates()
    return metrics


def print_results(metrics: EvalMetrics, args: argparse.Namespace):
    """Print evaluation results in a formatted table."""
    mode = "BASELINE" if args.baseline else "FINE-TUNED"
    
    print("\n" + "=" * 70)
    print(f"{'EVALUATION RESULTS':^70}")
    print(f"{'(' + mode + ')':^70}")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Environment: {args.env}")
    print(f"  Episodes: {metrics.n_episodes}")
    print("-" * 70)
    
    # Success Rate
    print(f"\n📊 Task Performance:")
    print(f"   ✅ Success Rate: {metrics.success_rate:.1f}% ({metrics.n_successes}/{metrics.n_episodes})")
    
    # Actual Time
    print(f"\n⏱️  Actual Completion Time:")
    print(f"   Mean: {metrics.actual_time_mean:.2f} sec")
    print(f"   Std:  {metrics.actual_time_std:.2f} sec")
    
    # Predictions (fine-tuned only)
    if not args.baseline:
        print(f"\n🔮 First-Step Predictions:")
        print(f"   Predicted Time (mean): {metrics.pred_time_mean:.2f} sec")
        print(f"   Predicted Std (mean):  {metrics.pred_std_mean:.2f} sec")
        
        print(f"\n📈 Prediction Accuracy:")
        print(f"   |Error| Mean: {metrics.time_error_mean:.2f} sec")
        print(f"   Error Std:    {metrics.time_error_std:.2f} sec")
        
        # Compare predicted vs actual variance
        print(f"\n📉 Variance Comparison:")
        print(f"   Actual Std:    {metrics.actual_time_std:.2f} sec")
        print(f"   Predicted Std: {metrics.pred_std_mean:.2f} sec")
        if metrics.actual_time_std > 0:
            ratio = metrics.pred_std_mean / metrics.actual_time_std
            calibration = "well-calibrated ✅" if 0.7 < ratio < 1.3 else "miscalibrated ⚠️"
            print(f"   Ratio (pred/actual): {ratio:.2f} ({calibration})")
    else:
        print(f"\n🔮 Time Predictions: SKIPPED (baseline model)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Time-Aware X-VLA")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint or HuggingFace model ID")
    parser.add_argument("--env", type=str, default="libero_goal",
                        help="Environment type (e.g., libero_goal)")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="outputs/eval",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Environment FPS for time calculation")
    parser.add_argument("--baseline", action="store_true",
                        help="Mark this as baseline evaluation (no time prediction)")
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = run_evaluation(
        checkpoint_path=args.checkpoint,
        env_type=args.env,
        n_episodes=args.n_episodes,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        is_baseline=args.baseline,
        fps=args.fps,
    )
    
    # Print results
    print_results(metrics, args)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"eval_{'baseline' if args.baseline else 'finetuned'}_{args.env}.json"
    with open(result_file, "w") as f:
        json.dump({
            "config": vars(args),
            "metrics": metrics.to_dict(is_baseline=args.baseline),
        }, f, indent=2)
    
    logger.info(f"Results saved to: {result_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
