#!/usr/bin/env python3
"""
Streamlined evaluation script for LIBERO datasets.
Assumes task indices are already remapped (0-9 for libero_goal).
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
import imageio
from tqdm import tqdm
from dotenv import load_dotenv
import multiprocessing
from functools import partial

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_episode(env, actions, output_path, task_idx, ep_idx):
    """
    Replay a single episode and return success status.
    """
    obs, info = env.reset()
    frames = [env.render()]
    success = False
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        
        if info.get('is_success', False):
            success = True
            break
        
        if terminated or truncated:
            break
            
    # Save video
    video_filename = output_path / f"task{task_idx}_ep{ep_idx}_{'success' if success else 'fail'}.mp4"
    imageio.mimsave(str(video_filename), frames, fps=10, quality=8)
    
    return success, len(frames)

def process_single_task(task_idx, dataset_path, output_dir, task_suite_name, control_mode):
    """
    Worker function to process all episodes for a single task.
    Assumes task_idx is already remapped to match LIBERO suite task IDs.
    """
    # Configure logging for this worker
    worker_logger = logging.getLogger(f"Task-{task_idx}")
    worker_logger.setLevel(logging.INFO)
    
    try:
        worker_logger.info(f"🚀 Starting Task {task_idx} | Suite: {task_suite_name} | Mode: {control_mode}")
        
        # Load dataset
        ds = load_from_disk(dataset_path)
        if hasattr(ds, 'keys'):
            ds = ds['train']
            
        # Filter for this task
        task_data = ds.filter(lambda x: x['task_index'] == task_idx)
        unique_episodes = sorted(list(set(task_data['episode_index'])))
        
        worker_logger.info(f"Task {task_idx}: Found {len(unique_episodes)} episodes")
        
        # Create output directory for this task
        task_out_dir = Path(output_dir) / f"task_{task_idx}"
        task_out_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Get Suite
        bench = benchmark.get_benchmark_dict()
        suite = bench[task_suite_name]()
        
        for ep_idx in unique_episodes:
            # Always use ep_idx % 50 for init state
            init_state_idx = ep_idx % 50

            # Initialize Env
            env = LiberoEnv(
                task_suite=suite,
                task_suite_name=task_suite_name,
                task_id=task_idx,  # task_idx is already remapped
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                control_mode=control_mode,
                init_states=True,
                episode_index=init_state_idx
            )
            
            # For absolute mode, ensure controller is set correctly
            if control_mode == "absolute":
                try:
                    controller = env._env.robots[0].controller
                    controller.use_delta = False
                except:
                    pass
            
            try:
                episode_data = task_data.filter(lambda x: x['episode_index'] == ep_idx)
                
                # Extract actions (handle both list and array formats)
                actions = np.array([np.array(x['action']) for x in episode_data])
                
                success, num_frames = evaluate_episode(env, actions, task_out_dir, task_idx, ep_idx)
                
                result_entry = {
                    "task_index": task_idx,
                    "episode_index": ep_idx,
                    "success": success,
                    "num_frames": num_frames
                }
                results.append(result_entry)
                
                worker_logger.info(f"Task {task_idx} Ep {ep_idx}: {'✅ Success' if success else '❌ Fail'}")
                
            except Exception as e:
                worker_logger.error(f"Error in Task {task_idx} Episode {ep_idx}: {e}")
            finally:
                env.close()
        
        # Save results for this task immediately
        df = pd.DataFrame(results)
        csv_path = task_out_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        worker_logger.info(f"💾 Saved results to {csv_path}")
        
        return results
        
    except Exception as e:
        worker_logger.error(f"CRITICAL Error in Task {task_idx}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Streamlined evaluation script for LIBERO datasets")
    parser.add_argument("--dataset_path", type=str, default="data/libero_hf_split/libero_goal", 
                        help="Path to remapped dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results", 
                        help="Output directory")
    parser.add_argument("--task_suite_name", type=str, default="libero_goal", 
                        help="LIBERO task suite name (e.g., libero_goal, libero_spatial, etc.)")
    parser.add_argument("--control_mode", type=str, default="absolute", 
                        choices=['relative', 'absolute'], help="Control mode")
    parser.add_argument("--num_workers", type=int, default=10, 
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler to logger
    file_handler = logging.FileHandler(output_dir / "evaluation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    # 1. Load Dataset Info
    logger.info(f"📂 Loading dataset from {args.dataset_path}")
    logger.info(f"📋 Task Suite: {args.task_suite_name}")
    logger.info(f"🎮 Control Mode: {args.control_mode}")
    
    ds = load_from_disk(args.dataset_path)
    if hasattr(ds, 'keys'):
        ds = ds['train']
        
    # 2. Identify Tasks to Process
    # Task indices are already remapped, so use them directly
    tasks_to_process = sorted(list(set(ds['task_index'])))
    logger.info(f"✅ Tasks found: {tasks_to_process}")
    logger.info(f"🚀 Starting parallel evaluation for {len(tasks_to_process)} tasks with {args.num_workers} workers")
    
    # 3. Run Parallel Processing
    process_func = partial(
        process_single_task, 
        dataset_path=args.dataset_path, 
        output_dir=args.output_dir,
        task_suite_name=args.task_suite_name,
        control_mode=args.control_mode
    )
    
    all_results = []
    with multiprocessing.Pool(args.num_workers) as pool:
        for task_results in tqdm(pool.imap_unordered(process_func, tasks_to_process), 
                                  total=len(tasks_to_process), 
                                  desc="Evaluating tasks"):
            all_results.extend(task_results)
            
    # 4. Merge and Save Final Report
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df = final_df.sort_values(by=["task_index", "episode_index"])
        final_csv_path = output_dir / "evaluation_results_all.csv"
        final_df.to_csv(final_csv_path, index=False)
        
        # Calculate summary stats
        summary = final_df.groupby("task_index").agg(
            success_rate=("success", "mean"),
            count=("success", "count"),
            avg_frames=("num_frames", "mean")
        )
        summary_path = output_dir / "evaluation_summary.csv"
        summary.to_csv(summary_path)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Evaluation Complete!")
        logger.info(f"📊 Full results: {final_csv_path}")
        logger.info(f"📈 Summary: {summary_path}")
        logger.info(f"{'='*60}")
        print("\n📊 Summary:")
        print(summary)
    else:
        logger.warning("⚠️ No results collected.")

if __name__ == "__main__":
    main()
