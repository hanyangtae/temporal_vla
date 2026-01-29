#!/usr/bin/env python3
"""
Unified evaluation script for LIBERO datasets.
Supports both:
1. Original Relative Dataset (requires task mapping via language)
2. Converted Absolute Dataset (tasks already mapped to 0-9)
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
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

def get_dataset_task_info(dataset_path):
    """
    Load task metadata from tasks.parquet and return a dictionary mapping task_index to language.
    Only used for 'original' dataset type.
    """
    try:
        tasks_path = hf_hub_download(
            repo_id='HuggingFaceVLA/libero', 
            filename='meta/tasks.parquet', 
            repo_type='dataset'
        )
        tasks_df = pd.read_parquet(tasks_path)
        # The index contains the task language
        return dict(zip(tasks_df['task_index'], tasks_df.index))
    except Exception as e:
        logger.error(f"Failed to load tasks.parquet: {e}")
        return {}

def find_libero_goal_task(language):
    """
    Find the task in libero_goal suite that matches the language.
    Returns (task_id, task_object) or (None, None).
    """
    bench = benchmark.get_benchmark_dict()
    if 'libero_goal' not in bench:
        return None, None
    
    suite = bench['libero_goal']()
    for i in range(len(suite.tasks)):
        task = suite.get_task(i)
        if task.language.lower().strip() == language.lower().strip():
            return i, task
            
    return None, None

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

def process_single_task(task_info, dataset_path, output_dir, dataset_type, control_mode):
    """
    Worker function to process all episodes for a single task.
    """
    if dataset_type == 'original':
        ds_task_idx, libero_task_id, task_lang = task_info
    else: # converted
        ds_task_idx = task_info # In converted, ds_task_idx IS the libero_task_id
        libero_task_id = ds_task_idx
        task_lang = "unknown" # Not strictly needed for converted

    # Configure logging for this worker
    worker_logger = logging.getLogger(f"Task-{ds_task_idx}")
    worker_logger.setLevel(logging.INFO)
    
    try:
        worker_logger.info(f"🚀 Starting Task {ds_task_idx} (Libero {libero_task_id}) Mode: {control_mode}")
        
        # Load dataset
        ds = load_from_disk(dataset_path)
        if hasattr(ds, 'keys'):
            ds = ds['train']
            
        # Filter for this task
        task_data = ds.filter(lambda x: x['task_index'] == ds_task_idx)
        unique_episodes = sorted(list(set(task_data['episode_index'])))
        
        worker_logger.info(f"Task {ds_task_idx}: Found {len(unique_episodes)} episodes")
        
        # Create output directory for this task
        task_out_dir = Path(output_dir) / f"task_{ds_task_idx}"
        task_out_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Get Suite
        bench = benchmark.get_benchmark_dict()
        suite = bench["libero_goal"]()
        
        for ep_idx in unique_episodes:
            # Determine correct episode index for init state
            # Original dataset: ep_idx is global (e.g. 727), needs % 50
            # Converted dataset: ep_idx is already % 50 (0-49)
            if dataset_type == 'original':
                init_state_idx = ep_idx % 50
            else:
                init_state_idx = ep_idx

            # Initialize Env
            env = LiberoEnv(
                task_suite=suite,
                task_suite_name="libero_goal",
                task_id=libero_task_id,
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
                
                # Extract actions
                if dataset_type == 'converted':
                    actions = np.array([np.array(x['action']) for x in episode_data])
                else:
                    actions = np.array([x['action'] for x in episode_data])
                
                success, num_frames = evaluate_episode(env, actions, task_out_dir, ds_task_idx, ep_idx)
                
                result_entry = {
                    "dataset_task_index": ds_task_idx,
                    "libero_task_id": libero_task_id,
                    "episode_index": ep_idx,
                    "success": success,
                    "num_frames": num_frames
                }
                results.append(result_entry)
                
                worker_logger.info(f"Task {ds_task_idx} Ep {ep_idx}: {'✅ Success' if success else '❌ Fail'}")
                
            except Exception as e:
                worker_logger.error(f"Error in Task {ds_task_idx} Episode {ep_idx}: {e}")
            finally:
                env.close()
        
        # Save results for this task immediately
        df = pd.DataFrame(results)
        csv_path = task_out_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        worker_logger.info(f"💾 Saved results to {csv_path}")
        
        return results
        
    except Exception as e:
        worker_logger.error(f"CRITICAL Error in Task {ds_task_idx}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script for LIBERO datasets")
    parser.add_argument("--dataset_path", type=str, default="data/libero_hf_abs/libero_goal", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_converted_abs", help="Output directory")
    parser.add_argument("--dataset_type", type=str, default="converted", choices=['original', 'converted'], 
                        help="'original' (relative, needs mapping) or 'converted' (absolute, already mapped)")
    parser.add_argument("--control_mode", type=str, default="absolute", choices=['relative', 'absolute'], help="Control mode")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel workers")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler to logger
    file_handler = logging.FileHandler(output_dir / "evaluation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    # 1. Load Dataset Info
    logger.info(f"Loading dataset info from {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    if hasattr(ds, 'keys'):
        ds = ds['train']
        
    # 2. Identify Tasks to Process
    dataset_task_indices = sorted(list(set(ds['task_index'])))
    tasks_to_process = []
    
    if args.dataset_type == 'original':
        logger.info("Mode: Original (Relative). Mapping tasks via language...")
        task_info_map = get_dataset_task_info(args.dataset_path)
        
        for ds_task_idx in dataset_task_indices:
            task_lang = task_info_map.get(ds_task_idx, "unknown")
            libero_task_id, _ = find_libero_goal_task(task_lang)
            
            if libero_task_id is not None:
                tasks_to_process.append((ds_task_idx, libero_task_id, task_lang))
                logger.info(f"  - Found: Dataset Task {ds_task_idx} -> Libero Goal {libero_task_id}")
            else:
                logger.warning(f"  - Skipping Dataset Task {ds_task_idx}: Not in Libero Goal")
                
    else: # converted
        logger.info("Mode: Converted (Absolute). Using task indices directly (0-9)...")
        # In converted dataset, we expect task indices to be 0-9 corresponding to Libero Goal
        tasks_to_process = dataset_task_indices
        logger.info(f"  - Tasks found: {tasks_to_process}")

    logger.info(f"Starting parallel evaluation for {len(tasks_to_process)} tasks with {args.num_workers} workers")
    
    # 3. Run Parallel Processing
    process_func = partial(process_single_task, 
                         dataset_path=args.dataset_path, 
                         output_dir=args.output_dir,
                         dataset_type=args.dataset_type,
                         control_mode=args.control_mode)
    
    all_results = []
    with multiprocessing.Pool(args.num_workers) as pool:
        for task_results in tqdm(pool.imap_unordered(process_func, tasks_to_process), total=len(tasks_to_process)):
            all_results.extend(task_results)
            
    # 4. Merge and Save Final Report
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df = final_df.sort_values(by=["dataset_task_index", "episode_index"])
        final_csv_path = output_dir / "evaluation_results_all.csv"
        final_df.to_csv(final_csv_path, index=False)
        
        # Calculate summary stats
        summary = final_df.groupby("dataset_task_index").agg(
            success_rate=("success", "mean"),
            count=("success", "count")
        )
        summary_path = output_dir / "evaluation_summary.csv"
        summary.to_csv(summary_path)
        
        logger.info(f"\nEvaluation Complete!")
        logger.info(f"Full results: {final_csv_path}")
        logger.info(f"Summary: {summary_path}")
        print("\nSummary:")
        print(summary)
    else:
        logger.warning("No results collected.")

if __name__ == "__main__":
    main()
