#!/usr/bin/env python3
"""
LIBERO 데이터셋의 상대 좌표 Action을 절대 좌표 Action으로 변환하는 스크립트.
src/utils/control_utils.py의 로직을 따릅니다.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk, Dataset, Features, Sequence, Value, Array2D, Array3D, concatenate_datasets
from tqdm import tqdm
import copy
from lerobot.envs.libero import LiberoEnv
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

# Add workspace root to path to import src modules
sys.path.append(os.getcwd())

try:
    from src.utils.control_utils import set_goal_position, set_goal_orientation
    import robosuite.utils.transform_utils as T
except ImportError:
    print("❌ src/utils/control_utils.py 또는 robosuite를 찾을 수 없습니다.")
    print("   먼저 src/utils/control_utils.py가 존재하는지 확인해주세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_task_language(dataset_path: str, task_index: int) -> str:
    """데이터셋의 task_index에 해당하는 language instruction 반환 (deploy_libero.py 참조)"""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    try:
        tasks_path = hf_hub_download(
            repo_id='HuggingFaceVLA/libero', 
            filename='meta/tasks.parquet', 
            repo_type='dataset'
        )
        tasks_df = pd.read_parquet(tasks_path)
        
        for task_name, row in tasks_df.iterrows():
            if row['task_index'] == task_index:
                return task_name
    except Exception as e:
        logger.warning(f"Task info 로드 실패: {e}")
        return "unknown"
    
    raise ValueError(f"task_index {task_index} not found in dataset")

# deprecated
# def find_libero_task_by_language(language: str):
#     """language instruction으로 libero task를 찾아 (suite_name, task_id) 반환 (deploy_libero.py 참조)"""
#     from libero.libero import benchmark
    
#     bench = benchmark.get_benchmark_dict()
#     suites_to_search = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
    
#     for suite_name in suites_to_search:
#         if suite_name not in bench:
#             continue
#         suite = bench[suite_name]()
#         for task_id in range(len(suite.tasks)):
#             task = suite.get_task(task_id)
#             if task.language.lower().strip() == language.lower().strip():
#                 return suite_name, task_id, suite, task
    
#     return None, None, None, None

def scale_action(action, input_min, input_max, output_min, output_max):
    """
    robosuite base_controller.py의 scale_action 로직 구현
    """
    # Calculate scale factors (usually done in __init__)
    action_scale = abs(output_max - output_min) / abs(input_max - input_min)
    action_output_transform = (output_max + output_min) / 2.0
    action_input_transform = (input_max + input_min) / 2.0
    
    # Clip and scale
    action = np.clip(action, input_min, input_max)
    transformed_action = (action - action_input_transform) * action_scale + action_output_transform
    
    return transformed_action

def convert_episode(env, actions, controller_config):
    """
    단일 에피소드에 대해 상대 좌표 Action을 절대 좌표 Action으로 변환
    """
    abs_actions = []
    
    # Controller parameters
    input_min = controller_config['input_min']
    input_max = controller_config['input_max']
    output_min = controller_config['output_min']
    output_max = controller_config['output_max']
    
    # Control dimensions (pos: 3, ori: 3)
    control_dim = 6 
    
    for action in actions:
        # 1. Get current EE state from environment (Before step)
        # robosuite controller stores these
        controller = env._env.robots[0].controller
        
        # Ensure controller state is updated
        # controller.update() is called inside env.step(), but we need current state here.
        # Accessing internal sim data directly as base_controller.py does.
        # Note: env.step() calls sim.forward(), so state should be valid from previous step/reset.
        
        current_pos = np.array(controller.ee_pos)
        current_ori_mat = np.array(controller.ee_ori_mat)
        
        # 2. Separate gripper action
        gripper_action = action[-1]
        delta_action = action[:-1] # 6D
        
        # 3. Scale action (Relative -> Scaled Delta)
        # Note: osc.py handles scaling based on impedance mode. 
        # Assuming "fixed" impedance mode which is standard for Libero.
        # In fixed mode, action is just delta.
        
        # Check if we need to expand limits for scaling
        # input/output limits in controller are usually for the full control dim (6)
        
        scaled_delta = scale_action(
            delta_action, 
            input_min, 
            input_max, 
            output_min, 
            output_max
        )
        
        # 4. Calculate Goal (Absolute) using temp/control_utils.py functions
        # osc.py logic:
        # scaled_delta[:3] -> position delta
        # scaled_delta[3:] -> orientation delta (axis-angle)
        
        # Position Goal
        goal_pos = set_goal_position(
            delta=scaled_delta[:3],
            current_position=current_pos,
            position_limit=controller.position_limits
        )
        
        # Orientation Goal
        goal_ori_mat = set_goal_orientation(
            delta=scaled_delta[3:],
            current_orientation=current_ori_mat,
            orientation_limit=controller.orientation_limits
        )
        
        # Convert Goal Orientation Matrix to Axis-Angle (or Quat) for action representation
        # Usually absolute actions are (x, y, z, ax, ay, az) or (x, y, z, qx, qy, qz, qw)
        # Let's use Axis-Angle to be consistent with 6D representation, 
        # OR we can keep it as rotation matrix if the downstream model supports it.
        # However, standard VLA inputs are usually vectors.
        # Let's convert rotation matrix back to axis-angle (absolute rotation vector)
        # But wait, axis-angle is usually relative. 
        # Absolute orientation is better represented as Euler or Quaternion.
        # LeRobot/Robosuite OSC accepts absolute goal as well.
        # If we look at osc.py set_goal:
        # if set_pos is None: set_pos = delta[:3] (if not using delta)
        # if set_ori is None: set_ori = ... (from delta)
        
        # Let's use 6D (Pos + Axis-Angle) or 7D (Pos + Quat).
        # Standard Libero/Robosuite uses Axis-Angle for relative, but for absolute?
        # Let's stick to 6D (Pos + Axis-Angle representing absolute rotation) 
        # OR 7D (Pos + Quat). 
        # Given X-VLA usually predicts 7D (Pos + Quat) or 6D (Pos + Euler/AxisAngle).
        # Let's use 6D Axis-Angle (Rotation Vector) of the absolute orientation.
        
        goal_ori_aa = T.quat2axisangle(T.mat2quat(goal_ori_mat))
        
        # Combine to form absolute action
        abs_action = np.concatenate([goal_pos, goal_ori_aa, [gripper_action]])
        abs_actions.append(abs_action)
        
        # 5. Step environment to update state
        # We must execute the ORIGINAL relative action to proceed the simulation correctly
        env.step(action)
        
    return np.array(abs_actions)

def process_task(task_idx, dataset_path, output_dir, suite_name_arg="libero_goal", max_episodes=None):
    """
    단일 Task 처리를 위한 함수 (멀티프로세싱용)
    """
    import logging
    # 로깅 설정 (멀티프로세싱에서는 각 프로세스마다 설정 필요)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(f"Task-{task_idx}")
    
    try:
        # 데이터셋 로드 (각 프로세스에서 별도로 로드해야 안전함)
        ds = load_from_disk(dataset_path)
        if hasattr(ds, 'keys'):
            ds = ds['train']
            
        # 언어 기반 검색 로직 제거하고 인자로 받은 suite_name 사용
        suite_name = suite_name_arg
        libero_task_id = task_idx
        
        # LiberoEnv 생성을 위해 suite 객체 로드
        from libero.libero import benchmark
        bench = benchmark.get_benchmark_dict()
        
        if suite_name not in bench:
            logger.error(f"❌ Suite '{suite_name}' not found in Libero benchmark.")
            return None, None
            
        suite = bench[suite_name]()
        
        # Task 정보 로깅 (옵션)
        task_language = "unknown"
        if libero_task_id < len(suite.tasks):
            task = suite.get_task(libero_task_id)
            task_language = task.language
        else:
            logger.warning(f"⚠️ Task ID {libero_task_id} out of range for {suite_name}")
        
        logger.info(f"🔄 Processing Task {task_idx}: {task_language} ({suite_name})")
        
        # Filter data for this task
        task_data = ds.filter(lambda x: x['task_index'] == task_idx)
        unique_episodes = sorted(list(set(task_data['episode_index'])))
        
        if max_episodes:
            unique_episodes = unique_episodes[:max_episodes]
            
        converted_samples = []
        
        for ep_idx in unique_episodes:
            episode_data = task_data.filter(lambda x: x['episode_index'] == ep_idx)
            
            # Extract actions
            actions = np.array([x['action'] for x in episode_data])
            
            # Create Env
            env = LiberoEnv(
                task_suite=suite,
                task_id=libero_task_id,
                task_suite_name=suite_name,
                obs_type="pixels_agent_pos",
                render_mode=None, # 렌더링 끄기 (속도 향상)
                init_states=True,
                episode_index=ep_idx % 50
            )
            env.reset()
            
            # Get Controller Config
            controller = env._env.robots[0].controller
            controller_config = {
                'input_min': controller.input_min,
                'input_max': controller.input_max,
                'output_min': controller.output_min,
                'output_max': controller.output_max,
            }
            
            try:
                abs_actions = convert_episode(env, actions, controller_config)
                
                for i, sample in enumerate(episode_data):
                    new_sample = copy.deepcopy(sample)
                    new_sample['action'] = abs_actions[i]
                    new_sample['episode_index'] = ep_idx % 50
                    # Use the remapped libero_task_id
                    new_sample['task_index'] = libero_task_id
                    converted_samples.append(new_sample)
                    
            except Exception as e:
                logger.error(f"❌ Error converting episode {ep_idx}: {e}")
            finally:
                env.close()
                
        # Save task data immediately to avoid OOM
        # Include mapped ID in folder name for clarity
        temp_dir_name = f"task_orig{task_idx}_mapped{libero_task_id}"
        temp_save_dir = Path(output_dir) / "temp" / suite_name / temp_dir_name
        temp_save_dir.mkdir(parents=True, exist_ok=True)
        
        data_dict = defaultdict(list)
        for item in converted_samples:
            for k, v in item.items():
                data_dict[k].append(v)
        
        task_ds = Dataset.from_dict(data_dict)
        task_ds.save_to_disk(str(temp_save_dir))
        
        logger.info(f"💾 Saved temporary task data to {temp_save_dir}")
        
        # Return path instead of data
        return suite_name, str(temp_save_dir)
        
    except Exception as e:
        logger.error(f"❌ Task {task_idx} failed: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO dataset from relative to absolute actions")
    parser.add_argument("--dataset_path", type=str, default="data/libero_hf/dataset", help="Input dataset path")
    parser.add_argument("--output_dir", type=str, default="data/libero_hf_abs", help="Output dataset path")
    parser.add_argument("--suite_name", type=str, default="libero_goal", help="Libero suite name (e.g., libero_goal, libero_10)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes per task to convert (for testing)")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Load dataset just to get task indices
    logger.info(f"📂 Loading dataset info from {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    if hasattr(ds, 'keys'):
        ds = ds['train']
    
    # Group by task
    task_indices = sorted(list(set(ds['task_index'])))
    logger.info(f"Found {len(task_indices)} tasks")
    
    # Test only the first task
    if args.max_episodes and len(task_indices) > 1 and args.max_episodes < 10: # Heuristic for testing
         # If testing small amount, maybe just run one task or few tasks
         pass

    # Parallel Processing
    from multiprocessing import Pool
    from functools import partial
    
    process_func = partial(process_task, dataset_path=args.dataset_path, output_dir=args.output_dir, suite_name_arg=args.suite_name, max_episodes=args.max_episodes)
    
    suite_paths = defaultdict(list)
    
    logger.info(f"🚀 Starting parallel processing with {args.num_workers} workers...")
    
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_func, task_indices), total=len(task_indices)))
        
    # Collect paths
    for suite_name, path in results:
        if suite_name and path:
            suite_paths[suite_name].append(path)

    # Merge and Save datasets by suite
    for suite_name, paths in suite_paths.items():
        if not paths:
            continue
            
        suite_output_dir = os.path.join(args.output_dir, suite_name)
        logger.info(f"📦 Merging {len(paths)} tasks for {suite_name}...")
        
        try:
            # Load all task datasets
            datasets = [load_from_disk(p) for p in paths]
            
            # Concatenate
            merged_ds = concatenate_datasets(datasets)
            
            # Save final suite dataset
            logger.info(f"💾 Saving {suite_name} dataset to {suite_output_dir} ({len(merged_ds)} frames)")
            merged_ds.save_to_disk(suite_output_dir)
            
            # Cleanup temp files (Optional)
            # import shutil
            # shutil.rmtree(os.path.join(args.output_dir, "temp", suite_name))
            
        except Exception as e:
            logger.error(f"❌ Failed to merge suite {suite_name}: {e}")
        
    logger.info("✅ Done!")

if __name__ == "__main__":
    main()
