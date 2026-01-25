"""
Debug script to inspect data flow between LIBERO env and X-VLA policy.
첫 번째 step에서의 데이터 흐름을 확인합니다:
1. env에서 나온 data (raw observation)
2. xvla로 들어간 data (preprocessed observation)
3. xvla에서 나온 data (raw action)
4. env로 들어간 data (postprocessed action)
"""

import json
import logging
import numpy as np
import torch
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat, pprint
from contextlib import nullcontext
from typing import Optional, Any
from copy import deepcopy

from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.libero import create_libero_envs
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

# LIBERO imports for task resolution
from libero.libero import benchmark

import gymnasium as gym


def describe_data(name: str, data: Any, indent: int = 0):
    """데이터 구조를 상세히 출력합니다."""
    prefix = "  " * indent
    print(f"\n{'='*60}")
    print(f"{prefix}📦 {colored(name, 'cyan', attrs=['bold'])}")
    print(f"{'='*60}")
    
    if isinstance(data, dict):
        print(f"{prefix}Type: dict with {len(data)} keys")
        for key, value in data.items():
            describe_value(key, value, indent + 1)
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        describe_tensor(name, data, indent)
    else:
        print(f"{prefix}Type: {type(data).__name__}")
        print(f"{prefix}Value: {data}")


def describe_value(key: str, value: Any, indent: int = 0):
    """단일 값을 상세히 출력합니다."""
    prefix = "  " * indent
    
    if isinstance(value, dict):
        print(f"{prefix}🔑 '{key}': dict with {len(value)} keys")
        for k, v in value.items():
            describe_value(k, v, indent + 1)
    elif isinstance(value, np.ndarray):
        print(f"{prefix}🔑 '{key}': numpy.ndarray")
        print(f"{prefix}    shape: {value.shape}, dtype: {value.dtype}")
        print(f"{prefix}    min: {value.min():.4f}, max: {value.max():.4f}, mean: {value.mean():.4f}")
        if value.size <= 20:
            print(f"{prefix}    values: {value.flatten()}")
    elif isinstance(value, torch.Tensor):
        print(f"{prefix}🔑 '{key}': torch.Tensor")
        print(f"{prefix}    shape: {value.shape}, dtype: {value.dtype}, device: {value.device}")
        print(f"{prefix}    min: {value.min().item():.4f}, max: {value.max().item():.4f}, mean: {value.float().mean().item():.4f}")
        if value.numel() <= 20:
            print(f"{prefix}    values: {value.flatten().tolist()}")
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}🔑 '{key}': {type(value).__name__} with {len(value)} items")
        if len(value) <= 5:
            for i, v in enumerate(value):
                describe_value(f"[{i}]", v, indent + 1)
    elif isinstance(value, str):
        print(f"{prefix}🔑 '{key}': str = '{value}'")
    else:
        print(f"{prefix}🔑 '{key}': {type(value).__name__} = {value}")


def describe_tensor(name: str, tensor: Any, indent: int = 0):
    """텐서를 상세히 출력합니다."""
    prefix = "  " * indent
    
    if isinstance(tensor, np.ndarray):
        print(f"{prefix}Type: numpy.ndarray")
        print(f"{prefix}Shape: {tensor.shape}")
        print(f"{prefix}Dtype: {tensor.dtype}")
        print(f"{prefix}Min: {tensor.min():.6f}")
        print(f"{prefix}Max: {tensor.max():.6f}")
        print(f"{prefix}Mean: {tensor.mean():.6f}")
        print(f"{prefix}Std: {tensor.std():.6f}")
        if tensor.size <= 50:
            print(f"{prefix}Values: {tensor.flatten()}")
    elif isinstance(tensor, torch.Tensor):
        print(f"{prefix}Type: torch.Tensor")
        print(f"{prefix}Shape: {tensor.shape}")
        print(f"{prefix}Dtype: {tensor.dtype}")
        print(f"{prefix}Device: {tensor.device}")
        print(f"{prefix}Min: {tensor.min().item():.6f}")
        print(f"{prefix}Max: {tensor.max().item():.6f}")
        print(f"{prefix}Mean: {tensor.float().mean().item():.6f}")
        print(f"{prefix}Std: {tensor.float().std().item():.6f}")
        if tensor.numel() <= 50:
            print(f"{prefix}Values: {tensor.flatten().tolist()}")


@dataclass
class DebugEvalConfig(EvalPipelineConfig):
    target_task_name: Optional[str] = None


@parser.wrap()
def debug_main(cfg: DebugEvalConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # 1. Load the benchmark suite to find the ID
    suite_name = cfg.env.task  # e.g., "libero_goal"
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown suite: {suite_name}")
    
    suite = benchmark_dict[suite_name]()
    target_id = -1
    
    # 2. Search for the task ID
    if cfg.target_task_name:
        for i, task in enumerate(suite.tasks):
            if task.name == cfg.target_task_name:
                target_id = i
                break
        
        if target_id == -1:
            available_tasks = [t.name for t in suite.tasks]
            raise ValueError(f"Task '{cfg.target_task_name}' not found. Available: {available_tasks}")
    else:
        target_id = 0  # Default to first task
        cfg.target_task_name = suite.tasks[0].name
        
    logging.info(f"Found task '{cfg.target_task_name}' at ID {target_id}")

    # 3. Create environment
    env_cls = gym.vector.SyncVectorEnv  # Use Sync for easier debugging
    
    envs = create_libero_envs(
        task=cfg.env.task,
        n_envs=1,  # Single env for debugging
        gym_kwargs={
            "task_ids": [target_id],
            "obs_type": "pixels_agent_pos",
        },
        camera_name=cfg.env.camera_name,
        init_states=cfg.env.init_states,
        env_cls=env_cls,
        control_mode=cfg.env.control_mode,
        episode_length=cfg.env.episode_length,
    )
    
    # Get the actual VectorEnv
    env = list(list(envs.values())[0].values())[0]
    
    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.eval()

    # Create preprocessors
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.policy
    )

    print("\n" + "=" * 80)
    print(colored("🔍 DEBUG: DATA FLOW INSPECTION (First Step)", "green", attrs=["bold"]))
    print("=" * 80)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        # Reset policy and environment
        policy.reset()
        
        # =====================================================================
        # STEP 1: env.reset() - ENV에서 나온 RAW DATA
        # =====================================================================
        raw_observation, info = env.reset(seed=[cfg.seed])
        
        describe_data("1️⃣  ENV에서 나온 DATA (raw observation from env.reset())", raw_observation)
        print(f"\n{colored('Info:', 'yellow')} {info}")
        
        # =====================================================================
        # STEP 2: Preprocessing - XVLA로 들어가기 전 변환
        # =====================================================================
        
        # 2a. preprocess_observation (numpy -> tensor, key rename)
        observation = preprocess_observation(deepcopy(raw_observation))
        describe_data("2a. preprocess_observation() 후", observation)
        
        # 2b. add_envs_task (task 정보 추가)
        observation = add_envs_task(env, observation)
        describe_data("2b. add_envs_task() 후", observation)
        
        # 2c. env_preprocessor (LIBERO specific)
        observation_env_preprocessed = env_preprocessor(deepcopy(observation))
        describe_data("2c. env_preprocessor() 후 (LIBERO specific)", observation_env_preprocessed)
        
        # 2d. policy preprocessor (device transfer, normalization, etc.)
        observation_policy_preprocessed = preprocessor(deepcopy(observation_env_preprocessed))
        describe_data("2️⃣  XVLA로 들어간 DATA (after all preprocessing)", observation_policy_preprocessed)
        
        # =====================================================================
        # STEP 3: Policy inference - XVLA에서 나온 ACTION
        # =====================================================================
        raw_action = policy.select_action(observation_policy_preprocessed)
        
        describe_data("3️⃣  XVLA에서 나온 DATA (raw action from policy.select_action())", raw_action)
        
        # =====================================================================
        # STEP 4: Postprocessing - ENV로 들어가는 ACTION
        # =====================================================================
        
        # 4a. policy postprocessor
        action_postprocessed = postprocessor(deepcopy(raw_action))
        describe_data("4a. policy postprocessor() 후", action_postprocessed)
        
        # 4b. env_postprocessor (extract action)
        action_transition = {"action": action_postprocessed}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]
        describe_data("4b. env_postprocessor() 후", action)
        
        # 4c. Convert to numpy
        action_numpy = action.to("cpu").numpy()
        describe_data("4️⃣  ENV로 들어간 DATA (action_numpy for env.step())", action_numpy)
        
        # =====================================================================
        # STEP 5: Execute action
        # =====================================================================
        print("\n" + "=" * 80)
        print(colored("🎬 Executing action...", "yellow", attrs=["bold"]))
        print("=" * 80)
        
        next_observation, reward, terminated, truncated, info = env.step(action_numpy)
        
        print(f"\n{colored('Reward:', 'green')} {reward}")
        print(f"{colored('Terminated:', 'red')} {terminated}")
        print(f"{colored('Truncated:', 'red')} {truncated}")
        print(f"{colored('Info:', 'yellow')} {info}")
        
        describe_data("5️⃣  다음 step의 ENV에서 나온 DATA", next_observation)

    # Clean up
    env.close()
    
    print("\n" + "=" * 80)
    print(colored("✅ Debug complete!", "green", attrs=["bold"]))
    print("=" * 80)


if __name__ == "__main__":
    debug_main()
