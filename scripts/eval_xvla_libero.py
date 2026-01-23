import json
import logging
import time
import concurrent.futures as cf
import numpy as np
import torch
import einops
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from contextlib import nullcontext
from typing import Optional, Any, TypedDict
from collections import defaultdict
from functools import partial

from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.libero import create_libero_envs
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

# Import necessary components from lerobot_eval to override
from lerobot.scripts.lerobot_eval import (
    eval_policy,
    ACC_KEYS,
)

# LIBERO imports for task resolution
from libero.libero import benchmark

@dataclass
class CustomEvalPipelineConfig(EvalPipelineConfig):
    # Add a new argument for filtering by task name
    target_task_name: Optional[str] = None

class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    # Add custom metrics
    execution_times: list[float]

def eval_one_custom(
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    
    # Calculate execution time for each episode based on steps and FPS
    fps = env.unwrapped.metadata["render_fps"]
    execution_times = []
    
    # We need to infer the number of steps from the reward/success/done tensors if available in task_result
    # But eval_policy returns aggregated per_episode info.
    # We can infer time from the video length if available, or we might need to modify eval_policy.
    # However, eval_policy in lerobot_eval.py does not return step counts directly in per_episode.
    # But it does return "eval_ep_s" in aggregated.
    
    # Let's look at how we can get the steps. 
    # eval_policy returns 'per_episode' list.
    # We can't easily get exact steps without modifying eval_policy or rollout.
    # BUT, we can approximate or if we look at the 'done' logic in rollout...
    
    # Wait, eval_policy returns 'episodes' if return_episode_data=True.
    # Let's use that if possible, or just rely on the fact that we want *wall clock* time per task?
    # The user asked for "execution time per task".
    
    # If the user means "simulation time" (steps / fps), we need steps.
    # If the user means "wall clock time" (how long it took to run), we can measure it here.
    
    # Let's assume the user wants the duration of the episode in the simulation (seconds).
    # Since we can't easily get steps from standard eval_policy without return_episode_data=True,
    # let's try to infer it or just measure wall clock time for the whole batch.
    
    # Actually, let's wrap the eval_policy call with a timer to get wall clock time for the set of episodes.
    # But for per-episode simulation time, we might need to rely on return_episode_data=True.
    
    # Let's check if we can get steps.
    # In eval_policy:
    # batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
    
    # We don't have access to 'mask' here.
    
    # Alternative: We can override run_one to measure wall clock time for the *entire task evaluation*.
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        execution_times=[], # Placeholder
    )

def run_one_custom(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    
    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    # We use the original eval_one logic but wrapped here
    metrics = eval_one_custom(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Add wall clock duration to metrics
    metrics["execution_time_wall"] = duration
    metrics["execution_time_per_episode"] = duration / n_episodes
    
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
        
    return task_group, task_id, metrics

def eval_policy_all_custom(
    envs: dict[str, dict[int, Any]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Custom implementation of eval_policy_all that includes timing metrics.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    # Add custom keys
    CUSTOM_ACC_KEYS = ACC_KEYS + ("execution_time_wall",)
    
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in CUSTOM_ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in CUSTOM_ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        
        # Aggregate wall time (summing them up for total time, or averaging? usually we want total resource usage)
        # But for 'overall' stats, maybe average per task?
        # Let's store raw values
        _append("execution_time_wall", metrics.get("execution_time_wall"))

        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one_custom,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "avg_execution_time": _agg_from_list(acc["execution_time_wall"]),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "avg_execution_time": _agg_from_list(overall["execution_time_wall"]),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }

@parser.wrap()
def eval_main(cfg: CustomEvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    
    # Custom Logic: Filter by task name if provided
    logging.info(f"Filtering for task: {cfg.target_task_name}")
    
    # 1. Load the benchmark suite to find the ID
    suite_name = cfg.env.task # e.g., "libero_goal"
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown suite: {suite_name}")
    
    suite = benchmark_dict[suite_name]()
    target_id = -1
    
    # 2. Search for the task ID
    for i, task in enumerate(suite.tasks):
        if task.name == cfg.target_task_name:
            target_id = i
            break
    
    if target_id == -1:
        available_tasks = [t.name for t in suite.tasks]
        raise ValueError(f"Task '{cfg.target_task_name}' not found in suite '{suite_name}'. Available: {available_tasks}")
        
    logging.info(f"Found task '{cfg.target_task_name}' at ID {target_id}")

    # 3. Create environment with specific task_id
    # We bypass make_env for LIBERO to inject gym_kwargs directly    # Note: make_env handles async logic internally, here we simplify to use create_libero_envs directly
    # which returns the dict structure expected by eval_policy_all
    
    # We need to handle async vs sync env class selection if we were fully replicating make_env,
    # but create_libero_envs handles the factory creation.
    import gymnasium as gym
    env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv

    envs = create_libero_envs(
        task=cfg.env.task,
        n_envs=cfg.eval.batch_size,
        gym_kwargs={
            "task_ids": [target_id],  # Inject the filter
            "obs_type": "pixels_agent_pos",  # Ensure robot state is included
        },
        camera_name=cfg.env.camera_name,
        init_states=cfg.env.init_states,
        env_cls=env_cls,
        control_mode=cfg.env.control_mode,
        episode_length=cfg.env.episode_length,
    )


    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        # Use custom eval_policy_all
        info = eval_policy_all_custom(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


if __name__ == "__main__":
    eval_main()
