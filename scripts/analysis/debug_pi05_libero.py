"""Debug script: pi0.5 + LIBERO pipeline with full normalization."""
import torch
import numpy as np
from pathlib import Path

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.normalize_processor import _NormalizationMixin

# Load dataset stats
print("Loading dataset stats from lerobot/libero...")
ds = LeRobotDataset("lerobot/libero")
stats = ds.meta.stats

cfg_policy = PreTrainedConfig.from_pretrained("lerobot/pi05_libero_base")
cfg_policy.pretrained_path = Path("lerobot/pi05_libero_base")
cfg_policy.device = "cuda"
cfg_env = LiberoEnvConfig(task="libero_10", task_ids=[0], obs_type="pixels_agent_pos")

envs_dict = make_env(cfg_env, n_envs=1, use_async_envs=False)
env = list(list(envs_dict.values())[0].values())[0]
policy = make_policy(cfg=cfg_policy, env_cfg=cfg_env)
policy.eval()

pre_ov = {"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": {}}}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg_policy, pretrained_path=cfg_policy.pretrained_path, preprocessor_overrides=pre_ov
)

# Inject stats AND features into normalizer steps
features = {**cfg_policy.input_features, **cfg_policy.output_features}
for step in preprocessor.steps:
    if isinstance(step, _NormalizationMixin):
        step.stats = stats
        step.features = features
        step._tensor_stats = {}
        step.__post_init__()
for step in postprocessor.steps:
    if isinstance(step, _NormalizationMixin):
        step.stats = stats
        step.features = features
        step._tensor_stats = {}
        step.__post_init__()

env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg_env, policy_cfg=cfg_policy)

policy.reset()
obs_raw, _ = env.reset(seed=[42])
obs = preprocess_observation(obs_raw)
obs = add_envs_task(env, obs)
task_text = obs.get("task", "MISSING")
print(f"task: {task_text}")

obs = env_pre(obs)
print("\nAfter env_preprocessor:")
for k in sorted(obs.keys()):
    v = obs[k]
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} [{v.min():.4f}, {v.max():.4f}]")

obs_p = preprocessor(obs)
print("\nAfter policy preprocessor (WITH stats+features):")
for k in sorted(obs_p.keys()):
    v = obs_p[k]
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} dev={v.device} [{v.min().item():.4f}, {v.max().item():.4f}]")

with torch.inference_mode():
    action = policy.select_action(obs_p)
action_out = postprocessor(action)
print(f"\nAction: {action_out.shape} values={action_out[0].cpu().numpy()}")

# Multiple steps
action_np = action_out.cpu().numpy()
obs_raw2, reward, term, trunc, info = env.step(action_np)

for i in range(9):
    obs = preprocess_observation(obs_raw2)
    obs = add_envs_task(env, obs)
    obs = env_pre(obs)
    obs_p = preprocessor(obs)
    with torch.inference_mode():
        action = policy.select_action(obs_p)
    action_out = postprocessor(action)
    action_np = action_out.cpu().numpy()
    print(f"Step {i+2}: action={action_np[0]}")
    obs_raw2, reward, term, trunc, info = env.step(action_np)

env.close()
print("\nDone.")
