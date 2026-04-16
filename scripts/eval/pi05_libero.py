"""pi0.5 LIBERO eval with dataset normalization stats injected."""
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import hotswap_stats
from lerobot.processor.normalize_processor import _NormalizationMixin
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Inject dataset normalization stats + features
    logging.info("Loading dataset stats from lerobot/libero...")
    ds = LeRobotDataset("lerobot/libero")
    preprocessor = hotswap_stats(preprocessor, ds.meta.stats)
    postprocessor = hotswap_stats(postprocessor, ds.meta.stats)
    features = {**cfg.policy.input_features, **cfg.policy.output_features}
    for step in preprocessor.steps:
        if isinstance(step, _NormalizationMixin) and not step.features:
            step.features = features
    for step in postprocessor.steps:
        if isinstance(step, _NormalizationMixin) and not step.features:
            step.features = features

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.policy
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
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

        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)

    close_envs(envs)

    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
