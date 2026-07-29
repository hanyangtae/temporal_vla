"""GR00T N1.5 + TTT Phase 2 finetune launcher.

N1.6 의 ``launch_finetune_ttt.py`` 와 동일한 의도지만 N1.5 codebase 직접 사용:
- ``GR00T_N1_5.from_pretrained`` (in N1.5 ``gr00t.model.gr00t_n1``)
- ``LeRobotSingleDataset`` (in N1.5 ``gr00t.data.dataset``)
- ``TrainRunner`` (in N1.5 ``gr00t.experiment.runner``)
- TTT 통합: ``attach_ttt_to_n1d5`` + ``LeRobotSingleDatasetWithTTT``

사용::

    python scripts/train/launch_finetune_ttt_n1d5.py \\
        --dataset-path .../atomic/<Task>/.../lerobot ... \\
        --base-model-path /temporal_vla/checkpoints/nvidia/GR00T-N1.5-3B \\
        --ttt-predictor-path .../phase1_*/epoch_NN.pt \\
        --ttt-eagle-cache-root /temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5 \\
        --output-dir .../outputs/groot_ttt_n1d5_target15
"""
from __future__ import annotations

import os
import sys
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# 1) PYTHONPATH 에 N1.5 + configs 추가
_N1d5_ROOT = "/temporal_vla/src/policies/Isaac-GR00T-N1.5"
_CFG_ROOT = "/temporal_vla/configs/policies"
for p in (_N1d5_ROOT, _CFG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# 2) pytorch3d stub (state_action transform import 만 통과)
import types as _types  # noqa: E402
if "pytorch3d" not in sys.modules:
    _pt = _types.ModuleType("pytorch3d")
    _pt.transforms = _types.ModuleType("pytorch3d.transforms")
    for _name in [
        "matrix_to_quaternion", "quaternion_to_matrix",
        "matrix_to_euler_angles", "euler_angles_to_matrix",
        "matrix_to_rotation_6d", "rotation_6d_to_matrix",
        "quaternion_apply", "axis_angle_to_quaternion",
    ]:
        setattr(_pt.transforms, _name, lambda *a, **k: None)
    sys.modules["pytorch3d"] = _pt
    sys.modules["pytorch3d.transforms"] = _pt.transforms

import tyro  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

# N1.5 codebase
from gr00t.data.schema import EmbodimentTag  # noqa: E402
from gr00t.experiment.runner import TrainRunner  # noqa: E402
from gr00t.model.gr00t_n1 import GR00T_N1_5  # registers gr00t_n1_5  # noqa: E402

# our integrations
from src.ttt.integrations.groot_wrapper_n1d5 import attach_ttt_to_n1d5  # noqa: E402
from src.ttt.integrations.dataset_patch_n1d5 import (  # noqa: E402
    LeRobotSingleDatasetWithTTT,
    make_ttt_collator,
)


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
@dataclass
class N1d5TTTFinetuneConfig:
    """N1.5 + TTT finetune CLI config."""
    dataset_path: List[str]
    output_dir: str = "/tmp/n1d5_ttt"
    base_model_path: str = "/temporal_vla/checkpoints/nvidia/GR00T-N1.5-3B"
    data_config: str = "robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig"
    embodiment_tag: str = "new_embodiment"

    # TTT
    ttt_predictor_path: str = ""
    ttt_eagle_cache_root: str = ""   # ex: /temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5
    ttt_predictor_input_dim: int = 2048
    ttt_predictor_proj_dim: int = 1536
    ttt_predictor_inner_model: str = "linear"
    ttt_predictor_eta_base: float = 0.1
    ttt_update_in_train: bool = True

    # GR00T finetune
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    max_steps: int = 20000
    save_steps: int = 5000
    save_total_limit: int = 4
    dataloader_num_workers: int = 2
    dataloader_prefetch_factor: int = 4
    gradient_accumulation_steps: int = 1
    video_backend: str = "decord"
    report_to: str = "none"

    # data cap
    max_episodes_per_task: int = 200
    resume: bool = False


# ─────────────────────────────────────────────────────────────────────
def _load_data_config(spec: str):
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)()
    raise ValueError(f"data_config must be 'module:Class', got {spec!r}")


def build_dataset(config: N1d5TTTFinetuneConfig, eagle_cache_root: str):
    """Build train_dataset (with TTT z_seq) + transforms / data_config_cls."""
    data_config_cls = _load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # Per-task datasets — LeRobotSingleDatasetWithTTT 의 max_episode_index 로
    # episode_index < max_episodes_per_task 인 sample 만 노출.
    # max_episodes_per_task <= 0 면 필터 비활성.
    max_ep = config.max_episodes_per_task if config.max_episodes_per_task > 0 else None
    datasets = []
    for path in config.dataset_path:
        ds = LeRobotSingleDatasetWithTTT(
            dataset_path=path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            eagle_cache_root=eagle_cache_root,
            max_episode_index=max_ep,
        )
        datasets.append(ds)
    # N1.5 TrainRunner 는 LeRobotSingleDataset / LeRobotMixtureDataset 만 받음.
    # MixtureDataset signature: data_mixture=[(ds, weight), ...], mode="train".
    if len(datasets) == 1:
        return datasets[0], data_config_cls, transforms
    from gr00t.data.dataset import LeRobotMixtureDataset
    data_mixture = [(ds, 1.0) for ds in datasets]
    mix = LeRobotMixtureDataset(
        data_mixture=data_mixture,
        mode="train",
        balance_dataset_weights=True,
        balance_trajectory_weights=True,
        seed=42,
    )
    return mix, data_config_cls, transforms


# ─────────────────────────────────────────────────────────────────────
def main(config: N1d5TTTFinetuneConfig):
    if not config.ttt_eagle_cache_root:
        raise ValueError("ttt_eagle_cache_root must be set for TTT finetune")

    print(f"[n1d5+ttt] datasets={len(config.dataset_path)}")
    print(f"[n1d5+ttt] eagle_cache_root={config.ttt_eagle_cache_root}")
    print(f"[n1d5+ttt] predictor_path={config.ttt_predictor_path}")

    train_dataset, data_config_cls, transforms = build_dataset(
        config, eagle_cache_root=config.ttt_eagle_cache_root,
    )
    print(f"[data] train_dataset built")

    # Action dim sanity (friend awr 패턴 mirror — minimal)
    from gr00t.model.transforms import GR00TTransform
    last_t = transforms.transforms[-1]
    assert isinstance(last_t, GR00TTransform), "last transform must be GR00TTransform"
    data_max_action_dim = last_t.max_action_dim
    data_action_horizon = len(data_config_cls.action_indices)

    # Model load
    print(f"[model] loading N1.5 from {config.base_model_path}")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    # Action head dim mismatch handling (friend awr_finetune L693-724 mirror)
    if data_action_horizon != model.action_head.config.action_horizon or \
       data_max_action_dim != model.action_head.config.action_dim:
        from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
        new_cfg = copy.deepcopy(model.action_head.config)
        new_cfg.action_horizon = data_action_horizon
        new_cfg.action_dim = data_max_action_dim
        new_head = FlowmatchingActionHead(new_cfg)
        try:
            new_head.load_state_dict(model.action_head.state_dict(), strict=False)
        except Exception as e:
            print(f"[warn] partial action_head load: {e}")
        model.action_head = new_head
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # Attach TTT
    print(f"[ttt] attaching TTT (inner={config.ttt_predictor_inner_model}, "
          f"input_dim={config.ttt_predictor_input_dim}, proj_dim={config.ttt_predictor_proj_dim})")
    attach_ttt_to_n1d5(
        model,
        predictor_input_dim=config.ttt_predictor_input_dim,
        predictor_proj_dim=config.ttt_predictor_proj_dim,
        predictor_inner_model=config.ttt_predictor_inner_model,
        predictor_eta_base=config.ttt_predictor_eta_base,
        predictor_state_path=config.ttt_predictor_path or None,
        ttt_update_in_train=config.ttt_update_in_train,
        ttt_update_at_inference=True,
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=Path(config.output_dir).name,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95, adam_beta2=0.999, adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=None if config.report_to == "none" else config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # Wrap collator: TTT z_seq dynamic pad
    orig_collator = experiment.trainer.data_collator
    experiment.trainer.data_collator = make_ttt_collator(orig_collator)

    print("[run] start training")
    experiment.train()


if __name__ == "__main__":
    config = tyro.cli(N1d5TTTFinetuneConfig)
    main(config)
