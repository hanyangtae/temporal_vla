"""GR00T N1.6 + TTT mirror entry of upstream ``gr00t.experiment.launch_finetune``.

Same CLI as upstream plus ``--ttt_predictor_path`` / ``--ttt_update_in_train``.
Upstream code stays untouched — we only monkey-patch ``Gr00tN1d6Pipeline._create_model``
to wrap the loaded model with ``Gr00tN1d6WithTTT`` after ``AutoModel.from_pretrained``.

Run inside the groot container::

    docker compose exec groot bash /temporal_vla/scripts/train/groot_ttt_robocasa_finetune.sh

Or direct CLI::

    docker compose exec -T groot bash -lc \\
        'cd /temporal_vla/src/policies/Isaac-GR00T && \\
         python /temporal_vla/scripts/train/launch_finetune_ttt.py \\
            --base_model_path /temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B \\
            --dataset_path /temporal_vla/data/datasets/robocasa_10tasks_lerobot_v21 \\
            --embodiment_tag ROBOCASA_PANDA_OMRON \\
            --modality_config_path /temporal_vla/configs/policies/groot_robocasa_panda_omron_config.py \\
            --ttt_predictor_path /temporal_vla/outputs/train/phase1_groot_robocasa/<run>/phase1_final.pt \\
            --output_dir /temporal_vla/outputs/groot_ttt_robocasa_10tasks \\
            --max_steps 20000 --global_batch_size 128 ...'
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# repo root + Isaac-GR00T 경로 보장
_REPO_ROOT = "/temporal_vla"
_GR00T_PATH = "/temporal_vla/src/policies/Isaac-GR00T"
for _p in (_REPO_ROOT, _GR00T_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run

from src.ttt.integrations.launch_patch import setup_ttt_pipeline_patch
from src.ttt.integrations.dataset_patch import setup_ttt_dataset_patch


@dataclass
class TTTFinetuneConfig(FinetuneConfig):
    """Upstream FinetuneConfig + TTT injection options.

    Empty ``ttt_predictor_path`` → baseline GR00T finetune (no TTT). 동일 entry
    로 baseline 도 돌릴 수 있다.

    ``ttt_eagle_cache_root`` 가 채워지고 ``ttt_predictor_path`` 도 있으면 episode-prefix
    모드 활성: dataset 이 sample 마다 ``ttt_z_seq`` (Eagle pre-LLM 캐시 0~t slice) +
    ``ttt_valid_mask`` 를 부착, wrapper 가 sequential inner-loop replay → LHT.
    비우면 single-frame fallback (current frame pre-LLM 만, contamination 위험 있음).
    """

    ttt_predictor_path: str = ""
    ttt_update_in_train: bool = False
    ttt_predictor_input_dim: int = 2048
    ttt_predictor_proj_dim: int = 2048
    ttt_predictor_inner_model: str = "linear"
    ttt_predictor_eta_base: float = 0.1
    ttt_eagle_cache_root: str = ""  # ex: /temporal_vla/data/robocasa_eagle_pre_llm


def load_modality_config(modality_config_path: str) -> None:
    """Same as upstream ``launch_finetune.load_modality_config``."""
    import importlib

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(TTTFinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # 1. modality config 등록 (upstream 과 동일 절차)
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # 2. TTT pipeline patch (upstream 모델 빌드 시 wrap)
    setup_ttt_pipeline_patch(
        predictor_path=ft_config.ttt_predictor_path,
        predictor_input_dim=ft_config.ttt_predictor_input_dim,
        predictor_proj_dim=ft_config.ttt_predictor_proj_dim,
        predictor_inner_model=ft_config.ttt_predictor_inner_model,
        predictor_eta_base=ft_config.ttt_predictor_eta_base,
        update_in_train=ft_config.ttt_update_in_train,
    )

    # 2b. TTT dataset patch — episode-prefix z_seq 주입 활성. eagle_cache_root + TTT 가
    # 둘 다 켜져 있을 때만 적용 (baseline 이면 skip).
    if ft_config.ttt_eagle_cache_root and ft_config.ttt_predictor_path:
        setup_ttt_dataset_patch(eagle_cache_root=ft_config.ttt_eagle_cache_root)

    # 3. config 빌드 (upstream launch_finetune.py 와 동일).
    # dataset_path 가 ":" 로 구분된 multi-path 면 per-task mixture, 아니면 단일 dataset.
    if ":" in ft_config.dataset_path:
        ds_paths = [p for p in ft_config.dataset_path.split(":") if p]
        datasets_cfg = [
            {
                "dataset_paths": [p],
                "mix_ratio": 1.0 / len(ds_paths),
                "embodiment_tag": embodiment_tag,
            }
            for p in ds_paths
        ]
    else:
        datasets_cfg = [
            {
                "dataset_paths": [ft_config.dataset_path],
                "mix_ratio": 1.0,
                "embodiment_tag": embodiment_tag,
            }
        ]
    config = get_default_config().load_dict({
        "data": {
            "download_cache": False,
            "datasets": datasets_cfg,
            # default 'torchcodec' 미설치 + fallback pyav 는 get_frames_by_indices
            # 가 NotImplementedError. ffmpeg binary 도 container 에 없음.
            # decord 가 가장 가벼움 (pip install decord, 친구 script 와 동일).
            "video_backend": "decord",
        }
    })
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6-ttt"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
