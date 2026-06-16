"""Adapter for LeRobot pi-family policies."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from checkpoint_profile import CheckpointProfile

from .common import LoadedPolicy, map_container_path

logger = logging.getLogger(__name__)


class PiPolicyAdapter:
    """LeRobot pi-family loader for regular LeRobot checkpoints."""

    def resolve_pretrained_path(
        self,
        profile: CheckpointProfile,
        repo_root: Path,
    ) -> str | Path:
        return map_container_path(profile.checkpoint_source.id, repo_root)

    def state_payload_keys(self, key: str) -> tuple[str, ...]:
        return (f"observation.state.{key}",)

    def state_dim(self, key: str, default: int) -> int:
        return default

    def transform_state_value(self, key: str, value):
        return value

    def build_remap_config(self, visual_keys: list, state_feat) -> tuple[dict[str, str], int]:
        unified_keys = [
            "observation.images.static",
            "observation.images.wrist",
            "observation.images.wrist2",
        ]
        raw = {
            unified_keys[i]: vk for i, vk in enumerate(visual_keys) if i < len(unified_keys)
        }
        camera_key_map = {src: dst for src, dst in raw.items() if src != dst}
        state_dim = state_feat.shape[0] if state_feat is not None else 0
        return camera_key_map, state_dim

    def load(
        self,
        *,
        profile: CheckpointProfile,
        policy_cls,
        pretrained_path: str | Path,
        dataset_stats: dict | None,
        device: str,
    ) -> LoadedPolicy:
        from lerobot.policies.factory import make_pre_post_processors

        ckpt_dir = Path(pretrained_path)
        has_lerobot_config = (
            (ckpt_dir / "config.json").exists() or not ckpt_dir.is_dir()
        )
        if not has_lerobot_config and (ckpt_dir / "model.safetensors").exists():
            logger.info("lerobot config 없음 — 외부 체크포인트 수동 로딩: %s", ckpt_dir)
            from safetensors.torch import load_file
            from lerobot.configs.policies import PolicyFeature
            from lerobot.configs.types import FeatureType

            ext = profile.model_specific.get("external_config", {})
            image_shape = list(ext.get("image_shape", [3, 200, 200]))
            state_shape = list(ext.get("state_shape", [7]))
            action_shape = list(ext.get("action_shape", [7]))
            camera_key = ext.get("camera_key", "observation.images.top")

            cfg = policy_cls.config_class()
            cfg.input_features = {
                camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
            }
            cfg.output_features = {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape),
            }

            if dataset_stats is None:
                norm_paths = list(ckpt_dir.rglob("norm_stats.json"))
                if norm_paths:
                    import json as _json

                    with open(norm_paths[0]) as _f:
                        raw_norm = _json.load(_f)
                    if "norm_stats" in raw_norm:
                        raw_norm = raw_norm["norm_stats"]
                    dataset_stats = {}
                    if "state" in raw_norm:
                        dataset_stats["observation.state"] = {
                            sk: torch.tensor(sv) for sk, sv in raw_norm["state"].items()
                        }
                    if "actions" in raw_norm:
                        dataset_stats["action"] = {
                            sk: torch.tensor(sv) for sk, sv in raw_norm["actions"].items()
                        }
                    logger.info(
                        "norm_stats loaded from %s: keys=%s",
                        norm_paths[0], list(dataset_stats.keys()),
                    )

            loaded_policy = policy_cls(cfg)
            state_dict = load_file(str(ckpt_dir / "model.safetensors"))
            embed_key = (
                "paligemma_with_expert.paligemma.model.language_model."
                "embed_tokens.weight"
            )
            lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
            if lm_head_key in state_dict and embed_key not in state_dict:
                state_dict[embed_key] = state_dict[lm_head_key]
            loaded_policy.model.load_state_dict(state_dict, strict=True)
            logger.info("외부 체크포인트 로드 성공")
        else:
            loaded_policy = policy_cls.from_pretrained(pretrained_path)

        loaded_policy.config.device = device
        prof_nas = getattr(profile, "n_action_steps", None)
        if prof_nas and hasattr(loaded_policy.config, "n_action_steps"):
            loaded_policy.config.n_action_steps = int(prof_nas)
        loaded_policy.to(device)
        loaded_policy.eval()

        loaded_pre, loaded_post = make_pre_post_processors(
            policy_cfg=loaded_policy.config,
            pretrained_path=pretrained_path if has_lerobot_config else None,
            dataset_stats=dataset_stats,
        )
        return LoadedPolicy(
            policy=loaded_policy,
            preprocessor=loaded_pre,
            postprocessor=loaded_post,
            pretrained_path=pretrained_path,
            has_lerobot_config=has_lerobot_config,
        )
