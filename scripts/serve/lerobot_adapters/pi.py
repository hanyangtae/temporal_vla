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

    def __init__(self) -> None:
        # load() 시 multi-camera 프로파일에서 채워짐: unified_src → model_dst 매핑
        self._camera_keys_override: dict[str, str] = {}

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
        state_dim = state_feat.shape[0] if state_feat is not None else 0
        # load()에서 camera_keys dict가 주입된 경우 → 직접 사용 (unified_key → model_key)
        if self._camera_keys_override:
            return dict(self._camera_keys_override), state_dim
        unified_keys = [
            "observation.images.static",
            "observation.images.wrist",
            "observation.images.wrist2",
        ]
        raw = {
            unified_keys[i]: vk for i, vk in enumerate(visual_keys) if i < len(unified_keys)
        }
        camera_key_map = {src: dst for src, dst in raw.items() if src != dst}
        return camera_key_map, state_dim

    def _has_camera_keys_in_profile(self, profile: CheckpointProfile) -> bool:
        """external_config.camera_keys dict 존재 시 True — multi-camera 외부 ckpt 판별."""
        ext = profile.model_specific.get("external_config", {})
        return bool(ext.get("camera_keys"))

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
        _force_external = self._has_camera_keys_in_profile(profile)
        has_lerobot_config = (
            (ckpt_dir / "config.json").exists() or not ckpt_dir.is_dir()
        )
        _loaded_external = (
            _force_external or not has_lerobot_config
        ) and (ckpt_dir / "model.safetensors").exists()
        if _loaded_external:
            if _force_external:
                logger.info(
                    "external_config.camera_keys 선언됨 — 외부 체크포인트 경로 강제 사용: %s",
                    ckpt_dir,
                )
            else:
                logger.info("lerobot config 없음 — 외부 체크포인트 수동 로딩: %s", ckpt_dir)
            from safetensors.torch import load_file
            from lerobot.configs.policies import PolicyFeature
            from lerobot.configs.types import FeatureType

            ext = profile.model_specific.get("external_config", {})
            image_shape = list(ext.get("image_shape", [3, 200, 200]))
            state_shape = list(ext.get("state_shape", [7]))
            action_shape = list(ext.get("action_shape", [7]))

            # multi-camera: camera_keys dict → {model_key: PolicyFeature}
            # single-camera fallback: camera_key str
            camera_keys_map: dict = ext.get("camera_keys") or {}
            camera_key_single: str = ext.get("camera_key", "observation.images.top")

            cfg = policy_cls.config_class()
            # config.json dtype=float32 는 16GB GPU 에서 OOM(~13.4GB). 체크포인트가 bf16 이면
            # bfloat16 으로 강제해 VRAM 을 절반으로 줄임. profile external_config.dtype 으로 override 가능.
            cfg_dtype = ext.get("dtype", "bfloat16")
            if hasattr(cfg, "dtype"):
                cfg.dtype = cfg_dtype

            # openpi norm_stats.json 은 mean/std 만 있고 q01/q99=None.
            # lerobot QUANTILES normalizer 는 q01/q99 를 필수로 요구 → MEAN_STD 로 override.
            norm_mapping_override = ext.get("normalization_mapping")
            if norm_mapping_override is None and hasattr(cfg, "normalization_mapping"):
                # stats 에 mean/std 만 있으면 MEAN_STD 로 강제 (QUANTILES 는 q01/q99 필요)
                cfg.normalization_mapping = {
                    k: ("MEAN_STD" if v == "QUANTILES" else v)
                    for k, v in cfg.normalization_mapping.items()
                }

            if camera_keys_map:
                # model_key = camera_keys_map value (e.g. "observation.images.base_0_rgb")
                # unified_src → model_key 를 adapter에 저장해 build_remap_config에서 사용
                self._camera_keys_override = dict(camera_keys_map)
                visual_features = {
                    model_key: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
                    for model_key in camera_keys_map.values()
                }
            else:
                visual_features = {
                    camera_key_single: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
                }
            cfg.input_features = {
                **visual_features,
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
                    def _safe_tensor(v):
                        """None 값 제외 — openpi norm_stats 의 q01/q99=None 대응."""
                        return None if v is None else torch.tensor(v)

                    dataset_stats = {}
                    if "state" in raw_norm:
                        dataset_stats["observation.state"] = {
                            sk: _safe_tensor(sv)
                            for sk, sv in raw_norm["state"].items()
                            if sv is not None
                        }
                    if "actions" in raw_norm:
                        dataset_stats["action"] = {
                            sk: _safe_tensor(sv)
                            for sk, sv in raw_norm["actions"].items()
                            if sv is not None
                        }
                    logger.info(
                        "norm_stats loaded from %s: keys=%s",
                        norm_paths[0], list(dataset_stats.keys()),
                    )

            loaded_policy = policy_cls(cfg)
            state_dict = load_file(str(ckpt_dir / "model.safetensors"))
            # LeRobot PI05Policy 가 inner module 을 self.model 로 감싸므로 export 시
            # 모든 키가 "model." prefix 를 가진다. 우리는 loaded_policy.model 에
            # 직접 로드(strict)하므로 그 prefix 를 벗겨 키를 정렬한다. 모든 키가
            # "model." 로 시작할 때만 적용(다른 export 포맷 호환 보존).
            if state_dict and all(k.startswith("model.") for k in state_dict):
                state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
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

        # external 수동 로딩 경로(_loaded_external)는 lerobot processor JSON
        # (policy_preprocessor.json 등)이 없으므로 pretrained_path 를 넘기면
        # ProcessorMigrationError 가 난다. 이 경우 dataset_stats(openpi norm_stats)만으로
        # 프로세서를 구성한다.
        _proc_path = None if _loaded_external else (
            pretrained_path if has_lerobot_config else None
        )
        loaded_pre, loaded_post = make_pre_post_processors(
            policy_cfg=loaded_policy.config,
            pretrained_path=_proc_path,
            dataset_stats=dataset_stats,
        )
        return LoadedPolicy(
            policy=loaded_policy,
            preprocessor=loaded_pre,
            postprocessor=loaded_post,
            pretrained_path=pretrained_path,
            has_lerobot_config=has_lerobot_config,
        )
