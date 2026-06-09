"""Adapter for raw Isaac-GR00T N1.5 checkpoints served through LeRobot."""

from __future__ import annotations

import json
import os
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from checkpoint_profile import CheckpointProfile

from .common import (
    FeatureSpec,
    LoadedPolicy,
    STATE_DIM,
    action_dim_from_profile,
    make_policy_features,
    map_container_path,
)
from .rotation import quat_wxyz_to_rotation_6d


GROOT_IMAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "side_0": ("side_0", "left", "static"),
    "side_1": ("side_1", "right"),
    "wrist_0": ("wrist_0", "wrist"),
}

GROOT_STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "eef_pos_rel": ("eef_pos",),
    "eef_quat_rel": ("eef_quat",),
    "base_position": ("base_pos",),
    "base_rotation": ("base_quat",),
}

GROOT_STATE_METADATA_KEYS: dict[str, str] = {
    "eef_pos_rel": "end_effector_position_relative",
    "eef_quat_rel": "end_effector_rotation_relative",
    "gripper_qpos": "gripper_qpos",
    "base_position": "base_position",
    "base_rotation": "base_rotation",
}

GROOT_ACTION_METADATA_KEYS: dict[str, str] = {
    "base_motion": "base_motion",
    "control_mode": "control_mode",
    "eef_pos": "end_effector_position",
    "eef_axisangle": "end_effector_rotation",
    "gripper": "gripper_close",
}

GROOT_STATS_FIELDS = ("min", "max", "mean", "std", "q01", "q99")
GROOT_STATE_ROTATION_6D_KEYS = {"eef_quat_rel", "base_rotation"}
GROOT_NATIVE_BINARY_ACTION_NAMES = {"gripper", "control_mode"}
GROOT_EAGLE_PROJECTOR_PREFIX = "backbone.eagle_model.mlp1."
GROOT_N15_RUNTIME_UTILS = (
    Path(__file__).resolve().parents[2] / "safe" / "groot_n15" / "robocasa" / "utils"
)
GROOT_N15_RUNTIME_MODULE = "_temporal_vla_groot_n15_runtime"
GROOT_ROTATION_6D_STATS = {
    "min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "q01": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "q99": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


def _load_groot_n15_runtime():
    module = sys.modules.get(GROOT_N15_RUNTIME_MODULE)
    if module is not None:
        return module

    runtime_path = GROOT_N15_RUNTIME_UTILS / "runtime.py"
    spec = importlib.util.spec_from_file_location(GROOT_N15_RUNTIME_MODULE, runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GR00T N1.5 runtime helper: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[GROOT_N15_RUNTIME_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _strip_order_prefix(view: str) -> str:
    if len(view) > 3 and view[:2].isdigit() and view[2] == "_":
        return view[3:]
    return view


def _ordered_image_view(index: int, view: str) -> str:
    return f"{index:02d}_{view}"


def _concat_metadata_stats(
    grouped_stats: dict[str, dict[str, Any]],
    keys: list[str],
    rotation_6d_keys: set[str] | None = None,
) -> dict:
    import torch

    rotation_6d_keys = rotation_6d_keys or set()
    missing = [key for key in keys if key not in grouped_stats]
    if missing:
        raise ValueError(f"GR00T metadata missing stats for keys: {missing}")

    flattened = {}
    for field in GROOT_STATS_FIELDS:
        parts = []
        for key in keys:
            if key in rotation_6d_keys:
                value = GROOT_ROTATION_6D_STATS.get(field)
            else:
                value = grouped_stats[key].get(field)
            if value is None:
                break
            parts.append(torch.tensor(value, dtype=torch.float32).flatten())
        if len(parts) == len(keys):
            flattened[field] = torch.cat(parts)
    return flattened


def _groot_state_feature_dim_for_keys(keys: list[str]) -> int:
    dim = 0
    unknown = []
    for key in keys:
        if key in GROOT_STATE_ROTATION_6D_KEYS:
            dim += 6
        elif key in STATE_DIM:
            dim += STATE_DIM[key]
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"Unknown observation state keys for lerobot serve: {unknown}")
    return dim


def load_groot_dataset_stats_from_metadata(
    profile: CheckpointProfile,
    metadata_path: str | Path,
) -> dict:
    """Build LeRobot flat stats from Isaac-GR00T experiment metadata."""
    metadata_path = Path(metadata_path)
    with metadata_path.open() as f:
        metadata = json.load(f)

    embodiment_tag = profile.model_specific.get("embodiment_tag", "new_embodiment")
    try:
        grouped = metadata[embodiment_tag]["statistics"]
    except KeyError as exc:
        raise ValueError(
            f"GR00T metadata has no statistics for embodiment {embodiment_tag!r}: "
            f"{metadata_path}"
        ) from exc

    profile_to_metadata = {
        key: GROOT_STATE_METADATA_KEYS.get(key, key)
        for key in profile.observation_requirements.state
    }
    state_keys = list(profile_to_metadata.values())
    rotation_6d_keys = {
        metadata_key
        for profile_key, metadata_key in profile_to_metadata.items()
        if profile_key in GROOT_STATE_ROTATION_6D_KEYS
    }
    action_keys = [
        GROOT_ACTION_METADATA_KEYS.get(action.name, action.name)
        for action in profile.action_layout
    ]
    return {
        "observation.state": _concat_metadata_stats(
            grouped["state"],
            state_keys,
            rotation_6d_keys=rotation_6d_keys,
        ),
        "action": _concat_metadata_stats(grouped["action"], action_keys),
    }


def build_groot_feature_specs(
    profile: CheckpointProfile,
) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
    resolution = int(profile.image_preprocess.resolution)
    input_specs = [
        FeatureSpec(
            key=f"observation.images.{_ordered_image_view(index, view)}",
            kind="VISUAL",
            shape=(3, resolution, resolution),
        )
        for index, view in enumerate(profile.observation_requirements.images)
    ]
    input_specs.append(
        FeatureSpec(
            key="observation.state",
            kind="STATE",
            shape=(_groot_state_feature_dim_for_keys(profile.observation_requirements.state),),
        )
    )
    output_specs = [
        FeatureSpec(
            key="action",
            kind="ACTION",
            shape=(action_dim_from_profile(profile),),
        )
    ]
    return input_specs, output_specs


def groot_snapshot_allow_patterns(subfolder: str) -> list[str]:
    prefix = subfolder.strip("/")
    return [
        f"{prefix}/config.json",
        f"{prefix}/model.safetensors.index.json",
        f"{prefix}/experiment_cfg/**",
        f"{prefix}/model-*.safetensors",
    ]


def load_groot_eagle_projector_from_checkpoint(
    policy: Any,
    pretrained_path: str | Path,
) -> list[str]:
    """Load Isaac-GR00T Eagle projector tensors into LeRobot's wrapped model."""
    from safetensors import safe_open

    groot_model = getattr(policy, "_groot_model", None)
    backbone = getattr(groot_model, "backbone", None)
    eagle_model = getattr(backbone, "eagle_model", None)
    projector = getattr(eagle_model, "mlp1", None)
    if projector is None:
        raise ValueError("LeRobot GR00T policy does not expose backbone.eagle_model.mlp1")

    checkpoint_dir = Path(pretrained_path)
    state = {}
    for shard_path in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key.startswith(GROOT_EAGLE_PROJECTOR_PREFIX):
                    state[key[len(GROOT_EAGLE_PROJECTOR_PREFIX) :]] = shard.get_tensor(key)

    expected = set(projector.state_dict())
    actual = set(state)
    if not state or expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "GR00T checkpoint/projector state mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    projector.load_state_dict(state, strict=True)
    return sorted(state)


def _build_groot_camera_key_map(visual_keys: list[str]) -> dict[str, str]:
    camera_key_map: dict[str, str] = {}
    for dst in visual_keys:
        view = str(dst).rsplit(".", 1)[-1]
        semantic_view = _strip_order_prefix(view)
        aliases = (view, semantic_view, *GROOT_IMAGE_ALIASES.get(semantic_view, ()))
        for alias in dict.fromkeys(aliases):
            src = f"observation.images.{alias}"
            if src != dst:
                camera_key_map.setdefault(src, dst)
    return camera_key_map


def _groot_action_name_slices(profile: CheckpointProfile) -> dict[str, slice]:
    offset = 0
    slices = {}
    for item in profile.action_layout:
        next_offset = offset + int(item.dims)
        slices[item.name] = slice(offset, next_offset)
        offset = next_offset
    return slices


def _groot_native_binary_action_indices(profile: CheckpointProfile) -> list[int]:
    slices = _groot_action_name_slices(profile)
    indices: list[int] = []
    for name in profile.model_specific.get(
        "native_binary_action_names",
        sorted(GROOT_NATIVE_BINARY_ACTION_NAMES),
    ):
        action_slice = slices.get(name)
        if action_slice is None:
            continue
        indices.extend(range(action_slice.start, action_slice.stop))
    return indices


class GrootNativeActionPostprocessor:
    """Apply native Isaac-GR00T binary inverse on top of LeRobot postprocessing."""

    def __init__(self, postprocessor: Any, binary_indices: list[int]):
        self.postprocessor = postprocessor
        self.binary_indices = tuple(binary_indices)

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        raw_action = action
        processed = self.postprocessor(action) if self.postprocessor is not None else action
        if (
            not self.binary_indices
            or not isinstance(raw_action, torch.Tensor)
            or not isinstance(processed, torch.Tensor)
        ):
            return processed

        raw = raw_action
        if raw.dim() == 3:
            raw = raw[:, -1, :]
        valid_indices = [idx for idx in self.binary_indices if idx < processed.shape[-1]]
        if not valid_indices:
            return processed

        raw = raw[..., : processed.shape[-1]].to(device=processed.device, dtype=processed.dtype)
        result = processed.clone()
        result[..., valid_indices] = (raw[..., valid_indices] > 0.5).to(dtype=processed.dtype)
        return result


def wrap_groot_native_action_postprocessor(
    postprocessor: Any,
    profile: CheckpointProfile,
    _dataset_stats: dict | None = None,
) -> Any:
    if not bool(profile.model_specific.get("native_action_unapply", False)):
        return postprocessor
    binary_indices = _groot_native_binary_action_indices(profile)
    if not binary_indices:
        return postprocessor
    return GrootNativeActionPostprocessor(postprocessor, binary_indices)


class GrootPolicyAdapter:
    """Adapter for raw Isaac-GR00T N1.5 checkpoints behind LeRobot GrootPolicy."""

    def resolve_pretrained_path(
        self,
        profile: CheckpointProfile,
        repo_root: Path,
    ) -> str | Path:
        if profile.checkpoint_source.type != "hf_repo":
            return map_container_path(profile.checkpoint_source.id, repo_root)

        subfolder = profile.model_specific.get("checkpoint_subfolder")
        if not subfolder:
            raise ValueError("groot hf_repo profile requires model_specific.checkpoint_subfolder")

        from huggingface_hub import snapshot_download

        setup_repo_local_hf_cache = _load_groot_n15_runtime().setup_repo_local_hf_cache

        setup_repo_local_hf_cache()
        snapshot = snapshot_download(
            repo_id=profile.checkpoint_source.id,
            repo_type="model",
            allow_patterns=groot_snapshot_allow_patterns(subfolder),
        )
        return Path(snapshot) / subfolder

    def state_payload_keys(self, key: str) -> tuple[str, ...]:
        keys = [key]
        keys.extend(alias for alias in GROOT_STATE_ALIASES.get(key, ()) if alias not in keys)
        return tuple(f"observation.state.{item}" for item in keys)

    def state_dim(self, key: str, default: int) -> int:
        if key in GROOT_STATE_ROTATION_6D_KEYS:
            return 6
        return default

    def transform_state_value(self, key: str, value):
        if key in GROOT_STATE_ROTATION_6D_KEYS:
            return quat_wxyz_to_rotation_6d(value)
        return value

    def build_remap_config(self, visual_keys: list, state_feat) -> tuple[dict[str, str], int]:
        state_dim = state_feat.shape[0] if state_feat is not None else 0
        return _build_groot_camera_key_map(visual_keys), state_dim

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
        from lerobot.policies.groot.configuration_groot import GrootConfig

        runtime = _load_groot_n15_runtime()
        patch_groot_runtime = runtime.patch_groot_runtime
        setup_repo_local_hf_cache = runtime.setup_repo_local_hf_cache

        setup_repo_local_hf_cache()
        if bool(profile.model_specific.get("raw_language", False)):
            os.environ["TEMPORAL_VLA_GROOT_RAW_LANGUAGE"] = "1"
        patch_groot_runtime()

        ms = profile.model_specific
        input_specs, output_specs = build_groot_feature_specs(profile)
        if dataset_stats is None:
            metadata_path = Path(pretrained_path) / "experiment_cfg" / "metadata.json"
            if metadata_path.is_file():
                dataset_stats = load_groot_dataset_stats_from_metadata(
                    profile,
                    metadata_path,
                )
        config = GrootConfig(
            base_model_path=str(pretrained_path),
            tokenizer_assets_repo=ms.get(
                "tokenizer_assets_repo",
                "lerobot/eagle2hg-processor-groot-n1p5",
            ),
            embodiment_tag=ms.get("embodiment_tag", "new_embodiment"),
            chunk_size=int(ms.get("chunk_size", 16)),
            n_action_steps=int(profile.n_action_steps),
            device=device,
            use_bf16=bool(ms.get("use_bf16", True)),
            tune_llm=False,
            tune_visual=False,
            tune_projector=False,
            tune_diffusion_model=False,
            input_features=make_policy_features(input_specs),
            output_features=make_policy_features(output_specs),
        )
        loaded_policy = policy_cls.from_pretrained(str(pretrained_path), config=config)
        if bool(ms.get("load_native_eagle_projector", False)):
            load_groot_eagle_projector_from_checkpoint(loaded_policy, pretrained_path)
        loaded_policy.config.device = device
        loaded_policy.to(device)
        loaded_policy.eval()
        loaded_pre, loaded_post = make_pre_post_processors(
            policy_cfg=loaded_policy.config,
            pretrained_path=None,
            dataset_stats=dataset_stats,
        )
        loaded_post = wrap_groot_native_action_postprocessor(
            loaded_post,
            profile,
            dataset_stats,
        )
        return LoadedPolicy(
            policy=loaded_policy,
            preprocessor=loaded_pre,
            postprocessor=loaded_post,
            pretrained_path=pretrained_path,
            has_lerobot_config=False,
        )
