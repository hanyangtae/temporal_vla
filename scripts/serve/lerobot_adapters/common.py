"""Shared helpers for LeRobot serving adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from checkpoint_profile import CheckpointProfile


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    kind: str
    shape: tuple[int, ...]


@dataclass
class LoadedPolicy:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    pretrained_path: str | Path
    has_lerobot_config: bool


STATE_DIM: dict[str, int] = {
    "eef_pos": 3,
    "eef_pos_rel": 3,
    "eef_euler": 3,
    "eef_axisangle": 3,
    "eef_quat": 3,
    "eef_quat_rel": 4,
    "eef_rot6d": 6,
    "gripper_qpos": 2,
    "gripper_opening": 1,
    "gripper_action": 1,
    "joint_pos": 7,
    "joint_vel": 7,
    "base_pos": 3,
    "base_quat": 4,
    "base_position": 3,
    "base_rotation": 4,
}


def map_container_path(path: str, repo_root: Path) -> str:
    if path.startswith("/temporal_vla/"):
        mapped = repo_root / path[len("/temporal_vla/") :]
        if mapped.exists():
            return str(mapped)
    return path


def load_dataset_stats(profile: CheckpointProfile) -> dict | None:
    stats_path = profile.model_specific.get("dataset_stats_path")
    if not stats_path:
        return None

    import json

    with open(stats_path) as f:
        raw = json.load(f)
    return {
        k: {sk: torch.tensor(sv) for sk, sv in sv_dict.items()}
        for k, sv_dict in raw.items()
    }


def preprocess_image_numpy(np_img: np.ndarray, image_preprocess) -> np.ndarray:
    if image_preprocess is None:
        return np_img

    arr = np.asarray(np_img, dtype=np.uint8)
    do_center_crop = bool(getattr(image_preprocess, "center_crop", False))
    resolution = int(getattr(image_preprocess, "resolution", 0) or 0)
    needs_resize = resolution > 0 and arr.shape[:2] != (resolution, resolution)
    if not do_center_crop and not needs_resize:
        return arr

    try:
        import torchvision.transforms.v2 as tv2
    except Exception:
        tv2 = None

    if tv2 is not None:
        frame = torch.from_numpy(arr.copy()).to(torch.float32) / 255.0
        frame = frame.permute(2, 0, 1).unsqueeze(0)
        if do_center_crop:
            scale = float(getattr(image_preprocess, "center_crop_scale", 0.95))
            if not 0.0 < scale <= 1.0:
                raise ValueError(f"invalid image_preprocess.center_crop_scale={scale}")
            height, width = frame.shape[-2:]
            frame = tv2.CenterCrop((int(height * scale), int(width * scale)))(frame)
        if needs_resize:
            frame = tv2.Resize(
                (resolution, resolution),
                interpolation=tv2.InterpolationMode.BILINEAR,
                antialias=True,
            )(frame)
        return (frame[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()

    if do_center_crop:
        scale = float(getattr(image_preprocess, "center_crop_scale", 0.95))
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"invalid image_preprocess.center_crop_scale={scale}")
        height, width = arr.shape[:2]
        crop_h = max(1, int(height * scale))
        crop_w = max(1, int(width * scale))
        top = max((height - crop_h) // 2, 0)
        left = max((width - crop_w) // 2, 0)
        arr = arr[top : top + crop_h, left : left + crop_w]

    if resolution > 0 and arr.shape[:2] != (resolution, resolution):
        arr = np.asarray(
            Image.fromarray(arr).resize((resolution, resolution), Image.BILINEAR),
            dtype=np.uint8,
        )
    return arr


def make_policy_features(specs: list[FeatureSpec]) -> dict:
    from lerobot.configs.types import FeatureType, PolicyFeature

    feature_types = {
        "VISUAL": FeatureType.VISUAL,
        "STATE": FeatureType.STATE,
        "ACTION": FeatureType.ACTION,
    }
    return {
        spec.key: PolicyFeature(type=feature_types[spec.kind], shape=spec.shape)
        for spec in specs
    }


def state_dim_for_keys(keys: list[str]) -> int:
    dim = 0
    unknown = []
    for key in keys:
        if key not in STATE_DIM:
            unknown.append(key)
            continue
        dim += STATE_DIM[key]
    if unknown:
        raise ValueError(f"Unknown observation state keys for lerobot serve: {unknown}")
    return dim


def action_dim_from_profile(profile: CheckpointProfile) -> int:
    return sum(a.dims for a in profile.action_layout)
