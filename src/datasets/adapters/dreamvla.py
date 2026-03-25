"""DreamVLA adapter for LeRobot datasets.

Uses LeRobotDataset directly — no custom generic dataset layer needed.
Converts LeRobot output to DreamVLA's 13-tuple collator format that
``train_one_epoch_calvin`` expects.

Responsibilities:
    - State dimension mapping (e.g. RoboCasa 16D → 8D)
    - Action dimension mapping (e.g. RoboCasa 12D → 7D)
    - Image preprocessing (model's image_processor)
    - Text tokenization (CLIP tokenizer)
    - Multi-step action chunking
    - Random shift augmentation
    - SAM feature / CoTracker trajectory loading
    - 13-tuple collator output
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Quaternion → Euler (intrinsic XYZ, i.e. roll-pitch-yaw)
# ---------------------------------------------------------------------------

def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


# ---------------------------------------------------------------------------
# State / Action converters (RoboCasa → DreamVLA Calvin-compatible)
# ---------------------------------------------------------------------------

def robocasa_state_to_dreamvla(state_16d: np.ndarray) -> np.ndarray:
    """RoboCasa 16D → DreamVLA 8D (eef_pos_rel 3 + euler 3 + gripper_qpos 2).

    Compatible with DreamVLA's arm_state_encoder(6D) + gripper_state_encoder(2D)
    when --gripper_width is set.
    """
    eef_pos = state_16d[..., 7:10]      # relative EEF position (3)
    eef_quat = state_16d[..., 10:14]    # relative EEF quaternion wxyz (4)
    eef_euler = quat_to_euler(eef_quat)  # → euler rpy (3)
    gripper = state_16d[..., 14:16]     # gripper joint positions (2)
    return np.concatenate([eef_pos, eef_euler, gripper], axis=-1)


def robocasa_action_to_dreamvla(action_12d: np.ndarray) -> np.ndarray:
    """RoboCasa 12D → DreamVLA 7D (eef_pos 3 + eef_rot 3 + gripper 1).

    Compatible with DreamVLA's arm_action_decoder(→6D) + gripper_action_decoder(→1D).
    """
    eef_pos = action_12d[..., 5:8]      # EEF position delta (3)
    eef_rot = action_12d[..., 8:11]     # EEF rotation delta (3)
    gripper = action_12d[..., 11:12]    # gripper close command (1)
    return np.concatenate([eef_pos, eef_rot, gripper], axis=-1)


# ---------------------------------------------------------------------------
# DreamVLA Adapter Dataset
# ---------------------------------------------------------------------------

class DreamVLAAdapter(Dataset):
    """Wraps a LeRobotDataset to produce DreamVLA-compatible training samples.

    Parameters
    ----------
    base_dataset : LeRobotDataset
        A LeRobotDataset instance with delta_timestamps configured.
        __getitem__ returns torch tensors + ``task`` string.
    image_fn : callable
        Image preprocessor: list[PIL.Image] → Tensor[S, C, H, W].
    text_fn : callable
        Text tokenizer: list[str] → Tensor.
    state_converter : callable | None
        Raw state → model state. If None, passes through unchanged.
    action_converter : callable | None
        Raw action → model action. If None, passes through unchanged.
    static_key : str
        LeRobot feature key for static camera images.
    wrist_key : str
        LeRobot feature key for wrist camera images.
    window_size : int
        Observation window size (frames used as input context).
    act_step : int
        Multi-step action prediction horizon.
    rgb_pad, gripper_pad : int
        Random shift augmentation size (-1 to disable).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        image_fn: Callable,
        text_fn: Callable,
        state_converter: Optional[Callable] = None,
        action_converter: Optional[Callable] = None,
        static_key: str = "observation.images.robot0_agentview_right",
        wrist_key: str = "observation.images.robot0_eye_in_hand",
        window_size: int = 10,
        act_step: int = 1,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        sam_features_path: Optional[str] = None,
        track_label_path: Optional[str] = None,
    ):
        self.base = base_dataset
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.state_converter = state_converter or (lambda x: x)
        self.action_converter = action_converter or (lambda x: x)
        self.static_key = static_key
        self.wrist_key = wrist_key
        self.window_size = window_size
        self.act_step = act_step
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.sam_features_path = sam_features_path
        self.track_label_path = track_label_path
        self.total_len = window_size + act_step - 1

        if self.rgb_pad != -1:
            from torchvision.transforms import RandomCrop
            self.rgb_shift = RandomCrop(
                size=(224, 224), padding=self.rgb_pad, pad_if_needed=True
            )
        if self.gripper_pad != -1:
            from torchvision.transforms import RandomCrop
            self.gripper_shift = RandomCrop(
                size=(224, 224), padding=self.gripper_pad, pad_if_needed=True
            )

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _tensor_to_pil_list(img_tensor: torch.Tensor) -> List[Image.Image]:
        """Convert (T, C, H, W) float [0,1] tensor → list[PIL.Image]."""
        imgs = []
        for t in range(img_tensor.shape[0]):
            arr = (img_tensor[t].permute(1, 2, 0) * 255).byte().numpy()
            imgs.append(Image.fromarray(arr))
        return imgs

    def _load_sam_feat(self, frame_idx: int, camera_name: str) -> torch.Tensor:
        """Load SAM feature for a single frame. Returns (C, 256) tensor."""
        path = os.path.join(
            self.sam_features_path, camera_name, "training", f"{frame_idx}.pt"
        )
        return torch.load(path, weights_only=True).to(torch.float32)

    def _load_track_label(self, frame_idx: int, camera_name: str) -> Dict[str, np.ndarray]:
        """Load CoTracker track label for a single frame."""
        path = os.path.join(
            self.track_label_path, camera_name, "training", f"{frame_idx}.npz"
        )
        data = np.load(path)
        return {"tracks": data["tracks"], "visibility": data["visibility"]}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]

        # Skip padded samples (frames beyond episode boundary)
        pad_key = f"{self.static_key}_is_pad"
        if pad_key in sample and sample[pad_key].any():
            return self[torch.randint(len(self), (1,)).item()]

        # LeRobot returns (T, C, H, W) tensors → convert to PIL for image_fn
        static_imgs = self._tensor_to_pil_list(sample[self.static_key])
        wrist_imgs = self._tensor_to_pil_list(sample[self.wrist_key])

        # State / action: tensor → numpy for converters
        state = sample["observation.state"].numpy()   # (T, 16)
        action = sample["action"].numpy()              # (T, 12)
        lang = sample.get("task", "")

        result = {
            "rgb_obs": {
                "rgb_static": static_imgs,
                "rgb_gripper": wrist_imgs,
            },
            "actions": self.action_converter(action),
            "robot_obs": self.state_converter(state),
            "lang": lang,
        }

        # Frame indices for the temporal window: [idx, idx+1, ..., idx+total_len-1]
        frame_indices = list(range(idx, idx + self.total_len))

        # SAM features
        if self.sam_features_path is not None:
            sam_static = torch.stack(
                [self._load_sam_feat(fi, "rgb_static") for fi in frame_indices]
            )  # (T, C, 256)
            sam_gripper = torch.stack(
                [self._load_sam_feat(fi, "rgb_gripper") for fi in frame_indices]
            )  # (T, C, 256)
            result["sam_static"] = sam_static
            result["sam_gripper"] = sam_gripper

        # Track labels
        if self.track_label_path is not None:
            tracks_list = [self._load_track_label(fi, "rgb_static") for fi in frame_indices]
            tracks_grip_list = [self._load_track_label(fi, "rgb_gripper") for fi in frame_indices]
            result["track_label"] = {
                "tracks": torch.from_numpy(
                    np.stack([t["tracks"] for t in tracks_list])
                ).float(),  # (T, 784, 2)
                "track_visibility": torch.from_numpy(
                    np.stack([t["visibility"] for t in tracks_list])
                ).float(),  # (T, 784)
                "tracks_gripper": torch.from_numpy(
                    np.stack([t["tracks"] for t in tracks_grip_list])
                ).float(),  # (T, 784, 2)
                "track_visibility_gripper": torch.from_numpy(
                    np.stack([t["visibility"] for t in tracks_grip_list])
                ).float(),  # (T, 784)
            }

        return result

    # ------------------------------------------------------------------
    # Collator — produces the 13-tuple for train_one_epoch_calvin
    # ------------------------------------------------------------------

    def collator(self, sample: List[Dict[str, Any]]):
        action_tensors = torch.from_numpy(
            np.array([np.stack(s["actions"]) for s in sample])
        ).float()
        state_tensors = torch.from_numpy(
            np.array([np.stack(s["robot_obs"]) for s in sample])
        ).float()
        image_tensors = torch.stack(
            [self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample]
        )
        gripper_tensors = torch.stack(
            [self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample]
        )

        text_tensors = self.text_fn([s["lang"] for s in sample])

        # train_one_epoch_calvin이 (B, T, action_dim) flat 시퀀스를 받아서
        # 자체적으로 action_pred_steps만큼 chunking하므로, 여기서는 하지 않는다.
        robot_obs = torch.zeros(1)

        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            image_tensors = image_tensors.reshape(bs * seq_len, *image_tensors.shape[2:])
            image_tensors = self.rgb_shift(image_tensors)
            image_tensors = image_tensors.reshape(bs, seq_len, *image_tensors.shape[1:])

        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            gripper_tensors = gripper_tensors.reshape(bs * seq_len, *gripper_tensors.shape[2:])
            gripper_tensors = self.gripper_shift(gripper_tensors)
            gripper_tensors = gripper_tensors.reshape(bs, seq_len, *gripper_tensors.shape[1:])

        # SAM features
        sam_static = None
        sam_wrist = None
        if "sam_static" in sample[0]:
            sam_static = torch.stack([s["sam_static"] for s in sample])   # (B, T, C, 256)
            sam_wrist = torch.stack([s["sam_gripper"] for s in sample])   # (B, T, C, 256)

        # Track labels
        track_dict = {}
        if "track_label" in sample[0]:
            track_dict = {
                "tracks": torch.stack(
                    [s["track_label"]["tracks"] for s in sample]
                ),  # (B, T, 784, 2)
                "track_visibility": torch.stack(
                    [s["track_label"]["track_visibility"] for s in sample]
                ),  # (B, T, 784)
                "tracks_gripper": torch.stack(
                    [s["track_label"]["tracks_gripper"] for s in sample]
                ),  # (B, T, 784, 2)
                "track_visibility_gripper": torch.stack(
                    [s["track_label"]["track_visibility_gripper"] for s in sample]
                ),  # (B, T, 784)
            }

        return (
            image_tensors,       # [0] images_primary
            text_tensors,        # [1] text
            action_tensors,      # [2] actions
            gripper_tensors,     # [3] images_wrist
            state_tensors,       # [4] states
            robot_obs,           # [5] robot_obs
            None,                # [6] depth_static
            None,                # [7] depth_wrist
            None,                # [8] dino_static
            None,                # [9] dino_wrist
            sam_static,          # [10] sam_static
            sam_wrist,           # [11] sam_wrist
            track_dict,          # [12] track_dict
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_robocasa_dreamvla_dataset(args, image_processor, tokenizer, epoch: int = 0):
    """Build RoboCasa + DreamVLA adapter dataloader.

    Returns a DataInfo compatible with DreamVLA's training loop.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from utils.data_utils import (
        DataInfo,
        SharedEpoch,
        preprocess_image,
        preprocess_text_calvin,
    )

    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(
        preprocess_text_calvin, tokenizer=tokenizer
    )

    # --- LeRobotDataset with temporal window via delta_timestamps ---
    static_key = getattr(args, "static_camera_key", "observation.images.robot0_agentview_right")
    wrist_key = getattr(args, "wrist_camera_key", "observation.images.robot0_eye_in_hand")

    fps = getattr(args, "fps", 20)
    total_len = args.window_size + args.multi_step_action - 1
    delta_ts = [i / fps for i in range(total_len)]

    delta_timestamps = {
        static_key: delta_ts,
        wrist_key: delta_ts,
        "observation.state": delta_ts,
        "action": delta_ts,
    }

    max_episodes = getattr(args, "max_episodes", None)

    base_dataset = LeRobotDataset(
        repo_id=getattr(args, "robocasa_repo_id", "robocasa_pickplace"),
        root=args.robocasa_dataset,
        delta_timestamps=delta_timestamps,
        episodes=list(range(max_episodes)) if max_episodes else None,
    )

    sam_features_path = getattr(args, "sam_features_path", None)
    track_label_path = getattr(args, "track_label_path", None)
    if not getattr(args, "load_sam_features", False):
        sam_features_path = None
    if not getattr(args, "load_track_labels", False):
        track_label_path = None

    dataset = DreamVLAAdapter(
        base_dataset=base_dataset,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        state_converter=robocasa_state_to_dreamvla,
        action_converter=robocasa_action_to_dreamvla,
        static_key=static_key,
        wrist_key=wrist_key,
        window_size=args.window_size,
        act_step=args.multi_step_action,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        sam_features_path=sam_features_path,
        track_label_path=track_label_path,
    )

    round_fn = math.ceil
    num_samples = len(dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=dataset.collator,
        drop_last=True,
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(
        dataloader=dataloader,
        shared_epoch=SharedEpoch(epoch=epoch),
        sampler=sampler,
        dataset=dataset,
    )
