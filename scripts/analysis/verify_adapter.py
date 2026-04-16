"""DreamVLA adapter 검증 스크립트.

컨테이너 내에서 실행:
    python /temporal_vla/scripts/analysis/verify_adapter.py

확인 항목:
    1. LeRobotDataset raw 데이터 shape + 실제 값
    2. Adapter 변환 후 shape + 실제 값
    3. Collator 13-tuple shape (최종 모델 입력)
"""

from __future__ import annotations

import sys
import functools
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---- path setup ----
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
import path_setup

path_setup.configure_repo_paths(
    include_dreamvla=True,
    include_datasets=True,
    include_script_utils=True,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from adapters.dreamvla import (
    DreamVLAAdapter,
    robocasa_state_to_dreamvla,
    robocasa_action_to_dreamvla,
)

# ---- config ----
DATASET_ROOT = "/temporal_vla/src/benchmarks/robocasa/datasets/v1.0/target/atomic/PickPlaceSinkToCounter/20250813/lerobot"
FPS = 20
WINDOW_SIZE = 13
ACT_STEP = 3
TOTAL_LEN = WINDOW_SIZE + ACT_STEP - 1  # 15


def dummy_image_fn(pil_images):
    """Dummy image preprocessor: list[PIL] → (T, 3, 224, 224)."""
    tensors = []
    for img in pil_images:
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors)


def dummy_text_fn(texts):
    """Dummy text tokenizer: list[str] → (B, 77) zeros."""
    return torch.zeros(len(texts), 77, dtype=torch.long)


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_tensor_info(name, t, indent=4):
    """텐서의 shape, dtype, 값 범위를 출력."""
    prefix = " " * indent
    if isinstance(t, torch.Tensor):
        print(f"{prefix}{name:40s} | shape {str(list(t.shape)):20s} | {t.dtype}")
        if t.numel() > 0 and t.is_floating_point():
            print(f"{prefix}{'':40s} | min={t.min().item():.4f}  max={t.max().item():.4f}  mean={t.mean().item():.4f}")
    elif isinstance(t, np.ndarray):
        print(f"{prefix}{name:40s} | shape {str(list(t.shape)):20s} | {t.dtype}")
        if t.size > 0 and np.issubdtype(t.dtype, np.floating):
            print(f"{prefix}{'':40s} | min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}")
    else:
        print(f"{prefix}{name:40s} | {type(t).__name__} = {repr(t)[:60]}")


def main():
    # ==================================================================
    # 1. LeRobotDataset — raw 데이터
    # ==================================================================
    print_section("1. LeRobotDataset RAW (변환 전)")

    delta_ts = [i / FPS for i in range(TOTAL_LEN)]
    ds = LeRobotDataset(
        repo_id="robocasa_pickplace",
        root=DATASET_ROOT,
        delta_timestamps={
            "observation.images.robot0_agentview_right": delta_ts,
            "observation.images.robot0_eye_in_hand": delta_ts,
            "observation.state": delta_ts,
            "action": delta_ts,
        },
        episodes=[0, 1, 2],
    )
    print(f"    총 프레임 수: {len(ds)}")

    sample = ds[0]

    print(f"\n    --- 이미지 ---")
    print_tensor_info("observation.images.robot0_agentview_right", sample["observation.images.robot0_agentview_right"])
    print_tensor_info("observation.images.robot0_eye_in_hand", sample["observation.images.robot0_eye_in_hand"])

    print(f"\n    --- State (16D raw) ---")
    state_raw = sample["observation.state"]
    print_tensor_info("observation.state", state_raw)
    print(f"      [0] joint_pos (7D):     {state_raw[0, :7].tolist()}")
    print(f"      [0] eef_pos_rel (3D):   {state_raw[0, 7:10].tolist()}")
    print(f"      [0] eef_quat_rel (4D):  {state_raw[0, 10:14].tolist()}")
    print(f"      [0] gripper_qpos (2D):  {state_raw[0, 14:16].tolist()}")

    print(f"\n    --- Action (12D raw) ---")
    action_raw = sample["action"]
    print_tensor_info("action", action_raw)
    print(f"      [0] joint_pos_delta (5D):  {action_raw[0, :5].tolist()}")
    print(f"      [0] eef_pos_delta (3D):    {action_raw[0, 5:8].tolist()}")
    print(f"      [0] eef_rot_delta (3D):    {action_raw[0, 8:11].tolist()}")
    print(f"      [0] gripper_cmd (1D):      {action_raw[0, 11:12].tolist()}")

    print(f"\n    --- Meta ---")
    print_tensor_info("task", sample["task"])
    print_tensor_info("action_is_pad", sample["action_is_pad"])

    # ==================================================================
    # 2. DreamVLAAdapter — 변환 후
    # ==================================================================
    print_section("2. DreamVLAAdapter (변환 후)")

    adapter = DreamVLAAdapter(
        base_dataset=ds,
        image_fn=dummy_image_fn,
        text_fn=dummy_text_fn,
        state_converter=robocasa_state_to_dreamvla,
        action_converter=robocasa_action_to_dreamvla,
        window_size=WINDOW_SIZE,
        act_step=ACT_STEP,
    )

    item = adapter[0]

    static_imgs = item["rgb_obs"]["rgb_static"]
    wrist_imgs = item["rgb_obs"]["rgb_gripper"]
    robot_obs = item["robot_obs"]
    actions = item["actions"]

    print(f"    --- 이미지 (tensor → PIL 변환) ---")
    print(f"      rgb_static:  {len(static_imgs)} x PIL.Image {static_imgs[0].size}")
    print(f"      rgb_gripper: {len(wrist_imgs)} x PIL.Image {wrist_imgs[0].size}")

    print(f"\n    --- State: 16D → 8D ---")
    print(f"      변환 전: observation.state         {list(state_raw.shape)}")
    print_tensor_info("변환 후: robot_obs", robot_obs)
    print(f"      [0] eef_pos_rel (3D):   {robot_obs[0, :3].tolist()}")
    print(f"      [0] eef_euler (3D):     {robot_obs[0, 3:6].tolist()}")
    print(f"      [0] gripper_qpos (2D):  {robot_obs[0, 6:8].tolist()}")

    print(f"\n    --- Action: 12D → 7D ---")
    print(f"      변환 전: action                    {list(action_raw.shape)}")
    print_tensor_info("변환 후: actions", actions)
    print(f"      [0] eef_pos_delta (3D):  {actions[0, :3].tolist()}")
    print(f"      [0] eef_rot_delta (3D):  {actions[0, 3:6].tolist()}")
    print(f"      [0] gripper_cmd (1D):    {actions[0, 6:7].tolist()}")

    print(f"\n    --- Lang ---")
    print(f"      {item['lang']!r}")

    # shape validation
    assert len(static_imgs) == TOTAL_LEN
    assert robot_obs.shape == (TOTAL_LEN, 8), f"State shape: {robot_obs.shape}"
    assert actions.shape == (TOTAL_LEN, 7), f"Action shape: {actions.shape}"

    # ==================================================================
    # 3. Collator — 최종 모델 입력
    # ==================================================================
    print_section("3. Collator → DreamVLA 모델 입력 (batch_size=2)")

    batch = [adapter[0], adapter[1]]
    result = adapter.collator(batch)

    names = [
        "images_primary", "text", "actions", "images_wrist", "states",
        "robot_obs", "depth_static", "depth_wrist", "dino_static",
        "dino_wrist", "sam_static", "sam_wrist", "track_dict",
    ]
    print(f"    act_step={ACT_STEP} 적용 → window 내 action chunking")
    print()
    for i, (name, val) in enumerate(zip(names, result)):
        if val is None:
            print(f"      [{i:2d}] {name:20s}: None")
        elif isinstance(val, dict):
            print(f"      [{i:2d}] {name:20s}: dict({list(val.keys())})")
        elif isinstance(val, torch.Tensor):
            shape_str = str(list(val.shape))
            print(f"      [{i:2d}] {name:20s}: {shape_str:30s} {val.dtype}")
        else:
            print(f"      [{i:2d}] {name:20s}: {type(val).__name__}")

    # shape validation
    B = 2
    T = WINDOW_SIZE
    assert result[0].shape == (B, T, 3, 224, 224), f"images: {result[0].shape}"
    assert result[2].shape == (B, T, ACT_STEP, 7), f"actions: {result[2].shape}"
    assert result[4].shape == (B, T, 8), f"states: {result[4].shape}"

    print_section("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
