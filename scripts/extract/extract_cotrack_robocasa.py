"""Extract CoTracker trajectory labels from RoboCasa LeRobot dataset.

Adapted from src/policies/dreamvla/data_process/cotrack_extractor.py for LeRobot format.
Saves per-frame optical flow tracks as .npz files.

Usage (in dreamvla container):
    python scripts/extract/extract_cotrack_robocasa.py \
        --dataset_path /temporal_vla/src/benchmarks/robocasa/datasets/v1.0/.../lerobot \
        --save_path /temporal_vla/src/benchmarks/robocasa/datasets/features/tracks \
        --checkpoint /temporal_vla/src/policies/dreamvla/co-tracker/checkpoints/scaled_offline.pth \
        --frame_gap 5 --patch_size 8 --num_workers 4
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# -- path setup --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "policies", "dreamvla"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "policies", "dreamvla", "co-tracker"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lerobot", "src"))

from cotracker.predictor import CoTrackerPredictor


def get_points_on_a_grid(patch_size, image_size, device):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    H, W = image_size
    ph, pw = patch_size
    y_centers = np.arange(ph // 2, H, ph)
    x_centers = np.arange(pw // 2, W, pw)
    xv, yv = np.meshgrid(x_centers, y_centers)
    centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
    return torch.from_numpy(centers).to(device)


def save_track_label(save_path, frame_idx, trk, vis):
    file_path = os.path.join(save_path, f"{frame_idx}.npz")
    with open(file_path, "wb") as f:
        np.savez_compressed(f, tracks=trk, visibility=vis)


def load_frame_image(dataset, idx, image_key):
    """Load a single frame as 224x224 PIL Image from LeRobotDataset."""
    sample = dataset[idx]
    img_tensor = sample[image_key]
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    arr = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()
    return Image.fromarray(arr).resize((224, 224))


def worker_fn(worker_id, ep_indices, args, image_key, camera_name):
    """Worker process: loads its own model and dataset, processes assigned episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    save_dir = os.path.join(args.save_path, camera_name, "training")
    os.makedirs(save_dir, exist_ok=True)

    base_dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_path)

    # Each worker loads its own model
    if args.checkpoint is not None:
        model = CoTrackerPredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to("cuda")

    H, W = 224, 224
    grid_pts = get_points_on_a_grid(args.patch_size, [H, W], device="cpu").float().unsqueeze(0)
    grid_size = H // args.patch_size
    n_grid_points = grid_pts.shape[1]

    ep_meta = base_dataset.meta.episodes
    ep_from = ep_meta["dataset_from_index"]
    ep_to = ep_meta["dataset_to_index"]

    zero_tracks = np.zeros((n_grid_points, 2), dtype=np.float32)
    zero_vis = np.ones((n_grid_points,), dtype=bool)

    saver = ThreadPoolExecutor(max_workers=4)

    for ep_idx in tqdm(ep_indices, desc=f"{camera_name}-w{worker_id}", position=worker_id):
        ep_start = ep_from[ep_idx]
        ep_end = ep_to[ep_idx]
        ep_len = ep_end - ep_start

        if ep_len < args.frame_gap + 1:
            for frame_idx in range(ep_start, ep_end):
                out_path = os.path.join(save_dir, f"{frame_idx}.npz")
                if not os.path.exists(out_path):
                    saver.submit(save_track_label, save_dir, frame_idx, zero_tracks, zero_vis)
            continue

        # Check if all frames already exist
        if not args.override:
            all_exist = all(
                os.path.exists(os.path.join(save_dir, f"{ep_start + i}.npz"))
                for i in range(ep_len)
            )
            if all_exist:
                continue

        # Load all frames for this episode
        frames = []
        for idx in range(ep_start, ep_end):
            img = load_frame_image(base_dataset, idx, image_key)
            frames.append(np.array(img))

        frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()

        # Create frame pairs
        num_pairs = ep_len - args.frame_gap
        video_pairs = frames_tensor.unfold(0, args.frame_gap + 1, 1)
        video_pairs = video_pairs[..., [0, -1]]
        video_pairs = video_pairs.permute(0, 4, 1, 2, 3).contiguous()

        queries = torch.cat(
            [torch.zeros(grid_pts.shape[0], grid_pts.shape[1], 1), grid_pts],
            dim=2,
        ).repeat(video_pairs.shape[0], 1, 1)

        batch_size = args.batch_size
        all_tracks = []
        all_vis = []

        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            with torch.no_grad():
                pred_tracks, pred_visibility = model(
                    video_pairs[start:end].cuda(),
                    queries=queries[start:end].cuda(),
                    grid_size=grid_size,
                    backward_tracking=False,
                )
            delta = (pred_tracks[:, 1:2, :, :] - pred_tracks[:, 0:1, :, :]).squeeze(1)
            all_tracks.append(delta.cpu().numpy())
            all_vis.append(pred_visibility[:, 1, :].cpu().numpy())

        if len(all_tracks) > 0:
            all_tracks = np.concatenate(all_tracks, axis=0)
            all_vis = np.concatenate(all_vis, axis=0)

        for i in range(ep_len):
            frame_idx = ep_start + i
            out_path = os.path.join(save_dir, f"{frame_idx}.npz")
            if os.path.exists(out_path) and not args.override:
                continue
            if i < num_pairs:
                trk = all_tracks[i]
                vis = all_vis[i]
            else:
                trk = zero_tracks
                vis = zero_vis
            saver.submit(save_track_label, save_dir, frame_idx, trk, vis)

    saver.shutdown(wait=True)


def extract_tracks_parallel(args, image_key, camera_name):
    """Extract CoTracker tracks using multiple worker processes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    save_dir = os.path.join(args.save_path, camera_name, "training")
    os.makedirs(save_dir, exist_ok=True)

    # Load metadata to get episode count
    base_dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_path)
    num_episodes = len(base_dataset.meta.episodes["dataset_from_index"])
    del base_dataset

    print(f"Extracting CoTracker tracks for {camera_name} ({num_episodes} episodes, {args.num_workers} workers)")

    # Split episodes across workers
    ep_lists = [[] for _ in range(args.num_workers)]
    for i in range(num_episodes):
        ep_lists[i % args.num_workers].append(i)

    processes = []
    for w_id, ep_indices in enumerate(ep_lists):
        p = Process(target=worker_fn, args=(w_id, ep_indices, args, image_key, camera_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"{camera_name} extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Extract CoTracker tracks from RoboCasa LeRobot dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--save_path", type=str, required=True, help="Output directory for track labels")
    parser.add_argument("--checkpoint", type=str, default=None, help="CoTracker model checkpoint path")
    parser.add_argument("--repo_id", type=str, default="robocasa_pickplace", help="LeRobot repo_id")
    parser.add_argument("--frame_gap", type=int, default=5, help="Temporal gap between frame pairs")
    parser.add_argument("--patch_size", type=int, default=8, help="Grid patch size (8 -> 28x28 = 784 points)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument(
        "--static_key", type=str, default="observation.images.robot0_agentview_right"
    )
    parser.add_argument(
        "--wrist_key", type=str, default="observation.images.robot0_eye_in_hand"
    )
    parser.add_argument(
        "--camera", type=str, default="both",
        choices=["static", "gripper", "both"],
        help="Which camera to process (for parallel execution)"
    )
    args = parser.parse_args()

    if args.camera in ("static", "both"):
        extract_tracks_parallel(args, args.static_key, "rgb_static")
    if args.camera in ("gripper", "both"):
        extract_tracks_parallel(args, args.wrist_key, "rgb_gripper")

    print("CoTracker trajectory extraction complete.")


if __name__ == "__main__":
    main()
