"""
RoboCasa lerobot v2.1 데이터셋을 DreamVLA (Seer/Calvin npz) 포맷으로 변환.

DreamVLA는 Calvin 스타일 npz를 사용:
  episode_XXXXXXX.npz 각각에 rgb_static, rgb_gripper, action, proprio 등 포함.
  + lang_annotations/auto_lang_ann.npy (에피소드별 언어 instruction 매핑)

dreamvla 컨테이너 내에서 실행:
  docker compose --profile dreamvla run --rm dreamvla \
    python /temporal_vla/scripts/convert_robocasa_to_dreamvla.py

robocasa 컨테이너에서도 실행 가능 (비디오 디코딩에 av/cv2 필요):
  docker compose exec robocasa python /temporal_vla/scripts/convert_robocasa_to_dreamvla.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def decode_video_frames(video_path: Path) -> np.ndarray:
    """mp4 -> numpy array [T, H, W, 3] uint8."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


def find_datasets(base: Path, task_type: str | None = None) -> list[Path]:
    if task_type:
        patterns = [f"{task_type}/*/*/lerobot"]
    else:
        patterns = ["atomic/*/*/lerobot", "composite/*/*/lerobot"]
    datasets = []
    for pattern in patterns:
        found = sorted(base.glob(pattern))
        valid = [p for p in found if (p / "meta" / "info.json").exists()]
        datasets.extend(valid)
    return datasets


def get_task_instructions(dataset_dir: Path) -> dict[int, str]:
    """tasks.jsonl에서 task_index -> instruction 매핑."""
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    mapping = {}
    for line in tasks_path.read_text().strip().split("\n"):
        entry = json.loads(line)
        mapping[entry["task_index"]] = entry["task"]
    return mapping


def convert_dataset(src: Path, dst: Path, image_size: int = 224):
    """단일 lerobot v2.1 데이터셋을 DreamVLA npz 포맷으로 변환."""
    dst.mkdir(parents=True, exist_ok=True)

    task_map = get_task_instructions(src)
    info = json.loads((src / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)

    lang_annotations = {"info": {"indx": []}}
    global_frame_idx = 0

    for ep in tqdm(range(total_episodes), desc=f"  episodes"):
        chunk = ep // chunks_size

        # parquet 데이터 로드
        parquet_path = src / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        ep_len = len(df)

        actions = np.stack(df["action"].to_list()).astype(np.float32)
        states = np.stack(df["observation.state"].to_list()).astype(np.float32)

        task_idx = int(df["task_index"].iloc[0])
        instruction = task_map.get(task_idx, "")

        # 비디오에서 이미지 프레임 추출 (agentview_left = static, eye_in_hand = gripper)
        vid_base = src / f"videos/chunk-{chunk:03d}"
        static_vid = vid_base / f"observation.images.robot0_agentview_left/episode_{ep:06d}.mp4"
        gripper_vid = vid_base / f"observation.images.robot0_eye_in_hand/episode_{ep:06d}.mp4"

        if static_vid.exists():
            static_frames = decode_video_frames(static_vid)
        else:
            static_frames = np.zeros((ep_len, image_size, image_size, 3), dtype=np.uint8)

        if gripper_vid.exists():
            gripper_frames = decode_video_frames(gripper_vid)
        else:
            gripper_frames = np.zeros((ep_len, image_size, image_size, 3), dtype=np.uint8)

        # 프레임 수 정렬 (비디오와 parquet 길이 불일치 가능)
        min_len = min(ep_len, len(static_frames), len(gripper_frames))
        actions = actions[:min_len]
        states = states[:min_len]
        static_frames = static_frames[:min_len]
        gripper_frames = gripper_frames[:min_len]

        # 리사이즈
        if static_frames.shape[1] != image_size or static_frames.shape[2] != image_size:
            static_frames = np.array([
                cv2.resize(f, (image_size, image_size)) for f in static_frames
            ])
            gripper_frames = np.array([
                cv2.resize(f, (image_size, image_size)) for f in gripper_frames
            ])

        ep_start = global_frame_idx
        for t in range(min_len):
            npz_path = dst / f"episode_{global_frame_idx:07d}.npz"
            np.savez_compressed(
                npz_path,
                rgb_static=static_frames[t],
                rgb_gripper=gripper_frames[t],
                action=actions[t],
                proprio=states[t],
            )
            global_frame_idx += 1

        ep_end = global_frame_idx - 1
        lang_annotations["info"]["indx"].append([ep_start, ep_end])

    # 언어 annotation 저장
    lang_dir = dst / "lang_annotations"
    lang_dir.mkdir(parents=True, exist_ok=True)

    # instruction을 indx와 매칭
    lang_annotations["language"] = {}
    lang_annotations["language"]["task_desc"] = []
    for ep in range(total_episodes):
        parquet_path = src / f"data/chunk-{ep // chunks_size:03d}/episode_{ep:06d}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        task_idx = int(df["task_index"].iloc[0])
        lang_annotations["language"]["task_desc"].append(task_map.get(task_idx, ""))

    np.save(lang_dir / "auto_lang_ann.npy", lang_annotations, allow_pickle=True)

    print(f"  Saved {global_frame_idx} frames, {total_episodes} episodes")


def main():
    parser = argparse.ArgumentParser(description="RoboCasa -> DreamVLA npz 포맷 변환")
    parser.add_argument("--input-base", type=str, default="/temporal_vla/data/datasets/v1.0")
    parser.add_argument("--output-base", type=str, default="/temporal_vla/data/datasets_dreamvla/v1.0")
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--task-type", type=str, default=None, choices=["atomic", "composite"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_base = Path(args.input_base) / args.split
    dst_base = Path(args.output_base) / args.split

    if not src_base.exists():
        print(f"Error: {src_base} does not exist")
        sys.exit(1)

    datasets = find_datasets(src_base, args.task_type)
    print(f"Found {len(datasets)} datasets under {src_base}")

    if args.dry_run:
        for ds in datasets:
            print(f"  {ds.relative_to(src_base)}")
        return

    success, fail = 0, 0
    for ds in datasets:
        rel = ds.relative_to(src_base)
        task_name = ds.parts[-3]
        dst = dst_base / task_name
        print(f"\n[{task_name}] {rel} -> {dst}")

        try:
            convert_dataset(ds, dst, args.image_size)
            success += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            fail += 1

    print(f"\nDone: {success} succeeded, {fail} failed out of {len(datasets)} total")


if __name__ == "__main__":
    main()
