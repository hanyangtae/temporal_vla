"""
VLA 평가 스크립트 (Calvin 환경, 모델 무관).

통일 API를 따르는 어떤 VLA 서버든 --server-url만 바꾸면 평가 가능.
calvin 컨테이너에서 실행:

# UP-VLA로 평가
docker compose run --rm calvin python /temporal_vla/scripts/calvin_eval.py \
  --dataset-path /temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset \
  --server-url http://localhost:8300 \
  --num-sequences 1 \
  --video-dir /temporal_vla/outputs/calvin_eval/video \
  --num-videos 1

# DreamVLA로 평가 (같은 스크립트, URL만 변경)
docker compose run --rm calvin python /temporal_vla/scripts/calvin_eval.py \
  --dataset-path /temporal_vla/data/calvin/task_ABC_D \
  --server-url http://localhost:8200 \
  --num-sequences 1000

VLA 서버(serve_*.py)가 먼저 실행 중이어야 한다.
"""

import argparse
import json
import logging
import os

# Calvin-native imports (calvin 컨테이너에 설치됨)
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/temporal_vla/scripts/utils")
sys.path.insert(0, "/temporal_vla/src/benchmarks/calvin/calvin_env")
sys.path.insert(0, "/temporal_vla/src/benchmarks/calvin/calvin_models")
sys.path.insert(0, "/temporal_vla/src")

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import count_success, get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── VLA 클라이언트 (통일 API) ────────────────────────────────────────────────

from vla_client import VLAClient
from processor.factory import make_calvin_processors
from processor.types import TransitionKey


def _draw_border(
    frames: List[np.ndarray], success: bool, border: int = 3, repeat: int = 5
) -> List[np.ndarray]:
    """원본 draw_outcome()과 동일: 마지막 프레임에만 테두리를 그리고 repeat번 반복해서 붙인다.
    성공=초록(0,255,0), 실패=빨강(255,0,0).
    """
    color = (0, 255, 0) if success else (255, 0, 0)
    last = frames[-1].copy()
    last[:border, :] = color  # top
    last[-border:, :] = color  # bottom
    last[:, :border] = color  # left
    last[:, -border:] = color  # right
    return frames + [last] * repeat


def _draw_instruction(frames: List[np.ndarray], text: str) -> List[np.ndarray]:
    """원본 add_text() (utils.py L160)와 동일하게 instruction 텍스트를 프레임에 오버레이한다.
    cv2.putText 두 번: 검정 두꺼운 외곽선 → 흰 텍스트 (stroke 효과).
    위치: 좌하단 (1, height-10), 폰트 크기: 너비 기준 비례.
    """
    import cv2

    result = []
    for f in frames:
        img = f.copy()
        h, w = img.shape[:2]
        coord = (1, int(h - 10))
        font_scale = (0.7 / 500) * w
        cv2.putText(
            img,
            text,
            coord,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            coord,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        result.append(img)
    return result


def _save_video(frames: List[np.ndarray], path: Path, fps: int = 20):
    """프레임 리스트 → MP4 저장. imageio 없으면 GIF로 fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        imageio.mimwrite(str(path), frames, fps=fps)
    except ImportError:
        gif_path = path.with_suffix(".gif")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        logger.warning("imageio not available, saved as GIF: %s", gif_path)


def _predict(
    vla_client: VLAClient,
    processed_obs: Dict,
    instruction: str,
) -> Tuple:
    """통일 API로 액션 예측. (actions [act_step, 7], None, None) 반환.

    processed_obs는 obs_pipeline 을 거친 결과:
      {"observation.images.static": uint8 HWC, ...}
    """
    images = {}
    for k, v in processed_obs.items():
        if k.startswith("observation.images."):
            cam_name = k.split("observation.images.")[-1]
            images[cam_name] = v

    actions, _ = vla_client.predict(images=images, instruction=instruction)
    return actions, None, None


# ─── Calvin env 셋업 ─────────────────────────────────────────────────────────


def _setup_egl(cuda_id: int = 0):
    if "EGL_VISIBLE_DEVICES" in os.environ:
        return
    try:
        egl_id = get_egl_device_id(cuda_id)
    except EglDeviceNotFoundError:
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    logger.info("EGL device: %d (CUDA %d)", egl_id, cuda_id)


def _get_raw_image(obs: Dict, key: str) -> np.ndarray:
    """Calvin raw obs에서 uint8 HxWx3 이미지 추출 (비디오 프레임 추출용).

    obs_pipeline은 observation.images.* 키를 생성하지만,
    비디오 프레임 저장 시 원본 obs에서 직접 이미지를 추출해야 할 때 사용.
    """
    from processor.obs.calvin import CalvinObsProcessor
    return CalvinObsProcessor._convert(obs["rgb_obs"][key])


# ─── 평가 루프 ───────────────────────────────────────────────────────────────


def _rollout(
    env,
    task_oracle,
    subtask: str,
    instruction: str,
    vla_client: VLAClient,
    obs_pipeline,
    action_pipeline,
    act_step: int,
    ep_len: int,
    record: bool = False,
    video_dir: Optional[Path] = None,
    seq_idx: int = 0,
    subtask_idx: int = 0,
) -> Tuple[bool, List[np.ndarray]]:
    """단일 subtask 롤아웃. (성공 여부, 프레임 리스트) 반환."""
    obs = env.get_obs()
    start_info = env.get_info()
    action_buffer = None
    frames = []

    for step in range(ep_len):
        if step % act_step == 0:
            processed = obs_pipeline({TransitionKey.OBSERVATION: obs})
            processed_obs = processed[TransitionKey.OBSERVATION]
            action_buffer, _, _ = _predict(
                vla_client, processed_obs, instruction
            )

        raw_action = action_buffer[step % act_step]
        processed = action_pipeline({TransitionKey.ACTION: raw_action})
        obs, _, _, current_info = env.step(processed[TransitionKey.ACTION])

        if record:
            frames.append(_get_raw_image(obs, "rgb_static"))

        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            return True, frames

    return False, frames


def _evaluate_sequence(
    env,
    task_oracle,
    initial_state: Dict,
    eval_sequence: List[str],
    val_annotations: Dict,
    vla_client: VLAClient,
    obs_pipeline,
    action_pipeline,
    act_step: int,
    ep_len: int,
    record: bool = False,
    video_dir: Optional[Path] = None,
    seq_idx: int = 0,
) -> int:
    """5-subtask 시퀀스 평가. 연속 성공 수 반환."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    all_frames = []
    for subtask_idx, subtask in enumerate(eval_sequence):
        instruction = val_annotations[subtask][0]
        success, frames = _rollout(
            env,
            task_oracle,
            subtask,
            instruction,
            vla_client,
            obs_pipeline,
            action_pipeline,
            act_step,
            ep_len,
            record=record,
            video_dir=video_dir,
            seq_idx=seq_idx,
            subtask_idx=subtask_idx,
        )
        if record and frames:
            frames = _draw_instruction(frames, instruction)
            frames = _draw_border(frames, success)
            all_frames.extend(frames)
        if success:
            success_counter += 1
        else:
            break

    # MP4 저장: 기록 대상이고 완전 성공(5/5)이 아닌 경우
    if record and video_dir is not None and all_frames and success_counter < 4:
        mp4_path = video_dir / "seq{:04d}_result{}.mp4".format(seq_idx, success_counter)
        _save_video(all_frames, mp4_path)

    return success_counter


def evaluate(
    dataset_path: Path,
    calvin_conf: Path,
    server_url: str,
    num_sequences: int,
    ep_len: int,
    act_step: int,
    output_path: Optional[Path],
    num_videos: int = 0,
    video_dir: Optional[Path] = None,
):
    # 통일 VLA 클라이언트
    vla_client = VLAClient(url=server_url)
    logger.info("VLA 서버 연결 대기 중: %s", server_url)
    server_info = vla_client.wait_until_ready()
    logger.info("VLA 서버 연결 완료: %s", server_info)

    # Processor pipeline (벤치마크별 obs/action 변환)
    obs_pipeline, action_pipeline = make_calvin_processors()

    # task oracle (calvin_env.envs.tasks.Tasks) — Calvin-native 클래스
    tasks_cfg = OmegaConf.load(
        calvin_conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(tasks_cfg)

    # 언어 어노테이션 (task_name → [annotation_strings])
    ann_cfg = OmegaConf.load(calvin_conf / "annotations/new_playtable_validation.yaml")
    val_annotations = OmegaConf.to_container(ann_cfg, resolve=True)

    # Calvin env
    # get_env()는 .hydra/config.yaml이 있는 split 디렉토리를 기대함
    # task_ABC_D/ 또는 calvin_debug_dataset/ 모두 하위에 validation/ 이 있는 구조
    env_path = dataset_path / "validation"
    if not env_path.exists():
        raise FileNotFoundError(
            f"{env_path} 가 없습니다. --dataset-path 는 split 루트 "
            f"(e.g. task_ABC_D/ 또는 calvin_debug_dataset/) 여야 합니다."
        )
    _setup_egl()
    env = get_env(str(env_path), show_gui=False)

    # 평가 시퀀스 생성
    sequences = get_sequences(num_sequences)

    results = []
    pbar = tqdm(sequences, desc="Evaluating")
    for seq_idx, (initial_state, eval_sequence) in enumerate(pbar):
        record = num_videos > 0 and seq_idx < num_videos
        vla_client.reset()
        result = _evaluate_sequence(
            env,
            task_oracle,
            initial_state,
            list(eval_sequence),
            val_annotations,
            vla_client,
            obs_pipeline,
            action_pipeline,
            act_step,
            ep_len,
            record=record,
            video_dir=video_dir,
            seq_idx=seq_idx,
        )
        results.append(result)

        success_rates = count_success(results)
        avg = sum(success_rates) / len(success_rates) * 5
        desc = " | ".join(f"{i+1}/5:{v*100:.1f}%" for i, v in enumerate(success_rates))
        pbar.set_description(f"{desc} | Avg:{avg:.2f}")

    # 결과 출력
    success_rates = count_success(results)
    avg_len = np.mean(results)
    print("\n=== VLA Calvin Evaluation Results ===")
    print(f"Sequences evaluated: {len(results)}")
    print(f"Average sequence length: {avg_len:.3f}")
    for i, sr in enumerate(success_rates):
        print(f"  {i+1}/5 success rate: {sr * 100:.1f}%")

    # 저장
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "avg_seq_len": float(avg_len),
            "chain_sr": {i + 1: sr for i, sr in enumerate(success_rates)},
            "num_sequences": len(results),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return results


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VLA Calvin 평가 (모델 무관)")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Calvin 데이터셋 경로 (e.g. /temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset)",
    )
    parser.add_argument(
        "--calvin-conf",
        type=str,
        default="/temporal_vla/src/policies/UP-VLA/policy_conf",
        help="Calvin task/annotation config 디렉토리 경로",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8300",
        help="VLA 서버 URL (통일 API)",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="평가 시퀀스 수",
    )
    parser.add_argument(
        "--ep-len",
        type=int,
        default=360,
        help="subtask당 최대 스텝 수",
    )
    parser.add_argument(
        "--act-step",
        type=int,
        default=10,
        help="모델이 한번에 예측하는 action step 수 (기본: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로 (.json)",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=0,
        help="비디오로 기록할 시퀀스 수 (0=비활성화). 성공 4회 미만인 경우만 MP4 저장.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="비디오/PNG 저장 디렉토리 (--num-videos > 0일 때 필수)",
    )
    args = parser.parse_args()

    if args.num_videos > 0 and args.video_dir is None:
        parser.error("--num-videos > 0 이면 --video-dir 를 지정해야 합니다.")

    evaluate(
        dataset_path=Path(args.dataset_path),
        calvin_conf=Path(args.calvin_conf),
        server_url=args.server_url,
        num_sequences=args.num_sequences,
        ep_len=args.ep_len,
        act_step=args.act_step,
        output_path=Path(args.output) if args.output else None,
        num_videos=args.num_videos,
        video_dir=Path(args.video_dir) if args.video_dir else None,
    )


if __name__ == "__main__":
    main()
