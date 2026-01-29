#!/usr/bin/env python3
"""
HuggingFace LIBERO 데이터셋을 사용하여 lerobot/libero 환경에서 action을 replay하는 스크립트.
데이터셋 변환 검증 및 단일 에피소드 디버깅용입니다.
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from datasets import load_from_disk
from PIL import Image
import imageio
from lerobot.envs.libero import LiberoEnv
from dotenv import load_dotenv
from libero.libero import benchmark

load_dotenv()

# Force logging configuration to override any previous config (e.g. from libraries)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def explore_dataset(dataset_path: str):
    """다운로드한 데이터셋 구조 탐색"""
    logger.info(f"📂 데이터셋 로드 중: {dataset_path}")
    ds = load_from_disk(dataset_path)
    
    logger.info(f"\n📊 데이터셋 구조:")
    logger.info(f"  - Split: {ds}")
    
    # train split 확인
    if hasattr(ds, 'keys'):
        for split in ds.keys():
            logger.info(f"\n  [{split}] 샘플 수: {len(ds[split])}")
            logger.info(f"  컬럼: {ds[split].column_names}")
            
            # 첫 번째 샘플 확인
            if len(ds[split]) > 0:
                sample = ds[split][0]
                logger.info(f"\n  첫 번째 샘플 키:")
                for key, value in sample.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        logger.info(f"    - {key}: shape={value.shape}, dtype={type(value)}")
                    elif isinstance(value, dict):
                        logger.info(f"    - {key}: dict with keys={list(value.keys())}")
                    elif isinstance(value, list):
                        logger.info(f"    - {key}: list, len={len(value)}")
                    else:
                        logger.info(f"    - {key}: {type(value).__name__} = {str(value)[:100]}")
    else:
        logger.info(f"  샘플 수: {len(ds)}")
        logger.info(f"  컬럼: {ds.column_names}")
        
        if len(ds) > 0:
            sample = ds[0]
            logger.info(f"\n  첫 번째 샘플 키:")
            for key, value in sample.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    logger.info(f"    - {key}: shape={value.shape}, dtype={type(value)}")
                elif isinstance(value, dict):
                    logger.info(f"    - {key}: dict")
                elif isinstance(value, list):
                    logger.info(f"    - {key}: list, len={len(value)}")
                elif hasattr(value, 'shape'):
                    logger.info(f"    - {key}: shape={value.shape}")
                else:
                    logger.info(f"    - {key}: {type(value).__name__} = {str(value)[:100]}")
    
    return ds


def get_task_language(dataset_path: str, task_index: int) -> str:
    """데이터셋의 task_index에 해당하는 language instruction 반환"""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    try:
        tasks_path = hf_hub_download(
            repo_id='HuggingFaceVLA/libero', 
            filename='meta/tasks.parquet', 
            repo_type='dataset'
        )
        tasks_df = pd.read_parquet(tasks_path)
        
        for task_name, row in tasks_df.iterrows():
            if row['task_index'] == task_index:
                return task_name
    except Exception as e:
        logger.warning(f"Task info 로드 실패: {e}")
        return "unknown"
    
    raise ValueError(f"task_index {task_index} not found in dataset")


def find_libero_task_by_language(language: str, target_suite: str = None):
    """language instruction으로 libero task를 찾아 (suite_name, task_id, suite, task) 반환"""
    bench = benchmark.get_benchmark_dict()
    
    if target_suite:
        suites_to_search = [target_suite]
    else:
        suites_to_search = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
    
    for suite_name in suites_to_search:
        if suite_name not in bench:
            continue
        suite = bench[suite_name]()
        for task_id in range(len(suite.tasks)):
            task = suite.get_task(task_id)
            if task.language.lower().strip() == language.lower().strip():
                return suite_name, task_id, suite, task
    
    return None, None, None, None


def replay_episode(
    dataset_path: str,
    task_index: int = 0,
    episode_id: int = 0,
    output_dir: str = "outputs/replay_videos",
    save_video: bool = True,
    control_mode: str = "absolute",
    task_suite: str = None,
    dataset_type: str = 'converted'
):
    """
    데이터셋에서 특정 에피소드의 action을 LIBERO 환경에서 replay
    
    Args:
        dataset_path: HuggingFace 데이터셋 경로
        task_index: 데이터셋의 task_index
        episode_id: replay할 에피소드 순번 (0부터 시작)
        output_dir: 비디오 저장 경로
        save_video: 비디오 저장 여부
        control_mode: 'absolute' 또는 'relative'
        task_suite: LIBERO task suite 이름 (None이면 자동 검색)
        dataset_type: 'original' or 'converted'
    """
    logger.info(f"\n🎬 데이터셋 Action Replay 시작")
    logger.info(f"  - Dataset Path: {dataset_path}")
    logger.info(f"  - Dataset Type: {dataset_type}")
    logger.info(f"  - Dataset Task Index: {task_index}")
    logger.info(f"  - Episode ID (Index): {episode_id}")
    logger.info(f"  - Control Mode: {control_mode}")
    
    # 1. Task 정보 및 Suite 찾기
    task_language = "unknown"
    suite_name = None
    libero_task_id = None
    suite = None
    task = None

    if dataset_type == 'converted':
        # Converted 데이터셋은 task_index가 이미 Libero Task ID와 동일하다고 가정
        if task_suite is None:
            task_suite = "libero_goal" # Default for converted
        
        bench = benchmark.get_benchmark_dict()
        if task_suite in bench:
            suite = bench[task_suite]()
            if task_index < len(suite.tasks):
                task = suite.get_task(task_index)
                task_language = task.language
                suite_name = task_suite
                libero_task_id = task_index
                logger.info(f"  - Converted Mode: Mapped directly to {suite_name} Task {libero_task_id}")
    
    else: # original
        # Original 데이터셋은 Language 기반 매핑 필요
        
        # 1-1. Suite가 지정된 경우 거기서 먼저 찾아봄
        if task_suite:
            bench = benchmark.get_benchmark_dict()
            if task_suite in bench:
                suite = bench[task_suite]()
                # task_index가 suite index와 일치하는지 확인 (우연히 맞을 수도 있음)
                if task_index < len(suite.tasks):
                    # 일단 가져와봄 (검증용)
                    temp_task = suite.get_task(task_index)
                    # 하지만 original 데이터셋의 task_index는 suite index와 다를 수 있음.
                    # Language를 먼저 가져와야 함.
                    pass

        # 1-2. 데이터셋에서 Language 가져오기
        try:
            task_language = get_task_language(dataset_path, task_index)
            logger.info(f"  - Task Language (from Dataset): {task_language}")
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to get task language: {e}")

        # 1-3. Language로 Libero Task 찾기
        suite_name, libero_task_id, suite, task = find_libero_task_by_language(task_language, target_suite=task_suite)

    if suite_name:
        logger.info(f"  - Libero Suite: {suite_name}")
        logger.info(f"  - Libero Task ID: {libero_task_id}")
        logger.info(f"  - Task Name: {task.name}")
    else:
        logger.error(f"❌ Failed to identify Libero task for language: {task_language}")
        return {'success': False, 'error': 'Task identification failed'}
    
    # 2. 데이터셋 로드
    ds = load_from_disk(dataset_path)
    if hasattr(ds, 'keys'):
        ds = ds['train']
    
    logger.info(f"\n📊 데이터셋 로드 완료: {len(ds)} 샘플")
    
    # 3. 데이터 필터링
    task_data = ds.filter(lambda x: x['task_index'] == task_index)
    logger.info(f"  - Task {task_index} 데이터: {len(task_data)} 샘플")
    
    unique_episodes = sorted(list(set(task_data['episode_index'])))
    logger.info(f"  - 고유 에피소드 수: {len(unique_episodes)}")
    
    if episode_id >= len(unique_episodes):
        logger.warning(f"  - episode_id {episode_id} >= {len(unique_episodes)}, 첫 번째 에피소드 사용")
        episode_id = 0
    
    target_episode_idx = unique_episodes[episode_id]
    episode_data = task_data.filter(lambda x: x['episode_index'] == target_episode_idx)
    logger.info(f"  - Target Episode Index: {target_episode_idx} (Sequence {episode_id})")
    logger.info(f"  - Frames: {len(episode_data)}")
    
    # 4. Action 추출
    actions = []
    for i in range(len(episode_data)):
        action = episode_data[i]['action']
        if isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        actions.append(action)
    
    # 5. LIBERO 환경 생성
    logger.info(f"\n🎮 LIBERO 환경 생성 중...")
    
    # Init State Index 계산
    if dataset_type == 'original':
        init_state_idx = target_episode_idx % 50
    else:
        init_state_idx = target_episode_idx # Converted는 이미 처리됨
        
    logger.info(f"  - Init State Index: {init_state_idx}")

    env = LiberoEnv(
        task_suite=suite, # 객체 전달
        task_id=libero_task_id, # 매핑된 ID 전달
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        init_states=True,
        episode_index=init_state_idx,
        control_mode=control_mode,
    )
    
    if control_mode == "absolute":
        logger.info(f"🔧 Controller set to ABSOLUTE action mode")
        try:
            controller = env._env.robots[0].controller
            controller.use_delta = False
        except Exception as e:
            logger.error(f"❌ Failed to set absolute mode: {e}")
    
    logger.info(f"  - 환경 생성 완료!")
    
    # 6. Replay 실행
    logger.info(f"\n🎬 Action Replay 실행 중...")
    
    obs, info = env.reset()
    frames = []
    frames.append(env.render())
    
    success = False
    for step, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        
        if info.get('is_success', False):
            success = True
            logger.info(f"  ✅ SUCCESS at step {step+1}!")
            break
        
        if terminated or truncated:
            logger.info(f"  Episode ended at step {step+1}")
            break
        
        if (step + 1) % 50 == 0:
            logger.info(f"  Step {step+1}/{len(actions)}")
    
    if not success:
        logger.info(f"  ❌ Episode ended without success after {len(actions)} steps")
    
    env.close()
    
    # 7. 비디오 저장
    if save_video and frames:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_path = output_path / f"replay_task{task_index}_ep{episode_id}.mp4"
        logger.info(f"\n💾 비디오 저장 중: {video_path}")
        
        imageio.mimsave(str(video_path), frames, fps=10, quality=8)
        
        # 첫 프레임 저장 (비교용)
        first_frame_path = output_path / f"replay_task{task_index}_ep{episode_id}_first_frame.png"
        Image.fromarray(frames[0]).save(first_frame_path)
        
        return {
            'video_path': str(video_path),
            'success': success,
            'task_name': task.name
        }
    
    return {'success': success}


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO 환경에서 HuggingFace 데이터셋 Replay (단일 에피소드 디버깅용)"
    )
    parser.add_argument("--dataset_path", type=str, default="data/libero_hf_abs/libero_goal", help="데이터셋 경로")
    parser.add_argument("--task_index", type=int, default=0, help="데이터셋의 task_index")
    parser.add_argument("--episode_id", type=int, default=0, help="Replay할 에피소드 순번 (0부터 시작)")
    parser.add_argument("--explore_only", action="store_true", help="데이터셋 구조만 탐색")
    parser.add_argument("--output_dir", type=str, default="outputs/replay_videos", help="비디오 저장 경로")
    parser.add_argument("--control_mode", type=str, default="absolute", choices=["absolute", "relative"], help="제어 모드")
    parser.add_argument("--task_suite", type=str, default=None, help="LIBERO task suite (옵션)")
    parser.add_argument("--dataset_type", type=str, default="converted", choices=['original', 'converted'], help="데이터셋 타입")
    
    args = parser.parse_args()
    
    if args.explore_only:
        explore_dataset(args.dataset_path)
    else:
        replay_episode(
            dataset_path=args.dataset_path,
            task_index=args.task_index,
            episode_id=args.episode_id,
            output_dir=args.output_dir,
            save_video=True,
            control_mode=args.control_mode,
            task_suite=args.task_suite,
            dataset_type=args.dataset_type
        )


if __name__ == "__main__":
    main()
