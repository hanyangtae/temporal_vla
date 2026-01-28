#!/usr/bin/env python3
"""
HuggingFace LIBERO 데이터셋을 사용하여 lerobot/libero 환경에서 action을 replay하는 스크립트
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from datasets import load_from_disk
from PIL import Image
import imageio


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    
    tasks_path = hf_hub_download(
        repo_id='HuggingFaceVLA/libero', 
        filename='meta/tasks.parquet', 
        repo_type='dataset'
    )
    tasks_df = pd.read_parquet(tasks_path)
    
    for task_name, row in tasks_df.iterrows():
        if row['task_index'] == task_index:
            return task_name
    
    raise ValueError(f"task_index {task_index} not found in dataset")


def find_libero_task_by_language(language: str, target_suite: str = None):
    """language instruction으로 libero task를 찾아 (suite_name, task_id) 반환
    
    Args:
        language: task의 language instruction
        target_suite: 검색할 suite 이름 (None이면 모든 suite 검색)
    """
    from libero.libero import benchmark
    
    bench = benchmark.get_benchmark_dict()
    
    # 검색할 suite 목록
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
            # language가 일치하면 반환
            if task.language.lower().strip() == language.lower().strip():
                return suite_name, task_id, suite, task
    
    return None, None, None, None


def replay_episode(
    dataset_path: str,
    task_index: int = 0,
    episode_id: int = 0,
    output_dir: str = "outputs/replay_videos",
    save_video: bool = True,
    use_absolute_action: bool = False,
):
    """
    데이터셋에서 특정 에피소드의 action을 LIBERO 환경에서 replay
    
    Args:
        dataset_path: HuggingFace 데이터셋 경로
        task_index: 데이터셋의 task_index (0-39)
        episode_id: replay할 에피소드 ID
        output_dir: 비디오 저장 경로
        save_video: 비디오 저장 여부
        use_absolute_action: 절대 좌표 모드 사용 여부 (controller.use_delta=False)
    """
    from libero.libero import benchmark
    
    logger.info(f"\n🎬 데이터셋 Action Replay 시작")
    logger.info(f"  - Dataset Task Index: {task_index}")
    logger.info(f"  - Episode ID: {episode_id}")
    
    # 데이터셋의 task language 가져오기
    task_language = get_task_language(dataset_path, task_index)
    logger.info(f"  - Task Language: {task_language}")
    
    # language로 libero task 찾기
    suite_name, libero_task_id, suite, task = find_libero_task_by_language(task_language)
    
    if suite_name is None:
        logger.error(f"❌ libero에서 '{task_language}' task를 찾을 수 없습니다!")
        logger.info("데이터셋 action만 사용하여 시뮬레이션 없이 진행합니다.")
        suite_name = "unknown"
        libero_task_id = -1
    else:
        logger.info(f"  - Libero Suite: {suite_name}")
        logger.info(f"  - Libero Task ID: {libero_task_id}")
    
    # 데이터셋 로드
    ds = load_from_disk(dataset_path)
    if hasattr(ds, 'keys'):
        ds = ds['train']
    
    logger.info(f"\n📊 데이터셋 로드 완료: {len(ds)} 샘플")
    
    # task_index로 필터링
    if 'task_index' in ds.column_names:
        task_data = ds.filter(lambda x: x['task_index'] == task_index)
        logger.info(f"  - Task {task_index} 데이터: {len(task_data)} 샘플")
    else:
        task_data = ds
        logger.info("  - task_index 컬럼 없음, 전체 데이터 사용")
    
    # episode_index로 필터링
    if 'episode_index' in task_data.column_names:
        unique_episodes = sorted(set(task_data['episode_index']))
        logger.info(f"  - 고유 에피소드 수: {len(unique_episodes)}")
        
        if episode_id >= len(unique_episodes):
            logger.warning(f"  - episode_id {episode_id} >= {len(unique_episodes)}, 첫 번째 에피소드 사용")
            episode_id = 0
        
        target_episode = unique_episodes[episode_id]
        episode_data = task_data.filter(lambda x: x['episode_index'] == target_episode)
        logger.info(f"  - Episode {target_episode} 선택: {len(episode_data)} 프레임")
    else:
        episode_data = task_data
        logger.info("  - episode_index 컬럼 없음")
    
    # Action 시퀀스 추출
    actions = []
    for i in range(len(episode_data)):
        action = episode_data[i]['action']
        if isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        actions.append(action)
    
    logger.info(f"  - Action 시퀀스 길이: {len(actions)}")
    logger.info(f"  - Action shape: {actions[0].shape}")
    
    # libero task를 찾지 못한 경우 데이터셋 이미지만 저장
    if suite is None:
        logger.info("\n⚠️ Libero 환경 없이 데이터셋 이미지만 저장합니다.")
        if save_video:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 데이터셋 이미지로 비디오 생성
            frames = []
            for i in range(len(episode_data)):
                img = episode_data[i]['observation.images.image']
                if isinstance(img, Image.Image):
                    frames.append(np.array(img))
            
            video_path = output_path / f"dataset_task{task_index}_ep{episode_id}.mp4"
            imageio.mimsave(str(video_path), frames, fps=10, quality=8)
            logger.info(f"  - 데이터셋 비디오 저장: {video_path}")
            
            return {'video_path': str(video_path), 'n_frames': len(frames), 'success': False}
        return {'n_frames': len(actions), 'success': False}
    
    logger.info(f"\n📌 Task 정보:")
    logger.info(f"  - Task Name: {task.name}")
    logger.info(f"  - Language: {task.language}")
    
    # LIBERO 환경 생성
    logger.info(f"\n🎮 LIBERO 환경 생성 중...")
    
    from lerobot.envs.libero import LiberoEnv
    
    env = LiberoEnv(
        task_suite=suite,
        task_id=libero_task_id,
        task_suite_name=suite_name,
        obs_type="pixels",
        render_mode="rgb_array",
        init_states=True,
        episode_index=episode_id,
        control_mode="absolute" if use_absolute_action else "relative",
    )
    
    if use_absolute_action:
        logger.info(f"🔧 Controller set to ABSOLUTE action mode (use_delta=False)")
        try:
            # Access internal robosuite controller
            # LiberoEnv -> OffScreenRenderEnv -> MujocoEnv -> robots -> controller
            controller = env._env.robots[0].controller
            controller.use_delta = False
        except Exception as e:
            logger.error(f"❌ Failed to set absolute mode: {e}")
    
    logger.info(f"  - 환경 생성 완료!")
    
    # Replay 실행
    logger.info(f"\n🎬 Action Replay 실행 중...")
    
    obs, info = env.reset()
    frames = []
    
    # 초기 프레임 저장
    frame = env.render()
    frames.append(frame)
    
    success = False
    for step, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 프레임 저장
        frame = env.render()
        frames.append(frame)
        
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
    
    # 비디오 저장
    if save_video and frames:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_path = output_path / f"replay_task{task_index}_ep{episode_id}.mp4"
        logger.info(f"\n💾 비디오 저장 중: {video_path}")
        
        # imageio로 저장
        imageio.mimsave(
            str(video_path),
            frames,
            fps=10,
            quality=8,
        )
        
        logger.info(f"  - 프레임 수: {len(frames)}")
        logger.info(f"  - 저장 완료: {video_path}")
        
        # 데이터셋 이미지와 비교용으로 첫 프레임 저장
        first_frame_path = output_path / f"replay_task{task_index}_ep{episode_id}_first_frame.png"
        Image.fromarray(frames[0]).save(first_frame_path)
        logger.info(f"  - 첫 프레임 저장: {first_frame_path}")
        
        # 데이터셋의 첫 이미지도 저장
        ds_first_image = episode_data[0]['observation.images.image']
        if isinstance(ds_first_image, Image.Image):
            ds_image_path = output_path / f"dataset_task{task_index}_ep{episode_id}_first_frame.png"
            ds_first_image.save(ds_image_path)
            logger.info(f"  - 데이터셋 첫 프레임 저장: {ds_image_path}")
        
        return {
            'video_path': str(video_path),
            'n_frames': len(frames),
            'success': success,
            'task_name': task.name,
            'task_language': task_language,
            'libero_suite': suite_name,
            'libero_task_id': libero_task_id,
        }
    
    return {
        'n_frames': len(frames),
        'success': success,
        'task_name': task.name,
        'task_language': task_language,
        'libero_suite': suite_name,
        'libero_task_id': libero_task_id,
    }


def deploy_on_libero(
    dataset_path: str,
    task_suite: str = "libero_goal",
    task_id: int = 0,
    n_episodes: int = 5,
    render: bool = False,
    device: str = "cuda",
):
    """
    LIBERO 환경에서 데이터셋 기반으로 deploy (랜덤 액션 테스트용)
    """
    from libero.libero import benchmark
    
    logger.info(f"\n🚀 LIBERO 환경 Deploy 시작")
    logger.info(f"  - Task Suite: {task_suite}")
    logger.info(f"  - Task ID: {task_id}")
    logger.info(f"  - Episodes: {n_episodes}")
    
    # 데이터셋 로드
    ds = load_from_disk(dataset_path)
    if hasattr(ds, 'keys'):
        ds = ds['train']
    
    # Task Suite 로드
    bench = benchmark.get_benchmark_dict()
    if task_suite not in bench:
        available = list(bench.keys())
        raise ValueError(f"Unknown task suite: {task_suite}. Available: {available}")
    
    suite = bench[task_suite]()
    n_tasks = len(suite.tasks)
    logger.info(f"  - Suite에 {n_tasks}개 task 존재")
    
    if task_id >= n_tasks:
        raise ValueError(f"task_id {task_id} >= n_tasks {n_tasks}")
    
    task = suite.get_task(task_id)
    logger.info(f"\n📌 선택된 Task:")
    logger.info(f"  - Task Name: {task.name}")
    logger.info(f"  - Language: {task.language}")
    
    # 데이터셋에서 해당 task의 데이터 필터링
    logger.info(f"\n📊 데이터셋에서 task 필터링 중...")
    
    # 컬럼 이름 확인
    columns = ds.column_names
    logger.info(f"  - 컬럼: {columns}")
    
    logger.info("  - task 컬럼을 찾을 수 없음. 전체 데이터 사용")
    task_data = ds
    
    # LIBERO 환경 생성
    logger.info(f"\n🎮 LIBERO 환경 생성 중...")
    
    from lerobot.envs.libero import LiberoEnv, get_task_init_states
    
    init_states = get_task_init_states(suite, task_id)
    logger.info(f"  - Init states: {init_states.shape}")
    
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=task_suite,
        obs_type="pixels",
        render_mode="rgb_array",
        init_states=True,
    )
    
    logger.info(f"  - 환경 생성 완료!")
    logger.info(f"  - Observation space: {env.observation_space}")
    logger.info(f"  - Action space: {env.action_space}")
    
    # 에피소드 실행 (랜덤 액션)
    logger.info(f"\n🎬 에피소드 실행 (랜덤 액션 테스트)...")
    
    success_count = 0
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 300:
            # 랜덤 액션
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if info.get('is_success', False):
                success_count += 1
                logger.info(f"  ✅ Episode {ep+1}: SUCCESS at step {steps}")
                break
        
        if not info.get('is_success', False):
            logger.info(f"  ❌ Episode {ep+1}: FAILED after {steps} steps")
    
    logger.info(f"\n📊 결과:")
    logger.info(f"  - Success Rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    
    env.close()
    logger.info("\n✅ Deploy 완료!")
    
    return {
        'success_count': success_count,
        'n_episodes': n_episodes,
        'task_name': task.name,
        'task_language': task.language,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO 환경에서 HuggingFace 데이터셋으로 Deploy/Replay"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/libero_hf/dataset",
        help="HuggingFace 데이터셋 경로"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="데이터셋의 task_index (0-39). libero 환경은 language로 자동 매칭"
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
        help="LIBERO task suite (random action 모드용)"
    )
    parser.add_argument(
        "--libero_task_id",
        type=int,
        default=0,
        help="Libero suite 내 task ID (random action 모드용)"
    )
    parser.add_argument(
        "--episode_id",
        type=int,
        default=0,
        help="Replay할 에피소드 ID (0부터 시작)"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="실행할 에피소드 수 (random action 모드)"
    )
    parser.add_argument(
        "--explore_only",
        action="store_true",
        help="데이터셋 구조만 탐색"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="데이터셋 action을 그대로 replay"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/replay_videos",
        help="비디오 저장 경로"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    parser.add_argument(
        "--use_absolute_action",
        action="store_true",
        help="Treat actions as absolute coordinates (sets controller.use_delta=False)"
    )
    
    args = parser.parse_args()
    
    if args.explore_only:
        explore_dataset(args.dataset_path)
    elif args.replay:
        replay_episode(
            dataset_path=args.dataset_path,
            task_index=args.task_index,
            episode_id=args.episode_id,
            output_dir=args.output_dir,
            save_video=True,
            use_absolute_action=args.use_absolute_action,
        )
    else:
        deploy_on_libero(
            dataset_path=args.dataset_path,
            task_suite=args.task_suite,
            task_id=args.libero_task_id,
            n_episodes=args.n_episodes,
            device=args.device,
        )


if __name__ == "__main__":
    main()
