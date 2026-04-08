"""GR00T rollout trajectory 수집 스크립트.

rollout_policy.py의 환경/모델 셋업을 재사용하되,
매 스텝의 action과 state를 기록하여 .npz로 저장한다.

사용법 (groot 컨테이너 내부):
    MUJOCO_GL=egl python /temporal_vla/scripts/collect_groot_trajectories.py \
        --model_path /temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B \
        --env_name robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env \
        --n_episodes 20 \
        --save_dir /temporal_vla/outputs/trajectories
"""

import argparse
import os
import time
import uuid
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")


def collect_trajectories(
    model_path: str,
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    n_action_steps: int,
    save_dir: str,
):
    # 지연 import (gr00t PYTHONPATH 필요)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
    from gr00t.eval.rollout_policy import (
        get_robocasa_env_fn,
        create_gr00t_sim_policy,
        WrapperConfigs,
        VideoConfig,
        MultiStepConfig,
        create_eval_env,
    )
    import gymnasium as gym

    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    # 모델 로딩
    print(f"Loading model from {model_path}...")
    policy = create_gr00t_sim_policy(model_path, embodiment_tag)

    # 비디오 저장 설정
    model_id = Path(model_path).name
    video_dir = f"/tmp/traj_videos_{model_id}_{uuid.uuid4().hex[:8]}"
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=video_dir, max_episode_steps=max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
        ),
    )

    env = create_eval_env(env_name, env_idx=0, total_n_envs=1, wrapper_configs=wrapper_configs)

    # 태스크 이름 추출
    task_short = env_name.split("/")[-1].replace("_PandaOmron_Env", "")
    save_path = Path(save_dir) / task_short
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {n_episodes} episodes for {env_name}")
    print(f"Saving to {save_path}")
    print(f"Videos to {video_dir}")

    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        policy.reset()

        actions_list = []
        states_list = []
        rewards_list = []
        success_flags = []

        for step in range(max_episode_steps):
            # state 기록 (가능한 키 수집)
            state_dict = {}
            for k, v in obs.items():
                if k.startswith("state.") or k.startswith("hand.") or k.startswith("body."):
                    state_dict[k] = np.array(v).flatten()

            # action 기록
            action, _ = policy.get_action(obs)
            action_flat = []
            for k in sorted(action.keys()):
                action_flat.append(np.array(action[k]).flatten())
            action_flat = np.concatenate(action_flat) if action_flat else np.array([])

            if state_dict:
                state_flat = np.concatenate([state_dict[k] for k in sorted(state_dict.keys())])
            else:
                state_flat = np.array([])

            actions_list.append(action_flat)
            states_list.append(state_flat)

            obs, reward, terminated, truncated, info = env.step(action)
            rewards_list.append(float(reward))

            success = info.get("success", False)
            if isinstance(success, (list, np.ndarray)):
                success = bool(np.any(success))
            success_flags.append(bool(success))

            if terminated or truncated:
                break

        # 에피소드 데이터 저장
        ep_data = {
            "actions": np.array(actions_list),
            "states": np.array(states_list) if states_list and states_list[0].size > 0 else np.array([]),
            "rewards": np.array(rewards_list),
            "success_flags": np.array(success_flags),
            "final_success": bool(any(success_flags)),
            "n_steps": len(actions_list),
            "env_name": env_name,
        }

        ep_file = save_path / f"ep_{ep_idx:03d}.npz"
        np.savez_compressed(ep_file, **ep_data)

        status = "SUCCESS" if ep_data["final_success"] else "FAIL"
        print(f"  Episode {ep_idx:3d}: {status} ({ep_data['n_steps']} steps) -> {ep_file}")

    env.close()
    print(f"\nDone. {n_episodes} episodes saved to {save_path}")
    print(f"Videos saved to {video_dir}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect GR00T rollout trajectories")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=720)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="/temporal_vla/outputs/trajectories")
    args = parser.parse_args()

    collect_trajectories(
        model_path=args.model_path,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
        save_dir=args.save_dir,
    )
