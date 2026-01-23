#!/usr/bin/env python3
"""
X-VLA Evaluation Script (Baseline & Subtask & Task-Level Support)

기능:
1. Baseline 평가: Full task instruction, 전체 에피소드 실행.
2. Subtask 평가: Subtask instruction 순차 실행 (Time/Uncertainty 활용 가능).
3. Task-Level Time 평가: Full task instruction + Task-level 시간 예측.

Usage:
    python scripts/eval_subtask_based.py --task open_the_middle_drawer_of_the_cabinet --n_episodes 10
"""

import argparse
import json
from re import T
import time
import sys
import os
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from unittest.mock import patch

import torch
import numpy as np
import cv2
from transformers import AutoTokenizer

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

# LeRobot & Libero Imports
# LIBERO import 시 불필요한 input() 호출 방지
with patch('builtins.input', return_value='N'):
    try:
        from libero.libero import benchmark
        HAS_LIBERO = True
    except ImportError:
        HAS_LIBERO = False
import gymnasium as gym
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.utils import rotate6d_to_axis_angle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.libero import create_libero_envs
from gymnasium.vector import SyncVectorEnv

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# === Constants ===
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@dataclass
class SubtaskConfig:
    max_steps_per_subtask: int = 100
    switch_mode: str = "time_prediction"
    fixed_steps_per_subtask: int = 50
    baseline: bool = False
    eval_subtask: bool = False
    task_level_time: bool = False

@dataclass
class EvalConfig:
    # 1. LeRobot Standard Sections
    env: LiberoEnv = field(default_factory=lambda: LiberoEnv(
        task="libero_goal",
        storage_path="data/libero_goal",
        gym_kwargs={"obs_type": "pixels_agent_pos", "render_mode": "rgb_array"}
    ))
    policy: Optional[PreTrainedConfig] = None
    eval: LeRobotStandardEvalConfig = field(default_factory=LeRobotStandardEvalConfig)
    
    # 2. Custom/Top-level Fields
    subtask: SubtaskConfig = field(default_factory=SubtaskConfig)
    
    # CLI 편의성을 위한 필드
    task_name: str = "open_the_middle_drawer_of_the_cabinet"
    output_dir: str = "outputs/eval"
    save_video: bool = False
    seed: int = 42

    def __post_init__(self):
        # Ensure env config is consistent
        if self.env:
             if self.env.gym_kwargs is None:
                self.env.gym_kwargs = {}
             # obs_type 등 강제 설정
             self.env.gym_kwargs["obs_type"] = "pixels_agent_pos"
             self.env.gym_kwargs["render_mode"] = "rgb_array"
class SubtaskResult:
    name: str
    steps: int
    actual_time: float
    predicted_time: Optional[float] = None
    predicted_var: Optional[float] = None
    completed: bool = False
    time_predictions: List[float] = field(default_factory=list)

@dataclass
class EpisodeResult:
    episode_idx: int
    success: bool
    total_steps: int
    total_time: float
    initial_predicted_time: Optional[float] = None
    subtask_results: List[SubtaskResult] = field(default_factory=list)

class SubtaskEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_env_and_model()
        self._load_subtasks()

    def _setup_env_and_model(self):
        """환경 및 모델 초기화"""
        logger.info(f"🔧 Setting up evaluator for task: {self.config.task_name}")
        
        # 1. Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        
        logger.info(f"🤖 Loading Policy via make_policy from {self.config.policy.pretrained_path if self.config.policy else 'scratch'}")
        
        # Use make_policy factory
        self.policy = make_policy(
            cfg=self.config.policy,
            ds_meta=None,
            env_cfg=self.config.env,
            rename_map={}
        )
        self.policy.eval()
        self.policy.to(self.device)

        # 2. Load Processors
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.config.policy.pretrained_path if self.config.policy else None,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": self.device}},
        )

        # 3. Setup LIBERO Environment
        if not HAS_LIBERO:
            raise RuntimeError("LIBERO library not found.")
            
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.config.env.task]() # Use env.task (e.g. libero_goal)
        # Find task ID
        task_id = next((i for i in range(task_suite.n_tasks) 
                       if task_suite.get_task(i).name == self.config.task_name), None)
        if task_id is None:
            raise ValueError(f"Task {self.config.task_name} not found in {self.config.env.task}")
            
        self.task = task_suite.get_task(task_id)
        self.full_task_description = self.task.language
        logger.info(f"📝 Full Task Description: {self.full_task_description}")

        # Create EnvConfig for make_env (already in self.config.env, just need to update gym_kwargs)
        # Update gym_kwargs with specific task_id
        if self.config.env.gym_kwargs is None:
            self.config.env.gym_kwargs = {}
        self.config.env.gym_kwargs["task_ids"] = [task_id]
        
        # Create Vector Env (LeRobot Style)
        env_map = make_env(
            self.config.env,
            n_envs=1
        )
        self.env = env_map[self.config.env.task][task_id]
        self.env.reset(seed=self.config.seed) 

    def _load_subtasks(self):
        """Subtask 메타데이터 로드"""
        if self.config.subtask.baseline or self.config.subtask.eval_subtask:
            self.subtasks = [self.full_task_description]
            self.mode = "baseline" if self.config.subtask.baseline else "task"
            return

        # Subtask 모드
        self.mode = "subtask"
        script_dir = Path(__file__).parent.parent
        metadata_path = script_dir / f"data/datasets/libero_goal_subtasks/{self.config.task_name}/subtask_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.subtasks = json.load(f)["subtasks"]
            logger.info(f"📋 Loaded {len(self.subtasks)} subtasks")
        else:
            logger.warning(f"⚠️ Subtask metadata not found at {metadata_path}. Fallback to full task.")
            self.subtasks = [self.full_task_description]
            self.mode = "baseline_fallback"

    def _process_obs(self, obs: Dict) -> Dict:
        """환경 관측값을 모델 입력 형식으로 변환"""
        # SyncVectorEnv의 배치 차원(0) 제거 및 Tensor 변환
        def process_img(img):
            
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.ndim == 4: img = img[0] # remove batch
            if img.shape[-1] == 3: img = img.permute(2, 0, 1) # HWC -> CHW
            if img.dtype == torch.uint8: img = img.float() / 255.0
            return img.unsqueeze(0).to(self.device) # Add batch dim back for model

        def process_state(state_dict):
            # Flatten robot state to vector (EEF Pos(3) + Quat(4) + Gripper(2) + Joint Pos(7) + Joint Vel(7))
            
            def get_t(x):
                if isinstance(x, np.ndarray): x = torch.from_numpy(x)
                if x.ndim > 1: x = x[0] # Remove batch
                return x.to(self.device).float()

            eef_pos = get_t(state_dict["eef"]["pos"])
            eef_quat = get_t(state_dict["eef"]["quat"])
            gripper_qpos = get_t(state_dict["gripper"]["qpos"])
            joint_pos = get_t(state_dict["joints"]["pos"])
            joint_vel = get_t(state_dict["joints"]["vel"])

            # Concatenate all state elements
            # Shape: [3+4+2+7+7] = [23]
            state_vec = torch.cat([eef_pos, eef_quat, gripper_qpos, joint_pos, joint_vel], dim=-1)
            return state_vec.unsqueeze(0) # Add batch dim [1, D]

        return {
            "observation.images.image": process_img(obs["pixels"]["image"]),
            "observation.images.image2": process_img(obs["pixels"]["image2"]),
            "observation.state": process_state(obs["robot_state"]),
        }

    def _predict(self, batch: Dict, instruction: str) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """Action 및 Time/Variance 예측"""
        # 1. Add task instruction for preprocessor (it handles tokenization)
        # Check if batch has batch dim, create list of instructions accordingly
        batch_size = batch["observation.images.image"].shape[0]
        batch["task"] = [instruction] * batch_size

        # 2. Preprocess
        batch = self.preprocessor(batch)

        # 3. Tokenize if preprocessor didn't (Safety net)
        if "observation.language.tokens" not in batch:
            tokens = self.tokenizer(
                instruction, return_tensors="pt", max_length=64, 
                padding="max_length", truncation=True
            ).input_ids.to(self.device)
            # Expand to batch size if needed
            if tokens.shape[0] != batch_size:
                tokens = tokens.expand(batch_size, -1)
            batch["observation.language.tokens"] = tokens

        # 4. Model Inference
        predicted_time, predicted_var = None, None
        
        try:
            # Time/Var 예측을 위해 generate_actions 직접 호출 시도
            # (X-VLA 모델이 Time/Var를 반환하도록 수정된 경우)
            with torch.no_grad():
                # generate_actions 호출을 위한 인자 준비
                img_agent = batch["observation.images.image"]
                proprio = batch["observation.state"]
                input_ids = batch["observation.language.tokens"]
                
                # Standard generate_actions inputs
                image_input = img_agent.unsqueeze(1) # [B, 1, C, H, W]
                image_mask = torch.ones(image_input.shape[0], 1, dtype=torch.bool, device=self.device)
                domain_id = torch.zeros(image_input.shape[0], dtype=torch.long, device=self.device)
                steps = getattr(self.policy.config, "num_denoising_steps", 10)

                # 호출
                outputs = self.policy.model.generate_actions(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_id,
                    proprio=proprio,
                    steps=steps
                )
                
                # 결과 언패킹 (Time Aware 모델인지 확인)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    actions, pred_time, pred_var = outputs
                    predicted_time = pred_time.item()
                    predicted_var = np.exp(0.5 * pred_var.item()) # log_var -> std
                    action_raw = actions[0, 0] # [Chunk, Dim] -> [Dim] (첫번째 청크)
                else:
                    # Standard Model (Action only)
                    actions = outputs
                    action_raw = actions[0, 0] if actions.ndim == 3 else actions[0]

        except Exception as e:
            # Fallback: Standard select_action
            # logger.debug(f"Fallback to select_action due to: {e}")
            action_raw = self.policy.select_action(batch)

        # 4. Postprocess
        # Ensure input to postprocessor is [B, D]
        if action_raw.ndim == 1:
            action_raw = action_raw.unsqueeze(0)
            
        action = self.postprocessor(action_raw)
        action = action.cpu().numpy()[0]

        # 5. Format Action (20D -> 7D conversion if needed)
        if action.shape[-1] == 20:
            pos = action[:3]
            rot6d = action[3:9]
            gripper = action[9:10]
            
            axis_angle = rotate6d_to_axis_angle(rot6d)
            gripper = np.where(gripper > 0.0, 1.0, -1.0)
            action = np.concatenate([pos, axis_angle, gripper])

        return action, predicted_time, predicted_var

    def _check_switch_condition(self, steps: int, obs: Dict, prev_gripper: float, pred_time: float) -> bool:
        """Subtask 전환 조건 확인"""
        mode = self.config.subtask.switch_mode
        if mode == "fixed_steps":
            return steps >= self.config.subtask.fixed_steps
        elif mode == "gripper_change":
            if prev_gripper is None: return False
            curr = obs["gripper"][0] if "gripper" in obs else obs["robot_state"]["gripper"]["qpos"][0][0]
            return abs(curr - prev_gripper) > 0.5 and steps >= 10
        elif mode == "time_prediction" and pred_time is not None:
            return pred_time < 0.5 and steps >= 5
        return steps >= self.config.subtask.max_steps_per_subtask

    def evaluate(self):
        results = []
        logger.info(f"🚀 Starting Evaluation: {self.mode.upper()} Mode")
        
        for ep_idx in range(self.config.eval.n_episodes):
            res = self.run_episode(ep_idx)
            results.append(res)
            
        self._save_results(results)
        self._print_summary(results)

    def run_episode(self, ep_idx: int) -> EpisodeResult:
        logger.info(f"🎮 Episode {ep_idx + 1}/{self.config.eval.n_episodes}")
        obs_raw, _ = self.env.reset(seed=self.config.seed + ep_idx)
        
        frames = []
        subtask_results = []
        total_steps = 0
        
        start_time = time.time()
        initial_pred_time = None
        
        # Subtask Loop
        for st_idx, instruction in enumerate(self.subtasks):
            logger.info(f"   📌 Subtask {st_idx+1}: {instruction}")
            st_start_time = time.time()
            st_steps = 0
            st_preds = []
            prev_gripper = None
            if self.mode == "subtask":
                max_steps = self.config.subtask.max_steps_per_subtask
            else:
                max_steps = self.config.env.episode_length
            
            while st_steps < max_steps:
                if total_steps >= self.config.env.episode_length: break
                
                # Prediction
                batch = self._process_obs(obs_raw)
                action, p_time, p_var = self._predict(batch, instruction)
                
                # Record predictions
                if p_time: st_preds.append(p_time)
                if total_steps == 0: initial_pred_time = p_time
                
                # Step Env
                obs_raw, _, terminated, truncated, info = self.env.step(np.expand_dims(action, 0))
                done = bool(terminated[0] or truncated[0])
                success = bool(info.get("is_success", [False])[0])
                
                # Video Frame
                if self.config.save_video:
                    frames.append(obs_raw["pixels"]["image"][0])

                total_steps += 1
                st_steps += 1
                
                # Logging
                if total_steps % 20 == 0:
                     logger.info(f"Step {total_steps}: Act={action[:3].round(2)} T={p_time if p_time else 'N/A'}")

                # Check Completion/Switch
                if done or success: break
                
                # Subtask Switching Logic (only in subtask mode)
                if self.mode == "subtask":
                    # Get gripper for switch logic
                    curr_gripper = obs_raw["robot_state"]["gripper"]["qpos"][0][0]
                    if self._check_switch_condition(st_steps, None, prev_gripper, p_time):
                        logger.info(f"      → Switching subtask after {st_steps} steps")
                        break
                    prev_gripper = curr_gripper
            
            # Subtask Result
            subtask_results.append(SubtaskResult(
                name=instruction,
                steps=st_steps,
                actual_time=time.time() - st_start_time,
                predicted_time=float(np.mean(st_preds)) if st_preds else None,
                completed=bool(success if (st_idx == len(self.subtasks)-1) else True) # Approximation
            ))
            
            if done or success: break
            
        # End Episode
        total_time = time.time() - start_time
        success = bool(info.get("is_success", [False])[0])
        
        if self.config.save_video:
            self._save_video(ep_idx, frames, success)
            
        return EpisodeResult(ep_idx, success, total_steps, total_time, initial_pred_time, subtask_results)

    def _save_video(self, idx, frames, success):
        if not frames: return
        path = Path(self.config.output_dir) / self.config.task_name / "videos"
        path.mkdir(parents=True, exist_ok=True)
        status = "success" if success else "failed"
        
        # Convert frames: RGB -> BGR & Flip
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(str(path / f"ep{idx:03d}_{status}.mp4"), 
                             cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
        for f in frames:
            out.write(cv2.flip(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), 0))
        out.release()

    def _save_results(self, results: List[EpisodeResult]):
        out_path = Path(self.config.output_dir) / self.config.task_name
        out_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": self.config.__dict__,
            "results": [
                {
                    "episode": r.episode_idx,
                    "success": r.success,
                    "total_time": r.total_time,
                    "initial_pred_time": r.initial_predicted_time,
                    "subtasks": [s.__dict__ for s in r.subtask_results]
                } for r in results
            ],
            "summary": {
                "success_rate": float(np.mean([r.success for r in results])),
                "avg_time": float(np.mean([r.total_time for r in results])),
            }
        }
        
        fname = f"results_{self.mode}.json"
        with open(out_path / fname, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Results saved to {out_path / fname}")

    def _print_summary(self, results):
        success_rate = np.mean([r.success for r in results])
        logger.info("\n" + "="*50)
        logger.info(f"📊 Summary ({self.mode})")
        logger.info(f"   Success Rate: {success_rate*100:.1f}%")
        logger.info(f"   Avg Time: {np.mean([r.total_time for r in results]):.2f}s")
        logger.info("="*50)

@parser.wrap()
def main(config: EvalConfig):
    evaluator = SubtaskEvaluator(config)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
