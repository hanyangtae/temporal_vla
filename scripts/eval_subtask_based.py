#!/usr/bin/env python3
"""
X-VLA Evaluation Script (Baseline & Subtask & Task-Level 모드 지원)

== Baseline 모드 (--baseline) ==
- Full task instruction 사용
- Subtask 전환 없이 전체 에피소드 실행
- 기존 X-VLA 모델 평가에 적합

== Subtask 모드 (기본) ==
- Subtask instruction 순차 실행
- Fine-tuned 모델 평가에 적합
- Subtask 단위 시간 예측 평가 포함

== Task-Level 모드 (--task_level_time) ==
- Full task instruction 사용하면서 Task-level 시간 예측 평가
- time_to_go_full로 학습된 모델 평가에 적합

Usage:
    # Baseline 평가 (full task instruction)
    python scripts/eval_subtask_based.py --task open_the_middle_drawer_of_the_cabinet --baseline --n_episodes 10
    
    # Subtask 기반 평가 (fine-tuned 모델)
    python scripts/eval_subtask_based.py --task open_the_middle_drawer_of_the_cabinet --n_episodes 10
    
    # Task-Level 시간 예측 평가
    python scripts/eval_subtask_based.py --task open_the_middle_drawer_of_the_cabinet --task_level_time --n_episodes 10
"""

import argparse
import json
import time
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer

# === LIBERO import with input() mocking ===
# LIBERO 초기화 시 input() 호출을 막기 위한 처리
from unittest.mock import patch

# 환경 변수로 LIBERO 데이터 경로 설정 (input 호출 방지)
if "LIBERO_DATA_DIR" not in os.environ:
    os.environ["LIBERO_DATA_DIR"] = "/workspace/vla_tset/data/libero"
    # 디렉토리가 없으면 생성
    os.makedirs(os.environ["LIBERO_DATA_DIR"], exist_ok=True)

# input()을 mocking하여 'N' 반환 (기본 경로 사용)
with patch('builtins.input', return_value='N'):
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        HAS_LIBERO = True
    except ImportError as e:
        HAS_LIBERO = False
        print(f"⚠️ LIBERO not available: {e}")

# LeRobot X-VLA imports
try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    HAS_XVLA = True
except ImportError:
    HAS_XVLA = False
    print("⚠️ X-VLA not available")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
# Force flush on every log
for handler in logging.root.handlers:
    handler.flush = lambda: None
    
logger = logging.getLogger(__name__)

# Force stdout/stderr flush
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ImageNet stats for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class SubtaskResult:
    """단일 Subtask 실행 결과"""
    subtask_name: str
    subtask_idx: int
    start_step: int
    end_step: int
    steps_taken: int
    predicted_time: Optional[float] = None  # 예측된 평균 시간
    predicted_var: Optional[float] = None   # 예측된 평균 불확실성
    actual_time: Optional[float] = None     # 실제 소요 시간 (초)
    completed: bool = False                  # Subtask 완료 여부
    time_predictions: List[float] = field(default_factory=list)  # 모든 예측값


@dataclass
class InstructionResult:
    """단일 Instruction 실행 결과"""
    task_name: str
    episode_idx: int
    success: bool
    total_steps: int
    total_time: float                    # 총 소요 시간 (초)
    initial_predicted_time: Optional[float] = None  # 첫 step의 시간 예측 (Task-level용)
    initial_predicted_var: Optional[float] = None   # 첫 step의 불확실성 예측
    subtask_results: List[SubtaskResult] = field(default_factory=list)


@dataclass
class EvalConfig:
    """평가 설정"""
    task_suite: str = "libero_goal"
    task_name: str = "open_the_middle_drawer_of_the_cabinet"
    model_path: str = "lerobot/xvla-libero"
    n_episodes: int = 10
    max_steps_per_episode: int = 300
    max_steps_per_subtask: int = 100     # Subtask당 최대 step
    subtask_switch_mode: str = "fixed_steps"  # "fixed_steps", "gripper_change", "time_prediction"
    fixed_steps_per_subtask: int = 50    # fixed_steps 모드에서 사용
    output_dir: str = "outputs/eval"
    save_video: bool = False
    seed: int = 42
    baseline: bool = False               # Full task instruction, 시간 예측 없음
    task_level_time: bool = False        # Full task instruction + Task-level 시간 예측 평가


class SubtaskBasedEvaluator:
    """Subtask 기반 평가기"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 결과 저장
        self.results: List[InstructionResult] = []
        
        # 모델 및 환경 초기화
        self._setup()
    
    def _setup(self):
        """모델과 환경 초기화"""
        logger.info("🔧 Setting up evaluator...")
        
        # Tokenizer 로드
        logger.info("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        
        # X-VLA 모델 로드
        logger.info(f"🤖 Loading X-VLA from {self.config.model_path}...")
        self.policy = XVLAPolicy.from_pretrained(self.config.model_path)
        self.policy.eval()
        if self.device == "cuda":
            self.policy = self.policy.cuda()
        logger.info(f"   Using device: {self.device}")
        
        # Subtask 메타데이터 로드 (baseline/task_level 모드가 아닐 때만)
        if not self.config.baseline and not self.config.task_level_time:
            self._load_subtask_metadata()
        else:
            # Baseline/Task-level 모드에서는 subtask 없음
            self.subtasks = None
            self.metadata = None
        
        # LIBERO 환경 설정
        self._setup_libero_env()
    
    def _load_subtask_metadata(self):
        """Subtask 메타데이터 로드"""
        # 스크립트 위치 기준으로 상대 경로 계산
        script_dir = Path(__file__).parent.parent  # vla_tset 루트
        subtask_dir = script_dir / f"data/datasets/libero_goal_subtasks/{self.config.task_name}"
        metadata_file = subtask_dir / "subtask_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
            self.subtasks = self.metadata["subtasks"]
            logger.info(f"📋 Loaded {len(self.subtasks)} subtasks:")
            for i, st in enumerate(self.subtasks):
                logger.info(f"   {i+1}. {st}")
        else:
            logger.warning(f"⚠️ Subtask metadata not found: {metadata_file}")
            logger.warning(f"   Falling back to full task instruction (like baseline mode)")
            # subtask 없으면 full task로 진행
            self.subtasks = None
            self.metadata = None
    
    def _get_task_description(self) -> str:
        """Task description 가져오기"""
        return self.config.task_name.replace("_", " ")
    
    def _setup_libero_env(self):
        """LIBERO 환경 설정"""
        if not HAS_LIBERO:
            raise RuntimeError("LIBERO is required for evaluation")
        
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.config.task_suite]()
        
        # Task 찾기
        task_id = None
        for i in range(task_suite.n_tasks):
            task = task_suite.get_task(i)
            if task.name == self.config.task_name:
                task_id = i
                break
        
        if task_id is None:
            raise ValueError(f"Task '{self.config.task_name}' not found in {self.config.task_suite}")
        
        self.task = task_suite.get_task(task_id)
        self.task_description = self.task.language
        
        logger.info(f"📋 Task: {self.task.name}")
        logger.info(f"📝 Description: {self.task_description}")
        
        # BDDL 파일 전체 경로 구성
        import libero
        libero_path = Path(libero.__file__).parent
        bddl_dir = libero_path / "libero" / "bddl_files" / self.config.task_suite
        bddl_file = bddl_dir / f"{self.config.task_name}.bddl"
        
        if not bddl_file.exists():
            raise FileNotFoundError(f"BDDL file not found: {bddl_file}")
        
        logger.info(f"📄 BDDL file: {bddl_file}")
        
        # 환경 생성
        env_args = {
            "bddl_file_name": str(bddl_file),
            "camera_heights": 128,
            "camera_widths": 128,
        }
        
        self.env = OffScreenRenderEnv(**env_args)
        logger.info("✅ Environment initialized")
    
    def _tokenize(self, instruction: str) -> torch.Tensor:
        """언어 인스트럭션 토큰화"""
        tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,
        )
        return tokens.input_ids
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        # HWC -> CHW
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        # [0, 255] -> [0, 1]
        img = img / 255.0
        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = (img - mean) / std
        return img
    
    def _parse_observation(self, obs: dict) -> dict:
        """환경에서 관찰 데이터 추출"""
        return {
            "agentview": obs["agentview_image"],
            "eye_in_hand": obs["robot0_eye_in_hand_image"],
            "ee_pos": obs["robot0_eef_pos"],
            "ee_quat": obs["robot0_eef_quat"],
            "gripper": obs["robot0_gripper_qpos"],
        }
    
    def _predict_action(self, obs: dict, instruction: str) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """X-VLA로 action + time + variance 예측
        
        Returns:
            tuple: (action [7D], predicted_time [float or None], predicted_var [float or None])
        """
        # 이미지 전처리
        agentview = self._preprocess_image(obs["agentview"]).unsqueeze(0)
        eye_in_hand = self._preprocess_image(obs["eye_in_hand"]).unsqueeze(0)
        
        # 로봇 상태 (7D -> 20D 패딩)
        state_7d = np.concatenate([
            obs["ee_pos"],
            obs["ee_quat"][:3],
            obs["gripper"][:1]
        ])
        state_20d = np.zeros(20)
        state_20d[:7] = state_7d
        state = torch.from_numpy(state_20d).float().unsqueeze(0)
        
        # 토큰화
        input_ids = self._tokenize(instruction)
        
        # Device로 이동
        agentview = agentview.to(self.device)
        eye_in_hand = eye_in_hand.to(self.device)
        state = state.to(self.device)
        input_ids = input_ids.to(self.device)
        
        # Inference - generate_actions 사용 (time/var 예측 포함)
        predicted_time = None
        predicted_var = None
        
        with torch.no_grad():
            # 먼저 generate_actions로 시도 (Fine-tuned 모델)
            try:
                # 이미지를 [B, 1, C, H, W] 형태로 변환
                image_input = agentview.unsqueeze(1)  # [1, 1, C, H, W]
                image_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
                domain_id = torch.zeros(1, dtype=torch.long, device=self.device)
                
                actions, pred_time, pred_var = self.policy.model.generate_actions(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_id,
                    proprio=state,
                    steps=10,  # diffusion steps
                )
                
                # Time/Var 추출
                if pred_time is not None:
                    predicted_time = pred_time[0, 0].item()  # [B, 1] -> scalar
                if pred_var is not None:
                    # log_var -> std (uncertainty)
                    predicted_var = np.exp(0.5 * pred_var[0, 0].item())
                
                action_raw = actions[0, 0].cpu().numpy()  # [B, chunk, action_dim] -> [action_dim]
                
            except Exception as e:
                # Fallback: 기존 select_action 사용 (Baseline 모델)
                logger.debug(f"Using fallback select_action: {e}")
                batch = {
                    "observation.images.image": agentview,
                    "observation.images.image2": eye_in_hand,
                    "observation.state": state,
                    "observation.language.tokens": input_ids,
                }
                action_raw = self.policy.select_action(batch)
                action_raw = action_raw.cpu().numpy()[0]
        
        # Action 변환: X-VLA (20D) -> X-VLA (10D) -> 환경 (7D)
        # generate_actions는 [B, Chunk, 20]을 반환 (Action Space Postprocess 후)
        # 하지만 XVLAModel.generate_actions는 이미 postprocess를 호출해서 반환하므로
        # action_space에 따라 차원이 다를 수 있음.
        # XVLA config가 'ee6d'라면 20D, 'auto'라면 20D.
        # 여기서 필요한 것은 앞의 10차원 (pos(3) + rot6d(6) + gripper(1))
        
        # [B, Chunk, Dim] -> [Dim] (첫번째 배치의 첫번째 스텝)
        # Chunking: 현재 스텝의 액션만 사용 (Chunk의 첫번째 요소)
        action_raw = actions[0, 0].cpu().numpy()
        
        return self._convert_action_to_env(action_raw), predicted_time, predicted_var
    
    def _convert_action_to_env(self, action_raw: np.ndarray) -> np.ndarray:
        """X-VLA action (20D/10D) -> 환경 action (7D) 변환
        
        X-VLA output (Pretrained): [pos (3), rot_6d (6), gripper (1), ...padding...]
        Environment: [pos (3), rot_axis_angle (3), gripper (1)] = 7D
        """
        # 1. Position (3D)
        pos = action_raw[:3]
        
        # 2. 6D rotation -> axis-angle (3D)
        # Indices 3:9 are 6D rotation
        rot_6d = action_raw[3:9]
        rot_axis_angle = self._rot6d_to_axis_angle(rot_6d)
        
        # 3. Gripper (1D)
        # Index 9 is gripper
        gripper = action_raw[9:10]
        # XVLA Gripper is usually [0, 1] (sigmoid applied)
        # LIBERO expects [-1, 1]
        # 0.5 threshold로 binary 변환 또는 선형 변환
        # 여기서는 threshold 사용 (-1: open/close?, 1: open/close?)
        # LIBERO: -1 is open, 1 is closed (usually) -> Check LIBERO convention
        # XVLA: 0 is open, 1 is closed
        # Let's map [0, 1] -> [-1, 1] linearly first: 2*x - 1
        gripper = 2.0 * gripper - 1.0
        
        # 7D action
        action = np.concatenate([pos, rot_axis_angle, gripper])
        return action
    
    def _rot6d_to_axis_angle(self, rot_6d: np.ndarray) -> np.ndarray:
        """6D rotation representation -> axis-angle
        
        6D: [col1 (3), col2 (3)] of rotation matrix
        """
        # 6D -> rotation matrix
        col1 = rot_6d[:3]
        col2 = rot_6d[3:6]
        
        # Gram-Schmidt orthogonalization
        col1 = col1 / (np.linalg.norm(col1) + 1e-8)
        col2 = col2 - np.dot(col2, col1) * col1
        col2 = col2 / (np.linalg.norm(col2) + 1e-8)
        col3 = np.cross(col1, col2)
        
        rot_mat = np.stack([col1, col2, col3], axis=1)  # (3, 3)
        
        # Rotation matrix -> axis-angle
        # Using Rodrigues' formula inverse
        angle = np.arccos(np.clip((np.trace(rot_mat) - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = np.array([
            rot_mat[2, 1] - rot_mat[1, 2],
            rot_mat[0, 2] - rot_mat[2, 0],
            rot_mat[1, 0] - rot_mat[0, 1],
        ])
        axis = axis / (2 * np.sin(angle) + 1e-8)
        
        return axis * angle
    
    def _should_switch_subtask(
        self,
        subtask_idx: int,
        steps_in_subtask: int,
        obs: dict,
        prev_gripper: Optional[float],
        predicted_time: Optional[float] = None,
    ) -> bool:
        """Subtask 전환 여부 판단"""
        mode = self.config.subtask_switch_mode
        
        if mode == "fixed_steps":
            return steps_in_subtask >= self.config.fixed_steps_per_subtask
        
        elif mode == "gripper_change":
            if prev_gripper is None:
                return False
            current_gripper = obs["gripper"][0]
            gripper_changed = abs(current_gripper - prev_gripper) > 0.5
            return gripper_changed and steps_in_subtask >= 10
        
        elif mode == "time_prediction":
            # 시간 예측 기반 (Fine-tuned 모델 필요)
            if predicted_time is not None:
                time_threshold = 0.5  # 0.5초 미만이면 완료로 간주
                if predicted_time < time_threshold and steps_in_subtask >= 5:
                    return True
            return steps_in_subtask >= self.config.max_steps_per_subtask
        
        else:
            return steps_in_subtask >= self.config.fixed_steps_per_subtask
    
    def run_episode(self, episode_idx: int) -> InstructionResult:
        """한 에피소드(Instruction) 실행"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎮 Episode {episode_idx + 1}/{self.config.n_episodes}")
        logger.info(f"{'='*60}")
        
        # 환경 초기화
        self.env.seed(self.config.seed + episode_idx)
        raw_obs = self.env.reset()
        
        episode_start_time = time.time()
        
        # === Baseline 모드 또는 Task-Level 모드 ===
        if self.config.baseline or self.config.task_level_time:
            return self._run_full_task_episode(
                episode_idx, raw_obs, episode_start_time,
                track_time=self.config.task_level_time
            )
        
        # === Subtask 모드 (subtask가 없으면 full task로 fallback) ===
        if self.subtasks is None:
            logger.warning("No subtask metadata, falling back to full task mode")
            return self._run_full_task_episode(
                episode_idx, raw_obs, episode_start_time,
                track_time=False
            )
        
        return self._run_subtask_episode(episode_idx, raw_obs, episode_start_time)
    
    def _run_full_task_episode(
        self, 
        episode_idx: int, 
        raw_obs, 
        episode_start_time: float,
        track_time: bool = False
    ) -> InstructionResult:
        """Full task instruction으로 전체 에피소드 실행
        
        Args:
            track_time: True면 시간 예측 추적 (Task-level 모드)
        """
        mode_name = "Task-Level Time" if track_time else "Baseline"
        logger.info(f"📋 {mode_name} Mode: Using full task instruction")
        logger.info(f"   Instruction: {self.task_description}")
        
        total_steps = 0
        done = False
        info = {}
        
        initial_predicted_time = None
        initial_predicted_var = None
        all_time_predictions = []
        all_var_predictions = []
        
        while total_steps < self.config.max_steps_per_episode:
            obs = self._parse_observation(raw_obs)
            
            # Action + Time 예측
            action, predicted_time, predicted_var = self._predict_action(obs, self.task_description)
            
            # 첫 step의 시간 예측 저장 (Task-level 평가용)
            if total_steps == 0:
                initial_predicted_time = predicted_time
                initial_predicted_var = predicted_var
                if track_time and predicted_time is not None:
                    logger.info(f"   📊 Initial time prediction: {predicted_time:.2f}s (±{predicted_var:.2f}s)" if predicted_var else f"   📊 Initial time prediction: {predicted_time:.2f}s")
            
            # 모든 시간 예측 저장
            if predicted_time is not None:
                all_time_predictions.append(predicted_time)
            if predicted_var is not None:
                all_var_predictions.append(predicted_var)
            
            # 환경 step
            raw_obs, reward, done, info = self.env.step(action)
            total_steps += 1
            
            # 로깅 (매 20 step)
            if total_steps % 20 == 0:
                time_str = f", pred_time={predicted_time:.2f}s" if predicted_time else ""
                logger.info(f"   Step {total_steps}: action[:3]={action[:3].round(3)}{time_str}")
            
            if done or info.get("success", False):
                logger.info(f"   ✅ Episode done at step {total_steps}")
                break
        
        # 결과 생성
        episode_end_time = time.time()
        success = info.get("success", False)
        actual_total_time = episode_end_time - episode_start_time
        
        # Full task를 단일 subtask로 기록
        avg_predicted_time = np.mean(all_time_predictions) if all_time_predictions else None
        avg_predicted_var = np.mean(all_var_predictions) if all_var_predictions else None
        
        subtask_result = SubtaskResult(
            subtask_name=self.task_description,
            subtask_idx=0,
            start_step=0,
            end_step=total_steps,
            steps_taken=total_steps,
            predicted_time=avg_predicted_time,
            predicted_var=avg_predicted_var,
            actual_time=actual_total_time,
            completed=success,
            time_predictions=all_time_predictions,
        )
        
        result = InstructionResult(
            task_name=self.config.task_name,
            episode_idx=episode_idx,
            success=success,
            total_steps=total_steps,
            total_time=actual_total_time,
            initial_predicted_time=initial_predicted_time,
            initial_predicted_var=initial_predicted_var,
            subtask_results=[subtask_result],
        )
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        time_info = ""
        if track_time and initial_predicted_time is not None:
            time_error = abs(initial_predicted_time - actual_total_time)
            time_info = f" | Pred: {initial_predicted_time:.2f}s, Actual: {actual_total_time:.2f}s, Error: {time_error:.2f}s"
        
        logger.info(f"\n{status} - Total: {total_steps} steps, {actual_total_time:.2f}s{time_info}")
        
        return result
    
    def _run_subtask_episode(
        self, 
        episode_idx: int, 
        raw_obs, 
        episode_start_time: float
    ) -> InstructionResult:
        """Subtask 순차 실행 모드"""
        logger.info(f"📋 Subtask Mode: {len(self.subtasks)} subtasks")
        
        subtask_results = []
        total_steps = 0
        done = False
        info = {}
        
        for subtask_idx, subtask_instruction in enumerate(self.subtasks):
            logger.info(f"\n📌 Subtask {subtask_idx + 1}/{len(self.subtasks)}: {subtask_instruction}")
            
            subtask_start_step = total_steps
            subtask_start_time = time.time()
            steps_in_subtask = 0
            prev_gripper = None
            subtask_completed = False
            time_predictions = []
            var_predictions = []
            
            while steps_in_subtask < self.config.max_steps_per_subtask:
                obs = self._parse_observation(raw_obs)
                
                # Action + Time 예측 (현재 subtask instruction 사용)
                action, predicted_time, predicted_var = self._predict_action(obs, subtask_instruction)
                
                if predicted_time is not None:
                    time_predictions.append(predicted_time)
                if predicted_var is not None:
                    var_predictions.append(predicted_var)
                
                # 환경 step
                raw_obs, reward, done, info = self.env.step(action)
                
                total_steps += 1
                steps_in_subtask += 1
                
                # 로깅
                if steps_in_subtask % 10 == 0:
                    time_str = f", pred_time={predicted_time:.2f}s" if predicted_time else ""
                    var_str = f", std={predicted_var:.2f}" if predicted_var else ""
                    logger.info(f"   Step {steps_in_subtask}: action[:3]={action[:3].round(3)}{time_str}{var_str}")
                
                # 전체 에피소드 완료 체크
                if done or info.get("success", False):
                    subtask_completed = True
                    logger.info(f"   ✅ Episode done at step {total_steps}")
                    break
                
                # Subtask 전환 체크
                if self._should_switch_subtask(subtask_idx, steps_in_subtask, obs, prev_gripper, predicted_time):
                    subtask_completed = True
                    time_info = f" (pred_time={predicted_time:.2f}s)" if predicted_time else ""
                    logger.info(f"   → Switching to next subtask after {steps_in_subtask} steps{time_info}")
                    break
                
                prev_gripper = obs["gripper"][0]
                
                if total_steps >= self.config.max_steps_per_episode:
                    logger.warning(f"   ⚠️ Max steps reached: {total_steps}")
                    break
            
            # Subtask 결과 기록
            subtask_end_time = time.time()
            avg_predicted_time = np.mean(time_predictions) if time_predictions else None
            avg_predicted_var = np.mean(var_predictions) if var_predictions else None
            
            subtask_result = SubtaskResult(
                subtask_name=subtask_instruction,
                subtask_idx=subtask_idx,
                start_step=subtask_start_step,
                end_step=total_steps,
                steps_taken=steps_in_subtask,
                predicted_time=avg_predicted_time,
                predicted_var=avg_predicted_var,
                actual_time=subtask_end_time - subtask_start_time,
                completed=subtask_completed,
                time_predictions=time_predictions,
            )
            subtask_results.append(subtask_result)
            
            pred_info = f", avg_pred={avg_predicted_time:.2f}s" if avg_predicted_time else ""
            logger.info(f"   Subtask {subtask_idx + 1}: {steps_in_subtask} steps, actual={subtask_result.actual_time:.2f}s{pred_info}")
            
            if done or info.get("success", False) or total_steps >= self.config.max_steps_per_episode:
                break
        
        # Instruction 결과
        episode_end_time = time.time()
        success = info.get("success", False)
        
        result = InstructionResult(
            task_name=self.config.task_name,
            episode_idx=episode_idx,
            success=success,
            total_steps=total_steps,
            total_time=episode_end_time - episode_start_time,
            initial_predicted_time=None,  # Subtask 모드에서는 사용 안함
            initial_predicted_var=None,
            subtask_results=subtask_results,
        )
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"\n{status} - Total: {total_steps} steps, {result.total_time:.2f}s")
        
        return result
    
    def evaluate(self) -> dict:
        """전체 평가 실행"""
        logger.info("\n" + "=" * 70)
        
        if self.config.baseline:
            mode_str = "Baseline"
        elif self.config.task_level_time:
            mode_str = "Task-Level Time"
        else:
            mode_str = "Subtask-based"
        
        logger.info(f"🧪 Starting {mode_str} Evaluation")
        logger.info("=" * 70)
        logger.info(f"Task: {self.config.task_name}")
        logger.info(f"Mode: {mode_str}")
        logger.info(f"Model: {self.config.model_path}")
        
        if not self.config.baseline and not self.config.task_level_time and self.subtasks:
            logger.info(f"Subtasks: {len(self.subtasks)}")
            logger.info(f"Switch mode: {self.config.subtask_switch_mode}")
        
        logger.info(f"Episodes: {self.config.n_episodes}")
        
        self.results = []
        
        for episode_idx in range(self.config.n_episodes):
            result = self.run_episode(episode_idx)
            self.results.append(result)
        
        # 결과 집계
        summary = self._compute_summary()
        
        # 결과 출력
        self._print_summary(summary)
        
        # 결과 저장
        self._save_results(summary)
        
        return summary
    
    def _compute_summary(self) -> dict:
        """결과 집계"""
        # === Instruction 단위 통계 ===
        successes = [r.success for r in self.results]
        success_rate = np.mean(successes)
        
        total_times = [r.total_time for r in self.results]
        total_steps_list = [r.total_steps for r in self.results]
        
        # === Task-Level 시간 예측 통계 (initial_predicted_time vs actual_time) ===
        task_level_time_stats = {}
        initial_predictions = [(r.initial_predicted_time, r.total_time) for r in self.results 
                               if r.initial_predicted_time is not None]
        
        if initial_predictions:
            pred_times, actual_times = zip(*initial_predictions)
            errors = [abs(p - a) for p, a in zip(pred_times, actual_times)]
            
            task_level_time_stats = {
                "n_predictions": len(initial_predictions),
                "mean_predicted_time": float(np.mean(pred_times)),
                "mean_actual_time": float(np.mean(actual_times)),
                "mae": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(np.array(errors)**2))),
                "correlation": float(np.corrcoef(pred_times, actual_times)[0, 1]) if len(pred_times) > 1 else None,
            }
        
        # === Subtask 단위 통계 ===
        subtask_times = {}
        for result in self.results:
            for st_result in result.subtask_results:
                name = st_result.subtask_name
                if name not in subtask_times:
                    subtask_times[name] = []
                subtask_times[name].append({
                    "steps": st_result.steps_taken,
                    "predicted_time": st_result.predicted_time,
                    "actual_time": st_result.actual_time,
                    "completed": st_result.completed,
                })
        
        # Subtask별 통계 계산
        subtask_stats = {}
        all_subtask_time_errors = []
        
        for name, times in subtask_times.items():
            steps = [t["steps"] for t in times]
            actual_times = [t["actual_time"] for t in times]
            predicted_times = [t["predicted_time"] for t in times if t["predicted_time"] is not None]
            completed = [t["completed"] for t in times]
            
            stats = {
                "mean_steps": float(np.mean(steps)),
                "std_steps": float(np.std(steps)),
                "mean_actual_time": float(np.mean(actual_times)),
                "std_actual_time": float(np.std(actual_times)),
                "completion_rate": float(np.mean(completed)),
            }
            
            # 시간 예측 통계
            if predicted_times:
                stats["mean_predicted_time"] = float(np.mean(predicted_times))
                stats["std_predicted_time"] = float(np.std(predicted_times))
                
                paired_errors = []
                for t in times:
                    if t["predicted_time"] is not None:
                        error = abs(t["predicted_time"] - t["actual_time"])
                        paired_errors.append(error)
                        all_subtask_time_errors.append(error)
                
                if paired_errors:
                    stats["time_mae"] = float(np.mean(paired_errors))
                    stats["time_rmse"] = float(np.sqrt(np.mean(np.array(paired_errors)**2)))
            
            subtask_stats[name] = stats
        
        # 전체 Subtask 시간 예측 통계
        subtask_time_prediction_stats = {}
        if all_subtask_time_errors:
            subtask_time_prediction_stats = {
                "overall_mae": float(np.mean(all_subtask_time_errors)),
                "overall_rmse": float(np.sqrt(np.mean(np.array(all_subtask_time_errors)**2))),
                "n_predictions": len(all_subtask_time_errors),
            }
        
        # 평가 모드 결정
        if self.config.baseline:
            mode = "baseline"
        elif self.config.task_level_time:
            mode = "task_level_time"
        else:
            mode = "subtask"
        
        return {
            "task_name": self.config.task_name,
            "model_path": self.config.model_path,
            "n_episodes": self.config.n_episodes,
            "mode": mode,
            "subtask_switch_mode": self.config.subtask_switch_mode if mode == "subtask" else None,
            
            # Instruction 단위 통계
            "instruction_stats": {
                "success_rate": float(success_rate),
                "mean_total_time": float(np.mean(total_times)),
                "std_total_time": float(np.std(total_times)),
                "mean_total_steps": float(np.mean(total_steps_list)),
                "std_total_steps": float(np.std(total_steps_list)),
            },
            
            # Task-Level 시간 예측 통계 (Exp 4용)
            "task_level_time_stats": task_level_time_stats,
            
            # Subtask 단위 통계
            "subtask_stats": subtask_stats,
            
            # Subtask 시간 예측 통계
            "subtask_time_prediction_stats": subtask_time_prediction_stats,
            
            # Raw results
            "raw_results": [
                {
                    "episode_idx": r.episode_idx,
                    "success": r.success,
                    "total_steps": r.total_steps,
                    "total_time": r.total_time,
                    "initial_predicted_time": r.initial_predicted_time,
                    "initial_predicted_var": r.initial_predicted_var,
                    "subtasks": [
                        {
                            "name": st.subtask_name,
                            "steps": st.steps_taken,
                            "predicted_time": st.predicted_time,
                            "predicted_var": st.predicted_var,
                            "actual_time": st.actual_time,
                            "completed": st.completed,
                        }
                        for st in r.subtask_results
                    ]
                }
                for r in self.results
            ]
        }
    
    def _print_summary(self, summary: dict):
        """결과 출력"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        inst_stats = summary["instruction_stats"]
        print(f"\n📋 Task: {summary['task_name']}")
        print(f"   Model: {summary['model_path']}")
        print(f"   Mode: {summary['mode'].upper()}")
        print(f"   Episodes: {summary['n_episodes']}")
        if summary['subtask_switch_mode']:
            print(f"   Switch mode: {summary['subtask_switch_mode']}")
        
        print(f"\n🎯 Instruction-Level Results:")
        print(f"   Success Rate: {inst_stats['success_rate']*100:.1f}%")
        print(f"   Avg Steps: {inst_stats['mean_total_steps']:.1f} ± {inst_stats['std_total_steps']:.1f}")
        print(f"   Avg Time: {inst_stats['mean_total_time']:.2f}s ± {inst_stats['std_total_time']:.2f}s")
        
        # Task-Level 시간 예측 결과 (Exp 4 또는 task_level_time 모드)
        task_time_stats = summary.get("task_level_time_stats", {})
        if task_time_stats:
            print(f"\n⏱️ Task-Level Time Prediction:")
            print(f"   N predictions: {task_time_stats['n_predictions']}")
            print(f"   Avg Predicted: {task_time_stats['mean_predicted_time']:.2f}s")
            print(f"   Avg Actual: {task_time_stats['mean_actual_time']:.2f}s")
            print(f"   MAE: {task_time_stats['mae']:.2f}s")
            print(f"   RMSE: {task_time_stats['rmse']:.2f}s")
            if task_time_stats.get('correlation') is not None:
                print(f"   Correlation: {task_time_stats['correlation']:.3f}")
        
        # Subtask 단위 결과 (subtask 모드일 때만)
        if summary['mode'] == 'subtask' and summary.get('subtask_stats'):
            print(f"\n⏱️ Subtask-Level Results:")
            for name, stats in summary["subtask_stats"].items():
                # 이름이 너무 길면 줄임
                display_name = name[:50] + "..." if len(name) > 50 else name
                print(f"\n   📌 {display_name}")
                print(f"      Completion Rate: {stats['completion_rate']*100:.1f}%")
                print(f"      Avg Steps: {stats['mean_steps']:.1f} ± {stats['std_steps']:.1f}")
                print(f"      Avg Actual Time: {stats['mean_actual_time']:.2f}s ± {stats['std_actual_time']:.2f}s")
                
                if "mean_predicted_time" in stats:
                    print(f"      Avg Predicted Time: {stats['mean_predicted_time']:.2f}s ± {stats['std_predicted_time']:.2f}s")
                    if "time_mae" in stats:
                        print(f"      Time MAE: {stats['time_mae']:.2f}s, RMSE: {stats['time_rmse']:.2f}s")
            
            # Subtask 전체 시간 예측 통계
            subtask_time_stats = summary.get("subtask_time_prediction_stats", {})
            if subtask_time_stats:
                print(f"\n   🎯 Subtask Time Prediction Overall:")
                print(f"      MAE: {subtask_time_stats['overall_mae']:.2f}s")
                print(f"      RMSE: {subtask_time_stats['overall_rmse']:.2f}s")
                print(f"      N predictions: {subtask_time_stats['n_predictions']}")
        
        print("\n" + "=" * 70)
    
    def _save_results(self, summary: dict):
        """결과 저장"""
        output_dir = Path(self.config.output_dir) / self.config.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 결정
        if self.config.baseline:
            filename = "eval_results_baseline.json"
        elif self.config.task_level_time:
            filename = "eval_results_task_level_time.json"
        else:
            filename = f"eval_results_{self.config.subtask_switch_mode}.json"
        
        output_file = output_dir / filename
        
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, "env"):
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description="X-VLA Evaluation (Baseline, Subtask, Task-Level modes)")
    parser.add_argument("--task_suite", type=str, default="libero_goal")
    parser.add_argument("--task", type=str, default="open_the_middle_drawer_of_the_cabinet")
    parser.add_argument("--model_path", type=str, default="lerobot/xvla-libero")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=300)
    parser.add_argument("--switch_mode", type=str, default="fixed_steps",
                       choices=["fixed_steps", "gripper_change", "time_prediction"])
    parser.add_argument("--fixed_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true",
                       help="Baseline mode: full task instruction, no time prediction tracking")
    parser.add_argument("--task_level_time", action="store_true",
                       help="Task-level time mode: full task instruction with time prediction tracking")
    
    args = parser.parse_args()
    
    # baseline과 task_level_time은 동시 사용 불가
    if args.baseline and args.task_level_time:
        print("❌ Cannot use --baseline and --task_level_time together")
        return
    
    config = EvalConfig(
        task_suite=args.task_suite,
        task_name=args.task,
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        subtask_switch_mode=args.switch_mode,
        fixed_steps_per_subtask=args.fixed_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        baseline=args.baseline,
        task_level_time=args.task_level_time,
    )
    
    if not HAS_LIBERO or not HAS_XVLA:
        print("❌ Required packages not available")
        print(f"   HAS_LIBERO: {HAS_LIBERO}")
        print(f"   HAS_XVLA: {HAS_XVLA}")
        return
    
    evaluator = SubtaskBasedEvaluator(config)
    
    try:
        summary = evaluator.evaluate()
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
