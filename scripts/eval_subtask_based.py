#!/usr/bin/env python3
"""
Subtask 기반 X-VLA Evaluation Script

실행 흐름:
1. 각 Instruction(Task)에 대해:
   - Subtask 목록을 순차적으로 실행
   - Subtask 1 → Subtask 2 → ... → Subtask N
   - 모든 Subtask 완료 후 Instruction 성공 여부 판정

채점 방식:
- 성공률: Instruction 단위 (전체 task 완료 여부)
- 시간: Subtask 단위 (각 subtask별 소요 시간)

Usage:
    python scripts/eval_subtask_based.py --task open_the_middle_drawer_of_the_cabinet --n_episodes 10
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer

# LIBERO imports
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    HAS_LIBERO = True
except ImportError:
    HAS_LIBERO = False
    print("⚠️ LIBERO not available")

# LeRobot X-VLA imports
try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    HAS_XVLA = True
except ImportError:
    HAS_XVLA = False
    print("⚠️ X-VLA not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    predicted_time: Optional[float] = None  # Fine-tuned 모델에서만
    actual_time: Optional[float] = None  # 실제 소요 시간 (초)
    completed: bool = False  # Subtask 완료 여부


@dataclass
class InstructionResult:
    """단일 Instruction 실행 결과"""
    task_name: str
    episode_idx: int
    success: bool
    total_steps: int
    total_time: float  # 총 소요 시간 (초)
    subtask_results: list[SubtaskResult] = field(default_factory=list)


@dataclass
class EvalConfig:
    """평가 설정"""
    task_suite: str = "libero_goal"
    task_name: str = "open_the_middle_drawer_of_the_cabinet"
    model_path: str = "lerobot/xvla-libero"
    n_episodes: int = 10
    max_steps_per_episode: int = 300
    max_steps_per_subtask: int = 100  # Subtask당 최대 step
    subtask_switch_mode: str = "fixed_steps"  # "fixed_steps", "gripper_change", "time_prediction"
    fixed_steps_per_subtask: int = 50  # fixed_steps 모드에서 사용
    output_dir: str = "outputs/eval"
    save_video: bool = False
    seed: int = 42


class SubtaskBasedEvaluator:
    """Subtask 기반 평가기"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 결과 저장
        self.results: list[InstructionResult] = []
        
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
        
        # Subtask 메타데이터 로드
        self._load_subtask_metadata()
        
        # LIBERO 환경 설정
        self._setup_libero_env()
    
    def _load_subtask_metadata(self):
        """Subtask 메타데이터 로드"""
        subtask_dir = Path(f"/workspace/vla_tset/data/datasets/libero_goal_subtasks/{self.config.task_name}")
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
            # 기본 subtask (전체 task를 하나의 subtask로)
            self.subtasks = [self._get_task_description()]
            self.metadata = {"subtasks": self.subtasks}
    
    def _get_task_description(self) -> str:
        """Task description 가져오기"""
        # Task name에서 description 추출
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
    
    def _predict_action(self, obs: dict, instruction: str) -> np.ndarray:
        """X-VLA로 action 예측"""
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
        
        # Inference
        with torch.no_grad():
            batch = {
                "observation.images.image": agentview,
                "observation.images.image2": eye_in_hand,
                "observation.state": state,
                "observation.language.tokens": input_ids,
            }
            action_raw = self.policy.select_action(batch)
        
        # Action 변환: X-VLA (10D) -> 환경 (7D)
        action_raw = action_raw.cpu().numpy()[0]
        action = self._convert_action_to_env(action_raw)
        
        return action
    
    def _convert_action_to_env(self, action_raw: np.ndarray) -> np.ndarray:
        """X-VLA action (10D) -> 환경 action (7D) 변환
        
        X-VLA output: [pos (3), rot_6d (6), gripper (1)] = 10D
        Environment: [pos (3), rot_axis_angle (3), gripper (1)] = 7D
        """
        # Position (3D)
        pos = action_raw[:3]
        
        # 6D rotation -> axis-angle (3D)
        rot_6d = action_raw[3:9]
        rot_axis_angle = self._rot6d_to_axis_angle(rot_6d)
        
        # Gripper (1D)
        gripper = action_raw[9:10]
        # 0.5 threshold로 binary 변환
        gripper = np.where(gripper > 0.5, 1.0, -1.0)
        
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
    ) -> bool:
        """Subtask 전환 여부 판단"""
        mode = self.config.subtask_switch_mode
        
        if mode == "fixed_steps":
            # 고정된 step 수
            return steps_in_subtask >= self.config.fixed_steps_per_subtask
        
        elif mode == "gripper_change":
            # Gripper 상태 변화 감지
            if prev_gripper is None:
                return False
            current_gripper = obs["gripper"][0]
            # Gripper가 열리거나 닫히면 subtask 전환
            gripper_changed = abs(current_gripper - prev_gripper) > 0.5
            # 최소 10 step 이후에만 전환
            return gripper_changed and steps_in_subtask >= 10
        
        elif mode == "time_prediction":
            # 시간 예측 기반 (Fine-tuned 모델 필요)
            # TODO: pred_time이 0에 가까워지면 전환
            return steps_in_subtask >= self.config.fixed_steps_per_subtask
        
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
        
        # 결과 초기화
        subtask_results = []
        total_steps = 0
        episode_start_time = time.time()
        done = False
        info = {}
        
        # 각 Subtask 순차 실행
        for subtask_idx, subtask_instruction in enumerate(self.subtasks):
            logger.info(f"\n📌 Subtask {subtask_idx + 1}/{len(self.subtasks)}: {subtask_instruction}")
            
            subtask_start_step = total_steps
            subtask_start_time = time.time()
            steps_in_subtask = 0
            prev_gripper = None
            subtask_completed = False
            
            while steps_in_subtask < self.config.max_steps_per_subtask:
                # 관찰
                obs = self._parse_observation(raw_obs)
                
                # Action 예측 (현재 subtask instruction 사용)
                action = self._predict_action(obs, subtask_instruction)
                
                # 환경 step
                raw_obs, reward, done, info = self.env.step(action)
                
                total_steps += 1
                steps_in_subtask += 1
                
                # 로깅
                if steps_in_subtask % 10 == 0:
                    logger.info(f"   Step {steps_in_subtask}: action[:3]={action[:3].round(3)}")
                
                # 전체 에피소드 완료 체크
                if done or info.get("success", False):
                    subtask_completed = True
                    logger.info(f"   ✅ Episode done at step {total_steps}")
                    break
                
                # Subtask 전환 체크
                if self._should_switch_subtask(subtask_idx, steps_in_subtask, obs, prev_gripper):
                    subtask_completed = True
                    logger.info(f"   → Switching to next subtask after {steps_in_subtask} steps")
                    break
                
                prev_gripper = obs["gripper"][0]
                
                # 최대 step 체크
                if total_steps >= self.config.max_steps_per_episode:
                    logger.warning(f"   ⚠️ Max steps reached: {total_steps}")
                    break
            
            # Subtask 결과 기록
            subtask_end_time = time.time()
            subtask_result = SubtaskResult(
                subtask_name=subtask_instruction,
                subtask_idx=subtask_idx,
                start_step=subtask_start_step,
                end_step=total_steps,
                steps_taken=steps_in_subtask,
                actual_time=subtask_end_time - subtask_start_time,
                completed=subtask_completed,
            )
            subtask_results.append(subtask_result)
            
            logger.info(f"   Subtask {subtask_idx + 1}: {steps_in_subtask} steps, {subtask_result.actual_time:.2f}s")
            
            # 전체 에피소드가 끝났으면 루프 종료
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
            subtask_results=subtask_results,
        )
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"\n{status} - Total: {total_steps} steps, {result.total_time:.2f}s")
        
        return result
    
    def evaluate(self) -> dict:
        """전체 평가 실행"""
        logger.info("\n" + "=" * 70)
        logger.info("🧪 Starting Subtask-based Evaluation")
        logger.info("=" * 70)
        logger.info(f"Task: {self.config.task_name}")
        logger.info(f"Subtasks: {len(self.subtasks)}")
        logger.info(f"Episodes: {self.config.n_episodes}")
        logger.info(f"Switch mode: {self.config.subtask_switch_mode}")
        
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
        # Instruction 단위 성공률
        successes = [r.success for r in self.results]
        success_rate = np.mean(successes)
        
        # 총 소요 시간 통계
        total_times = [r.total_time for r in self.results]
        total_steps_list = [r.total_steps for r in self.results]
        
        # Subtask 단위 시간 통계
        subtask_times = {}
        for result in self.results:
            for st_result in result.subtask_results:
                name = st_result.subtask_name
                if name not in subtask_times:
                    subtask_times[name] = []
                subtask_times[name].append({
                    "steps": st_result.steps_taken,
                    "time": st_result.actual_time,
                    "completed": st_result.completed,
                })
        
        # Subtask별 통계 계산
        subtask_stats = {}
        for name, times in subtask_times.items():
            steps = [t["steps"] for t in times]
            actual_times = [t["time"] for t in times]
            completed = [t["completed"] for t in times]
            
            subtask_stats[name] = {
                "mean_steps": np.mean(steps),
                "std_steps": np.std(steps),
                "mean_time": np.mean(actual_times),
                "std_time": np.std(actual_times),
                "completion_rate": np.mean(completed),
            }
        
        return {
            "task_name": self.config.task_name,
            "n_episodes": self.config.n_episodes,
            "subtask_switch_mode": self.config.subtask_switch_mode,
            
            # Instruction 단위 통계
            "instruction_stats": {
                "success_rate": success_rate,
                "mean_total_time": np.mean(total_times),
                "std_total_time": np.std(total_times),
                "mean_total_steps": np.mean(total_steps_list),
                "std_total_steps": np.std(total_steps_list),
            },
            
            # Subtask 단위 통계
            "subtask_stats": subtask_stats,
            
            # Raw results
            "raw_results": [
                {
                    "episode_idx": r.episode_idx,
                    "success": r.success,
                    "total_steps": r.total_steps,
                    "total_time": r.total_time,
                    "subtasks": [
                        {
                            "name": st.subtask_name,
                            "steps": st.steps_taken,
                            "time": st.actual_time,
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
        print(f"   Episodes: {summary['n_episodes']}")
        print(f"   Switch mode: {summary['subtask_switch_mode']}")
        
        print(f"\n🎯 Instruction-Level Results:")
        print(f"   Success Rate: {inst_stats['success_rate']*100:.1f}%")
        print(f"   Avg Steps: {inst_stats['mean_total_steps']:.1f} ± {inst_stats['std_total_steps']:.1f}")
        print(f"   Avg Time: {inst_stats['mean_total_time']:.2f}s ± {inst_stats['std_total_time']:.2f}s")
        
        print(f"\n⏱️ Subtask-Level Results:")
        for name, stats in summary["subtask_stats"].items():
            print(f"\n   📌 {name[:50]}...")
            print(f"      Completion Rate: {stats['completion_rate']*100:.1f}%")
            print(f"      Avg Steps: {stats['mean_steps']:.1f} ± {stats['std_steps']:.1f}")
            print(f"      Avg Time: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s")
        
        print("\n" + "=" * 70)
    
    def _save_results(self, summary: dict):
        """결과 저장"""
        output_dir = Path("/workspace/vla_tset") / self.config.output_dir / self.config.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 저장
        output_file = output_dir / f"eval_results_{self.config.subtask_switch_mode}.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, "env"):
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Subtask-based X-VLA Evaluation")
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
    
    args = parser.parse_args()
    
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
    )
    
    if not HAS_LIBERO or not HAS_XVLA:
        print("❌ Required packages not available")
        return
    
    evaluator = SubtaskBasedEvaluator(config)
    
    try:
        summary = evaluator.evaluate()
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
