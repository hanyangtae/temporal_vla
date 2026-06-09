"""Factory functions for creating benchmark-specific processor pipelines.

각 벤치마크에 대해 (obs_pipeline, action_pipeline) 튜플을 반환한다.
벤치마크 eval 스크립트에서는 이 팩토리만 호출하면 된다.

Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Tuple

from .base import DataProcessorPipeline
from .obs.calvin import CalvinObsProcessor
from .obs.groot_robocasa import GrootRoboCasaObsProcessor
from .obs.libero import LIBEROObsProcessor
from .obs.robocasa import RoboCasaObsProcessor
from .action.calvin import CalvinActionProcessor
from .action.groot_robocasa import GrootRoboCasaActionProcessor
from .action.libero import LIBEROActionProcessor
from .action.robocasa import RoboCasaActionProcessor


def make_calvin_processors(
    use_wrist: bool = False,
    use_state: bool = True,
    image_size: int = 200,
    gripper_threshold: float = 0.0,
    action_type: str = "relative",
    gripper_invert: bool = False,
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """Calvin 벤치마크용 (obs_pipeline, action_pipeline) 생성.

    Args:
        use_wrist: wrist 카메라 이미지를 포함할지 여부.
        use_state: robot_obs에서 state sub-keys를 추출할지 여부.
        image_size: Calvin 이미지 해상도 (기본 200).
        gripper_threshold: gripper 이산화 임계값.
                          relative + 표준 컨벤션 (DreamVLA): 0.0 (기본).
                          absolute + invert 컨벤션 (X-VLA sigmoid 1=close): 0.8 권장.
        action_type: "relative" 또는 "absolute".
                    absolute 시 CalvinActionProcessor가 3-tuple 반환 →
                    Calvin env가 절대좌표로 직접 처리.
        gripper_invert: gripper emit 컨벤션 invert.
                       False (default, 표준): emit 클수록 open. DreamVLA 같은 ±1
                         binarized 출력. `g > threshold → open(+1)`.
                       True: emit 클수록 close. X-VLA 같은 sigmoid (1=close) 출력.
                         `g < threshold → open(+1)`.

    Returns:
        (obs_pipeline, action_pipeline) 튜플.
    """
    obs_pipeline = DataProcessorPipeline(
        steps=[CalvinObsProcessor(use_wrist=use_wrist, use_state=use_state, image_size=image_size)],
        name="calvin_obs",
    )
    action_pipeline = DataProcessorPipeline(
        steps=[
            CalvinActionProcessor(
                threshold=gripper_threshold,
                action_type=action_type,
                gripper_invert=gripper_invert,
            )
        ],
        name="calvin_action",
    )
    return obs_pipeline, action_pipeline


def make_libero_processors(
    image_size: int = 256,
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """LIBERO 벤치마크용 (obs_pipeline, action_pipeline) 생성.

    Args:
        image_size: 이미지 해상도 (LIBERO env `camera_heights`/`camera_widths`).
                   openvla-oft 학습 기본 = 256.

    데이터 흐름:
        env obs (raw orientation) → obs_processor →
        unified payload (`observation.images.{static,wrist}`, `observation.state.{eef_pos,eef_quat,gripper_qpos}`)
        → /act 서버 (학습 시 사용된 dataset_statistics 기준 normalize/denorm) →
        sub-key action (이미 디노말 + gripper 후처리 완료) → action_processor →
        7D LIBERO env action `[pos(3), axisangle(3), gripper(1)]`.

    Returns:
        (obs_pipeline, action_pipeline) 튜플.
    """
    obs_pipeline = DataProcessorPipeline(
        steps=[LIBEROObsProcessor(image_size=image_size)],
        name="libero_obs",
    )
    action_pipeline = DataProcessorPipeline(
        steps=[LIBEROActionProcessor()],
        name="libero_action",
    )
    return obs_pipeline, action_pipeline


def make_robocasa_processors(
    static_cam: str = "robot0_agentview_left",
    static_cam2: str = None,
    wrist_cam: str = "robot0_eye_in_hand",
    robot_prefix: str = "robot0_",
    image_size: int = 224,
    arm_dim: int = 6,
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """RoboCasa 벤치마크용 (obs_pipeline, action_pipeline) 생성.

    Args:
        static_cam: 정적 카메라 이름 (robosuite 환경 기준).
        static_cam2: 두 번째 정적 카메라 (3-camera 모델용, None이면 2-camera).
        wrist_cam: 손목 카메라 이름.
        robot_prefix: 로봇 observable 키 접두사 (기본 "robot0_").
        image_size: 이미지 해상도 (기본 224).
        arm_dim: arm action 차원 수 (기본 6).

    Returns:
        (obs_pipeline, action_pipeline) 튜플.
    """
    obs_pipeline = DataProcessorPipeline(
        steps=[RoboCasaObsProcessor(
            static_cam=static_cam,
            static_cam2=static_cam2,
            wrist_cam=wrist_cam,
            robot_prefix=robot_prefix,
            image_size=image_size,
        )],
        name="robocasa_obs",
    )
    action_pipeline = DataProcessorPipeline(
        steps=[RoboCasaActionProcessor(arm_dim=arm_dim)],
        name="robocasa_action",
    )
    return obs_pipeline, action_pipeline


def make_groot_robocasa_processors(
    strict: bool = True,
    action_mode: str = "step",
    image_size: int = 256,
    action_horizon: int = 16,
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """GR00T GrootRoboCasaEnv용 (obs_pipeline, action_pipeline) 생성.

    Generic robosuite RoboCasa processor와 달리 GrootRoboCasaEnv native keys를
    ``src.policies.groot.robocasa.io`` adapter contract에 맞춰 처리한다.
    """
    obs_pipeline = DataProcessorPipeline(
        steps=[GrootRoboCasaObsProcessor(strict=strict, image_size=image_size)],
        name="groot_robocasa_obs",
    )
    action_pipeline = DataProcessorPipeline(
        steps=[
            GrootRoboCasaActionProcessor(
                mode=action_mode,
                action_horizon=action_horizon,
            )
        ],
        name="groot_robocasa_action",
    )
    return obs_pipeline, action_pipeline
