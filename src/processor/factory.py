"""Factory functions for creating benchmark-specific processor pipelines.

각 벤치마크에 대해 (obs_pipeline, action_pipeline) 튜플을 반환한다.
벤치마크 eval 스크립트에서는 이 팩토리만 호출하면 된다.

Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Tuple

from .base import DataProcessorPipeline
from .obs.calvin import CalvinObsProcessor
from .obs.robocasa import RoboCasaObsProcessor
from .action.calvin import CalvinActionProcessor
from .action.robocasa import RoboCasaActionProcessor


def make_calvin_processors(
    use_wrist: bool = False,
    use_state: bool = True,
    image_size: int = 200,
    gripper_threshold: float = 0.0,
    action_type: str = "relative",
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """Calvin 벤치마크용 (obs_pipeline, action_pipeline) 생성.

    Args:
        use_wrist: wrist 카메라 이미지를 포함할지 여부.
        use_state: robot_obs에서 state sub-keys를 추출할지 여부.
        image_size: Calvin 이미지 해상도 (기본 200).
        gripper_threshold: gripper 이산화 임계값.
                          relative 모델(DreamVLA 등): 0.0 (기본).
                          absolute 모델(X-VLA 등): 0.8 권장.
        action_type: "relative" 또는 "absolute".
                    absolute 시 CalvinActionProcessor가 3-tuple 반환 →
                    Calvin env가 절대좌표로 직접 처리.

    Returns:
        (obs_pipeline, action_pipeline) 튜플.
    """
    obs_pipeline = DataProcessorPipeline(
        steps=[CalvinObsProcessor(use_wrist=use_wrist, use_state=use_state, image_size=image_size)],
        name="calvin_obs",
    )
    action_pipeline = DataProcessorPipeline(
        steps=[CalvinActionProcessor(threshold=gripper_threshold, action_type=action_type)],
        name="calvin_action",
    )
    return obs_pipeline, action_pipeline


def make_robocasa_processors(
    static_cam: str = "robot0_agentview_left",
    wrist_cam: str = "robot0_eye_in_hand",
    robot_prefix: str = "robot0_",
    image_size: int = 224,
    arm_dim: int = 6,
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """RoboCasa 벤치마크용 (obs_pipeline, action_pipeline) 생성.

    Args:
        static_cam: 정적 카메라 이름 (robosuite 환경 기준).
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
