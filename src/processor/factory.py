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
    image_size: int = 200,
    gripper_threshold: float = 0.0,
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """Calvin 벤치마크용 (obs_pipeline, action_pipeline) 생성.

    Args:
        use_wrist: wrist 카메라 이미지를 포함할지 여부.
        image_size: Calvin 이미지 해상도 (기본 200).
        gripper_threshold: gripper 이산화 임계값.

    Returns:
        (obs_pipeline, action_pipeline) 튜플.
    """
    obs_pipeline = DataProcessorPipeline(
        steps=[CalvinObsProcessor(use_wrist=use_wrist, image_size=image_size)],
        name="calvin_obs",
    )
    action_pipeline = DataProcessorPipeline(
        steps=[CalvinActionProcessor(threshold=gripper_threshold)],
        name="calvin_action",
    )
    return obs_pipeline, action_pipeline


def make_robocasa_processors(
    static_cam: str = "robot0_agentview_left",
    wrist_cam: str = "robot0_eye_in_hand",
    state_key: str = "robot0_proprio-state",
    image_size: int = 224,
    arm_dim: int = 6,
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """RoboCasa 벤치마크용 (obs_pipeline, action_pipeline) 생성.

    Args:
        static_cam: 정적 카메라 이름 (robosuite 환경 기준).
        wrist_cam: 손목 카메라 이름.
        state_key: proprioceptive state 키.
        image_size: 이미지 해상도 (기본 224).
        arm_dim: arm action 차원 수 (기본 6).

    Returns:
        (obs_pipeline, action_pipeline) 튜플.
    """
    obs_pipeline = DataProcessorPipeline(
        steps=[RoboCasaObsProcessor(
            static_cam=static_cam,
            wrist_cam=wrist_cam,
            state_key=state_key,
            image_size=image_size,
        )],
        name="robocasa_obs",
    )
    action_pipeline = DataProcessorPipeline(
        steps=[RoboCasaActionProcessor(arm_dim=arm_dim)],
        name="robocasa_action",
    )
    return obs_pipeline, action_pipeline
