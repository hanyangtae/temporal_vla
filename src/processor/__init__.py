"""Generic benchmark processor pipeline for Project FastAPI evaluation.

LeRobot의 ProcessorStep/DataProcessorPipeline 인터페이스를 따르는
경량 구현. numpy 기반, Python 3.8 호환.

이 package는 RoboCasa/Calvin/LIBERO eval script가 env-native obs/action을
project HTTP schema로 바꾸는 benchmark-side pipeline이다. GR00T
`GrootRoboCasaEnv` adapter도 같은 pipeline shape를 제공하지만, 실제 native
key mapping 구현은 `src.policies.groot.robocasa.io`를 단일 출처로 둔다.
"""
from .base import (
    ActionProcessorStep,
    DataProcessorPipeline,
    ObservationProcessorStep,
    ProcessorStep,
)
from .types import (
    FeatureType,
    Features,
    PipelineFeatureType,
    PolicyFeature,
    Transition,
    TransitionKey,
    create_transition,
)
from .factory import make_calvin_processors, make_robocasa_processors
from .factory import make_groot_robocasa_processors

__all__ = [
    # Base
    "ProcessorStep",
    "ObservationProcessorStep",
    "ActionProcessorStep",
    "DataProcessorPipeline",
    # Types
    "FeatureType",
    "PipelineFeatureType",
    "PolicyFeature",
    "Features",
    "Transition",
    "TransitionKey",
    "create_transition",
    # Factory
    "make_calvin_processors",
    "make_groot_robocasa_processors",
    "make_robocasa_processors",
]
