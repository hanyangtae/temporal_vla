"""TTT × VLA backbone integrations.

각 모듈은 한 backbone 모델에 TTT predictor 를 붙이는 wrapper / 헬퍼를 노출한다.
"""

from src.ttt.integrations.groot_wrapper import (
    Gr00tN1d6WithTTT,
    attach_ttt_to_groot,
)

__all__ = ["Gr00tN1d6WithTTT", "attach_ttt_to_groot"]
