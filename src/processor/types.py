"""Feature types and constants for the processor pipeline.

Mirrors LeRobot's configs/types.py conventions so that processor configs
are interoperable when bridging to LeRobot later.

Python 3.8 compatible.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Feature type enums (mirrors lerobot.configs.types)
# ---------------------------------------------------------------------------


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ACTION = "ACTION"
    LANGUAGE = "LANGUAGE"


class PipelineFeatureType(str, Enum):
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


# ---------------------------------------------------------------------------
# Feature descriptor
# ---------------------------------------------------------------------------


class PolicyFeature:
    """Describes a single feature's type and shape.

    Equivalent to ``lerobot.configs.types.PolicyFeature`` but without
    requiring ``dataclasses`` on every Python version.
    """

    def __init__(self, type: FeatureType, shape: Tuple[int, ...]):
        self.type = type
        self.shape = shape

    def __repr__(self) -> str:
        return "PolicyFeature(type={!r}, shape={!r})".format(self.type, self.shape)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolicyFeature):
            return NotImplemented
        return self.type == other.type and self.shape == other.shape


# ---------------------------------------------------------------------------
# Feature dict type alias (plain dict for 3.8 compat)
# ---------------------------------------------------------------------------
# features = {
#   PipelineFeatureType.OBSERVATION: {
#       "observation.images.static": PolicyFeature(VISUAL, (3, 224, 224)),
#       "observation.state":         PolicyFeature(STATE,  (7,)),
#   },
#   PipelineFeatureType.ACTION: {
#       "action": PolicyFeature(ACTION, (7,)),
#   },
# }
Features = Dict[PipelineFeatureType, Dict[str, PolicyFeature]]


# ---------------------------------------------------------------------------
# Transition keys (mirrors lerobot.processor.core.TransitionKey)
# ---------------------------------------------------------------------------


class TransitionKey(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    INFO = "info"


# ---------------------------------------------------------------------------
# Transition type (plain dict, not TypedDict, for 3.8 compat)
# ---------------------------------------------------------------------------
# Transition = {
#   "observation": {"observation.images.static": np.ndarray, ...},
#   "action": np.ndarray | None,
#   "info": dict | None,
# }
Transition = Dict[str, Any]


def create_transition(
    observation: Optional[Dict[str, Any]] = None,
    action: Any = None,
    info: Optional[Dict[str, Any]] = None,
) -> Transition:
    """Create a Transition dict with sensible defaults."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.INFO: info if info is not None else {},
    }
