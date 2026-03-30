"""Base classes for the processor pipeline.

Follows LeRobot's ProcessorStep / DataProcessorPipeline interface contract
but is self-contained, numpy-centric, and Python 3.8 compatible.

Python 3.8 compatible.
"""

from __future__ import annotations

import importlib
import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import Features, PolicyFeature, Transition, TransitionKey, create_transition

# ============================================================================
# ProcessorStep — single transformation
# ============================================================================


class ProcessorStep(ABC):
    """Abstract base class for a single step in a data processing pipeline.

    Interface mirrors ``lerobot.processor.pipeline.ProcessorStep``.
    """

    @abstractmethod
    def __call__(self, transition: Transition) -> Transition:
        return transition

    @abstractmethod
    def transform_features(self, features: Features) -> Features:
        return features

    def get_config(self) -> Dict[str, Any]:
        return {}

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass

    def reset(self) -> None:
        pass


# ============================================================================
# Specialised base classes — narrow __call__ to a single field
# ============================================================================


class ObservationProcessorStep(ProcessorStep, ABC):
    """Step that only transforms the *observation* part of a transition."""

    @abstractmethod
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]: ...

    def __call__(self, transition: Transition) -> Transition:
        t = dict(transition)
        obs = t.get(TransitionKey.OBSERVATION)
        if obs is None:
            return t
        t[TransitionKey.OBSERVATION] = self.process_observation(dict(obs))
        return t


class ActionProcessorStep(ProcessorStep, ABC):
    """Step that only transforms the *action* part of a transition."""

    @abstractmethod
    def process_action(self, action: Any) -> Any: ...

    def __call__(self, transition: Transition) -> Transition:
        t = dict(transition)
        act = t.get(TransitionKey.ACTION)
        if act is None:
            return t
        t[TransitionKey.ACTION] = self.process_action(act)
        return t


# ============================================================================
# DataProcessorPipeline — chains steps
# ============================================================================


class DataProcessorPipeline:
    """Sequential pipeline that chains multiple :class:`ProcessorStep` instances.

    Mirrors ``lerobot.processor.pipeline.DataProcessorPipeline`` but without
    HuggingFace Hub / torch / safetensors dependencies.
    """

    def __init__(
        self,
        steps: Sequence[ProcessorStep],
        name: str = "DataProcessorPipeline",
    ):
        self.steps: Sequence[ProcessorStep] = steps
        self.name = name

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def __call__(self, data: Transition) -> Transition:
        transition = data
        for step in self.steps:
            transition = step(transition)
        return transition

    def reset(self) -> None:
        for step in self.steps:
            step.reset()

    # ------------------------------------------------------------------
    # Feature tracking
    # ------------------------------------------------------------------

    def transform_features(self, features: Features) -> Features:
        """Propagate feature metadata through the full pipeline."""
        for step in self.steps:
            features = step.transform_features(features)
        return features

    # ------------------------------------------------------------------
    # Serialisation (lightweight JSON, no safetensors)
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """Save pipeline config + state as JSON files."""
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "name": self.name,
            "steps": [],
        }  # type: Dict[str, Any]

        for idx, step in enumerate(self.steps):
            entry = {
                "class": "{}.{}".format(
                    step.__class__.__module__, step.__class__.__name__
                ),
                "config": step.get_config(),
            }  # type: Dict[str, Any]

            state = step.state_dict()
            if state:
                state_file = "step_{}_{}.json".format(idx, step.__class__.__name__)
                _save_state_json(state, os.path.join(save_directory, state_file))
                entry["state_file"] = state_file

            config["steps"].append(entry)

        with open(os.path.join(save_directory, "pipeline.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "DataProcessorPipeline":
        """Load pipeline from a directory saved by :meth:`save_pretrained`."""
        config_path = os.path.join(save_directory, "pipeline.json")
        with open(config_path) as f:
            config = json.load(f)

        steps = []  # type: List[ProcessorStep]
        for idx, entry in enumerate(config["steps"]):
            step_class = _import_class(entry["class"])
            step = step_class(**entry.get("config", {}))

            if "state_file" in entry:
                state = _load_state_json(
                    os.path.join(save_directory, entry["state_file"])
                )
                step.load_state_dict(state)

            steps.append(step)

        return cls(steps=steps, name=config.get("name", "DataProcessorPipeline"))

    def __repr__(self) -> str:
        step_names = [s.__class__.__name__ for s in self.steps]
        return "{}(name={!r}, steps={})".format(
            self.__class__.__name__, self.name, step_names
        )


# ============================================================================
# Helpers (private)
# ============================================================================


def _import_class(full_class_path: str) -> type:
    module_path, class_name = full_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _save_state_json(state: Dict[str, Any], path: str) -> None:
    serialisable = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = {
                "__ndarray__": True,
                "data": v.tolist(),
                "dtype": str(v.dtype),
            }
        else:
            serialisable[k] = v
    with open(path, "w") as f:
        json.dump(serialisable, f)


def _load_state_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        raw = json.load(f)
    state = {}  # type: Dict[str, Any]
    for k, v in raw.items():
        if isinstance(v, dict) and v.get("__ndarray__"):
            state[k] = np.array(v["data"], dtype=v["dtype"])
        else:
            state[k] = v
    return state
