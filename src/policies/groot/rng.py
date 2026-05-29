"""RNG helpers shared by GR00T serving transports."""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch


@contextlib.contextmanager
def temporary_inference_seed(seed: Any):
    """Optionally make one inference call deterministic without leaking RNG state."""
    if seed is None:
        yield
        return

    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inference_seed must be an integer: {seed!r}") from exc

    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(seed_int)
        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
