"""Explicit LeRobot policy adapter registry."""

from __future__ import annotations

from .groot import GrootPolicyAdapter
from .pi import PiPolicyAdapter


_POLICY_ADAPTERS = {
    "pi0": PiPolicyAdapter,
    "pi05": PiPolicyAdapter,
    "pi0_fast": PiPolicyAdapter,
    "groot": GrootPolicyAdapter,
}


def make_policy_adapter(policy_type: str):
    try:
        adapter_cls = _POLICY_ADAPTERS[policy_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_POLICY_ADAPTERS))
        raise ValueError(
            f"Unsupported LeRobot policy_type={policy_type!r}; supported: {supported}"
        ) from exc
    return adapter_cls()
