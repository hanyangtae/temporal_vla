"""Small shared serving response helpers."""

from __future__ import annotations

from typing import Any


def policy_status(policy: Any | None) -> str:
    return "ok" if policy is not None else "not_loaded"


def reset_policy(policy: Any | None) -> dict[str, str]:
    if policy is not None:
        policy.reset()
    return {"status": "reset"}


def health_response(
    *,
    policy: Any | None,
    model: str,
    profile: Any,
    n_action_steps: int,
    action_type: str,
    action_keys: list[str],
    **extra: Any,
) -> dict[str, Any]:
    response = {
        "status": policy_status(policy),
        "model": model,
        "profile": profile.name,
        "n_action_steps": n_action_steps,
        "action_type": action_type,
        "action_keys": action_keys,
    }
    response.update(extra)
    return response
