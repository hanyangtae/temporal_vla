"""RoboCasa scenario replay helpers for GR00T evaluation and SAFE wiring."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np


EP_META_MANIFEST_FORMAT = "robocasa_ep_meta_manifest.v1"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def ep_meta_manifest_stem(env_name: str, scenario_seed: int) -> str:
    safe_env_name = "".join(c if c.isalnum() else "_" for c in env_name).strip("_")
    return f"{safe_env_name}--seed{scenario_seed}"


def ep_meta_manifest_path(
    ep_meta_dir: Path, env_name: str, scenario_seed: int
) -> Path:
    return ep_meta_dir / f"{ep_meta_manifest_stem(env_name, scenario_seed)}.json"


def load_ep_meta_manifest(
    path: Path, env_name: str, scenario_seed: int
) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if payload.get("env_name") != env_name:
        raise ValueError(f"ep_meta manifest env mismatch: {path}")
    if payload.get("scenario_seed") != scenario_seed:
        raise ValueError(f"ep_meta manifest seed mismatch: {path}")
    ep_meta = payload.get("ep_meta")
    if not isinstance(ep_meta, dict):
        raise ValueError(f"ep_meta manifest is missing ep_meta dict: {path}")
    return ep_meta


def write_ep_meta_manifest(
    path: Path,
    *,
    env_name: str,
    scenario_seed: int,
    ep_meta: dict[str, Any],
    robocasa_env_source: str | None = None,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": EP_META_MANIFEST_FORMAT,
        "env_name": env_name,
        "scenario_seed": scenario_seed,
        "ep_meta": json_safe(ep_meta),
    }
    if robocasa_env_source is not None:
        payload["robocasa_env_source"] = robocasa_env_source
    with path.open("w") as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        )


def find_env_method(env: Any, method_name: str):
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        method = getattr(current, method_name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)
    raise RuntimeError(f"Could not find RoboCasa env method {method_name}()")


def get_robocasa_ep_meta(env: Any) -> dict[str, Any]:
    return json_safe(find_env_method(env, "get_ep_meta")())


def set_robocasa_ep_meta(env: Any, ep_meta: dict[str, Any]) -> None:
    find_env_method(env, "set_ep_meta")(deepcopy(ep_meta))
