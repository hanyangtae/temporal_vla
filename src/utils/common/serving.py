"""Shared serving helpers: API responses + serve-script bootstrap."""

from __future__ import annotations

import argparse
from typing import Any


def setup_serve_logging(name: str) -> None:
    """모듈 로거를 구성한다. create_module_logger 사용, 불가 시 basicConfig 폴백.

    기존 serve 스크립트들이 main() 첫머리에서 반복하던 try/except 블록과
    동일한 동작.
    """
    import logging

    try:
        from src.utils.common.logger import create_module_logger

        create_module_logger(name)
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def add_server_args(
    parser: argparse.ArgumentParser,
    *,
    default_port: int,
    host_default: str = "0.0.0.0",
) -> None:
    """통일 serve 스크립트 공통 --host/--port 인자를 등록."""
    parser.add_argument("--host", type=str, default=host_default)
    parser.add_argument("--port", type=int, default=default_port)


def run_uvicorn(app: Any, args: argparse.Namespace) -> None:
    """uvicorn.run(app, host=args.host, port=args.port). uvicorn 은 지연 import."""
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


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
