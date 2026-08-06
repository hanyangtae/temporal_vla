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


def serve_provenance(profile: Any | None = None) -> dict[str, Any]:
    """docs/04 규약 — rollout 인덱스의 `machine`·`ckpt` 원천.

    **serve 가 정본이다.** serve 가 도는 머신이 수집기와 다를 수 있어(컨테이너·원격
    serve) 클라이언트가 자기 호스트명을 쓰면 틀린 값이 기록된다. HTTP(`/health`) 든
    ZMQ(응답 dict) 든 전송 방식과 무관하게 이 헬퍼를 쓴다.

    왜 필요한가:
      - machine: 머신이 다르면 hidden state 원소 93% 가 갈리고 개별 판정이 12.7%
        뒤집힌다(docs/04 §3.2 실측). 좌표 경로의 `<machine>` 층이자 층화 요인이다.
      - ckpt: `model_family` 는 계열명이라 베이스와 파인튜닝을 구분하지 못한다.
        프로파일명이 실제 체크포인트 식별자다.

    **GPU 는 넣지 않는다** — 같은 머신이면 GPU 가 달라도 bitwise 동일(§3.2)하므로
    GPU 를 포함하면 같은 칸이 GPU 별로 갈려 병렬 배정이 막힌다. 머신명만 기록한다.

    2026-08 정리에서 이 둘이 없어 activation 526 판의 머신을 영구히 잃었다.
    """
    import socket

    return {
        "serve_machine": socket.gethostname(),
        "serve_ckpt": getattr(profile, "name", None),
    }


def collector_provenance(server_payload: dict[str, Any] | None) -> dict[str, Any]:
    """serve 응답(HTTP /health 또는 ZMQ 응답)에서 `machine`·`ckpt` 를 뽑는다.

    serve 가 값을 안 주면(구 버전 serve) 키는 만들되 None 으로 둔다 —
    인덱서 계약을 지키고, **추측으로 채우지 않는다**(docs/04 §5).
    """
    body = server_payload or {}
    return {
        "machine": body.get("serve_machine"),
        "ckpt": body.get("serve_ckpt"),
    }


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
