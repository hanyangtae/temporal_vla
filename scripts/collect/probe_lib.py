#!/usr/bin/env python3
# 유래: exp5-4 (2026-08-10 기능명 재배치 — docs/review/RENAME_PLAN.md)
"""exp5-4 probe 경로 — t=0 관측 1개로 seed 별 record 0 활성만 캡처.

구 `collect/http_feature_collect.py` 에서 분리(2026-07-31). 그 파일은 라운드마다
스위치가 얹혀 1200줄을 넘었고, 이 probe 경로 205줄은 exp5-4 전용이라 rollout·저장
본류와 섞일 이유가 없다. 호출 계약은 유지한다 — `http_feature_collect.py` 가
`--probe-seeds/--probe-out` 을 받아 여기 `run_probe()` 로 위임하므로
`probe_collect.sh` 는 수정 불필요.

exp5-4 라운드 자체는 "노이즈 1-step 선별 = seed 주효과" 로 종결됐지만
(docs/steering/37), 여기 정의된 `_ProbeClient` / `_obs_hash` /
`_flatten_action_chunk` 는 같은 디렉토리의 **무결성 도구가 계속 쓴다**:
  - `smoke_probe.py`      — 캡처 hook 이 추론을 바꾸는가 · 서버 상태 오염 · 서버 간 차이
  - `check_probe_identity.py` — probe 활성 ↔ rollout record 0 활성 bit 동일성
따라서 라운드 종결과 무관하게 보존 대상이다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# 본류 수집 모듈에서 재사용 — probe 는 같은 env 생성·관측 변환·serve 지문 경로를 탄다.
from http_feature_collect import (
    _get_serve_identity,
    _lerobot_action_to_official_chunk,
    _normalize_instruction,
    _prefer_present,
    make_env,
    official_obs_to_lerobot_inputs,
)
from src.policies.groot.robocasa.scenario_replay import get_robocasa_ep_meta, json_safe
from vla_client import VLAClient

class _ProbeClient(VLAClient):
    """probe 전용 최소 클라이언트.

    수집용 ``N15LerobotHttpFeatureClient`` 를 쓰지 않는다 — records/n_calls 계수와
    get_action 의 ``call_inference_seed = seed_base + calls_done`` 가산 경로를 아예
    타지 않기 위해서다. ``_post_and_decode`` 만 얇게 감싸 서버의 inference_seed echo
    를 검증한다.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_result: dict[str, Any] | None = None

    def _post_and_decode(self, endpoint, payload):
        result, latency_ms = super()._post_and_decode(endpoint, payload)
        self.last_result = result
        return result, latency_ms


def _obs_hash(images: dict[str, Any], states: dict[str, Any] | None) -> str:
    """probe 입력(이미지+state)의 sha256 — rollout t=0 입력과 동일성 대조용."""
    import hashlib

    h = hashlib.sha256()
    for key in sorted(images):
        arr = np.ascontiguousarray(np.asarray(images[key]))
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    for key in sorted(states or {}):
        arr = np.ascontiguousarray(np.asarray(states[key], dtype=np.float64))
        h.update(key.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _flatten_action_chunk(action: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """sub-keyed action chunk → ([n_steps, dim] float32, 사용 키 순서)."""
    chunk = _lerobot_action_to_official_chunk(action)
    keys = sorted(chunk)
    if not keys:
        return np.zeros((0, 0), dtype=np.float32), []
    pieces = []
    for key in keys:
        arr = np.asarray(chunk[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        pieces.append(arr)
    return np.concatenate(pieces, axis=1), keys


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """exp5-4 probe-only: t=0 관측 1개로 seed 별 record 0 활성만 캡처하고 종료.

    rollout 을 돌리지 않는다 — env.step / labeler.step / pkl·json 산출 경로를 전혀 타지
    않으며, 후보 seed 마다 ``/reset`` 1회 + ``predict_with_features`` 단일 호출만 한다
    (collector policy wrapper 의 get_action 경유 금지 — calls_done 가산 오염 방지).

    seed s 의 활성 = 같은 scene 을 --inference-seed s 로 수집했을 때의 record 0 과 동일:
    수집 경로의 call_inference_seed = inference_seed + calls_done 이고 record 0 은
    calls_done=0 이라 정확히 s 이다.
    """
    raw_seeds = [tok.strip() for tok in str(args.probe_seeds).split(",")]
    probe_seeds = [int(tok) for tok in raw_seeds if tok != ""]
    if not probe_seeds:
        raise ValueError("--probe-seeds 가 비어 있음")
    if len(set(probe_seeds)) != len(probe_seeds):
        raise ValueError(f"--probe-seeds 에 중복: {probe_seeds}")
    if args.seed is None:
        raise ValueError("probe 모드는 --seed (scenario seed) 필수")
    if args.n_action_steps <= 0:
        raise ValueError(f"--n_action_steps must be positive: {args.n_action_steps}")
    out_path = Path(args.probe_out)
    if out_path.suffix != ".npz":
        raise ValueError(f"--probe-out 은 .npz 경로여야 함: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = _ProbeClient(args.vla_server, timeout=args.timeout)
    if args.wait_ready:
        client.wait_until_ready(max_wait=args.timeout)
    serve_identity = _get_serve_identity(args.vla_server)

    env = make_env(
        args.task,
        args.split,
        env_name=args.env_name,
        scenario_seed=args.seed,
        video_dir=None,          # 영상 wrapper 미사용 (렌더 비용·산출물 없음)
        video_fps=args.video_fps,
        steps_per_render=args.steps_per_render,
        overlay_text=False,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )
    try:
        obs, _info = env.reset(seed=args.seed)
        ep_meta = get_robocasa_ep_meta(env)
        images, states, instruction = official_obs_to_lerobot_inputs(obs)
        canonical = getattr(args, "canonical_instruction", None)
        if canonical is not None and (
            _normalize_instruction(instruction) != _normalize_instruction(canonical)
        ):
            raise RuntimeError(
                "probe scene instruction mismatch: "
                f"{instruction!r} != {canonical!r} (scene seed {args.seed})"
            )
        obs_hash = _obs_hash(images, states)
        hidden: list[np.ndarray] = []
        chunks: list[np.ndarray] = []
        chunk_keys: list[str] = []
        seed_echo: list[int] = []
        capture_layers = None
        feature_axes = None
        capture_token_mode = None
        for seed in probe_seeds:
            # 후보마다 서버 히스토리 초기화 후 단일 호출 (get_action 경유 없음).
            client.reset()
            actions, features, _latency_ms = client.predict_with_features(
                images, states, instruction, inference_seed=int(seed)
            )
            raw = client.last_result or {}
            if "inference_seed" in raw:
                echoed = int(raw["inference_seed"])
                if echoed != int(seed):
                    raise RuntimeError(
                        f"probe: 서버 inference_seed echo 불일치 {echoed} != {seed}"
                    )
                seed_echo.append(echoed)
            else:
                # echo 미지원 serve — 요청 payload 값을 그대로 기록해 감사 가능하게 둔다.
                seed_echo.append(-1)
                print(f"[probe] serve 가 inference_seed 를 echo 하지 않음 — 요청값 {seed} 기록")
            if not features or features.get("hidden_states") is None:
                raise RuntimeError(
                    "probe: /act_with_features 가 features.hidden_states 를 반환하지 않음 "
                    "(serve 를 --groot-dit-capture-layers 로 띄웠는지 확인)"
                )
            arr = np.asarray(features["hidden_states"], dtype=np.float32)
            if arr.ndim == 5 and arr.shape[0] == 1:
                arr = arr[0]  # 혹시 batch 축이 있으면 제거 ([1,L,K,T,D] -> [L,K,T,D])
            if hidden and arr.shape != hidden[0].shape:
                raise RuntimeError(
                    f"probe: seed {seed} 활성 shape {arr.shape} != {hidden[0].shape}"
                )
            hidden.append(arr)
            chunk, keys = _flatten_action_chunk(actions)
            if chunks and chunk.shape != chunks[0].shape:
                raise RuntimeError(
                    f"probe: seed {seed} action chunk shape {chunk.shape} != {chunks[0].shape}"
                )
            if chunk_keys and keys != chunk_keys:
                raise RuntimeError("probe: action sub-key 구성이 요청 간 변경됨")
            chunk_keys = keys
            chunks.append(chunk)
            layers = features.get("capture_layers", features.get("layer_indices"))
            if layers is not None:
                layers = [int(layer) for layer in layers]
                if capture_layers is not None and layers != capture_layers:
                    raise RuntimeError("probe: capture_layers 가 요청 간 변경됨")
                capture_layers = layers
            feature_axes = _prefer_present(features.get("axes"), feature_axes)
            capture_token_mode = _prefer_present(
                features.get("capture_token_mode"), capture_token_mode
            )
        stacked = np.stack(hidden, axis=0)  # [k, L, K, T, D]
        action_chunk = np.stack(chunks, axis=0)  # [k, n_steps, dim]
        np.savez(
            out_path,
            seeds=np.asarray(probe_seeds, dtype=np.int64),
            hidden=stacked,
            action_chunk=action_chunk,
            action_chunk_keys=np.asarray(json.dumps(chunk_keys)),
            obs_hash=np.asarray(obs_hash),
            seed_echo=np.asarray(seed_echo, dtype=np.int64),
            capture_layers=np.asarray(capture_layers or [], dtype=np.int64),
            scenario_seed=np.asarray(args.seed, dtype=np.int64),
            instruction=np.asarray(instruction),
            # 문자열 목록은 json 으로 — object dtype 은 load 시 allow_pickle 을 강제한다.
            feature_axes=np.asarray(json.dumps(list(feature_axes or []))),
            capture_token_mode=np.asarray(capture_token_mode or ""),
            env_name=np.asarray(args.env_name),
            n_action_steps=np.asarray(args.n_action_steps, dtype=np.int64),
            serve_identity=np.asarray(json.dumps(json_safe(serve_identity))),
            ep_meta=np.asarray(json.dumps(json_safe(ep_meta), ensure_ascii=False)),
        )
        print(
            f"probe wrote {out_path}: scene={args.seed} seeds={probe_seeds} "
            f"hidden={stacked.shape} action_chunk={action_chunk.shape} "
            f"obs_hash={obs_hash[:16]} layers={capture_layers} instr={instruction!r}"
        )
        return {
            "probe": {
                "path": str(out_path),
                "scenario_seed": args.seed,
                "seeds": probe_seeds,
                "obs_hash": obs_hash,
                "shape": list(stacked.shape),
                "capture_layers": capture_layers,
                "instruction": instruction,
            }
        }
    finally:
        env.close()
