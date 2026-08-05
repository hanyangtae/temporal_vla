"""Policy client transports for GR00T N1.6 RoboCasa SAFE collection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import zmq
# vla_client 는 scripts/utils 소속 — 러너가 path_setup 으로 sys.path 를 구성한 뒤에만
# import 가능하다. src → scripts 역방향 의존이라 장기적으로는 vla_client 의 src 이동 대상.
from vla_client import VLAClient

from src.collect.schema import (
    _extract_groot_action_vector,
    _extract_safe_action_vector,
    _to_pickleable_numpy,
)
from src.processor.factory import make_groot_robocasa_processors
from src.processor.types import TransitionKey
from src.policies.groot.robocasa.io import (
    http_server_url,
    prepare_groot_robocasa_observation,
)
from src.policies.groot.core.schema import extract_groot_instruction
from src.policies.safe_metadata import normalize_feature_metadata


MsgSerializer: Any | None = None


def _prefer_present(new: Any, old: Any) -> Any:
    return old if new is None else new


def _load_msg_serializer() -> Any:
    global MsgSerializer
    if MsgSerializer is None:
        from gr00t.policy.server_client import MsgSerializer as _MsgSerializer

        MsgSerializer = _MsgSerializer
    return MsgSerializer


def _split_vla_observation(
    processed_obs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    images = {}
    states = {}
    for key, value in processed_obs.items():
        if key.startswith("observation.images."):
            images[key.split("observation.images.", 1)[1]] = value
        elif key.startswith("observation.state."):
            states[key] = value
    instruction = processed_obs.get("task", "")
    return images, states, "" if instruction is None else str(instruction)


class SafeFeatureRecordMixin:
    """Shared SAFE pkl record state for ZMQ and HTTP policy transports."""

    def _init_safe_feature_records(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.task_description: str | None = None
        self.feature_kind: str | None = None
        # docs/04 규약 — serve 응답이 정본 (serve 머신 != 수집기 머신일 수 있음)
        self.machine: str | None = None
        self.ckpt: str | None = None
        self.feature_axes: list[str] | None = None
        self.feature_slice: str | None = None
        self.exported_action_token_count: int | None = None
        self.feature_action_horizon: int | None = None
        self.valid_action_horizon: int | None = None
        self.model_action_horizon: int | None = None
        self.num_inference_timesteps: int | None = None
        self.capture_layers: list[int] | None = None
        self.layer_indices: list[int] | None = None
        self.layer_count: int | None = None
        self.token_count: int | None = None
        # VL(goal) pathway feature metadata (multilayer endpoint --capture-vl 시).
        self.vl_feature_kind: str | None = None
        self.vl_feature_axes: list[str] | None = None
        self.vl_feature_dim: int | None = None

    def _reset_safe_feature_records(self) -> None:
        self.records.clear()
        self.task_description = None
        self.feature_kind = None
        self.machine = None
        self.ckpt = None
        self.feature_axes = None
        self.feature_slice = None
        self.exported_action_token_count = None
        self.feature_action_horizon = None
        self.valid_action_horizon = None
        self.model_action_horizon = None
        self.num_inference_timesteps = None
        self.capture_layers = None
        self.layer_indices = None
        self.layer_count = None
        self.token_count = None
        self.vl_feature_kind = None
        self.vl_feature_axes = None
        self.vl_feature_dim = None

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del options
        self._reset_safe_feature_records()
        return {}

    def _update_safe_feature_metadata(self, payload: dict[str, Any]) -> None:
        metadata = normalize_feature_metadata(payload)
        self.machine = _prefer_present(payload.get("serve_machine"), self.machine)
        self.ckpt = _prefer_present(payload.get("serve_ckpt"), self.ckpt)
        self.feature_kind = _prefer_present(metadata.feature_kind, self.feature_kind)
        self.feature_axes = _prefer_present(metadata.feature_axes, self.feature_axes)
        self.feature_slice = _prefer_present(metadata.feature_slice, self.feature_slice)
        self.exported_action_token_count = _prefer_present(
            metadata.exported_action_token_count, self.exported_action_token_count
        )
        self.feature_action_horizon = _prefer_present(
            metadata.feature_action_horizon, self.feature_action_horizon
        )
        self.valid_action_horizon = _prefer_present(
            metadata.valid_action_horizon, self.valid_action_horizon
        )
        self.model_action_horizon = _prefer_present(
            metadata.model_action_horizon, self.model_action_horizon
        )
        self.num_inference_timesteps = _prefer_present(
            metadata.num_inference_timesteps, self.num_inference_timesteps
        )
        layer_indices = payload.get("layer_indices")
        capture_layers = payload.get("capture_layers", layer_indices)
        normalized_layers = (
            None if capture_layers is None else [int(layer) for layer in capture_layers]
        )
        self.capture_layers = _prefer_present(normalized_layers, self.capture_layers)
        self.layer_indices = _prefer_present(
            None if layer_indices is None else [int(layer) for layer in layer_indices],
            self.layer_indices,
        )
        if self.layer_indices is None:
            self.layer_indices = self.capture_layers
        self.layer_count = _prefer_present(payload.get("layer_count"), self.layer_count)
        self.token_count = _prefer_present(payload.get("token_count"), self.token_count)
        # VL pathway metadata (multilayer --capture-vl). 없으면 그대로 None 유지.
        self.vl_feature_kind = _prefer_present(
            payload.get("vl_feature_kind"), self.vl_feature_kind
        )
        self.vl_feature_axes = _prefer_present(
            payload.get("vl_feature_axes"), self.vl_feature_axes
        )
        self.vl_feature_dim = _prefer_present(
            payload.get("vl_feature_dim"), self.vl_feature_dim
        )


class N16SafeCollectingPolicyClient(SafeFeatureRecordMixin):
    def __init__(
        self,
        host: str,
        port: int,
        endpoint: str = "get_action_with_features",
        timeout_ms: int = 120000,
        inference_seed: int | None = None,
    ):
        self._init_safe_feature_records()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")
        self.endpoint = endpoint
        self.inference_seed = inference_seed

    def call_endpoint(
        self, endpoint: str, data: dict[str, Any] | None = None, requires_input: bool = True
    ) -> Any:
        serializer = _load_msg_serializer()
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        self.socket.send(serializer.to_bytes(request))
        response = serializer.from_bytes(self.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"N1.6 server error: {response['error']}")
        return response

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        filtered = prepare_groot_robocasa_observation(observation, strict=True)
        if self.task_description is None:
            self.task_description = extract_groot_instruction(filtered)

        request_options = dict(options or {})
        call_inference_seed = (
            None if self.inference_seed is None else self.inference_seed + len(self.records)
        )
        if call_inference_seed is not None:
            request_options["inference_seed"] = call_inference_seed
        request: dict[str, Any] = {"observation": filtered}
        if request_options:
            request["options"] = request_options
        response = self.call_endpoint(self.endpoint, request)
        action = response["action"]
        hidden_states = np.asarray(response["hidden_states"])
        if hidden_states.ndim == 4 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states[0]

        record: dict[str, Any] = {
            "hidden_state": torch.from_numpy(np.ascontiguousarray(hidden_states)),
            "action_vector": _extract_safe_action_vector(action),
            "groot_action_vector": _extract_groot_action_vector(action),
            "action": _to_pickleable_numpy(action),
        }
        vl_hidden = response.get("vl_hidden_states")
        if vl_hidden is not None:
            record["vl_hidden_state"] = torch.from_numpy(
                np.ascontiguousarray(np.asarray(vl_hidden))
            )
        self._update_safe_feature_metadata(response)
        self.records.append(record)
        return action, {}


class HttpN16SafeCollectingPolicyClient(SafeFeatureRecordMixin, VLAClient):
    """VLAClient-backed GR00T policy adapter that records SAFE features."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 120.0,
        inference_seed: int | None = None,
    ):
        self._init_safe_feature_records()
        VLAClient.__init__(
            self,
            http_server_url(host, port),
            timeout=timeout_s,
        )
        self.inference_seed = inference_seed
        self.obs_pipeline, self.action_pipeline = make_groot_robocasa_processors(
            strict=True,
            action_mode="chunk",
        )

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        SafeFeatureRecordMixin.reset(self, options)
        VLAClient.reset(self)
        return {}

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        processed = self.obs_pipeline({TransitionKey.OBSERVATION: observation})
        images, states, instruction = _split_vla_observation(
            processed[TransitionKey.OBSERVATION]
        )
        if self.task_description is None:
            self.task_description = instruction
        call_inference_seed = (
            None if self.inference_seed is None else self.inference_seed + len(self.records)
        )
        actions, features, _latency_ms = self.predict_with_features(
            images,
            states or None,
            instruction,
            inference_seed=call_inference_seed,
        )
        if not isinstance(actions, dict):
            raise RuntimeError("HTTP /act_with_features must return sub-keyed actions")

        processed_act = self.action_pipeline({TransitionKey.ACTION: actions})
        action = processed_act[TransitionKey.ACTION]
        hidden_states = np.asarray(features["hidden_states"])
        if hidden_states.ndim == 4 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states[0]

        record: dict[str, Any] = {
            "hidden_state": torch.from_numpy(np.ascontiguousarray(hidden_states)),
            "action_vector": _extract_safe_action_vector(action),
            "groot_action_vector": _extract_groot_action_vector(action),
            "action": _to_pickleable_numpy(action),
        }
        self._update_safe_feature_metadata(features)
        self.records.append(record)
        return action, {}
