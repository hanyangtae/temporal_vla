"""VLA 모델 추론 서버와 통신하는 통일 HTTP 클라이언트.

벤치마크 환경(robocasa, calvin 등)에서 실행되며,
모델 서버(GR00T, pi0.5, UP-VLA, X-VLA 등)에 관측값을 보내고 액션을 받는다.

통일 API 규격:
  POST /act
    요청: {
      "observation.images.static": base64png,
      "observation.images.wrist": base64png,
      "observation.state.eef_pos": [float, float, float],
      "observation.state.eef_quat": [float x4],
      "observation.state.joint_pos": [float x7],
      "observation.state.eef_pos_rel": [float x3],      # GR00T RoboCasa
      "observation.state.eef_quat_rel": [float x4],     # GR00T RoboCasa
      "observation.state.base_position": [float x3],    # GR00T RoboCasa
      "observation.state.base_rotation": [float x4],    # GR00T RoboCasa
      ...
      "task": "str"
    }
    응답:
      신규 sub-keyed:
      {
        "action.eef_pos": [[float x3], ...],
        "action.eef_axisangle": [[float x3], ...],
        "action.gripper": [[float x1], ...],
        "latency_ms": float
      }
      하위호환 flat:
      {
        "action": [[float...], ...],
        "latency_ms": float
      }

  POST /act_with_features
    요청: /act 와 동일
    응답: /act sub-keyed 응답 + features.hidden_states feature blob + metadata
          + optional features.vl_hidden_states feature blob

  POST /reset   ← 에피소드 시작 시 히스토리 초기화 (모델이 필요 없으면 no-op)
  GET  /health  ← 서버 상태 확인 + feature 정보
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import requests
from PIL import Image

from src.policies.safe_metadata import normalize_feature_metadata
from src.utils.common.feature_blob import decode_feature_blob, decode_legacy_feature_array


def encode_image(img: np.ndarray) -> str:
    """HxWx3 uint8 numpy → base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class VLAClient:
    """통일 VLA 추론 클라이언트.

    모든 모델 서버가 동일한 API를 따르므로 클라이언트는 1개로 충분하다.
    """

    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health_check(self, timeout: float = 5.0) -> dict | None:
        try:
            r = requests.get(f"{self.url}/health", timeout=timeout)
            if r.status_code == 200:
                return r.json()
            return None
        except (requests.RequestException, ValueError):
            return None

    def wait_until_ready(self, max_wait: float = 180.0, poll_interval: float = 3.0):
        """서버가 준비될 때까지 대기."""
        deadline = time.time() + max_wait
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            info = self.health_check(timeout=min(5.0, remaining))
            if info and info.get("status") == "ok":
                return info

            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval, remaining))
        raise TimeoutError(f"Server at {self.url} not ready after {max_wait}s")

    def reset(self):
        """에피소드 시작 시 서버 히스토리 초기화."""
        r = requests.post(f"{self.url}/reset", timeout=self.timeout)
        r.raise_for_status()

    def _build_payload(
        self,
        images: dict[str, np.ndarray],
        states: dict[str, np.ndarray] | None,
        instruction: str,
        inference_seed: int | None = None,
    ) -> dict:
        payload: dict = {}
        for k, v in images.items():
            payload[f"observation.images.{k}"] = encode_image(v)
        payload["task"] = instruction
        if inference_seed is not None:
            payload["inference_seed"] = int(inference_seed)
        if states is not None:
            for k, v in states.items():
                payload[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return payload

    def _post_and_decode(self, path: str, payload: dict) -> tuple[dict, float]:
        t0 = time.time()
        r = requests.post(f"{self.url}{path}", json=payload, timeout=self.timeout)
        latency_ms = (time.time() - t0) * 1000

        try:
            result = r.json()
        except ValueError:
            result = None

        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            if isinstance(result, dict):
                message = result.get("error") or result.get("detail")
                if message:
                    raise RuntimeError(f"VLA server error: {message}") from exc
            raise

        if not isinstance(result, dict):
            raise RuntimeError("VLA server response must be a JSON object")

        if "error" in result:
            raise RuntimeError(f"VLA server error: {result['error']}")

        return result, latency_ms

    def _parse_action(self, result: dict) -> dict | np.ndarray:
        """sub-keyed dict 우선, 없으면 flat ``action`` array."""
        action_keys = [k for k in result if k.startswith("action.")]
        if action_keys:
            action_dict = {}
            for k in action_keys:
                arr = np.array(result[k], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                action_dict[k] = arr
            return action_dict

        if "action" not in result:
            keys = ", ".join(sorted(result)) or "<empty>"
            raise RuntimeError(f"VLA server response missing action keys: {keys}")

        actions = np.array(result["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        return actions

    def predict(
        self,
        images: dict[str, np.ndarray],
        states: dict[str, np.ndarray] | None = None,
        instruction: str = "",
        inference_seed: int | None = None,
    ):
        """모델 서버에 액션 예측 요청.

        Args:
            images: 카메라 이름 → HxWx3 uint8 numpy 딕셔너리.
                    키 이름은 "static", "wrist" 등 벤치마크가 정하는 이름.
            states: observation.state.* 키 → numpy 배열 딕셔너리.
                    예: {"observation.state.eef_pos": np.array([...]), ...}
                    GR00T RoboCasa는 eef_pos_rel/eef_quat_rel/base_* 키를 사용.
                    None이면 state 관련 키를 전송하지 않음.
            instruction: 태스크 언어 지시문.

        Returns:
            (actions, latency_ms)

            actions 형식은 서버 응답에 따라 결정:
              - Sub-keyed (신규): dict[str, np.ndarray]
                예: {"action.eef_pos": [N, 3], "action.eef_rot6d": [N, 6], ...}
              - Flat (하위호환): np.ndarray [N, action_dim]
                예: [N, 7]
        """
        payload = self._build_payload(images, states, instruction, inference_seed)
        result, latency_ms = self._post_and_decode("/act", payload)
        return self._parse_action(result), latency_ms

    def predict_with_features(
        self,
        images: dict[str, np.ndarray],
        states: dict[str, np.ndarray] | None = None,
        instruction: str = "",
        inference_seed: int | None = None,
        extra_payload: dict | None = None,
    ):
        """``/act_with_features`` 호출. action sub-key dict + features dict 반환.

        Returns:
            (actions, features, latency_ms)
              actions: ``predict`` 와 같은 sub-keyed dict[str, np.ndarray]
                       (서버가 sub-keyed 응답을 보장 — GR00T HTTP serve 한정)
              features: dict | None
                hidden_states: np.ndarray (``[B, K, H, D]`` 등 서버 shape 그대로)
                kind: str   (e.g. "groot_n16_dit_valid_action_tokens_pre_velocity")
                axes: list[str]
                slice: str  ("valid" | "all")
                exported_action_token_count: int
                feature_action_horizon: int
                valid_action_horizon: int
                model_action_horizon: int
                num_inference_timesteps: int
                capture_layers / layer_count / token_count, optional for DiT block residual features
                vl_hidden_states: np.ndarray, optional
                vl_feature_kind / vl_feature_axes / vl_feature_dim, optional
        """
        payload = self._build_payload(images, states, instruction, inference_seed)
        if extra_payload:
            payload.update(extra_payload)
        result, latency_ms = self._post_and_decode("/act_with_features", payload)
        actions = self._parse_action(result)

        blob = result.get("features.hidden_states")
        if isinstance(blob, dict):
            metadata = normalize_feature_metadata(result)
            features = {
                "hidden_states": decode_feature_blob(blob),
                "kind": metadata.feature_kind,
                "axes": metadata.feature_axes,
                "slice": metadata.feature_slice,
                "exported_action_token_count": metadata.exported_action_token_count,
                "feature_action_horizon": metadata.feature_action_horizon,
                "valid_action_horizon": metadata.valid_action_horizon,
                "model_action_horizon": metadata.model_action_horizon,
                "num_inference_timesteps": metadata.num_inference_timesteps,
            }
            for key in (
                "capture_layers",
                "layer_indices",
                "layer_count",
                "token_count",
                "feature_dim",
                "capture_token_mode",
            ):
                if key in result:
                    features[key] = result.get(key)
            vl_blob = result.get("features.vl_hidden_states")
            if isinstance(vl_blob, dict):
                features["vl_hidden_states"] = decode_feature_blob(vl_blob)
                features["vl_feature_kind"] = result.get("vl_feature_kind")
                features["vl_feature_axes"] = result.get("vl_feature_axes")
                features["vl_feature_dim"] = result.get("vl_feature_dim")
            return actions, features, latency_ms

        if result.get("has_feature") is False:
            return actions, None, latency_ms

        legacy_blob = result.get("hidden_states_b64")
        if not isinstance(legacy_blob, str):
            raise RuntimeError(
                "/act_with_features response missing features.hidden_states blob"
            )

        metadata = normalize_feature_metadata(result)
        features = {
            "hidden_states": decode_legacy_feature_array(legacy_blob),
            "kind": metadata.feature_kind,
            "axes": metadata.feature_axes,
            "slice": metadata.feature_slice,
            "exported_action_token_count": metadata.exported_action_token_count,
            "feature_action_horizon": metadata.feature_action_horizon,
            "valid_action_horizon": metadata.valid_action_horizon,
            "model_action_horizon": metadata.model_action_horizon,
            "num_inference_timesteps": metadata.num_inference_timesteps,
        }
        for key in (
            "capture_layers",
            "layer_indices",
            "layer_count",
            "token_count",
            "feature_dim",
            "capture_token_mode",
        ):
            if key in result:
                features[key] = result.get(key)
        return actions, features, latency_ms
