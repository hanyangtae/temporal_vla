"""VLA 모델 추론 서버와 통신하는 통일 HTTP 클라이언트.

벤치마크 환경(robocasa, calvin 등)에서 실행되며,
모델 서버(DreamVLA, UP-VLA, X-VLA 등)에 관측값을 보내고 액션을 받는다.

통일 API 규격:
  POST /act
    요청: {
      "observation.images.static": base64png,
      "observation.images.wrist": base64png,
      "observation.state.eef_pos": [float, float, float],
      "observation.state.eef_quat": [float x4],
      "observation.state.joint_pos": [float x7],
      ...
      "task": "str"
    }
    응답: {
      "action": [[float...], ...],   ← 항상 2D
      "latency_ms": float
    }

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


def encode_image(img: np.ndarray) -> str:
    """HxWx3 uint8 numpy → base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_ndarray(b64_str: str) -> np.ndarray:
    """base64(np.save bytes) → ndarray (서버 _encode_ndarray 의 역변환)."""
    return np.load(io.BytesIO(base64.b64decode(b64_str)), allow_pickle=False)


class VLAClient:
    """통일 VLA 추론 클라이언트.

    모든 모델 서버가 동일한 API를 따르므로 클라이언트는 1개로 충분하다.
    """

    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> dict | None:
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
            return None
        except requests.ConnectionError:
            return None

    def wait_until_ready(self, max_wait: float = 180.0, poll_interval: float = 3.0):
        """서버가 준비될 때까지 대기."""
        start = time.time()
        while time.time() - start < max_wait:
            info = self.health_check()
            if info and info.get("status") == "ok":
                return info
            time.sleep(poll_interval)
        raise TimeoutError(f"Server at {self.url} not ready after {max_wait}s")

    def reset(self):
        """에피소드 시작 시 서버 히스토리 초기화."""
        r = requests.post(f"{self.url}/reset", timeout=self.timeout)
        r.raise_for_status()

    def predict(
        self,
        images: dict[str, np.ndarray],
        states: dict[str, np.ndarray] | None = None,
        instruction: str = "",
    ):
        """모델 서버에 액션 예측 요청.

        Args:
            images: 카메라 이름 → HxWx3 uint8 numpy 딕셔너리.
                    키 이름은 "static", "wrist" 등 벤치마크가 정하는 이름.
            states: observation.state.* 키 → numpy 배열 딕셔너리.
                    예: {"observation.state.eef_pos": np.array([...]), ...}
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
        payload = self._build_payload(images, states, instruction)

        t0 = time.time()
        r = requests.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        r.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        return self._parse_actions(r.json()), latency_ms

    def predict_with_features(
        self,
        images: dict[str, np.ndarray],
        states: dict[str, np.ndarray] | None = None,
        instruction: str = "",
    ):
        """SAFE 수집용: /act_with_features 호출. predict() 와 인자 동일.

        Returns:
            (actions, features, latency_ms)

            actions: predict() 와 동일(sub-keyed dict 또는 flat ndarray).
            features: dict | None — 정책이 이번 step 에 새 추론을 돌렸을 때만 채워짐.
              {"hidden_states": ndarray ([K,H,D] flow-matching | [1,n_tokens,D] pi0_fast),
               "feature_kind": str, "feature_axes": list[str],
               "num_inference_timesteps": int|None, ...}
              버퍼된 action 만 반환한 step(queue pop)에서는 None.
        """
        payload = self._build_payload(images, states, instruction)

        t0 = time.time()
        r = requests.post(f"{self.url}/act_with_features", json=payload, timeout=self.timeout)
        r.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        result = r.json()
        actions = self._parse_actions(result)

        features = None
        if result.get("has_feature"):
            features = {"hidden_states": _decode_ndarray(result["hidden_states_b64"])}
            for k in (
                "feature_kind",
                "feature_axes",
                "num_inference_timesteps",
                "action_horizon",
                "feature_dim",
                "n_action_tokens",
            ):
                if k in result:
                    features[k] = result[k]
        return actions, features, latency_ms

    @staticmethod
    def _build_payload(
        images: dict[str, np.ndarray],
        states: dict[str, np.ndarray] | None,
        instruction: str,
    ) -> dict:
        payload: dict = {}
        for k, v in images.items():
            payload[f"observation.images.{k}"] = encode_image(v)
        payload["task"] = instruction
        if states is not None:
            for k, v in states.items():
                payload[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return payload

    @staticmethod
    def _parse_actions(result: dict):
        # Sub-keyed 포맷 감지: "action.*" 키가 하나라도 있으면 dict 반환
        action_keys = [k for k in result if k.startswith("action.")]
        if action_keys:
            action_dict = {}
            for k in action_keys:
                arr = np.array(result[k], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                action_dict[k] = arr
            return action_dict

        # 하위호환: flat "action" 배열
        actions = np.array(result["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        return actions
