"""VLA 모델 추론 서버와 통신하는 통일 HTTP 클라이언트.

벤치마크 환경(robocasa, calvin 등)에서 실행되며,
모델 서버(DreamVLA, UP-VLA, X-VLA 등)에 관측값을 보내고 액션을 받는다.

통일 API 규격:
  POST /act
    요청: {
      "images": {"static": base64png, "wrist": base64png, ...},
      "state": [float...],
      "instruction": "str"
    }
    응답: {
      "actions": [[float...], ...],   ← 항상 2D
      "latency_ms": float
    }

  POST /reset   ← 에피소드 시작 시 히스토리 초기화 (모델이 필요 없으면 no-op)
  GET  /health  ← 서버 상태 확인
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
        state: np.ndarray | list[float] | None = None,
        instruction: str = "",
    ) -> tuple[np.ndarray, float]:
        """모델 서버에 액션 예측 요청.

        Args:
            images: 카메라 이름 → HxWx3 uint8 numpy 딕셔너리.
                    키 이름은 "static", "wrist" 등 벤치마크가 정하는 이름.
            state: proprioceptive state (1-D float). None이면 전송하지 않음.
            instruction: 태스크 언어 지시문.

        Returns:
            (actions [N, action_dim], latency_ms)
            N은 모델의 action prediction steps (1 이상).
        """
        payload = {
            "images": {k: encode_image(v) for k, v in images.items()},
            "instruction": instruction,
        }
        if state is not None:
            if isinstance(state, np.ndarray):
                payload["state"] = state.tolist()
            else:
                payload["state"] = state

        t0 = time.time()
        r = requests.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        r.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        result = r.json()
        actions = np.array(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        return actions, latency_ms
