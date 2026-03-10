"""VLA 모델 추론 서버와 통신하는 HTTP 클라이언트.

robocasa 환경에서 실행되며, X-VLA/DreamVLA 서버에 관측값을 보내고 액션을 받는다.
서버가 별도 Docker 컨테이너에서 돌고 있어도 network_mode: host이므로 localhost로 접근.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import requests


class VLAClient:
    """HTTP 기반 VLA 추론 클라이언트."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def wait_until_ready(self, max_wait: float = 120.0, poll_interval: float = 3.0):
        """서버가 준비될 때까지 대기."""
        start = time.time()
        while time.time() - start < max_wait:
            if self.health_check():
                return True
            time.sleep(poll_interval)
        raise TimeoutError(f"Server at {self.url} not ready after {max_wait}s")


class XVLAClient(VLAClient):
    """X-VLA 서버 클라이언트."""

    def __init__(self, url: str = "http://localhost:8100", timeout: float = 30.0):
        super().__init__(url, timeout)

    def predict(
        self,
        image0: np.ndarray,
        proprio: np.ndarray,
        instruction: str,
        domain_id: int = 0,
        steps: int = 10,
        image1: Optional[np.ndarray] = None,
        image2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        payload = {
            "proprio": proprio.tolist(),
            "language_instruction": instruction,
            "image0": image0.tolist(),
            "domain_id": domain_id,
            "steps": steps,
        }
        if image1 is not None:
            payload["image1"] = image1.tolist()
        if image2 is not None:
            payload["image2"] = image2.tolist()

        t0 = time.time()
        r = requests.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        r.raise_for_status()
        latency = time.time() - t0

        result = r.json()
        actions = np.array(result["action"], dtype=np.float32)
        return actions, latency


class DreamVLAClient(VLAClient):
    """DreamVLA 서버 클라이언트.

    서버는 에피소드 단위 히스토리를 유지하므로,
    에피소드 시작 시 반드시 reset()을 호출해야 한다.
    """

    def __init__(self, url: str = "http://localhost:8200", timeout: float = 30.0):
        super().__init__(url, timeout)

    def reset(self):
        """에피소드 시작 시 서버 히스토리 초기화."""
        r = requests.post(f"{self.url}/reset", timeout=self.timeout)
        r.raise_for_status()

    def predict(
        self,
        static_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> tuple[np.ndarray, float]:
        """
        Args:
            static_image: HxWx3 uint8, robot0_agentview_left
            wrist_image:  HxWx3 uint8, robot0_eye_in_hand
            state:        1-D float array (proprioceptive state)
            instruction:  task language string
        Returns:
            (action [7], latency_ms)
        """
        payload = {
            "images": {
                "robot0_agentview_left": static_image.tolist(),
                "robot0_eye_in_hand": wrist_image.tolist(),
            },
            "state": state.tolist(),
            "language_instruction": instruction,
        }

        t0 = time.time()
        r = requests.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        r.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        result = r.json()
        action = np.array(result["action"], dtype=np.float32)
        return action, latency_ms
