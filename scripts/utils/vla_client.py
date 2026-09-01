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
        # 마지막 /act_with_features 응답의 failure detector 신호 (serve 가 안 보내면 None)
        self.last_failure: dict | None = None
        # 마지막 /act_with_features 응답의 cluster phase 자체판정
        # (serve --cluster-phase-bundle 전용 — 안 보내면 None). detector 유무와 독립이라
        # last_failure 와 별도 속성으로 둔다(detector 없는 순수 라벨링 수집도 읽을 수 있게).
        self.last_cluster: dict | None = None

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
        # serve 의 detector 상태도 /reset 에서 초기화된다 — 클라이언트 캐시도 같이 비운다.
        self.last_failure = None
        self.last_cluster = None

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

        # online failure detector (serve --failure-detector): plain JSON 스칼라.
        # feature blob 유무와 무관하게 오므로(--no-features eval 은 blob 억제 + score 만)
        # features dict 가 아니라 클라이언트 속성으로 노출한다.
        # per-step 게이팅(docs/steering/47): serve 가 1차 무개입 pass 점수(y_t)로 발화를
        # 판정하고 발화 시 DiT 만 2차 재실행한다. 그 감사 필드(post 점수·발화 flag·op·
        # 2차 seed)는 있을 때만 실어 준다 — 기존 arm 응답 스키마는 불변(하위 호환).
        if "features.failure_score" in result:
            failure = {
                "score": result.get("features.failure_score"),
                "fired": bool(result.get("features.failure_fired")),
                "delta": result.get("features.failure_delta"),
                "step": result.get("features.failure_step"),
            }
            for src_key, dst_key in (
                ("features.failure_score_post", "score_post"),
                ("features.perstep_fired", "perstep_fired"),
                ("features.perstep_op", "perstep_op"),
                ("features.perstep_seed2", "perstep_seed2"),
                ("features.perstep_gate_skipped", "gate_skipped"),
                ("features.perstep_debug_max_action_diff", "debug_max_action_diff"),
                # cluster phase 자체판정(serve --cluster-phase-bundle): "c0".."c7" + 거리.
                ("features.perstep_cluster", "cluster"),
                ("features.perstep_cluster_dist", "cluster_dist"),
                # best-of-N 재샘플(rsn_*): 후보 수·LLR·선택 idx·기각 사유.
                ("features.perstep_cand_n", "perstep_cand_n"),
                ("features.perstep_cand_llr", "perstep_cand_llr"),
                ("features.perstep_cand_sel", "perstep_cand_sel"),
                ("features.perstep_cand_reject", "perstep_cand_reject"),
                # LLR 선별 불가 사유(fallback=후보 0 개입 — gate_skipped 와 다름: 개입은 일어남)
                ("features.perstep_cand_entry", "perstep_cand_entry"),
                ("features.perstep_cand_logs", "perstep_cand_logs"),
                ("features.perstep_llr_fallback", "perstep_llr_fallback"),
                # 2차 pass 소요(ms)·후보별 소요·미등록 phase 대체개입 사유
                ("features.perstep_rerun_ms", "perstep_rerun_ms"),
                ("features.perstep_cand_ms", "perstep_cand_ms"),
                ("features.perstep_fallback", "perstep_fallback"),
            ):
                if src_key in result:
                    failure[dst_key] = result.get(src_key)
            self.last_failure = failure
        else:
            self.last_failure = None

        # cluster phase 는 detector 없이도 올 수 있으므로 별도 속성으로도 노출한다.
        if "features.perstep_cluster" in result:
            self.last_cluster = {
                "cluster": result.get("features.perstep_cluster"),
                "cluster_dist": result.get("features.perstep_cluster_dist"),
            }
        else:
            self.last_cluster = None

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
            # online phase readout (serve --phase-readout): plain JSON, pass through as-is.
            if "features.phase" in result:
                features["phase"] = result.get("features.phase")
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
