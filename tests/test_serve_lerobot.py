"""serve_lerobot.py 유닛 테스트 (TDD).

Docker 없이 로컬에서 실행 가능. serve_lerobot.py 구현 전에 인터페이스를 명세한다.

실행:
  python3 -m pytest tests/test_serve_lerobot.py -v
"""

from __future__ import annotations

import base64
import io
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from serve_lerobot import STATE_KEY_ORDER, _build_remap_config, parse_payload

# ─── 헬퍼 ────────────────────────────────────────────────────────────────────


def _make_b64_image(h: int = 200, w: int = 200) -> str:
    """HxWx3 uint8 numpy → base64 PNG string."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_full_state_payload() -> dict:
    """STATE_KEY_ORDER 전체 키를 포함한 payload."""
    dims = {
        "observation.state.eef_pos": [1.0, 2.0, 3.0],
        "observation.state.eef_euler": [0.1, 0.2, 0.3],
        "observation.state.gripper_opening": [0.5],
        "observation.state.joint_pos": list(range(7, 14, 1)),
        "observation.state.gripper_action": [1.0],
    }
    return dims


# ─── parse_payload ────────────────────────────────────────────────────────────


class TestParsePayloadImage(unittest.TestCase):
    """이미지 변환: base64 PNG → float CHW tensor."""

    def test_static_image_shape_and_dtype(self):
        """base64 PNG → [1, 3, H, W] float32 tensor."""
        payload = {"observation.images.static": _make_b64_image(200, 200)}
        batch = parse_payload(payload)

        t = batch["observation.images.static"]
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, (1, 3, 200, 200))
        self.assertEqual(t.dtype, torch.float32)

    def test_image_range_0_to_1(self):
        """픽셀값 [0, 255] uint8 → [0.0, 1.0] float."""
        # 순수 흰 이미지 (255) → 1.0
        white = np.full((4, 4, 3), 255, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(white).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        batch = parse_payload({"observation.images.static": b64})
        self.assertAlmostEqual(batch["observation.images.static"].max().item(), 1.0)

        # 순수 검정 이미지 (0) → 0.0
        black = np.zeros((4, 4, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(black).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        batch = parse_payload({"observation.images.static": b64})
        self.assertAlmostEqual(batch["observation.images.static"].min().item(), 0.0)

    def test_multiple_cameras(self):
        """static + wrist 두 카메라 모두 독립적으로 변환."""
        payload = {
            "observation.images.static": _make_b64_image(200, 200),
            "observation.images.wrist": _make_b64_image(84, 84),
        }
        batch = parse_payload(payload)

        self.assertIn("observation.images.static", batch)
        self.assertIn("observation.images.wrist", batch)
        self.assertEqual(batch["observation.images.static"].shape, (1, 3, 200, 200))
        self.assertEqual(batch["observation.images.wrist"].shape, (1, 3, 84, 84))

    def test_no_image_key_absent(self):
        """이미지 없으면 observation.images.* 키 없음."""
        batch = parse_payload({"task": "test"})
        img_keys = [k for k in batch if k.startswith("observation.images.")]
        self.assertEqual(len(img_keys), 0)


class TestParsePayloadState(unittest.TestCase):
    """state 변환: observation.state.* sub-keys → 고정 순서 concatenate."""

    def test_full_state_shape(self):
        """STATE_KEY_ORDER 전체 키 → [1, 15] tensor."""
        batch = parse_payload(_make_full_state_payload())

        t = batch["observation.state"]
        self.assertIsInstance(t, torch.Tensor)
        expected_dim = sum(len(v) for v in _make_full_state_payload().values())
        self.assertEqual(t.shape, (1, expected_dim))
        self.assertEqual(t.dtype, torch.float32)

    def test_state_concatenation_order(self):
        """STATE_KEY_ORDER 순서대로 concatenate 됨을 값으로 검증."""
        payload = _make_full_state_payload()
        batch = parse_payload(payload)
        t = batch["observation.state"][0]  # [D]

        offset = 0
        for key in STATE_KEY_ORDER:
            if key not in payload:
                continue
            expected = torch.tensor(payload[key], dtype=torch.float32)
            actual = t[offset : offset + len(expected)]
            np.testing.assert_allclose(
                actual.numpy(),
                expected.numpy(),
                err_msg=f"state key '{key}' not in correct position",
            )
            offset += len(expected)

    def test_partial_state_keys(self):
        """일부 sub-key만 있으면 있는 것만 포함, 순서 유지."""
        payload = {
            "observation.state.eef_pos": [1.0, 2.0, 3.0],
            "observation.state.eef_euler": [0.1, 0.2, 0.3],
        }
        batch = parse_payload(payload)

        self.assertEqual(batch["observation.state"].shape, (1, 6))
        np.testing.assert_allclose(
            batch["observation.state"][0, :3].numpy(), [1.0, 2.0, 3.0]
        )

    def test_no_state_keys_absent(self):
        """state 키가 없으면 observation.state 키 없음."""
        batch = parse_payload({"observation.images.static": _make_b64_image()})
        self.assertNotIn("observation.state", batch)

    def test_state_unknown_subkey_ignored(self):
        """STATE_KEY_ORDER에 없는 sub-key는 무시."""
        payload = {
            "observation.state.eef_pos": [1.0, 2.0, 3.0],
            "observation.state.unknown_sensor": [9.9, 8.8],
        }
        batch = parse_payload(payload)
        # eef_pos(3)만 포함
        self.assertEqual(batch["observation.state"].shape, (1, 3))


class TestParsePayloadMisc(unittest.TestCase):
    """instruction 및 기타 키 처리."""

    def test_instruction_extracted(self):
        """task 키 → batch["task"] string."""
        payload = {
            "observation.images.static": _make_b64_image(),
            "task": "pick up the cup",
        }
        batch = parse_payload(payload)
        self.assertEqual(batch["task"], "pick up the cup")

    def test_missing_instruction_defaults_empty_string(self):
        """task 없으면 빈 문자열."""
        batch = parse_payload({"observation.images.static": _make_b64_image()})
        self.assertEqual(batch["task"], "")

    def test_unknown_top_level_keys_ignored(self):
        """observation.images/state/task 이외 키는 batch에 포함되지 않음."""
        payload = {
            "observation.images.static": _make_b64_image(),
            "latency_ms": 123.4,
            "some_random_key": "value",
        }
        batch = parse_payload(payload)
        self.assertNotIn("latency_ms", batch)
        self.assertNotIn("some_random_key", batch)


# ─── _apply_input_remap ──────────────────────────────────────────────────────


class TestApplyInputRemap(unittest.TestCase):
    """_apply_input_remap: 카메라 키 리맵핑 + state 차원 슬라이싱."""

    def setUp(self):
        import serve_lerobot as srv

        self.srv = srv
        # 각 테스트 전에 globals 초기화
        srv._camera_key_map = {}
        srv._state_dim = 0

    def tearDown(self):
        self.srv._camera_key_map = {}
        self.srv._state_dim = 0

    def test_noop_when_empty(self):
        """리맵핑 없으면 batch 그대로 반환."""
        batch = {
            "observation.images.static": torch.zeros(1, 3, 200, 200),
            "observation.state": torch.zeros(1, 15),
            "task": "test",
        }
        result = self.srv._apply_input_remap(batch)
        self.assertIn("observation.images.static", result)
        self.assertEqual(result["observation.state"].shape, (1, 15))

    def test_camera_key_renamed(self):
        """_camera_key_map에 따라 이미지 키가 rename됨."""
        self.srv._camera_key_map = {
            "observation.images.static": "observation.images.top"
        }
        batch = {"observation.images.static": torch.zeros(1, 3, 200, 200), "task": "t"}
        result = self.srv._apply_input_remap(batch)
        self.assertIn("observation.images.top", result)
        self.assertNotIn("observation.images.static", result)

    def test_multiple_camera_keys_renamed(self):
        """여러 카메라 키 동시 리맵핑."""
        self.srv._camera_key_map = {
            "observation.images.static": "observation.images.top",
            "observation.images.wrist": "observation.images.wrist_left",
        }
        batch = {
            "observation.images.static": torch.zeros(1, 3, 200, 200),
            "observation.images.wrist": torch.zeros(1, 3, 84, 84),
        }
        result = self.srv._apply_input_remap(batch)
        self.assertIn("observation.images.top", result)
        self.assertIn("observation.images.wrist_left", result)
        self.assertNotIn("observation.images.static", result)
        self.assertNotIn("observation.images.wrist", result)

    def test_state_dim_slicing(self):
        """_state_dim > 0이면 observation.state를 앞 N차원으로 truncate."""
        self.srv._state_dim = 7
        batch = {
            "observation.state": torch.arange(15, dtype=torch.float32).unsqueeze(0)
        }
        result = self.srv._apply_input_remap(batch)
        self.assertEqual(result["observation.state"].shape, (1, 7))
        np.testing.assert_allclose(
            result["observation.state"][0].numpy(), np.arange(7, dtype=np.float32)
        )

    def test_state_dim_zero_no_slice(self):
        """_state_dim == 0이면 슬라이싱 없음."""
        self.srv._state_dim = 0
        batch = {"observation.state": torch.zeros(1, 15)}
        result = self.srv._apply_input_remap(batch)
        self.assertEqual(result["observation.state"].shape, (1, 15))

    def test_missing_src_key_ignored(self):
        """리맵핑 src 키가 batch에 없으면 에러 없이 무시."""
        self.srv._camera_key_map = {
            "observation.images.wrist": "observation.images.wrist_left"
        }
        batch = {"observation.images.static": torch.zeros(1, 3, 200, 200)}
        result = self.srv._apply_input_remap(batch)
        self.assertIn("observation.images.static", result)
        self.assertNotIn("observation.images.wrist_left", result)


# ─── _build_remap_config ─────────────────────────────────────────────────────


class _StateFeat:
    """robot_state_feature stub."""

    def __init__(self, dim: int):
        self.shape = (dim,)


class TestBuildRemapConfig(unittest.TestCase):
    """_build_remap_config: policy input_features → 카메라 키맵 + state dim 도출."""

    # ── 카메라 키맵 ──────────────────────────────────────────────────────────

    def test_single_camera_different_key(self):
        """policy 키가 다르면 static → policy 키로 매핑."""
        key_map, _ = _build_remap_config(["observation.images.top"], None)
        self.assertEqual(
            key_map, {"observation.images.static": "observation.images.top"}
        )

    def test_single_camera_same_key_removed(self):
        """policy 키가 static과 동일하면 맵에서 제거."""
        key_map, _ = _build_remap_config(["observation.images.static"], None)
        self.assertEqual(key_map, {})

    def test_two_cameras_wrist_same_key_removed(self):
        """static+wrist 두 카메라, wrist키가 동일하면 static만 매핑."""
        key_map, _ = _build_remap_config(
            ["observation.images.top", "observation.images.wrist"], None
        )
        self.assertEqual(
            key_map, {"observation.images.static": "observation.images.top"}
        )

    def test_two_cameras_all_same_keys_empty(self):
        """두 카메라 모두 src==dst이면 빈 맵."""
        key_map, _ = _build_remap_config(
            ["observation.images.static", "observation.images.wrist"], None
        )
        self.assertEqual(key_map, {})

    def test_three_cameras_all_mapped(self):
        """세 카메라 모두 다른 키면 전부 매핑."""
        key_map, _ = _build_remap_config(
            ["observation.images.top", "observation.images.wrist_left", "observation.images.wrist_right"],
            None,
        )
        self.assertEqual(key_map, {
            "observation.images.static": "observation.images.top",
            "observation.images.wrist": "observation.images.wrist_left",
            "observation.images.wrist2": "observation.images.wrist_right",
        })

    # ── state dim ────────────────────────────────────────────────────────────

    def test_state_feat_none_gives_zero(self):
        """state_feat=None → state_dim=0 (슬라이싱 없음)."""
        _, state_dim = _build_remap_config(["observation.images.top"], None)
        self.assertEqual(state_dim, 0)

    def test_state_feat_shape_extracted(self):
        """state_feat.shape[0] → state_dim."""
        _, state_dim = _build_remap_config(["observation.images.top"], _StateFeat(7))
        self.assertEqual(state_dim, 7)

    def test_state_feat_14dim(self):
        _, state_dim = _build_remap_config(["observation.images.top"], _StateFeat(14))
        self.assertEqual(state_dim, 14)


# ─── FastAPI endpoints ────────────────────────────────────────────────────────


def _make_mock_policy(action_dim: int = 7, n_action_steps: int = 1) -> MagicMock:
    """select_action → [action_dim] tensor, reset → None."""
    if n_action_steps == 1:
        action = torch.zeros(action_dim)
    else:
        action = torch.zeros(n_action_steps, action_dim)

    policy = MagicMock()
    policy.select_action.return_value = action
    policy.reset.return_value = None
    return policy


def _make_mock_processors():
    """preprocessor: batch → batch (pass-through mock), postprocessor: identity."""
    pre = MagicMock(side_effect=lambda x: x)
    post = MagicMock(side_effect=lambda x: x)
    return pre, post


class TestActEndpoint(unittest.TestCase):
    """POST /act 엔드포인트."""

    def setUp(self):
        import serve_lerobot as srv
        from fastapi.testclient import TestClient

        srv.policy = _make_mock_policy(action_dim=7)
        srv.preprocessor, srv.postprocessor = _make_mock_processors()
        self.client = TestClient(srv.app)
        self.srv = srv

    def _act(self, extra: dict | None = None) -> dict:
        payload = {"observation.images.static": _make_b64_image(), "task": "test"}
        if extra:
            payload.update(extra)
        r = self.client.post("/act", json=payload)
        self.assertEqual(r.status_code, 200)
        return r.json()

    def test_response_has_action_and_latency(self):
        """응답에 action, latency_ms 키 존재."""
        data = self._act()
        self.assertIn("action", data)
        self.assertIn("latency_ms", data)

    def test_action_is_2d(self):
        """action은 항상 2D array [[...]]."""
        data = self._act()
        action = np.array(data["action"])
        self.assertEqual(action.ndim, 2)

    def test_action_dim(self):
        """action shape: [n_steps, 7]."""
        data = self._act()
        action = np.array(data["action"])
        self.assertEqual(action.shape[1], 7)

    def test_multi_step_action(self):
        """policy가 [N, 7] 반환 시 그대로 2D로 응답."""
        self.srv.policy = _make_mock_policy(action_dim=7, n_action_steps=10)
        data = self._act()
        action = np.array(data["action"])
        self.assertEqual(action.shape, (10, 7))

    def test_latency_ms_is_positive_float(self):
        data = self._act()
        self.assertIsInstance(data["latency_ms"], float)
        self.assertGreater(data["latency_ms"], 0.0)

    def test_model_not_loaded_returns_error(self):
        """policy가 None이면 500 또는 에러 응답."""
        self.srv.policy = None
        r = self.client.post(
            "/act",
            json={"observation.images.static": _make_b64_image(), "task": "test"},
        )
        self.assertIn(r.status_code, [200, 500])
        if r.status_code == 200:
            self.assertIn("error", r.json())


class TestResetEndpoint(unittest.TestCase):
    """POST /reset 엔드포인트."""

    def setUp(self):
        import serve_lerobot as srv
        from fastapi.testclient import TestClient

        self.mock_policy = _make_mock_policy()
        srv.policy = self.mock_policy
        srv.preprocessor, srv.postprocessor = _make_mock_processors()
        self.client = TestClient(srv.app)

    def test_reset_returns_ok(self):
        r = self.client.post("/reset")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "reset")

    def test_reset_calls_policy_reset(self):
        """policy.reset() 호출됨."""
        self.client.post("/reset")
        self.mock_policy.reset.assert_called_once()


class TestHealthEndpoint(unittest.TestCase):
    """GET /health 엔드포인트."""

    def setUp(self):
        import serve_lerobot as srv
        from fastapi.testclient import TestClient

        srv.policy = _make_mock_policy()
        srv.preprocessor, srv.postprocessor = _make_mock_processors()
        self.client = TestClient(srv.app)
        self.srv = srv

    def test_health_when_loaded(self):
        """policy 로드 시 status=ok."""
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_health_when_not_loaded(self):
        """policy=None이면 status=not_loaded."""
        self.srv.policy = None
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "not_loaded")

    def test_health_has_model_field(self):
        """응답에 model 필드 존재."""
        r = self.client.get("/health")
        self.assertIn("model", r.json())

    def test_health_has_n_action_steps(self):
        """응답에 n_action_steps 필드 존재 (calvin_eval.py가 act_step 자동 감지에 사용)."""
        r = self.client.get("/health")
        self.assertIn("n_action_steps", r.json())
        self.assertIsInstance(r.json()["n_action_steps"], int)

    def test_health_has_action_dim(self):
        """응답에 action_dim 필드 존재 (calvin_eval.py가 action 차원 검증에 사용)."""
        r = self.client.get("/health")
        self.assertIn("action_dim", r.json())
        self.assertIsInstance(r.json()["action_dim"], int)


if __name__ == "__main__":
    unittest.main()
