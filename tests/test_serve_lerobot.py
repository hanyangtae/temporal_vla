"""scripts/serve/lerobot.py 유닛 테스트 (TDD).

Docker 없이 로컬에서 실행 가능. LeRobot serving interface를 명세한다.

실행:
  python3 -m pytest tests/test_serve_lerobot.py -v
"""

from __future__ import annotations

import base64
import importlib.util
import io
import random
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "serve"))

_scripts_pkg = types.ModuleType("scripts")
_scripts_pkg.__path__ = [str(REPO_ROOT / "scripts")]
_serve_pkg = types.ModuleType("scripts.serve")
_serve_pkg.__path__ = [str(REPO_ROOT / "scripts" / "serve")]
sys.modules["scripts"] = _scripts_pkg
sys.modules["scripts.serve"] = _serve_pkg

_lerobot_spec = importlib.util.spec_from_file_location(
    "scripts.serve.lerobot",
    REPO_ROOT / "scripts" / "serve" / "lerobot.py",
)
assert _lerobot_spec and _lerobot_spec.loader
serve_lerobot = importlib.util.module_from_spec(_lerobot_spec)
sys.modules["scripts.serve.lerobot"] = serve_lerobot
_lerobot_spec.loader.exec_module(serve_lerobot)

from lerobot_adapters import make_policy_adapter  # noqa: E402
from lerobot_adapters.common import FeatureSpec  # noqa: E402
from lerobot_adapters.groot import (  # noqa: E402
    GrootPolicyAdapter,
    build_groot_feature_specs,
    groot_snapshot_allow_patterns,
    load_groot_eagle_projector_from_checkpoint,
    load_groot_dataset_stats_from_metadata,
    wrap_groot_native_action_postprocessor,
)
from lerobot_adapters.pi import PiPolicyAdapter  # noqa: E402
from checkpoint_profile import load_profile  # noqa: E402

STATE_KEY_ORDER = serve_lerobot.STATE_KEY_ORDER
_build_remap_config = serve_lerobot._build_remap_config
parse_payload = serve_lerobot.parse_payload

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


def _make_groot_profile() -> SimpleNamespace:
    return SimpleNamespace(
        image_preprocess=SimpleNamespace(rotate_180=False),
        observation_requirements=SimpleNamespace(
            images=["side_0", "side_1", "wrist_0"],
            state=[
                "eef_pos_rel",
                "eef_quat_rel",
                "gripper_qpos",
                "base_position",
                "base_rotation",
            ],
            allow_conversions=[],
        ),
    )


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

    def test_profile_center_crop_resize(self):
        """profile image_preprocess can center-crop and resize images."""
        from scripts.serve import lerobot as srv

        old_profile = srv._profile
        srv._profile = SimpleNamespace(
            image_preprocess=SimpleNamespace(
                rotate_180=False,
                center_crop=True,
                center_crop_scale=0.5,
                resolution=4,
            ),
            observation_requirements=SimpleNamespace(state=[]),
        )
        try:
            batch = parse_payload({"observation.images.static": _make_b64_image(8, 8)})
        finally:
            srv._profile = old_profile

        self.assertEqual(batch["observation.images.static"].shape, (1, 3, 4, 4))

    def test_profile_crop_resize_matches_groot_torchvision_rounding(self):
        """GR00T image preprocessing uses torchvision float resize before uint8 cast."""
        try:
            import torchvision.transforms.v2 as tv2
        except Exception as exc:
            self.skipTest(f"torchvision v2 unavailable: {exc}")

        from scripts.serve import lerobot as srv

        arr = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        old_profile = srv._profile
        srv._profile = SimpleNamespace(
            image_preprocess=SimpleNamespace(
                rotate_180=False,
                center_crop=True,
                center_crop_scale=0.75,
                resolution=4,
            ),
            observation_requirements=SimpleNamespace(state=[]),
        )
        try:
            batch = parse_payload({"observation.images.static": b64})
        finally:
            srv._profile = old_profile

        frame = torch.from_numpy(arr.copy()).to(torch.float32) / 255.0
        frame = frame.permute(2, 0, 1).unsqueeze(0)
        frame = tv2.CenterCrop((6, 6))(frame)
        frame = tv2.Resize(
            (4, 4),
            interpolation=tv2.InterpolationMode.BILINEAR,
            antialias=True,
        )(frame)
        expected = (frame[0].permute(1, 2, 0) * 255).to(torch.uint8).numpy()
        got = (batch["observation.images.static"][0].permute(1, 2, 0) * 255).to(torch.uint8).numpy()

        np.testing.assert_array_equal(got, expected)

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


class TestInferenceSeed(unittest.TestCase):
    """Optional per-request seed controls stochastic policy sampling."""

    def test_missing_inference_seed_is_noop(self):
        self.assertIsNone(serve_lerobot._apply_inference_seed({}))

    def test_inference_seed_replays_python_numpy_and_torch_rng(self):
        seed = 12345

        serve_lerobot._apply_inference_seed({"inference_seed": seed})
        first = (
            random.random(),
            np.random.rand(3),
            torch.rand(3),
        )

        serve_lerobot._apply_inference_seed({"inference_seed": seed})
        second = (
            random.random(),
            np.random.rand(3),
            torch.rand(3),
        )

        self.assertEqual(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])


# ─── _apply_input_remap ──────────────────────────────────────────────────────


class TestApplyInputRemap(unittest.TestCase):
    """_apply_input_remap: 카메라 키 리맵핑 + state 차원 슬라이싱."""

    def setUp(self):
        from scripts.serve import lerobot as srv

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


class TestPolicyAdapterFactory(unittest.TestCase):
    """LeRobot policy_type -> explicit adapter registry."""

    def test_pi_family_uses_pi_adapter(self):
        for policy_type in ("pi0", "pi05", "pi0_fast"):
            with self.subTest(policy_type=policy_type):
                self.assertIsInstance(make_policy_adapter(policy_type), PiPolicyAdapter)

    def test_groot_uses_groot_adapter(self):
        self.assertIsInstance(make_policy_adapter("groot"), GrootPolicyAdapter)

    def test_unknown_policy_type_is_rejected(self):
        with self.assertRaises(ValueError):
            make_policy_adapter("unknown_policy")


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


class TestGrootRobocasaObsBridge(unittest.TestCase):
    """GR00T RoboCasa365 HTTP obs bridge contract."""

    def setUp(self):
        from scripts.serve import lerobot as srv

        self.srv = srv
        self._old_profile = srv._profile
        self._old_policy_type = srv._policy_type
        self._old_policy_adapter = srv._policy_adapter
        self._old_camera_key_map = dict(srv._camera_key_map)
        self._old_state_dim = srv._state_dim
        srv._profile = _make_groot_profile()
        srv._policy_type = "groot"
        srv._policy_adapter = GrootPolicyAdapter()
        srv._camera_key_map = {}
        srv._state_dim = 0

    def tearDown(self):
        self.srv._profile = self._old_profile
        self.srv._policy_type = self._old_policy_type
        self.srv._policy_adapter = self._old_policy_adapter
        self.srv._camera_key_map = self._old_camera_key_map
        self.srv._state_dim = self._old_state_dim

    def test_groot_state_uses_robocasa_generic_aliases(self):
        """generic RoboCasa state aliases populate the 20D official GR00T state vector."""
        payload = {
            "observation.state.eef_pos": [1.0, 2.0, 3.0],
            "observation.state.eef_quat": [0.0, 0.0, 0.0, 1.0],
            "observation.state.gripper_qpos": [0.1, 0.2],
            "observation.state.base_pos": [4.0, 5.0, 6.0],
            "observation.state.base_quat": [0.0, 0.0, 1.0, 0.0],
        }

        batch = self.srv.parse_payload(payload)

        self.assertEqual(batch["observation.state"].shape, (1, 20))
        np.testing.assert_allclose(
            batch["observation.state"][0].numpy(),
            np.array(
                [
                    1.0, 2.0, 3.0,
                    -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                    0.1, 0.2,
                    4.0, 5.0, 6.0,
                    -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_groot_camera_aliases_do_not_clobber_direct_side_keys(self):
        """wrist alias maps to wrist_0, never overwrites side_1."""
        key_map, state_dim = self.srv._build_remap_config(
            [
                "observation.images.00_side_0",
                "observation.images.01_side_1",
                "observation.images.02_wrist_0",
            ],
            _StateFeat(16),
        )
        self.srv._camera_key_map = key_map
        self.srv._state_dim = state_dim

        side_1 = torch.ones(1, 3, 4, 4)
        wrist = torch.full((1, 3, 4, 4), 2.0)
        batch = {
            "observation.images.side_1": side_1.clone(),
            "observation.images.wrist": wrist.clone(),
            "observation.state": torch.zeros(1, 16),
        }

        result = self.srv._apply_input_remap(batch)

        np.testing.assert_allclose(
            result["observation.images.01_side_1"].numpy(),
            side_1.numpy(),
        )
        np.testing.assert_allclose(
            result["observation.images.02_wrist_0"].numpy(),
            wrist.numpy(),
        )
        self.assertNotIn("observation.images.side_1", result)
        self.assertNotIn("observation.images.wrist", result)


class TestGrootAdapterSpecs(unittest.TestCase):
    """GR00T policy adapter: profile → LeRobot GrootConfig feature specs."""

    def test_robocasa365_profile_matches_n15_data_config_order(self):
        profile = load_profile(
            REPO_ROOT
            / "configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml"
        )

        self.assertEqual(
            profile.observation_requirements.images,
            ["side_0", "side_1", "wrist_0"],
        )
        self.assertEqual(
            profile.observation_requirements.state,
            [
                "eef_pos_rel",
                "eef_quat_rel",
                "gripper_qpos",
                "base_position",
                "base_rotation",
            ],
        )
        self.assertEqual(
            [(item.name, item.dims) for item in profile.action_layout],
            [
                ("eef_pos", 3),
                ("eef_axisangle", 3),
                ("gripper", 1),
                ("base_motion", 4),
                ("control_mode", 1),
            ],
        )
        self.assertEqual(profile.action_type, "absolute")
        self.assertEqual(profile.action_dim, 12)
        self.assertEqual(profile.image_preprocess.resolution, 224)
        self.assertTrue(profile.image_preprocess.center_crop)
        self.assertEqual(profile.image_preprocess.center_crop_scale, 0.95)
        self.assertTrue(profile.model_specific["raw_language"])
        self.assertTrue(profile.model_specific["native_action_unapply"])
        self.assertTrue(profile.model_specific["load_native_eagle_projector"])

    def test_native_action_postprocessor_binarizes_gripper_and_control_mode(self):
        """Native GR00T action inverse uses binary mode for gripper/control_mode."""
        profile = load_profile(
            REPO_ROOT
            / "configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml"
        )
        stats = {
            "action": {
                "min": torch.tensor([-1.0] * 12),
                "max": torch.tensor([1.0] * 12),
            }
        }

        class _Post:
            def __call__(self, action):
                return action

        post = wrap_groot_native_action_postprocessor(_Post(), profile, stats)
        action = torch.tensor(
            [[
                0.0, 0.25, -0.25,
                0.5, -0.5, 0.0,
                0.49,
                -1.0, 0.0, 1.0, 0.25,
                0.51,
            ]],
            dtype=torch.float32,
        )

        result = post(action)

        self.assertEqual(result.shape, (1, 12))
        torch.testing.assert_close(result[:, 6], torch.tensor([0.0]))
        torch.testing.assert_close(result[:, 11], torch.tensor([1.0]))
        torch.testing.assert_close(result[:, :3], torch.tensor([[0.0, 0.25, -0.25]]))

    def test_robocasa_profile_builds_three_view_state_and_action_specs(self):
        profile = SimpleNamespace(
            image_preprocess=SimpleNamespace(resolution=224),
            observation_requirements=SimpleNamespace(
                images=["side_0", "side_1", "wrist_0"],
                state=[
                    "eef_pos_rel",
                    "eef_quat_rel",
                    "gripper_qpos",
                    "base_position",
                    "base_rotation",
                ],
            ),
            action_layout=[
                _ActionDim("eef_pos", 3),
                _ActionDim("eef_axisangle", 3),
                _ActionDim("gripper", 1),
                _ActionDim("base_motion", 4),
                _ActionDim("control_mode", 1),
            ],
        )

        input_specs, output_specs = build_groot_feature_specs(profile)

        self.assertEqual(
            input_specs,
            [
                FeatureSpec(
                    "observation.images.00_side_0", "VISUAL", (3, 224, 224)
                ),
                FeatureSpec(
                    "observation.images.01_side_1", "VISUAL", (3, 224, 224)
                ),
                FeatureSpec(
                    "observation.images.02_wrist_0", "VISUAL", (3, 224, 224)
                ),
                FeatureSpec("observation.state", "STATE", (20,)),
            ],
        )
        self.assertEqual(
            sorted(spec.key for spec in input_specs if spec.kind == "VISUAL"),
            [
                "observation.images.00_side_0",
                "observation.images.01_side_1",
                "observation.images.02_wrist_0",
            ],
        )
        self.assertEqual(
            output_specs,
            [FeatureSpec("action", "ACTION", (12,))],
        )

    def test_emit_subkeys_uses_metadata_action_order(self):
        from scripts.serve import lerobot as srv

        profile = load_profile(
            REPO_ROOT
            / "configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml"
        )
        action = np.arange(12, dtype=np.float32).reshape(1, 12)

        result = srv._emit_subkeys(action, profile)

        self.assertEqual(result["action.eef_pos"], [[0.0, 1.0, 2.0]])
        self.assertEqual(result["action.eef_axisangle"], [[3.0, 4.0, 5.0]])
        self.assertEqual(result["action.gripper"], [[6.0]])
        self.assertEqual(result["action.base_motion"], [[7.0, 8.0, 9.0, 10.0]])
        self.assertEqual(result["action.control_mode"], [[11.0]])

    def test_checkpoint_metadata_stats_are_flattened_in_profile_order(self):
        profile = load_profile(
            REPO_ROOT
            / "configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml"
        )
        metadata_path = (
            REPO_ROOT
            / "data/huggingface/hub/models--robocasa--robocasa365_checkpoints"
            / "snapshots/14895998fe7c8f8f2441cc8957ec2c510302758b"
            / "gr00t_n1-5/multitask_learning/checkpoint-120000"
            / "experiment_cfg/metadata.json"
        )

        stats = load_groot_dataset_stats_from_metadata(profile, metadata_path)

        self.assertEqual(set(stats), {"observation.state", "action"})
        self.assertEqual(stats["observation.state"]["min"].shape, (20,))
        self.assertEqual(stats["action"]["min"].shape, (12,))
        np.testing.assert_allclose(
            stats["observation.state"]["min"][:3].numpy(),
            [-0.6509041, -0.90334606, -0.3715313],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            stats["observation.state"]["min"][3:9].numpy(),
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            stats["action"]["min"][:5].numpy(),
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            rtol=1e-5,
        )

    def test_groot_hf_snapshot_patterns_download_only_runtime_files(self):
        patterns = groot_snapshot_allow_patterns(
            "gr00t_n1-5/multitask_learning/checkpoint-120000",
        )

        self.assertIn(
            "gr00t_n1-5/multitask_learning/checkpoint-120000/config.json",
            patterns,
        )
        self.assertIn(
            "gr00t_n1-5/multitask_learning/checkpoint-120000/model-*.safetensors",
            patterns,
        )
        self.assertIn(
            "gr00t_n1-5/multitask_learning/checkpoint-120000/experiment_cfg/**",
            patterns,
        )
        self.assertFalse(any("optimizer" in pattern for pattern in patterns))
        self.assertFalse(any("scheduler" in pattern for pattern in patterns))

    def test_native_eagle_projector_loader_reads_raw_groot_shards(self):
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            self.skipTest(f"safetensors unavailable: {exc}")

        projector = torch.nn.Sequential(torch.nn.Linear(2, 3))
        policy = SimpleNamespace(
            _groot_model=SimpleNamespace(
                backbone=SimpleNamespace(
                    eagle_model=SimpleNamespace(mlp1=projector),
                ),
            ),
        )
        expected_weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        expected_bias = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(
                {
                    "backbone.eagle_model.mlp1.0.weight": expected_weight,
                    "backbone.eagle_model.mlp1.0.bias": expected_bias,
                    "backbone.other": torch.ones(1),
                },
                str(Path(tmpdir) / "model-00001-of-00001.safetensors"),
            )

            loaded_keys = load_groot_eagle_projector_from_checkpoint(policy, tmpdir)

        self.assertEqual(loaded_keys, ["0.bias", "0.weight"])
        torch.testing.assert_close(projector[0].weight, expected_weight)
        torch.testing.assert_close(projector[0].bias, expected_bias)


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


class _ActionDim:
    def __init__(self, name: str, dims: int):
        self.name = name
        self.dims = dims


class _MockProfile:
    name = "lerobot_test"
    action_type = "relative"
    emits_subkeys = [
        "action.eef_pos",
        "action.eef_axisangle",
        "action.gripper",
    ]
    action_layout = [
        _ActionDim("eef_pos", 3),
        _ActionDim("eef_axisangle", 3),
        _ActionDim("gripper", 1),
    ]

    def dim_slice(self, name: str) -> slice:
        offset = 0
        for item in self.action_layout:
            if item.name == name:
                return slice(offset, offset + item.dims)
            offset += item.dims
        raise KeyError(name)


class TestActEndpoint(unittest.TestCase):
    """POST /act 엔드포인트."""

    def setUp(self):
        from scripts.serve import lerobot as srv
        from fastapi.testclient import TestClient

        srv.policy = _make_mock_policy(action_dim=7)
        srv.preprocessor, srv.postprocessor = _make_mock_processors()
        srv._profile = _MockProfile()
        srv._policy_type = "lerobot_test"
        srv._n_action_steps = 1
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
        """응답에 action subkeys, latency_ms 키 존재."""
        data = self._act()
        self.assertIn("action.eef_pos", data)
        self.assertIn("action.eef_axisangle", data)
        self.assertIn("action.gripper", data)
        self.assertIn("latency_ms", data)

    def test_action_is_2d(self):
        """action subkey는 항상 2D array [[...]]."""
        data = self._act()
        for key in _MockProfile.emits_subkeys:
            self.assertEqual(np.array(data[key]).ndim, 2)

    def test_action_dim(self):
        """action subkey dims add up to the 7D profile layout."""
        data = self._act()
        dim = sum(np.array(data[key]).shape[1] for key in _MockProfile.emits_subkeys)
        self.assertEqual(dim, 7)

    def test_multi_step_action(self):
        """policy가 [N, 7] 반환 시 subkey도 N-step으로 응답."""
        self.srv.policy = _make_mock_policy(action_dim=7, n_action_steps=10)
        self.srv._n_action_steps = 10
        data = self._act()
        self.assertEqual(np.array(data["action.eef_pos"]).shape, (10, 3))
        self.assertEqual(np.array(data["action.eef_axisangle"]).shape, (10, 3))
        self.assertEqual(np.array(data["action.gripper"]).shape, (10, 1))

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


class TestActWithFeaturesEndpoint(unittest.TestCase):
    """POST /act_with_features endpoint contract."""

    def setUp(self):
        from scripts.serve import lerobot as srv
        from fastapi.testclient import TestClient

        srv.policy = _make_mock_policy(action_dim=7)
        srv.preprocessor, srv.postprocessor = _make_mock_processors()
        srv._profile = _MockProfile()
        srv._policy_type = "pi05"
        srv._n_action_steps = 1
        self.client = TestClient(srv.app)
        self.srv = srv

    def tearDown(self):
        self.srv.policy = None

    def test_response_includes_unified_feature_horizon_metadata(self):
        hidden = np.zeros((10, 50, 4), dtype=np.float16)
        action = torch.zeros(7, dtype=torch.float32)
        meta = {
            "feature_kind": "pi05_action_expert_pre_velocity",
            "feature_axes": ["denoising_step", "action_step", "feature_dim"],
            "num_inference_timesteps": 10,
            "action_horizon": 50,
            "feature_dim": 4,
        }

        with unittest.mock.patch.object(
            self.srv.safe_hooks,
            "run_with_features",
            return_value=(action, hidden, meta["feature_axes"], meta),
        ):
            r = self.client.post(
                "/act_with_features",
                json={"observation.images.static": _make_b64_image(), "task": "test"},
            )

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["has_feature"])
        self.assertEqual(data["features.kind"], "pi05_action_expert_pre_velocity")
        self.assertEqual(data["features.exported_action_token_count"], 50)
        self.assertEqual(data["features.feature_action_horizon"], 50)
        self.assertEqual(data["features.model_action_horizon"], 50)
        self.assertEqual(data["features.num_inference_timesteps"], 10)
        self.assertIn("features.hidden_states", data)
        self.assertIn("hidden_states_b64", data)

    def test_response_maps_autoregressive_token_count_to_feature_horizon(self):
        hidden = np.zeros((1, 12, 4), dtype=np.float16)
        action = torch.zeros(7, dtype=torch.float32)
        meta = {
            "feature_kind": "pi0_fast_prelogit_action_tokens",
            "feature_axes": ["token_singleton", "action_token", "feature_dim"],
            "num_inference_timesteps": None,
            "action_horizon": None,
            "n_action_tokens": 12,
            "exported_action_token_count": None,
            "model_action_horizon": None,
            "feature_dim": 4,
        }

        with unittest.mock.patch.object(
            self.srv.safe_hooks,
            "run_with_features",
            return_value=(action, hidden, meta["feature_axes"], meta),
        ):
            r = self.client.post(
                "/act_with_features",
                json={"observation.images.static": _make_b64_image(), "task": "test"},
            )

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["features.kind"], "pi0_fast_prelogit_action_tokens")
        self.assertEqual(data["features.exported_action_token_count"], 12)
        self.assertEqual(data["features.feature_action_horizon"], 12)
        self.assertEqual(data["features.model_action_horizon"], 12)


class TestResetEndpoint(unittest.TestCase):
    """POST /reset 엔드포인트."""

    def setUp(self):
        from scripts.serve import lerobot as srv
        from fastapi.testclient import TestClient

        self.mock_policy = _make_mock_policy()
        srv.policy = self.mock_policy
        srv.preprocessor, srv.postprocessor = _make_mock_processors()
        srv._profile = _MockProfile()
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
        from scripts.serve import lerobot as srv
        from fastapi.testclient import TestClient

        srv.policy = _make_mock_policy()
        srv.preprocessor, srv.postprocessor = _make_mock_processors()
        srv._profile = _MockProfile()
        srv._policy_type = "lerobot_test"
        srv._n_action_steps = 1
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

    def test_health_has_action_keys(self):
        """응답에 action_keys 필드 존재."""
        r = self.client.get("/health")
        self.assertEqual(r.json()["action_keys"], list(_MockProfile.emits_subkeys))


if __name__ == "__main__":
    unittest.main()
