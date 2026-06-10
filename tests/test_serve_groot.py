"""GR00T HTTP server wiring tests."""

from __future__ import annotations

import argparse
import base64
import asyncio
import importlib.util
import io
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import httpx
import torch
from fastapi.encoders import jsonable_encoder
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
GROOT_SERVER_PATH = REPO_ROOT / "scripts" / "serve" / "groot.py"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))

from src.policies.groot.safe.features import decode_feature_tensor_blob  # noqa: E402


class _ModalityConfig:
    def __init__(self, modality_keys: list[str], delta_indices: list[int] | None = None):
        self.modality_keys = modality_keys
        self.delta_indices = delta_indices or []


class _ActionDim:
    def __init__(self, name: str, dims: int):
        self.name = name
        self.dims = dims


class _Profile:
    base_model = "groot"
    name = "groot_http_test"
    emits_subkeys = [
        "action.eef_pos",
        "action.eef_axisangle",
        "action.gripper",
        "action.base_motion",
        "action.control_mode",
    ]
    action_layout = [
        _ActionDim("eef_pos", 3),
        _ActionDim("eef_axisangle", 3),
        _ActionDim("gripper", 1),
    ]
    action_type = "relative"
    n_action_steps = 1

    def dim_slice(self, name: str) -> slice:
        offset = 0
        for item in self.action_layout:
            if item.name == name:
                return slice(offset, offset + item.dims)
            offset += item.dims
        raise KeyError(name)


class _HookHandle:
    def __init__(self, target: "_HookTarget"):
        self.target = target

    def remove(self) -> None:
        self.target.hook = None


class _HookTarget:
    """``policy.model.action_head.model`` DiT 더미. forward hook 으로 출력 캡처."""

    def __init__(self, output: torch.Tensor):
        self.output = output
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _HookHandle(self)

    def emit(self) -> None:
        if self.hook is not None:
            self.hook(self, (), self.output)


class _ActionHead:
    def __init__(self, model_action_horizon: int, feature_dim: int):
        # captured tensor shape: [B=1, model_action_horizon, feature_dim].
        # 각 step 의 leading H 토큰만 잘려서 export 됨.
        output = torch.arange(
            model_action_horizon * feature_dim, dtype=torch.float32
        ).reshape(1, model_action_horizon, feature_dim)
        self.action_horizon = model_action_horizon
        self.model = _HookTarget(output)


class _RootModel:
    def __init__(self, model_action_horizon: int, feature_dim: int):
        self.action_head = _ActionHead(model_action_horizon, feature_dim)


class _InnerPolicy:
    def __init__(self, model_action_horizon: int, feature_dim: int):
        self.model = _RootModel(model_action_horizon, feature_dim)


class _FakePolicy:
    """Gr00tSimPolicyWrapper 더미 — ``/act`` 와 ``/act_with_features`` 양쪽에서 사용."""

    def __init__(self, model_action_horizon: int = 4, feature_dim: int = 2):
        self.policy = _InnerPolicy(model_action_horizon, feature_dim)
        self.last_obs = None
        self.last_torch_initial_seed = None
        self.reset_count = 0
        self._modality_config: dict | None = None

    def set_modality_config(self, modality_config: dict) -> None:
        self._modality_config = modality_config

    def reset(self):
        self.reset_count += 1

    def get_modality_config(self):
        return self._modality_config

    def get_action(self, obs, options=None):
        self.last_obs = obs
        self.last_torch_initial_seed = torch.initial_seed()
        self.policy.model.action_head.model.emit()
        return (
            {
                "action.end_effector_position": np.array([[[1.0, 2.0, 3.0]]]),
                "gripper_close": np.array([[[0.5]]]),
            },
            {},
        )


def _import_groot_server():
    spec = importlib.util.spec_from_file_location("serve_groot_under_test", GROOT_SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_b64_image() -> str:
    arr = np.full((8, 8, 3), 127, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestBuildGrootObs(unittest.TestCase):
    def setUp(self):
        self.srv = _import_groot_server()

    def _set_service_runtime(self, modality_configs, state_dims):
        self.srv._service = self.srv.GrootPolicyService()
        self.srv._service.set_runtime_state(
            profile=_Profile(),
            policy=_FakePolicy(),
            modality_configs=modality_configs,
            state_dims=state_dims,
        )

    def test_new_embodiment_language_key_is_populated(self):
        modality_configs = {
            "video": _ModalityConfig(["robot0_agentview_left"]),
            "state": _ModalityConfig(["gripper_qpos"]),
            "language": _ModalityConfig(["annotation.human.task_description"]),
            "action": _ModalityConfig(["end_effector_position"], delta_indices=[0]),
        }
        self._set_service_runtime(modality_configs, {"gripper_qpos": 1})

        obs = self.srv._build_groot_obs(
            {
                "observation.images.left": _make_b64_image(),
                "observation.state.gripper_qpos": [0.25],
                "task": "open the cabinet",
            }
        )

        self.assertEqual(obs["annotation.human.task_description"], ("open the cabinet",))
        self.assertEqual(
            obs["annotation.human.action.task_description"], ("open the cabinet",)
        )
        self.assertIn("video.robot0_agentview_left", obs)
        self.assertIn("state.gripper_qpos", obs)

    def test_action_language_key_remains_populated(self):
        modality_configs = {
            "video": _ModalityConfig(["res256_image_side_0"]),
            "state": _ModalityConfig(["gripper_qpos"]),
            "language": _ModalityConfig(["annotation.human.action.task_description"]),
            "action": _ModalityConfig(["end_effector_position"], delta_indices=[0]),
        }
        self._set_service_runtime(modality_configs, {"gripper_qpos": 1})

        obs = self.srv._build_groot_obs(
            {
                "observation.images.left": _make_b64_image(),
                "observation.state.gripper_qpos": [0.25],
                "task": "pick up the object",
            }
        )

        self.assertEqual(
            obs["annotation.human.action.task_description"], ("pick up the object",)
        )
        self.assertEqual(obs["annotation.human.task_description"], ("pick up the object",))
        self.assertIn("video.res256_image_side_0", obs)


class TestPredictActionEndpoint(unittest.TestCase):
    def setUp(self):
        self.srv = _import_groot_server()
        self.policy = _FakePolicy()
        modality_configs = {
            "video": _ModalityConfig(["robot0_agentview_left"]),
            "state": _ModalityConfig(["gripper_qpos"]),
            "language": _ModalityConfig(["annotation.human.task_description"]),
            "action": _ModalityConfig(["end_effector_position"], delta_indices=[0]),
        }
        self.policy.set_modality_config(modality_configs)
        self.srv._service = self.srv.GrootPolicyService()
        self.srv._service.set_runtime_state(
            profile=_Profile(),
            policy=self.policy,
            modality_configs=modality_configs,
            state_dims={"gripper_qpos": 2},
        )
        self.srv.app.state.args = argparse.Namespace(
            feature_slice="valid",
            feature_dtype="float16",
            feature_action_horizon=None,
        )
        self.srv.app.router.on_startup.clear()

    def test_fastapi_routes_are_registered(self):
        route_methods = {
            route.path: set(route.methods or [])
            for route in self.srv.app.routes
            if hasattr(route, "path")
        }

        self.assertIn("POST", route_methods["/act"])
        self.assertIn("POST", route_methods["/act_with_features"])
        self.assertIn("POST", route_methods["/reset"])
        self.assertIn("GET", route_methods["/health"])

    def test_predict_action_maps_policy_output_to_http_subkeys(self):
        response = asyncio.run(
            self.srv.predict_action(
                {
                    "observation.images.left": _make_b64_image(),
                    "observation.state.gripper_qpos": [0.25, 0.75],
                    "task": "close the fridge",
                }
            )
        )

        self.assertEqual(response["action.eef_pos"], [[1.0, 2.0, 3.0]])
        self.assertEqual(response["action.gripper"], [[0.5]])
        self.assertEqual(response["action.base_motion"], [[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(response["action.control_mode"], [[0.0]])
        self.assertIn("latency_ms", response)
        self.assertEqual(
            self.policy.last_obs["annotation.human.task_description"],
            ("close the fridge",),
        )

    def test_predict_action_honors_inference_seed_without_leaking_rng_state(self):
        torch.manual_seed(777)
        before_state = torch.random.get_rng_state()

        response = asyncio.run(
            self.srv.predict_action(
                {
                    "observation.images.left": _make_b64_image(),
                    "task": "close the fridge",
                    "inference_seed": 4242,
                }
            )
        )

        self.assertIn("action.eef_pos", response)
        self.assertEqual(self.policy.last_torch_initial_seed, 4242)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_state))

    def test_predict_action_rejects_invalid_inference_seed(self):
        response = asyncio.run(
            self.srv.predict_action(
                {
                    "observation.images.left": _make_b64_image(),
                    "task": "close the fridge",
                    "inference_seed": "not-an-int",
                }
            )
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("inference_seed", response.body.decode())

    def test_http_act_endpoint_accepts_json_payload(self):
        async def run_request():
            transport = httpx.ASGITransport(app=self.srv.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/act",
                    json={
                        "observation.images.left": _make_b64_image(),
                        "observation.state.gripper_qpos": [0.25, 0.75],
                        "task": "close the fridge",
                    },
                )

        response = asyncio.run(run_request())

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["action.eef_pos"], [[1.0, 2.0, 3.0]])
        self.assertEqual(data["action.gripper"], [[0.5]])
        self.assertEqual(
            self.policy.last_obs["annotation.human.task_description"],
            ("close the fridge",),
        )

    def test_http_act_endpoint_returns_503_when_model_not_loaded(self):
        self.srv._service.policy = None

        async def run_request():
            transport = httpx.ASGITransport(app=self.srv.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/act",
                    json={
                        "observation.images.left": _make_b64_image(),
                        "task": "close the fridge",
                    },
                )

        response = asyncio.run(run_request())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"error": "model not loaded"})

    def test_act_response_is_json_serializable(self):
        data = asyncio.run(
            self.srv.predict_action(
                {
                    "observation.images.left": _make_b64_image(),
                    "observation.state.gripper_qpos": [0.25, 0.75],
                    "task": "close the fridge",
                }
            )
        )
        encoded = jsonable_encoder(data)

        self.assertEqual(data["action.eef_pos"], [[1.0, 2.0, 3.0]])
        self.assertEqual(data["action.gripper"], [[0.5]])
        self.assertEqual(data["action.base_motion"], [[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(data["action.control_mode"], [[0.0]])
        self.assertIn("latency_ms", data)
        self.assertEqual(encoded["action.eef_pos"], [[1.0, 2.0, 3.0]])

    def test_health_response_is_json_serializable_and_exposes_language_keys(self):
        data = asyncio.run(self.srv.health())
        encoded = jsonable_encoder(data)

        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["language_keys"], [
            "annotation.human.action.task_description",
            "annotation.human.task_description",
        ])
        self.assertEqual(encoded["language_keys"], data["language_keys"])

    def test_health_advertises_safe_feature_support(self):
        data = asyncio.run(self.srv.health())

        self.assertTrue(data["supports_features"])
        self.assertEqual(
            data["feature_kind"], "groot_n16_dit_valid_action_tokens_pre_velocity"
        )
        self.assertEqual(
            data["feature_axes"],
            ["denoising_step", "valid_action_step", "feature_dim"],
        )
        self.assertTrue(data["supports_inference_seed"])
        self.assertEqual(data["feature_slice"], "valid")
        self.assertEqual(data["feature_dtype"], "float16")
        self.assertIsNone(data["feature_action_horizon"])

    def test_act_with_features_returns_action_subkeys_and_feature_blob(self):
        response = asyncio.run(
            self.srv.predict_action_with_features(
                {
                    "observation.images.left": _make_b64_image(),
                    "observation.state.gripper_qpos": [0.25, 0.75],
                    "task": "close the fridge",
                }
            )
        )

        # action sub-key payload 는 /act 와 같은 모양.
        self.assertEqual(response["action.eef_pos"], [[1.0, 2.0, 3.0]])
        self.assertEqual(response["action.gripper"], [[0.5]])
        self.assertEqual(response["action.base_motion"], [[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(response["action.control_mode"], [[0.0]])
        self.assertIn("latency_ms", response)

        # features.* namespace.
        blob = response["features.hidden_states"]
        self.assertIsInstance(blob, dict)
        self.assertEqual(blob["dtype"], "float16")
        self.assertEqual(
            response["features.kind"],
            "groot_n16_dit_valid_action_tokens_pre_velocity",
        )
        self.assertEqual(
            response["features.axes"],
            ["denoising_step", "valid_action_step", "feature_dim"],
        )
        self.assertEqual(response["features.slice"], "valid")
        self.assertEqual(response["features.model_action_horizon"], 4)
        # valid_action_horizon = len(modality_configs["action"].delta_indices) = 1.
        self.assertEqual(response["features.valid_action_horizon"], 1)
        self.assertEqual(response["features.exported_action_token_count"], 1)
        self.assertEqual(response["features.feature_action_horizon"], 1)
        self.assertEqual(response["features.num_inference_timesteps"], 1)

        decoded = decode_feature_tensor_blob(blob)
        # [B, K, H, D] with B=1, K=1 denoising step, H=1 valid token, D=2.
        self.assertEqual(decoded.shape, (1, 1, 1, 2))
        # _ActionHead.output = arange(4*2).reshape(1,4,2); leading slice
        # [:, -4:, :][:, :1, :] picks tokens at position 0 → [[0., 1.]].
        np.testing.assert_allclose(decoded[0, 0, 0], np.array([0.0, 1.0]))

    def test_act_with_features_is_json_serializable(self):
        data = asyncio.run(
            self.srv.predict_action_with_features(
                {
                    "observation.images.left": _make_b64_image(),
                    "observation.state.gripper_qpos": [0.25, 0.75],
                    "task": "close the fridge",
                }
            )
        )

        encoded = jsonable_encoder(data)
        self.assertEqual(
            encoded["features.kind"], data["features.kind"]
        )
        self.assertEqual(
            encoded["features.hidden_states"]["shape"], list(data["features.hidden_states"]["shape"])
        )

    def test_act_with_features_returns_503_when_model_not_loaded(self):
        self.srv._service.policy = None

        async def run_request():
            transport = httpx.ASGITransport(app=self.srv.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/act_with_features",
                    json={
                        "observation.images.left": _make_b64_image(),
                        "task": "close the fridge",
                    },
                )

        response = asyncio.run(run_request())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"error": "model not loaded"})

    def test_act_with_features_returns_400_for_invalid_slice(self):
        self.srv.app.state.args = argparse.Namespace(
            feature_slice="bogus",
            feature_dtype="float16",
            feature_action_horizon=None,
        )

        async def run_request():
            transport = httpx.ASGITransport(app=self.srv.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/act_with_features",
                    json={
                        "observation.images.left": _make_b64_image(),
                        "observation.state.gripper_qpos": [0.25, 0.75],
                        "task": "close the fridge",
                    },
                )

        response = asyncio.run(run_request())
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertIn("error", body)
        self.assertIn("feature slice", body["error"].lower())


class TestGrootServeMain(unittest.TestCase):
    def test_device_arg_defaults_to_none_so_profile_device_can_apply(self):
        srv = _import_groot_server()

        with (
            mock.patch.object(sys, "argv", ["groot.py", "--profile", "profile.yaml"]),
            mock.patch.object(srv, "load_profile", return_value=_Profile()),
            mock.patch.object(srv, "run_uvicorn"),
        ):
            srv.main()

        self.assertIsNone(srv.app.state.args.device)

    def test_device_arg_can_explicitly_override_profile_device(self):
        srv = _import_groot_server()

        with (
            mock.patch.object(
                sys,
                "argv",
                ["groot.py", "--profile", "profile.yaml", "--device", "cpu"],
            ),
            mock.patch.object(srv, "load_profile", return_value=_Profile()),
            mock.patch.object(srv, "run_uvicorn"),
        ):
            srv.main()

        self.assertEqual(srv.app.state.args.device, "cpu")


if __name__ == "__main__":
    unittest.main()
