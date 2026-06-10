"""HTTP VLA client response handling tests."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import numpy as np
import requests

from src.utils.common.feature_blob import (
    encode_feature_blob,
    encode_legacy_feature_array,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))

import vla_client  # noqa: E402
from vla_client import VLAClient  # noqa: E402


class _Response:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


def _image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


class _JsonResponse:
    def __init__(self, payload=None, status_code: int = 200, json_error: Exception | None = None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class _ActHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode()) if body else {}
        self.server.requests.append({"path": self.path, "payload": payload})

        response_payload = json.dumps(self.server.response_payload).encode()
        self.send_response(self.server.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_payload)))
        self.end_headers()
        self.wfile.write(response_payload)

    def log_message(self, *args):
        pass


class _LoopbackActServer:
    def __init__(self, response_payload, status_code: int = 200):
        self.response_payload = response_payload
        self.status_code = status_code
        self.server = None
        self.thread = None

    def __enter__(self):
        try:
            self.server = HTTPServer(("127.0.0.1", 0), _ActHandler)
        except PermissionError as exc:
            raise unittest.SkipTest("local socket binding is not permitted") from exc

        self.server.response_payload = self.response_payload
        self.server.response_status = self.status_code
        self.server.requests = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=1)

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_port}"

    @property
    def requests(self) -> list[dict]:
        assert self.server is not None
        return self.server.requests


class TestVLAClientPredict(unittest.TestCase):
    def test_health_check_returns_none_for_invalid_json(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.get",
            return_value=_JsonResponse(json_error=ValueError("bad json")),
        ):
            self.assertIsNone(client.health_check())

    def test_health_check_uses_requested_timeout(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.get",
            return_value=_JsonResponse({"status": "ok"}),
        ) as get:
            self.assertEqual(client.health_check(timeout=1.25), {"status": "ok"})

        self.assertEqual(get.call_args.kwargs["timeout"], 1.25)

    def test_health_check_returns_none_for_request_exception(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.get",
            side_effect=requests.Timeout("timeout"),
        ):
            self.assertIsNone(client.health_check())

    def test_wait_until_ready_does_not_oversleep_max_wait(self):
        client = VLAClient("http://server")
        now = [0.0]
        sleeps = []

        def fake_sleep(duration):
            sleeps.append(duration)
            now[0] += duration

        with (
            mock.patch.object(client, "health_check", return_value=None),
            mock.patch.object(vla_client.time, "time", side_effect=lambda: now[0]),
            mock.patch.object(vla_client.time, "sleep", side_effect=fake_sleep),
        ):
            with self.assertRaisesRegex(TimeoutError, "not ready"):
                client.wait_until_ready(max_wait=1.0, poll_interval=3.0)

        self.assertEqual(sleeps, [1.0])

    def test_wait_until_ready_caps_health_timeout_to_remaining_budget(self):
        client = VLAClient("http://server")
        now = [0.0]
        health_timeouts = []

        def fake_health_check(timeout):
            health_timeouts.append(timeout)
            return None

        def fake_sleep(duration):
            now[0] += duration

        with (
            mock.patch.object(client, "health_check", side_effect=fake_health_check),
            mock.patch.object(vla_client.time, "time", side_effect=lambda: now[0]),
            mock.patch.object(vla_client.time, "sleep", side_effect=fake_sleep),
        ):
            with self.assertRaisesRegex(TimeoutError, "not ready"):
                client.wait_until_ready(max_wait=1.0, poll_interval=3.0)

        self.assertEqual(health_timeouts, [1.0])

    def test_wait_until_ready_returns_ok_health_payload(self):
        client = VLAClient("http://server")
        now = [0.0]

        def fake_sleep(duration):
            now[0] += duration

        with (
            mock.patch.object(
                client,
                "health_check",
                side_effect=[{"status": "not_loaded"}, {"status": "ok", "model": "groot"}],
            ),
            mock.patch.object(vla_client.time, "time", side_effect=lambda: now[0]),
            mock.patch.object(vla_client.time, "sleep", side_effect=fake_sleep) as sleep,
        ):
            info = client.wait_until_ready(max_wait=5.0, poll_interval=3.0)

        self.assertEqual(info["model"], "groot")
        sleep.assert_called_once_with(3.0)

    def test_predict_returns_subkeyed_action_dict(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"action.eef_pos": [[1.0, 2.0, 3.0]]}),
        ) as post:
            actions, _ = client.predict({"left": _image()}, instruction="open drawer")

        self.assertEqual(post.call_args.args[0], "http://server/act")
        self.assertEqual(post.call_args.kwargs["json"]["task"], "open drawer")
        self.assertEqual(sorted(actions), ["action.eef_pos"])
        np.testing.assert_allclose(actions["action.eef_pos"], [[1.0, 2.0, 3.0]])

    def test_predict_sends_optional_inference_seed(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"action.eef_pos": [[1.0, 2.0, 3.0]]}),
        ) as post:
            client.predict({"left": _image()}, instruction="open drawer", inference_seed=4242)

        self.assertEqual(post.call_args.kwargs["json"]["inference_seed"], 4242)

    def test_predict_returns_flat_action_array(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"action": [1.0, 2.0, 3.0]}),
        ):
            actions, _ = client.predict({"left": _image()})

        np.testing.assert_allclose(actions, [[1.0, 2.0, 3.0]])

    def test_predict_raises_runtime_error_for_server_error_body(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"error": "model not loaded"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "model not loaded"):
                client.predict({"left": _image()})

    def test_predict_raises_runtime_error_for_http_error_detail(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"detail": "bad payload"}, status_code=422),
        ):
            with self.assertRaisesRegex(RuntimeError, "bad payload"):
                client.predict({"left": _image()})

    def test_predict_raises_runtime_error_for_http_error_body(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"error": "model not loaded"}, status_code=503),
        ):
            with self.assertRaisesRegex(RuntimeError, "model not loaded"):
                client.predict({"left": _image()})

    def test_predict_raises_runtime_error_for_missing_action_keys(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"latency_ms": 1.0}),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing action keys: latency_ms"):
                client.predict({"left": _image()})

    def test_predict_posts_to_loopback_act_endpoint(self):
        with _LoopbackActServer({"action.eef_pos": [[4.0, 5.0, 6.0]]}) as server:
            client = VLAClient(server.url)
            actions, _ = client.predict(
                {"left": _image()},
                states={"observation.state.eef_pos_rel": np.array([0.1, 0.2, 0.3])},
                instruction="close the drawer",
            )

        self.assertEqual(server.requests[0]["path"], "/act")
        self.assertEqual(server.requests[0]["payload"]["task"], "close the drawer")
        self.assertEqual(
            server.requests[0]["payload"]["observation.state.eef_pos_rel"],
            [0.1, 0.2, 0.3],
        )
        self.assertIn("observation.images.left", server.requests[0]["payload"])
        np.testing.assert_allclose(actions["action.eef_pos"], [[4.0, 5.0, 6.0]])


def _feature_blob(arr: np.ndarray) -> dict:
    return encode_feature_blob(arr)


def _legacy_feature_blob(arr: np.ndarray) -> str:
    return encode_legacy_feature_array(arr)


class TestVLAClientPredictWithFeatures(unittest.TestCase):
    def _features_response(self, feature_arr: np.ndarray) -> dict:
        return {
            "action.eef_pos": [[1.0, 2.0, 3.0]],
            "action.gripper": [[0.5]],
            "latency_ms": 12.5,
            "features.hidden_states": _feature_blob(feature_arr),
            "features.kind": "groot_n16_dit_valid_action_tokens_pre_velocity",
            "features.axes": ["denoising_step", "valid_action_step", "feature_dim"],
            "features.slice": "valid",
            "features.exported_action_token_count": 16,
            "features.feature_action_horizon": 16,
            "features.valid_action_horizon": 16,
            "features.model_action_horizon": 50,
            "features.num_inference_timesteps": 4,
        }

    def test_predict_with_features_decodes_blob_and_metadata(self):
        feature_arr = np.arange(2 * 3 * 4, dtype=np.float16).reshape(1, 2, 3, 4)
        payload = self._features_response(feature_arr)
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post", return_value=_Response(payload)
        ) as post:
            actions, features, _latency = client.predict_with_features(
                {"left": _image()},
                states={"observation.state.eef_pos_rel": np.array([0.0, 0.0, 0.0])},
                instruction="open the cabinet",
                inference_seed=4242,
            )

        self.assertEqual(post.call_args.args[0], "http://server/act_with_features")
        self.assertEqual(post.call_args.kwargs["json"]["task"], "open the cabinet")
        self.assertEqual(post.call_args.kwargs["json"]["inference_seed"], 4242)
        np.testing.assert_allclose(actions["action.eef_pos"], [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(actions["action.gripper"], [[0.5]])

        np.testing.assert_array_equal(features["hidden_states"], feature_arr)
        self.assertEqual(
            features["kind"], "groot_n16_dit_valid_action_tokens_pre_velocity"
        )
        self.assertEqual(
            features["axes"],
            ["denoising_step", "valid_action_step", "feature_dim"],
        )
        self.assertEqual(features["slice"], "valid")
        self.assertEqual(features["exported_action_token_count"], 16)
        self.assertEqual(features["feature_action_horizon"], 16)
        self.assertEqual(features["valid_action_horizon"], 16)
        self.assertEqual(features["model_action_horizon"], 50)
        self.assertEqual(features["num_inference_timesteps"], 4)

    def test_predict_with_features_accepts_namespaced_legacy_horizon_alias(self):
        feature_arr = np.arange(1 * 1 * 12 * 4, dtype=np.float16).reshape(1, 1, 12, 4)
        payload = self._features_response(feature_arr)
        payload["features.feature_action_horizon"] = None
        payload["features.exported_action_token_count"] = None
        payload["features.model_action_horizon"] = None
        payload["features.action_horizon"] = None
        payload["features.n_action_tokens"] = 12
        client = VLAClient("http://server")

        with mock.patch("vla_client.requests.post", return_value=_Response(payload)):
            _actions, features, _latency = client.predict_with_features({"left": _image()})

        assert features is not None
        self.assertEqual(features["exported_action_token_count"], 12)
        self.assertEqual(features["feature_action_horizon"], 12)
        self.assertEqual(features["model_action_horizon"], 12)

    def test_predict_with_features_raises_when_blob_missing(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"action.eef_pos": [[0.0, 0.0, 0.0]]}),
        ):
            with self.assertRaisesRegex(RuntimeError, "features.hidden_states"):
                client.predict_with_features({"left": _image()})

    def test_predict_with_features_returns_none_when_lerobot_queue_has_no_feature(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response(
                {
                    "action.eef_pos": [[0.0, 0.0, 0.0]],
                    "has_feature": False,
                }
            ),
        ):
            actions, features, _latency = client.predict_with_features({"left": _image()})

        np.testing.assert_allclose(actions["action.eef_pos"], [[0.0, 0.0, 0.0]])
        self.assertIsNone(features)

    def test_predict_with_features_decodes_legacy_lerobot_feature_response(self):
        feature_arr = np.arange(10 * 50 * 4, dtype=np.float16).reshape(10, 50, 4)
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response(
                {
                    "action.eef_pos": [[1.0, 2.0, 3.0]],
                    "has_feature": True,
                    "hidden_states_b64": _legacy_feature_blob(feature_arr),
                    "feature_kind": "pi05_action_expert_pre_velocity",
                    "feature_axes": ["denoising_step", "action_step", "feature_dim"],
                    "num_inference_timesteps": 10,
                    "action_horizon": None,
                    "n_action_tokens": 50,
                    "exported_action_token_count": None,
                    "model_action_horizon": None,
                }
            ),
        ):
            _actions, features, _latency = client.predict_with_features({"left": _image()})

        self.assertIsNotNone(features)
        assert features is not None
        np.testing.assert_array_equal(features["hidden_states"], feature_arr)
        self.assertEqual(features["kind"], "pi05_action_expert_pre_velocity")
        self.assertEqual(
            features["axes"], ["denoising_step", "action_step", "feature_dim"]
        )
        self.assertEqual(features["exported_action_token_count"], 50)
        self.assertEqual(features["feature_action_horizon"], 50)
        self.assertEqual(features["model_action_horizon"], 50)
        self.assertEqual(features["num_inference_timesteps"], 10)

    def test_predict_with_features_propagates_server_error(self):
        client = VLAClient("http://server")

        with mock.patch(
            "vla_client.requests.post",
            return_value=_Response({"error": "model not loaded"}, status_code=503),
        ):
            with self.assertRaisesRegex(RuntimeError, "model not loaded"):
                client.predict_with_features({"left": _image()})

    def test_predict_with_features_round_trips_loopback_server(self):
        feature_arr = np.linspace(-1.0, 1.0, 12, dtype=np.float16).reshape(1, 2, 2, 3)
        payload = self._features_response(feature_arr)

        with _LoopbackActServer(payload) as server:
            client = VLAClient(server.url)
            actions, features, _latency = client.predict_with_features(
                {"left": _image()},
                states={"observation.state.eef_pos_rel": np.array([0.1, 0.2, 0.3])},
                instruction="turn on the sink",
            )

        self.assertEqual(server.requests[0]["path"], "/act_with_features")
        self.assertEqual(server.requests[0]["payload"]["task"], "turn on the sink")
        np.testing.assert_allclose(actions["action.eef_pos"], [[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(features["hidden_states"], feature_arr)


if __name__ == "__main__":
    unittest.main()
