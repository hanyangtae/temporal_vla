#!/usr/bin/env python3
"""Native GR00T **N1.6** ZMQ serve + optional COAST conceptor steering.

This is the N1.6 analogue of ``native_steered_serve.py`` (N1.5). It exists so an
N1.6 checkpoint can run inside the *identical faithful stack* as N1.5 — driven by
the same client ``native_official_zmq_eval.py`` (official ``robocasa/<Task>`` env,
``split=pretrain``, ``replan=5``). That gives an apples-to-apples N1.5-vs-N1.6
steering ΔSR comparison.

Two things differ from the N1.5 serve and are the whole reason this file exists:

1. **Different submodule / loader.** N1.6 lives in ``src/policies/Isaac-GR00T``
   (NOT ``Isaac-GR00T-N1.5``). Both expose a top-level ``gr00t`` package, so they
   cannot coexist in one process. This script must run with that submodule on
   ``PYTHONPATH`` (the ``groot`` docker service / ``configure_repo_paths(include_groot=True)``).
   The policy is built exactly like the proven ``feature_server.py`` /
   ``run_gr00t_server.py``: ``Gr00tPolicy(embodiment_tag, model_path, device,
   strict)`` wrapped in ``Gr00tSimPolicyWrapper`` (flat sim-env obs/action keys).

2. **Different ZMQ wire protocol.** The N1.6 ``PolicyServer.run()`` dispatches
   ``handler(**data)`` and N1.6's ``Gr00tSimPolicyWrapper.get_action`` returns a
   ``(action, info)`` tuple. But the N1.5 client (``native_official_zmq_eval.py``)
   sends ``data = <flat obs dict>`` (positional-style) and expects a **dict**
   response (it does ``action_dict.get("actions", action_dict)``). To stay a thin
   wrapper with NO submodule edit, this serve runs its own minimal ZMQ REP loop
   that mirrors N1.5's ``BaseInferenceServer.run()`` dispatch (positional
   ``handler(data)``), reuses N1.6's ``MsgSerializer`` (msgpack + ``__ndarray_class__``
   — byte-compatible with the client's ``N15MsgSerializer``), and unwraps the
   ``(action, info)`` tuple to a flat action dict before replying.

The COAST steering hook is identical to N1.5/feature_server: ``ConceptorSteering``
on the DiT output (``pathway="dit"``, ``--steering-layer None`` => ``action_head.model``
output, feature_dim=1024 — matching the N1.6 C_steer conceptors). β=0 (or omitting
``--steering-npz``) => baseline serve (hook absent / M=I).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import zmq

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# N1.6 submodule on path. In the `groot` docker service PYTHONPATH already points
# at Isaac-GR00T; this is a host-checkout safety net. We deliberately do NOT add
# Isaac-GR00T-N1.5 (the two `gr00t` packages must not collide).
_N16_ROOT = _REPO / "src" / "policies" / "Isaac-GR00T"
if _N16_ROOT.exists() and str(_N16_ROOT) not in sys.path:
    sys.path.insert(0, str(_N16_ROOT))

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import (  # noqa: E402
    Gr00tPolicy,
    Gr00tSimPolicyWrapper,
)
from gr00t.policy.server_client import MsgSerializer  # noqa: E402

from scripts.serve.steering_hooks import (  # noqa: E402
    ConceptorSteering,
    load_steering_matrix,
)


class N15CompatInferenceServer:
    """Minimal ZMQ REP server speaking the **N1.5 client** protocol on an N1.6 policy.

    Mirrors ``Isaac-GR00T-N1.5/gr00t/eval/service.py:BaseInferenceServer.run()``:
    the request is ``{"endpoint": ..., "data": <obs dict>}`` and the handler is
    called positionally as ``handler(data)`` (NOT ``handler(**data)`` like N1.6's
    own ``PolicyServer``). ``get_action`` returns the flat action dict (the
    ``(action, info)`` tuple from ``Gr00tSimPolicyWrapper`` is unwrapped to its
    action component) so the N1.5 client's ``action_dict.get("actions", action_dict)``
    resolves to the flat ``action.*`` map.
    """

    def __init__(self, policy: Any, host: str = "*", port: int = 5558,
                 api_token: str | None = None):
        self.policy = policy
        self.api_token = api_token
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, Any] = {
            "get_action": self._get_action,
            "get_modality_config": self._get_modality_config,
            "ping": self._ping,
            "kill": self._kill,
        }

    # ── endpoints ────────────────────────────────────────────────────────────
    def _get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = self.policy.get_action(observation)
        # Gr00tSimPolicyWrapper.get_action -> (action_dict, info_dict). The N1.5
        # client expects a flat dict, so return the action component only.
        if isinstance(result, tuple):
            return result[0]
        return result

    def _get_modality_config(self) -> Any:
        return self.policy.get_modality_config()

    def _ping(self) -> dict[str, str]:
        return {"status": "ok", "message": "Server is running"}

    def _kill(self) -> None:
        self.running = False

    # ── loop ─────────────────────────────────────────────────────────────────
    def _validate_token(self, request: dict) -> bool:
        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"[serve] N15CompatInferenceServer ready on {addr}", flush=True)
        while self.running:
            try:
                request = MsgSerializer.from_bytes(self.socket.recv())
                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: invalid API token"})
                    )
                    continue
                endpoint = request.get("endpoint", "get_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                handler = self._endpoints[endpoint]
                if endpoint in ("get_action",):
                    # positional dispatch (N1.5 protocol): data == obs dict.
                    result = handler(request.get("data", {}))
                else:
                    result = handler()
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as exc:  # noqa: BLE001
                print(f"[serve] error: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                self.socket.send(MsgSerializer.to_bytes({"error": str(exc)}))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True,
                    help="N1.6 checkpoint dir (e.g. .../grootn16_.../checkpoint-120000).")
    ap.add_argument("--embodiment-tag", default="ROBOCASA_PANDA_OMRON",
                    help=(
                        "EmbodimentTag enum NAME. Default ROBOCASA_PANDA_OMRON: its "
                        "modality keys (video res256_image_side_0/1, res256_image_wrist_0; "
                        "language annotation.human.action.task_description) match the RAW "
                        "obs keys emitted by the official robocasa/<Task> env, so the N1.5 "
                        "client (native_official_zmq_eval.py — which does NOT alias obs keys) "
                        "works without a remap layer. Use NEW_EMBODIMENT only if obs keys are "
                        "pre-aliased to robot0_agentview_left/right + annotation.human.task_description."
                    ))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strict", action="store_true", default=False,
                    help="Enforce Gr00tPolicy obs/action validation (default off, as feature_server).")
    # --data-config accepted for CLI symmetry with the N1.5 serve; N1.6 derives
    # transforms from the checkpoint processor, so it is ignored.
    ap.add_argument("--data-config", default=None,
                    help="Ignored for N1.6 (transforms come from the checkpoint processor).")
    ap.add_argument("--denoising-steps", type=int, default=None,
                    help="Ignored for N1.6 (fixed by the checkpoint). Accepted for CLI symmetry.")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--host", default="*")
    ap.add_argument("--api-token", default=None)
    # COAST steering (optional). Omit --steering-npz for the baseline serve.
    ap.add_argument("--steering-npz", default=None)
    ap.add_argument("--steering-beta", type=float, default=0.1)
    ap.add_argument("--steering-alpha", type=float, default=None)
    ap.add_argument("--steering-layer", type=int, default=None,
                    help="DiT transformer_block index. None = action_head.model output (D=1024).")
    ap.add_argument("--steering-pathway", choices=("dit", "vl"), default="dit")
    ap.add_argument("--steering-key", default="C_steer",
                    choices=("C_steer", "C_success", "C_failure"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base_policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag[args.embodiment_tag],
        model_path=args.model_path,
        device=args.device,
        strict=args.strict,
    )
    policy = Gr00tSimPolicyWrapper(base_policy, strict=args.strict)

    if args.steering_npz:
        M = load_steering_matrix(
            args.steering_npz, args.steering_beta,
            alpha=args.steering_alpha, key=args.steering_key,
        )
        steer_layer = None if args.steering_pathway == "vl" else args.steering_layer
        hook = ConceptorSteering(
            base_policy.model, M,
            pathway=args.steering_pathway, layer=steer_layer,
        ).register()
        print(
            f"[steer] REGISTERED npz={args.steering_npz} pathway={args.steering_pathway} "
            f"layer={steer_layer} alpha={args.steering_alpha} beta={args.steering_beta} "
            f"key={args.steering_key} token_select={hook.token_select} "
            f"horizon={hook.horizon} M={M.shape}",
            flush=True,
        )
    else:
        print("[steer] BASELINE (no steering hook)", flush=True)

    N15CompatInferenceServer(
        policy, host=args.host, port=args.port, api_token=args.api_token
    ).run()


if __name__ == "__main__":
    main()
