#!/usr/bin/env python3
"""GR00T N1.6 ZMQ server with a SAFE flow-feature endpoint.

The SAFE feature exported here is the flow-matching action-token tensor
immediately before the embodiment-specific Action Decoder projects it to the
velocity field:

    model_output[:, -model_action_horizon:, :][:, :feature_action_horizon, :]

Across denoising steps this is serialized as ``[K, H, D]`` per rollout step.
For RoboCasa PandaOmron, the GR00T N1.6 checkpoint has a model-level max action
horizon of 50, while the embodiment's decoded action horizon is 16. The default
SAFE export keeps only those 16 valid action-token positions. Use
``--feature-action-horizon`` to export a leading subset for action-horizon
ablations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from path_setup import configure_repo_paths  # noqa: E402

configure_repo_paths(include_script_utils=True, include_groot=True)

from checkpoint_profile import load_profile  # noqa: E402
from gr00t.policy.server_client import PolicyServer  # noqa: E402
from src.policies.groot.loader import load_groot_policy  # noqa: E402
from src.policies.groot.rng import temporary_inference_seed  # noqa: E402
from src.policies.groot.safe_features import (  # noqa: E402
    SAFE_FEATURE_AXES_ALL,
    SAFE_FEATURE_AXES_VALID,
    SAFE_FEATURE_KIND_ALL,
    SAFE_FEATURE_KIND_VALID,
    SafeFeatureExtractor,
)


def _to_numpy_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_numpy_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_numpy_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy_tree(v) for v in value)
    return value


class SafeN16FeaturePolicy:
    def __init__(
        self,
        sim_policy: Any,
        feature_dtype: str = "float16",
        feature_slice: str = "valid",
        feature_action_horizon: int | None = None,
    ):
        self.sim_policy = sim_policy
        self.policy = sim_policy.policy
        self.extractor = SafeFeatureExtractor(
            sim_policy,
            feature_dtype=feature_dtype,
            feature_slice=feature_slice,
            feature_action_horizon=feature_action_horizon,
        )

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.sim_policy.reset(options)

    def get_modality_config(self) -> dict[str, Any]:
        return self.sim_policy.get_modality_config()

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.sim_policy.get_action(observation, options)

    def get_action_with_features(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        inference_seed = None if options is None else options.get("inference_seed")
        with temporary_inference_seed(inference_seed):
            captured = self.extractor.capture(observation, options=options)
        metadata = captured.metadata
        return {
            "action": _to_numpy_tree(captured.action),
            "hidden_states": captured.hidden_states.cpu().numpy(),
            **metadata.asdict(),
            "action_horizon": metadata.exported_action_token_count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T N1.6 SAFE feature ZMQ server")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default=None)
    parser.add_argument("--api-token", default=None)
    parser.add_argument(
        "--model-path-override",
        default=None,
        help="Override profile.checkpoint_source.id for host/container path differences.",
    )
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="dtype used for serialized SAFE hidden states.",
    )
    parser.add_argument(
        "--feature-slice",
        choices=("valid", "all"),
        default="valid",
        help=(
            "valid exports only the embodiment's decoded action-token positions; "
            "all exports the model-level max action-token horizon."
        ),
    )
    parser.add_argument(
        "--feature-action-horizon",
        type=int,
        default=None,
        help=(
            "Optional leading action-token count to export. Defaults to the valid "
            "decoded horizon for --feature-slice valid, or the model horizon for "
            "--feature-slice all."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    if args.model_path_override is not None:
        profile.checkpoint_source.id = args.model_path_override
    loaded = load_groot_policy(profile, device=args.device)
    safe_policy = SafeN16FeaturePolicy(
        loaded.policy,
        feature_dtype=args.feature_dtype,
        feature_slice=args.feature_slice,
        feature_action_horizon=args.feature_action_horizon,
    )
    server = PolicyServer(safe_policy, host=args.host, port=args.port, api_token=args.api_token)
    server.register_endpoint("get_action_with_features", safe_policy.get_action_with_features)
    server.run()


if __name__ == "__main__":
    main()
