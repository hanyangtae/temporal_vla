#!/usr/bin/env python3
"""GR00T N1.6 ZMQ server with a SAFE flow-feature endpoint.

The SAFE feature exported here is the flow-matching action-token tensor
immediately before the embodiment-specific Action Decoder projects it to the
velocity field:

    model_output[:, -model_action_horizon:, :][:, :valid_action_horizon, :]

Across denoising steps this is serialized as ``[K, H, D]`` per rollout step.
For RoboCasa PandaOmron, the GR00T N1.6 checkpoint has a model-level max action
horizon of 50, while the embodiment's decoded action horizon is 16. The default
SAFE export keeps only those 16 valid action-token positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/utils"))
sys.path.insert(0, str(REPO_ROOT / "src/policies/Isaac-GR00T"))

from checkpoint_profile import load_profile  # noqa: E402
from gr00t.policy.server_client import PolicyServer  # noqa: E402
from src.policies.groot.loader import load_groot_policy  # noqa: E402


SAFE_FEATURE_KIND_VALID = "groot_n16_dit_valid_action_tokens_pre_velocity"
SAFE_FEATURE_KIND_ALL = "groot_n16_dit_all_action_tokens_pre_velocity"
SAFE_FEATURE_AXES_VALID = ["denoising_step", "valid_action_step", "feature_dim"]
SAFE_FEATURE_AXES_ALL = ["denoising_step", "model_action_token", "feature_dim"]


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


def _cast_feature_tensor(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "float32":
        return tensor.float()
    if dtype_name == "float16":
        return tensor.half()
    raise ValueError(f"Unsupported feature dtype: {dtype_name}")


class SafeN16FeaturePolicy:
    def __init__(
        self,
        sim_policy: Any,
        feature_dtype: str = "float16",
        feature_slice: str = "valid",
    ):
        self.sim_policy = sim_policy
        self.policy = sim_policy.policy
        self.feature_dtype = feature_dtype
        if feature_slice not in {"valid", "all"}:
            raise ValueError(f"Unsupported feature slice: {feature_slice}")
        self.feature_slice = feature_slice

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.sim_policy.reset(options)

    def get_modality_config(self) -> dict[str, Any]:
        return self.sim_policy.get_modality_config()

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.sim_policy.get_action(observation, options)

    def _valid_action_horizon(self) -> int:
        modality_configs = self.sim_policy.get_modality_config()
        return len(modality_configs["action"].delta_indices)

    def _get_action_and_safe_features(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], torch.Tensor, int, int]:
        action_head = self.policy.model.action_head
        model_action_horizon = int(action_head.action_horizon)
        valid_action_horizon = self._valid_action_horizon()
        safe_features: list[torch.Tensor] = []

        def capture_dit_output(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            model_output = output[0] if isinstance(output, tuple) else output
            action_tokens = model_output[:, -model_action_horizon:]
            if self.feature_slice == "valid":
                # decode_action uses only the embodiment's valid leading action steps.
                safe_features.append(action_tokens[:, :valid_action_horizon].detach())
            else:
                safe_features.append(action_tokens.detach())

        handle = action_head.model.register_forward_hook(capture_dit_output)
        try:
            action, _ = self.sim_policy.get_action(observation, options)
        finally:
            handle.remove()

        if not safe_features:
            raise RuntimeError("Failed to capture GR00T N1.6 DiT SAFE features")
        return action, torch.stack(safe_features, dim=1), model_action_horizon, valid_action_horizon

    def get_action_with_features(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with torch.inference_mode():
            action, safe_feature_tensor, model_action_horizon, valid_action_horizon = (
                self._get_action_and_safe_features(observation, options)
            )

        safe_features = _cast_feature_tensor(safe_feature_tensor, self.feature_dtype)
        feature_kind = (
            SAFE_FEATURE_KIND_VALID if self.feature_slice == "valid" else SAFE_FEATURE_KIND_ALL
        )
        feature_axes = (
            SAFE_FEATURE_AXES_VALID if self.feature_slice == "valid" else SAFE_FEATURE_AXES_ALL
        )
        return {
            "action": _to_numpy_tree(action),
            "hidden_states": safe_features.cpu().numpy(),
            "feature_kind": feature_kind,
            "feature_axes": feature_axes,
            "feature_slice": self.feature_slice,
            "exported_action_token_count": int(safe_features.shape[2]),
            "action_horizon": int(safe_features.shape[2]),
            "valid_action_horizon": valid_action_horizon,
            "model_action_horizon": model_action_horizon,
            "num_inference_timesteps": int(safe_features.shape[1]),
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
    )
    server = PolicyServer(safe_policy, host=args.host, port=args.port, api_token=args.api_token)
    server.register_endpoint("get_action_with_features", safe_policy.get_action_with_features)
    server.run()


if __name__ == "__main__":
    main()
