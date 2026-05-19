#!/usr/bin/env python3
"""GR00T N1.5 ZMQ server with a SAFE feature-export endpoint.

This keeps the normal ``get_action`` endpoint used by the RoboCasa rollout
client and adds ``get_action_with_features``.  The extra endpoint returns the
same action plus the pre-action-head backbone token features that SAFE can use
as hidden states.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src/policies/Isaac-GR00T-N1.5"))
sys.path.insert(0, str(REPO_ROOT / "configs/policies"))

from gr00t.eval.service import BaseInferenceServer  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import (  # noqa: E402
    COMPUTE_DTYPE,
    Gr00tPolicy,
    squeeze_dict_values,
    unsqueeze_dict_values,
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


def _cast_feature_tensor(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "float32":
        return tensor.float()
    if dtype_name == "bfloat16":
        return tensor.to(torch.bfloat16)
    if dtype_name == "float16":
        return tensor.half()
    raise ValueError(f"Unsupported feature dtype: {dtype_name}")


class SafeFeaturePolicy:
    def __init__(self, policy: Gr00tPolicy, feature_dtype: str = "float16"):
        self.policy = policy
        self.feature_dtype = feature_dtype

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        return self.policy.get_action(observations)

    def get_modality_config(self) -> dict[str, Any]:
        return self.policy.get_modality_config()

    def get_action_with_features(self, observations: dict[str, Any]) -> dict[str, Any]:
        obs_copy = observations.copy()
        is_batch = self.policy._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)

        normalized_input = self.policy.apply_transforms(obs_copy)

        device_type = (
            "cuda"
            if torch.cuda.is_available() and str(self.policy.device).startswith("cuda")
            else "cpu"
        )
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=COMPUTE_DTYPE)
            if device_type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            backbone_inputs, action_inputs = self.policy.model.prepare_input(normalized_input)
            backbone_outputs = self.policy.model.backbone(backbone_inputs)

            raw_features = backbone_outputs["backbone_features"].detach()
            raw_attention_mask = backbone_outputs.get("backbone_attention_mask")
            if raw_attention_mask is not None:
                raw_attention_mask = raw_attention_mask.detach()

            action_head_outputs = self.policy.model.action_head.get_action(
                backbone_outputs, action_inputs
            )
            self.policy.model.validate_data(
                action_head_outputs, backbone_outputs, is_training=False
            )

        normalized_action = action_head_outputs["action_pred"].float()
        unnormalized_action = self.policy._get_unnormalized_action(normalized_action)

        features = _cast_feature_tensor(raw_features, self.feature_dtype)
        if not is_batch:
            features = features.squeeze(0)
            normalized_action = normalized_action.squeeze(0)
            if raw_attention_mask is not None:
                raw_attention_mask = raw_attention_mask.squeeze(0)
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        response = {
            "action": _to_numpy_tree(unnormalized_action),
            "hidden_states": features.cpu().numpy(),
            "feature_kind": "groot_n15_backbone_features_pre_action_head",
            "normalized_action": normalized_action.detach().cpu().numpy(),
        }
        if raw_attention_mask is not None:
            response["attention_mask"] = raw_attention_mask.cpu().numpy()
        return response


class SafeFeatureInferenceServer(BaseInferenceServer):
    def __init__(
        self,
        model: SafeFeaturePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
    ):
        super().__init__(host=host, port=port, api_token=api_token)
        self.register_endpoint("get_action", model.get_action)
        self.register_endpoint("get_action_with_features", model.get_action_with_features)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T N1.5 SAFE feature ZMQ server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--api-token", default=None)
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="dtype used for serialized hidden states.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_data_config(args.data_config)

    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    safe_policy = SafeFeaturePolicy(policy, feature_dtype=args.feature_dtype)
    server = SafeFeatureInferenceServer(
        safe_policy, host=args.host, port=args.port, api_token=args.api_token
    )
    server.run()


if __name__ == "__main__":
    main()
