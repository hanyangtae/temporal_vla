"""Object-oriented runtime service for the GR00T HTTP server."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .loader import LoadedGrootModel, load_groot_policy
from .preprocess import FINAL_IMAGE_RESOLUTION, decode_b64_image, process_img
from .rng import temporary_inference_seed
from ..safe.features import SafeFeatureExtractor, encode_feature_tensor_blob, feature_metadata
from .schema import (
    GROOT_ENV_LANGUAGE_KEYS,
    GROOT_TO_UNIFIED_ACTION,
    UNIFIED_TO_STATE_KEY,
    build_video_mapping,
    normalize_modality_key,
)
from src.utils.common.serving import health_response, reset_policy


logger = logging.getLogger(__name__)

FALLBACK_ACTION_DIMS = {
    "base_motion": 4,
    "control_mode": 1,
}


class GrootModelNotLoadedError(RuntimeError):
    """Raised when an inference endpoint is called before a policy is loaded."""


@dataclass
class GrootFeatureConfig:
    feature_slice: str = "valid"
    feature_dtype: str = "float16"
    feature_action_horizon: int | None = None

    @classmethod
    def from_args(cls, args: Any | None) -> "GrootFeatureConfig":
        if args is None:
            return cls()
        return cls(
            feature_slice=getattr(args, "feature_slice", "valid"),
            feature_dtype=getattr(args, "feature_dtype", "float16"),
            feature_action_horizon=getattr(args, "feature_action_horizon", None),
        )


class GrootPolicyService:
    """Runtime state and behavior behind the GR00T FastAPI routes."""

    def __init__(
        self,
        *,
        profile: Any | None = None,
        feature_config: GrootFeatureConfig | None = None,
    ):
        self.profile = profile
        self.feature_config = feature_config or GrootFeatureConfig()
        self.policy: Any | None = None
        self.embodiment_tag: Any | None = None
        self.modality_configs: dict | None = None
        self.state_dims: dict[str, int] = {}
        self.device = "cuda"
        self.video_key_mapping: dict[str, str] = {}
        self.warned_missing_video_keys: set[str] = set()
        self.warned_missing_state_keys: set[str] = set()
        self.step_count = 0

    def load_policy(self, *, device: str | None = None) -> None:
        if self.profile is None:
            raise RuntimeError("profile must be set before loading GR00T")

        loaded = load_groot_policy(self.profile, device=device)
        self.set_loaded_model(loaded)

    def set_loaded_model(self, loaded: LoadedGrootModel) -> None:
        self.policy = loaded.policy
        self.embodiment_tag = loaded.embodiment_tag
        self.modality_configs = loaded.modality_configs
        self.state_dims = loaded.state_dims
        self.device = loaded.device
        self._refresh_video_mapping()

    def set_runtime_state(
        self,
        *,
        profile: Any,
        policy: Any,
        modality_configs: dict,
        state_dims: dict[str, int] | None = None,
        embodiment_tag: Any | None = None,
        device: str = "cuda",
    ) -> None:
        """Test/helper hook for setting an already-loaded policy runtime."""
        self.profile = profile
        self.policy = policy
        self.modality_configs = modality_configs
        self.state_dims = state_dims or {}
        self.embodiment_tag = embodiment_tag
        self.device = device
        self._refresh_video_mapping()

    def _refresh_video_mapping(self) -> None:
        if self.modality_configs is None:
            self.video_key_mapping = {}
            return
        video_modality_keys = list(self.modality_configs["video"].modality_keys)
        self.video_key_mapping = build_video_mapping(video_modality_keys)
        logger.info(
            "[video] modality_keys=%s -> mapping=%s",
            video_modality_keys,
            self.video_key_mapping,
        )

    def language_keys(self) -> list[str]:
        keys = list(GROOT_ENV_LANGUAGE_KEYS)
        if self.modality_configs is not None and "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                if key not in keys:
                    keys.append(key)
        return keys

    def action_horizon(self) -> int:
        if self.modality_configs is not None:
            return len(self.modality_configs["action"].delta_indices)
        if self.profile is not None:
            return self.profile.n_action_steps
        return 1

    def build_groot_obs(self, payload: dict[str, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        for unified_key, groot_key in self.video_key_mapping.items():
            b64 = payload.get(unified_key)
            if b64 is None or groot_key in obs:
                continue
            np_img = process_img(decode_b64_image(b64))
            obs[groot_key] = np_img[np.newaxis, np.newaxis, ...]

        for unified_key, groot_key in UNIFIED_TO_STATE_KEY.items():
            val = payload.get(unified_key)
            if val is not None:
                arr = np.array(val, dtype=np.float32).flatten()
                obs[groot_key] = arr[np.newaxis, np.newaxis, :]

        if self.modality_configs is not None:
            h, w = FINAL_IMAGE_RESOLUTION
            zero_img = np.zeros((1, 1, h, w, 3), dtype=np.uint8)

            for vk_raw in self.modality_configs["video"].modality_keys:
                vk = normalize_modality_key(vk_raw, "video")
                if vk not in obs:
                    obs[vk] = zero_img
                    if vk not in self.warned_missing_video_keys:
                        logger.warning("video key %s missing -> zero image fallback", vk)
                        self.warned_missing_video_keys.add(vk)

            for sk_raw in self.modality_configs["state"].modality_keys:
                sk = normalize_modality_key(sk_raw, "state")
                if sk not in obs:
                    local = sk[len("state.") :]
                    dim = self.state_dims.get(local, 1)
                    obs[sk] = np.zeros((1, 1, dim), dtype=np.float32)
                    if sk not in self.warned_missing_state_keys:
                        logger.warning(
                            "state key %s missing -> zero(%dD) fallback",
                            sk,
                            dim,
                        )
                        self.warned_missing_state_keys.add(sk)

        task = (payload.get("task", ""),)
        for key in self.language_keys():
            obs[key] = task
        return obs

    def reset(self) -> dict[str, str]:
        response = reset_policy(self.policy)
        inner = getattr(self.policy, "policy", None) if self.policy is not None else None
        model = getattr(inner, "model", None) if inner is not None else None
        if model is not None and hasattr(model, "reset_predictor"):
            model.reset_predictor()
        return response

    def convert_native_action_to_subkeys(
        self, action_dict: dict[str, Any], latency_ms: float
    ) -> dict[str, Any]:
        if self.profile is None:
            raise RuntimeError("profile must be set before action conversion")

        out: dict[str, Any] = {"latency_ms": latency_ms}
        for key_raw, arr in action_dict.items():
            arr = np.asarray(arr)
            key = key_raw[len("action.") :] if key_raw.startswith("action.") else key_raw
            unified = GROOT_TO_UNIFIED_ACTION.get(key)
            if unified is None:
                unified = f"action.{key}"
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim == 1:
                arr = arr[np.newaxis, :]
            elif arr.ndim == 0:
                arr = arr.reshape(1, 1)
            out[unified] = arr.tolist()

        horizon = self.action_horizon()
        profile_action_names = {action.name for action in self.profile.action_layout}
        for subkey in self.profile.emits_subkeys:
            if subkey in out:
                continue
            local = subkey[len("action.") :]
            if local in profile_action_names:
                dim_slice = self.profile.dim_slice(local)
                dim = dim_slice.stop - dim_slice.start
            else:
                dim = FALLBACK_ACTION_DIMS.get(local, 1)
            out[subkey] = [[0.0] * dim for _ in range(horizon)]
            logger.warning("emit sub-key %s missing from GR00T output, filled zeros", subkey)

        return out

    def _log_debug_call(self, out: dict[str, Any], payload: dict[str, Any]) -> None:
        self.step_count += 1
        if self.step_count > 5:
            return
        logger.info(
            "call=%d horizon=%d pos=%s grip=%s task=%r",
            self.step_count,
            self.action_horizon(),
            np.array(out.get("action.eef_pos", [[0] * 3]))[0].round(4).tolist(),
            np.array(out.get("action.gripper", [[0]]))[0].round(4).tolist(),
            payload.get("task", "")[:40],
        )

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.policy is None:
            raise GrootModelNotLoadedError("model not loaded")

        t0 = time.time()
        groot_obs = self.build_groot_obs(payload)
        with temporary_inference_seed(payload.get("inference_seed")):
            action_dict, _info = self.policy.get_action(groot_obs)
        latency_ms = (time.time() - t0) * 1000

        out = self.convert_native_action_to_subkeys(action_dict, latency_ms)
        self._log_debug_call(out, payload)
        return out

    def act_with_features(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.policy is None:
            raise GrootModelNotLoadedError("model not loaded")

        cfg = self.feature_config
        t0 = time.time()
        groot_obs = self.build_groot_obs(payload)
        with temporary_inference_seed(payload.get("inference_seed")):
            captured = SafeFeatureExtractor(
                self.policy,
                feature_slice=cfg.feature_slice,
                feature_dtype=cfg.feature_dtype,
                feature_action_horizon=cfg.feature_action_horizon,
            ).capture(groot_obs)
        latency_ms = (time.time() - t0) * 1000

        out = self.convert_native_action_to_subkeys(captured.action, latency_ms)
        metadata = captured.metadata
        out["features.hidden_states"] = encode_feature_tensor_blob(
            captured.hidden_states, cfg.feature_dtype
        )
        out["features.kind"] = metadata.feature_kind
        out["features.axes"] = metadata.feature_axes
        out["features.slice"] = metadata.feature_slice
        out["features.exported_action_token_count"] = (
            metadata.exported_action_token_count
        )
        out["features.feature_action_horizon"] = metadata.feature_action_horizon
        out["features.valid_action_horizon"] = metadata.valid_action_horizon
        out["features.model_action_horizon"] = metadata.model_action_horizon
        out["features.num_inference_timesteps"] = metadata.num_inference_timesteps

        self._log_debug_call(out, payload)
        return out

    def health(self) -> dict[str, Any]:
        if self.profile is None:
            return {"status": "not_loaded", "model": "groot"}

        cfg = self.feature_config
        feature_kind, feature_axes = feature_metadata(cfg.feature_slice)
        return health_response(
            policy=self.policy,
            model="groot-n1.6",
            profile=self.profile,
            n_action_steps=self.action_horizon(),
            action_type=self.profile.action_type,
            action_keys=list(self.profile.emits_subkeys),
            embodiment_tag=self.embodiment_tag.value if self.embodiment_tag else None,
            language_keys=self.language_keys(),
            supports_features=True,
            supports_inference_seed=True,
            feature_kind=feature_kind,
            feature_axes=feature_axes,
            feature_slice=cfg.feature_slice,
            feature_dtype=cfg.feature_dtype,
            feature_action_horizon=cfg.feature_action_horizon,
        )
