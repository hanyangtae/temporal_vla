"""GR00T N1.6 SAFE feature extraction utilities.

HTTP `/act_with_features` (`scripts/serve/groot.py`) 와 ZMQ
``get_action_with_features`` (`scripts/safe/groot_n16/robocasa/serve/feature_server.py`)
가 공유하는 DiT pre-velocity feature 캡처/직렬화 helpers.

캡처 대상은 flow-matching DiT 출력에서 Action Decoder 가 velocity field 로
projection 하기 직전의 텐서:

    model_output[:, -model_action_horizon:, :][:, :feature_action_horizon, :]

K denoising step 동안 매 step 의 텐서를 새 축으로 stack 해서
``[B, K, H, D]`` 로 export 한다. RoboCasa PandaOmron N1.6 기준
``model_action_horizon=50``, ``valid_action_horizon=16``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.policies.safe_capture import SafeForwardCapture
from src.policies.safe_metadata import (
    GROOT_N16_ALL_FEATURE_AXES,
    GROOT_N16_ALL_FEATURE_KIND,
    GROOT_N16_VALID_FEATURE_AXES,
    GROOT_N16_VALID_FEATURE_KIND,
    groot_n16_feature_metadata,
)


SAFE_FEATURE_KIND_VALID = GROOT_N16_VALID_FEATURE_KIND
SAFE_FEATURE_KIND_ALL = GROOT_N16_ALL_FEATURE_KIND
SAFE_FEATURE_AXES_VALID = GROOT_N16_VALID_FEATURE_AXES
SAFE_FEATURE_AXES_ALL = GROOT_N16_ALL_FEATURE_AXES

FEATURE_SLICES: tuple[str, ...] = ("valid", "all")
FEATURE_DTYPES: tuple[str, ...] = ("float16", "float32")


@dataclass(frozen=True)
class SafeFeatureMetadata:
    feature_kind: str | None
    feature_axes: list[str] | None
    feature_slice: str | None
    exported_action_token_count: int | None
    feature_action_horizon: int | None
    valid_action_horizon: int | None
    model_action_horizon: int | None
    num_inference_timesteps: int | None

    def asdict(self) -> dict[str, Any]:
        return {
            "feature_kind": self.feature_kind,
            "feature_axes": self.feature_axes,
            "feature_slice": self.feature_slice,
            "exported_action_token_count": self.exported_action_token_count,
            "feature_action_horizon": self.feature_action_horizon,
            "valid_action_horizon": self.valid_action_horizon,
            "model_action_horizon": self.model_action_horizon,
            "num_inference_timesteps": self.num_inference_timesteps,
        }


@dataclass(frozen=True)
class SafeFeatureCaptureResult:
    """Transport-neutral SAFE feature capture result."""

    action: dict[str, Any]
    hidden_states: torch.Tensor
    metadata: SafeFeatureMetadata


class SafeFeatureExtractor:
    """Captures and normalizes GR00T SAFE features for any serving transport."""

    def __init__(
        self,
        sim_policy: Any,
        *,
        feature_dtype: str = "float16",
        feature_slice: str = "valid",
        feature_action_horizon: int | None = None,
    ):
        if feature_dtype not in FEATURE_DTYPES:
            raise ValueError(f"Unsupported feature dtype: {feature_dtype}")
        feature_metadata(feature_slice)
        if feature_action_horizon is not None and feature_action_horizon <= 0:
            raise ValueError(
                f"feature_action_horizon must be positive: {feature_action_horizon}"
            )
        self.sim_policy = sim_policy
        self.feature_dtype = feature_dtype
        self.feature_slice = feature_slice
        self.feature_action_horizon = feature_action_horizon

    def capture(
        self,
        observation: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
    ) -> SafeFeatureCaptureResult:
        with torch.inference_mode():
            captured = capture_dit_features(
                self.sim_policy,
                observation,
                feature_slice=self.feature_slice,
                feature_action_horizon=self.feature_action_horizon,
                options=options,
            )

        hidden_states = cast_feature_tensor(captured["features"], self.feature_dtype)
        feature_kind, feature_axes = feature_metadata(self.feature_slice)
        metadata = SafeFeatureMetadata(
            feature_kind=feature_kind,
            feature_axes=feature_axes,
            feature_slice=self.feature_slice,
            exported_action_token_count=int(hidden_states.shape[2]),
            feature_action_horizon=captured["feature_action_horizon"],
            valid_action_horizon=captured["valid_action_horizon"],
            model_action_horizon=captured["model_action_horizon"],
            num_inference_timesteps=captured["num_inference_timesteps"],
        )
        return SafeFeatureCaptureResult(
            action=captured["action"],
            hidden_states=hidden_states,
            metadata=metadata,
        )


def feature_metadata(feature_slice: str) -> tuple[str, list[str]]:
    return groot_n16_feature_metadata(feature_slice)


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def normalize_feature_metadata(payload: dict[str, Any]) -> SafeFeatureMetadata:
    """Normalize ZMQ, HTTP-response, or VLAClient feature metadata keys.

    Canonical fields match the SAFE rollout pkl schema. Accepted aliases:
    ``feature_kind`` / ``kind`` / ``features.kind`` and the same pattern for
    axes/slice/horizon metadata.
    """
    feature_action_horizon = _optional_int(
        _first_present(
            payload,
            "feature_action_horizon",
            "features.feature_action_horizon",
        )
    )
    exported_action_token_count = _optional_int(
        _first_present(
            payload,
            "exported_action_token_count",
            "features.exported_action_token_count",
        )
    )
    if exported_action_token_count is None:
        exported_action_token_count = feature_action_horizon

    axes = _first_present(payload, "feature_axes", "axes", "features.axes")
    if axes is not None:
        axes = list(axes)

    return SafeFeatureMetadata(
        feature_kind=_first_present(payload, "feature_kind", "kind", "features.kind"),
        feature_axes=axes,
        feature_slice=_first_present(payload, "feature_slice", "slice", "features.slice"),
        exported_action_token_count=exported_action_token_count,
        feature_action_horizon=feature_action_horizon,
        valid_action_horizon=_optional_int(
            _first_present(payload, "valid_action_horizon", "features.valid_action_horizon")
        ),
        model_action_horizon=_optional_int(
            _first_present(payload, "model_action_horizon", "features.model_action_horizon")
        ),
        num_inference_timesteps=_optional_int(
            _first_present(
                payload,
                "num_inference_timesteps",
                "features.num_inference_timesteps",
            )
        ),
    )


def resolve_feature_action_horizon(
    *,
    feature_slice: str,
    model_action_horizon: int,
    valid_action_horizon: int,
    feature_action_horizon: int | None,
) -> int:
    if feature_slice not in FEATURE_SLICES:
        raise ValueError(f"Unsupported feature slice: {feature_slice}")
    if feature_action_horizon is not None and feature_action_horizon <= 0:
        raise ValueError(
            f"feature_action_horizon must be positive: {feature_action_horizon}"
        )
    max_export = (
        valid_action_horizon if feature_slice == "valid" else model_action_horizon
    )
    fah = feature_action_horizon or max_export
    if fah > max_export:
        raise ValueError(
            "feature_action_horizon exceeds exportable horizon: "
            f"{fah} > {max_export} (feature_slice={feature_slice})"
        )
    return fah


def capture_dit_features(
    sim_policy: Any,
    observation: dict[str, Any],
    *,
    feature_slice: str = "valid",
    feature_action_horizon: int | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``sim_policy.get_action`` 을 실행하면서 DiT pre-velocity 출력을 hook 으로 가로챈다.

    Hook 은 ``sim_policy.policy.model.action_head.model`` 에 호출당 한 번 붙고
    반환 직전에 제거된다.

    Returns:
        dict
          action: native GR00T action dict (``sim_policy.get_action`` 원형)
          features: ``[B, K, H, D]`` torch.Tensor (device 위)
          model_action_horizon: int
          valid_action_horizon: int
          feature_action_horizon: int
          num_inference_timesteps: int (=K)
    """
    policy = sim_policy.policy
    action_head = policy.model.action_head
    model_action_horizon = int(action_head.action_horizon)
    valid_action_horizon = len(
        sim_policy.get_modality_config()["action"].delta_indices
    )
    fah = resolve_feature_action_horizon(
        feature_slice=feature_slice,
        model_action_horizon=model_action_horizon,
        valid_action_horizon=valid_action_horizon,
        feature_action_horizon=feature_action_horizon,
    )

    def slice_action_tokens(output: torch.Tensor) -> torch.Tensor:
        model_output = output[0] if isinstance(output, tuple) else output
        action_tokens = model_output[:, -model_action_horizon:]
        return action_tokens[:, :fah]

    with SafeForwardCapture(
        action_head.model,
        "post",
        slice_action_tokens,
    ) as capture:
        action, _ = sim_policy.get_action(observation, options)

    if not capture.buf:
        raise RuntimeError("Failed to capture GR00T N1.6 DiT SAFE features")

    feature_tensor = torch.stack(capture.buf, dim=1)
    return {
        "action": action,
        "features": feature_tensor,
        "model_action_horizon": model_action_horizon,
        "valid_action_horizon": valid_action_horizon,
        "feature_action_horizon": fah,
        "num_inference_timesteps": int(feature_tensor.shape[1]),
    }


def cast_feature_tensor(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "float32":
        return tensor.float()
    if dtype == "float16":
        return tensor.half()
    raise ValueError(f"Unsupported feature dtype: {dtype}")


def encode_features_to_base64(
    tensor: torch.Tensor, dtype: str
) -> dict[str, Any]:
    """Cast feature tensor → JSON-safe ``{data, shape, dtype}`` blob."""
    cast = cast_feature_tensor(tensor, dtype)
    arr = cast.detach().cpu().numpy()
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def decode_features_from_base64(blob: dict[str, Any]) -> np.ndarray:
    """``encode_features_to_base64`` 역연산. shape/dtype 메타로 reshape."""
    raw = base64.b64decode(blob["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(blob["dtype"]))
    return arr.reshape(blob["shape"]).copy()
