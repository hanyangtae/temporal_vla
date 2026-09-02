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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.utils.common.feature_blob import decode_feature_blob, encode_feature_blob

from src.policies.safe_capture import SafeForwardCapture
from src.policies.safe_metadata import (
    GROOT_N16_ALL_FEATURE_AXES,
    GROOT_N16_ALL_FEATURE_KIND,
    GROOT_N16_VALID_FEATURE_AXES,
    GROOT_N16_VALID_FEATURE_KIND,
    SafeFeatureMetadata,
    groot_n16_feature_metadata,
)


SAFE_FEATURE_KIND_VALID = GROOT_N16_VALID_FEATURE_KIND
SAFE_FEATURE_KIND_ALL = GROOT_N16_ALL_FEATURE_KIND
SAFE_FEATURE_AXES_VALID = GROOT_N16_VALID_FEATURE_AXES
SAFE_FEATURE_AXES_ALL = GROOT_N16_ALL_FEATURE_AXES

FEATURE_SLICES: tuple[str, ...] = ("valid", "all")
FEATURE_DTYPES: tuple[str, ...] = ("float16", "float32")

# ── 다층 residual 캡처 (COAST Stage1 layer sweep · grid all_token_full 대응) ──
# ZMQ feature_server 의 get_action_with_multilayer_features 와 **동일 수식**을
# HTTP /act_with_features 에서도 쓰기 위한 공유 구현 (2026-09-01 이식).
#   valid/all : action token 만 → K·H mean → [L, D]
#   full      : block 출력 시퀀스 T 전체 보존 → K mean → [L, T, D]
MULTILAYER_FEATURE_KIND = "groot_n16_dit_block_residual_pooled_multilayer"
MULTILAYER_FEATURE_AXES = ["layer", "feature_dim"]
MULTILAYER_FEATURE_KIND_PERT = "groot_n16_dit_block_residual_kmean_perT_multilayer"
MULTILAYER_FEATURE_AXES_PERT = ["layer", "token_pos", "feature_dim"]
CAPTURE_TOKEN_MODES: tuple[str, ...] = ("valid", "all", "full")


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


@dataclass(frozen=True)
class MultilayerCaptureResult:
    """다층 DiT residual 캡처 결과 (transport 무관)."""

    action: dict[str, Any]
    hidden_states: torch.Tensor          # [L, D] 또는 [L, T, D]
    feature_kind: str
    feature_axes: list[str]
    capture_token_mode: str
    layer_indices: list[int]
    token_count: int | None
    valid_action_horizon: int
    model_action_horizon: int
    num_inference_timesteps: int | None
    vl_hidden_states: torch.Tensor | None = None


class MultilayerFeatureExtractor:
    """DiT transformer_blocks residual stream 을 layer 별로 캡처한다.

    `scripts/safe/groot_n16/robocasa/serve/feature_server.py` 의
    `_get_action_and_multilayer_features` 와 같은 수식을 쓰며, HTTP·ZMQ 어느
    transport 에서도 동일 텐서가 나오도록 이 모듈이 단일 출처다.
    """

    def __init__(
        self,
        sim_policy: Any,
        *,
        feature_dtype: str = "float16",
        feature_slice: str = "valid",
        capture_token_mode: str = "full",
        capture_layers: list[int] | None = None,
        capture_vl: bool = False,
    ):
        if feature_dtype not in FEATURE_DTYPES:
            raise ValueError(f"Unsupported feature dtype: {feature_dtype}")
        if capture_token_mode not in CAPTURE_TOKEN_MODES:
            raise ValueError(f"Unsupported capture_token_mode: {capture_token_mode}")
        feature_metadata(feature_slice)
        self.sim_policy = sim_policy
        self.feature_dtype = feature_dtype
        self.feature_slice = feature_slice
        self.capture_token_mode = capture_token_mode
        self.capture_layers = capture_layers
        self.capture_vl = capture_vl

    def _action_head(self) -> Any:
        """transport 별 policy 래핑 차이를 흡수한다.

        ZMQ feature_server 는 raw ``Gr00tPolicy`` 를, HTTP service 는
        ``Gr00tSimPolicyWrapper`` (``.policy`` 로 내부 정책 노출) 를 넘긴다.
        """
        node = self.sim_policy
        for _ in range(3):
            model = getattr(node, "model", None)
            if model is not None and hasattr(model, "action_head"):
                return model.action_head
            node = getattr(node, "policy", None)
            if node is None:
                break
        raise RuntimeError("action_head 를 찾을 수 없다 (policy 래핑 확인 필요)")

    def _resolve_layers(self) -> list[int]:
        blocks = self._action_head().model.transformer_blocks
        if self.capture_layers is None:
            return list(range(len(blocks)))
        return [int(i) for i in self.capture_layers]

    def capture(
        self,
        observation: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
    ) -> MultilayerCaptureResult:
        mode = self.capture_token_mode
        action_head = self._action_head()
        blocks = action_head.model.transformer_blocks
        model_action_horizon = int(action_head.action_horizon)
        valid_action_horizon = len(
            self.sim_policy.get_modality_config()["action"].delta_indices
        )
        layers = self._resolve_layers()
        captured: dict[int, list[torch.Tensor]] = {ell: [] for ell in layers}

        def make_hook(ell: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                if mode == "full":
                    act = out                                   # [B, T, D] 전체 보존
                else:
                    act = out[:, -model_action_horizon:]
                    if mode == "valid":
                        act = act[:, :valid_action_horizon]
                captured[ell].append(act.detach())
            return hook

        handles = [blocks[ell].register_forward_hook(make_hook(ell)) for ell in layers]
        vl_captured: list[torch.Tensor] = []
        if self.capture_vl:
            def vl_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                vl_captured.append(out.detach().mean(dim=1))
            handles.append(action_head.vlln.register_forward_hook(vl_hook))
        try:
            with torch.inference_mode():
                action, _ = self.sim_policy.get_action(observation, options)
        finally:
            for h in handles:
                h.remove()

        num_inference = None
        token_count = None
        pooled_layers: list[torch.Tensor] = []
        for ell in layers:
            feats = captured[ell]
            if not feats:
                raise RuntimeError(f"Failed to capture DiT block {ell} residual stream")
            stack = torch.stack(feats, dim=0)          # [K, B, H_or_T, D]
            num_inference = int(stack.shape[0])
            token_count = int(stack.shape[2])
            if mode == "full":
                pooled_layers.append(stack.mean(dim=0))       # → [B, T, D]
            else:
                pooled_layers.append(stack.mean(dim=(0, 2)))  # → [B, D]
        pooled = torch.stack(pooled_layers, dim=1)     # [B, L, (T,) D]
        hidden = cast_feature_tensor(pooled, self.feature_dtype)[0]
        if mode == "full":
            kind, axes = MULTILAYER_FEATURE_KIND_PERT, list(MULTILAYER_FEATURE_AXES_PERT)
        else:
            kind, axes = MULTILAYER_FEATURE_KIND, list(MULTILAYER_FEATURE_AXES)
        vl_hidden = None
        if vl_captured:
            vl_hidden = cast_feature_tensor(vl_captured[0], self.feature_dtype)[0]
        return MultilayerCaptureResult(
            action=action,
            hidden_states=hidden,
            feature_kind=kind,
            feature_axes=axes,
            capture_token_mode=mode,
            layer_indices=list(layers),
            token_count=token_count,
            valid_action_horizon=valid_action_horizon,
            model_action_horizon=model_action_horizon,
            num_inference_timesteps=num_inference,
            vl_hidden_states=vl_hidden,
        )


def feature_metadata(feature_slice: str) -> tuple[str, list[str]]:
    return groot_n16_feature_metadata(feature_slice)


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


def encode_feature_tensor_blob(
    tensor: torch.Tensor, dtype: str
) -> dict[str, Any]:
    """Cast feature tensor → JSON-safe ``{data, shape, dtype}`` blob."""
    cast = cast_feature_tensor(tensor, dtype)
    arr = cast.detach().cpu().numpy()
    return encode_feature_blob(arr)


def decode_feature_tensor_blob(blob: dict[str, Any]) -> np.ndarray:
    """Decode a JSON-safe feature tensor blob."""
    return decode_feature_blob(blob)


def encode_features_to_base64(
    tensor: torch.Tensor, dtype: str
) -> dict[str, Any]:
    """Compatibility alias for callers that still use the old name."""
    return encode_feature_tensor_blob(tensor, dtype)


def decode_features_from_base64(blob: dict[str, Any]) -> np.ndarray:
    """Compatibility alias for callers that still use the old name."""
    return decode_feature_tensor_blob(blob)
