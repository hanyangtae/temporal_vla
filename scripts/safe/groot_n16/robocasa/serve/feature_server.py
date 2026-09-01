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
from src.policies.groot.core.loader import load_groot_policy  # noqa: E402
from src.policies.groot.core.rng import temporary_inference_seed  # noqa: E402
from src.policies.groot.safe.features import (  # noqa: E402
    SAFE_FEATURE_AXES_ALL,
    SAFE_FEATURE_AXES_VALID,
    SAFE_FEATURE_KIND_ALL,
    SAFE_FEATURE_KIND_VALID,
    SafeFeatureExtractor,
    cast_feature_tensor,
)
from src.utils.common.serving import serve_provenance  # noqa: E402
from src.policies.safe_metadata import (  # noqa: E402
    GROOT_N16_VL_FEATURE_KIND,
    GROOT_VL_FEATURE_AXES,
)

# Multi-layer capture (COAST A.10.2 Stage1 layer sweep용): DiT transformer_blocks 의
# residual stream(inner_dim=1536)을 layer별로 캡처.
#
# capture_token_mode 에 따라 token 축 처리가 다름:
#   - "valid"  (현행): action 16 token 만, K·H mean → env step 당 [L, D]
#   - "all"           : action 50 token (model_action_horizon), K·H mean → [L, D]
#   - "full"          : transformer block output 의 sequence T 전체, K mean 만
#                       (T 축 보존) → [L, T, D]  (COAST p24 명세 모방용)
MULTILAYER_FEATURE_KIND = "groot_n16_dit_block_residual_pooled_multilayer"
MULTILAYER_FEATURE_AXES = ["layer", "feature_dim"]
MULTILAYER_FEATURE_KIND_PERT = "groot_n16_dit_block_residual_kmean_perT_multilayer"
MULTILAYER_FEATURE_AXES_PERT = ["layer", "token_pos", "feature_dim"]
CAPTURE_TOKEN_MODES = ("valid", "all", "full")

# VL(goal) pathway feature: action_head.vlln 출력(post-LayerNorm VL features, D=2048).
# get_action 당 1회 발화 → VL token 축(seq) mean-pool (NOTALL: GR00T VL-SA 는 pooling 이득).
# DiT multilayer 와 동일 forward 에서 함께 캡처해 rollout 비용 중복 없음.
VL_FEATURE_KIND = GROOT_N16_VL_FEATURE_KIND
VL_FEATURE_AXES = GROOT_VL_FEATURE_AXES


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
        profile: Any | None = None,
    ):
        self.sim_policy = sim_policy
        # docs/04 — 응답에 실을 ckpt 식별자 원천
        self.profile = profile
        self.policy = sim_policy.policy
        self.extractor = SafeFeatureExtractor(
            sim_policy,
            feature_dtype=feature_dtype,
            feature_slice=feature_slice,
            feature_action_horizon=feature_action_horizon,
        )
        # Multi-layer capture (COAST Stage1 layer sweep)용. None=전체 layer 캡처.
        self.capture_layers: list[int] | None = None
        # Multi-layer hook 의 token 축 처리 모드. CAPTURE_TOKEN_MODES 참조.
        self.capture_token_mode: str = "valid"
        # VL(goal) pathway feature 동시 캡처 여부 (action_head.vlln seq-mean-pool).
        self.capture_vl: bool = False

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

    # ─── Multi-layer residual stream capture (COAST Stage1 layer sweep) ──────────
    def _resolve_capture_layers(self) -> list[int]:
        blocks = self.policy.model.action_head.model.transformer_blocks
        n = len(blocks)
        if self.capture_layers is None:
            return list(range(n))
        return [int(i) for i in self.capture_layers]

    def _get_action_and_multilayer_features(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], torch.Tensor, list[int], int, int, int, int]:
        """선택한 DiT transformer_blocks 의 residual stream 을 layer별로 캡처.

        ``self.capture_token_mode`` 에 따라 token 축 처리가 다르다:
          - ``valid``: action token (마지막 model_action_horizon, valid 슬라이스) 만,
            K·H mean → env step 당 [B, L, D] (COAST global 집계 = K·H mean).
          - ``all``  : action token (마지막 model_action_horizon 전체), K·H mean → [B, L, D].
          - ``full`` : transformer block output 의 sequence T 전체 보존, K mean 만 →
            env step 당 [B, L, T, D] (COAST p24 명세 "S=49 전체 mean" 모방 등 사후 pool 용).

        반환 마지막 정수는 token_count (env step 당 token 축 길이):
          valid → valid_action_horizon, all → model_action_horizon, full → T.
        """
        mode = self.capture_token_mode
        if mode not in CAPTURE_TOKEN_MODES:
            raise ValueError(f"Unsupported capture_token_mode: {mode}")
        action_head = self.policy.model.action_head
        blocks = action_head.model.transformer_blocks
        model_action_horizon = int(action_head.action_horizon)
        valid_action_horizon = len(
            self.sim_policy.get_modality_config()["action"].delta_indices
        )
        layers = self._resolve_capture_layers()
        captured: dict[int, list[torch.Tensor]] = {ell: [] for ell in layers}

        def make_hook(ell: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                if mode == "full":
                    act = out  # [B, T, D] — T 전체 보존
                else:
                    act = out[:, -model_action_horizon:]  # 마지막 50
                    if mode == "valid":
                        act = act[:, :valid_action_horizon]  # 첫 16
                captured[ell].append(act.detach())
            return hook

        handles = [blocks[ell].register_forward_hook(make_hook(ell)) for ell in layers]
        # VL(goal) pathway: vlln 출력(get_action당 1회) seq-mean-pool 동시 캡처.
        vl_captured: list[torch.Tensor] = []
        if self.capture_vl:
            def vl_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                vl_captured.append(out.detach().mean(dim=1))  # [B, seq, D] → [B, D]
            handles.append(action_head.vlln.register_forward_hook(vl_hook))
        try:
            action, _ = self.sim_policy.get_action(observation, options)
        finally:
            for h in handles:
                h.remove()

        num_inference = None
        pooled_layers: list[torch.Tensor] = []
        token_count = None
        for ell in layers:
            feats = captured[ell]
            if not feats:
                raise RuntimeError(f"Failed to capture DiT block {ell} residual stream")
            stack = torch.stack(feats, dim=0)  # [K, B, H_or_T, D]
            num_inference = int(stack.shape[0])
            token_count = int(stack.shape[2])
            if mode == "full":
                pooled_layers.append(stack.mean(dim=0))  # K mean → [B, T, D]
            else:
                pooled_layers.append(stack.mean(dim=(0, 2)))  # K·H mean → [B, D]
        pooled = torch.stack(pooled_layers, dim=1)  # [B, L, D] or [B, L, T, D]
        vl_pooled = vl_captured[0] if vl_captured else None  # [B, D_bb] or None
        return (action, pooled, layers, model_action_horizon, valid_action_horizon,
                num_inference, token_count, vl_pooled)

    def get_action_with_multilayer_features(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # per-call inference_seed 적용 — get_action_with_features 와 동일 규약.
        # (2026-09-01 수정: 여기 빠져 있어 다층 캡처가 호출 seed 를 무시했고,
        #  HTTP 이식본과 값이 갈렸다. denoise RNG 가 seed 로 고정돼야 replay·
        #  셀-paired 비교가 성립한다.)
        inference_seed = None if options is None else options.get("inference_seed")
        with temporary_inference_seed(inference_seed), torch.inference_mode():
            (action, pooled, layers, model_action_horizon, valid_action_horizon,
             num_inference, token_count, vl_pooled) = (
                self._get_action_and_multilayer_features(observation, options)
            )
        feats = cast_feature_tensor(pooled, self.extractor.feature_dtype)
        # full 모드: [B, L, T, D] → batch 0 → [L, T, D]
        # 기타 (valid/all): [B, L, D] → batch 0 → [L, D]
        mode = self.capture_token_mode
        if mode == "full":
            kind = MULTILAYER_FEATURE_KIND_PERT
            axes = MULTILAYER_FEATURE_AXES_PERT
        else:
            kind = MULTILAYER_FEATURE_KIND
            axes = MULTILAYER_FEATURE_AXES
        response = {
            "action": _to_numpy_tree(action),
            "hidden_states": feats[0].cpu().numpy(),
            "feature_kind": kind,
            "feature_axes": axes,
            "feature_slice": self.extractor.feature_slice,
            "capture_token_mode": mode,
            "layer_indices": list(layers),
            "feature_dim": int(feats.shape[-1]),
            "token_count": int(token_count) if token_count is not None else None,
            "valid_action_horizon": valid_action_horizon,
            "model_action_horizon": model_action_horizon,
            "num_inference_timesteps": num_inference,
            # docs/04 규약 — rollout 인덱스의 machine·ckpt 원천. ZMQ 는 /health 가
            # 없으므로 매 응답에 실어 보낸다 (HTTP serve 와 동일 헬퍼).
            **serve_provenance(getattr(self, "profile", None)),
        }
        if vl_pooled is not None:
            vl_feats = cast_feature_tensor(vl_pooled, self.extractor.feature_dtype)
            response["vl_hidden_states"] = vl_feats[0].cpu().numpy()  # [D_bb]
            response["vl_feature_kind"] = VL_FEATURE_KIND
            response["vl_feature_axes"] = list(VL_FEATURE_AXES)
            response["vl_feature_dim"] = int(vl_feats.shape[-1])
        return response


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
    # ─── Multi-layer capture (COAST Stage1 layer sweep) ───
    parser.add_argument(
        "--capture-layers",
        default=None,
        help=(
            "Comma-separated DiT transformer_block indices for "
            "get_action_with_multilayer_features. Default(None)=all blocks."
        ),
    )
    parser.add_argument(
        "--capture-token-mode",
        choices=CAPTURE_TOKEN_MODES,
        default="valid",
        help=(
            "Multi-layer hook 의 token 축 처리. "
            "valid=action 16(K·H mean), all=action 50(K·H mean), "
            "full=T 전체 보존(K mean). full 응답은 [L, T, D]."
        ),
    )
    parser.add_argument(
        "--capture-vl",
        action="store_true",
        help=(
            "multilayer endpoint 에서 VL(goal) pathway feature(action_head.vlln "
            "seq-mean-pool, D=2048)도 동시 캡처해 vl_hidden_states 로 반환."
        ),
    )
    # ─── COAST conceptor steering (옵션) ───
    parser.add_argument(
        "--steering-npz",
        default=None,
        help="Conceptor npz path. 지정 시 steering hook을 영구 등록한다.",
    )
    parser.add_argument("--steering-beta", type=float, default=0.3)
    parser.add_argument("--steering-alpha", type=float, default=None)
    parser.add_argument(
        "--steering-key",
        choices=("C_steer", "C_success", "C_failure"),
        default="C_steer",
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="DiT block index to steer (pathway=dit). None=action_head.model 출력(pre-velocity).",
    )
    parser.add_argument(
        "--steering-pathway",
        choices=("dit", "vl"),
        default="dit",
        help=(
            "Steering 주입 pathway. dit=motor(action token, K회/step), "
            "vl=goal(action_head.vlln, 전체 VL token, get_action당 1회·K step 전파)."
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
        profile=profile,
    )
    if args.capture_layers:
        safe_policy.capture_layers = [int(x) for x in args.capture_layers.split(",")]
    safe_policy.capture_token_mode = args.capture_token_mode
    safe_policy.capture_vl = args.capture_vl

    # COAST conceptor steering hook (옵션). DiT layer(또는 pre-velocity 출력)에 영구 등록 →
    # get_action / get_action_with_features 모두 steered. β=0 이면 M=I (baseline 동일).
    steering = None
    if args.steering_npz:
        sys.path.insert(0, str(REPO_ROOT / "scripts/serve"))
        from steering_hooks import ConceptorSteering, load_steering_matrix

        M = load_steering_matrix(
            args.steering_npz,
            beta=args.steering_beta,
            alpha=args.steering_alpha,
            key=args.steering_key,
        )
        steer_layer = None if args.steering_pathway == "vl" else args.steering_layer
        steering = ConceptorSteering(
            safe_policy.policy.model,
            M,
            pathway=args.steering_pathway,
            layer=steer_layer,
        ).register()
        print(
            f"[steering] ConceptorSteering 등록: npz={args.steering_npz} "
            f"pathway={args.steering_pathway} beta={args.steering_beta} "
            f"alpha={args.steering_alpha} key={args.steering_key} "
            f"layer={steer_layer} token_select={steering.token_select} "
            f"horizon={steering.horizon}",
            flush=True,
        )

    server = PolicyServer(safe_policy, host=args.host, port=args.port, api_token=args.api_token)
    server.register_endpoint("get_action_with_features", safe_policy.get_action_with_features)
    server.register_endpoint(
        "get_action_with_multilayer_features", safe_policy.get_action_with_multilayer_features
    )
    server.run()


if __name__ == "__main__":
    main()
