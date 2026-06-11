"""SAFE feature 추출 hook (lerobot 정책용, HTTP serve 에서 사용).

SAFE latent 한 점의 의미
------------------------
한 점 = **policy 추론(action chunk) 1회**에서 얻은, action 이 velocity field
(flow-matching) 또는 token logit(autoregressive)으로 디코딩되기 **직전**의
마지막-레이어 hidden state. SAFE 논문(arXiv 2506.09937)의 feature 정의와 동일.

대부분의 lerobot 정책은 내부 action queue 가 빌 때만(= ``config.n_action_steps`` env step
마다) 새 추론을 돌린다. 그 사이 step 의 ``select_action`` 은 버퍼된 action 을 popleft 만
하고 추론을 하지 않으므로 latent 가 생기지 않는다. GR00T N1.5 feature collection은
N1.6 SAFE collector와 같은 chunk execution을 맞추기 위해 ``predict_action_chunk``를
직접 호출하고, 그 action chunk와 같은 추론에서 나온 latent를 record한다.

모델별 hook 지점 (v0.5.1 소스에서 검증)
  pi0, pi05 : ``model.action_out_proj`` 입력            (denoise_step 마다 [B, H, D])
  xvla      : ``model.transformer.action_decoder`` 입력  (denoise step 마다 [B, H, D])
  groot     : ``action_head.model``(DiT) 출력의 마지막 H (denoise step 마다 [B, H, D])
  pi0_fast  : ``paligemma.lm_head`` 입력                 (생성 토큰마다 [B, 1, D])

per-step 저장 텐서
  flow-matching : ``[K_denoise, H_action, D]``
  pi0_fast      : ``[1, n_action_tokens, D]``  (downstream SAFE 집계가 3D ``[K,H,D]``
                  를 요구하므로 singleton K 로 감싼다)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.policies.safe_capture import SafeForwardCapture  # noqa: E402
from src.policies.safe_metadata import (  # noqa: E402
    GROOT_N15_DIT_BLOCK_RESIDUAL_FEATURE_AXES,
    GROOT_N15_DIT_BLOCK_RESIDUAL_FEATURE_KIND,
    GROOT_N15_VL_FEATURE_KIND,
    GROOT_VL_FEATURE_AXES,
    lerobot_feature_axes,
    lerobot_feature_kind,
)


FLOW_MATCHING_TYPES = {"pi0", "pi05", "xvla", "groot"}
AUTOREGRESSIVE_TYPES = {"pi0_fast"}
SUPPORTED_TYPES = FLOW_MATCHING_TYPES | AUTOREGRESSIVE_TYPES


def _resolve_target(
    policy: Any, policy_type: str
) -> tuple[torch.nn.Module, str, Callable[[torch.Tensor], torch.Tensor] | None]:
    """SAFE hook 대상 ``(module, mode, slicer)`` 반환.

    mode 는 ``"pre"``(forward_pre_hook, 입력 캡처) 또는 ``"post"``(forward_hook, 출력 캡처).
    slicer 는 캡처 텐서를 action-token 위치로 자르는 함수(없으면 None).
    """
    if policy_type in ("pi0", "pi05"):
        # action_out_proj 입력 suffix_out 은 이미 [B, chunk_size, width] 로 슬라이싱됨.
        return policy.model.action_out_proj, "pre", None
    if policy_type == "xvla":
        # action_decoder 입력은 norm(x[:, :num_actions]) = [B, num_actions, hidden].
        return policy.model.transformer.action_decoder, "pre", None
    if policy_type == "groot":
        head = policy._groot_model.action_head
        horizon = int(head.action_horizon)
        # DiT 출력은 (state + future + action) 토큰 → 마지막 action_horizon 개만.
        return head.model, "post", (lambda t: t[:, -horizon:])
    if policy_type == "pi0_fast":
        return policy.model.paligemma_with_expert.paligemma.lm_head, "pre", None
    raise ValueError(f"Unsupported policy_type: {policy_type}")


def _resolve_vl_target(policy: Any, policy_type: str) -> torch.nn.Module | None:
    if policy_type != "groot":
        return None
    head = policy._groot_model.action_head
    return getattr(head, "vlln", None)


class SafeFeatureCapture:
    """lerobot 정책에 hook 을 걸어 SAFE feature 를 누적하는 context manager.

    hook 이 한 번 발화할 때마다 ``self.buf`` 에 텐서 1개 append:
      - flow-matching: denoise step 마다 ``[B, H, D]`` (총 K 개)
      - pi0_fast: 생성 토큰마다 ``[B, 1, D]`` (총 n_tokens 개)
    """

    def __init__(
        self,
        policy: Any,
        policy_type: str,
        *,
        capture_vl: bool = False,
        groot_dit_layers: tuple[int, ...] | list[int] | None = None,
    ):
        if policy_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported policy_type: {policy_type}")
        self.policy = policy
        self.policy_type = policy_type
        self.groot_dit_layers = (
            None if groot_dit_layers is None else [int(layer) for layer in groot_dit_layers]
        )
        self.block_modules: list[torch.nn.Module] = []
        self.block_bufs: dict[int, list[torch.Tensor]] = {}
        self._block_capture_ctxs: list[SafeForwardCapture] = []
        self.groot_action_horizon: int | None = None
        if self.groot_dit_layers is not None:
            if policy_type != "groot":
                raise ValueError("groot_dit_layers is supported only for policy_type='groot'")
            head = policy._groot_model.action_head
            blocks = head.model.transformer_blocks
            self.groot_action_horizon = int(head.action_horizon)
            self.block_modules = [blocks[layer] for layer in self.groot_dit_layers]
            self.module = None
            self.mode = None
            self.slicer = None
        else:
            self.module, self.mode, self.slicer = _resolve_target(policy, policy_type)
        self.vl_module = _resolve_vl_target(policy, policy_type) if capture_vl else None
        self.buf: list[torch.Tensor] = []
        self.vl_buf: list[torch.Tensor] = []
        self._capture_ctx: SafeForwardCapture | None = None
        self._vl_capture_ctx: SafeForwardCapture | None = None

    def __enter__(self) -> "SafeFeatureCapture":
        self.buf = []
        self.vl_buf = []
        self.block_bufs = {}
        self._block_capture_ctxs = []
        if self.groot_dit_layers is not None:
            for layer, module in zip(self.groot_dit_layers, self.block_modules):
                ctx = SafeForwardCapture(
                    module,
                    "post",
                    None,
                    to_cpu=True,
                    dtype=torch.float32,
                )
                ctx.__enter__()
                self._block_capture_ctxs.append(ctx)
                self.block_bufs[layer] = ctx.buf
        else:
            self._capture_ctx = SafeForwardCapture(
                self.module,
                self.mode,
                self.slicer,
                to_cpu=True,
                dtype=torch.float32,
            )
            self._capture_ctx.__enter__()
        if self.vl_module is not None:
            self._vl_capture_ctx = SafeForwardCapture(
                self.vl_module,
                "post",
                lambda t: t.mean(dim=1),
                to_cpu=True,
                dtype=torch.float32,
            )
            self._vl_capture_ctx.__enter__()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        if self._vl_capture_ctx is not None:
            self.vl_buf = list(self._vl_capture_ctx.buf)
            self._vl_capture_ctx.__exit__(*_exc)
            self._vl_capture_ctx = None
        if self._block_capture_ctxs:
            for layer, ctx in zip(self.groot_dit_layers or [], self._block_capture_ctxs):
                self.block_bufs[layer] = list(ctx.buf)
                ctx.__exit__(*_exc)
            self._block_capture_ctxs = []
        if self._capture_ctx is not None:
            self.buf = list(self._capture_ctx.buf)
            self._capture_ctx.__exit__(*_exc)
            self._capture_ctx = None
        return False

    def assemble(self) -> np.ndarray | None:
        """캡처된 텐서를 per-step SAFE latent 으로 결합.

        이번 호출에서 추론이 발화하지 않았으면(queue pop 만) None 반환. GR00T N1.5
        collect는 queue pop이 아니라 direct chunk inference이므로 매 호출마다 latent가
        생긴다.
        flow-matching → ``[K, H, D]`` (float16), pi0_fast → ``[1, n_tokens, D]`` (float16).
        """
        if not self.buf:
            return None
        if self.policy_type in AUTOREGRESSIVE_TYPES:
            # [B, 1, D] 들을 token 축으로 concat → [B, n_tokens, D] → [n_tokens, D] → [1, n_tokens, D]
            toks = torch.cat(self.buf, dim=1)[0]
            arr = toks.unsqueeze(0).numpy()
        else:
            # [B, H, D] 들을 K 축으로 stack → [K, B, H, D] → [K, H, D]
            arr = torch.stack(self.buf, dim=0)[:, 0].numpy()
        return arr.astype(np.float16)

    def assemble_blocks(self) -> np.ndarray | None:
        """GR00T DiT block residual features: K mean over ``[K, B, T, D]`` -> ``[L, T, D]``."""
        if self.groot_dit_layers is None:
            return None
        if not self.groot_dit_layers:
            raise ValueError("groot_dit_layers must not be empty")
        pooled_layers: list[torch.Tensor] = []
        any_capture = False
        for layer in self.groot_dit_layers:
            feats = self.block_bufs.get(layer, [])
            if feats:
                any_capture = True
            else:
                raise RuntimeError(f"Failed to capture GR00T DiT block {layer} residual stream")
            stack = torch.stack(feats, dim=0)  # [K, B, T, D]
            pooled_layers.append(stack[:, 0].mean(dim=0))  # [T, D]
        if not any_capture:
            return None
        return torch.stack(pooled_layers, dim=0).numpy().astype(np.float16)

    def block_num_inference_timesteps(self) -> int | None:
        if self.groot_dit_layers is None:
            return None
        first_layer = self.groot_dit_layers[0]
        feats = self.block_bufs.get(first_layer, [])
        return int(len(feats)) if feats else None

    def assemble_vl(self) -> np.ndarray | None:
        """GR00T VL pathway feature: ``action_head.vlln`` output seq-mean-pool."""
        if not self.vl_buf:
            return None
        return self.vl_buf[0][0].numpy().astype(np.float16)


def run_with_features(
    policy: Any,
    batch: dict[str, Any],
    policy_type: str,
    *,
    capture_vl: bool = False,
    groot_dit_layers: tuple[int, ...] | list[int] | None = None,
) -> tuple[torch.Tensor, np.ndarray | None, list[str] | None, dict[str, Any]]:
    """SAFE hook 을 건 채 ``select_action`` 을 실행.

    Returns ``(action, hidden_states | None, feature_axes | None, meta)``.
    GR00T N1.5는 N1.6 SAFE collection과 같은 chunk execution을 맞추기 위해
    ``predict_action_chunk``를 호출한다. 다른 policy에서 추론이 발화하지 않은
    step(queue pop 만)에서는 hidden_states=None — 호출자는 hidden_states가 있을 때만
    record 한다. ``capture_vl``은 GR00T에서만 ``action_head.vlln`` 출력의 sequence
    mean-pool을 optional VL pathway feature로 추가한다. ``groot_dit_layers``가 있으면
    GR00T DiT final output 대신 선택한 ``transformer_blocks`` residual stream을
    ``[layer, model_token, feature_dim]``으로 캡처한다.
    """
    cap = SafeFeatureCapture(
        policy,
        policy_type,
        capture_vl=capture_vl,
        groot_dit_layers=groot_dit_layers,
    )
    with torch.inference_mode(), cap:
        if policy_type == "groot" and hasattr(policy, "predict_action_chunk"):
            action = policy.predict_action_chunk(batch)
        else:
            action = policy.select_action(batch)

    hidden = cap.assemble_blocks() if groot_dit_layers is not None else cap.assemble()
    if hidden is None:
        return action, None, None, {}

    if groot_dit_layers is not None:
        axes = list(GROOT_N15_DIT_BLOCK_RESIDUAL_FEATURE_AXES)
        meta = {
            "feature_kind": GROOT_N15_DIT_BLOCK_RESIDUAL_FEATURE_KIND,
            "feature_axes": axes,
            "num_inference_timesteps": cap.block_num_inference_timesteps(),
            "capture_layers": [int(layer) for layer in groot_dit_layers],
            "layer_count": int(hidden.shape[0]),
            "token_count": int(hidden.shape[1]),
            "feature_dim": int(hidden.shape[2]),
            "model_action_horizon": cap.groot_action_horizon,
        }
    else:
        axes = lerobot_feature_axes(policy_type)
    if groot_dit_layers is None and policy_type in AUTOREGRESSIVE_TYPES:
        meta = {
            "feature_kind": lerobot_feature_kind(policy_type),
            "feature_axes": axes,
            "num_inference_timesteps": None,
            "n_action_tokens": int(hidden.shape[1]),
            "feature_dim": int(hidden.shape[2]),
        }
    elif groot_dit_layers is None:
        meta = {
            "feature_kind": lerobot_feature_kind(policy_type),
            "feature_axes": axes,
            "num_inference_timesteps": int(hidden.shape[0]),
            "action_horizon": int(hidden.shape[1]),
            "feature_dim": int(hidden.shape[2]),
        }
    vl_hidden = cap.assemble_vl()
    if vl_hidden is not None:
        meta["vl_hidden_states"] = vl_hidden
        meta["vl_feature_kind"] = GROOT_N15_VL_FEATURE_KIND
        meta["vl_feature_axes"] = list(GROOT_VL_FEATURE_AXES)
        meta["vl_feature_dim"] = int(vl_hidden.shape[-1])
    return action, hidden, axes, meta
