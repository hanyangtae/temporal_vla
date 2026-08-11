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
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES,
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND,
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_AXES,
    GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_KIND,
    GROOT_N15_VL_FEATURE_KIND,
    GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_AXES,
    GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_KIND,
    GROOT_VL_FEATURE_AXES,
    PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES,
    PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND,
    lerobot_feature_axes,
    lerobot_feature_kind,
)


FLOW_MATCHING_TYPES = {"pi0", "pi05", "xvla", "groot"}
AUTOREGRESSIVE_TYPES = {"pi0_fast"}
SUPPORTED_TYPES = FLOW_MATCHING_TYPES | AUTOREGRESSIVE_TYPES

# GR00T DiT block residual capture 의 token 풀링 모드 (exp3(구 pq3) COAST 토큰 축 정렬).
#   action_token_mean: 구·default — 마지막 action_horizon 토큰 mean → [L, K, D]
#   all_token_full   : 신규 — 전체 시퀀스 토큰(state+future+action) 보존 → [L, K, T, D]
#                      (fp16, fit 수집 전용 — mean 은 fit 시점에 수행)
GROOT_DIT_TOKEN_POOLS = ("action_token_mean", "all_token_full")
# GR00T VL pathway capture 지점.
#   vlln_mean       : 구·default — action_head.vlln 출력 seq-mean → [D_vl]
#   post_vl_sa_full : 신규 — vl_self_attention 출력(=DiT cross-attn 입력) full-token → [T_vl, D_vl]
GROOT_VL_CAPTURE_POINTS = ("vlln_mean", "post_vl_sa_full")


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


def _resolve_vl_target(
    policy: Any, policy_type: str, vl_capture_point: str = "vlln_mean"
) -> torch.nn.Module | None:
    if policy_type != "groot":
        return None
    head = policy._groot_model.action_head
    if vl_capture_point == "post_vl_sa_full":
        return getattr(head, "vl_self_attention", None)
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
        pi05_expert_layers: tuple[int, ...] | list[int] | None = None,
        groot_dit_token_pool: str = "action_token_mean",
        vl_capture_point: str = "vlln_mean",
    ):
        if policy_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported policy_type: {policy_type}")
        if groot_dit_token_pool not in GROOT_DIT_TOKEN_POOLS:
            raise ValueError(
                f"Unsupported groot_dit_token_pool: {groot_dit_token_pool} "
                f"(expected {GROOT_DIT_TOKEN_POOLS})"
            )
        if vl_capture_point not in GROOT_VL_CAPTURE_POINTS:
            raise ValueError(
                f"Unsupported vl_capture_point: {vl_capture_point} "
                f"(expected {GROOT_VL_CAPTURE_POINTS})"
            )
        self.policy = policy
        self.policy_type = policy_type
        self.groot_dit_token_pool = groot_dit_token_pool
        self.vl_capture_point = vl_capture_point
        self.groot_dit_layers = (
            None if groot_dit_layers is None else [int(layer) for layer in groot_dit_layers]
        )
        self.pi05_expert_layers = (
            None if pi05_expert_layers is None else [int(layer) for layer in pi05_expert_layers]
        )
        self.block_modules: list[torch.nn.Module] = []
        self.block_bufs: dict[int, list[torch.Tensor]] = {}
        self._block_capture_ctxs: list[SafeForwardCapture] = []
        self.groot_action_horizon: int | None = None
        # pi05 expert block residual capture 시 마지막 chunk_size action token mean-pool.
        self.pi05_chunk_size: int | None = None
        if self.groot_dit_layers is not None and self.pi05_expert_layers is not None:
            raise ValueError("groot_dit_layers and pi05_expert_layers are mutually exclusive")
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
        elif self.pi05_expert_layers is not None:
            if policy_type != "pi05":
                raise ValueError("pi05_expert_layers is supported only for policy_type='pi05'")
            # COAST A.7.1: action expert(Gemma2) decoder layer residual stream 출력.
            layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
            self.pi05_chunk_size = int(policy.model.config.chunk_size)
            self.block_modules = [layers[layer] for layer in self.pi05_expert_layers]
            self.module = None
            self.mode = None
            self.slicer = None
        else:
            self.module, self.mode, self.slicer = _resolve_target(policy, policy_type)
        self.vl_module = (
            _resolve_vl_target(policy, policy_type, vl_capture_point)
            if capture_vl
            else None
        )
        self.buf: list[torch.Tensor] = []
        self.vl_buf: list[torch.Tensor] = []
        self._capture_ctx: SafeForwardCapture | None = None
        self._vl_capture_ctx: SafeForwardCapture | None = None

    @property
    def _block_layers(self) -> list[int] | None:
        """block residual capture 대상 layer 목록 (groot DiT 또는 pi05 expert)."""
        if self.groot_dit_layers is not None:
            return self.groot_dit_layers
        if self.pi05_expert_layers is not None:
            return self.pi05_expert_layers
        return None

    def __enter__(self) -> "SafeFeatureCapture":
        self.buf = []
        self.vl_buf = []
        self.block_bufs = {}
        self._block_capture_ctxs = []
        block_layers = self._block_layers
        if block_layers is not None:
            for layer, module in zip(block_layers, self.block_modules):
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
            # vlln_mean(구): seq-mean → [B, D_vl] / post_vl_sa_full(신규): full-token
            # 무슬라이스 → [B, T_vl, D_vl] (mean 은 fit 시점에).
            vl_slicer = (
                None
                if self.vl_capture_point == "post_vl_sa_full"
                else (lambda t: t.mean(dim=1))
            )
            self._vl_capture_ctx = SafeForwardCapture(
                self.vl_module,
                "post",
                vl_slicer,
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
            for layer, ctx in zip(self._block_layers or [], self._block_capture_ctxs):
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
        """GR00T DiT block residual features, COAST-faithful (A.7.2).

        ``groot_dit_token_pool`` 에 따라:
          - ``action_token_mean`` (구·default): per layer ``[K, B, T, D]`` -> slice the
            last ``action_horizon`` action tokens -> mean-pool across those action
            tokens -> ``[K, D]``. Stack layers -> ``[L, K, D]``. COAST §3: "mean-pooling
            across action tokens to obtain one vector h per denoising step".
          - ``all_token_full`` (exp3): 토큰 축 T(state+future+action 전체) 보존 ->
            per layer ``[K, T, D]`` -> stack layers -> ``[L, K, T, D]`` (fp16, fit
            수집 전용 — mean 은 fit 시점에 수행).
        """
        if self.groot_dit_layers is None:
            return None
        if not self.groot_dit_layers:
            raise ValueError("groot_dit_layers must not be empty")
        horizon = self.groot_action_horizon
        if not horizon:
            raise RuntimeError("groot_action_horizon unavailable for action-token slice")
        # 추론이 발화하지 않은 step → 전 block buf 가 비면 None (assemble() 과 동일).
        # GR00T 는 direct chunk inference 라 보통 매 호출 발화하지만, 일관성을 위해 가드.
        if not any(self.block_bufs.get(layer) for layer in self.groot_dit_layers):
            return None
        layer_feats: list[torch.Tensor] = []
        for layer in self.groot_dit_layers:
            feats = self.block_bufs.get(layer, [])
            if not feats:
                raise RuntimeError(f"Failed to capture GR00T DiT block {layer} residual stream")
            stack = torch.stack(feats, dim=0)  # [K, B, T, D]
            if self.groot_dit_token_pool == "all_token_full":
                layer_feats.append(stack[:, 0, :, :])  # [K, T, D]
            else:
                # COAST: mean-pool the last `horizon` action tokens, keep denoise step K.
                layer_feats.append(stack[:, 0, -horizon:, :].mean(dim=1))  # [K, D]
        # [L, K, D] (action_token_mean) | [L, K, T, D] (all_token_full)
        return torch.stack(layer_feats, dim=0).numpy().astype(np.float16)

    def assemble_expert_blocks(self) -> np.ndarray | None:
        """pi05 action expert block residual features, COAST-faithful (A.7.1).

        action expert(Gemma2) decoder layer 출력 per layer ``[K, B, S_full, D]`` →
        마지막 ``chunk_size`` action token slice → 그 action token 들에 대해 mean-pool →
        ``[K, D]`` (denoise step K 보존). layer stack → ``[L, K, D]``. GR00T
        ``assemble_blocks`` 와 동일한 집계, slice 길이만 ``chunk_size`` 로 바뀐다.
        """
        if self.pi05_expert_layers is None:
            return None
        if not self.pi05_expert_layers:
            raise ValueError("pi05_expert_layers must not be empty")
        chunk = self.pi05_chunk_size
        if not chunk:
            raise RuntimeError("pi05_chunk_size unavailable for action-token slice")
        # pi05 는 internal action queue → 추론은 n_action_steps 마다만 발화한다. queue-pop
        # step(추론 안 함)에서는 expert layer hook 이 안 불려 block_bufs 가 전부 빈다 →
        # assemble() 과 동일하게 None 반환(이 step 은 feature 없음). 일부 layer 만 비는
        # 부분 캡처만 진짜 오류로 취급한다.
        if not any(self.block_bufs.get(layer) for layer in self.pi05_expert_layers):
            return None
        layer_feats: list[torch.Tensor] = []
        for layer in self.pi05_expert_layers:
            feats = self.block_bufs.get(layer, [])
            if not feats:
                raise RuntimeError(
                    f"Failed to capture pi05 expert block {layer} residual stream"
                )
            stack = torch.stack(feats, dim=0)  # [K, B, S_full, D]
            # COAST: mean-pool the last `chunk_size` action tokens, keep denoise step K.
            layer_feats.append(stack[:, 0, -chunk:, :].mean(dim=1))  # [K, D]
        return torch.stack(layer_feats, dim=0).numpy().astype(np.float16)  # [L, K, D]

    def block_num_inference_timesteps(self) -> int | None:
        block_layers = self._block_layers
        if block_layers is None:
            return None
        first_layer = block_layers[0]
        feats = self.block_bufs.get(first_layer, [])
        return int(len(feats)) if feats else None

    def assemble_vl(self) -> np.ndarray | None:
        """GR00T VL pathway feature.

        ``vlln_mean``: ``action_head.vlln`` 출력 seq-mean-pool → ``[D_vl]``.
        ``post_vl_sa_full``: ``vl_self_attention`` 출력 full-token → ``[T_vl, D_vl]``.
        어느 쪽이든 get_action 당 1회 발화 → 첫 buf 의 batch 0 사용.
        """
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
    pi05_expert_layers: tuple[int, ...] | list[int] | None = None,
    groot_dit_token_pool: str = "action_token_mean",
    vl_capture_point: str = "vlln_mean",
) -> tuple[torch.Tensor, np.ndarray | None, list[str] | None, dict[str, Any]]:
    """SAFE hook 을 건 채 ``select_action`` 을 실행.

    Returns ``(action, hidden_states | None, feature_axes | None, meta)``.
    GR00T N1.5는 N1.6 SAFE collection과 같은 chunk execution을 맞추기 위해
    ``predict_action_chunk``를 호출한다. 다른 policy에서 추론이 발화하지 않은
    step(queue pop 만)에서는 hidden_states=None — 호출자는 hidden_states가 있을 때만
    record 한다. ``capture_vl``은 GR00T에서만 ``action_head.vlln`` 출력의 sequence
    mean-pool을 optional VL pathway feature로 추가한다. ``groot_dit_layers``가 있으면
    GR00T DiT final output 대신 선택한 ``transformer_blocks`` residual stream을 COAST A.7.2
    대로 action-token mean + denoise step 보존하여 ``[layer, denoise_step, feature_dim]``으로
    캡처한다. ``pi05_expert_layers``가 있으면 pi05 action expert(Gemma2) decoder layer
    residual stream을 COAST A.7.1 대로 마지막 ``chunk_size`` action token mean + denoise step
    보존하여 동일한 ``[layer, denoise_step, feature_dim]``으로 캡처한다.

    """
    cap = SafeFeatureCapture(
        policy,
        policy_type,
        capture_vl=capture_vl,
        groot_dit_layers=groot_dit_layers,
        pi05_expert_layers=pi05_expert_layers,
        groot_dit_token_pool=groot_dit_token_pool,
        vl_capture_point=vl_capture_point,
    )
    with torch.inference_mode(), cap:
        if policy_type == "groot" and hasattr(policy, "predict_action_chunk"):
            action = policy.predict_action_chunk(batch)
        else:
            action = policy.select_action(batch)

    ca_meta: dict[str, Any] = {}

    if groot_dit_layers is not None:
        hidden = cap.assemble_blocks()
    elif pi05_expert_layers is not None:
        hidden = cap.assemble_expert_blocks()
    else:
        hidden = cap.assemble()
    if hidden is None:
        return action, None, None, ca_meta

    if groot_dit_layers is not None:
        if groot_dit_token_pool == "all_token_full":
            # hidden is [L, K, T, D]: full-token capture (mean 은 fit 시점에).
            axes = list(GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_AXES)
            meta = {
                "feature_kind": GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FULLTOKEN_FEATURE_KIND,
                "feature_axes": axes,
                "num_inference_timesteps": int(hidden.shape[1]),
                "capture_layers": [int(layer) for layer in groot_dit_layers],
                "layer_count": int(hidden.shape[0]),
                "denoise_step_count": int(hidden.shape[1]),
                "token_count": int(hidden.shape[2]),
                "feature_dim": int(hidden.shape[3]),
                "model_action_horizon": cap.groot_action_horizon,
                "capture_token_mode": groot_dit_token_pool,
            }
        else:
            # hidden is [L, K, D]: layer x denoise_step x feature_dim (action-token pooled).
            axes = list(GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES)
            meta = {
                "feature_kind": GROOT_N15_DIT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND,
                "feature_axes": axes,
                "num_inference_timesteps": int(hidden.shape[1]),
                "capture_layers": [int(layer) for layer in groot_dit_layers],
                "layer_count": int(hidden.shape[0]),
                "denoise_step_count": int(hidden.shape[1]),
                "feature_dim": int(hidden.shape[2]),
                "model_action_horizon": cap.groot_action_horizon,
                "capture_token_mode": groot_dit_token_pool,
            }
    elif pi05_expert_layers is not None:
        # hidden is [L, K, D]: layer x denoise_step x feature_dim (action-token pooled).
        axes = list(PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_AXES)
        meta = {
            "feature_kind": PI05_EXPERT_BLOCK_RESIDUAL_DENOISE_FEATURE_KIND,
            "feature_axes": axes,
            "num_inference_timesteps": int(hidden.shape[1]),
            "capture_layers": [int(layer) for layer in pi05_expert_layers],
            "layer_count": int(hidden.shape[0]),
            "denoise_step_count": int(hidden.shape[1]),
            "feature_dim": int(hidden.shape[2]),
            "model_action_horizon": cap.pi05_chunk_size,
        }
    else:
        axes = lerobot_feature_axes(policy_type)
    _block_mode = groot_dit_layers is not None or pi05_expert_layers is not None
    if not _block_mode and policy_type in AUTOREGRESSIVE_TYPES:
        meta = {
            "feature_kind": lerobot_feature_kind(policy_type),
            "feature_axes": axes,
            "num_inference_timesteps": None,
            "n_action_tokens": int(hidden.shape[1]),
            "feature_dim": int(hidden.shape[2]),
        }
    elif not _block_mode:
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
        if vl_capture_point == "post_vl_sa_full":
            meta["vl_feature_kind"] = GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_KIND
            meta["vl_feature_axes"] = list(GROOT_N15_VL_POST_SA_FULLTOKEN_FEATURE_AXES)
            meta["vl_token_count"] = int(vl_hidden.shape[0])
        else:
            meta["vl_feature_kind"] = GROOT_N15_VL_FEATURE_KIND
            meta["vl_feature_axes"] = list(GROOT_VL_FEATURE_AXES)
        meta["vl_feature_dim"] = int(vl_hidden.shape[-1])
        meta["vl_capture_point"] = vl_capture_point
    meta.update(ca_meta)
    return action, hidden, axes, meta
