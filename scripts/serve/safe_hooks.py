"""SAFE feature 추출 hook (lerobot 정책용, HTTP serve 에서 사용).

SAFE latent 한 점의 의미
------------------------
한 점 = **policy 추론(action chunk) 1회**에서 얻은, action 이 velocity field
(flow-matching) 또는 token logit(autoregressive)으로 디코딩되기 **직전**의
마지막-레이어 hidden state. SAFE 논문(arXiv 2506.09937)의 feature 정의와 동일.

lerobot 정책은 내부 action queue 가 빌 때만(= ``config.n_action_steps`` env step
마다) 새 추론을 돌린다. 그 사이 step 의 ``select_action`` 은 버퍼된 action 을
popleft 만 하고 추론을 하지 않으므로 latent 가 생기지 않는다. collector 는 latent
가 생긴 step 에만 record 한다. 이렇게 하면 rollout 충실도가 일반 배포와 동일하게
유지되면서(매 step replan 아님) SAFE 의 "추론당 feature" 시퀀스와 일치한다.

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


class SafeFeatureCapture:
    """lerobot 정책에 hook 을 걸어 SAFE feature 를 누적하는 context manager.

    hook 이 한 번 발화할 때마다 ``self.buf`` 에 텐서 1개 append:
      - flow-matching: denoise step 마다 ``[B, H, D]`` (총 K 개)
      - pi0_fast: 생성 토큰마다 ``[B, 1, D]`` (총 n_tokens 개)
    """

    def __init__(self, policy: Any, policy_type: str):
        if policy_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported policy_type: {policy_type}")
        self.policy = policy
        self.policy_type = policy_type
        self.module, self.mode, self.slicer = _resolve_target(policy, policy_type)
        self.buf: list[torch.Tensor] = []
        self._capture_ctx: SafeForwardCapture | None = None

    def __enter__(self) -> "SafeFeatureCapture":
        self.buf = []
        self._capture_ctx = SafeForwardCapture(
            self.module,
            self.mode,
            self.slicer,
            to_cpu=True,
            dtype=torch.float32,
        )
        self._capture_ctx.__enter__()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        if self._capture_ctx is not None:
            self.buf = list(self._capture_ctx.buf)
            self._capture_ctx.__exit__(*_exc)
            self._capture_ctx = None
        return False

    def assemble(self) -> np.ndarray | None:
        """캡처된 텐서를 per-step SAFE latent 으로 결합.

        이번 호출에서 추론이 발화하지 않았으면(queue pop 만) None 반환.
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


def run_with_features(
    policy: Any, batch: dict[str, Any], policy_type: str
) -> tuple[torch.Tensor, np.ndarray | None, list[str] | None, dict[str, Any]]:
    """SAFE hook 을 건 채 ``select_action`` 을 실행.

    Returns ``(action, hidden_states | None, feature_axes | None, meta)``.
    추론이 발화하지 않은 step(queue pop 만)에서는 hidden_states=None — 호출자는
    hidden_states 가 있을 때만 record 한다.
    """
    cap = SafeFeatureCapture(policy, policy_type)
    with torch.inference_mode(), cap:
        action = policy.select_action(batch)

    hidden = cap.assemble()
    if hidden is None:
        return action, None, None, {}

    axes = lerobot_feature_axes(policy_type)
    if policy_type in AUTOREGRESSIVE_TYPES:
        meta = {
            "feature_kind": lerobot_feature_kind(policy_type),
            "feature_axes": axes,
            "num_inference_timesteps": None,
            "n_action_tokens": int(hidden.shape[1]),
            "feature_dim": int(hidden.shape[2]),
        }
    else:
        meta = {
            "feature_kind": lerobot_feature_kind(policy_type),
            "feature_axes": axes,
            "num_inference_timesteps": int(hidden.shape[0]),
            "action_horizon": int(hidden.shape[1]),
            "feature_dim": int(hidden.shape[2]),
        }
    return action, hidden, axes, meta
