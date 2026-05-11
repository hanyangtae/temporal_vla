"""GR00T N1.6 × TTT integration wrapper.

Stage 2 학습/추론에서 GR00T 옆에 frozen ProgressPredictor 를 두고, Eagle pre-LLM
임베딩을 TTT 입력으로 연결, TTT 출력을 DiT cross-attention KV 에 token 으로
**direct prepend** (projector 없음) 한다. TTT 방향 backprop 차단을 위해 token
은 ``.detach()`` 후 concat 된다.

Architecture (Stage 2 forward)
------------------------------
::

    backbone_inputs ─► Eagle.model(output_hidden_states=True)
                         │
                         ├─ hidden_states[0]  (pre-LLM)
                         │       │
                         │       ▼ masked-mean-pool over T_vl
                         │   z [B, 2048]  (fp32)
                         │       │
                         │       ▼ ProgressPredictor (frozen outer; inner-loop optional)
                         │   h_TTT [B, 2048]
                         │       │
                         │       ▼ .detach().unsqueeze(1)
                         │   ttt_token [B, 1, 2048]
                         │       │
                         └─ hidden_states[-1] (vl_embeds, [B, T_vl, 2048])
                                 │
                                 ▼ concat ([vl_embeds, ttt_token])
                         DiT cross-attn KV   [B, T_vl + 1, 2048]
                                 │
                                 ▼
                            action_head (DiT) → action_loss

Backward gradient routing
-------------------------
- ``action_loss`` flows back through DiT/state·action enc/dec; stops at the
  ``.detach()`` on ``ttt_token``. Predictor outer params are also frozen via
  ``requires_grad=False``.
- Predictor's inner-loop adaptation (``ttt_step``) is independently controlled by
  ``ttt_update_in_train``. Default False during training (deterministic forward),
  enable for episode-aware experiments.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Optional

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

# Isaac-GR00T submodule path (containers / dev hosts)
_GR00T_PATH = "/temporal_vla/src/policies/Isaac-GR00T"
if _GR00T_PATH not in sys.path:
    sys.path.insert(0, _GR00T_PATH)

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import (  # noqa: E402
    Gr00tN1d6,
    Gr00tN1d6Config,
)

from src.ttt.predictor import ProgressPredictor

logger = logging.getLogger(__name__)


class Gr00tN1d6WithTTT(Gr00tN1d6):
    """Gr00tN1d6 with a parallel TTT predictor whose embedding is injected
    into the DiT cross-attention KV stream as one extra token.

    Members in addition to base ``Gr00tN1d6``:

    - ``self.predictor``: ``ProgressPredictor`` (input_dim=2048, proj_dim=2048 by default,
        matching Eagle Qwen3-1.7B hidden = DiT cross-attn KV dim).
        Outer params are frozen for Stage 2; inner-loop adaptation is gated by
        ``ttt_update_in_train``.

    Forward augments ``backbone_output.backbone_features`` and
    ``backbone_output.backbone_attention_mask`` (and ``image_mask`` for the
    alternate-VL-DiT path) with one extra TTT token before delegating to the
    upstream ``action_head`` — so both training and inference (DiT denoising loop)
    transparently see the augmented KV stream.
    """

    config_class = Gr00tN1d6Config

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
        predictor_input_dim: int = 2048,
        predictor_proj_dim: int = 2048,
        predictor_inner_model: str = "linear",
        predictor_eta_base: float = 0.1,
        ttt_update_in_train: bool = False,
        ttt_update_at_inference: bool = True,
    ):
        super().__init__(config, transformers_loading_kwargs=transformers_loading_kwargs)

        self.predictor = ProgressPredictor(
            input_dim=predictor_input_dim,
            proj_dim=predictor_proj_dim,
            inner_model_type=predictor_inner_model,
            eta_base=predictor_eta_base,
            learnable_eta=False,
        )
        self.ttt_update_in_train = ttt_update_in_train
        self.ttt_update_at_inference = ttt_update_at_inference
        self._predictor_loaded = False

        self._freeze_predictor_outer()

    # ─────────────────────────────────────────────
    # Predictor lifecycle
    # ─────────────────────────────────────────────

    def _freeze_predictor_outer(self) -> None:
        """Freeze **every** predictor param including the inner model (f_adapt).

        Why all-frozen rather than f_adapt-unfrozen: HF Trainer's default
        ``create_optimizer`` (`transformers/trainer.py:1281-1305`) auto-includes
        every ``requires_grad=True`` parameter. If we leave f_adapt unfrozen
        permanently, Adam ends up tracking it and (a) applying weight_decay
        slowly destroys θ_init even with detached forward, and (b) Adam
        momentum collides with ``ttt_step`` 's in-place ``param.data`` update.

        ``ttt_step`` only needs ``requires_grad=True`` on f_adapt for its
        ``torch.autograd.grad`` call, so we flip it transiently inside
        ``_enable_inner_grad`` whenever ``predictor(z, update=True)`` is invoked.
        """
        for p in self.predictor.parameters():
            p.requires_grad = False

    @contextmanager
    def _enable_inner_grad(self):
        """Temporarily set f_adapt params to ``requires_grad=True`` so
        ``torch.autograd.grad`` inside ``predictor.ttt.ttt_step`` succeeds.
        Restored on exit so HF Trainer's optimizer never sees them as trainable.
        """
        flips: list[torch.nn.Parameter] = []
        for n, p in self.predictor.named_parameters():
            if "f_adapt" in n and not p.requires_grad:
                p.requires_grad = True
                flips.append(p)
        try:
            yield
        finally:
            for p in flips:
                p.requires_grad = False

    def load_predictor_state(self, state_dict_path: str) -> None:
        """Load Phase 1 ProgressPredictor weights and snapshot inner state as θ_init."""
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        self.predictor.load_state_dict(sd)
        self.predictor.save_init()
        self._freeze_predictor_outer()
        self._predictor_loaded = True
        logger.info("Loaded predictor state from %s", state_dict_path)

    def reset_predictor(self) -> None:
        """Reset TTT inner state to θ_init. Call at episode boundary."""
        self.predictor.reset()

    # ─────────────────────────────────────────────
    # Eagle forward — get pre-LLM + post-LLM together
    # ─────────────────────────────────────────────

    def _eagle_full_forward(self, backbone_inputs: dict):
        """Run the Eagle backbone with ``output_hidden_states=True``.

        Returns:
            pre_llm: ``hidden_states[0]`` [B, T_vl, hidden] — embeddings before any
                     LLM transformer layer, i.e. the LLM's input embedding sequence.
            post_llm: ``hidden_states[-1]`` [B, T_vl, hidden] — what DiT cross-attends to.
            attention_mask_bool: [B, T_vl] bool — valid token mask.
            image_mask_bool: [B, T_vl] bool — image-token positions (for use_alternate_vl_dit).
        """
        backbone = self.backbone
        backbone.set_frozen_modules_to_eval_mode()
        keys = ["input_ids", "attention_mask", "pixel_values"]
        vl_input = {k: backbone_inputs[k] for k in keys}
        outputs = backbone.model(**vl_input, output_hidden_states=True)
        pre_llm = outputs.hidden_states[0]
        post_llm = outputs.hidden_states[-1]
        attn_mask_bool = vl_input["attention_mask"] == 1
        image_token_id = backbone.model.config.image_token_index
        image_mask_bool = vl_input["input_ids"] == image_token_id
        return pre_llm, post_llm, attn_mask_bool, image_mask_bool

    # ─────────────────────────────────────────────
    # TTT branch
    # ─────────────────────────────────────────────

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B, T, D] + [B, T] bool → [B, D] (mean over T over valid positions)."""
        m = mask.to(x.dtype).unsqueeze(-1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def _ttt_token(
        self,
        pre_llm: torch.Tensor,
        attention_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Pool pre-LLM, run predictor, return ``(ttt_token, pred_out)``.

        ``ttt_token`` is detached and dtype-matched to ``pre_llm`` so it can be
        concatenated with ``post_llm`` (DiT KV) without grad flowing back to TTT.

        ``do_update`` paths:
          - train: ``ttt_update_in_train`` (default False — deterministic forward).
          - eval/serve: ``ttt_update_at_inference`` (default True — inner-loop
            adaptation per the user-specified design).
        Inside the update branch we flip f_adapt requires_grad transiently so
        ``ttt_step`` 's ``autograd.grad`` works without leaking into HF Trainer.
        """
        z = self._masked_mean(pre_llm, attention_mask_bool).float()

        do_update = (
            (self.training and self.ttt_update_in_train)
            or (not self.training and self.ttt_update_at_inference)
        )
        if do_update:
            with self._enable_inner_grad():
                pred_out = self.predictor(z, update=True)
        else:
            pred_out = self.predictor(z, update=False)

        h_TTT = pred_out["ttt_output"]                 # [B, proj_dim]
        ttt_token = h_TTT.detach().unsqueeze(1)        # [B, 1, proj_dim]
        ttt_token = ttt_token.to(pre_llm.dtype)
        return ttt_token, pred_out

    @staticmethod
    def _augment_backbone_output(
        post_llm: torch.Tensor,
        attention_mask_bool: torch.Tensor,
        image_mask_bool: torch.Tensor,
        ttt_token: torch.Tensor,
    ) -> BatchFeature:
        """Concat TTT token to post-LLM features + extend masks."""
        B = post_llm.size(0)
        device = post_llm.device

        new_features = torch.cat([post_llm, ttt_token], dim=1)        # [B, T_vl+1, D]

        ones_attn = torch.ones(B, 1, dtype=attention_mask_bool.dtype, device=device)
        new_attn = torch.cat([attention_mask_bool, ones_attn], dim=1)  # always valid

        zeros_img = torch.zeros(B, 1, dtype=image_mask_bool.dtype, device=device)
        new_img = torch.cat([image_mask_bool, zeros_img], dim=1)       # not an image token

        return BatchFeature(data={
            "backbone_features": new_features,
            "backbone_attention_mask": new_attn,
            "image_mask": new_img,
        })

    def _backbone_with_ttt(self, backbone_inputs: dict):
        pre_llm, post_llm, attn, img_mask = self._eagle_full_forward(backbone_inputs)
        ttt_token, pred_out = self._ttt_token(pre_llm, attn)
        backbone_outputs = self._augment_backbone_output(post_llm, attn, img_mask, ttt_token)
        return backbone_outputs, pred_out

    # ─────────────────────────────────────────────
    # Public forward / get_action — drop-in for upstream Trainer
    # ─────────────────────────────────────────────

    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs, pred_out = self._backbone_with_ttt(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)
        # 모니터링 정보 (action loss 에 영향 없음)
        try:
            action_outputs["progress_pred"] = pred_out["progress"].detach()
            action_outputs["ttt_ssl_loss"] = pred_out["ssl_loss"].detach()
        except Exception:
            pass
        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs, _pred_out = self._backbone_with_ttt(backbone_inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs)


# ──────────────────────────────────────────────────────────────────────────
# Helpers for instantiation / weight transfer
# ──────────────────────────────────────────────────────────────────────────


def attach_ttt_to_groot(
    base_model: Gr00tN1d6,
    predictor_state_path: Optional[str] = None,
    *,
    predictor_proj_dim: int = 2048,
    predictor_input_dim: int = 2048,
    predictor_inner_model: str = "linear",
    predictor_eta_base: float = 0.1,
    ttt_update_in_train: bool = False,
    ttt_update_at_inference: bool = True,
) -> Gr00tN1d6WithTTT:
    """기존 ``Gr00tN1d6`` 인스턴스의 가중치를 그대로 가지고 ``Gr00tN1d6WithTTT`` 로
    승격. upstream ``AutoModel.from_pretrained`` 호출 직후에 한 번 부르면 된다.

    ``predictor_state_path`` 가 주어지면 Phase 1 체크포인트도 함께 로드.
    """
    config = base_model.config
    new_model = Gr00tN1d6WithTTT(
        config,
        predictor_input_dim=predictor_input_dim,
        predictor_proj_dim=predictor_proj_dim,
        predictor_inner_model=predictor_inner_model,
        predictor_eta_base=predictor_eta_base,
        ttt_update_in_train=ttt_update_in_train,
        ttt_update_at_inference=ttt_update_at_inference,
    )
    # base model 의 가중치 → 새 인스턴스로 복사. predictor 키는 base 에 없으므로
    # strict=False (missing keys 만 발생, predictor 는 새로 init 된 상태 유지).
    missing, unexpected = new_model.load_state_dict(base_model.state_dict(), strict=False)
    pred_missing = [k for k in missing if not k.startswith("predictor.")]
    if pred_missing:
        logger.warning("Unexpected missing keys (non-predictor): %s", pred_missing[:10])
    if unexpected:
        logger.warning("Unexpected keys when copying base→ttt: %s", unexpected[:10])

    # device / dtype 맞추기
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    new_model = new_model.to(device=device)
    # predictor 만 fp32 유지 (autograd.grad 가 fp32 에서 안전)
    for n, p in new_model.named_parameters():
        if not n.startswith("predictor."):
            p.data = p.data.to(dtype=dtype)
        else:
            p.data = p.data.to(dtype=torch.float32)

    if predictor_state_path:
        new_model.load_predictor_state(predictor_state_path)

    return new_model
