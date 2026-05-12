"""GR00T N1.6 × TTT integration wrapper.

Stage 2 학습/추론: TTT module 은 Phase 1 ckpt 로 **outer-frozen** 으로 로드되고,
각 sample 마다 episode prefix Eagle pre-LLM 시퀀스 ``z_0..z_t`` 를 sequential 하게
inner-loop SSL 로 누적 → 마지막 timestep 의 TTT 출력 = **Latent History Token (LHT)**.
LHT 는 ``.detach()`` 후 DiT cross-attention KV 에 token 으로 **direct prepend**
(projector 없음) — action loss 의 gradient 가 TTT 로 흐르지 않게 차단. TTT 는
오로지 inner-loop SSL 로만 적응하고 outer 학습은 안 받음.

Architecture (Stage 2 forward, episode-prefix mode)
---------------------------------------------------
::

    inputs:
      ├─ ttt_z_seq      [B, T_max, 2048]   (Eagle pre-LLM 캐시 0..t 까지)
      ├─ ttt_valid_mask [B, T_max] bool
      └─ <GR00T 표준 obs at frame t (image, state, lang)>

    ttt_z_seq, valid_mask
        │
        ▼ predictor.meta_forward (inner-loop SSL, create_graph=False)
    ttt_outputs_seq [T_max, B, 2048]
        │
        ▼ gather last-valid timestep per item
    LHT [B, 2048]
        │
        ▼ .detach().unsqueeze(1)
    ttt_token [B, 1, 2048]

    <obs at frame t> ─► Eagle.model(hidden_states[-1]) ─► post_llm [B, T_vl, 2048]
        │
        ▼ concat(post_llm, ttt_token) → DiT KV [B, T_vl+1, 2048]
        ▼ DiT → action_loss

Backward gradient routing
-------------------------
- ``action_loss`` flows back through DiT/state·action enc/dec + GR00T LLM 상위
  4 layer (config.model.tune_top_llm_layers=4).
- ``.detach()`` on ``ttt_token`` + ``requires_grad=False`` on all predictor params
  → TTT outer 파라미터 (P_K, P_V, P_Q, θ_0, f_adapt) 는 Phase 2 에서 학습 X.
- Predictor's inner-loop adaptation (``ttt_step``) 는 ``ttt_update_in_train`` 으로
  토글. **Default True**: episode prefix replay 시 frame 별 SSL 적용 → LHT 가
  trajectory history 누적. Phase 2 학습/추론 모두 동일.

Single-frame fallback
---------------------
``inputs`` 에 ``ttt_z_seq`` 가 없으면 (=배치가 trajectory 컨텍스트를 안 실어 줌)
기존 single-frame ``_ttt_token(pre_llm, attn)`` 경로로 fallback. Phase 1 ckpt
호환성 / smoke test 용.
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
        ttt_update_in_train: bool = True,
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
        """Single-frame fallback — `ttt_z_seq` 가 inputs 에 없을 때 사용.

        Pool pre-LLM, run predictor 1 step, return ``(ttt_token, pred_out)``.
        ``ttt_token`` is detached and dtype-matched to ``pre_llm`` so it can be
        concatenated with ``post_llm`` (DiT KV) without grad flowing back to TTT.

        ``do_update`` paths:
          - train: ``ttt_update_in_train`` (default True — frame 별 inner-loop).
          - eval/serve: ``ttt_update_at_inference`` (default True — inner-loop
            adaptation per episode 누적, serve 시 ``reset_predictor`` 명시 필요).
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

    def _ttt_token_from_zseq(
        self,
        z_seq: torch.Tensor,           # [B, T_max, D]  fp16/bf16/fp32
        valid_mask: torch.Tensor,      # [B, T_max] bool
    ) -> tuple[torch.Tensor, dict]:
        """**Episode-prefix replay → Latent History Token (LHT).**

        Phase 2 의 핵심 경로. dataset 이 sample 마다 Eagle pre-LLM cache 에서
        episode 시작부터 현재 frame 직전까지의 z_seq 를 cubidic 으로 채워 batch
        로 보내주면, 여기서 sequential 하게 inner-loop SSL 을 돌려 trajectory
        history 가 누적된 LHT 를 만든다. 각 batch item 의 마지막 valid timestep
        에서의 TTT output 을 LHT 로 채택, detach 후 dtype 정합하여 token 으로 반환.

        Notes
        -----
        * meta_forward 는 **batch-shared inner_params** 로 sequential 업데이트.
          서로 다른 episode prefix 가 같은 inner_params 를 공유 → 평균화 효과로
          contamination. Phase 1 meta-learning 과 동일한 batching 가정.
          엄밀한 per-item independent replay 는 cost 가 batch 배여서 1차 implementation
          에서는 생략. inference 시 batch=1 이라 무관.
        * ``create_graph=False``: outer backprop 차단 (TTT 는 frozen).
        * predictor.reset() 은 호출 안 함 — meta_forward 가 매 호출마다
          ``get_inner_params()`` 로 θ_0 부터 시작.
        """
        B, T_max, D = z_seq.shape
        target_dtype = z_seq.dtype

        # meta_forward 는 [T, B, D] fp32 기대.
        z_seq_t = z_seq.permute(1, 0, 2).contiguous().float()
        valid_mask_t = valid_mask.permute(1, 0).contiguous()  # [T_max, B]

        with self._enable_inner_grad():
            result = self.predictor.meta_forward(
                z_seq_t, create_graph=False, valid_mask=valid_mask_t,
            )

        ttt_outputs_seq = result["ttt_outputs_seq"]  # [T_max, B, D]

        # batch item 별 마지막 valid timestep
        last_valid = valid_mask.long().sum(dim=1) - 1            # [B], -1 if 모두 invalid
        last_valid = last_valid.clamp(min=0)
        arange_B = torch.arange(B, device=z_seq.device)
        lht = ttt_outputs_seq[last_valid, arange_B, :]           # [B, D]

        ttt_token = lht.detach().unsqueeze(1).to(target_dtype)   # [B, 1, D]

        progress_seq = result["progress_seq"]                    # [T_max, B, 1]
        progress_at_last = progress_seq[last_valid, arange_B, :]
        # ssl_losses 는 list of T scalars; 마지막 valid timestep 의 평균 ssl
        ssl_list = result["ssl_losses"]
        if ssl_list:
            ssl_at_last = ssl_list[-1] if torch.is_tensor(ssl_list[-1]) else torch.tensor(0.0)
        else:
            ssl_at_last = torch.tensor(0.0, device=z_seq.device)

        pred_out = {
            "progress": progress_at_last.detach(),
            "ssl_loss": ssl_at_last.detach() if torch.is_tensor(ssl_at_last) else ssl_at_last,
            "ttt_output": lht.detach(),
        }
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

    def _backbone_with_ttt(
        self,
        backbone_inputs: dict,
        ttt_z_seq: Optional[torch.Tensor] = None,
        ttt_valid_mask: Optional[torch.Tensor] = None,
    ):
        pre_llm, post_llm, attn, img_mask = self._eagle_full_forward(backbone_inputs)

        if ttt_z_seq is not None and ttt_valid_mask is not None:
            # Phase 2 primary path: episode-prefix replay → LHT
            ttt_token, pred_out = self._ttt_token_from_zseq(ttt_z_seq, ttt_valid_mask)
        else:
            # Fallback: single-frame pre-LLM pooling (legacy / smoke test)
            ttt_token, pred_out = self._ttt_token(pre_llm, attn)

        backbone_outputs = self._augment_backbone_output(post_llm, attn, img_mask, ttt_token)
        return backbone_outputs, pred_out

    # ─────────────────────────────────────────────
    # Public forward / get_action — drop-in for upstream Trainer
    # ─────────────────────────────────────────────

    @staticmethod
    def _pop_ttt_inputs(inputs):
        """Pop ``ttt_z_seq`` / ``ttt_valid_mask`` so they don't reach `prepare_input`
        (which would error on unknown keys). Returns (z_seq, valid_mask) or (None, None)."""
        if not isinstance(inputs, dict):
            return None, None
        z_seq = inputs.pop("ttt_z_seq", None)
        valid_mask = inputs.pop("ttt_valid_mask", None)
        return z_seq, valid_mask

    def forward(self, inputs: dict) -> BatchFeature:
        ttt_z_seq, ttt_valid_mask = self._pop_ttt_inputs(inputs)
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs, pred_out = self._backbone_with_ttt(
            backbone_inputs, ttt_z_seq=ttt_z_seq, ttt_valid_mask=ttt_valid_mask,
        )
        action_outputs = self.action_head(backbone_outputs, action_inputs)
        # 모니터링 정보 (action loss 에 영향 없음)
        try:
            action_outputs["progress_pred"] = pred_out["progress"].detach()
            ssl = pred_out["ssl_loss"]
            action_outputs["ttt_ssl_loss"] = ssl.detach() if torch.is_tensor(ssl) else ssl
        except Exception:
            pass
        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        ttt_z_seq, ttt_valid_mask = self._pop_ttt_inputs(inputs)
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs, _pred_out = self._backbone_with_ttt(
            backbone_inputs, ttt_z_seq=ttt_z_seq, ttt_valid_mask=ttt_valid_mask,
        )
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
    ttt_update_in_train: bool = True,
    ttt_update_at_inference: bool = True,
) -> Gr00tN1d6:
    """기존 ``Gr00tN1d6`` 인스턴스에 **in place** 로 TTT 를 부착.

    이전 구현은 ``Gr00tN1d6WithTTT(config)`` 로 새 인스턴스를 만들고 weight 를 copy
    했는데, ``__init__`` 안의 ``super().__init__()`` 가 Eagle3 backbone 을 from_config
    로 다시 빌드하면서 ``Siglip2VisionModel.post_init()`` → ``_init_weights`` 의 upstream
    ``NameError: Siglip2Model is not defined`` 버그를 트리거. base_model 은 이미 from_pretrained
    로 정상 로드되어 있으므로, **재인스턴스화하지 않고** Gr00tN1d6WithTTT 의 메서드들을
    기존 instance 에 bind + ``predictor`` submodule 만 추가하는 방식으로 동등 동작.

    ``predictor_state_path`` 가 주어지면 Phase 1 체크포인트도 함께 로드.
    """
    from types import MethodType

    device = next(base_model.parameters()).device

    # 1) Predictor 생성 & attach (fp32 로 — autograd.grad 안전성)
    predictor = ProgressPredictor(
        input_dim=predictor_input_dim,
        proj_dim=predictor_proj_dim,
        inner_model_type=predictor_inner_model,
        eta_base=predictor_eta_base,
        learnable_eta=False,
    ).to(device=device, dtype=torch.float32)
    base_model.predictor = predictor

    # 2) TTT 관련 flag
    base_model.ttt_update_in_train = ttt_update_in_train
    base_model.ttt_update_at_inference = ttt_update_at_inference
    base_model._predictor_loaded = False

    # 3) Predictor outer params freeze — 기본은 모두 freeze, ttt_step 호출 시
    # transient 하게 f_adapt 만 requires_grad=True (_enable_inner_grad context).
    for p in base_model.predictor.parameters():
        p.requires_grad = False

    # 4) Gr00tN1d6WithTTT 의 메서드들을 base_model instance 에 bind.
    #    static method 는 함수 그대로, instance method 는 MethodType 으로.
    instance_methods = [
        "forward",
        "get_action",
        "_backbone_with_ttt",
        "_eagle_full_forward",
        "_ttt_token",
        "_ttt_token_from_zseq",
        "_enable_inner_grad",
        "_freeze_predictor_outer",
        "load_predictor_state",
        "reset_predictor",
    ]
    for name in instance_methods:
        fn = getattr(Gr00tN1d6WithTTT, name)
        setattr(base_model, name, MethodType(fn, base_model))

    # static methods — class에서 가져온 그대로 (이미 underlying function 반환)
    for name in ["_pop_ttt_inputs", "_masked_mean", "_augment_backbone_output"]:
        setattr(base_model, name, getattr(Gr00tN1d6WithTTT, name))

    # 5) Phase 1 ckpt 로드 (옵션)
    if predictor_state_path:
        base_model.load_predictor_state(predictor_state_path)

    logger.info(
        "[attach_ttt] in-place attached to %s (update_in_train=%s, update_at_inference=%s)",
        type(base_model).__name__, ttt_update_in_train, ttt_update_at_inference,
    )
    return base_model
