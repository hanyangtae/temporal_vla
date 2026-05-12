"""
ProgressPredictor — TTT + ProgressHead 진행률 예측 모듈 (Phase 1).
TTTVLAAdapter — ProgressPredictor + VLAProjector 통합 모듈 (Phase 2 + 추론).

분리 이유:
    Phase 1에서 실물 dataset으로 TTT + ProgressHead만 학습하여 진행률 예측기를
    먼저 만든 뒤, Phase 2에서 VLA와 연결하여 VLAProjector를 finetuning해야 함.
    ProgressPredictor를 독립적으로 학습/저장/로드할 수 있어야 한다.

Phase 1 — Progress Predictor 학습:
    expert trajectory에서 TTTModule + ProgressHead 학습.
    매 timestep: z_t → TTT self-update → TTT forward → ProgressHead → v_t
    label: y_t = t/T, Loss: MSE + λ_self·SSL + λ_mono·mono

Phase 2 — VLA Projector 학습 (Δv 기반):
    pretrained ProgressPredictor를 freeze, VLAProjector만 학습.
    Δv = v(s_{t+1}) - v(s_t)를 이용:
    Loss = -Δv · log P(a_expert | s_t, θ_t + projector injection)

추론:
    TTTVLAAdapter가 실패 감지 (ProgressPredictor) + action 수정 (VLAProjector) 모두 수행.

참고: VITA Section 3.1, Sun et al. (2024) Section 2
"""

import torch
import torch.nn as nn

from typing import Optional, Literal

from src.ttt.ttt_module import TTTModule
from src.ttt.progress_head import ProgressHead
from src.ttt.vla_projector import VLAProjector, InjectionMode


class ProgressPredictor(nn.Module):
    """
    TTT + ProgressHead 진행률 예측 모듈 (Phase 1 학습 단위).

    Phase 1에서 독립적으로 학습 → 체크포인트 저장 → Phase 2에서 로드.

    Args:
        input_dim: VLA encoder 출력 차원
        proj_dim: TTT inner model의 입출력 차원
        inner_model_type: TTT inner model 종류 ("linear" or "mlp")
        inner_expansion: MLP expansion factor
        eta_base: inner-loop base learning rate
        learnable_eta: η를 학습할지 여부
        head_hidden_dim: ProgressHead MLP hidden dim
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 64,
        inner_model_type: Literal["linear", "mlp", "linear_preLN"] = "linear",
        inner_expansion: int = 4,
        eta_base: float = 0.1,
        learnable_eta: bool = False,
        head_hidden_dim: int = 128,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # TTT module (SSL self-update backbone)
        self.ttt = TTTModule(
            input_dim=input_dim,
            proj_dim=proj_dim,
            inner_model_type=inner_model_type,
            inner_expansion=inner_expansion,
            eta_base=eta_base,
            learnable_eta=learnable_eta,
        )

        # Progress head (진행률 예측 + 실패 감지)
        self.progress_head = ProgressHead(
            input_dim=proj_dim,
            hidden_dim=head_hidden_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
        update: bool = True,
    ) -> dict:
        """
        단일 timestep 처리.

        Args:
            z: VLA encoder output [batch, input_dim]
            update: TTT self-update 수행 여부

        Returns:
            dict with:
                - "progress": 진행률 예측 [batch, 1]
                - "ttt_output": TTT adapted representation [batch, proj_dim]
                - "ssl_loss": self-supervised loss (학습 시)
        """
        result = {}

        ssl_loss = self.ttt.compute_ssl_loss(z)
        result["ssl_loss"] = ssl_loss

        if update:
            self.ttt.ttt_step(z)

        ttt_output = self.ttt(z)
        result["ttt_output"] = ttt_output

        progress = self.progress_head(ttt_output)
        result["progress"] = progress

        return result

    def meta_forward(
        self,
        z_sequence: torch.Tensor,
        create_graph: bool = True,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Meta-learning 학습용 sequential forward (VITA Section 3.2.1).

        에피소드 전체를 순차 처리하면서 computational graph 유지.
        L_pred의 gradient가 inner update를 통해 θ_0, P_K, P_V, P_Q, h, η로 흐름.

        학습 루프에서의 사용:
            result = predictor.meta_forward(z_seq, valid_mask=pad_mask)  # [T, B, D]
            pred_loss = F.mse_loss(result["progress_seq"][pred_mask], targets[pred_mask])
            ssl_loss = torch.stack(result["ssl_losses"]).mean()
            total = pred_loss + lambda_self * ssl_loss
            total.backward()

        Args:
            z_sequence: [T, batch, input_dim] 시간순 representation 시퀀스
            valid_mask: [T, batch] bool. True = valid frame (not padded).
                        None이면 전 frame valid로 처리.

        Returns:
            dict with:
                - "progress_seq": [T, batch, 1]
                - "ssl_losses": list of T scalar SSL losses (패딩 frame은 0)
                - "ttt_outputs_seq": [T, batch, proj_dim]
        """
        T = z_sequence.size(0)
        inner_params = self.ttt.get_inner_params()

        progress_list = []
        ssl_loss_list = []
        ttt_output_list = []

        for t in range(T):
            z_t = z_sequence[t]  # [B, D]

            if valid_mask is not None and not valid_mask[t].any():
                # 배치 내 모든 아이템이 패딩 → TTT 업데이트 스킵, ssl_loss = 0
                ssl_loss_list.append(torch.tensor(0.0, device=z_t.device))
                ttt_output = self.ttt.forward_functional(z_t, inner_params)
            else:
                # 1. Inner update (θ_{t-1} → θ_t) + SSL loss
                inner_params, ssl_loss = self.ttt.ttt_step_functional(
                    z_t, inner_params, create_graph=create_graph
                )
                # 일부만 패딩인 경우: valid fraction만큼 ssl_loss 스케일
                if valid_mask is not None:
                    valid_frac = valid_mask[t].float().mean()
                    ssl_loss = ssl_loss * valid_frac
                ssl_loss_list.append(ssl_loss)

                # 2. Forward with updated params (θ_t)
                ttt_output = self.ttt.forward_functional(z_t, inner_params)

            ttt_output_list.append(ttt_output)

            # 3. Progress prediction
            progress = self.progress_head(ttt_output)
            progress_list.append(progress)

        return {
            "progress_seq": torch.stack(progress_list),
            "ssl_losses": ssl_loss_list,
            "ttt_outputs_seq": torch.stack(ttt_output_list),
        }

    def predict_progress(self, z: torch.Tensor, update: bool = True) -> torch.Tensor:
        """진행률만 예측. 실패 감지 시 사용."""
        if update:
            self.ttt.ttt_step(z)
        ttt_output = self.ttt(z)
        return self.progress_head(ttt_output)

    # ──────────────────────────────────────────────
    # Episode lifecycle
    # ──────────────────────────────────────────────

    def reset(self) -> None:
        """새로운 trajectory/episode 시작 시 inner state 리셋."""
        self.ttt.reset_inner_state()

    def save_init(self) -> None:
        """현재 inner model 상태를 초기값으로 저장."""
        self.ttt.save_init_state()


class TTTVLAAdapter(nn.Module):
    """
    Phase 2 학습 + 추론용 통합 모듈.

    pretrained ProgressPredictor (frozen) + VLAProjector를 결합.
    Phase 2에서는 VLAProjector만 학습하고,
    추론 시에는 실패 감지 + action 수정을 모두 수행.

    사용법:
        # Phase 1 체크포인트 로드
        predictor = ProgressPredictor(input_dim=768, proj_dim=64)
        predictor.load_state_dict(torch.load("phase1_ckpt.pt"))

        # Phase 2 어댑터 생성 (predictor 자동 freeze)
        adapter = TTTVLAAdapter(predictor, vla_dim=2048, injection_mode="hidden_add")

    Args:
        predictor: 학습된 ProgressPredictor 인스턴스
        vla_dim: VLA hidden/logit 차원
        injection_mode: VLA 주입 방식
        alpha: injection scale
        learnable_alpha: α를 학습 가능하게 할지 여부
    """

    def __init__(
        self,
        predictor: ProgressPredictor,
        vla_dim: int,
        injection_mode: InjectionMode = "hidden_add",
        alpha: float = 1.0,
        learnable_alpha: bool = False,
    ):
        super().__init__()

        self.predictor = predictor
        self.vla_projector = VLAProjector(
            ttt_dim=predictor.proj_dim,
            vla_dim=vla_dim,
            mode=injection_mode,
            alpha=alpha,
            learnable_alpha=learnable_alpha,
        )

        # Phase 2 기본 세팅: predictor freeze
        self.freeze_predictor()

    def freeze_predictor(self) -> None:
        """ProgressPredictor (TTT + Head)를 freeze."""
        for param in self.predictor.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """모든 파라미터 학습 가능 (joint fine-tuning 시)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        z: torch.Tensor,
        update: bool = True,
        vla_hidden: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        단일 timestep 처리: 진행률 예측 + VLA action 수정.

        Args:
            z: VLA encoder output [batch, input_dim]
            update: TTT self-update 수행 여부
            vla_hidden: VLA hidden state [batch, vla_dim] (주입 시 필요)

        Returns:
            dict with:
                - "progress": 진행률 예측 [batch, 1]
                - "ttt_output": TTT adapted representation [batch, proj_dim]
                - "ssl_loss": self-supervised loss
                - "vla_injection": VLA 수정 결과
        """
        result = self.predictor(z, update=update)
        vla_injection = self.vla_projector(result["ttt_output"], vla_hidden)
        result["vla_injection"] = vla_injection
        return result

    def inject_to_vla(
        self,
        z: torch.Tensor,
        vla_hidden: Optional[torch.Tensor] = None,
        update: bool = True,
    ) -> torch.Tensor:
        """VLA action 수정만 수행. 추론 시 사용."""
        if update:
            self.predictor.ttt.ttt_step(z)
        ttt_output = self.predictor.ttt(z)
        return self.vla_projector(ttt_output, vla_hidden)

    # ──────────────────────────────────────────────
    # Episode lifecycle (predictor에 위임)
    # ──────────────────────────────────────────────

    def reset(self) -> None:
        """새로운 trajectory/episode 시작 시 inner state 리셋."""
        self.predictor.reset()

    def save_init(self) -> None:
        """현재 inner model 상태를 초기값으로 저장."""
        self.predictor.save_init()
