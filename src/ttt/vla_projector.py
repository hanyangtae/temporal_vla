"""
VLA Projector — TTT 출력을 VLA action 수정에 사용하기 위한 projection 모듈.

Phase 2에서 Δv 기반으로 학습된다:
    Δv = v(s_{t+1}) - v(s_t)  (frozen ProgressHead가 계산)
    Loss = -Δv × log P(a_expert | s_t, θ_t)
    → 진전 구간(Δv > 0)의 expert action을 강화, 퇴보 구간은 억제.

주입 방식:
    1. hidden_add: h_final = h_LLM + α · Proj(h_TTT)
       → VLA output head 직전 hidden state에 add. 연속 출력 모델(pi0 등).
    2. logit_shift: L_final = L_VLA + α · Proj(h_TTT)
       → VLA 출력 logit에 직접 add. 이산화(tokenized) 출력 모델 전용.
    3. film: h' = γ(h_TTT) ⊙ x + β(h_TTT)
       → Diffusion action expert에 FiLM conditioning.
    4. token: VLA input token 형태로 변환하여 입력 시퀀스에 추가.
       → 가장 단순. 낮은 성능 예상이나 baseline으로 유용.
"""

import torch
import torch.nn as nn

from typing import Literal


InjectionMode = Literal["hidden_add", "logit_shift", "film", "token"]


class VLAProjector(nn.Module):
    """
    TTT output → VLA action 수정을 위한 projection.

    Phase 2 학습 시 이 모듈만 학습되고, TTT + ProgressHead는 frozen.
    Δv 신호를 통해 "진전하는 action"의 방향을 학습한다.

    Args:
        ttt_dim: TTT module output 차원 (= proj_dim)
        vla_dim: VLA hidden/logit/token 차원 (모델마다 다름)
        mode: 주입 방식
        alpha: injection scale (α). 수정 강도 제어. 학습 가능하게 설정 가능.
        learnable_alpha: alpha를 학습 가능한 파라미터로 만들지 여부
    """

    def __init__(
        self,
        ttt_dim: int,
        vla_dim: int,
        mode: InjectionMode = "hidden_add",
        alpha: float = 1.0,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.ttt_dim = ttt_dim
        self.vla_dim = vla_dim

        # Injection scale α
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

        if mode == "hidden_add":
            # h_final = h_LLM + α · Proj(h_TTT)
            self.proj = nn.Sequential(
                nn.Linear(ttt_dim, vla_dim),
                nn.LayerNorm(vla_dim),
            )

        elif mode == "logit_shift":
            # L_final = L_VLA + α · Proj(h_TTT)
            # vla_dim = vocab_size (또는 discrete action bins)
            self.proj = nn.Sequential(
                nn.Linear(ttt_dim, vla_dim),
            )

        elif mode == "film":
            # h' = γ(h_TTT) ⊙ x + β(h_TTT)
            self.scale_proj = nn.Linear(ttt_dim, vla_dim)
            self.shift_proj = nn.Linear(ttt_dim, vla_dim)
            # scale을 1 근처로 초기화 (identity에 가깝게 시작)
            nn.init.zeros_(self.scale_proj.weight)
            nn.init.ones_(self.scale_proj.bias)
            nn.init.zeros_(self.shift_proj.weight)
            nn.init.zeros_(self.shift_proj.bias)

        elif mode == "token":
            # h_TTT → VLA input token embedding 차원으로 변환
            self.proj = nn.Sequential(
                nn.Linear(ttt_dim, vla_dim),
                nn.LayerNorm(vla_dim),
            )

        else:
            raise ValueError(f"Unknown injection mode: {mode}")

    def forward(
        self,
        ttt_output: torch.Tensor,
        vla_hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            ttt_output: TTT module output [batch, ttt_dim]
            vla_hidden: VLA hidden state / logit [batch, vla_dim]
                - hidden_add: output head 직전 hidden state
                - logit_shift: VLA 출력 logit
                - film: diffusion expert의 중간 feature
                - token: 불필요 (None)
        Returns:
            mode별 결과:
            - hidden_add: [batch, vla_dim] (h + α·proj)
            - logit_shift: [batch, vla_dim] (L + α·proj)
            - film: [batch, vla_dim] (γ⊙h + β)
            - token: [batch, 1, vla_dim] (추가 input token)
        """
        if self.mode == "hidden_add":
            projected = self.alpha * self.proj(ttt_output)
            if vla_hidden is not None:
                return vla_hidden + projected
            return projected

        elif self.mode == "logit_shift":
            projected = self.alpha * self.proj(ttt_output)
            if vla_hidden is not None:
                return vla_hidden + projected
            return projected

        elif self.mode == "film":
            gamma = self.scale_proj(ttt_output)
            beta = self.shift_proj(ttt_output)
            return gamma * vla_hidden + beta

        elif self.mode == "token":
            projected = self.proj(ttt_output)
            # [batch, vla_dim] → [batch, 1, vla_dim]
            return projected.unsqueeze(-2)

    def get_injection_only(self, ttt_output: torch.Tensor) -> torch.Tensor:
        """
        VLA hidden 없이 injection 값만 반환.
        vla_hidden과 합치는 것은 호출자가 수행.

        hidden_add, logit_shift: α · Proj(h_TTT)
        film: (γ, β) tuple
        token: [batch, 1, vla_dim]
        """
        if self.mode in ("hidden_add", "logit_shift"):
            return self.alpha * self.proj(ttt_output)
        elif self.mode == "film":
            gamma = self.scale_proj(ttt_output)
            beta = self.shift_proj(ttt_output)
            return gamma, beta
        elif self.mode == "token":
            return self.proj(ttt_output).unsqueeze(-2)
