"""
TTT (Test-Time Training) Module

VLA encoder 출력을 받아 self-supervised loss로 자체 업데이트되는 adaptation module.
hidden state = 이 모듈의 weights (W). 매 timestep마다 gradient step으로 W가 업데이트되며,
이를 통해 temporal context를 파라미터에 축적한다.

실패 루프 탈출에서의 역할:
    추론 시 매 timestep SSL로 자체 업데이트됨으로써, 현재 trajectory의
    temporal context를 inner model 파라미터에 축적. 이 축적된 정보가
    ProgressHead와 VLAProjector의 입력이 되어 실패 감지 + action 수정을 가능하게 한다.

핵심 구성요소:
    - f_adapt: inner model (Linear 또는 MLP). 이 weights가 TTT의 hidden state.
    - P_K, P_V: meta-learned projection. SSL reconstruction task를 정의.
    - P_Q: meta-learned projection. output(test view)을 정의.
    - η (eta): inner-loop learning rate. learnable or fixed.

Update rule (매 timestep t):
    ℓ_self(z_t; θ_{t-1}) = ||f(P_K·z_t; θ_{t-1}) - P_V·z_t||^2
    θ_t = θ_{t-1} - η·∇_θ ℓ_self(z_t; θ_{t-1})

Output rule:
    output_t = f(P_Q·z_t; θ_t)

참고: Sun et al. (2024) Section 2.1-2.3, Figure 5
참고: VITA (Ziakas & Russo, 2026) Section 3.1.2, Eq. (1)-(2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Literal


class TTTInnerModel(nn.Module):
    """
    TTT의 inner model f.
    hidden state로 사용되는 모델. Linear 또는 MLP.

    f(x) = x + LN(f_res(x))  (residual + LayerNorm for stability)

    Args:
        input_dim: 입력 차원 (= projection dimension d')
        model_type: "linear" (TTT-Linear) 또는 "mlp" (TTT-MLP)
        expansion_factor: MLP일 때 hidden dim = input_dim * expansion_factor
    """

    def __init__(
        self,
        input_dim: int,
        model_type: Literal["linear", "mlp"] = "linear",
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.model_type = model_type

        if model_type == "linear":
            self.f_res = nn.Linear(input_dim, input_dim, bias=False)
        elif model_type == "mlp":
            hidden_dim = input_dim * expansion_factor
            self.f_res = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x + LN(f_res(x))"""
        return x + self.ln(self.f_res(x))

    def functional_forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Explicit params를 사용한 functional forward.
        Meta-learning에서 inner update를 통한 gradient 흐름 유지용.

        Args:
            x: input [batch, input_dim]
            params: {name: tensor} dict. self.state_dict()와 같은 키 구조.
        """
        if self.model_type == "linear":
            res = F.linear(x, params["f_res.weight"])
        elif self.model_type == "mlp":
            h = F.linear(x, params["f_res.0.weight"], params["f_res.0.bias"])
            h = F.gelu(h)
            res = F.linear(h, params["f_res.2.weight"], params["f_res.2.bias"])

        res = F.layer_norm(
            res, [res.size(-1)], params["ln.weight"], params["ln.bias"]
        )
        return x + res


class TTTModule(nn.Module):
    """
    TTT (Test-Time Training) Module.

    VLA encoder 출력을 받아:
    1. meta-learned SSL task로 inner model을 self-update
    2. updated inner model로 output 생성

    학습 시 (outer loop):
        - P_K, P_V, P_Q, θ_init (inner model의 초기 weights), η가 최적화됨
        - meta-learning: SSL update 후 prediction loss가 줄어들도록 학습

    추론 시 (inner loop):
        - 매 timestep마다 ℓ_self로 inner model weights를 gradient step
        - sequential update로 temporal context를 파라미터에 축적

    Args:
        input_dim: VLA encoder 출력 차원
        proj_dim: projection 차원 (d'). inner model의 입출력 차원.
        inner_model_type: inner model 종류 ("linear" or "mlp")
        inner_expansion: MLP inner model의 expansion factor
        eta_base: inner-loop base learning rate
        learnable_eta: eta를 입력에 따라 학습할지 여부
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 64,
        inner_model_type: Literal["linear", "mlp"] = "linear",
        inner_expansion: int = 4,
        eta_base: float = 0.1,
        learnable_eta: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.eta_base = eta_base

        # Meta-learned projections (outer-loop parameters)
        # P_K: training view (corrupted input for SSL)
        self.P_K = nn.Linear(input_dim, proj_dim, bias=False)
        # P_V: label view (reconstruction target for SSL)
        self.P_V = nn.Linear(input_dim, proj_dim, bias=False)
        # P_Q: test view (output을 위한 projection)
        self.P_Q = nn.Linear(input_dim, proj_dim, bias=False)

        # Inner model f (its weights = TTT hidden state)
        self.f_adapt = TTTInnerModel(
            input_dim=proj_dim,
            model_type=inner_model_type,
            expansion_factor=inner_expansion,
        )

        # Learnable η: η(x) = η_base * σ(θ_lr · x)
        self.learnable_eta = learnable_eta
        if learnable_eta:
            self.theta_lr = nn.Linear(input_dim, 1, bias=False)

    def compute_ssl_loss(
        self, z: torch.Tensor, params: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Self-supervised reconstruction loss.

        ℓ_self(z; θ) = ||f(P_K·z; θ) - P_V·z||^2

        Args:
            z: input representation [batch, input_dim] or [input_dim]
            params: inner model params (None이면 현재 self.f_adapt 사용)
        Returns:
            scalar loss
        """
        train_view = self.P_K(z)  # corrupted input
        label_view = self.P_V(z)  # reconstruction target

        # inner model forward (functional forward if params given)
        if params is not None:
            pred = self._functional_forward(train_view, params)
        else:
            pred = self.f_adapt(train_view)

        return F.mse_loss(pred, label_view)

    def get_eta(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inner-loop learning rate.
        η(z) = η_base * σ(θ_lr · z) if learnable, else η_base.
        """
        if self.learnable_eta:
            return self.eta_base * torch.sigmoid(self.theta_lr(z))
        return self.eta_base

    def ttt_step(self, z: torch.Tensor) -> None:
        """
        하나의 TTT update step (in-place).
        θ_t = θ_{t-1} - η·∇_θ ℓ_self(z_t; θ_{t-1})

        추론 시 매 timestep마다 호출. 이를 통해 temporal context가
        inner model의 파라미터에 축적된다.

        Args:
            z: input representation [input_dim] or [1, input_dim]
        """
        ssl_loss = self.compute_ssl_loss(z)
        eta = self.get_eta(z)

        # inner model의 파라미터만 업데이트
        grads = torch.autograd.grad(
            ssl_loss, self.f_adapt.parameters(), create_graph=self.training
        )
        with torch.no_grad() if not self.training else torch.enable_grad():
            for param, grad in zip(self.f_adapt.parameters(), grads):
                param.data = param - eta * grad

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Output rule: output = f(P_Q·z; θ_t)

        ttt_step()이 먼저 호출된 후 사용.

        Args:
            z: input representation [batch, input_dim] or [input_dim]
        Returns:
            adapted output [batch, proj_dim] or [proj_dim]
        """
        test_view = self.P_Q(z)
        return self.f_adapt(test_view)

    def reset_inner_state(self) -> None:
        """
        Inner model을 초기 상태(θ_0)로 리셋.
        새로운 trajectory/episode 시작 시 호출.

        Note: 학습 시 θ_init = W_0는 outer-loop parameter로 학습됨.
              여기서는 현재 저장된 초기값으로 복원.
        """
        if not hasattr(self, "_init_state"):
            return
        self.f_adapt.load_state_dict(self._init_state)

    def save_init_state(self) -> None:
        """현재 inner model 상태를 초기 상태로 저장."""
        self._init_state = {
            k: v.clone() for k, v in self.f_adapt.state_dict().items()
        }

    def get_inner_params(self) -> dict:
        """
        Inner model의 파라미터를 dict로 반환.
        Meta-learning의 시작점 (θ_0). 반환값은 실제 파라미터 텐서 참조.
        """
        return {name: param for name, param in self.f_adapt.named_parameters()}

    def compute_ssl_loss_functional(
        self, z: torch.Tensor, inner_params: dict
    ) -> torch.Tensor:
        """
        Functional SSL loss.
        ℓ_self(z; θ) = ||f(P_K·z; θ) - P_V·z||²

        P_K, P_V는 outer-loop params (autograd 자동 추적),
        inner model은 explicit inner_params로 forward.
        """
        train_view = self.P_K(z)
        label_view = self.P_V(z)
        pred = self.f_adapt.functional_forward(train_view, inner_params)
        return F.mse_loss(pred, label_view)

    def ttt_step_functional(
        self, z: torch.Tensor, inner_params: dict, create_graph: bool = True
    ) -> tuple:
        """
        Functional inner update. computational graph를 유지한 채 새 params dict 반환.
        θ_t = θ_{t-1} - η·∇_θ ℓ_self(z_t; θ_{t-1})

        Args:
            create_graph: True이면 outer gradient용 2차 graph 유지 (학습 시).
                          False이면 inner update만 수행하고 graph를 끊음 (eval 시).
        Returns:
            (new_inner_params, ssl_loss)
        """
        ssl_loss = self.compute_ssl_loss_functional(z, inner_params)
        eta = self.get_eta(z)
        # z가 [batch, D]일 때 eta는 [batch, 1] → 배치 평균으로 scalar화
        if isinstance(eta, torch.Tensor) and eta.dim() > 0:
            eta = eta.mean()
        grads = torch.autograd.grad(
            ssl_loss, inner_params.values(), create_graph=create_graph
        )
        new_params = {
            k: inner_params[k] - eta * g
            for k, g in zip(inner_params.keys(), grads)
        }
        return new_params, ssl_loss

    def forward_functional(
        self, z: torch.Tensor, inner_params: dict
    ) -> torch.Tensor:
        """
        Functional output. explicit inner params로 계산.
        output = f(P_Q·z; θ_t)
        """
        test_view = self.P_Q(z)
        return self.f_adapt.functional_forward(test_view, inner_params)

    def _functional_forward(
        self, x: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """Inner model의 functional forward. TTTInnerModel에 위임."""
        return self.f_adapt.functional_forward(x, params)
