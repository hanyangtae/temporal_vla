"""
Progress Head

TTT module의 출력을 받아 task 진행률(0~1)을 예측하는 regression head.

실패 루프 탈출에서의 역할:
    1. Phase 1 학습: expert trajectory에서 v(s_t) ≈ t/T 예측 학습.
    2. Phase 2: frozen 상태에서 Δv = v(s_{t+1}) - v(s_t) 계산.
       이 Δv가 VLAProjector 학습의 핵심 신호.
    3. 추론: 진행률 예측의 단조증가 이탈을 모니터링하여 실패 감지.

입력: TTT module output [batch, proj_dim]
출력: progress scalar [batch, 1] in [0, 1]

참고: VITA Section 3.1.3, Eq. (3)
"""

import torch
import torch.nn as nn


class ProgressHead(nn.Module):
    """
    Task 진행률 예측 head.

    Two-layer MLP → sigmoid로 [0, 1] 출력.

    Args:
        input_dim: TTT module output 차원 (= proj_dim)
        hidden_dim: MLP hidden dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: TTT module output [batch, input_dim]
        Returns:
            progress: [batch, 1] in [0, 1]
        """
        return self.head(x)
