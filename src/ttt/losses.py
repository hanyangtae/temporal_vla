"""
Loss functions for TTT 실패 루프 탈출 모듈 학습.

Phase 1 — Progress Predictor 학습:
    L_phase1 = L_pred + λ_self · L_ssl + λ_mono · L_mono
    - L_pred: MSE(v_pred, t/T)  진행률 예측
    - L_ssl: TTT self-supervised reconstruction loss
    - L_mono: 단조증가 위반 페널티

Phase 2 — VLA Projector 학습 (Δv 기반):
    L_phase2 = -Δv · log P(a_expert | s_t, θ_t)
    - Δv = v(s_{t+1}) - v(s_t)  (frozen ProgressHead 계산)
    - Δv > 0 → 진전 구간 → expert action 강화
    - Δv < 0 → 퇴보 구간 → expert action 억제
    - θ_t = VLA params + Projector injection

참고:
    - VITA Section 3.2.1 (meta-learning training objective)
    - 연구 계획서: Δv 기반 projector learning
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Phase 1: Progress Predictor 학습용
# ──────────────────────────────────────────────

def progress_prediction_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Progress prediction loss (MSE).

    Args:
        pred: predicted progress [batch, 1] in [0, 1]
        target: ground truth progress [batch, 1] (= t/T for expert trajectories)
    Returns:
        scalar MSE loss
    """
    return F.mse_loss(pred, target)


def monotonicity_loss(progress_sequence: torch.Tensor) -> torch.Tensor:
    """
    단조증가 위반 페널티.
    expert trajectory에서 진행률은 단조증가해야 함.
    연속된 예측 간 감소가 있으면 페널티.

    Args:
        progress_sequence: [T, 1] or [T] 형태의 시간순 progress 예측
    Returns:
        scalar loss (위반 정도)
    """
    progress = progress_sequence.squeeze()
    diffs = progress[1:] - progress[:-1]
    violations = F.relu(-diffs)  # 감소분만 페널티
    return violations.mean()


def phase1_loss(
    ssl_loss: torch.Tensor,
    pred_loss: torch.Tensor,
    mono_loss: torch.Tensor = None,
    lambda_self: float = 0.5,
    lambda_mono: float = 0.1,
) -> torch.Tensor:
    """
    Phase 1 meta-learning outer loss (VITA Section 3.2.1).

    L = L_pred + λ_self · L_ssl (+ λ_mono · L_mono)

    주의: pred_loss는 ProgressPredictor.meta_forward()에서 나온 값이어야 함.
    meta_forward가 inner update를 통한 computational graph를 유지하므로,
    이 loss를 backward()하면 θ_0, P_K, P_V, P_Q, h, η가 모두 최적화됨.

    Args:
        ssl_loss: TTT self-supervised reconstruction loss (meta_forward ssl_losses의 평균)
        pred_loss: progress prediction MSE loss (meta_forward progress_seq 기반)
        mono_loss: 단조증가 위반 페널티 (optional, VITA 논문에는 없는 추가 항)
        lambda_self: SSL loss 가중치 (VITA 기본값 0.5)
        lambda_mono: monotonicity loss 가중치
    """
    total = pred_loss + lambda_self * ssl_loss
    if mono_loss is not None:
        total = total + lambda_mono * mono_loss
    return total


# ──────────────────────────────────────────────
# Phase 2: VLA Projector 학습용 (Δv 기반)
# ──────────────────────────────────────────────

def compute_delta_v(
    progress_current: torch.Tensor,
    progress_next: torch.Tensor,
) -> torch.Tensor:
    """
    Δv = v(s_{t+1}) - v(s_t).

    frozen ProgressHead가 예측한 연속 두 timestep의 진행률 차이.
    Δv > 0: 진전 (이 구간의 action이 task 완수 방향)
    Δv < 0: 퇴보 (이 구간의 action이 실패 방향)

    Args:
        progress_current: v(s_t) [batch, 1]
        progress_next: v(s_{t+1}) [batch, 1]
    Returns:
        delta_v [batch, 1]
    """
    return (progress_next - progress_current).detach()


def projector_loss(
    log_prob_expert: torch.Tensor,
    delta_v: torch.Tensor,
) -> torch.Tensor:
    """
    Phase 2 Δv-weighted action loss.

    L = -Δv · log P(a_expert | s_t, θ_t)

    Δv > 0 (진전): -Δv · logP < 0 → logP 최대화 → expert action 강화
    Δv < 0 (퇴보): -Δv · logP > 0 → logP 최소화 → expert action 억제

    θ_t는 VLA original params + Projector injection이 반영된 상태에서 계산.
    이 loss로 Projector만 학습 (VLA, TTT, ProgressHead는 frozen).

    Args:
        log_prob_expert: log P(a_expert | s_t, θ_t) [batch]
            VLA가 projector injection을 받은 상태에서 expert action에 대한 log-prob.
        delta_v: Δv [batch] or [batch, 1]
            frozen ProgressHead가 계산한 progress 차이.
    Returns:
        scalar loss
    """
    delta_v = delta_v.squeeze()
    log_prob_expert = log_prob_expert.squeeze()
    return -(delta_v * log_prob_expert).mean()


def phase2_loss(
    log_prob_expert: torch.Tensor,
    delta_v: torch.Tensor,
    reg_loss: torch.Tensor = None,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    """
    Phase 2 전체 loss: VLAProjector 학습.

    L = L_proj + λ_reg · L_reg

    Args:
        log_prob_expert: log P(a_expert | s_t, θ_t) [batch]
        delta_v: Δv [batch] or [batch, 1]
        reg_loss: projector weight regularization (optional, 과도한 수정 방지)
        lambda_reg: regularization 가중치
    """
    total = projector_loss(log_prob_expert, delta_v)
    if reg_loss is not None:
        total = total + lambda_reg * reg_loss
    return total
