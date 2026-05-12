"""linear_preLN 의 sequential meta_forward 가 closed-form affine recurrence 와
수치 동치인지 검증.

수식 (linear_preLN):
    K_t = LN(P_K z_t),  V_t = P_V z_t,  Q_t = LN(P_Q z_t)
    pred_t = K_t + K_t W_{t-1}ᵀ                          (f(x) = x + Wx)
    ssl_t  = (1/(B·D')) ||pred_t - V_t||²
    ∇_W    = (2/(B·D')) · errᵀ K_t   where err = K_t + K_t Wᵀ - V_t
           = (2/(B·D')) [W · (K_tᵀ K_t) + (K_t - V_t)ᵀ K_t]
    W_t    = W_{t-1} · A_t + b_t
        A_t = I - (2η/(B·D')) K_tᵀ K_t          (rank ≤ B perturbation of I)
        b_t = -(2η/(B·D')) (K_t - V_t)ᵀ K_t
    output_t = Q_t + Q_t W_tᵀ

이 affine recurrence 가 성립하면 mini-batch TTT (Sun 2024 §2.3) / associative scan
직접 적용 가능. 본 test 는 그 수식이 sequential autograd.grad 경로와 같은 결과
내는지 확인.

사용:
    docker exec lerobot python /temporal_vla/scripts/test/test_linear_preLN_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/temporal_vla")
sys.path.insert(0, str(REPO_ROOT))

from src.ttt.predictor import ProgressPredictor


def closed_form_recurrence(
    pred: ProgressPredictor,
    z_seq: torch.Tensor,   # [T, B, D_in] fp32
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """수식대로 직접 W_t 를 굴려서 ttt_outputs / progress 를 계산.

    Returns
    -------
    ttt_outputs_seq : [T, B, D']
    progress_seq    : [T, B, 1]
    """
    ttt = pred.ttt
    device = z_seq.device
    T, B, _ = z_seq.shape
    D = ttt.proj_dim

    # W_0 = inner model 의 현재 weight. linear_preLN 의 f_adapt 는 nn.Linear 한 개라 bias 없음.
    W = ttt.f_adapt.f_res.weight.detach().clone()  # [D', D']
    I = torch.eye(D, device=device, dtype=W.dtype)
    scale = 2.0 * eta / (B * D)

    ttt_outs = []
    progs = []

    with torch.no_grad():
        for t in range(T):
            z_t = z_seq[t]                              # [B, D_in]
            K_t = ttt.outer_ln_K(ttt.P_K(z_t))          # [B, D']
            V_t = ttt.P_V(z_t)                          # [B, D']
            Q_t = ttt.outer_ln_Q(ttt.P_Q(z_t))          # [B, D']

            A_t = I - scale * (K_t.T @ K_t)             # [D', D']
            b_t = -scale * ((K_t - V_t).T @ K_t)        # [D', D']
            W = W @ A_t + b_t                           # affine update

            # output = Q_t + Q_t Wᵀ  (F.linear(Q_t, W) = Q_t @ Wᵀ)
            out_t = Q_t + Q_t @ W.T                     # [B, D']
            ttt_outs.append(out_t)
            progs.append(pred.progress_head(out_t))     # [B, 1]

    return torch.stack(ttt_outs), torch.stack(progs)


def run_case(T: int, B: int, D_in: int, D_prime: int, eta: float, device: str):
    torch.manual_seed(0)
    pred = ProgressPredictor(
        input_dim=D_in,
        proj_dim=D_prime,
        inner_model_type="linear_preLN",
        eta_base=eta,
        learnable_eta=False,
        head_hidden_dim=16,
    ).to(device).float()

    # autograd.grad 가 inner_params 에 대해 미분 가능하려면 requires_grad=True 필요
    for n, p in pred.named_parameters():
        p.requires_grad = "f_adapt" in n

    z_seq = torch.randn(T, B, D_in, device=device, dtype=torch.float32)

    # (1) Reference: sequential meta_forward (autograd.grad 경로)
    out_ref = pred.meta_forward(z_seq, create_graph=False, valid_mask=None)
    ttt_ref = out_ref["ttt_outputs_seq"]        # [T, B, D']
    prog_ref = out_ref["progress_seq"]          # [T, B, 1]

    # (2) Closed-form: 직접 affine recurrence
    ttt_cf, prog_cf = closed_form_recurrence(pred, z_seq, eta)

    diff_ttt = (ttt_ref - ttt_cf).abs()
    diff_prog = (prog_ref - prog_cf).abs()

    print(f"--- T={T}, B={B}, D_in={D_in}, D'={D_prime}, eta={eta} ---")
    print(f"  TTT output  max={diff_ttt.max().item():.3e}  mean={diff_ttt.mean().item():.3e}"
          f"  ref_norm={ttt_ref.abs().mean().item():.3e}")
    print(f"  Progress    max={diff_prog.max().item():.3e}  mean={diff_prog.mean().item():.3e}")

    assert diff_ttt.max().item() < 1e-4, (
        f"TTT output diff too large: {diff_ttt.max().item()}"
    )
    assert diff_prog.max().item() < 1e-4, (
        f"Progress diff too large: {diff_prog.max().item()}"
    )


def main():
    device = "cpu"  # 작은 모델 + 검증만 → cpu 로 GPU 학습과 충돌 없게.
    cases = [
        # (T, B, D_in, D', eta)
        (1,  4, 16,  8, 0.1),   # 단일 step
        (5,  4, 16,  8, 0.1),   # 짧은 sequence
        (20, 4, 16,  8, 0.1),
        (50, 8, 32, 16, 0.1),   # 좀 더 큰
        (10, 4, 16,  8, 0.5),   # 다른 eta
        (10, 4, 16,  8, 0.01),
    ]
    for case in cases:
        run_case(*case, device=device)
    print("[test] all equivalence cases OK")


if __name__ == "__main__":
    main()
