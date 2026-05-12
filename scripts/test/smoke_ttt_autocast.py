"""TTT meta_forward bf16 autocast smoke test.

목적
----
Phase 2 GR00T finetune 의 TTT inner-loop (`predictor.meta_forward`) 를 bf16 autocast
안에서 돌리는 최적화의 안전성/효과를 검증.

검증 항목
---------
1. NaN/Inf 발생 여부 (bf16 정밀도 손실로 발산하지 않는지)
2. fp32 baseline 대비 output diff (progress_seq) — bf16 정밀도 한계 (~1e-2) 내인지
3. wall-clock speedup

**측정 결과 (2026-05-12, A6000, T=485, B=8, D=2048)**:
- 정확도: 안전 (NaN 0, progress max_abs_diff ~3e-3)
- 속도: **bf16 이 14% 더 느림** (1054 ms → 1221 ms per iter)
- 추정 원인: 작은 Linear (2048→2048) sequential T 반복 → bf16 cast overhead >
  tensor core 이득. autocast context 의 매 iter enter/exit 비용도 큰 듯.
- 결론: bf16 autocast 미적용. fp32 그대로 유지.

사용
----
    docker exec groot python /temporal_vla/scripts/test/smoke_ttt_autocast.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/temporal_vla")
sys.path.insert(0, str(REPO_ROOT))

from src.ttt.predictor import ProgressPredictor


PHASE1_CKPT = REPO_ROOT / "outputs/train/phase1_groot_robocasa/20260511_1040/epoch_08.pt"
INPUT_DIM = 2048
PROJ_DIM = 2048
T_LIST = [200, 485]  # avg, p99
BATCH = 8
N_ITER = 5
WARMUP = 2


def build_predictor(device: torch.device) -> ProgressPredictor:
    pred = ProgressPredictor(
        input_dim=INPUT_DIM,
        proj_dim=PROJ_DIM,
        inner_model_type="linear",
        inner_expansion=4,
        eta_base=0.1,
        learnable_eta=False,
        head_hidden_dim=128,
    )
    state = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    pred.load_state_dict(state)
    pred.eval()
    pred.to(device)
    # f_adapt requires_grad=True 로 풀어줘야 autograd.grad 가 inner update 계산 가능
    for n, p in pred.named_parameters():
        p.requires_grad = "f_adapt" in n
    return pred


def run_once(pred: ProgressPredictor, z_seq: torch.Tensor, valid_mask: torch.Tensor,
             use_autocast: bool) -> dict:
    if use_autocast:
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = torch.autocast(device_type="cuda", enabled=False)
    with ctx:
        out = pred.meta_forward(z_seq, create_graph=False, valid_mask=valid_mask)
    return out


def time_iters(pred, z_seq, valid_mask, use_autocast: bool, n: int) -> tuple[float, dict]:
    # warmup
    for _ in range(WARMUP):
        _ = run_once(pred, z_seq, valid_mask, use_autocast)
    torch.cuda.synchronize()
    t0 = time.time()
    last_out = None
    for _ in range(n):
        last_out = run_once(pred, z_seq, valid_mask, use_autocast)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    return dt, last_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"
    print(f"[smoke] device={device}, ckpt={PHASE1_CKPT}")

    pred = build_predictor(device)

    for T in T_LIST:
        print(f"\n=== T={T}, B={BATCH}, D={INPUT_DIM} ===")
        torch.manual_seed(0)
        z_seq = torch.randn(T, BATCH, INPUT_DIM, device=device, dtype=torch.float32)
        valid_mask = torch.ones(T, BATCH, dtype=torch.bool, device=device)

        # fp32 baseline
        dt_fp32, out_fp32 = time_iters(pred, z_seq, valid_mask, use_autocast=False, n=N_ITER)
        # bf16 autocast
        dt_bf16, out_bf16 = time_iters(pred, z_seq, valid_mask, use_autocast=True, n=N_ITER)

        ps_fp32 = out_fp32["progress_seq"].float()
        ps_bf16 = out_bf16["progress_seq"].float()
        diff = (ps_fp32 - ps_bf16).abs()
        nan_inf_fp32 = (~torch.isfinite(ps_fp32)).sum().item()
        nan_inf_bf16 = (~torch.isfinite(ps_bf16)).sum().item()

        print(f"  fp32        : {dt_fp32*1000:8.1f} ms/iter")
        print(f"  bf16 auto   : {dt_bf16*1000:8.1f} ms/iter   (speedup x{dt_fp32/dt_bf16:.2f})")
        print(f"  progress diff: max_abs={diff.max().item():.4e}  "
              f"mean_abs={diff.mean().item():.4e}")
        print(f"  NaN/Inf      : fp32={nan_inf_fp32}  bf16={nan_inf_bf16}")
        assert nan_inf_bf16 == 0, "bf16 path produced NaN/Inf"
        # progress 는 [0,1] 범위 sigmoid output → 1e-2 absolute diff 이면 무시 가능
        assert diff.max().item() < 0.05, f"progress diff too large: {diff.max().item()}"

    print("\n[smoke] OK")


if __name__ == "__main__":
    main()
