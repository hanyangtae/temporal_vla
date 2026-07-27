"""src/sae 코어 회귀 테스트.

동료 레포(task_classification@88543a2)에는 SAE forward/loss 단위 테스트가 없었다
(tests 는 metrics·colormap 만 — docs/steering/29_sae_port_review.md §6 "위험·함정").
그래서 이식하면서 top-k 마스킹·단위노름 사전·학습 수렴을 고정하는 테스트를 새로 쓴다.
핸드아웃 체크리스트 = docs/steering/30_sae_g1_port_handout.md §4 Phase A4.

전부 CPU·소형 텐서로 몇 초 안에 끝나야 한다 (CI 부담 없이 자주 돌리는 게 목적).
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
import torch

from src.sae.cluster import dead_fraction
from src.sae.models import BaseAE, DecoderLinearDict, EncoderTopK, build_model, sae_config
from src.sae.train import encode_all, set_seed, train_sae

SAE_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "sae"


# ---------------------------------------------------------------- (i) top-k
def test_topk_activates_exactly_k():
    """top-k 마스킹 후 행마다 활성(>0) feature 수가 정확히 k."""
    set_seed(0)
    m, k, D, N = 64, 8, 16, 256
    enc = EncoderTopK(D, m, k)
    h = enc(torch.randn(N, D))

    n_active = (h > 0).sum(-1)
    assert h.shape == (N, m)
    # ReLU 양수 개수가 k 이상이면(랜덤 초기화·랜덤 입력이면 m=64 중 ~32개) 정확히 k개가 산다.
    assert torch.all(n_active == k), f"활성 수 분포: {torch.unique(n_active).tolist()}"
    # 마스킹은 상한이기도 하다 — 어떤 경우에도 k를 넘지 않아야 한다.
    assert int(n_active.max()) <= k
    assert EncoderTopK.density(h) == pytest.approx(k / m, rel=1e-6)


def test_topk_no_mask_when_k_ge_m():
    """k >= m 이면 top-k 가 무의미 → ReLU 결과 그대로 (마스킹 없음)."""
    set_seed(0)
    m, D, N = 32, 16, 64
    enc = EncoderTopK(D, m, k=m)
    x = torch.randn(N, D)
    h = enc(x)
    assert torch.allclose(h, torch.relu(enc.net(x)))


def test_topk_preserves_values_not_gate():
    """마스킹은 0/1 게이트가 아니라 살아남은 원소의 '값'을 보존한다."""
    set_seed(1)
    enc = EncoderTopK(8, 16, k=4)
    x = torch.randn(32, 8)
    raw = torch.relu(enc.net(x))
    h = enc(x)
    kept = h > 0
    assert torch.allclose(h[kept], raw[kept])


# ---------------------------------------------------------------- (ii) 단위노름 사전
def test_decoder_columns_unit_norm_at_init():
    """DecoderLinearDict 가 실제로 쓰는 사전의 열 노름이 1."""
    set_seed(0)
    dec = DecoderLinearDict(m=32, out_dim=12)
    norms = dec.dictionary().norm(dim=0)
    assert torch.allclose(norms, torch.ones(32), atol=1e-5)


def test_decoder_columns_unit_norm_after_training_steps():
    """학습 스텝 후에도 열 노름이 1 — 정규화가 파라미터가 아니라 forward 에서 일어나므로
    옵티마이저가 lin.weight 를 아무리 키워도 유효 사전은 계속 단위노름이어야 한다."""
    set_seed(0)
    D, m, k = 12, 32, 4
    model = BaseAE(EncoderTopK(D, m, k), DecoderLinearDict(m, D), loss="mse")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(128, D)

    w0 = model.dec.lin.weight.detach().clone()
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        model.loss(x).backward()
        opt.step()

    # 파라미터는 실제로 움직였는데(테스트가 공회전이 아님을 확인)
    assert not torch.allclose(w0, model.dec.lin.weight.detach())
    # 유효 사전 열 노름은 여전히 1이다.
    norms = model.dec.dictionary().norm(dim=0)
    assert torch.allclose(norms, torch.ones(m), atol=1e-5)
    # 원 파라미터 열 노름은 1이 아니어도 된다(정규화가 forward 에 있다는 사실 고정).
    assert model.dec.lin.weight.norm(dim=0).shape == (m,)


def test_mse_loss_freezes_logvar():
    """loss='mse' 면 decoder.logvar 가 0에 동결된다(σ=1 고정 가우시안)."""
    model = BaseAE(EncoderTopK(8, 16, 4), DecoderLinearDict(16, 8), loss="mse")
    assert torch.allclose(model.dec.logvar, torch.zeros(8))
    assert model.dec.logvar.requires_grad is False


def test_build_model_rejects_bad_config():
    with pytest.raises(ValueError):
        build_model({"input_dim": 8, "latent_dim": 16, "encoder": {"type": "topk", "k": 2}})
    with pytest.raises(ValueError):
        build_model({"input_dim": 8, "latent_dim": 16,
                     "encoder": {"type": "nope"}, "decoder": {"type": "linear_dict"}})


# ---------------------------------------------------------------- (iii) 학습으로 loss 감소
def test_reconstruction_loss_decreases():
    """미니배치 학습으로 재구성 손실(val)이 유의하게 떨어진다."""
    set_seed(0)
    D, m, k, N = 24, 96, 8, 2000
    rng = np.random.default_rng(0)
    # 저차원 구조가 있는 데이터(랜덤 노이즈만이면 재구성 여지가 없다)
    A = rng.normal(size=(6, D))
    X = (rng.normal(size=(N, 6)) @ A + 0.05 * rng.normal(size=(N, D))).astype(np.float32)

    model = build_model(sae_config(D, expansion=m / D, k=k))
    best = train_sae(model, X[:1600], X[1600:], epochs=60, patience=60, min_epochs=60,
                     batch_size=256, lr=1e-2, device="cpu", seed=0, verbose=False)

    hist = best["history"]
    assert len(hist) == 60
    first_val, last_val = hist[0][2], hist[-1][2]
    assert last_val < first_val, f"val loss 가 안 줄었다: {first_val} → {last_val}"
    assert best["loss"] <= last_val
    # train loss 도 단조 추세로 감소 (초반 평균 > 후반 평균)
    tr = np.array([h[1] for h in hist])
    assert tr[:10].mean() > tr[-10:].mean()


# ---------------------------------------------------------------- (iv) 합성 사전 복원
def test_synthetic_sparse_dictionary_recovery():
    """알려진 sparse 사전으로 만든 데이터를 SAE 가 높은 R² 로 재구성한다.

    참 생성모델: x = A @ s, A[D, m0] 단위노름 열, s 는 행마다 k0 개만 0이 아님.
    top-k SAE(m > m0, k = k0)가 이 구조를 잡으면 재구성 R² 가 높아야 한다.
    (사전 열 자체의 일대일 복원이 아니라 '희소 구조로 재구성 가능한가' sanity.)
    """
    set_seed(0)
    rng = np.random.default_rng(7)
    D, m0, k0, N = 32, 48, 3, 6000

    A = rng.normal(size=(D, m0))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    S = np.zeros((N, m0), np.float32)
    for i in range(N):
        idx = rng.choice(m0, k0, replace=False)
        S[i, idx] = rng.uniform(1.0, 2.0, k0)
    X = (S @ A.T).astype(np.float32)

    n_tr = 4800
    model = build_model(sae_config(D, expansion=4, k=k0))   # m = 128 > m0 = 48
    train_sae(model, X[:n_tr], X[n_tr:], epochs=250, patience=250, min_epochs=250,
              batch_size=256, lr=3e-3, device="cpu", seed=0, verbose=False)

    with torch.no_grad():
        xt = torch.as_tensor(X[n_tr:])
        xhat = model.reconstruct(xt).numpy()
    x_te = X[n_tr:]
    sse = float(((x_te - xhat) ** 2).sum())
    sst = float(((x_te - x_te.mean(0)) ** 2).sum())
    r2 = 1.0 - sse / sst
    assert r2 > 0.8, f"held-out 재구성 R² = {r2:.3f} (기대 > 0.8)"

    # 희소 코드가 실제로 희소하고, 사전이 통째로 죽지 않았는지 (dead feature 진단)
    z = encode_all(model, X[:n_tr], batch_size=1024)
    assert z.shape == (n_tr, 128)
    assert np.all((z > 0).sum(1) <= k0)
    assert dead_fraction(z) < 1.0


# ---------------------------------------------------------------- (v) scipy 금지
def test_no_scipy_import_in_src_sae():
    """src/sae 어디에도 scipy import 가 없어야 한다.

    분석·fit 을 돌리는 원격 노드(승준)에 scipy 가 설치돼 있지 않다
    (memory `remote-compute-workflow`, 핸드아웃 §5·§6-5). 코어가 scipy 를 끌어오면
    원격에서 그대로 죽는다.
    """
    pat = re.compile(r"^\s*(import\s+scipy|from\s+scipy\b)", re.M)
    files = sorted(SAE_DIR.rglob("*.py"))
    assert files, f"검사할 파일이 없다: {SAE_DIR}"
    offenders = [str(p) for p in files if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"scipy import 발견: {offenders}"
