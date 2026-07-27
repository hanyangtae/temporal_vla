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


# ---------------------------------------------------------------- (v) aux-k loss
def _dead_prone_data(n=6000, D=32, m0=256, k0=4, seed=11):
    """참 사전의 **모든** 열이 쓰이는 희소 데이터.

    이렇게 해야 dead feature 가 '최적해라서 안 쓰이는 열'이 아니라 순수한 최적화 실패가
    된다 — aux-k 가 고치라고 만들어진 그 병리. (참 원자보다 사전이 크면 dead 가 정상이라
    aux-k 를 켜면 오히려 재구성이 나빠지는 게 맞다.)
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D, m0))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    S = np.zeros((n, m0), np.float32)
    for i in range(n):
        S[i, rng.choice(m0, k0, replace=False)] = rng.uniform(1.0, 2.0, k0)
    return (S @ A.T).astype(np.float32)


def test_aux_k_zero_is_bit_identical_to_legacy_path():
    """aux_k=0 이면 손실 값이 기존 경로와 **완전히 동일**하고 버퍼도 안 생긴다.

    (기존 15 run 의 재현성이 aux-k 도입으로 깨지지 않는다는 회귀 고정.)
    """
    set_seed(0)
    D, m, k = 12, 48, 4
    legacy = BaseAE(EncoderTopK(D, m, k), DecoderLinearDict(m, D), loss="mse")
    set_seed(0)
    withzero = BaseAE(EncoderTopK(D, m, k), DecoderLinearDict(m, D), loss="mse", aux_k=0)

    x = torch.randn(64, D)
    assert torch.equal(legacy.loss(x), withzero.loss(x))
    # 버퍼 미등록 → state_dict 키가 기존과 같다 (구 체크포인트 strict 로드 유지)
    assert "steps_since_active" not in dict(withzero.named_buffers())
    assert set(legacy.state_dict()) == set(withzero.state_dict())
    # sae_config 도 aux_k=0 이면 키를 넣지 않는다
    assert "aux_k" not in sae_config(D, expansion=2, k=k)
    assert sae_config(D, expansion=2, k=k, aux_k=8)["aux_k"] == 8


@pytest.mark.parametrize("seed", [0, 1])
def test_aux_k_reduces_dead_features(seed):
    """aux-k 를 켜면 같은 seed·같은 데이터에서 dead feature 비율이 줄어든다.

    실측(seed 0/1/2): dead 0.16~0.26 → 0.055, held-out R² 도 조금 올라간다.
    aux_k 는 사전 크기 대비 비율이 중요하다 — m=256 에 16 (≈6%) 은 실험용 512/6144(8%)와
    같은 대역. m 의 25% 를 넘게 잡으면 rarely-firing feature 까지 잔차 재구성으로 끌려가
    오히려 dead 가 늘었다(실측 aux_k=64/m=256: 0.23→0.28).
    """
    D, m0, k = 32, 256, 4
    X = _dead_prone_data(D=D, m0=m0, k0=k)
    m, n_tr = m0, 4800

    def run(aux_k):
        set_seed(seed)
        cfg = sae_config(D, expansion=m / D, k=k, aux_k=aux_k, aux_alpha=1 / 32,
                         dead_window=25)
        model = build_model(cfg)
        train_sae(model, X[:n_tr], X[n_tr:], epochs=30, patience=30, min_epochs=30,
                  batch_size=256, lr=3e-3, device="cpu", seed=seed, verbose=False)
        return model, dead_fraction(encode_all(model, X[:n_tr], batch_size=1024))

    m_off, dead_off = run(0)
    m_on, dead_on = run(16)
    assert dead_off > 0.1, f"기준 조건이 dead 를 충분히 만들지 못했다: {dead_off:.3f}"
    assert dead_on < dead_off, f"aux-k 가 dead 를 못 줄였다: {dead_off:.3f} → {dead_on:.3f}"
    # aux-k 를 켜도 재구성이 망가지면 안 된다 (보조손실이 주 목적을 잡아먹지 않았는지)
    with torch.no_grad():
        xt = torch.as_tensor(X[n_tr:])
        r2 = [1.0 - float(((xt - mm.reconstruct(xt)) ** 2).sum()
                          / ((xt - xt.mean(0)) ** 2).sum()) for mm in (m_off, m_on)]
    assert r2[1] >= r2[0] - 0.02, f"재구성 R²: off={r2[0]:.3f} on={r2[1]:.3f}"


def test_aux_k_keeps_dictionary_unit_norm():
    """aux-k 학습 후에도 유효 사전 열 노름이 1 (decode_nobias 경로도 정규화를 통과)."""
    D, m, k = 32, 128, 4
    X = _dead_prone_data(D=D, m0=m, k0=k, n=1500)
    set_seed(0)
    model = build_model(sae_config(D, expansion=m / D, k=k, aux_k=8, dead_window=5))
    train_sae(model, X[:1200], X[1200:], epochs=25, patience=25, min_epochs=25,
              batch_size=128, lr=3e-3, device="cpu", seed=0, verbose=False)
    norms = model.dec.dictionary().norm(dim=0)
    assert torch.allclose(norms, torch.ones(m), atol=1e-5)
    # 편향 없는 aux 재구성도 같은 단위노름 사전을 쓴다
    h = torch.zeros(4, m)
    h[:, 0] = 1.0
    assert torch.allclose(model.dec.decode_nobias(h)[0],
                          model.dec.dictionary()[:, 0], atol=1e-6)


def _force_dead(model, dead_idx, m):
    """steps_since_active 를 직접 세팅해 특정 feature 만 dead 로 만든다 (테스트 통제용)."""
    model.steps_since_active.zero_()
    model.steps_since_active[list(dead_idx)] = model.dead_window + 1


def test_aux_k_selected_values_pass_through_relu():
    """리뷰 #8 — aux 후보는 **사전활성**으로 고르되 **값에는 ReLU**를 적용한다.

    (OpenAI TopK-SAE 공식 구현 정합: dead 마스킹된 pre-act 에 TopK 활성화(topk+ReLU).)
    음의 사전활성을 그대로 계수로 쓰면 그 dead feature 가 사전 열의 **반대 방향**으로
    잔차를 맞추게 된다 — 소생이 아니라 부호 반전 학습.

    통제: dead feature 4개의 사전활성을 전부 음수로 고정 → h_aux 가 통째로 0 →
    aux 손실 = 0.5·Σe² (잔차를 하나도 설명하지 못함). pre-ReLU 구현이었다면 음수 계수로
    e 를 부분 설명해 이 값보다 **작아진다**.
    """
    set_seed(0)
    D, m, k = 4, 8, 2
    model = BaseAE(EncoderTopK(D, m, k), DecoderLinearDict(m, D), loss="mse",
                   aux_k=4, aux_alpha=1 / 32, dead_window=5)
    with torch.no_grad():                        # dead 4개는 항상 음의 사전활성
        model.enc.net.weight[4:].zero_()
        model.enc.net.bias[4:] = -10.0
    model.eval()                                 # 카운터 갱신 없이 아래 세팅을 유지
    _force_dead(model, range(4, m), m)

    x = torch.randn(16, D)
    c = model.enc(x)
    x_hat, _lv = model.dec(c)
    e = (x - x_hat).detach()
    aux = model._aux_loss(x, x_hat, c)

    expected_zero_code = 0.5 * (e ** 2).sum(-1).mean()
    assert torch.allclose(aux, expected_zero_code, atol=1e-6), (
        f"음의 사전활성이 aux 재구성에 계수로 들어갔다: aux={float(aux):.6f} "
        f"vs ReLU 기대값={float(expected_zero_code):.6f}")

    # 대조군: dead feature 하나가 **양의** 사전활성이면 그 값이 그대로 계수로 쓰인다
    with torch.no_grad():
        model.enc.net.bias[4] = 3.0
    c = model.enc(x)
    x_hat, _lv = model.dec(c)
    e = (x - x_hat).detach()
    h_aux = torch.zeros(len(x), m)
    h_aux[:, 4] = 3.0                            # ReLU(3.0)=3.0, 나머지 dead 는 ReLU(-10)=0
    e_hat = model.dec.decode_nobias(h_aux)
    assert torch.allclose(model._aux_loss(x, x_hat, c),
                          0.5 * ((e - e_hat) ** 2).sum(-1).mean(), atol=1e-6)


def test_aux_k_requires_topk_components():
    """조밀 AE 조합에서 aux_k>0 을 주면 조용히 무시하지 않고 막는다."""
    from src.sae.models import Decoder, Encoder      # noqa: PLC0415
    with pytest.raises(ValueError):
        BaseAE(Encoder(8, 4), Decoder(4, 8), loss="mse", aux_k=16)


# ---------------------------------------------------------------- (vi) scipy 금지
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
