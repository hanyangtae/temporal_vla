"""top-k SAE 모델 코어 (동료 레포에서 lift).

출처: task_classification@88543a2 `phase/models/autoencoder.py` + `phase/models/factory.py`
      (https://github.com/robots-oh/task_classification)
이식 근거·갭 분석: docs/steering/29_sae_port_review.md §2.1, §5.A
이식 계획:        docs/steering/30_sae_g1_port_handout.md §2.1, §4 Phase A

[원본 대비 변경]
- hydra/omegaconf 의존 제거 — `build_model`은 평범한 dict(Mapping)만 받는다.
  (레포 규약: 설정은 우리 방식(dict/json), 경로는 scripts/path_setup.py 단일 소스.)
- 사용하지 않는 컴포넌트 제외: `EncoderVariational`·`VariationalAE`(원본에서 DEPRECATED),
  `DecoderSeedConditioned`(ctx 소유자 미정), `PhaseClassifier`(지도 상한 트랙 — G1 무관).
- 원 소스의 설계 근거 주석(docstring)은 그대로 보존한다.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, get_args

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)

# 재구성 손실 축. LossName은 정적 힌트, LOSSES는 그 힌트에서 파생된 런타임 허용 집합
# — 한 곳(LossName)만 고치면 애노테이션·검증·argparse choices가 함께 따라온다.
LossName = Literal["mse", "log_likelihood"]
LOSSES = get_args(LossName)                 # ("mse", "log_likelihood")


# ================================================================ 모듈
class Encoder(nn.Module):
    """추론망 q(c_t | x_t) — 시점별 독립 MLP. 시간 맥락 없음(인과적).

    시간 결합은 오직 sticky Markov prior에서만 온다 → '지속성의 출처'가 prior임을
    분리한다. (인코더가 미래를 보면 온라인 인과성을 위배하므로 시퀀스 인코더는 쓰지 않는다.)

    [우리 용도] 조밀 AE 기준선 + `EncoderTopK(hidden=...)`의 MLP 백본.
    """

    def __init__(self, input_dim, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, d)
        self.d = d

    def forward(self, x):
        h = self.net(x)
        o = self.head(h)
        return o


class Decoder(nn.Module):
    """p(obs_t | c_t) — 대각 가우시안, 차원별 로그분산.

    [반환] (x̂, logvar). logvar를 학습하면 상대오차, 0에 얼리면 절대오차(=MSE)가 된다.
    그 선택은 이 모듈을 만드는 쪽에서 requires_grad로 정한다 (BaseAE 계약 참고).
    """

    def __init__(self, d, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        self.logvar = nn.Parameter(torch.zeros(out_dim))

    def forward(self, c):
        inp = c
        return self.net(inp), self.logvar.clamp(-8, 4)


class EncoderTopK(nn.Module):
    """희소 코드 h = top-k(ReLU(W_e x)) ∈ R^m — 과완비 사전 + 구조적 희소성.

    Encoder의 조밀한 병목(d=16)과 달리 m > input_dim으로 차원을 늘리는 대신 매 시점
    k개만 켠다. 희소성이 top-k 마스크라는 **구조**로 강제되므로 손실에 L1 페널티가
    필요 없다 — "학습 손실은 재구성뿐"이라는 AE와의 비교 조건이 유지된다.

    [인자] hidden=None이면 nn.Linear(표준 SAE의 선형 사전), 값이 있으면 Encoder를
    백본으로 써서 AE와 capacity를 맞춘 변형이 된다 (디코더는 어느 쪽이든 선형 유지
    → '희소 vs 조밀' 축만 분리된다).

    [반환] [..., m]

    [우리 재설계 메모] 동료 세팅은 input=PCA64, m=128 → 원본 1536-d 기준으로는 사실상
    undercomplete. G1은 원본 D=1536을 그대로 넣고 m=4~8×D로 키운다
    (docs/steering/30_sae_g1_port_handout.md §2.2-2, §4 Phase C).
    """

    def __init__(self, input_dim, m, k, hidden=None):
        super().__init__()
        self.net = nn.Linear(input_dim, m) if hidden is None else Encoder(input_dim, m, hidden)
        self.m, self.k = m, k
        self.d = m                      # Encoder와 같은 이름으로 잠재 폭을 노출

    def forward(self, x):
        """[구현 메모] topk의 k번째 값을 임계값으로 삼아 where로 마스킹한다 —
        gather/scatter보다 짧고, 동점이 있으면 k개보다 많이 살아남을 수 있지만
        연속값이라 사실상 발생하지 않는다. 마스킹은 살아남은 원소의 값을 보존하므로
        (0/1 게이트가 아니라) 그 원소들로만 gradient가 흐른다.
        k >= m이면 top-k가 무의미하므로 ReLU 결과를 그대로 반환한다.
        """
        h = F.relu(self.net(x))
        if self.k < self.m:
            thr = torch.topk(h, self.k, dim=-1).values[..., -1:]     # k번째 값
            h = torch.where(h >= thr, h, torch.zeros_like(h))
        return h

    @staticmethod
    def density(h):
        """평균 활성 비율 (진단용). 이론상 k/m 근처여야 하고, 이보다 낮으면
        ReLU가 k개를 못 채운 것 = dead feature 신호.
        """
        return (h > 0).float().mean().item()


class DecoderLinearDict(nn.Module):
    """단위노름 사전 선형 디코더 (top-k SAE 표준) — Decoder와 같은 (x̂, logvar) 계약.

    [왜 단위노름인가] 사전 열의 크기를 자유롭게 두면 모델이 열을 키우고 계수 h를
    줄여도 같은 출력을 낸다 — 희소도(활성 개수)는 그대로인데 계수 스케일만 수축해
    코드의 크기 정보가 의미를 잃는다. 매 forward마다 열을 노름으로 나눠 이 자유도를
    없앤다(정규화가 파라미터가 아니라 forward에서 일어나므로 gradient도 정규화된
    형태를 통해 흐른다). clamp_min(1e-8)은 죽은 열에서 0-나눗셈을 막는다.

    비선형층이 없다는 점이 Decoder와의 유일한 구조 차이다 (해석성 전제).
    """

    def __init__(self, m, out_dim):
        super().__init__()
        self.lin = nn.Linear(m, out_dim)
        self.logvar = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h):
        w = self.lin.weight                                          # [out_dim, m]
        wn = w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)
        return F.linear(h, wn, self.lin.bias), self.logvar.clamp(-8, 4)

    @torch.no_grad()
    def dictionary(self):
        """단위노름으로 정규화된 사전 [out_dim, m] — forward가 실제로 쓰는 행렬.

        (우리 추가: 파라미터 `lin.weight`는 정규화 전 값이라 사전 열 노름 검사·
        feature 방향 분석은 이 메서드를 통해야 한다.)
        """
        w = self.lin.weight
        return w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)


class BaseAE(nn.Module):
    """표준 AutoEncoder. 잠재가 결정적이고 prior(KL)도 이산 구조도 없다.

    [역할] x_t → c_t → x̂_t 의 단순 인코더/디코더. 학습 목적함수는 재구성 하나뿐이며,
    이산 상태 z는 학습이 끝난 뒤 c를 클러스터링해서 만든다.

    [조립] 인코더/디코더를 **주입**받는다 — 결정적이냐 변분이냐, 조밀이냐 희소냐는
    이 클래스가 모른다. Base가 소유하는 것은 목적함수(loss)뿐이다. 그래서 AE와 SAE는
    같은 BaseAE이고 주입한 컴포넌트만 다르다.

    [디코더에 요구하는 계약] dec(c) → (x̂, logvar).
        logvar는 x와 브로드캐스트 가능한 차원별 로그분산 파라미터(`.logvar` 속성).
        loss='mse'면 __init__이 그 파라미터를 0에 동결한다 → σ=1 고정 가우시안이 되어
        loss() 본문이 분기 없이 그대로 절대오차(MSE와 argmin 동일)가 된다.
    """

    def __init__(self, encoder, decoder, loss: LossName = "log_likelihood"):
        """인코더/디코더를 받아 조립한다.

        loss='mse'면 주입받은 디코더의 logvar를 0에 동결한다(아래 참고).
        """
        # 인자 검증 — 부작용(logvar 동결) 이전에 걸러 반쯤 만들어진 객체를 남기지 않는다
        if loss not in LOSSES:
            raise ValueError(f"loss는 {LOSSES} 중 하나여야 합니다: {loss!r}")

        # 객체 생성
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.loss_name = loss           # 메서드 loss()와 이름이 겹치지 않도록 _name
        if self.loss_name == "mse":
            # σ=1 고정 → dec.forward가 logvar.clamp(-8,4)=0 을 반환하므로
            # loss()가 그대로 0.5·Σ(x-x̂)² + const 가 된다 (본문 분기 불필요).
            nn.init.zeros_(self.dec.logvar)
            self.dec.logvar.requires_grad_(False)

    def loss(self, x):
        """재구성 음의 로그가능도(스텝 평균). 페널티/KL 항 없음 = 순수 AE 목적함수.

        [역할] 학습 루프가 **최소화**하는 스칼라를 만든다 — `model.loss(x).backward()`.

        [인자]
            x  [N, D]  step 행렬. 데이터가 step 단위라 배치 축이 곧 스텝 축이다.
                       (마지막 축만 feature로 보고 나머지를 평균하므로 leading 축이
                        여러 개여도 동작한다.)

        [계산 흐름]
            1) c = enc(x)                                 잠재 [..., d]
            2) x̂, logvar = dec(c)
            3) nll = 0.5·[(x-x̂)²·exp(-lv) + lv + log2π]   차원별 가우시안 NLL
            4) feature 축 합 → 나머지 축 평균 = 스텝 평균

        [반환] 0-dim 스칼라. 작을수록 재구성이 좋다.

        logvar≡0(loss='mse')이면 nll = 0.5·Σ(x-x̂)² + const → MSE와 argmin이 같다.
        """
        c = self.enc(x)                                   # [..., d] 잠재
        x_hat, logvar = self.dec(c)                       # 타깃은 입력 자기재구성

        se = (x - x_hat) ** 2
        nll = 0.5 * (se * torch.exp(-logvar) + logvar + LOG2PI)
        return nll.sum(-1).mean()

    @torch.no_grad()
    def latent(self, x):
        """사후 클러스터링·probe 에 넣을 표현을 뽑는다 (여러 모델의 공통 인터페이스).

        [반환] [..., d] 결정적 코드. SAE의 희소 코드 h와 같은 자리다.
        이 메서드가 모델들을 fit_clusters/assign에서 동일하게 다룰 수 있게 하는
        유일한 접점 — 이 계약(no_grad + 입력과 같은 leading shape)을 유지해야 한다.
        """
        return self.enc(x)          # [..., d] 결정적 코드

    @torch.no_grad()
    def reconstruct(self, x):
        """x̂ 만 돌려준다 (우리 추가 — 재구성 R²·잔차화 진단용)."""
        return self.dec(self.enc(x))[0]


# ================================================================ 팩토리
# 출처: task_classification@88543a2 phase/models/factory.py:24-95
#
# 설정 → 모델 조립. AE와 SAE는 '주입한 컴포넌트'만 다른 같은 BaseAE다.
#     ae   encoder: mlp   + decoder: mlp          → 조밀 병목 AutoEncoder
#     sae  encoder: topk  + decoder: linear_dict  → top-k Sparse AutoEncoder
#
# 레지스트리의 각 빌더는 `(input_dim, latent_dim, **컴포넌트 고유 파라미터)`를 받는다.
# 공유 차원을 팩토리가 양쪽에 꽂기 때문에 인코더 출력차원과 디코더 입력차원이 어긋날 수 없다.
# 디코더의 출력차원은 항상 input_dim이다 (타깃이 x 자기재구성이므로).
ENCODERS = {
    "mlp": lambda input_dim, latent_dim, hidden=256:
        Encoder(input_dim, latent_dim, hidden),
    "topk": lambda input_dim, latent_dim, k, hidden=None:
        EncoderTopK(input_dim, latent_dim, k, hidden),
}

DECODERS = {
    "mlp": lambda input_dim, latent_dim, hidden=256:
        Decoder(latent_dim, input_dim, hidden),
    "linear_dict": lambda input_dim, latent_dim:
        DecoderLinearDict(latent_dim, input_dim),
}

MODELS = {"ae": BaseAE}

REQUIRED = ("input_dim", "latent_dim", "encoder", "decoder")


def _plain(obj):
    """Mapping → 평범한 dict. (원본은 omegaconf DictConfig 대응이었고, 여기서는
    dict 만 쓰지만 중첩 Mapping 정규화는 그대로 유지한다.)"""
    return {str(k): (_plain(v) if isinstance(v, Mapping) else v)
            for k, v in obj.items()}


def _build(registry, spec, field, input_dim, latent_dim):
    """spec {'type': 이름, ...파라미터} → 모듈. type은 소비하고 나머지는 빌더로 넘긴다."""
    if not isinstance(spec, Mapping):
        raise ValueError(f"{field}는 type을 가진 매핑이어야 합니다: {spec!r}")
    spec = _plain(spec)
    name = spec.pop("type", None)
    if name not in registry:
        raise ValueError(f"{field}.type은 {tuple(registry)} 중 하나여야 합니다: {name!r}")
    try:
        return registry[name](input_dim=input_dim, latent_dim=latent_dim, **spec)
    except TypeError as e:
        # 오타·누락 파라미터는 빌더 호출에서 잡힌다. 어느 컴포넌트인지만 얹어준다
        # (빌더가 람다라 원본 메시지만으로는 위치를 알 수 없다).
        raise TypeError(f"{field}(type={name!r}) 조립 실패: {e}") from e


def build_model(cfg, input_dim=None):
    """model 설정 블록(dict) → 모델.

    [인자]
        cfg        model 블록 dict. REQUIRED 키가 있어야 한다.
        input_dim  주면 cfg의 input_dim을 덮어쓴다 — 실제 데이터 차원이 설정값과
                   다를 때 설정을 고치지 않고 맞추기 위한 통로.

    예:
        cfg = {"kind": "ae", "input_dim": 1536, "latent_dim": 6144,
               "encoder": {"type": "topk", "k": 32},
               "decoder": {"type": "linear_dict"}, "loss": "mse"}
        model = build_model(cfg)
    """
    if not isinstance(cfg, Mapping):
        raise ValueError(f"model 설정은 매핑이어야 합니다: {cfg!r}")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"model 설정에 {missing} 키가 없습니다")

    D = cfg["input_dim"] if input_dim is None else input_dim
    d = cfg["latent_dim"]

    enc = _build(ENCODERS, cfg["encoder"], "encoder", D, d)
    dec = _build(DECODERS, cfg["decoder"], "decoder", D, d)

    kind = cfg.get("kind", "ae")
    if kind not in MODELS:
        raise ValueError(f"model.kind는 {tuple(MODELS)} 중 하나여야 합니다: {kind!r}")
    # loss는 모델 클래스가 기본값과 검증을 갖고 있으므로 있을 때만 넘긴다
    kwargs = {k: cfg[k] for k in ("loss",) if k in cfg}
    return MODELS[kind](enc, dec, **kwargs)


def sae_config(input_dim, expansion=4, k=32, loss="mse"):
    """G1 표준 top-k SAE 설정 dict 생성 헬퍼 (overcomplete: m = expansion × D).

    동료 하이퍼(input=PCA64·m=128·k=16)는 원본 1536-d 기준으로 과완비가 아니라
    그대로 쓰지 않는다 — docs/steering/30_sae_g1_port_handout.md §2.1, §4 Phase C.
    """
    return {"kind": "ae", "input_dim": int(input_dim),
            "latent_dim": int(round(expansion * input_dim)),
            "encoder": {"type": "topk", "k": int(k)},
            "decoder": {"type": "linear_dict"},
            "loss": loss}
