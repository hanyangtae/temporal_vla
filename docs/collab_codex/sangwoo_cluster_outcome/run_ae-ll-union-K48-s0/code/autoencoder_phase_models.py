import math
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


class EncoderVariational(Encoder):
    """변분 추론망 q(c_t | x_t) = N(mu, exp(logvar)) — 결정적 Encoder의 확률적 짝.

    구조 차이는 head 하나뿐이다: 출력이 2d(mu, logvar 2-head)이고 forward가 튜플을
    돌려준다. 그래서 부모를 출력차원 2d로 조립하고 forward만 오버라이드한다.

    [반환] (mu, logvar) 각 [..., d]. logvar는 exp 폭주를 막으려고 [-8, 4]로 clamp.
    """

    def __init__(self, input_dim, d, hidden=256):
        super().__init__(input_dim, 2 * d, hidden)   # head: hidden → 2d
        self.d = d                                   # 부모가 넣은 2d를 잠재 차원으로 정정

    def forward(self, x):
        mu, logvar = super().forward(x).chunk(2, -1)
        return mu, logvar.clamp(-8, 4)


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

class DecoderSeedConditioned(Decoder):
    """p(obs_t | c_t, cell) — cell(seed) 정체성을 ctx로 함께 받는 디코더.

    디코더가 cell 정체성을 공짜로 받으면 c는 cell을 인코딩할 이유가 없어지고,
    잠재에는 cell 잔차가 아닌 구조만 남는다 (nuisance 통제).
    stage0 진단: hidden state의 cell 선형 probe 정확도 = 1.000 → 통제 없이는 새어든다.

    구조 차이는 입력이 [c, ctx] concat이라는 것뿐 → 부모를 입력차원 d+ctx_dim으로
    조립하고 forward만 오버라이드한다.

    [주의] forward가 ctx를 **요구**하므로 Decoder의 드롭인 대체재가 아니다.
    ctx(=cell 임베딩)를 누가 소유하고 어떻게 넘길지는 아직 정하지 않았다.
    """

    def __init__(self, d, out_dim, hidden=256, ctx_dim=8):
        super().__init__(d + ctx_dim, out_dim, hidden)   # net 입력: d + ctx_dim
        self.ctx_dim = ctx_dim

    def forward(self, c, ctx):
        inp = torch.cat([c, ctx], -1)
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


class BaseAE(nn.Module):
    """표준 AutoEncoder. 잠재가 결정적이고 prior(KL)도 이산 구조도 없다.

    [역할] x_t → c_t → x̂_t 의 단순 인코더/디코더. 학습 목적함수는 재구성 하나뿐이며,
    이산 상태 z는 학습이 끝난 뒤 c를 클러스터링해서 만든다.

    [조립] 인코더/디코더를 **주입**받는다 — 결정적이냐 변분이냐, 조밀이냐 희소냐는
    이 클래스가 모른다. Base가 소유하는 것은 목적함수(loss)뿐이다. 그래서 AE와 SAE는
    같은 BaseAE이고 주입한 컴포넌트만 다르며(factory.py 참고), VAE는 loss에 KL을
    더한 서브클래스로만 갈라지면 된다.

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
        VAE는 여기에 KL을 더한 loss로 오버라이드하면 되고, 두 항이 모두 최소화
        방향이라 합성이 그냥 덧셈이다 (부호를 뒤집는 지점이 어디에도 없다).

        [인자]
            x  [N, D]  step 행렬. 데이터가 step 단위라 배치 축이 곧 스텝 축이다.
                       (마지막 축만 feature로 보고 나머지를 평균하므로 leading 축이
                        여러 개여도 동작하지만, 이 파이프라인에서는 항상 [N, D]다.)

        [계산 흐름]
            1) c = enc(x)                                 잠재 [..., d]
            2) x̂, logvar = dec(c)
            3) nll = 0.5·[(x-x̂)²·exp(-lv) + lv + log2π]   차원별 가우시안 NLL
                     = -log p(x | x̂, exp(lv)) — 로그밀도에 -1을 곱한 값
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
        """사후 클러스터링에 넣을 표현을 뽑는다 (여러 모델의 공통 인터페이스).

        [반환] [..., d] 결정적 코드. SAE의 희소 코드 h와 같은 자리다.
        이 메서드가 모델들을 fit_clusters/assign에서 동일하게 다룰 수 있게 하는
        유일한 접점 — 이 계약(no_grad + 입력과 같은 leading shape)을 유지해야 한다.
        """
        return self.enc(x)          # [..., d] 결정적 코드


class VariationalAE(BaseAE):
    """**DEPRECATED — 실험에 쓰지 않는다.**

    이론적 근거 부족과 실험 성능 부족으로 실험 축에서 제외됐다. conf/model에 항목이 없고
    sweep 축에도 들어가지 않는다. 구조적 완결성(변분 인코더를 주입했을 때의 목적함수)을
    남겨두기 위해 정의만 유지하며, 여기에 더 이상 자원을 쓰지 않는다.

    [요구 사항] encoder가 (mu, logvar) 튜플을 돌려줘야 한다 → EncoderVariational.
    """

    def __init__(self, encoder, decoder, loss: LossName = "log_likelihood", beta=1.0):
        super().__init__(encoder, decoder, loss)
        self.beta = beta

    def loss(self, x):
        """-ELBO = NLL + β·KL. 두 항 모두 최소화 방향이라 합성이 그냥 덧셈이다."""
        mu, logvar = self.enc(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # 재파라미터화
        x_hat, out_logvar = self.dec(z)

        se = (x - x_hat) ** 2
        nll = 0.5 * (se * torch.exp(-out_logvar) + out_logvar + LOG2PI)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())         # vs N(0, I)
        return nll.sum(-1).mean() + self.beta * kl.sum(-1).mean()

    @torch.no_grad()
    def latent(self, x):
        """샘플이 아니라 mu — 클러스터링 대상은 결정적 코드여야 한다."""
        return self.enc(x)[0]




