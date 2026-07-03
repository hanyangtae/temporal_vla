# Toy Models of Superposition (Elhage et al., Anthropic 2022)

- 출처: Transformer Circuits Thread, 2022-09-14 (arXiv 2209.10652) · PDF `docs/Activation_steering_basic/ToyModelsSuperposition_2209.10652.pdf` · 읽은 범위: Abstract, Definitions and Motivation(§1), Demonstrating Superposition~Discussion/Related Work(§2, 실질적으로 논문 전체 본문, 부록 제외) · tier: must · 한 줄 역할: "왜 activation의 개별 방향이 깨끗한 단일 개념이 아닌가"를 최초로 엄밀히 증명한 이론적 토대 논문 — steering(ActAdd/CAA 단일벡터)과 SAE(사전학습 없는 overcomplete basis 탐색) 양쪽의 존재 이유를 설명.

## 문제/동기

- 이상적으로는 뉴런 하나 = 해석 가능한 feature 하나(monosemantic)이길 바라지만, 실제로는 많은 뉴런이 서로 무관한 여러 개념에 반응하는 "polysemanticity"가 관찰됨 (LLM일수록 더 심함).
- 왜 어떤 뉴런은 깨끗이 해석되고 어떤 뉴런은 안 되는지, 왜 어떤 모델/태스크는 monosemantic 뉴런이 많고 어떤 건 거의 없는지에 대한 설명이 없었음.
- 저자들의 실용적 동기: "모델에 어떤 기능(circuit)이 존재/부재하는지 전수 열거"가 안전성 검증의 핵심 primitive인데, superposition이 이를 근본적으로 방해함.

## 핵심 아이디어 (novelty 3-5문장)

1. Superposition 가설: 신경망은 뉴런 수(차원)보다 많은 sparse feature를 "거의 직교(almost-orthogonal)"한 방향에 압축해서 저장한다 — feature 수 > 차원 수. 이는 Johnson–Lindenstrauss(고차원에서 거의 직교 벡터가 지수적으로 많이 존재)와 compressed sensing(sparse 벡터는 저차원 투영에서도 복원 가능)의 조합으로 정당화됨.
2. 이를 처음으로 "toy model"(합성 sparse 데이터 + ReLU-output 선형 오토인코더)에서 명확히 재현·정량화함 — linear model(순수 PCA류)은 superposition을 절대 하지 않지만, 출력에 ReLU 하나만 붙여도 sparsity가 커지면 급격히 superposition으로 전환됨(phase change).
3. Superposition은 이분법(0/1)이 아니라 feature마다 "fractional dimensionality"(0, 3/8, 2/5, 1/2, 2/3, 3/4, 1 등)를 갖는 연속적 스펙트럼이며, 이 값들이 정확히 uniform polytope(digon/삼각형/사각반프리즘/오각형/사면체) 기하 구조와 대응된다 — Thomson problem(구면 위 점 배치 최적화)의 변형과 동형.
4. Superposition은 저장뿐 아니라 계산(computation)에도 관여할 수 있음을 입증(ReLU 기반 absolute-value 회로가 겹친 feature 위에서도 동작) — 즉 활성화가 얽혀 있어도 모델이 그 위에서 유의미한 연산을 수행할 수 있다는 것.
5. "Superposition을 푸는 것"(feature enumeration이 가능해지는 것)은 (a) superposition 없는 모델을 학습하거나, (b) 학습 후 overcomplete basis(=sparse dictionary, 후속 SAE 연구의 이론적 전신)를 찾거나, (c) 이 둘의 hybrid로만 가능하다고 정리 — 이는 이후 Anthropic "Towards Monosemanticity"(SAE) 및 다수 mech-interp 연구의 출발점.

## 방법 (메커니즘)

- **합성 데이터**: 입력 x는 n차원, 각 차원(feature) i는 확률 S_i로 0, 아니면 [0,1](또는 abs-value 실험에서는 [-1,1])에서 균등분포. Importance I_i(더 중요한 feature는 loss 가중치 ↑)로 feature 중요도 불균형을 모사.
- **두 모델 비교**: Linear model `h=Wx, x'=Wᵀh+b` (superposition 없음, PCA와 동치) vs ReLU output model `x'=ReLU(WᵀWx+b)` (m<n로 저차원 투영 후 ReLU로 필터링). 차이는 오직 출력의 ReLU 유무.
- **Loss**: feature importance로 가중된 MSE, `L=Σ_x Σ_i I_i (x_i - x'_i)²`.
- **가시화 도구**: WᵀW(feature 간 간섭 행렬), ‖W_i‖(feature가 표현됐는지), feature dimensionality `D_i = ‖W_i‖² / Σ_j(Ŵ_i·W_j)²` (한 차원을 몇 개 feature가 나눠 쓰는지의 척도), Frobenius norm `D=m/‖W‖_F²`("feature당 차원 수").
- **privileged basis 버전**: hidden layer에도 ReLU를 추가(`h=ReLU(Wx)`)하면 뉴런 기저가 특별해져 feature가 뉴런에 정렬되기 시작 → 이때 뉴런별 stacked weight plot으로 monosemantic/polysemantic 뉴런을 직접 시각화.
- **computation-in-superposition 실험**: `y=abs(x)=ReLU(x)+ReLU(-x)`를 계산하도록 W1(encode), W2(decode)를 독립적으로 학습 → sparse 영역에서 뉴런 수보다 많은 feature에 대해 절댓값 연산이 근사적으로 성립함을 확인. 일부 뉴런은 "asymmetric superposition + inhibition" motif(비대칭 가중치로 한쪽만 간섭시키고 별도 뉴런으로 억제)를 자체 발견.

## 실험/결과 (핵심 수치)

- **Phase change**: 2-feature/1-hidden-dim 실험에서 sparsity·상대 importance에 따라 "feature 미학습 / superposition 학습 / 전용 차원 할당" 3개 solution 사이에 불연속적(1차 phase transition) 전환이 이론·실험 모두에서 확인됨.
- **Fractional dimensionality**: uniform superposition(n=400, m=30)에서 feature dimensionality가 0, 3/8(정사각 반프리즘), 2/5(오각형), 1/2(antipodal pair), 2/3(삼각형), 3/4(사면체) 등 이산적인 값에 "달라붙는(sticky)" 현상 관찰. D=1(전용 차원)과 D=1/2(antipodal pair 저장)이 특히 넓은 sparsity 구간에서 안정.
- **상관 feature**: 상관된 feature는 orthogonal한 local basis를 선호하거나(가능하면), 불가능하면 서로 인접 배치, sparsity가 낮아지면 아예 주성분으로 collapse(PCA와 동치화). 반상관 feature는 반대로 antipodal(음의 간섭)을 선호.
- **적대적 취약성**: superposition이 형성될수록 L2 adversarial example에 대한 취약성이 3배 이상 증가하며, 그 정도는 "feature당 차원 수의 역수(=차원당 feature 밀도)"와 강하게 상관됨.
- **정성적 검증**: (i) InceptionV1은 층이 깊어질수록 polysemantic 뉴런 비율 증가(고차 feature일수록 sparse해서 superposition 예측과 일치), (ii) Transformer 첫 MLP층 뉴런이 극도로 polysemantic(다국어 토큰 중의성 등 매우 sparse한 feature 다수 때문으로 설명).

## activation-steering 흐름에서의 위치

이 논문은 steering "기법"이 아니라 steering이 왜 잘 안 통하거나 얽히는지에 대한 **이론적 기초**다. 계보상 위치:

- ActAdd/CAA류(단일 방향 additive steering)가 가정하는 "concept 하나 = 방향 하나"는 이 논문의 "non-superposition, basis-aligned" 특수 케이스에서만 정확히 성립. 실제로는 sparse feature가 many-to-few로 압축되어 있어 하나의 방향을 밀면 무관한 다른 feature들도 (interference를 통해) 같이 흔들릴 수 있음 — steering이 부작용(off-target 효과)을 내는 이유의 1차 설명.
- "Three Ways Out" 중 (b) overcomplete basis를 사후에 찾는 접근이 정확히 Sparse Autoencoder(SAE, Cunningham et al. 2023 / Towards Monosemanticity)의 이론적 동기. 즉 이 논문 → SAE → (SAE feature 방향으로) steering/clamping이라는 계보의 시작점.
- Conceptors(우리가 쓰는 C_success ∧ ¬C_failure)류의 subspace/operator 기반 steering은 이 논문이 보여준 "feature가 방향들의 다차원 다발(polytope)로 얽혀 있다"는 사실과 정합적 — 단일 벡터가 아니라 다차원 연산자가 필요한 이유를 뒷받침.

## 우리 프로젝트 연결

- **conceptor(다차원 contrastive) 설계 정당화**: 논문의 핵심 결론 중 하나가 "single direction ≠ single concept"이다. 우리가 success/failure를 additive 단일벡터가 아니라 `C_steer = C_success ∧ ¬C_failure`라는 다차원 연산자로 잡는 설계는, VLA의 hidden state 역시 sparse feature superposition을 겪고 있을 가능성이 높다는 이 논문의 시사점과 직접 연결된다 — 성공/실패 concept이 다른 무관한 feature와 뒤섞여(polysemantic) 있을 수 있으므로, 단일 축이 아니라 부분공간을 타깃해야 interference를 줄일 수 있다.
- **pathway 분리(VL vs DiT) 근거 보강**: 논문의 "상관된 feature는 orthogonal한 local basis를 형성하는 경향(local almost-orthogonal basis)" 결과는, VL(goal)과 DiT(motor)처럼 기능적으로 구분되는 feature 군집이 표현공간에서 상대적으로 분리된 local basis를 이룰 수 있다는 가설과 유사한 형태 — 다만 이 논문은 correlated feature set 내부에서의 국소 직교성을 말한 것이라 pathway 분리를 직접 증명하진 않음(유비 수준).
- **confound 재해석 가능성**: length(timeout) confound, instruction-skew 같은 우리가 관찰한 아티팩트도 "성공/실패 방향"이 실제로는 다른 sparse feature(에피소드 길이, 특정 instruction)와 superposition으로 얽혀 있어서 생기는 현상일 수 있다는 프레임을 제공 — 다만 이건 직접 검증한 바 없음(직접연결 약함, 유비적 설명).
- **한계**: 이 논문 자체는 VLA/steering 실험을 다루지 않으며 완전히 별개의 도메인(합성 toy MLP)이라, 우리 프로젝트에 대한 연결은 실험적 근거가 아니라 개념적/이론적 정당화 수준이다.

## 면접 포인트 (Q→A 1-3개)

- Q: Superposition이 왜 발생하는가, 그리고 ReLU가 왜 결정적인가?
  A: feature 수가 뉴런(차원) 수보다 훨씬 많고 각 feature가 sparse(거의 항상 0)할 때, 고차원 공간에는 "거의 직교"인 방향이 지수적으로 많이 존재(Johnson–Lindenstrauss)하고 compressed sensing 논리로 sparse 벡터는 저차원 투영에서도 복원 가능하다. 문제는 이 복원에 필연적으로 약한 간섭(interference)이 남는데, 선형 모델은 이를 걸러낼 방법이 없어 항상 PCA(=차원 수만큼만 표현)로 수렴한다. ReLU 같은 비선형이 작은 간섭 노이즈를 0으로 깎아내는 필터 역할을 하기 때문에, 그 대가로 더 많은 feature를 압축 저장하는 것이 손실 관점에서 유리해진다.
- Q: 이게 SAE(Sparse Autoencoder)와 어떻게 연결되는가?
  A: 논문은 "superposition을 푸는" 3가지 길 중 하나로 "학습 후 overcomplete(sparse) basis를 찾아 feature를 풀어내는" 접근을 제시하는데, 이것이 정확히 SAE가 나중에 구현한 것이다(activation을 더 넓은 sparse dictionary로 재투영해 monosemantic-에 가까운 latent를 복원). 즉 이 논문은 SAE가 "왜 필요한가"에 대한 이론적 답이고, SAE는 그 처방을 실행한 후속 연구다.
- Q: 이 논문의 결과를 단일벡터 steering(ActAdd/CAA)에 대한 비판으로 쓸 수 있는가?
  A: 그렇다. 단일벡터 steering은 "하나의 concept = 하나의 orthogonal 방향"을 암묵적으로 가정하는데, 이 논문은 실제로는 sparse feature들이 방향을 공유(polytope 구조로 얽힘)한다는 것을 보였다. 따라서 하나의 방향을 세게 밀면 그 방향과 간섭하는 다른(무관한) feature들도 같이 움직여 부작용을 낼 수 있다 — 이는 우리가 다차원 conceptor를 쓰는 이유의 이론적 근거로 인용 가능하다.

## 한계/비판

- 순수 toy model(1~2층 선형+ReLU, 합성 sparse 데이터)이라 실제 LLM/VLA의 표현에 얼마나 일반화되는지는 저자들도 불확실하다고 명시(§Discussion "How realistic are these toy models?"). 특히 기하학적 구조(polytope)와 학습 동역학 결과는 저자 스스로 "toy model에 고유한 아티팩트일 가능성"을 인정.
- 실제 모델에는 ground-truth feature 라벨이 없어 이 논문의 예측(모노/폴리세맨틱 전환, sparsity-superposition 상관 등)을 직접 검증하기 어렵다 — 정성적 정합성(InceptionV1 깊이별 polysemanticity, 초기 MLP층 등)만 제시되고 정량적 검증은 없음.
- feature importance/sparsity curve를 실제 모델에서 추정하는 방법이 없어(논문이 스스로 Open Question으로 남김), 이 이론이 언제·얼마나 문제가 되는지 사전에 예측하기 어렵다.
- "superposition을 없애는 것이 항상 가능하다"는 낙관적 톤이지만, 경쟁력 있는(성능 손실 없는) superposition-free 모델을 만드는 것은 여전히 미해결 — MoE가 후보로 제시되나 사변적 수준.
