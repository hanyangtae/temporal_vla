# The Linear Representation Hypothesis and the Geometry of Large Language Models (Park, Choe & Veitch 2023/2024)

- 출처: arXiv 2311.03658v2 (ICML 2024, PMLR 235) · PDF: docs/Activation_steering_basic/ParkLRH_2311.03658.pdf · 섹션=§1 Introduction(무엇&왜, §2~4도 정독) · tier=must · 한 줄 역할: "activation steering이 왜 통하는가"의 이론적 토대 — "선형표현"의 세 통속적 정의(subspace/measurement/intervention)를 counterfactual 언어로 통일하고, 탐지(probe)와 개입(steering vector)이 같은 방향임을 증명.

## 문제/동기
"Linear Representation Hypothesis"(고수준 개념이 표현공간의 한 방향으로 인코딩된다)라는 말은 널리 쓰이지만 정확히 뭘 의미하는지 불명확했다. 실무에서는 최소 세 가지 다른 의미로 섞여 쓰인다: (1) subspace — 대응 단어쌍 차이가 공통 방향에 놓임(king-queen ≈ man-woman), (2) measurement — 선형 프로브로 개념값을 읽어낼 수 있음, (3) intervention — 벡터를 더하면 개념이 바뀜(steering). 이 셋이 같은 것인지, 다르다면 어떻게 연결되는지 불명확했다. 또한 코사인유사도·직교성 같은 기하 연산을 쓰려면 inner product가 필요한데, LM 학습(softmax)은 표현을 affine 변환까지만 identify하므로 "유클리드 inner product를 그냥 쓰는 것"은 정당화되지 않는다 — 어떤 inner product가 의미구조를 존중하는지가 두 번째 미해결 문제였다.

## 핵심 아이디어 (novelty)
1. counterfactual pair 언어로 subspace 표현을 두 공간에 각각 정식화: unembedding(출력 단어) 공간의 subspace, embedding(입력 문맥) 공간의 subspace.
2. 정리로 이 둘을 실전 개념에 연결: unembedding 표현은 measurement(선형 프로브)와 동치(Thm 2.2), embedding 표현은 intervention(steering vector 더하기)과 동치(Thm 2.5) — "탐지 방향"과 "개입 방향"이 별개가 아니라 각 공간에서의 같은 개념의 두 얼굴임을 증명.
3. "causally separable한 개념은 직교해야 한다"는 원칙으로 causal inner product를 정의(Def 3.1). 이 inner product 하에서 Riesz isomorphism이 unembedding 표현과 embedding 표현을 동일한 벡터로 매핑함을 증명(Thm 3.2) — 즉 프로브 방향 = steering 방향(같은 inner product 아래).
4. 실전 계산 가능한 explicit closed form 유도: causally separable 개념들의 unembedding projection이 독립이라는 가정(Assump 3.3) 하에 M = Cov(γ)^-1 (γ는 무작위 단어의 unembedding 벡터) — LLM의 unembedding 행렬만으로 causal inner product 추정 가능.
5. LLaMA-2-7B, 27개 concept으로 세 notion(subspace/measurement/intervention)의 상호 일치를 실증.

## 방법 (메커니즘)
- 개념 방향 추정(subspace): counterfactual word pair {(y0,y1)}의 unembedding 벡터 차이 γ(y1)-γ(y0)를 다수 쌍에 대해 평균 후 정규화 → 개념 방향 γ̄_W (unembedding cone 대표).
- causal inner product: ⟨γ̄,γ̄'⟩_C = γ̄ᵀ Cov(γ)^-1 γ̄' (closed form, D=Id 선택).
- intervention(embedding) 방향: λ̄_W = Cov(γ)^-1 γ̄_W (unembedding→embedding 매핑, Thm 3.2의 isomorphism 활용 — 실제 counterfactual context pair를 찾는 어려움을 우회).
- 적용형태: **additive**. λ_{W,α}(x) = λ(x) + α·λ̄_W, 즉 context의 최종 표현(다음 토큰 확률을 결정하는 표현, 사실상 마지막 층 hidden state)에 개념 방향을 스칼라 α로 스케일해 더함. 별도 residual-stream 중간층 개입은 다루지 않음(한계 절 참조). 강도 α는 0→0.4로 스윕해 로그확률의 선형 변화를 확인.

## 실험/결과 (LLaMA-2-7B)
- Fig.2: 27개 concept 중 대부분에서 counterfactual pair projection이 random pair보다 유의하게 오른쪽으로 치우침(subspace 존재 확인). 예외: thing⇒part(선형표현 없음).
- Fig.3: 추정 causal inner product 하에서 causally-separable concept 쌍들의 |⟨γ̄_W, γ̄_Z⟩_C|가 대부분 거의 0(직교) — 의미적으로 유사한 개념끼리만(예: 언어쌍끼리) block-diagonal 비직교 구조.
- Table 1: "Long live the ___" 프롬프트에 male⇒female 방향을 α=0→0.4로 더하면 top-1 예측이 king→queen으로 바뀌고, 대소문자(King, 직교 개념)는 영향받지 않음 — target만 바뀌고 off-target 보존을 확인.

## activation-steering 흐름에서의 위치 (앞뒤 논문 연결)
word2vec류 단어벡터 산술(Mikolov 2013, subspace 개념의 원조) → 2023년 전후 실전 activation steering 기법군(ActAdd 2308, CAA 2312, RepE 2310, ITI 2306 — 이 논문과 거의 동시기, 서로 인용관계 약함)이 "activation에 벡터를 더하면 행동이 바뀐다"를 경험적으로 보임 → 본 논문이 그 뒤(또는 병행)에 "왜 되는가"를 counterfactual causal 언어로 사후 정당화. Wang et al. 2023(diffusion model의 counterfactual latent concept)에서 개념 형식화에 영감을 받음. 후속 연구(Conceptors 2410 등)는 이 논문의 1차원 cone(단일벡터) 표현을 multi-dim subspace/projector로 확장 — 본 논문은 이 확장의 이론적 출발점(measurement=intervention 동치)을 제공.

## 우리 프로젝트 연결 (pathway/phase/conceptor/online검출과의 관계)
- Thm 3.2의 "measurement 표현(탐지기/프로브 방향) = intervention 표현(steering 벡터)" 동치성은 우리 핵심 난제(online phase/pathway 실패 식별)에 이론적 힌트를 준다: 실패 유형을 잘 분류하는 검출기 방향을 찾으면, 그 방향이 곧(적절한 inner product 아래) steering에 쓸 개입 방향이 될 수 있다는 정당화 — 우리 conceptor C_steer 구성(succ/fail 분류 경계 → steering 연산자)과 같은 논리 구조.
- "causally separable concept = 직교" 원칙은 VL(goal/what)과 DiT(motor/how) pathway를 "각각 따로" steer하는 우리 설계의 이론적 유비가 될 수 있다 — 단, 원문의 orthogonality는 discrete vocabulary 위 통계적 독립(무작위 단어 기준)에서 유도된 것이라, Eagle→VL-SA→DiT처럼 직렬 결합된 continuous latent에 그대로 이식하기엔 근거가 약하다(우리 project note에도 있는 "진짜 독립 아님" 주의와 정합).
- conceptor(C_success ∧ ¬C_failure)는 이 논문의 단일방향(1-dim cone) notion보다 일반화된 multi-dim subspace 버전 — 이 논문이 formalize한 subspace/measurement/intervention 삼각관계의 dim>1 확장이 실질적으로 우리가 쓰는 도구.
- 다만 이론 자체는 discrete softmax 출력(vocab)을 전제한 LM 전용이라, GR00T DiT의 continuous action-generation latent에는 **직접연결 약함** — 유비 수준에서만 차용 가능.

## 면접 포인트 (Q→A)
1. Q: "activation steering이 이론적으로 왜 통하나요?" A: "고수준 개념이 표현공간의 한 방향(subspace)으로 인코딩된다는 선형표현가설을, 이 논문이 counterfactual pair 언어로 formalize했다. 핵심 결과는 '측정(프로브)'과 '개입(steering vector 더하기)'이 서로 다른 실전 관행처럼 보이지만, causal inner product 아래에서는 같은 벡터를 가리킨다는 증명이다. 그래서 좋은 분류기 방향을 찾으면 그게 곧 steering 방향이 된다."
2. Q: "왜 inner product 선택이 중요한가요?" A: "코사인유사도·직교성은 inner product에 의존하는데, LM 학습(softmax)은 표현을 affine변환까지만 identify해서 유클리드 inner product 사용이 정당화되지 않는다. 이 논문은 'causally separable한 개념은 직교해야 한다'는 원칙으로 causal inner product를 정의하고, unembedding covariance로 M=Cov(γ)^-1이라는 closed form을 유도해 실전에서 계산 가능하게 만들었다."
3. Q(우리 프로젝트 관점): "이 이론을 GR00T VLA steering에 그대로 쓸 수 있나요?" A: "아니다. 이론은 discrete vocabulary·softmax 출력을 전제해서 continuous action latent(DiT)에는 직접 이식되지 않는다. 다만 '탐지 방향=개입 방향' 통찰은 온라인 phase/pathway 실패 검출기를 그대로 steering 방향으로 재사용할 수 있다는 이론적 근거로 차용할 수 있다."

## 한계/비판 (실패·confound 지점)
- 이론 전체가 embedding(입력 문맥)/unembedding(출력 단어) 공간에서만 전개됨. 저자 스스로 Discussion에서 "we do not address interpretability of ... the activations of intermediate layers"라고 명시 — 실제 activation steering 논문들(ActAdd, CAA, ITI 등)이 개입하는 residual-stream 중간층에는 이 이론이 직접 적용되지 않고 future work로 남김.
- causal inner product는 유일하지 않다(d degrees of freedom, D는 free parameter). 논문은 임의로 D=Id를 선택했을 뿐 "왜 이 선택이 맞는지"에 대한 원칙은 제시하지 못함.
- Assumption 3.3(causally-separable 개념의 unembedding projection이 무작위 단어 기준 독립)은 검증되지 않은 가정이며, "자연어 코퍼스에는 non-causal correlation이 존재할 수 있다"고 저자도 각주에서 인정 — 이는 우리 프로젝트의 instruction confound(SlideDishwasherRack VL AUROC 0.93이 instruction-skew artifact) 문제와 구조적으로 동일한 함정.
- 실험이 사람이 미리 대칭적으로 정의한 이항 개념(성별/언어/시제 등, single-token counterfactual pair)에 크게 의존 — "성공/실패"처럼 pair가 자연스럽지 않고 비대칭·다차원적인 개념(behavior outcome)에 그대로 일반화되는지는 불명확.
- LLaMA-2-7B 단일 모델, 27개 사람이 고른 concept에 한정된 검증 — 모델 스케일/아키텍처(비-LM, VLA 등) 일반화는 검증되지 않음.
