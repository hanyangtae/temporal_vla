# Scaling and Evaluating Sparse Autoencoders (Gao et al. 2024)

- 출처: OpenAI, Superalignment Interpretability team. arXiv preprint(비게재).
- arXiv: 2406.04093 (v1, 2024-06-06)
- PDF: `docs/Activation_steering_basic/TopKSAE_2406.04093.pdf`
- 정독 섹션: §2(방법, 집중) + Abstract·§1·§3(scaling)·§4(evaluation)·§5·§6 훑음
- tier: must
- 한줄역할: ReLU+L1 SAE(Cunningham/Bricken 계열)를 **TopK 활성화**로 대체해 학습 난이도(dead latent, λ tuning)를 없애고, GPT-4 스케일(1600만 latent)까지 SAE를 안정적으로 키우는 법을 제시한 "SAE 스케일업" 표준 레시피 논문.

## 문제·동기

- SAE는 유망하지만 학습이 어려움: reconstruction-sparsity 두 목적을 L1 계수 λ로 간접 조절해야 하고, 대형 SAE일수록 학습 중 완전히 죽는 latent(dead latent) 비율이 커짐(Templeton et al. 2024는 3400만 latent 중 1200만만 생존).
- 선행연구(Cunningham 2023, Bricken/Anthropic 2023)는 작은 LM에 작은 SAE만 검증 — "임의 크기 LM에, 극도로 넓고 sparse한 SAE를 안정적으로" 학습하는 방법론 부재.
- 목표: (1) 어떤 LM(GPT-4 포함)에도 적용 가능한 SOTA 학습 레시피, (2) n(latent 수)·sparsity·subject model 크기에 대한 scaling law, (3) reconstruction-sparsity 외에 "실제로 쓸모 있는 feature인가"를 재는 새 평가지표.

## 핵심 아이디어

- k-sparse autoencoder(Makhzani & Frey 2013)의 **TopK 활성화함수**를 LLM residual stream SAE에 도입: encoder 출력에서 top-k만 남기고 나머지를 0으로 강제 clamp.
- L1 penalty(간접·근사) 대신 L0(=k)를 **직접** 설정 → hyperparameter tuning 단순화 + 학습 자체가 더 좋은 sparsity-reconstruction frontier를 달성.
- 여기에 encoder=decoder^T 초기화 + auxiliary loss(AuxK) 두 장치를 더해 dead latent를 원천 억제 → 1600만 latent, GPT-4 40B 토큰 규모까지 학습 성공을 실증.

## 방법(TopK 활성화, sparsity 직접통제, dead latent 방지, scaling law)

- **베이스라인(ReLU+L1)**: z = ReLU(Wenc(x−bpre)+benc), x̂ = Wdec z + bpre, L = ‖x−x̂‖² + λ‖z‖1 (Bricken et al. 2023 재현).
- **TopK**: z = TopK(Wenc(x−bpre)), decoder는 동일(선형), **L = ‖x−x̂‖² 뿐**(L1 항 삭제). 이득: (a) λ tuning 불필요, (b) L0를 직접 지정해 모델 비교·반복이 단순, (c) 임의 활성함수와 결합 가능, (d) ReLU 대비 sparsity-reconstruction frontier 우위(격차는 scale 커질수록 확대), (e) 작은 활성값을 강제로 0으로 눌러 monosemanticity 상승.
- **dead latent 방지**: ① encoder를 decoder의 transpose 방향으로 초기화(읽기/쓰기 방향 정렬, 파라미터는 안 묶음 — Cunningham의 tied weight와 다름). ② **auxiliary loss(AuxK)**: 10M 토큰 동안 한 번도 안 켜진 latent를 dead로 표시 → top-kaux(보통 512)개 dead latent만으로 주 모델의 residual e=x−x̂를 재구성(ê=Wdec z)하도록 Laux=‖e−ê‖² 추가, 전체 loss=L+αLaux(α≈1/32, 계산비용 +10%). 결과: 1600만 latent 규모에서도 dead latent 7%(무대책 시 최대 90%).
- **scaling law(§3)**: (1) L(C) — 고정 compute에서 최적 MSE, 사전학습 scaling law(Kaplan/Hoffmann)와 같은 방식(Lindsey et al. 2024 계승), power law이나 최소형 모델은 off-trend. (2) L(N) — 수렴까지 학습, 최적 LR ∝ 1/√n, 수렴까지 토큰 수 ∝ n^0.6(GPT-2 small)~n^0.65(GPT-4). (3) irreducible loss 항을 넣어야 fit이 좋아짐 → activation에 SAE로 설명 안 되는 unstructured noise floor 존재 시사. (4) joint law L(n,k)(식 3, GPT-4 fit): k가 커질수록 L(N) 기울기 가팔라짐(γ<0), irreducible loss는 k가 커질수록 감소(η<0). (5) subject model이 커질수록(GPT-4 계열) 같은 MSE에 필요한 latent 수·exponent 모두 나빠짐(Ls(N)).

## 실험·결과(GPT-4 규모, 지표)

- **GPT-4 스케일 실증**: 1600만 latent SAE를 GPT-4 residual stream(전체 깊이의 5/6 지점 층)에 40B 토큰으로 학습, context 64 토큰. dead latent 7%.
- 4가지 신규 평가지표(§4, 모두 대체로 n이 클수록 개선, L0 효과는 지표별로 반대 방향):
  1. **downstream loss**: SAE reconstruction으로 residual을 치환 후 LM의 KL/cross-entropy 저하 측정. zero-ablation 기준 "fraction recovered" 대신 **compute-등가 지표** 제안(같은 loss를 내려면 얼마의 pretraining compute가 필요한가) — 1600만 latent GPT-4 SAE는 GPT-4 사전학습 compute의 **10%**에 해당하는 loss(zero-ablation fidelity로는 98.2%로 후하게 나와 이 지표의 함정을 스스로 지적).
  2. **probe loss**: 직접 큐레이션한 이진분류 61개 태스크에 대해 latent별 1-d logistic probe(Newton-Raphson) → 최고 성능 latent의 loss. TopK가 ReLU와 raw residual channel 모두를 능가.
  3. **explainability(N2G)**: wildcard n-gram 기반 저비용 설명 생성, precision/recall 측정. TopK가 같은 n에서 ReLU 대비 recall 1.5배 이상(precision은 0.9배 정도)로 F1 개선 — "가짜 해석성(illusion of interpretability)"의 recall 편향(Bolukbasi 2021, Bills 2023 언급) 문제를 precision까지 같이 봄으로써 일부 완화.
  4. **ablation sparsity**: latent 하나씩 ablation 후 logit 변화의 (L1/L2)² 로 "영향받는 유효 vocab 토큰 수" 측정. latent 효과는 절대적으로도 sparse(10~14% vs residual channel ablation 60%). k가 커질수록 sparser하다가 k≈dmodel 근처(512/768)에서 역전(solution이 dense해짐).

## activation-steering 흐름 위치

- Cunningham(2023)·Bricken/Towards Monosemanticity(2023)가 ReLU+L1 SAE를 residual stream에 처음 성공적으로 적용한 "발견" 단계라면, 이 논문은 그 레시피를 **GPT-4급으로 안정적으로 스케일업**하는 "엔지니어링 표준화" 단계 — dead latent·λ tuning이라는 실무적 병목을 TopK+AuxK로 해소.
- steering을 직접 하지 않지만 §6 Limitations에서 "finding vectors for steering behavior"를 명시적 future-work로 지목 — 이후 SAE-feature steering 연구(Golden Gate Claude, AxBench, feature-steering 논문류, GemmaScope의 JumpReLU 계승)가 이 TopK/AuxK 레시피(혹은 그 후속인 JumpReLU)를 기본 아키텍처로 채택.
- 계보: Cunningham/Bricken(ReLU+L1, 발견) → **Gao et al. TopK(이 논문, 스케일 표준화)** → GemmaScope 등 JumpReLU/Gated 계열(§5.2에서 이미 비교 대상) → SAE-feature 기반 steering 응용. NOTALL(우리 프로젝트가 참조하는 GR00T DiT per-token SAE)도 이 세대의 SAE 아키텍처 계보 위에 있을 가능성이 높음(TopK류가 현재 사실상 표준).

## 우리 프로젝트 연결

- 우리 rollout 데이터(수천 episode 단위)는 GPT-4의 40B 토큰과 규모 차이가 극단적 — §3 scaling law가 시사하듯 latent 수 n은 훨씬 작게 잡아야 하고, 절대 데이터량이 적을수록 dead latent 위험은 더 커짐(활성 신호 자체가 희소) → AuxK+init 조합이 우리 규모에서 더 필수적일 수 있음.
- TopK의 "L0 직접 설정" 성질은 phase-matched framing에 유용: phase(초반/중반/후반)마다 별도 SAE 혹은 k를 다르게 잡아 후보 feature 수를 통제된 방식으로 비교할 수 있음(λ 재튜닝 없이).
- §4.2 probe-loss 방법(latent별 1-d logistic probe, task=이진 라벨)은 그대로 succ/fail 라벨에 적용 가능 — "이 latent가 succ/fail을 분리하는가"를 SAE 전체 activation space 대신 개별 latent 단위로 스크리닝하는 틀을 재사용할 수 있음. 단 우리 confound(길이·instruction skew)를 통제하려면 probe 라벨을 phase-matched/length-controlled로 재정의해야 함(이 논문은 그런 confound 자체를 다루지 않음).
- 다만 이 논문은 여전히 offline·대량corpus·task-agnostic SAE 학습이라, 우리 핵심 난제(**online phase/failure-type 식별**)에 직접 답은 아님 — 적용한다면 "오프라인에서 phase별 SAE dictionary를 미리 학습 → 추론 중 activation을 투영해 monitor" 2단계 구조가 되고, VITA/ProgressHead류 online 신호 공급원과 결합해야 함.

## 면접 포인트(Q→A)

Q1. TopK가 L1 대비 나은 점은?
A. L1은 L0의 근사이며 모든 양의 활성값을 0쪽으로 미는 shrinkage bias가 있다(Tibshirani 1996). TopK는 top-k만 남기는 hard selection이라 (a) λ tuning 없이 L0를 정확히 지정해 모델 비교가 쉽고, (b) §5.1 refinement 실험에서 확인되듯 shrinkage가 사실상 없으며(ReLU+L1은 refinement 후 활성값이 체계적으로 커짐, TopK는 안 그럼), (c) 실측 sparsity-reconstruction frontier에서 ReLU를 능가하고 그 격차가 scale과 함께 벌어지며(Fig 2b), (d) 작은 활성값을 강제로 0으로 눌러 monosemanticity를 높인다.

Q2. dead latent 문제와 해결책은?
A. 큰 SAE일수록 학습 중 완전히 안 켜지는 latent 비율이 커짐(무대책 시 최대 90%, Templeton 2024는 3400만 중 1200만만 생존). 해결책 두 가지 — encoder를 decoder의 transpose 방향으로 초기화, 그리고 AuxK 보조 loss(10M 토큰 무활성 latent를 dead로 표시 → top-kaux=512개 dead latent만으로 주 모델 residual e=x−x̂를 재구성하는 Laux=‖e−ê‖²를 α≈1/32로 더함, 계산비용 +10%). 이 조합으로 1600만 latent 규모에서도 dead latent를 7%로 억제.

Q3. scaling law 결과가 실무적으로 왜 중요한가?
A. n, k에 대한 joint power law(식 3)가 나온다는 것은 LM 사전학습처럼 SAE도 compute-optimal 크기를 예측 가능하다는 뜻. irreducible loss 항이 fit을 개선한 것은 activation에 SAE로 설명 못하는 noise floor가 있다는 신호이고, subject model(GPT-4 계열)이 커질수록 같은 MSE에 필요한 latent 수·exponent가 모두 나빠짐(Ls(N)) — 더 큰 백본일수록 SAE 학습이 비례 이상으로 비싸진다는 뜻.

Q4. "10% of GPT-4 pretraining compute" downstream loss 지표의 의미는?
A. 기존 zero-ablation 대비 fraction-recovered 지표는 residual을 아예 0으로 지우는 게 워낙 loss를 크게 올리기 때문에 나쁜 재구성도 후하게 점수가 나오는 함정이 있다(이 논문의 16M SAE도 zero-ablation fidelity로는 98.2%로 매우 좋게 보임). 대신 "이 정도 loss 저하를 내려면 얼마만큼의 사전학습 compute가 필요한가"로 환산하면 절대적 스케일 감각이 생기고, 10%라는 수치는 여전히 상당한 정보 손실이 있음을 보여준다.

Q5. 우리 VLA latent steering 프로젝트에 적용한다면 어떤 한계에 부딪히나?
A. (1) rollout 데이터가 GPT-4의 40B 토큰과 규모가 다름 → latent 수는 훨씬 작아야 하고 dead latent 위험은 오히려 커짐. (2) VLA activation은 task/phase마다 강한 non-stationarity가 있어 "균질 corpus, context 64 토큰" 가정이 안 맞음 — phase-conditioned SAE 같은 변형이 필요할 수 있음. (3) steering은 이 논문의 §6 future-work로만 언급되고 직접 검증되지 않음 — probe-loss/ablation-sparsity 지표는 succ/fail 분리 latent의 사전 스크리닝엔 쓸 수 있지만, 우리 핵심 난제인 online phase/failure-type 식별 자체엔 답이 아니다.

## 한계·비판

- TopK는 모든 토큰에 정확히 k개 latent를 강제 — 토큰마다 필요한 feature 수가 다를 수 있는데 이를 무시(저자도 E[L0] 제약이 더 원칙적이라고 인정, §6).
- N2G 설명 방식은 계산은 싸지만 wildcard n-gram 패턴만 표현 가능 — 복잡한 semantic 패턴은 못 담고, 진짜 GPT-4 기반 precision 평가(Bills et al. 방식)는 너무 비싸 시도하지 않음.
- probe 지표는 61개 사전 큐레이션 태스크에 의존 — "우리가 자연스럽다고 미리 가정한 concept"에만 편향, 발견되지 않은 feature는 이 지표로 평가 불가.
- context length 64 토큰 제한 — GPT-4의 장거리·복잡한 behavior는 못 담을 가능성(저자도 §6에서 지적).
- 저자 스스로 인정: GPT-4에서 random activating example의 상당수가 "아직 충분히 monosemantic하지 않음".
- irreducible loss 항의 원인(진짜 unstructured noise인지 SAE가 아직 못 담는 구조인지)이 불명확 — scaling law 해석에 모호함 남음.
- steering을 직접 실행·검증하지 않음 — "steering vector 탐색"은 future work로만 언급, causal steering 실험/결과 없음.
- GPT-4 자체가 비공개 모델이라 재현성 제한(체크포인트·데이터 비공개, 코드/시각화 도구만 공개) — 완전 재현은 GPT-2 small급에서만 가능.
