# Steering Large Language Models using Conceptors: Improving Addition-Based Activation Engineering (Postmus & Abreu 2024)

- 출처: arXiv 2410.16314v4 (NeurIPS 2024 MINT Workshop) · PDF: docs/Activation_steering_basic/Conceptors_2410.16314.pdf · 섹션=§3 Conceptors as Steering Matrices(방법 중심, 전체 정독) · tier=must★★ · 한 줄 역할: 우리 프로젝트 headline method(C_steer = C_success ∧ ¬C_failure, h' = h·Mᵀ)의 **직접 수학적 근거** — activation steering을 "벡터 덧셈(translation)"에서 "conceptor 행렬 곱(soft projection)"으로 바꾸고, Boolean 대수(AND/OR/NOT)로 스티어링 목표를 조합하는 최초의 정식화. COAST의 직계 선행연구(COAST = 이 논문 수식을 VLA rollout에 적용).

## 문제/동기 (additive steering의 한계)
기존 activation steering(ActAdd, CAA, function vectors 등)은 대조군 activation을 **평균낸 단일 벡터** h̄를 h' = β·h̄ + h 형태로 더하는 translation이다. 문제는: (1) 평균은 activation cloud를 점 하나로 요약해 성분 간 **상관·분산 구조를 버림** — 복잡한 패턴을 표현 못함. (2) 대조 프롬프트를 구성하기 어려운 개념도 있음. (3) additive steering 성능이 불안정(Turner et al. 2024 인용). Mean-centering(이방성 편향 제거)이 일부 개선하지만 여전히 점 표현의 한계 안에 있음.

## 핵심 아이디어
Jaeger(2014)의 **conceptor**(RNN 제어용 신경계산 구조)를 LLM steering에 최초 도입. conceptor 행렬 C는 활성화 벡터 집합의 주축·분산을 인코딩하는 양의 준정부호 행렬로, 고차원 **타원체(ellipsoid)**로 activation 패턴의 "상태공간 영역"을 표현한다(점이 아니라 영역). 스티어링을 벡터 덧셈이 아니라 **행렬-벡터 곱(soft projection)**으로 재정의하고, 여러 스티어링 목표를 conceptor의 **Boolean 대수(AND/OR/NOT)**로 조합 가능하게 만든다.

## 방법 (메커니즘)
**Conceptor 정의(식3)**: 재구성오차+정규화를 최소화하는 최적화의 closed-form 해:
  min_C ‖X − XC‖²_F + α⁻²‖C‖²_F  →  C(R,α) = R(R + α⁻²I)⁻¹,  R = XᵀX/n
- X = 활성화 벡터를 행으로 쌓은 행렬, n = 샘플 수, R = (비중심화) 상관행렬, α = **aperture**(정규화 강도).
- C의 고유값 μ_i과 R의 고유값 λ_i 관계: μ_i = λ_i/(λ_i+α⁻²) (0<λ_i<1일 때). α→∞이면 μ_i→1(C→I, 전부 통과); α→0이면 μ_i→0(C→영행렬, 전부 억제). λ_i=1인 성분은 α와 무관하게 μ_i=1(항상 통과), λ_i=0이면 항상 0.
- **soft projection**: 하드 프로젝터는 고유값이 0 또는 1뿐이지만, C는 [0,1] 사이 "부드러운" 고유값을 가져 activation을 주축 방향으로 **연속적으로 스케일**한다 — additive의 "고정 이동"과 질적으로 다름(위치 의존적 선형 사상).
- **적용(식4)**: h'_ℓ = β_c · C^f_ℓ · h_ℓ (덧셈이 아니라 곱셈). mean-centered 버전: h'_ℓ = β_c·C^mc·(h_ℓ−μ_train) + μ_train.
- **Boolean 연산(§3.1, Jaeger 2017 기반)**:
  - OR(공분산 합치기, 식5-6): C1∨C2 = (R1+R2)(R1+R2+α⁻²I)⁻¹ = [I + (C1(I−C1)⁻¹+C2(I−C2)⁻¹)⁻¹]⁻¹
  - NOT(역상관행렬, 식7-8): ¬C = R⁻¹(R⁻¹+α⁻²I)⁻¹ = **I − C** (aperture 무관, 정확한 항등식 — 계산 매우 쉬움)
  - AND(드모르간 a∧b=¬(a∨b), 식9-10): C1∧C2 = (R1⁻¹+R2⁻¹)⁻¹[(R1⁻¹+R2⁻¹)⁻¹+α⁻²I]⁻¹ = **(C1⁻¹ + C2⁻¹ − I)⁻¹**
- **계산복잡도(§3.2)**: fit은 O(n³)(오프라인, 1회, 상환됨), 저장은 O(n²)(fp32 4096차원=67MB, weight matrix와 동급). 추론 시 conceptor를 W_x^C = W_x·C처럼 후속 weight에 **fuse**하면 추가 비용 0 — 단, 이는 C가 고정된 채 항상 켜져 있을 때만 성립하고, 켜고 끄거나 여러 C를 스위칭하면 계산그래프 변경 오버헤드가 남는다고 저자도 명시.

## 실험·결과 (수치)
- 모델: GPT-J(6B), GPT-NeoX(20B). Todd et al.(2023) function-vector 6개 태스크(antonyms/present-past/english-french/singular-plural/country-capital/capitalize), ICL 프롬프트 마지막 토큰 residual stream 사용, 5회 반복 평균, 전 레이어·하이퍼파라미터 그리드서치.
- Table1(GPT-J, best-of-all-layers/hparams, top-1 accuracy): Addition 20.54/93.16/32.04/18.88/69.66% → Addition+MC 31.20/95.00/63.90/34.32/83.32% → **Conceptor** 52.14/96.68/81.62/59.02/91.56% → Conceptor+MC 52.82/96.26/85.32/61.32/91.88% (순서: antonyms/capitalize/country-capital/english-french/present-past). **Conceptor(MC 없이도)가 Addition+MC를 전 태스크에서 능가.**
- 최적 레이어: GPT-J 9-16층, GPT-NeoX 10-30층. 최적 aperture는 대체로 **α=0.1이 전 실험의 10% 이내**(GPT-J 최적 α=0.05·β_c=2.0, 일부 태스크는 다른 값).
- 합성함수 실험(§4.2, GPT-J): 2개 conceptor를 AND(C1∧C2)로 합친 것이 스티어링 벡터 산술평균((h̄1+h̄2)/2)보다 전 태스크에서 우세, english-french & antonyms 조합에서는 합성 태스크 직접 fit한 conceptor 베이스라인마저 능가(진정한 조합적 일반화 신호).

## activation-steering 흐름에서의 위치 (additive→projective 전환점)
ActAdd(2308)·CAA(2312)·mean-centering(2312)·function vectors(2310)까지는 모두 "평균 벡터를 더하는" translation 계열. 이 논문(2410, NeurIPS 2024 워크숍)이 처음으로 **활성화의 2차 모멘트(공분산) 구조 전체**를 conceptor 행렬로 인코딩해 **soft projection**으로 개입하는 방식을 도입 — Jaeger의 RNN 제어 conceptor(2014/2017)와 conceptor 기반 BERT/GPT debiasing(Yifei et al. 2023)을 LLM steering에 처음 연결한 지점. 동시에 Boolean 대수로 "스티어링 목표 조합"을 벡터평균이 아니라 집합연산으로 재정의 — 이후 **COAST**가 이 식(특히 AND/NOT)을 VLA rollout activation에 그대로 이식(단, 전 timestep을 하나의 R로 pool)하는 직계 후속.

## 우리 프로젝트 연결
- **C_steer = C_success ∧ ¬C_failure의 유도**는 이 논문 식을 그대로 조합한 것: ① ¬C_failure = I − C_failure (식8, aperture 무관 정확식 — C_failure만 fit하면 공짜). ② C_success ∧ ¬C_failure = (C_success⁻¹ + (¬C_failure)⁻¹ − I)⁻¹ (식10에 C2=¬C_failure 대입). 의미상 "성공 패턴에 속하면서 동시에 실패 패턴 궤적에서는 벗어난" 활성화 영역을 골라내는 연산자.
- 우리 적용식 h' = h·Mᵀ, M=(1−β)I+β·C_steer는 논문 식4(h'=β_c·C·h, identity blending 없음)의 **소프트 확장**이다 — β<1이면 필터링된 방향에서도 원신호 일부를 남겨 개입을 덜 파괴적으로 만든다. LM의 단일토큰 생성과 달리 VLA의 연속 action 생성/control 안정성을 고려한 선택으로 보임(논문엔 이런 identity-blend 항 없음).
- **COAST와의 관계**: COAST = 이 논문의 AND/NOT Boolean 대수를 VLA rollout activation에 그대로 적용한 것 — 클래스(succ/fail)별 R=E[hhᵀ]를 **전 timestep을 하나로 pool**해서 fit. 우리가 COAST를 비판하는 지점(길이confound·phase 미분리)은 이 논문의 conceptor 수학 자체의 결함이 아니라 **COAST의 집계(aggregation) 선택**의 결함이다 — 이 논문의 정식화는 X(활성화 벡터 집합)에 아무 제약이 없어, phase별로 별도 R_phase를 fit해도 동일한 closed-form/Boolean 연산이 그대로 성립한다. 우리의 phase-matched 확장 = "전 timestep 대신 phase-bin별 X_phase로 conceptor를 따로 fit" — 수학은 이 논문 그대로, 집계 축만 바꾸는 것.
- aperture α: 논문은 GPT-J/GPT-NeoX·6개 언어함수 태스크에서 α=0.1 근방이 거의 항상 최적이라 보고하지만, 저자 스스로 "새 모델·태스크엔 재검증 필요"라 명시. GR00T DiT activation(고차원·denoising-step 간 강한 상관)은 이 경험적 사전과 다를 가능성이 커 우리 실험에서 별도 스윕 필요.

## 면접 포인트 (Q→A)
1. Q: "conceptor가 단일 스티어링 벡터보다 우월한 이유는?" A: "단일벡터는 activation cloud를 평균(점)으로만 요약해 상관·분산 구조를 버리지만, conceptor C=R(R+α⁻²I)⁻¹는 공분산행렬 R 전체를 인코딩해 타원체로 패턴의 주축·퍼짐까지 표현한다. 적용도 h'=h+βv̄(고정 이동)가 아니라 h'=β_c·C·h(위치 의존적 soft projection)라, 이미 패턴 안에 있는 성분은 그대로 통과시키고 밖의 성분만 눌러준다. 실험적으로도 GPT-J/GPT-NeoX 6개 태스크 전부에서 conceptor가 mean-centered additive보다 우월(예: country-capital 63.90%→81.62%)."
2. Q: "aperture α의 의미는?" A: "C의 고유값 μ_i=λ_i/(λ_i+α⁻²)를 조절하는 정규화 강도다. α→∞면 C→I(개입 없음), α→0이면 C→영행렬(전부 억제) — PCA에서 몇 개 주성분을 남길지와 비슷한 편향-분산 트레이드오프 다이얼이다. 논문에서는 여러 태스크·모델에서 α=0.1 근방이 강건하게 최적이었지만, 새 도메인엔 재검증이 필요하다고 저자 스스로 밝힌다."
3. Q(우리 프로젝트): "C_success∧¬C_failure는 이 논문의 어떤 연산에서 나온 건가?" A: "이 논문 §3.1의 Boolean conceptor 대수 그대로다. ¬C_failure=I−C_failure(식8, aperture 무관 정확식)로 실패 방향의 여집합을 구하고, AND는 C1∧C2=(C1⁻¹+C2⁻¹−I)⁻¹(식10)로 두 conceptor의 교집합을 구한다. COAST가 이 식을 VLA rollout에 처음 적용했는데 전 timestep을 하나의 R로 pool했고, 우리는 phase-bin별로 별도 R을 fit해 COAST의 길이confound를 통제하는 확장이다 — 수학은 이 논문 그대로다."

## 한계·비판
- 데이터 요구량: conceptor는 정확한 R(공분산) 추정에 평균보다 더 많은 샘플이 필요하다고 저자 스스로 인정 — 우리가 phase별로 더 잘게 쪼개면 phase당 rollout 수가 줄어드는 문제와 직결(성공/실패 표본 부족).
- 계산·운영비용: O(n³) fit·O(n²) 저장(4096차원=67MB)로 additive보다 훨씬 비쌈. weight-fusion으로 추론비용을 지울 수 있다지만 이는 **하나의 고정 C를 상시 적용**할 때만 유효 — 우리처럼 phase/pathway 조건부로 여러 C를 온라인 스위칭해야 하면 fusion 이득을 못 받고 계산그래프 변경 오버헤드가 다시 발생(논문도 명시).
- 새 하이퍼파라미터 α: "거의 항상 α=0.1"은 GPT-J/GPT-NeoX·언어함수 태스크에 국한된 경험적 관찰. GR00T처럼 아키텍처·모달리티가 전혀 다른 모델엔 그대로 이식 불가, 재스윕 필요.
- 검증 범위: 태스크가 전부 "명확한 단일토큰 함수 실행"(antonym 등)에 국한 — "성공/실패"처럼 비대칭·다차원·연속적인 행동 결과에 대한 conceptor 성능은 저자도 "more complex behaviors/tasks"를 future work로 명시. 정확히 우리 프로젝트가 도전하는 미검증 영역.
- 합성함수(AND) 실험은 GPT-J 한 모델·2개 조합만 테스트 — 더 많은 concept·더 큰 모델로의 확장성은 미검증.
- 모델 규모 상한 20B(GPT-NeoX). 저자 스스로 "scalability to larger models"를 future work로 남김 — 70B급 LLM이나 GR00T 같은 diffusion/flow 기반 action decoder에는 검증되지 않음.
