# Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (Li et al. 2023)

- 출처: NeurIPS 2023 · arXiv 2306.03341v6 · PDF: docs/Activation_steering_basic/ITI_2306.03341.pdf · 섹션=§3 방법(§3.1 Setup, §3.2 Probing, §3.3 Inference-Time Intervention 중심, §4 실험·§5 분석 보조) · tier=must · 한 줄 역할: probe로 "진실성" 관련 attention head를 소수만 골라 그 head의 활성화 공간에만 truthful 방향으로 shift를 가하는 sparse, head-level activation steering의 원형 — 우리 pathway/phase 선택적 개입과 "어디에 개입할지 먼저 고른다"는 설계철학이 가장 가깝다.

## 문제·동기
LLM은 정답을 "아는" 것과 "말하는" 것 사이에 괴리가 있다(generation-discrimination gap). 저자들은 이를 probe accuracy(중간 활성화로 분류)와 generation accuracy(실제 출력)의 차이로 조작적으로 정의했고, LLaMA-7B에서 TruthfulQA 기준 둘 사이에 40%p 차이를 관측했다. 즉 모델 내부에는 진실 관련 정보가 있는데 표면 생성이 그것을 반영하지 못한다. RLHF 등 기존 해법은 대량의 annotation과 연산이 필요하고, 인간/AI 평가자를 "만족시키는" 방향으로 학습되므로 기만(sycophancy)이 최적 전략이 될 위험이 있다. ITI는 이를 추가 학습 없이, 추론 시 활성화만 이동시켜 해결하려 한다.

## 핵심 아이디어 (novelty)
1. Transformer의 MHA를 residual stream에 head별로 독립적으로 벡터를 더하는 형태로 재서술(식 1: x_{l+1} = x_l + Σ_h Q_h Att_h(P_h x_l))하고, Att 직후·Q_h 이전의 head별 활성화 x_h_l 공간에서 개입.
2. head마다 별도 linear probe를 학습해 "진실성"과 상관된 head를 순위화 — 전체 head 중 일부만 강하게 진실성을 인코딩(층별로도 비균질, 초중반 층에 집중).
3. 개입은 (a) 상위 K개 head만 선택(sparse, "minimally invasive")하고 (b) 그 head들의 활성화에 표준편차 σ 단위로 스케일한 방향 θ를 더하는 두 축으로 구성 — head 선택(어디)과 방향/강도(얼마나)를 분리해서 설계.
4. 방향 자체는 probe 가중치가 아니라 mass mean shift(참/거짓 활성화 평균의 차 벡터)가 실전에서 더 낫다는 것을 실증 — probe의 분류 경계 법선과 클래스 평균차 벡터가 anisotropic 분포에서는 다른 방향이고, 개입에 causal한 쪽은 후자.

## 방법 (§3 상세)
- probe: p_theta(x_h_l) = sigmoid(inner(theta, x_h_l)), head·층마다 1개, TruthfulQA 참/거짓 QA pair의 마지막 토큰 활성화로 학습(4:1 train/val). 최고 성능 head는 14층 18번째 head, 검증정확도 83.3%(baseline 50%).
- 방향 후보 두 가지 비교(Table 3): probe weight direction(분류 경계에 수직) vs mass mean shift(참/거짓 클래스 평균 벡터 차이) — anisotropic 분포에서 최적 분리 초평면이 두 평균의 차이 방향과 다를 수 있음(부록 B, 토이 가우시안 예시로 시각화). CCS(비지도, 쌍 정보만 사용) baseline도 비교.
- 개입 수식(식 2): x_{l+1} = x_l + Σ_h Q_h(Att_h(P_h x_l) + α·σ_h·θ_h). 선택 안 된 head는 θ=0. α는 개입 강도(표준편차 배수), K는 개입 대상 head 개수. σ_h는 train+val 활성화의 표준편차로 추정.
- 자기회귀 생성 매 스텝마다 동일 개입 반복(디코딩 알고리즘과 무관하게 적용 가능).
- 계산비용: head 수와 무관하게 층당 상수벡터 1개(식 3, α·Σ Q_h(σ_h θ_h))만 residual에 더하면 되므로 거의 0 — MHA bias term에 baking해 오프라인 편집도 가능(공개 checkpoint 존재).
- 하이퍼파라미터 K, α: 5% TruthfulQA로 그리드서치, 성능·침습성(CE, KL) 트레이드오프로 K=48, α=15 선택(§4.3 기준 실험) — 이후 실험은 mass mean shift 기준 K=48, α=15 혹은 방향별 재탐색(Table 3의 mass mean shift는 α=20).
- 모델선택은 2-fold cross-validation(head 선정·방향추정에 test 누출 방지)으로 엄격 분리.

## 실험·결과
- TruthfulQA generation track, 지표 true×informative(%): Baseline 30.5 → Baseline+ITI 43.5(Table 1); few-shot prompting과 결합 시 49.5 → 51.4.
- Instruction-tuned 모델에 적용(Table 2): Alpaca 32.5 → 65.1(약 2배), Vicuna 51.5 → 74.0 — 논문 초록 헤드라인 수치.
- 방향 비교(Table 3): random 31.2, CCS 33.4, probe weight 34.8, mass mean shift 42.3 — mass mean shift가 최고이며 더 강한 α에도 안정적.
- head 선택 방식 비교(Table 5, §5.5): head-wise selection(42.3) > point-wise selection(39.2, informativeness 붕괴) > no selection(35.4) — "어디에 개입할지"를 sparsify하는 것 자체가 성능에 기여, 전 head에 개입하면 오히려 나쁨.
- α-K 스윕(Fig.4)에서 true×informative는 역U자형 — 강도가 과하면 informativeness(예: "I have no comment" 남발)가 무너짐, truthfulness-helpfulness trade-off 명시적 관측.
- OOD 일반화(§5.3, Table 4): TruthfulQA에서 학습한 방향/하이퍼파라미터를 그대로 Natural Questions/TriviaQA/MMLU에 적용 — 소폭이지만 전 벤치마크에서 개선(MMLU가 가장 큼), 성능 저하는 없음.
- 부록 B 토이 실험: anisotropic Gaussian에서 최적분리 초평면(정확도 82.5%)과 그 법선(70.5%)이 다른 방향임을 시각적으로 증명 — probe 방향과 개입에 유효한 방향의 괴리를 뒷받침.

## activation-steering 흐름에서의 위치
CCS(Burns 2022, 비지도 대조 탐색으로 truthful 방향 발견, causal 여부 불명)를 계승해 "탐지"에서 "개입"으로 확장한 초기 논문 중 하나(2306, 거의 ActAdd/CAA/RepE와 동시기). 기존 weight-editing(ROME 등)과 달리 활성화만 바꾸는 activation editing 계열. 본 논문의 핵심 기여는 residual-stream 전체가 아니라 **attention head 단위의 국소·희소 개입**이라는 점 — 이후 연구들(RepE는 층 전체에 개입, Conceptors류는 multi-dim subspace로 일반화)과 비교해 "어디에 개입할지"를 probe 정확도로 명시적으로 선별한다는 게 차별점. Park et al.(LRH, 2311)의 "probe 방향=steering 방향" 이론적 동치성은 본 논문의 mass-mean-shift가 probe-weight보다 낫다는 실증 결과와는 살짝 결이 다름(이론과 실전 사이 간극 — 저자도 anisotropic 분포에서 둘이 다르다고 명시).

## 우리 프로젝트 연결
- **head-wise sparse selection**(K개만 개입) 은 우리의 pathway(VL/DiT) 분리 steering과 개념적으로 가장 가깝다: ITI가 "이 head는 진실성과 관련 있다/없다"를 probe accuracy로 판정해 개입 범위를 좁히듯, 우리는 "이 pathway/이 phase가 실패 유형과 관련 있다"를 판정해 개입을 라우팅해야 한다 — 둘 다 "전체에 개입하면 오히려 나쁘다"(Table 5, no-selection 35.4 < head-wise 42.3)는 실증적 근거를 공유.
- ITI는 head 선택을 **정적**으로(TruthfulQA 전체에서 한 번 뽑은 K개 head, 매 토큰 동일하게 적용) 한다 — 우리가 미해결로 남긴 "온라인 phase/pathway 식별"과 달리 ITI는 입력별로 다른 head를 켜고 끄지 않는다. 즉 ITI는 "어디"는 고정, "얼마나"만 스칼라(α)로 조절 — 우리의 phase-matched(rollout 진행에 따라 개입 대상이 바뀜) 설계보다 훨씬 단순한 세팅이라는 점이 한계이자 대조점.
- mass mean shift가 probe weight보다 우월했다는 결과는 우리 conceptor(C_success ∧ ¬C_failure, 대조 통계 기반)가 단순 판별경계보다 나을 수 있다는 방법론적 정당화로 재사용 가능 — 다만 ITI는 단일벡터(1-dim) shift, 우리는 multi-dim 연산자라는 점에서 일반화 방향이 같다.
- σ_h 단위 스케일링(α·σ)은 우리 개입 강도 스윕(사다리식 ablation)에서 activation의 자연스러운 scale에 맞춰 강도를 정규화하는 관행의 선례로 참고 가능.

## 면접 포인트 (Q→A)
1. Q: "ITI가 다른 steering과 다른 점은?" A: "대부분의 activation steering(ActAdd, CAA 등)은 residual stream 전체 혹은 특정 층 전체에 한 벡터를 더한다. ITI는 그보다 세밀하게, attention head 단위로 probe 정확도가 높은 상위 K개 head만 골라 그 head의 출력에만 개입한다. 이 sparse selection 자체가 성능에 기여한다는 걸 ablation(Table 5)으로 보였다."
2. Q: "왜 probe 방향이 아니라 mass mean shift를 쓰나?" A: "분포가 anisotropic이면 최적 분류 경계(probe weight)의 법선과 두 클래스 평균의 차이 벡터가 다른 방향을 가리킬 수 있다(부록 B 토이 예시). 실험적으로 mass mean shift가 더 강한 개입 강도에서도 안정적이고 성능이 더 좋았다 — 분류에 좋은 방향이 곧 개입에 좋은 방향은 아니라는 교훈."
3. Q(우리 프로젝트 관점): "ITI를 우리 VLA steering에 어떻게 참고하나?" A: "ITI의 head-wise sparse selection은 우리 pathway 분리(VL/DiT 각각 따로 steer)와 같은 설계철학이다. 다만 ITI는 개입 대상(head 집합)을 정적으로 한 번 정하고 고정하는 반면, 우리는 rollout phase에 따라 개입 대상이 바뀌어야 하는(phase-matched) 더 어려운 문제를 풀어야 한다 — ITI에는 없는 온라인 라우팅이 핵심 난제다."

## 한계·비판
- head 선택과 방향 추정이 TruthfulQA 특유의 이진(참/거짓) 레이블에 의존 — "성공/실패"처럼 원인이 다양하고(goal 실패 vs motor 실패) 비대칭적인 우리 문제에 그대로 이식하기엔 레이블 설계가 더 어렵다.
- 개입이 **정적**이다: 매 토큰 동일한 K개 head, 동일한 α로 개입 — 문맥에 따라 개입 여부/강도를 바꾸는 online routing은 다루지 않음(저자도 §6에서 "unsupervised discovery"를 future work로 남김, 온라인 적응은 아예 스코프 밖).
- truthfulness-helpfulness trade-off(Fig.6B)가 명시적으로 존재 — α를 올리면 "I have no comment" 남발로 informativeness가 붕괴. 우리 문제에서도 steering 강도를 올리면 task-relevant capability가 훼손될 위험이 유사하게 있을 수 있음(ΔSR만 보면 놓치는 부작용).
- 저자 스스로 "ITI가 내부적으로 무엇을 하는지 mechanistic하게 이해한다고 주장하지 않는다"(§2)고 명시 — probe 정확도가 높다고 그 head가 causal하게 진실성을 담당한다는 보장은 약하다(상관 vs 인과 문제, mass-mean-shift가 더 나은 이유도 사후적 설명일 뿐 이론적 근거는 약함).
- 평가가 GPT-judge(자동 채점기)에 크게 의존하며 저자도 사람 평가와 약간의 괴리(truthfulness 과대추정, informativeness 과소추정)를 인정 — 지표 자체의 노이즈.
- 단일 아키텍처 계열(LLaMA/Alpaca/Vicuna, 텍스트 전용 decoder-only LM)에서만 검증 — continuous action-generation을 하는 VLA의 DiT 같은 비-autoregressive-token 구조에 head 개념이 그대로 대응되지 않는다.
