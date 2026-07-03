# Representation Engineering: A Top-Down Approach to AI Transparency (Zou et al. 2023)

- 출처: arXiv:2310.01405 (v4, 2025-03-03) · Center for AI Safety 외 (Zou, Phan, Chen, Campbell, Guo, Ren 등, 공동 1저자 다수)
- PDF: `docs/Activation_steering_basic/RepE_2310.01405.pdf`
- 정독 섹션: §1 Introduction, §3 Representation Engineering (3.1 Reading / 3.2 Control), §4 Honesty in-depth example, §7 Conclusion, Appendix A
- tier: must
- 한줄 역할: activation steering을 포함해 "표현을 읽고(reading) 조작한다(control)"는 **population-level·top-down 프레임 자체를 정의한 시조 논문**. 이후 CAA/ActAdd/COAST 등 개별 steering 기법들이 이 우산(RepE) 안의 control 방법론 중 하나로 위치한다.

## 문제·동기

LLM 내부는 대부분 블랙박스로 취급된다. 기존 mechanistic interpretability(MI)는 뉴런·회로 단위의 **상향식(bottom-up)** 분석에 집중하는데, 이는 복잡한 고수준 인지 현상(정직성, 권력 추구 성향 등)을 설명하는 데 한계가 있다(P.W. Anderson "More Is Different" 인용). 저자들은 인지신경과학의 Sherringtonian(뉴런·회로 중심) vs Hopfieldian(뉴런 집단의 활동 패턴=표현 중심) 대비를 빌려와, **표현(representation) 자체를 분석 단위로 삼는 하향식(top-down) 접근**을 제안한다. 목표는 이해(transparency)뿐 아니라 실질적 **제어(safety-relevant control)**까지 하나의 틀로 묶는 것.

## 핵심 아이디어(3-5문장)

RepE는 두 축으로 구성된다: **Representation Reading**(모델 내부에 특정 개념/기능에 대응하는 방향이 존재하는지 찾아내는 것, 즉 read-out)과 **Representation Control**(찾아낸 방향으로 활성화를 이동시켜 모델 행동을 바꾸는 것, 즉 steering). 두 절차 모두 개별 뉴런이 아니라 **은닉 상태 벡터 집합(population of activations)** 을 다루며, 대조적 자극(contrastive stimuli, 예: "정직하게 답하라" vs "거짓말하라")에 대한 활성화 분포 차이에서 방향을 뽑아낸다는 점에서 population-level 통계 프레임이다. 논문은 reading의 baseline으로 **LAT(Linear Artificial Tomography)**를, control의 baseline으로 **Reading Vector / Contrast Vector / LoRRA**를 제시하고, 이 둘이 서로 다른 목적(상관 vs 인과)을 가지므로 "읽기에 좋은 방향이 곧 제어에 좋은 방향은 아니다"라는 점을 명시한다. 저자들은 이를 개별 stimulus에 activation vector를 적용하는 좁은 의미의 "activation engineering"(Turner et al. ActAdd)과 구분해, reading+control을 통합하고 representation tuning(LoRRA)까지 포함하는 **더 넓은 우산 개념**으로 RepE를 정의한다.

## 방법(메커니즘; reading vs control, LAT 등)

**Reading — LAT 파이프라인 (4단계, §3.1.1):**
1. Stimulus/Task 설계: 개념(concept, 예: 정직성)은 `"Consider the amount of <concept> in: <stimulus>. The amount of <concept> is"` 템플릿, 기능(function, 예: 거짓말하기)은 `USER: <instruction> <experimental/reference prompt> ASSISTANT: <output>` 형태의 대조 템플릿(T+/T−)으로 유도.
2. 신경 활동 수집: concept은 마지막 토큰(또는 concept 토큰) 표현, function은 응답 내 모든 토큰 표현을 각 레이어에서 수집.
3. 선형 모델 구성: 기본은 **비지도 PCA**를 대조쌍 차이벡터(paired difference, `A(i) − A(j)`)에 적용해 1번째 주성분을 "reading vector" v로 삼음(지도 방식인 linear probing/mean-difference도 가능). 예측은 내적 `Rep(x)^T v`.
4. Monitoring: 레이어×토큰 위치별로 v와의 내적을 시각화(LAT scan)해 개념/기능의 활성 강도를 추적.
   - 평가는 4범주: Correlation(상관) / Manipulation(자극·억제 인과) / Termination(제거 시 성능 저하, lesion) / Recovery(제거 후 재주입으로 복구, rescue). 즉 상관을 넘어 인과를 확립하려면 조작 실험까지 요구.

**Control — baseline transformations (§3.2):**
- Operand(조작 대상 벡터) 3종: (a) **Reading Vector** — stimulus 독립적, 항상 같은 방향(간단하지만 약함). (b) **Contrast Vector** — 매 입력마다 대조 프롬프트 쌍을 실제로 통과시켜 얻는 stimulus-dependent 방향(강하지만 추론 시 3배 이상 연산 필요, 레이어 전파로 인한 cascading effect 보정 위해 얕은 레이어부터 순차 적용). (c) **LoRRA(Low-Rank Representation Adaptation)** — Contrast Vector를 타깃으로 하는 loss로 저랭크 adapter를 파인튜닝(Algorithm 1), 추론 시 추가 비용 없음.
- Operator(조작 연산) 3종: **Linear Combination** R' = R ± v(자극/억제), **Piece-wise** R' = R + sign(RᵀV)v(조건부 증폭), **Projection** R' = R − (RᵀV/‖v‖²)v(해당 방향 성분 제거).
- 우리 프로젝트의 contrastive conceptor(C_success ∧ ¬C_failure, multi-dim)는 이 논문의 단일 Contrast Vector(1차원)를 다차원 subspace 연산자로 확장한 형태로 이해할 수 있음(COAST 계열이 이 확장을 담당).

## 실험·결과(핵심 수치)

- **TruthfulQA MC1 (reading, Table 1)**: LLaMA-2-Chat 7B/13B/70B 평균 — Zero-shot 32.3% → Heuristic(언어화 자기평가) 47.2% → **LAT 60.6~60.7%**. 표준 zero-shot보다 28%p 이상 개선, 소수(5~10개) stimulus로도 유사 성능 → 모델이 "정직성"에 대한 **일관된 내부 개념**을 갖는다는 근거로 제시.
- **Honesty control (Table 2, TruthfulQA MC1)**: 7B-Chat None 31.0 → ActAdd 33.7 → Reading 34.1 → **Contrast Vector 47.9(SOTA)** → LoRRA 42.3(3배 이상 적은 연산으로 근접 성능). 13B에서도 동일 경향(None 35.9 → Contrast 54.0). 13B 모델이 대조 벡터 제어로 GPT-4에 근접.
- **Lie/hallucination detection (§4.3.2, 정성)**: 토큰 단위 부정직성 점수가 긴 시나리오에서 실제 로짓 분포(정답 D에도 C 확률 37% 등)와 정합, "거짓말할 결과를 고민하는" 서술 구간도 탐지.
- **Memorization 억제 (Table 7)**: 유명 인용구 완성 과제에서 원 모델 EM 90%+ → 기억 방향을 빼는 control로 EM/SIM 대폭 하락(예: LATQuote EM 89.3→47.6), 역사 사실 QA 정확도는 97.2%→96.2%로 거의 손상 없음 → steering이 world knowledge는 보존하며 표적 행동만 억제 가능함을 보여주는 selectivity 근거.
- Reading은 honesty 외 utility/morality/power/emotion/harmlessness/fairness/fact editing에도 반복 적용되어 넓은 적용성을 주장(§5–6, 본 정독에서는 개관만 확인).

## activation-steering 흐름 위치

이 논문은 "activation steering"이라는 실무적 기법 계보(ActAdd/CAA 등)를 **reading+control 통합 프레임(RepE)**으로 재정의한 시조 논문이다. 저자들은 명시적으로 Turner et al.의 "activation engineering"(개별 stimulus에서 뽑은 하나의 벡터를 얼려진 모델에 더하는 좁은 기법)과 자신들을 구분하면서, (1) reading을 통한 내부 개념의 **존재 검증**, (2) 그 개념 방향을 이용한 **인과적 control**, (3) 나아가 **representation tuning(LoRRA)**까지 하나의 우산으로 묶는다. 이후 문헌 지형에서 이 논문은 "reading vs control", "population-level activation 통계", "correlation→manipulation→termination→recovery" 4단계 인과성 검증 틀의 표준 어휘를 제공했고, CAA/ActAdd류(단일 additive steering vector)와 COAST류(contrastive operator, 우리 프로젝트가 쓰는 C_steer)는 모두 이 §3.2의 Contrast Vector / Linear Combination 연산의 확장·정교화로 위치시킬 수 있다.

## 우리 프로젝트 연결

- 우리 method의 `C_steer = C_success ∧ ¬C_failure`는 RepE §3.2의 **Contrast Vector**(성공/실패라는 대조적 조건에서 뽑은 population-level 차이 방향)를 다차원 subspace/operator로 일반화한 것 — RepE가 제시한 "reading vector(약함) vs contrast vector(강하지만 비쌈) vs LoRRA(파인튜닝, 저비용)" 트레이드오프 구도를 우리 conceptor fit(사전에 오프라인으로 fit해 두고 추론 시 저비용 적용)이 그대로 물려받는다.
- RepE의 **Manipulation/Termination/Recovery** 인과 검증 틀은 우리가 "steering이 실제로 SR을 올리는가"를 인과적으로 재측정(ΔSR)하는 절차와 동일 논리 — 단순 상관(succ/fail latent가 분리된다)만으로는 부족하고 개입 실험이 필요하다는 근거를 이 논문에서 재확인할 수 있음.
- 다만 RepE는 **LLM 텍스트 도메인·population-level·오프라인(정적) 개념** 조작이며, 우리의 핵심 난제인 **online phase/failure-type 식별**(추론 중 어느 pathway가 언제 실패하는지)은 다루지 않는다. RepE의 LAT monitoring(레이어×토큰 위치별 스캔)은 사후 시각화 도구에 가깝고, 우리처럼 rollout 진행 중 실시간으로 phase를 라우팅에 쓰는 문제의식은 없음 — 이 논문은 "steer의 수학적 정의"를 빌려오는 토대 역할이지, phase-matched 문제의 해법은 아니다.

## 면접 포인트(Q→A 1-3개)

**Q1. RepE가 말하는 "reading"과 "control"은 왜 분리해서 취급해야 하나?**
A. reading은 특정 방향과 개념 사이의 **상관**만 보장한다(분류 정확도가 높다고 해서 그 방향을 조작했을 때 실제로 행동이 바뀐다는 보장은 없음). 논문은 "reading에 효과적인 방법이 반드시 control에도 효과적인 것은 아니다"라고 명시하고, 인과성 확립을 위해 Manipulation/Termination/Recovery 3종 개입 실험을 요구한다. 우리 프로젝트에서도 succ/fail latent 분리(reading 성격)와 steering으로 인한 ΔSR(control 성격)을 분리해서 검증해야 하는 이유가 여기서 온다.

**Q2. Reading Vector, Contrast Vector, LoRRA의 차이와 트레이드오프는?**
A. Reading Vector는 LAT로 뽑은 stimulus-독립적 고정 방향(구현 쉬움, 효과 약함). Contrast Vector는 매 입력마다 대조 프롬프트 쌍을 실제로 실행해 얻는 stimulus-dependent 방향(효과 최강, TruthfulQA SOTA, 그러나 추론 연산 3배 이상 증가 + cascading effect 보정을 위해 레이어를 얕은 것부터 순차 재계산해야 함). LoRRA는 Contrast Vector를 타깃으로 하는 loss로 LoRA adapter를 미리 학습해 두면, 추론 시엔 추가 비용 없이 Contrast Vector에 근접한 성능을 낸다. 우리의 conceptor도 오프라인 fit(LoRRA와 유사한 위치, 사전 계산) + 추론 시 저비용 적용(h·Mᵀ)이라는 동일한 트레이드오프 해법을 택한 것.

**Q3. RepE와 mechanistic interpretability(MI)의 차이는? 왜 top-down인가?**
A. MI는 뉴런·회로 단위의 상향식(bottom-up) 분석으로 "작은 메커니즘을 찾아 조합"하는 접근인 반면, RepE는 Hopfieldian 관점을 빌려 "표현(뉴런 집단의 활동 패턴)"을 그 자체로 분석 단위로 삼는 하향식(top-down) 접근이다(Appendix A). 저자들은 두 접근이 상호 배타적이지 않고 "생물학이 응용화학이 아니듯 RepE도 응용 MI가 아니다"라며 상보적 관계로 규정한다. VLA steering 맥락에서는 개별 attention head/뉴런을 찾기보다 hidden state 전체의 population 통계(평균·공분산·conceptor)로 개입하는 우리 접근이 이 top-down 계보에 속한다.

## 한계·비판

- **모두 LLM 텍스트 도메인**: 실험이 LLaMA-2-Chat/Vicuna 등 언어모델에 국한되며 continuous control(로봇 액션 시퀀스)이나 diffusion-style action head(VLA의 DiT)로의 이전 가능성은 검증되지 않음.
- **정적·population-level**: reading vector/contrast vector는 오프라인에 고정된 대조 프롬프트로 뽑고 추론 중 값이 바뀌지 않는다 — "언제(rollout의 어느 시점) 적용해야 하는가"라는 phase-matched 문제의식이 없음. 우리 프로젝트의 핵심 난제(online phase/failure-type 식별)는 이 논문 범위 밖.
- **Contrast Vector의 비용**: 최고 성능 기법이 추론 시 3배 이상 연산과 레이어별 순차 재계산을 요구 — 실시간 로봇 제어(steering latency 예산이 촉박한 VLA 서빙)에는 그대로 적용하기 부담스러움. LoRRA로 우회하지만 이는 파인튜닝을 요구해 "백본 재학습 없음" 제약과 충돌.
- **평가의 자기참조성**: 일부 지표(예: TruthfulQA 방향 선택에 validation 라벨을 일부 사용, 언어화 heuristic baseline)는 완전한 비지도라 보기 어렵고, 저자 스스로도 "LAT의 불안정성"을 부록에서 언급함(Table 8 표준편차가 최대 5.6%p로 큼, 특히 13B).
- **lie detection의 해석 모호성**: §4.3.2에서 저자들도 "거짓말 자체"와 "거짓말의 결과를 고민하는 사고 과정"을 탐지기가 구분하지 못한다고 인정 — 단일 방향이 여러 다른 인지 과정을 뭉뚱그릴 위험(우리 프로젝트가 VL/DiT pathway를 분리하려는 동기와 상통: 단일 global 방향은 서로 다른 실패 유형을 뭉갤 수 있음).
