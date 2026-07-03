# The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets (Marks & Tegmark, 2023/2024 COLM)

- 출처: COLM 2024 (conference paper) / arXiv:2310.06824v3
- PDF 경로: `docs/Activation_steering_basic/GeometryOfTruth_2310.06824.pdf`
- 정독 섹션: §2 데이터셋 구성 (전체 통독, §2 중심)
- tier: must
- 한줄역할: "성공/실패(참/거짓) 활성화를 어떻게 나누고(diff-of-means, PCA), 그 방향이 실제로
  모델 출력을 인과적으로 바꾸는지(patching 개입)"를 가장 깨끗한 세팅에서 보여주는 probing→steering
  다리 논문. 우리 project의 succ/fail latent 분리·conceptor 방향추출의 방법론적 원형.

## 문제·동기

LLM은 내부적으로 "이 문장이 참/거짓"임을 알면서도 거짓을 출력할 수 있다(예: GPT-4가 CAPTCHA를
풀기 위해 시각장애인이라고 거짓말). 활성화에 truth를 분류하는 probe를 학습하는 선행연구
(Azaria&Mitchell 2023, Li et al. 2023b ITI, Burns et al. 2023 CCS)들이 있었지만, 이 방법들이
"진짜 truth 방향"을 잡는지 아니면 truth와 상관관계만 있는 다른 feature(예: "not" 단어 존재,
그럴듯한 문장 여부)를 잡는지가 논쟁적이었다(Levinstein & Herrmann 2023이 일반화 실패 지적).
저자들은 의도적으로 아주 단순·명확·무논쟁적인 참/거짓 문장 데이터셋(cities, sp_en_trans,
larger_than 등)을 구성해, "truth가 선형으로 표상되는가"를 최대한 깨끗하게 검증하고자 한다.

## 핵심 아이디어

3갈래 증거로 "LLM은 (충분한 스케일에서) 참/거짓을 선형으로 표상한다"를 논증한다.
1. PCA 시각화: 활성화를 2D로 투영하면 참/거짓이 뚜렷이 선형 분리됨(Fig.1).
2. Transfer/일반화: 한 데이터셋에서 학습한 probe가 구조·주제가 다른 데이터셋에도 일반화됨.
3. 인과 개입(patching/activation steering): 특정 hidden state를 다른 방향으로 이동시키면
   모델이 거짓 문장을 참으로(또는 그 반대로) 취급하게 만들 수 있음.
핵심 기여는 로지스틱회귀(LR)의 결함을 지적하고, 더 단순한 **mass-mean(difference-in-means)
probe**가 분류 정확도는 LR/CCS와 비슷하면서 **인과적으로는 훨씬 더 강하다**는 것을 보인 점.

## 방법

**Step 0 — patching으로 위치 특정(§3)**: "Tokyo는 일본에 있다...Chicago는 캐나다에 있다"
프롬프트 쌍(거짓판 pF, 참판 pT)에서 pT의 residual stream 활성화를 pF에 토큰별·레이어별로
patch하며 log P(TRUE)-log P(FALSE) 변화를 측정. 3개 그룹의 인과적 hidden state 발견:
(a) 개체명(Chicago/Toronto) 표상, (b) **문장 종결부(마침표 토큰) — "요약된 문장 전체 truth"
표상, (c) 최종 예측 직전. 이후 분석은 그룹 (b)의 최하류(most downstream) hidden state 하나를
씀(few-shot 없이, 각 데이터셋 평균으로 centering).

**PCA 시각화(§4)**: 그룹(b) 활성화를 dataset별로 중심화 후 PCA. curated dataset(비-truth
변동 최소화하도록 설계)에서는 top-2 PC에서 참/거짓이 뚜렷이 갈림. 이 분리축의 데이터셋 간
정렬(alignment) 여부로 truth가 "얼마나 추상적인 개념으로 통합되어 있는지"를 논증
(정렬=공유 truth 방향, 반대극성=surface-level feature인 "close association" 등이 섞임,
직교=두 feature 다 salient).

**Mass-mean(diff-of-means) probing(§5.1)**: θ_mm = μ+ - μ-  (평균 truth 활성화 - 평균 false
활성화). LR은 max-margin separator로 수렴하는데, truth 방향 θt와 orthogonal하지 않은 다른
feature θf가 있으면 LR이 θf 쪽으로 "interference 제거"를 위해 방향을 왜곡시킨다는 문제를
지적(Fig.4). IID 보정판은 whitening: p_mm^iid(x) = σ(θ_mm^T Σ^-1 x) (Σ = 클래스별 중심화 후
공분산; Fisher LDA와 동치). 이론적으로(App.F) 가우시안 등분산 가정 하에서 Σ^-1 θ_mm ∝ θ_lr
— 즉 mass-mean+whitening은 LR과 "평균적으로" 같은 방향을 향하지만, non-IID/OOD 상황에서
whitening 없이 θ_mm 자체를 쓰면 LR보다 훨씬 causal하게 작동.

**인과 개입 실험(§6)**: θ를 normalize해서 "평균 false 활성화 + θ = 평균 true 활성화 취급"이
되도록 스케일 조정. false 문장 활성화 x → x+θ (또는 true 문장 → x-θ)로 그룹(b) hidden state를
치환하고, downstream P(TRUE)-P(FALSE) 변화를 정규화 간접효과(NIE)로 측정. 핵심: OOD 데이터
(sp_en_trans로 평가, 다른 데이터셋에서 학습한 probe로 개입)에서 평가 — Li et al. 2023b(ITI)와
차별화되는 지점.

## 실험·결과

- Table2 (NIE): MM이 8개 조건 중 7개에서 LR·CCS를 능가, 종종 큰 차이로(예: cities 학습·13B·
  false→true: LR .13 vs MM .77). 그런데 LR/MM/CCS의 **분류 정확도는 거의 동일** — "정확도는
  같아도 인과성은 다르다"가 이 논문 최대 발견.
- 모델 스케일이 커질수록(7B→13B→70B) probe 일반화(OOD 정확도)가 좋아짐 → truth가 더 추상적
  general concept으로 통합됨(scale에서 emergent linear structure).
- "statements+negation"(cities+neg_cities)으로 학습하면 negation 포함 OOD에 대한 일반화가
  향상(단, larger_than+smaller_than은 다른 패턴).
- likely(그럴듯함) 데이터로 학습한 probe는 분류 정확도는 나쁘지만 causal 개입에서는 의외로
  강함(특히 MM) → truth와 plausibility가 얽혀있지만 완전히 같지는 않음을 시사.
- likely dataset을 별도로 설계해 "truth ≠ probable text" 통제(neg_cities/neg_sp_en_trans는
  진실성과 확률이 음의 상관, r=-.63/-.89).

## activation-steering 흐름 위치(읽기→쓰기 다리)

이 논문은 "probing(읽기)에서 causal steering(쓰기)으로" 넘어가는 다리를 **그 자체가 방법론적
주제**로 삼은 논문이다. §3 patching으로 "어디를 읽어야 하는지"(hidden state 위치)를 먼저
localize하고, §4-5에서 "무슨 방향을 어떻게 추출할지"(PCA/LR/CCS/MM), §6에서 "그 방향을 실제로
밀었을 때 출력이 바뀌는지"(인과 개입)를 분리해서 각각 측정한 뒤 상호 대조한다. 이 구조 자체가
"read-out ≠ write-in" 문제의 명시적 실증: 같은 정확도의 probe라도 causal steering 효과는
천차만별이라는 것을 정량적으로 보여준 최초급 사례 중 하나. mass-mean(diff-of-means)이 LEACE
(Belrose et al. 2023, App.G)의 최적 선형 지우기(erasure) 방향과 동치라는 것도 부록에서 증명 —
"분리를 최적으로 지우는 방향 = steering에 가장 적합한 방향"이라는 대칭성.

## 우리 프로젝트 연결(방향추출·인과검증)

- 우리 project의 C_steer = C_success ∧ ¬C_failure(contrastive conceptor)는 이 논문의
  difference-in-means(θ_mm = μ_success - μ_failure)의 multi-dimensional 확장으로 볼 수 있음.
  이 논문이 diff-of-means 단일벡터가 LR/PCA보다 causal하다고 보인 것은, COAST 계열의 conceptor
  접근이 "정확도"보다 "causal steering 효율"을 기준으로 방향을 골라야 한다는 근거를 준다.
- §3의 patching localization(어느 layer·어느 token position이 causal한가)은 우리의
  "pathway(VL/DiT) 분리 + phase-matched" hook 지점 선정과 방법론적으로 동일한 절차 —
  이 논문은 "문장 종결 토큰의 특정 레이어대"로 좁혔듯, 우리는 phase-bin별·pathway별로
  hook 지점을 좁혀야 함 (현재 action_head DiT block 고정 hook과 비교 가능).
- OOD causal 검증(§6, sp_en_trans 학습 안 한 데이터로 평가)은 우리 project의 "unseen task
  causal steering ΔSR 재측정" 표준과 정확히 같은 사상 — in-distribution 정확도만으로 방향을
  고르면 안 된다는 근거를 우리 실험 설계에도 직접 인용 가능.
- 길이confound와 유사한 논점: 이 논문의 antipodal/orthogonal misalignment 논의(§4, cities vs
  neg_cities)는 "표면적 feature(surface-level, 예: '단어 not의 존재')가 진짜 target feature와
  혼입"되는 문제로, 우리 project의 "길이가 늘 실패=timeout과 혼입"되는 confound와 구조적으로
  동일한 패턴(대리 feature가 salient하면 분리축이 왜곡됨).
- MM probe가 whitening(Σ^-1) 적용 시 LR과 이론적으로 근접한다는 F.1 정리는, conceptor의
  공분산 기반 연산(C_success, C_failure)이 "우연이 아니라 whitened diff-of-means"라는 수학적
  해석을 뒷받침 — conceptor는 본질적으로 whitened contrastive direction의 subspace 일반화.

## 면접 포인트(Q→A)

Q1. "probing 정확도가 높으면 그 방향으로 steering해도 잘 될까?"
A1. 아니다. 이 논문의 핵심 결과가 정확히 그 반례다: LR, MM, CCS는 sp_en_trans 분류 정확도가
거의 동일하지만, 인과 개입(NIE)에서는 MM이 7/8 조건에서 LR·CCS를 크게 앞선다(예: NIE .13 vs
.77). 이유는 LR이 max-margin separator를 찾는 과정에서 truth와 non-orthogonal한 다른 feature
(θf)의 "interference 제거"에 최적화되어, 결정경계는 좋지만 실제 feature 방향에서 벗어나기
때문(Fig.4). 즉 probing은 정보가 선형으로 "복원 가능(decodable)"한지를 보고, steering은 그
방향이 "인과적으로 사용됨(causally used)"을 요구 — 둘은 다른 질문이다. 이 구분이 read-out↔
write-in 다리의 핵심 함정.

Q2. "그럼 왜 diff-of-means가 더 causal한가?"
A2. 직관적으로 diff-of-means는 데이터의 실제 class-conditional mean shift를 그대로 따라가는
방향이라, 모델이 실제로 그 축을 따라 표상을 이동시켰을 때(=진짜 생성 과정의 방향과 가까울
때) 개입 효과가 크다. 반대로 LR/max-margin은 결정경계 최적화가 목적이라, 분류에는 필요없는
"noise 방향으로의 오버피팅"이 섞여도 정확도엔 표시가 안 나지만 개입 시엔 그 방향이 모델
내부 계산과 어긋난다. 부록 F는 가우시안+등분산 가정 하에서 whitened MM(Σ^-1θ_mm)이 LR과
"평균적으로" 같은 방향을 향함을 증명 — 즉 LR의 문제는 유한 표본·비등분산 실전에서 불안정한
방향을 고르는 것.

Q3. "우리 project의 conceptor 방향추출과 무슨 관계인가?"
A3. C_steer = C_success ∧ ¬C_failure는 diff-of-means의 multi-dim/공분산-aware 일반화다.
이 논문이 단일 방향(diff-of-means)의 인과성을 검증한 프로토콜(§6: OOD causal NIE 측정)을
그대로 우리 conceptor 검증에 적용해야 한다 — in-distribution 분류 정확도(길이 confound로
쉽게 왜곡됨)가 아니라 unseen task에서의 causal ΔSR을 기준으로 방향을 골라야 한다는 근거.

## 한계·비판

- LLaMA-2 계열(7B/13B/70B)만 검증 — 다른 아키텍처·모델군 일반화는 미확인.
- "truth"를 의도적으로 단순·무논쟁 진술로 scope 제한(App.A: QA 정답, 기만, 순응 등은 제외) —
  실전 hallucination/deception 탐지로 바로 확장되지 않음.
- 왜 likely 데이터로 학습한 MM probe가 분류는 나쁘지만 causal 개입엔 강한지, 왜
  cities+neg_cities MM이 70B에서 유독 나쁜지 등 미해결 질문을 저자 스스로 인정(§7.1) —
  diff-of-means의 causal 우위 메커니즘이 완전히 설명되지는 않음.
- 개입은 단일 hidden state(문장 종결 토큰의 한 layer)에 국한 — 다층·다토큰 동시 개입이나
  우리 project처럼 phase에 따라 동적으로 바뀌는 개입 지점은 다루지 않음(정적 single-point
  steering).
- 길이/구조 confound 통제는 "curated dataset 설계"로 사전 차단하는 방식이라, 우리 project
  처럼 사후에 confound를 통계적으로 통제하는 문제(길이-실패 혼입)에는 직접적 해법을 주지
  않음 — 데이터 설계 단계에서 미리 균형을 맞추는 전략의 참고 사례 정도.
