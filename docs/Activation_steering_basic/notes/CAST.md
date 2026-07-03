# Programming Refusal with Conditional Activation Steering (Lee et al. 2025, CAST)

- 출처: arXiv:2409.05907v3 (2025-02-17), ICLR 2025 accepted · IBM Research + UPenn (Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, Amit Dhurandhar). 코드: github.com/IBM/activation-steering
- PDF: `docs/Activation_steering_basic/CAST_2409.05907.pdf`
- §5파트: 주로 (2)왜 중요한가 + (4)전망. §5(3) "왜 안 쓰이나"에도 간접 기여(무조건 steering이 실사용을 막는 구체적 병목을 정량 실증).
- 3축: **읽기+쓰기 결합**(condition vector=읽기/probe 역할, behavior vector=쓰기/steer 역할이 한 파이프라인에 순차 결합) / **데모+OSS 라이브러리**(실서비스 배포 사례는 아니고 재현가능한 오픈소스 툴킷 단계) / **inference-only**(가중치 업데이트 전혀 없음, 초록부터 "without weight optimization" 강조).
- 한줄역할: activation steering에 "조건(if-then)" 차원을 추가해 "항상 켜진 전역 개입"을 "조건이 맞을 때만 켜지는 국소 개입"으로 바꾼 논문 — 우리가 지향하는 phase-matched steering의 LLM 도메인 원형(prototype)에 가장 가까운 선행연구.

## 문제·동기

기존 activation steering(ActAdd/CAA/RepE 계열)은 무조건(indiscriminate) 개입이다: refusal behavior vector를 더하면 harmful/harmless 가리지 않고 거부율이 오른다. 콘텐츠 모더레이션이나 도메인 특화 어시스턴트(예: 의료봇은 의료상담을 허용해야 함)처럼 "언제, 무엇을" 개입할지 조건화해야 하는 실사용 시나리오에서 이는 치명적 결함이다. Preference modeling(RLHF/DPO)은 자원집약적이고 harmful의 정의가 맥락마다 달라(같은 의료상담이 맥락별 harmful/필수) universal harm model 자체가 어렵다는 문제도 있다.

## 핵심 아이디어

"condition vector"라는 새 벡터 타입 도입. 기존 steering vector는 "behavior vector"(행동 유도)로 재명명하고, condition vector는 "지금 개입해도 되는 조건인지"를 판정하는 트리거로 쓴다.

h' ← h + f(sim(h, proj_c h)) · α · v

proj_c h = 조건벡터 c 위로의 projection, f = 이진 threshold step function(sim > θ 또는 < θ). 조건이 만족될 때만 behavior vector v를 더한다. Duality(비교방향 뒤집기 → 정확히 여집합 조건에 개입) / Modulation(θ로 조건 민감도 폭 조절) / Multi-conditioning(여러 condition vector를 OR/AND로 논리결합)까지 세 성질을 실증한다.

## 방법(조건 감지 → 조건부 steer 메커니즘)

- **벡터 추출**: 대조쌍(D+/D-)을 모델에 통과시켜 layer별 hidden state에 mean-centering + PCA, 1st PC를 vector_l로 채택. Behavior vector는 응답 접미사(suffix) 위치 hidden state 평균(국소·행동 초점), condition vector는 프롬프트 전체 토큰 hidden state 평균(전역·주제 초점) — "무엇에 관한 프롬프트인가"는 holistic해야 하고 "어떻게 반응하는가"는 국소적이어야 한다는 설계 근거.
- **감지(조건 체크)**: 생성의 첫 forward pass(prompt caching 시점)에서 **딱 1회만** sim(h, proj_c h) 계산. 조건이 만족되면 이후 모든 생성 스텝에서 behavior vector를 계속 더함(트리거 후 always-on, 토큰마다 재검사하지 않음) — 효율성을 위한 설계지만 "정적/단발성 조건부"라는 한계이기도 함.
- **최적 개입점 탐색**: (layer, threshold θ, 비교방향 >/<) 3-튜플을 F1 최대화 grid search로 탐색(전체 layer 중 앞쪽 절반만 탐색). 대부분 실험이 grid search 포함 1시간 내 재현.
- 실무 API: `steer()`(behavior만), `steer(condition_vector=...)`(조건부 단일), `multisteer(rules=["if C1 then B1", ...])`(다중 규칙) 3단계로 라이브러리화.

## 실험·결과

- **§4 단일조건(harmful/harmless)**: 8개 모델(Qwen1.5 1.8B/32B, Llama2-13B, Llama3.1-8B, NeuralDaredevil, Hermes2Pro, OLMo, Zephyr, Danube3)에서 무조건 steering(AST)은 harmless 거부율도 90%대로 치솟지만(Zephyr 0.2%→94.8%), CAST는 harmful 거부율은 비슷하게 올리면서(Hermes2Pro 19.3%→83.3%) harmless는 거의 그대로(1.0%→2.4%).
- **Saturation/속도**: 조건벡터 추출 데이터를 늘려도 성능은 빨리 정체(가중치 최적화와 다른 특성, 모델의 기존 표현력에 의존); 추출 시간은 샘플 수에 선형 비례.
- **§5 세분화조건(성적내용/법률의견/증오발언/범죄계획/건강상담 5종)**: 개별 카테고리 유도/억제, OR 논리결합으로 다중 규칙, duality로 "특정 도메인에만 응답"(constraining) 구현. Unseen 카테고리(도박/금융/사생활침해/멀웨어)에도 일반화 — 카테고리 간 semantic distance가 클수록 constraining 효과 좋음(상관관계, Fig 9c). Prompting-only baseline은 CAST보다 일관되게 열등(강제 조건화 불가).

## §5(산업)에서의 위치

이 논문 자체는 "실서비스 배포 보고"가 아니라 "실서비스에 필요한 조건부 제어 능력을 프레임화·검증한 연구"다. 다만 (a) IBM Research 산업연구소 소속, (b) 재현가능한 OSS 라이브러리 공개, (c) 문제의식이 "콘텐츠 모더레이션", "도메인 특화 어시스턴트"라는 명시적 산업 유스케이스에서 출발한다는 점에서, §5의 (2)"왜 중요한가"(무조건 개입이 실사용을 막는다는 병목을 정량 실증) + (4)"전망"(규칙기반·프로그래밍 가능한 steering이 실배포로 가는 경로)에 기여한다. Circuit Breakers가 "이미 배포된 안전 스펙"이라면, CAST는 그보다 한 단계 앞 — "배포하려면 이런 제어가능성(controllability)이 필요하다"는 요구사항을 최초로 프레임화·프로토타입한 사례.

## 우리 프로젝트 연결(조건부=우리 phase-matched와의 유사/차이)

- **구조적 동형성**: "condition vector로 감지 → behavior vector로 개입"은 우리의 "phase/failure-type 온라인 식별 → C_steer=C_success∧¬C_failure로 조건부 steer"와 파이프라인 골격이 그대로 대응한다. projection+cosine sim으로 조건 충족을 판정하는 f 자리는 우리 online detector의 판별식과 원리적으로 동일 위치.
- **결정적 차이 1 (조건의 시간성)**: CAST의 condition check는 프롬프트 첫 forward pass에서 **1회만** 일어나고 이후 고정된다(정적 조건부). 우리가 필요한 phase-matched steering은 rollout 매 스텝마다 phase가 바뀌므로 조건을 반복 재평가해야 한다(동적/시변 조건부) — CAST는 이 축을 아예 다루지 않는다(애초에 1회성 프롬프트 분류 문제였기 때문). 이것이 CAST를 그대로 못 가져오는 핵심 갭이자, 우리 프로젝트의 "★ 중심 미해결 문제"(online phase/failure-type 식별)가 바로 이 갭을 메우는 자리다.
- **결정적 차이 2 (조건의 성격)**: CAST 조건 = "입력 프롬프트의 정적 토픽/카테고리"(이산적 분류). 우리 조건 = "현재 rollout의 시간적 위치(phase)와 실패 유형"(연속적/시변 상태추정, causal하게 online에서만 알 수 있음) — 문제 난도 자체가 다르다.
- **결정적 차이 3 (도메인/모달리티)**: CAST는 텍스트 전용, 단일 hidden state stream. 우리는 VL(Eagle)/DiT(action head) 두 pathway가 직렬 결합된 VLA — CAST의 단일 condition vector는 우리에게 "pathway별 조건벡터"로 분해되어야 하고, multi-conditioning(OR/AND, §5 Fig8)은 "VL 조건 OR DiT 조건" 식 pathway 결합 규칙에 참고 가능.
- **재사용 가능한 것**: (1) grid search로 (layer, threshold, direction) 최적화하는 방법론 → 우리 online detector threshold 튜닝에 그대로 적용 가능. (2) duality(비교방향 뒤집기=여집합 개입) → 우리가 "성공 방향으로 밀기" vs "실패 방향에서 밀어내기"를 conceptor 축 부호만 바꿔 표현하는 것과 수학적으로 동일. (3) "behavior=국소 통계, condition=전역 통계" 구분 → 우리도 phase 식별용 conceptor(누적/전역 통계)와 steering용 conceptor(현재 시점/국소 통계)를 다른 통계량으로 fit해야 할 수 있음을 시사.
- **못 가져오는 것(=우리 기여 지점)**: 온라인·연속·시변 조건 감지 자체. CAST는 이 문제를 풀지 않고 1회 정적 분류로 우회했다.

## 면접 포인트(Q→A)

**Q1. CAST의 condition vector와 우리의 online phase/failure-type 식별기는 뭐가 다른가?**
A. 판정 메커니즘(hidden state를 조건벡터에 projection→cosine sim→threshold)은 원리적으로 유사하지만, CAST는 "프롬프트 시점 1회 정적 분류"이고 우리는 "rollout 매 스텝 동적 재평가"가 필요하다. CAST는 생성이 시작되면 조건을 다시 검사하지 않는데, 우리 문제는 정확히 "언제 phase가 바뀌었고 언제 실패가 시작됐는지"를 매 스텝 알아내야 하는 시계열 문제라 CAST 설계를 그대로 못 쓴다.

**Q2. CAST가 "무조건 steering은 못 쓴다"는 걸 어떻게 실증했나?**
A. 같은 refusal behavior vector를 조건 없이 더하면(AST) harmless 프롬프트 거부율도 90%대로 치솟는다(예: Zephyr 0.2%→94.8%). 조건부(CAST)를 걸면 harmful 거부율은 비슷하게 오르면서 harmless는 거의 그대로(6.8%)다 — "targeted intervention의 필요성"을 정량 대조로 보여준 것이 핵심 기여.

**Q3. duality/modulation이 우리 conceptor 설계에 어떻게 대응되는가?**
A. duality(비교방향 뒤집기=여집합 개입)는 C_steer 부호를 뒤집어 "성공쪽으로 밀기" ↔ "실패쪽에서 밀어내기"를 표현하는 것과 같은 축이다. modulation(θ로 조건 민감도 조절)은 우리 steering 강도나 온라인 detector 판정 threshold 조절과 대응한다 — 다만 CAST는 이를 정적 분류에, 우리는 시변 상태추정에 써야 하는 차이가 있다.

**Q4. CAST가 정말 우리 메인 방법에 가장 가까운 선행연구인가, 이름만 "conditional"이라 끌어온 건 아닌가?**
A. 수식 구조 자체(h' = h + f(sim(h,proj_c h))·α·v)가 "감지 함수 f와 개입 v를 분리하고 f로 v의 활성화를 게이팅한다"는 점에서 phase-matched steering의 골격과 1:1 대응한다. 다만 f가 다루는 조건의 시간성(정적 vs 동적)이 근본적으로 다르므로 "구조는 원형, 시간축 확장은 우리 기여"라고 구분해서 말해야 한다.

## 한계·비판

- 조건 감지가 이산적 텍스트/토픽 분류로 한정 — 대조쌍(D+/D-)이 명확히 다른 카테고리라는 전제. 연속·시변 상태(phase, 실패유형)에 같은 PCA-1st-PC 추출이 통할지 불확실(저자도 §A.2에서 "벡터 추출 방법 개선은 향후 연구"라 인정).
- 첫 forward pass에서 1회만 조건 체크 → 멀티턴/긴 생성 중 주제(우리 식 phase)가 바뀌는 경우 대응 불가, 스코프 자체에서 배제.
- 3개 하이퍼파라미터(threshold/layer/direction) grid search 필요, cosine sim 유효범위가 모델마다 크게 다름(Hermes2Pro 0.0~0.1 vs Zephyr 0.4~0.6) — 새 모델·조건마다 재탐색, 자동화 안 됨.
- 다중 조건 결합 시 성능 저하 관찰(§5, 억제조건과 유도조건 충돌 시 유도 효과 감소) — pathway별 조건(VL/DiT)을 동시에 걸 때도 유사 간섭 가능성 시사.
- 평가가 전부 텍스트 refusal 도메인 — continuous control(로봇 액션), diffusion action head로의 이전 가능성은 전혀 다루지 않음. "condition vector"가 action-chunk 단위로 어떻게 정의돼야 하는지도 미지수.
- threshold 선택이 train-set F1 최대화라는 사후적 최적화 — unseen에 대한 일반화는 §5 constraining 실험(unseen harm category)에서만 간접 검증, 실패 사례 심층분석은 부족.
