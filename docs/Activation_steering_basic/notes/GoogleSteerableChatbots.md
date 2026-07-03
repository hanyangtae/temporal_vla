# Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering (Bo et al. 2025)

- 출처: arXiv:2505.04260 (v2, 2025-05-13) · Jessica Y. Bo(Univ. of Toronto, Google Student Researcher로 수행) + Tianyu Xu, Ishan Chatterjee, Katrina Passarella-Ward, Achin Kulshrestha, D Shin(전원 Google AR 소속)
- PDF: docs/Activation_steering_basic/GoogleSteerableChatbots_2505.04260.pdf
- §5(산업 적용) 파트: 개인화(personalization) 응용 + 배포 현실(실사용자 n=14 HCI 유저스터디) — steering을 소비자 대면 챗봇 인터페이스로 프로토타입한 흔치 않은 사례
- 3축: 쓰기(write, additive activation steering) · 연구(research prototype — 학회 preprint + 유저스터디, 상용 배포 근거는 없음) · inference-time(추론 시 스칼라 강도로 residual stream에 개입, 백본 재학습 없음)
- 한줄역할: LLM 개인화를 "steering 강도"라는 단일 선형 파라미터로 노출하고 이를 3종 인터페이스(직접 슬라이더/페어와이즈 캘리브레이션/대화 중 자동학습)로 감싸 실사용자 반응까지 측정 — 산업 응용에서 steering을 "얼마나 세게 개입할지"를 최종 사용자/시스템에 넘기는 UX 패턴의 사례.

## 문제·동기

LLM은 RLHF로 "평균 사용자"에 맞춰 학습되므로, 개인별 선호(예산 vs 럭셔리, 관광지 vs 힙스터 동네 등)가 크게 갈리는 라이프스타일 플래닝 과제(선물 고르기, 여행 계획, 레스토랑 추천)에서 어긋나기 쉽다. 비전문 사용자는 프롬프트 명세 능력이 약하고, 자신의 선호를 모집단 평균 대비 상대적으로 인지하기도 어렵다("나는 평균보다 얼마나 럭셔리를 좋아하나?"). 기존 개인화는 memory/RAG(과거 발화 검색 — cold-start 불가) 또는 finetuning/RLHF(재학습 비용) 중심인데, steering은 재학습 없이 저비용으로, 선형 강도 인자 하나로 세밀하게 조절 가능한 대안으로 제시된다.

## 핵심 아이디어

선호 차원 d = {cost, ambiance, age, time, culture} 각각을 양/음 trait의 연속 강도로 파라미터화하고, 사용자 선호 벡터 d^u를 h_steered = h + d^u · v 형태로 residual stream에 주입한다(steer(d^u)를 통해 M(x, steer(d^u)) 출력을 사용자 맞춤화). steering vector v_i는 GPT-4o로 합성한 대조 데이터(각 trait 50-80개, 실제 챗봇 응답 포맷 모사)에 layer-wise logistic probe를 학습해 계수를 방향으로 취한다(von Rütte et al. 2024 방식 계승 — mean-diff/PCA도 비교했으나 probe 채택). top-k layer(모델별 16-32개 층)도 probe accuracy 기준으로 선택. 평가는 LLM 출력 임베딩과 Yelp 리뷰 positive/negative 코퍼스 평균 임베딩의 상대 코사인 유사도(Effect)에 perplexity로 정규화한 PNE(품질 저하 통제)를 더한 것.

## 방법(preference-based activation steering, 강도 조절)

- 5개 오픈소스 LLM(1.6B~9B: StableLM2, Gemma2-2B/9B, Mistral-7B, Qwen2.5-7B)에 동일 절차 적용, 계산실험 E1-E4 수행.
- E1(강도 스윕, -30~30): "functional steering range" 내에서 강도-효과가 근사 선형(품질 저하 없이). 범위는 모델별로 다름(민감한 모델은 ±10).
- E2(프롬프트 상호작용): 프롬프트 기반 선호("나는 아침형 인간이다")와 steering이 가산적으로 상호작용 — steering이 대화의 기본 맥락을 설정하고 프롬프트가 그 위에 변주를 더함.
- E3(다중 선호): 상관 최소 두 차원(culture, age)을 동시 steering, 가중합으로 적용 → 성능 저하 크지 않게 결합 가능.
- E4(숨은 선호 학습): GPT-4o-mini가 극단적 숨은 선호(budget 100%/luxury 100%)를 가진 사용자를 역할극, 감성 기반 학습 알고리즘 d^{u*}_{t+1} = d^{u*}_t + p(dissatisfaction(x_t)·direction(x_t))이 12턴 내 대체로 올바른 강도로 수렴.
- 유저스터디 인터페이스 3종 + PROMPT 베이스라인: SELECT(슬라이더 직접 조작), CALIBRATE(A/B 페어와이즈 비교 2-3회로 강도 이분탐색 수렴), LEARN(대화 중 감성으로 강도 자동 추론, 투명성 위해 학습된 %를 사용자에 노출).

## 실험·결과

- User study n=14(within-subjects), gemma-2-9b-it 백엔드(Gradio + Colab L4 GPU), 4개 과제(선물/여행/레스토랑/레시피) × 4개 인터페이스.
- U1(steering 강도 대 실제 표현된 선호 상관): SELECT r=0.54, LEARN r=0.40, 전체 steering 결합 r=0.37(모두 p<0.001) — CALIBRATE만 r=0.04, p=0.78로 무의미(2-3회의 시간 제약 캘리브레이션이 원인으로 추정).
- U2(표현된 선호 대 참가자 실제 ground-truth 선호 상관): SELECT r=0.46, CALIBRATE r=0.59(모두 p<0.001) — LEARN r=0.14(p=0.20, 비유의), PROMPT r=0.03(p=0.84, 사실상 무관) — steering 전체 결합 r=0.38(p<0.001)로 프롬프트 단독보다 유의하게 우수.
- 강도값 자체의 정합: SELECT가 최저 오차(MAE 27.8±31.9)·최고 방향 일치율(86%), CALIBRATE 유사 일치율·더 큰 오차(41.4±34.6), LEARN 최저 일치율(57%)·최대 오차(50.1±40.9).
- 주관 평가: 개별 조건은 PROMPT 대비 대체로 유의차 없음(CALIBRATE만 Satisfaction·Persistency에서 유의하게 높음) — 그러나 참가자별 "선호 인터페이스"로 묶으면 전 항목에서 PROMPT 대비 강한 유의차. 선호 인터페이스 투표는 SELECT 6, LEARN 5, CALIBRATE 3으로 개인 가치관(통제감/사용편의/투명성)에 따라 극명히 갈림.

## §5(산업)에서의 위치(개인화 응용, 연구 vs 제품)

저자 6인 중 5인이 Google AR 소속(1저자는 Google 재직 중 Student Researcher로 수행)이라는 점에서 §5 산업 근거로 무게가 있지만, 논문 자체는 ACM 학회 preprint 형식("Preprint, 2024" placeholder venue)이며 실제 제품 출시·배포 근거는 없다 — 3축 상 "연구(research)"에 해당하고, Circuit Breakers(RepE를 실제 안전 gate로 배포한 스펙)나 Anthropic Constitutional Classifiers처럼 "이미 배포된" 사례와는 위상이 다르다.

산업 적용 각도가 안전(harm 억제)이 아니라 개인화(personalization)라는 점에서, 서베이 §5의 기존 두 갈래(안전 gate=Circuit Breakers/Constitutional Classifiers, interpretability 인프라=Goodfire Ember/Gemma Scope)와 구분되는 세 번째 각도 — "steering strength를 사용자 대면 UX 컨트롤로 노출"하는 제품 설계 패턴을 §5에 추가한다. 논문이 직접 주장하는 산업적 강점은 (1) 저비용 추론(가중치 미변경, 온디바이스·AR/VR 후보로 저자들이 명시), (2) 개인식별정보 없이 사용자를 표현(steering strength 벡터 자체가 PII를 담지 않음 — RAG/finetuning 프로파일 대비 프라이버시 이점).

동시에 산업 배포의 미성숙함도 드러난다. 계산실험(E1-E4)은 통제된 재현 가능 결과지만, 실사용자 반응(U1-U4)은 노이즈가 커 CALIBRATE의 강도-표현 상관이 무너지고(r=0.04), 캘리브레이션·학습 알고리즘 모두 저자 스스로 "proof-of-concept 수준"(SOTA 분류기·베이지안 최적화가 아닌 단순 구현)이라 명시한다 — 서베이가 강조하는 "범용 steering 제품화는 아직 어렵다"는 §5 교훈의 또 다른 실증 사례.

## 우리 프로젝트 연결

- "선형 강도 인자 d로 steering 세기를 조절"하는 프레임은 우리 conceptor 기반 steering(h' = h·Mᵀ)에도 적용 가능한 별도 축이다 — 우리도 개입 방향(C_steer)뿐 아니라 강도 스칼라를 노출·스윕하는 실험을 사다리식 ablation에 추가할 여지가 있다.
- 더 직접적인 연결은 LEARN 인터페이스의 온라인 강도 추정 알고리즘(식 1, 감성 기반 업데이트) — "대화 중 latent 신호를 읽어 개입 강도를 실시간 조정"하는 구조로, 우리의 "online phase/failure-type 식별 → steering 라우팅" 문제와 구조적으로 동형이다. 다만 신호원이 본질적으로 다르다: 이 논문은 사용자 텍스트 감성(외부에서 직접 관측 가능)을 쓰지만, 우리는 VLA 내부 activation(직접 관측 불가, 별도 검출기로 추론해야 함)을 신호로 써야 한다.
- CALIBRATE(사전 페어와이즈 캘리브레이션, 대화 시작 전 고정)가 LEARN(대화 중 실시간 추정)보다 오히려 실사용자 선호와 상관이 더 높았다(r=0.59 대 0.14)는 결과는, 정교한 online 신호보다 간단하지만 안정적인 사전 phase-bin 캘리브레이션이 실전에서 더 신뢰할 수 있다는 가능성을 시사한다(단, 텍스트-대화 도메인에서 로봇 rollout 도메인으로의 전이는 별도 검증 필요).
- 상관 최소 두 축(E3)을 동시 steering해도 가산적으로 잘 결합된다는 결과는, 우리 VL/DiT pathway 분리 steering을 "두 방향을 각각 다른 강도로 동시에 가산"하는 설계로 볼 때 참고할 수 있는 선례다(다만 이 논문은 5개 선호 전부가 아니라 최소상관 2개 쌍만 검증해 일반화는 제한적).

## 면접 포인트(Q→A)

Q1. 이 논문에서 "steering" 자체는 얼마나 새로운 방법인가.
A. 기술적으로는 단순하다. 대조 데이터에 layer-wise logistic probe를 학습해 그 계수를 steering vector로 쓰고, additive(h + d·v)로 residual stream에 더하는 von Rütte et al.(2024) 절차를 그대로 가져다 썼다. 방법론적 기여는 크지 않고, 논문의 기여는 이를 "개인화 UX 문제"에 적용해 3종 인터페이스로 구현하고 실사용자 반응까지 측정한 응용·HCI 쪽에 있다.

Q2. Google 소속이라는 점이 왜 §5(산업 적용) 근거로 의미가 있나.
A. 저자 대부분이 Google AR 소속 연구자이고, 문제 설정 자체가 온디바이스/AR·VR 개인화 어시스턴트를 겨냥한다고 논문이 명시한다. 즉 학계 단독 연구가 아니라 실제 소비자 제품을 만드는 조직이 steering을 개인화 UX 후보로 진지하게 검토한 흔적이다. 다만 논문은 학회 preprint 단계이고 실제 Google 제품에 반영됐다는 근거는 없어 "제품"이 아니라 "연구" 축에 머문다.

Q3. CALIBRATE가 LEARN보다 실제 선호와 더 잘 맞은 이유는, 우리 프로젝트에 어떤 시사점이 있나.
A. LEARN은 실시간 텍스트 감성 분류(TweetEval)로 방향·강도를 추정하는데, 시간 제약(≤5분, 몇 턴) 속 대화에서는 노이즈가 커 실제 선호와 상관이 낮았다(r=0.14, 비유의). 반면 CALIBRATE는 대화 시작 전 A/B 페어와이즈 비교 2-3회로 강도를 이분탐색해 고정하므로 노이즈에 덜 취약했다(r=0.59). 다만 CALIBRATE는 이후 대화 중 사용자가 마음을 바꿔도 재조정이 안 되는 경직성이 있다고 저자도 지적한다. 우리 프로젝트에서 "online phase 신호"를 정교하게 실시간 추론하려는 시도가, 오히려 "rollout 특정 시점에 한 번 판별해 고정 적용"하는 것보다 노이즈에 약할 수 있다는 경고로 읽을 수 있다.

Q4. 이 논문이 산업 적용 가능성에 대해 스스로 인정하는 한계는 무엇인가.
A. 계산실험은 통제됐지만 유저스터디의 캘리브레이션·학습 알고리즘은 "proof-of-concept" 수준이라 SOTA 대비 정밀도가 낮다고 저자가 명시한다. 실사용자 대화는 계산실험보다 노이즈가 훨씬 크고(CALIBRATE의 U1 상관 붕괴), 사용자가 선호를 자주 바꾸며 지속성(persistency)에 대한 의견도 참가자 간 극단적으로 갈린다. "steering이 개인화에 효과적"이라는 결론은 통제된 조건에서는 강하지만, 실제 배포 시 안정성·강건성은 아직 증명되지 않았다.

## 한계·비판

- steering vector 학습 데이터가 GPT-4o 합성 텍스트에 의존한다 — 실제 인간이 쓴 "선호별 이상적 챗봇 응답" 데이터가 없어 대리 데이터로 학습했고, 평가도 Yelp 리뷰(실사용자 언어지만 챗봇 응답 포맷과는 다른 도메인)와의 코사인 유사도라는 간접 지표에 의존한다. 벡터 생성과 평가 코퍼스 간의 도메인 격차를 논문이 완전히 통제하지는 못한다.
- 5개 선호 차원·1개 태스크 도메인(라이프스타일 플래닝)·Yelp 코퍼스에 국한 — 창작, 기술 커뮤니케이션, 페르소나 등 저자 스스로 언급한 더 넓은 개인화 영역으로의 일반화는 미검증.
- CALIBRATE/LEARN 알고리즘 모두 단순 구현(평균 이분탐색, 사전학습 감성분류기 임계값 조합)이라 성능 상한을 논문이 실제로 보여주지 못했을 가능성이 있음을 저자도 인정.
- n=14 소규모 within-subjects 유저스터디, 60분 제한 세션 — 장기 지속 사용(persistency)에 대한 실측이 없고, 참가자 자기보고 ground-truth 선호도 실제로는 과제 중 자주 바뀌는 것으로 관찰돼 "정답" 자체가 불안정하다.
- 텍스트 LLM 챗봇 도메인 전용 — 연속 제어(action)·다단계 rollout·"성공/실패"가 이산 라벨이 아닌 우리 VLA 세팅으로의 전이는 검증되지 않았다. 특히 "강도 인자를 최종 사용자에게 그대로 노출"하는 UX 패턴(SELECT/CALIBRATE)은 로봇 조작처럼 실시간·안전 critical한 세팅에서 그대로 이식하기 어렵다 — 우리가 필요로 하는 것은 사람이 수동 조절하는 슬라이더가 아니라 phase/failure-type이 시스템 내부에서 자동으로 판별되는 구조이기 때문이다.
