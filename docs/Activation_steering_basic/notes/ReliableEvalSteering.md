# Towards Reliable Evaluation of Behavior Steering Interventions in LLMs (Pres 2024)

- 출처: arXiv:2410.17245v1 (2024-10-22) · NeurIPS 2024 Workshop on Foundation Model Interventions · Itamar Pres(Michigan/ERA Fellowship), Laura Ruis(UCL), Ekdeep Singh Lubana(Michigan/Harvard CBS), David Krueger(Cambridge)
- PDF: `docs/Activation_steering_basic/ReliableEvalSteering_2410.17245.pdf`
- §5 파트: 장벽(평가 부실) — 서베이 §5의 기존 두 틈새(안전 gate / interpretability API) 옆에 놓일 세 번째 축
- 3축: 안전 필터 / interpretability API / **평가 장벽(evaluation barrier)** — 이 논문은 세 번째 축을 채움
- 한줄 역할: steering이 "잘 듣는다"는 보고 대부분이 MCQ propensity 또는 정성적 생성 데모에 의존했고, open-context·likelihood·행동간 비교·baseline 4속성을 갖춘 파이프라인으로 재평가하면 CAA/ITI 효과가 원 보고보다 약하거나 방향이 비대칭(promote 실패·suppress만 성공)임을 실증 — steering이 QA/회귀 테스트 가능한 제품 표준이 되기 전에 반드시 넘어야 할 장벽을 정면 제시.

## 문제·동기
CAA/ITI 등 RepE 계열 방법의 성공 보고는 (a) MCQ 이지선다 propensity 비교 아니면 (b) LLM-judge/사람이 읽은 정성적 생성 샘플에 의존해왔다. Tan et al.(2024, 같은 서베이 TanSteeringReliability 참고)이 이미 "per-sample steerability가 요동친다"는 신뢰성 문제를 제기했지만 그 논문도 MCQ logit-diff 틀 안에 있었다. 이 논문은 한 단계 더 나아가 "평가 프로토콜 자체"의 결함을 지적한다: 같은 corrigible CAA 벡터가 MCQ에서는 성공하지만 동일 내용의 open-ended 생성에서는 실패하고(Table 1), 생성 텍스트만 보면 실패로 보이는 개입이 실은 top-2 토큰이 거의 동률이라 샘플링 시드에 따라 성공/실패가 갈릴 수 있다(Table 2-3 myopia 사례). 지금까지 보고된 steering 성공률의 상당수가 "무엇을, 어떻게 쟀는지"에 오염돼 있다는 것이 동기.

## 핵심 아이디어
평가가 갖춰야 할 4개 속성(Property 1-4)을 정의하고 이를 만족하는 단일 파이프라인을 제안한다.
1. Generalizability(개방형 문맥) — MCQ가 아니라 chat-format 프롬프트 + positive/negative continuation을 이어붙여 open-ended 배포 상황과 유사한 문맥에서 평가.
2. Consistency(모델 confidence) — 생성 텍스트 하나만 보지 않고 토큰 log-likelihood를 측정해 top-1 샘플링의 우연성을 배제.
3. Cross-behavioral comparability — 행동별 전용 어휘(예: wedding 키워드 빈도) 대신 positive/negative continuation likelihood라는 행동-불문 지표로 모든 behavior를 같은 척도에서 비교.
4. Baseline — intervened와 baseline(개입 없음) likelihood를 항상 나란히 비교, baseline이 이미 원하는 답을 선호하는 천장효과 구간은 제외.
파이프라인 절차: 프롬프트에 pos/neg continuation을 이어붙여 baseline·intervened 모델 양쪽에서 토큰 log-likelihood를 잰다. "가장 그럴듯한 negative"와 "가장 그럴듯하지 않은 positive"의 평균으로 두 모델의 likelihood를 재정규화하고, baseline 기준 오름차순으로 positive/negative를 각각 정렬해 산점도로 시각화(Figure 1b). 메트릭은 baseline이 가장 약하게 선호하는 구간(top 25/50/75%)에서 intervened−baseline 평균 log-likelihood 차이를 positive/negative 별도로 보고한다 — "promote"(positive 상승)와 "suppress"(negative 하강)를 분리해 보는 것이 기존 지표에 없던 축.

## 방법(평가 프로토콜 결여 진단)
- 데이터: Panickssery(CAA) 원 50개 open-ended 프롬프트 + GPT-4 생성 continuation으로 truthfulness/myopia/corrigible 등 데이터셋 구성.
- 모델: Llama-2-7B-Chat(CAA, layer13 벡터 ×2 스칼라, seed 42), ITI는 별도 체크포인트(likenneth/honest_llama2_chat_7B).
- 진단1(Property1 결여, Appendix D): 동일 corrigible CAA 벡터를 (i) MCQ 포맷 (ii) 동일 내용의 open-ended 포맷에 적용 — MCQ에서는 target behavior 유도 성공, open-ended에서는 실패(Table1). 지금까지 논문이 MCQ에서 잰 "성공"이 배포 상황(open-ended)으로 일반화 안 됨을 실증.
- 진단2(Property2 결여, Appendix E): myopia CAA 생성 텍스트만 보면 실패로 보이나(farsighted 응답), top-10 토큰 분포를 보면 myopic 토큰이 다수이고 top-1/top-2(각각 myopic/non-myopic)가 거의 동률(0.39/0.39) — nucleus sampling 시드에 따라 성공/실패가 바뀔 수 있음. 텍스트 하나만 보는 정성 평가는 이 variance를 놓친다.
- 진단3(Property3 결여): Turner의 wedding-word-frequency·perplexity 지표는 어휘가 명확한 topic behavior에만 통하고 truthfulness처럼 이지선다적 abstract behavior엔 적용 불가 — behavior 간 비교가 원천적으로 불가능.
- 진단4(Property4 결여): baseline 비교 없이 절대적 행동 강도만 보면, 원래도 그 행동을 잘 하던(천장효과) 모델과 개입으로 실제로 바뀐 모델을 구분 못한다.

## 실험·결과(재평가 시 효과 과장 확인)
Figure2·Table4로 ITI(truthfulness) + CAA(neg-hallucination, corrigible, myopia, hallucination, sycophancy)를 재평가:
- ITI truthfulness: top25%에서 positive +0.08 / negative −0.08 — 유일하게 both-direction이 뚜렷하게 성공.
- CAA neg-hallucination: negative 억제(top50% 0.07)는 강하지만 positive 증가는 미미(0.02) — "promote는 약하고 suppress만 강함" 비대칭, 논문이 강조하는 신규 관찰.
- CAA corrigible/myopia: 매우 erratic. myopia는 모든 샘플의 likelihood가 눌려 metric이 음수(positive: −0.02~−0.03)까지 나옴 — promote는 사실상 실패.
- CAA sycophancy: 전 구간 0.01 안팎으로 거의 무효과 — Panickssery 원논문이 조심스레 제기한 "sycophancy CAA가 truthfulness를 낮출 수 있다"는 가설을 이 pipeline에서는 사실상 기각.
결론: 동일 CAA/ITI를 4속성을 갖춘 프로토콜로 재면 다수 behavior에서 원 보고보다 약하거나 promote/suppress 중 하나로 치우친 비대칭 효과만 남는다 — "steering이 잘 듣는다"는 서사가 평가 방식에 크게 의존했음을 정량적으로 확인.

## §5(산업)에서의 위치(평가 미성숙=채택 장벽)
서베이 §5는 현재 "안전 gate(Circuit Breakers, Constitutional Classifiers)"·"interpretability API(Ember, Gemma Scope)" 두 틈새로 구성돼 있다. 이 논문은 그 옆에 세 번째 축 — "평가 장벽"을 채운다. 산업이 steering을 제품에 QA/회귀 테스트 대상으로 채택하려면 표준화·재현 가능·baseline 대비 정량 지표가 필요한데, 현재 필드는 (i) MCQ propensity(배포 문맥과 불일치) (ii) 정성적 LLM-judge/사람 판정 생성 샘플(confidence 무시) (iii) behavior 전용 지표(교차 비교 불가) (iv) baseline 부재 중 최소 하나를 결여한다. 이는 Circuit Breakers/Constitutional Classifiers가 안전 gate로 배포될 때조차 성공 기준이 narrow red-team ASR 벤치마크에 갇혀 있고, 범용 steering API(Goodfire Ember)가 select-partner로 축소된 상황과 직결된다 — 회귀 테스트 스위트를 못 갖춘 채로는 "성능이 유지된다"를 방어할 수 없기 때문이다. 즉 §5의 두 틈새가 "왜 더 넓어지지 못하는가"에 대한 원인 진단이 이 논문의 위치.

## 우리 프로젝트 연결(우리 ΔSR·baseline 규율과 대비)
- 우리 project는 이미 이 논문이 요구하는 4속성을 로봇 도메인 형태로 강제하고 있다: (1) open-ended 대응물 = MCQ가 아니라 실제 env rollout 실행, (2) confidence 대응물 = 단일 demo가 아니라 N_EP=20 다수 episode SR 평균(표본분산까지 관측), (3) cross-behavior comparability 대응물 = 모든 조건을 ΔSR 단일 스칼라로 비교하는 사다리식 ablation(global→pathway-split→+phase-bin), (4) baseline 비교 = EVAL_SEED=100000 고정 하 조건-쌍 ΔSR — Property4와 정확히 동형.
- 이 논문의 "promote vs suppress 분리" 통찰은 우리에게도 적용 가능: 우리 steering이 SR을 올릴 때 그게 baseline에서도 이미 성공하던 episode를 더 굳힌 것(promote)인지, baseline 실패 episode를 구제한 것(suppress/rescue)인지 구분해야 진짜 인과 효과를 주장할 수 있다. episode-level 페어링(같은 EVAL_SEED, steer on/off) 없이는 우리도 이 논문이 지적한 착시(easy episode SR만 오른 것)에 빠질 위험이 있다.
- Appendix D의 corrigible MCQ vs open-ended 불일치는, COAST/NOTALL 등 다수 VLA steering 선행연구가 여전히 offline latent-space 분리도(AUROC)나 짧은 정성 데모로만 성공을 보고하는 것과 같은 함정이다 — 우리가 "steer가 듣나"를 latent 분리도가 아니라 실제 ΔSR 인과 재측정으로 판정한다는 규율(project direction 문서)의 직접적 근거.
- 반면 이 논문의 한계도 우리에게 그대로 적용된다: 우리 SR은 episode 단위 이산 성공/실패라 이 논문이 강조하는 "연속 likelihood/confidence" 축이 없다. action-level에서 목표물과의 거리·phase 도달 여부 같은 soft margin을 보조 지표로 병행하면 Property2(confidence) 정신을 로봇 도메인에 이식할 수 있다.

## 면접 포인트(Q→A)
1. Q: "steering 평가에서 가장 흔한 함정이 뭔가?" A: "MCQ에서 잰 성공이 open-ended 배포 상황으로 일반화된다는 암묵적 가정이다. Pres et al.(2024)은 동일 corrigible CAA 벡터가 MCQ에서는 성공, 동일 내용의 open-ended 생성에서는 실패함을 실증했다."
2. Q: "정성적(사람이 읽은) 생성 샘플 평가가 왜 부족한가?" A: "top-1 샘플링 결과만 보면 성공/실패로 보이지만, myopia 사례에서 top-2 토큰이 거의 동률(0.39/0.39)이라 랜덤 시드에 따라 뒤집힐 수 있었다. 텍스트 하나로 판정하면 이 variance를 놓친다."
3. Q: "이 논문이 재평가해서 뒤집은 결과가 있나?" A: "CAA sycophancy는 전 구간 0.01 안팎으로 거의 무효과가 나와 Panickssery 원논문의 '아첨이 truthfulness를 낮출 수 있다'는 가설을 사실상 기각했고, corrigible/myopia CAA도 promote(positive 상승)는 실패하고 suppress(negative 하강)만 작동하는 비대칭이 드러났다."
4. Q(우리 프로젝트): "우리 평가는 이 논문의 기준을 만족하나?" A: "EVAL_SEED 고정 하 조건별 ΔSR 비교, 사다리식 ablation으로 단일 스칼라 교차비교, N_EP=20 다수 episode로 이산 confidence를 확보 — Property1/3/4는 로봇 실행이라는 형태로 이미 만족한다. 다만 Property2(연속 confidence)에 대응하는 soft-margin 지표는 아직 없어 보완 여지가 있다."
5. Q: "이 논문의 지표는 만능인가?" A: "아니다. 저자 스스로 top-10 정도만 보고 전체 next-token 분포는 다 안 본다고 인정했고, GPT-4로 만든 continuation이 대상 모델(Llama2 7B)엔 OOD일 수 있다는 한계도 남긴다."

## 한계·비판
- next-token 분포를 top-10 정도만 확인하고 전체 분포를 다루지 않음 — 저자 스스로 Appendix B에서 명시한 한계이자 Property2를 완전히 만족하지 못한다는 자기모순.
- continuation을 GPT-4로 생성해 대상 모델(Llama-2-7B-Chat)에는 OOD일 수 있음 — Property1(open-ended context 재현) 자체가 완전히는 지켜지지 않음.
- NeurIPS 2024 workshop 논문(본문 6쪽)으로 스케일이 작다: 모델 1개(Llama-2-7B, ITI만 별도 체크포인트), behavior 5-6개, 개입 방법 2개(CAA, ITI)뿐 — Tan et al.(2024)의 40개 MWE 데이터셋·4모델 규모에 비해 일반성 근거가 얕음.
- 메트릭이 "top 25/50/75% 최약체 구간"이라는 임의의 컷을 쓰는데, 이 컷이 결과 해석(예: myopia negative 강함/positive 약함)에 얼마나 민감한지 sensitivity 분석이 없음.
- 코드·데이터셋 공개가 "review 종료 후"로 예정(Appendix A)이라 arXiv v1 시점엔 재현 불가 — reproducibility를 요구하는 논문 자신이 재현 가능성을 아직 못 갖춤.
- VLA/로봇처럼 연속·시계열 action space에는 이 pipeline(이산 토큰 log-likelihood 기반)이 직접 적용되지 않는다 — 우리가 차용하려면 "토큰"을 "action chunk" 또는 "phase 도달"로 재정의하는 추가 설계가 필요하다.
