# Robots That Ask For Help: Uncertainty Alignment for LLM Planners (KnowNo) (Ren et al. 2023)

- 출처: RSS 2023 (arXiv v2, 2023.09) · arXiv:2307.01928v2 [cs.RO] (Princeton + Google DeepMind) · PDF: docs/references/KnowNo_AskForHelp_2307.01928.pdf · 섹션=§7 VLA방향으로 배정됐으나 본 논문 자체는 §1~§6 + Appendix A1~A9까지만 존재(§7 없음) — 본문 전체 정독, VLA 확장 관련 서술은 §6 Discussion의 "저수준 action policy 불확실성을 CP calibration에 통합"이라는 future-work 한 문장뿐 · tier=must · 한줄역할: conformal prediction(CP)으로 LLM planner의 다음-스텝 신뢰도를 정렬해 "언제 사람에게 도움을 요청할지"를 통계적 커버리지 보장과 함께 결정 — online 불확실성 검출의 통계적 프레이밍 선례.

## 문제·동기
LLM 기반 planner는 자연어 지시의 모호성(예: "그릇을 전자레인지에 넣어줘" — 금속 그릇/플라스틱 그릇 중 무엇인지)에도 confidently hallucinate하는 경향이 있어, 잘못된 계획을 그대로 실행하면 비효율을 넘어 위험(화재 등)까지 초래할 수 있다. 기존 LLM-planning 연구는 (a) 명확화 요청 자체를 하지 않거나(SayCan 계열) (b) 프롬프트만으로 애매하면 물어보게 하는데(Inner Monologue), 후자는 프롬프트 엔지니어링에 크게 의존하고 "도움 요청 시 목표 성공률을 실제로 달성한다"는 보장이 전혀 없다. 저자는 이를 uncertainty alignment 문제로 정식화: (i) calibrated confidence — 사용자가 지정한 성공률 1-ε을 통계적으로 달성, (ii) minimal help — 그 안에서 사람 개입을 최소화.

## 핵심 아이디어
next-step planning을 multiple-choice QA(MCQA)로 재구성한다: LLM이 먼저 의미적으로 다른 후보 plan 4개(A~D) + "E) 목록에 없음"을 생성하고, 그중 정답 하나를 고르는 next-token 확률을 신뢰도 점수로 쓴다. 이는 free-form 문장 확률이 길이에 민감해(길수록 확률↓) 나쁜 스코어링 함수가 되는 문제를 MCQA의 단일 토큰 예측으로 우회하는 트릭이다. 이 점수를 CP에 넣어 prediction set C(x)를 만들고, 집합이 singleton이면 그대로 실행, 아니면 사람에게 도움을 요청한다 — CP의 coverage 보장이 그대로 "task completion rate ≥ 1-ε" 보장으로 직결된다(도움 요청이 실패 확률을 흡수). 멀티스텝으로의 확장에서는 시퀀스 전체를 하나의 calibration 데이터포인트로 승격시키고, "전 스텝 중 최저 신뢰도"를 시퀀스 점수로 써서 시퀀스 레벨 CP를 적용한 뒤, 이 non-causal 계산 결과가 causal(각 시점에서 과거 정보만 사용) 재구성과 동치임을 증명(Claim 1)해 온라인 배포에서도 동일 보장이 유지되게 만든 것이 핵심 기여다.

## 방법 (conformal prediction, prediction set, coverage 보장, ask-for-help 트리거)
- **CP 배경**: calibration set N개 (x_i, y_i) i.i.d, nonconformity score κ_i = 1 - f̂(x_i)_{y_i}. 경험적 분위수 q̂ = ⌈(N+1)(1-ε)⌉/N quantile of {κ_i}. test에서 prediction set C(x_test) = {y ∈ Y : f̂(x_test)_y ≥ 1-q̂}. marginal 보장: P(y_test ∈ C(x_test)) ≥ 1-ε — 모델 f̂의 정확도와 무관하게, calibration/test가 exchangeable(i.i.d.)하다는 가정만으로 성립.
- **dataset-conditional 보장**(Vovk 2012 인용): 매번 새 calibration set을 뽑을 필요 없이 "이 특정 calibration set이 주어졌을 때"의 하한을 Beta 분포 분위수로 제공(N=400, δ=0.01) — 실전에서 재사용 가능한 단일 calibration set으로 배포 가능하게 함.
- **최소 prediction set 크기**: f̂가 true conditional probability를 모델링한다면, CP가 coverage 제약 하에서 평균 집합 크기를 최소화하는 최적 스킴이라는 정리(Sadinle et al. 2019, Thm.1)를 인용 — "minimal help" 목표를 이론적으로 뒷받침.
- **single-step 절차** (Proposition 1): calibration set 라벨링(사람이 정답 제공) → ε에 맞춰 q̂ 계산 → test에서 singleton이면 자동 실행, non-singleton이면 사람에게 prediction set을 자연어 질문으로 변환해 제시하고 선택받음.
- **multi-step 절차** (Proposition 2): 스텝 간 context가 이전 행동에 의존해 i.i.d.가 깨짐 → 시퀀스를 하나의 데이터포인트로 lift, 시퀀스 점수 f̂(x)_y := min_t f̂(x_t)_{y_t} 로 정의 → 표준 CP로 non-causal하게 시퀀스 레벨 집합 C(x_test) 계산 → 각 시점에서 동일 quantile q̂로 독립적으로 만든 causal 집합들의 곱집합 C̄(x_test)가 C(x_test)와 논리적으로 동치임을 증명(Claim 1: y∈C ⇔ y∈C̄) → causal 온라인 적용에도 시퀀스 레벨 task completion 보장이 그대로 유지됨.
- **multiple acceptable options 확장** (Proposition 3, 4): 정답이 여러 개인 스텝에서는 β 연산자로 정답 후보 중 f̂ confidence가 가장 높은 것을 골라 단일 정답처럼 취급한 뒤 동일 CP 절차를 적용.
- **ask-for-help 트리거**: prediction set 크기 >1이면 사람에게 물어봄(집합을 자연어 질문으로 변환, 일부는 LLM이 직접 질문 문장을 생성), =1이면 자동 실행. ε(목표 실패율)은 사용자가 다이얼처럼 조절 가능한 유일한 하이퍼파라미터.

## 실험·결과
- PaLM-2L 기본(추가로 PaLM-2L-IF, GPT-3.5 비교). 시뮬레이션(PyBullet tabletop rearrangement, attribute/numeric/spatial 모호성), 하드웨어 UR5 multi-step food sorting, 모바일 매니퓰레이터 주방(Winograd schema·안전성 모호성), Kuka bimanual.
- baseline: Simple Set(누적 threshold, coverage 보장 없음), Ensemble Set(20-shot 프롬프트 앙상블, 20배 연산비용), Prompt Set(LLM에 직접 집합 생성 요청), Binary(불확실 여부만 이진 출력), No Help.
- KNOWNO가 지정 성공률(1-ε)과 실제 달성률 사이 편차가 가장 작음(CP coverage 보장 덕분). Simple Set/Ensemble Set은 coverage를 일관되게 못 맞추고, Prompt Set/Binary는 애초에 ε을 조절할 방법이 없음.
- 하드웨어 multi-step food sorting: help율 step-wise 14%p, trial-wise 8%p 감소, 평균 prediction set 크기도 축소하면서 동일 목표 성공률 달성(Table 1). 모바일 매니퓰레이션: help율 14%p 감소(Table 2).
- LLM을 PaLM-2L-IF, GPT-3.5로 교체해도 coverage 보장은 유지됨 — GPT-3.5는 MCQA 옵션 편향(D/E 선호, A/B 기피)이 있음에도 CP가 도움 요청을 늘려서 목표 성공률을 여전히 달성. 즉 CP 보장은 f̂의 정확도·캘리브레이션 수준에 의존하지 않는다는 점을 실증.
- Ablation(Appendix A9): MCQA 대신 sentence-level raw probability/perplexity 점수를 쓰면 성능이 뚜렷이 나빠짐 — MCQA 재구성이 단순 길이-편향 제거를 넘어 LLM으로부터 더 정확한 confidence를 끌어내는 추가 이득이 있음을 시사.

## activation-steering 흐름 위치(검출/introspection 레이어)
이 논문 자체는 activation을 조작하는 steering이 아니라, LLM 출력(다음-토큰 확률) 레벨의 introspection·검출이다. steering 파이프라인 관점에서 보면 "언제 개입해야 하는가"를 판단하는 트리거/게이팅 레이어에 해당하며, 개입 방식은 latent 조작이 아니라 "사람에게 위임(ask-for-help)"이다. 우리가 벤치마크로 삼을 지점은 CP의 coverage-보장 threshold 설계 그 자체이지, activation을 어떻게 바꾸는지가 아니다 — 즉 이 논문은 검출·게이팅 축의 "통계적 보장" 선례로만 참고하고, latent steering 메커니즘 자체는 이식 대상이 아니다.

## 우리 프로젝트 연결(online 검출에 conformal 보장 이식 가능성)
- **threshold 설계 이식**: 우리 online phase/failure-type detector(예: N16 DiT block31, t_d=11에서 AUROC 0.92)는 지금 AUROC로 검증된 고정 threshold를 쓴다. KnowNo처럼 성공 rollout(calibration set)만으로 conformal quantile을 잡으면, "steer 미개입으로 인한 실패 누락률 상한 ≤ ε"이라는 통계적 보장을 다이얼처럼 조절 가능한 형태로 부여할 수 있다.
- **causal 재구성 트릭 이식**: Claim 1(non-causal calibration → causal 재구성해도 coverage 동치)은 phase-matched steering의 phase-bin별 독립 threshold 설계에 그대로 대응된다 — 전체 rollout(시퀀스)을 한 번에 calibrate하고, 온라인에서는 각 phase-bin마다 causal하게(과거 정보만으로) threshold를 적용해도 시퀀스 레벨 보장이 유지된다는 논리를 빌려올 수 있다. FIPER(스크래치패드 메모리 fiper-failure-prediction 참고)의 sliding-window conformal band와 계열은 같지만, KnowNo는 discrete decision(도움 요청 여부)에 특화해 이론이 더 tight하다.
- **핵심 차이 — ground-truth 라벨의 부재**: KnowNo의 calibration set은 매 샘플마다 사람이 직접 "정답 다음 스텝"을 라벨링한다. 우리는 succ/fail 이진 라벨만 있고 "왜 실패했는지(어느 pathway·어느 phase)"의 ground-truth 라벨이 없다. conformal 프레임을 온전히 이식하려면 pathway_step_attribution.py류의 귀인 결과를 proxy label로 삼아 calibration set을 구성해야 한다는 추가 작업이 필요하다.
- **이산 게이팅 설계의 선례**: "help 트리거 = prediction set 비singleton"이라는 이산적 게이팅 로직은, steer 강도를 연속으로 줄지 이산적으로 on/off할지의 설계 논쟁에서 "이산 게이팅 + 통계적 보장" 조합이 실전에서 작동함을 보여주는 선례로 참고 가능하다.

## 면접 포인트(Q→A; conformal prediction 개념)
1. Q: "conformal prediction이 정확히 뭘 보장하나?" A: "모델의 예측 정확도 자체가 아니라, '예측 집합이 정답을 포함할 확률'을 분포에 대한 가정 없이(model-agnostic, distribution-free) 보장한다. calibration set에서 nonconformity score의 경험적 분위수를 threshold로 잡으면, exchangeability(대체로 i.i.d.)만 가정한 채로 새 테스트 샘플에 대해 P(y_test ∈ C(x_test)) ≥ 1-ε이 성립한다 — 모델이 잘 맞든 못 맞든 통계적으로 보장된다는 게 핵심이고, 대신 모델이 부정확하면 집합 크기(help 비용)가 커지는 방식으로 그 부정확함을 흡수한다."
2. Q: "KnowNo가 multi-step(순차적) 상황에서 conformal prediction의 i.i.d. 가정 위반을 어떻게 해결했나?" A: "각 스텝의 context가 이전 행동에 의존해 i.i.d.가 깨지므로, 개별 스텝이 아니라 전체 시퀀스를 하나의 데이터포인트로 승격(lift)시키고, 시퀀스 점수를 '전 스텝 중 최저 신뢰도(min)'로 정의해 시퀀스 레벨에서 표준 CP를 적용한다. 이후 non-causal하게 계산된 시퀀스 집합이 causal(각 시점에서 과거 정보만 사용) 재구성과 논리적으로 동치임을 증명(Claim 1)해서, 온라인 배포 시에도 동일한 커버리지 보장이 유지되게 만들었다."
3. Q(우리 프로젝트 관점): "이 방법을 우리 VLA online steering 트리거에 어떻게 가져올 수 있나?" A: "지금 우리 phase/failure-type detector는 AUROC로 검증된 고정 threshold를 쓰는데, KnowNo처럼 성공 rollout만으로 conformal quantile을 잡으면 'steer 미개입 시 실패율 상한 ≤ ε'이라는 통계적 보장을 조절 가능한 다이얼로 만들 수 있다. 다만 우리는 KnowNo처럼 매 스텝 사람이 준 ground-truth 라벨이 없으므로, pathway attribution 결과를 proxy label로 써서 calibration set을 구성해야 한다는 차이가 있다."

## 한계·비판
- coverage 보장은 "환경(물체)이 텍스트 컨텍스트에 완전히 grounding되어 있고, LLM이 고른 action이 저수준 정책으로 성공적으로 실행된다"는 가정에 의존한다 — perception 오류나 저수준 제어 실패(모바일 매니퓰레이션 저수준 성공률 약 86%)는 CP 보장 범위 밖이며 저자도 §6에서 명시적 한계로 인정.
- 사람이 "항상 정확하게" 도움을 준다는 가정(human faithfully provides help)에 의존 — 사람의 실수·모델링은 프레임워크에 반영되어 있지 않다.
- calibration에 400개 i.i.d. 시나리오 라벨링이 필요(멀티스텝 약 4시간, 싱글스텝 약 1.5시간) — 새 배포 환경마다 재수집이 필요하고, 분포 변화(distribution shift)가 있으면 보장이 깨진다.
- MCQA 후보 생성 자체가 LLM few-shot prompt에 의존한다 — "진짜 정답"이 생성된 4개 후보에도 "E) 목록에 없음"에도 해당하지 않는 상황을 원칙적으로 보장하지 못한다(경험적으로는 잘 커버됨을 보였을 뿐).
- 실패의 원인(왜 애매한지, spatial/attribute/numeric 등 모호성 종류)을 진단하지 않는 이진적 트리거다 — ask-for-help는 "무엇이 잘못됐는가"에 대한 설명을 주지 않는다. KnowNo가 답하는 질문은 "애매한가?"이고 우리 프로젝트가 답해야 하는 질문은 "어느 pathway가 실패했는가?"로, 질문 자체의 층위가 다르다.
- GPT-3.5는 MCQA 옵션 편향(D/E 선호, A/B 기피)이 뚜렷했는데, CP는 이를 "도움 요청 증가"로만 흡수할 뿐 근본적인 LLM 편향 자체는 해결하지 못한다.
