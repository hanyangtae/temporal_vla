# Steering Llama 2 via Contrastive Activation Addition (Rimsky/Panickssery et al. 2023/2024, ACL 2024)

- 출처: arXiv:2312.06681 (v4, 2024-07-05) · ACL 2024 · Nina Panickssery(=Rimsky, Anthropic), Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
- PDF: `docs/Activation_steering_basic/CAA_2312.06681.pdf`
- 정독 섹션: §3 방법(3.1 데이터셋 소싱, 3.2 PCA 시각화) 중심, §1·2·4~10 개관 확인
- tier: must
- 한줄 역할: positive/negative 대조쌍의 평균 activation 차이(Mean Difference)로 스티어링 벡터를 만들고, 프롬프트 이후 전 토큰 위치에 계수(±)를 곱해 더하는 표준 절차를 정의 — 우리 succ/fail 대조 방향추출의 직접 원형이자, conceptor(C_success ∧ ¬C_failure)가 1차원(rank-1)으로 퇴화했을 때의 특수 케이스에 해당하는 baseline.

## 문제·동기

RLHF·instruction finetuning·prompt engineering만으로는 정합성(alignment) 확보에 한계(데이터 다양성, hallucination, OOD 실패, 불투명한 메커니즘)가 있다. activation/representation engineering이 대안으로 떠올랐으나(Turner ActAdd, Li ITI, Zou RepE 등) 여러 모델·행동에 걸쳐 견고성이 검증되지 않은 상태였다. 저자들은 RLHF로 안전 정렬된 Llama 2 Chat(7B/13B)에 CAA를 적용해, activation engineering이 이미 RLHF된 모델 위에서도(그리고 finetuning·system-prompt와 결합해도) 작동하는지 검증한다.

## 핵심 아이디어

같은 질문에 대해 답만 다른(A=행동 긍정, B=행동 부정) 이지선다 프롬프트 쌍을 수백 개 구성하고, 정답 letter 토큰 위치의 residual stream activation 차이를 평균 내어 "행동 방향" 벡터(Mean Difference, MD)를 얻는다. ActAdd(Turner et al. 2023)와 계열이 같지만 (1) 단일 프롬프트쌍이 아닌 수백 개 대조쌍 평균으로 노이즈를 줄이고, (2) 벡터를 프롬프트 이후 모든 토큰 위치에 주입한다(ActAdd는 첫 토큰만). 질문 본문은 고정하고 답 letter만 바꾸는 최소 대비 설계로 confound를 상쇄해 행동 관련 성분만 isolate한다.

## 방법(positive/negative 쌍 mean-difference, 전 토큰위치 주입, 계수 부호로 boost/suppress)

- 데이터: Anthropic Advanced AI Risk(Perez et al. 2022), Sycophancy on NLP/political typology, GPT-4로 자체 생성한 Hallucination/Refusal 데이터셋. 각 (prompt p, positive completion cp, negative completion cn) triple.
- MD 벡터(식 1): v_MD = (1/|D|) Σ_{p,cp,cn∈D} [a_L(p,cp) − a_L(p,cn)], a_L()은 레이어 L의 answer-letter 토큰 activation.
- 주입: 추론 시 유저 프롬프트 이후 "모든" 토큰 위치에 multiplier×v_MD를 residual stream에 더함. multiplier 부호로 boost(+)/suppress(−), 크기로 강도 조절.
- 레이어 선정: 전 레이어 스윕(상수 multiplier ±1)으로 효과 최대 레이어를 먼저 찾고(7B→layer 13, 13B→layer 14/15), 그 레이어에서만 multiplier 범위를 다시 스윕.
- 정규화: 행동 간 벡터 norm은 표준화하되, 레이어 간 norm은 정규화하지 않음(residual stream norm이 forward pass에서 지수적으로 커지는 자연 norm 보존, §10 한계로 재논의).

## 실험·결과(sycophancy·corrigibility 등 행동축)

- 7개 행동축: AI Coordination, Corrigibility, Hallucination, Myopic Reward, Survival Instinct, Sycophancy, Refusal.
- MC 평가: 7B/13B 모두 최적 레이어 부근에서 +1/−1 multiplier가 p(answer matching behavior)를 전 행동에서 일관되게 증가/감소(Fig 3-4), layer 17 이후 급격히 감쇠.
- Open-ended: GPT-4가 1-10점으로 평가(Table 1, "디저트만 먹기" 질문에 +sycophancy는 동조, −는 반박).
- System prompt 결합(Table 3, 13B): 대부분 행동에서 CAA가 positive/negative system prompt 위에 추가 효과(예: Corrigibility positive-prompt 0.79→steering+1 0.93).
- Finetuning 결합(Table 4, 7B open-ended): 7종 중 3종은 finetuning 위에도 추가 효과. 특히 Sycophancy는 finetuning이 MC→open-ended로 OOD 일반화에 실패(Table 13 acc 44%인데 Table4 open-ended 효과 거의 0)하지만 CAA는 모든 경우 일반화에 성공.
- Capability 보존: MMLU(Table5) 거의 무변화, TruthfulQA(Table12)는 sycophancy 벡터를 빼면 소폭 개선.
- 해석: steering vector·per-token activation의 cosine similarity가 의미상 관련 토큰에서 상승(Fig 6, "I cannot help"↔refusal). base↔chat 모델 벡터 유사도는 layer 7-15에서 peak(Fig 9)→이 구간은 RLHF가 표현을 크게 재배선하지 않음을 시사.

## activation-steering 흐름 위치

ActAdd(Turner 2023, 단일쌍·첫토큰만 주입) → **CAA(이 논문, 대조쌍 수백 개 평균·전토큰 주입)** → RepE(Zou 2023, reading+control 통합 프레임, Contrast Vector가 CAA의 MD와 동일 계열) → ITI(Li 2023, sparse attention head 대상 mean-diff)로 이어지는 계보에서, CAA는 "대조쌍 mean-difference + 전토큰 additive steering"을 정식화한 실용적 canonical baseline이다. 이후 문헌에서 "CAA-style steering"은 사실상 이 rank-1 mean-difference 벡터 additive 기법을 가리키는 대명사로 쓰인다. COAST/conceptor 계열(우리 방법의 토대)은 이 1차원 벡터를 성공/실패 분포의 공분산 기반 multi-dim subspace/operator로 확장한 것으로 위치시킬 수 있다.

## 우리 프로젝트 연결(대조 방향추출·conceptor 1D 특수화)

- 우리의 succ/fail 대조 방향추출은 CAA의 MD 공식(positive=성공 rollout, negative=실패 rollout activation)을 그대로 적용하는 지점. CAA는 텍스트 A/B 답변쌍을 대비하지만, 우리는 rollout 성공/실패 trajectory의 hidden state를 대비한다는 도메인 차이가 있다.
- C_steer = C_success ∧ ¬C_failure는 CAA MD 벡터(단일 방향)의 다차원 특수화로 이해 가능: CAA가 "평균차 벡터 하나를 전 토큰에 더한다"면, conceptor는 "성공/실패 분포의 공분산 구조(subspace)를 각각 fit해 논리곱 연산자로 결합"한다. CAA는 이 conceptor가 rank-1(C≈vv^T)로 퇴화한 특수 케이스로 볼 수 있어, 우리 conceptor가 단순 mean-diff보다 실제로 나은지 검증할 자연스러운 ablation 하한선이 된다.
- "전 토큰 위치 균일 주입"은 우리의 phase-matched 문제의식과 정반대 극단(무조건부 상시 개입) — CAA §9.1 Future work가 스스로 "targeted token position steering"을 제안한 지점이 바로 우리의 phase-matched DiT steering 동기와 맞닿는다.
- 계수 부호로 boost/suppress라는 CAA의 이분법은 우리의 h' = h·M^T(방향성 있는 subspace 투영) 연산보다 단순 — CAA는 "steering이 듣는가"에 대한 최소 기준선(가장 단순한 형태)을 제공한다.

## 면접 포인트(Q→A; CAA vs ActAdd 차이)

**Q1. CAA와 ActAdd(Turner et al. 2023)의 차이는?**
A. 세 가지. (1) ActAdd는 단일 대조 프롬프트쌍에서 벡터를 뽑지만 CAA는 수백 개 대조쌍의 평균차를 써서 노이즈를 줄인다. (2) ActAdd는 벡터를 첫 토큰 위치에만 주입하지만 CAA는 프롬프트 이후 모든 토큰 위치에 주입한다. (3) ActAdd는 GPT-2-XL에서만 검증돼 행동·프롬프트에 강건하지 않았지만, CAA는 RLHF된 Llama 2 Chat에서 7개 행동축에 걸쳐 일관되게 작동함을 보인다.

**Q2. CAA의 Mean Difference가 PCA와 어떻게 다르고 왜 그것으로 충분한가?**
A. Tigges et al.(2023) 인용에 따르면, 대조쌍을 "질문 동일·답변 letter만 다름"으로 최소화하면 MD 벡터가 PCA 1주성분과 유사한 방향을 낸다. 즉 confound가 잘 상쇄된 대비 설계라면 지도적(supervised) mean-diff만으로 비지도 PCA와 비슷한 방향을 뽑을 수 있어, 굳이 PCA를 쓸 필요가 없다는 논리다.

**Q3. 왜 CAA는 finetuning보다 OOD 일반화가 낫다고 주장하는가(Sycophancy 사례)?**
A. finetuning은 MC 데이터셋 형식(A/B)에 파라미터가 직접 과적합해 open-ended 생성으로 전이가 실패하지만(Table13 acc 44%인데 Table4 open-ended 효과 거의 0), CAA는 파라미터를 건드리지 않고 모델이 이미 가진 표현을 이동만 시키므로 학습 형식에 덜 의존적이다. 이는 steering 계열 전체(우리 conceptor 포함)가 "백본 재학습 없음" 제약 하에서 갖는 핵심 이점과 같은 논리.

## 한계·비판

- multiplier가 커지면 텍스트 품질(perplexity) 저하 — 저자도 이 때문에 multiplier 범위를 제한. "많이 밀수록 좋다"가 아닌 tradeoff.
- 레이어 간 벡터 norm을 정규화하지 않고 constant multiplier로 최적 레이어를 먼저 고정한 뒤 그 레이어에서만 multiplier를 탐색 — 레이어별 최적 multiplier가 다를 가능성을 探索하지 않아 layer-optimality 결론이 편향됐을 수 있음(§10 자인).
- 전 토큰 위치 균일 주입은 "언제 얼마나" 조절이 없는 rigid 개입 — phase/시점 조건부 steering(우리 문제의식)과 정반대.
- finetuning/system-prompt 베이스라인이 충분히 최적화되지 않음(hyperparameter search 부재) — CAA 우위가 과대평가됐을 가능성을 저자 스스로 인정.
- GPT-4 rater 의존 open-ended 평가는 rater 편향·prompt wording sensitivity에 취약.
- MC 이지선다 A/B라는 인위적 대조 형식에 의존 — 연속 행동 공간·이산 선택지가 없는 도메인(VLA action)으로의 전이 가능성은 미검증(우리가 직접 확인해야 할 도메인 격차).
