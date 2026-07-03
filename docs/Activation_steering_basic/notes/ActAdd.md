# Activation Addition (Turner et al. 2023)

- 출처: arXiv:2308.10248 (원제 "Steering Language Models with Activation Engineering", 방법명 ActAdd), ICML 2024 accepted / v5 2024-10-10
- PDF: docs/Activation_steering_basic/ActAdd_2308.10248.pdf
- 정독 섹션: §3 (How Activation Addition Works) 중심, §2(관련연구 분류표) · §4(실험) · §5·부록(한계·ablation) 보조 참고
- tier: must
- 한줄역할: additive single-vector activation steering의 원형(prototype). 우리 conceptor(C_success ∧ ¬C_failure) 및 CAA와 나란히 놓는 baseline — "최적화 없이 대조쌍 차이만으로 스티어링 벡터를 만든다"는 계보의 시작점.

## 문제·동기

- prompt engineering은 신뢰성이 낮고(Ye & Durrett 2022), finetuning/RLHF는 비용이 크고 weight 전체를 바꿔 부작용(capability 손상)이 있음.
- 저자들의 가설: LLM은 이미 가진 능력을 다 "이끌어내지(elicit)" 못하는 **elicitation overhang** 상태 — 예: eloquent와 mathematical이 학습데이터에 동시 등장 안 하면, 그 조합을 뽑아낼 prompt가 존재하지 않을 수 있음. activation을 직접 조작하면 이 조합을 강제로 활성화할 수 있다는 아이디어.
- 목표: gradient/backward pass 없이, inference-time에 residual stream만 건드려서 output 속성(topic, sentiment, toxicity)을 제어.

## 핵심 아이디어

- 대조 프롬프트 쌍 (p+, p-) (예: "Love" vs "Hate")을 모델에 각각 forward, 특정 layer l에서의 활성화 차이 h_A = h+^l − h-^l 를 스티어링 벡터로 삼는다.
- 이 벡터를 "virtual bias term"처럼 user prompt의 forward pass 도중 residual stream에 그대로 더해(add) 이후 계산을 계속한다.
- 핵심: 학습/최적화 없음(optimization-free). 프롬프트 쌍 1개(=2개 데이터포인트)만 있으면 되고, 저자들은 "몇 분 만에" 많은 예시 벡터를 찾았다고 보고.

## 방법 (§3, Algorithm 1)

입력: 스티어링 프롬프트쌍 (p+, p-), user prompt p*, 대상 layer l, injection coefficient c, 정렬 위치 a, 모델 M.

```
(p'+, p'-) = pad_right_same_token_len(p+, p-)
h+^l = M.forward(p'+).activations[l]
h-^l = M.forward(p'-).activations[l]
h_A^l = h+^l − h-^l                      # 스티어링 벡터 (사전 계산, inference 이전)
h^l   = M.forward(p*).activations[l]
S = M.continue_forward( c·h_A^l + h^l @ a )   # 즉 h' = h + c·h_A  (h_A ≈ h+ − h-)
```

- 우리 표기로 옮기면 h' = h + α·v (α=c, v=h_A). 완전한 additive single-vector steering.
- 주입 위치: residual stream, layer l (middle layer가 경험적으로 최적, Fig3에서 GPT-2-XL layer≈6 부근 피크, 이후 감소). c는 보통 |c|<15, layer sweep은 대략 l∈[6,24] 그리드서치.
- sequence 정렬 a는 논문 전체에서 a=1로 고정("front" addition, 첫 토큰 스트림부터 개입). 개입 위치의 다음 토큰 예측은 마스킹하여 개입 지점 자체의 logit에 오염되지 않게 함.
- h+ 단독(대조 없이)은 비효과적(Appendix Table 7, Love 벡터 단독 실패) — 대조 프롬프트가 방향을 암묵적으로 "특정"해주는 역할이 필수.
- 짧은 프롬프트는 오른쪽 공백 패딩으로 토큰 길이를 맞춤. 최적화/gradient 없이 forward pass 몇 번으로 끝남.

## 실험·결과

- GPT-2-XL: wedding 벡터(p+=weddings, p-=' ', l=16, c=1)로 wedding-related 문장 perplexity ratio 감소(최대 -4%), 무관 문장엔 영향 거의 없음(Fig2).
- layer sweep: 최적 injection에서 wedding 단어 포함 completion 비율 >90% (baseline ~2%, Fig3).
- 토픽 제어(finance/music/politics/science/weddings 등, GPT-3.5 relevance 채점): c=2에서 5~20%p 상승(art 제외).
- 독성 감소(RealToxicityPrompts, Perspective API): ActAdd-OPT toxicity .112 (baseline .134, 2등 PREADD-D .122보다 8%↓), ActAdd-LLaMA3 .108 (baseline .114).
- 감정 제어(IMDb): negative→positive에서 SOTA (LLaMA3 0.669), positive→negative는 PREADD가 우세하지만 fluency 손상 훨씬 큼(68.4 vs 24.2).
- 일반지식 보존(ConceptNet, P@K): steering 부작용이 거의 없음(Fig5) — off-target 성능 유지 주장의 핵심 근거.
- ablation: random 방향 벡터는 anger 벡터보다 오히려 KL-divergence가 큼(=덜 targeted, Fig10) → steering 방향이 무작위가 아니라 유의미한 feature 방향임을 시사. partial ActAdd(차원 일부만 더함)도 완만하게 효과 스케일(Fig14).
- 재현성: GPT-J-6B, Llama-1-13B, OPT-6.7B, LLaMA-3-8B로 확장 재현했으나 Llama-13B의 일부 벡터(Eiffel→Rome, anger, harm)는 재현 실패.

## activation-steering 흐름 위치 (additive 계보 출발)

- 논문 Table 2 분류축: "벡터를 어떻게 얻는가"(gradient 탐색 vs 프롬프트쌍 차이) × "무엇에 개입하는가"(weight vs activation). ActAdd는 "프롬프트쌍 차이 × activation 개입" 칸의 대표(Li et al. 2023b ITI와 동일 칸이지만 ITI는 probe 기반 head 선택 + 모든 시퀀스 위치에 동일 벡터 주입, dozens 표본 필요 — ActAdd는 위치 subset + 최소 2 표본).
- Subramani 2022(gradient search로 벡터 추출), Dathathri 2020 PPLM(classifier 기반 perturbation)은 이 논문의 "optimization 필요" 계열 선구자.
- 후속: CAA(Rimsky 2023, contrastive activation addition — ActAdd를 여러 층/많은 대조쌍으로 일반화), Zou 2023 RepE, Liu 2023 in-context vectors 모두 "이후(followed) 연구"로 명시.
- 즉 ActAdd는 additive single-vector steering 계보의 사실상 출발점(원형)이고, 우리 conceptor(다차원 projective 연산자)는 이 계보에서 "단일 벡터 덧셈"을 "subspace 정렬"로 일반화한 지점에 위치.

## 우리 프로젝트 연결

- **additive vs projective**: ActAdd는 h' = h + c·v, v는 rank-1 방향(단일 대조쌍의 차이), c는 사람이 그리드서치하는 자유 스칼라. 우리 conceptor는 h' = h·Mᵀ, C_steer = C_success ∧ ¬C_failure로 fit한 다차원(soft) subspace 프로젝션 — 단일 방향이 아니라 성공 활성화 분포 전체가 만드는 공간으로 정렬시키는 연산. ActAdd의 "c를 얼마나 키울지"가 우리에겐 conceptor aperture/threshold 하이퍼파라미터에 대응(둘 다 완전히 hyperparameter-free는 아님).
- **boosting 위험**: Appendix F가 결정적 — c=+10 Anger−Calm 벡터는 residual stream 크기의 최대 ~90%까지 차지(destructive interference 무시 시). Appendix Table 7은 c=100처럼 과하면 텍스트가 붕괴하는 실패 사례를 명시. 우리 VLA 세팅은 연속 action 공간이라 "문법 붕괴"에 해당하는 명시적 안전판이 없어 이 위험이 더 크다 — over-steering이 action을 조용히 망가뜨릴 수 있음(텍스트처럼 "말이 안 됨"으로 바로 드러나지 않음). conceptor의 norm-bounded/subspace-projection 성격이 이 위험을 완화할 후보지만, 검증되지 않은 가설이며 별도 ablation 필요.
- **phase-matched 축 부재**: ActAdd는 고정 layer l, 고정 c, 시퀀스 위치 a=1(front) 하나로 정적으로 개입 — "언제(rollout phase)" 개입 강도를 바꾼다는 개념이 아예 없음. 우리 DiT phase-matched steering(rollout-t 조건부)은 ActAdd에 없는 완전히 새로운 축.
- **pathway 분리 축 부재**: ActAdd는 단일 residual stream, 단일 layer 개입. VL/DiT처럼 서로 다른 pathway를 따로 조준한다는 개념 없음. 다만 §4.1.3의 layer sweep 방법론(어느 층이 최적인지 실험적으로 찾기)은 우리가 VL-SA/DiT block별 최적 개입 지점을 찾는 것과 방법론적으로 유사.
- **선형 표현 가설의 실험적 근거**(§5 Interpretability)는 우리 conceptor/steering 전체가 딛고 서 있는 전제(성공/실패가 activation space의 방향/부분공간으로 분리 가능)를 뒷받침하는 초기 증거로 인용 가능.

## 면접 포인트 (Q→A)

- Q: ActAdd가 "optimization-free"라는 게 왜 중요한가?
  A: gradient/backward pass 불필요, 프롬프트쌍 2개의 forward pass만으로 벡터 계산 → 빠른 반복(수 분 단위), 학습 인프라 없이 inference 가능한 어떤 환경에서도 적용 가능. Subramani 2022류 gradient-search 방식과의 핵심 차별점.
- Q: 왜 h+ 단독이 아니라 h+ − h- 대조를 쓰는가?
  A: Appendix Table 7에서 Love 벡터 단독은 거의 효과 없음. 대조쌍의 차이가 "의미 있는 방향"을 암묵적으로 특정해주기 때문 — 절대 activation 값이 아니라 두 조건 간 차이(방향)가 유의미하다는 선형 표현 가설과 일치.
- Q: 왜 middle layer 주입이 최적인가?
  A: 실험적 관찰(Fig3): 너무 이른 층은 feature가 아직 덜 형성되었고, 너무 늦은 층(특히 마지막 층 근처)은 이후 attention이 통합할 시간이 부족해 문법 붕괴를 유발. 이는 hard theory가 아니라 경험적 sweep 결과.
- Q: c는 무한히 키워도 되나?
  A: 아니다. Appendix F에서 c=10이면 residual stream의 최대 ~90%가 steering vector로 결정됨(원래 forward-pass 정보를 거의 압도) — 이 이상은 coherence 붕괴(Table 7). steering 강도와 fluency/일관성 사이 trade-off가 명시적으로 존재.
- Q: ActAdd가 단순 프롬프트/토큰 주입과 뭐가 다른가?
  A: Appendix B 세 가지 실험으로 반박 — (1) 3×wedding − 3×whitespace 같은 연속적 스태킹은 discrete token으로 불가능, (2) embedding만 later layer에 주입(0→20)하면 효과 미미(즉 효과 대부분이 중간 layer들의 계산 작업에서 나옴, 단순 토큰 주입이 아님), (3) perplexity 실험에서 prompting은 무관 문장에 큰 penalty를 주지만 ActAdd는 그렇지 않음.
- Q: 우리 conceptor steering과 ActAdd의 핵심 차이는?
  A: (1) rank-1 additive 방향 vs rank-k projective subspace, (2) 정적 프롬프트쌍 1개 vs succ/fail 활성화 분포 전체로 fit, (3) phase/pathway 조건 없음 vs phase-matched + pathway-split.

## 한계·비판

- c, l이 사람이 튜닝해야 하는 자유 하이퍼파라미터(재사용 가능하다곤 하나 완전 자동은 아님).
- API-only 모델엔 적용 불가 — 중간 activation을 캐시/노출해야 함(현재 대부분 상용 API가 미지원).
- reasoning 능력에 대한 영향은 검증하지 않음(저자 스스로 명시).
- 모델 간 재현성 불완전: Llama-13B에서 일부 벡터(Eiffel→Rome, anger, harm) 재현 실패.
- 모든 steering 방법에서 fluency 저하 불가피(ActAdd도 1.5~3배 disfluency 증가) — steering-fidelity와 품질 사이 근본적 트레이드오프.
- 정성적 예시는 K=3 completion 중 "가장 그럴듯한 것"을 보고(best-of-K) — cherry-picking 가능성, 대표성 의문.
- Appendix F·H의 메커니즘 설명("axis-aligned feature", QKV readout 가설)은 저자 스스로 "미검증(undemonstrated)"이라 인정 — 왜 되는지에 대한 이론은 약함, 경험적 성공이 이론을 앞섬.
- 단일 벡터·단일 층 설계라서 다중 속성 동시 제어나 조건부(phase/타입-dependent) 제어를 위한 프레임워크가 없음 — 우리 문제(phase-matched, pathway-split)로 확장하려면 근본적인 구조 확장이 필요.
