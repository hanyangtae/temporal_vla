# Refusal in Language Models Is Mediated by a Single Direction (Arditi et al. 2024)

- 출처: NeurIPS 2024 · arXiv:2406.11717v3 · Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery(Anthropic), Wes Gurnee, Neel Nanda
- PDF: `docs/Activation_steering_basic/ArditiRefusal_2406.11717.pdf`
- 정독 섹션: §3 방법(difference-in-means 추출·directional ablation·activation addition), §4 weight orthogonalization jailbreak, §5 adversarial suffix mechanistic 분석 중심, §1·2·6·7 개관 확인
- tier: must
- 한줄 역할: "행동(거부) 전체가 residual stream의 단일 1차원 subspace로 매개된다"를 13개 모델·72B까지 실증하고, 그 방향을 지워버리는(directional ablation/projective ablation) 개입만으로 안전학습을 무력화 — projective **suppression**(erasure)이 additive steering과 별개 축임을 보여주는 대표 사례이자 우리 ¬C_failure(실패 방향 억제) 설계의 직접 선례.

## 문제·동기

Chat 모델은 RLHF/instruction-finetuning으로 유해 요청을 거부하도록 정렬되지만, 이 거부 행동의 내부 메커니즘은 잘 알려져 있지 않다. 기존 jailbreak(적대적 접미사, 파인튜닝 기반 탈옥 등)은 거부를 우회하는 데는 성공하지만 "왜/어떻게" 우회되는지에 대한 메커니즘적 설명이 없다. 저자들은 mechanistic interpretability와 activation steering(Panickssery/CAA, Turner/ActAdd, Zou/RepE)의 최근 성과에 착안해, 거부라는 안전-관련 행동이 모델 내부에서 어떻게 표상되는지를 규명하고자 한다.

## 핵심 아이디어

거부는 단일 모델·단일 층에 국한된 우연한 상관관계가 아니라, **13개 오픈소스 chat 모델(1.8B~72B) 전반에 걸쳐 residual stream의 단 하나의 1차원 방향(subspace)에 의해 매개된다.** 이 방향을 활성화에서 제거(directional ablation)하면 유해 지시에도 거부하지 않게 되고(necessary), 이 방향을 무해한 지시의 활성화에 더하면(activation addition) 거부가 유도된다(sufficient) — 즉 필요충분조건을 모두 인과적으로(단순 상관이 아니라 개입으로) 입증한 것이 핵심 기여. 나아가 이 방향을 아예 가중치에서 orthogonalize(직교화)해버리는 rank-1 weight edit만으로, 추론 시 개입 없이도 영구적으로 거부를 해제할 수 있음을 보인다.

## 방법(refusal direction 추출=diff-of-means, directional ablation=projection 제거, addition으로 유도)

- 데이터: Dharmful(ADVBENCH·MaliciousInstruct·TDC2023·HarmBench 취합), Dharmless(Alpaca 샘플). 각 train 128 / val 32.
- **추출(diff-of-means)**: 매 층 l·post-instruction 토큰 위치 i마다 유해 프롬프트 평균활성 μ와 무해 프롬프트 평균활성 ν의 차 r=μ−ν 를 계산(식 2) — CAA의 Mean Difference와 정확히 같은 연산이나, 여기서는 (layer×position) 후보 집합 전체를 만들고 validation 지표로 **단 하나**를 선택한다는 점이 다르다.
- **후보 선택 알고리즘(§C.1)**: 각 후보 r(l)_i에 대해 bypass_score(ablation 시 D_val_harmful에서 거부율 하락 정도), induce_score(addition 시 D_val_harmless에서 거부율 상승 정도), kl_score(ablation이 무해 프롬프트 처리를 얼마나 교란하는지)를 계산. induce_score>0, kl_score<0.1(부작용 최소), l<0.8L(unembedding 방향에 너무 가까운, 즉 "I"/"As" 같은 거부 토큰을 직접 못 쓰게 막는 표층적 트릭 배제) 제약 하에서 bypass_score를 최소화하는 (l*, i*) 하나를 뽑는다.
- **Directional ablation(식 4)**: x' ← x − r̂r̂ᵀx. 모든 층·모든 토큰 위치의 모든 residual stream 활성화에서 r̂ 방향 성분을 투영-제거(projective ablation) — 모델이 그 방향을 아예 표상하지 못하게 만듦.
- **Activation addition(식 3)**: x^(l)' ← x^(l) + r^(l). 추출된 층 l에만, 모든 토큰 위치에 원벡터(단위벡터 아님, norm 포함) 그대로 더함 — CAA와 달리 단일 층에만 적용.
- **Weight orthogonalization(식 5, §4.1)**: directional ablation과 수학적으로 동치인 정적 개입. residual stream에 쓰는 모든 W_out(임베딩·위치임베딩·attn out·MLP out)의 열벡터를 r̂에 대해 직교화 → 추론 시 개입 없이 가중치 자체가 그 방향을 영원히 못 쓰게 됨(rank-1 edit, gradient 불필요, 유해 completion 예시 불필요).
- 평가: refusal_score(문자열 매칭 "I'm sorry" 등) + safety_score(Llama Guard 2가 유해성 판정) 이중 지표로 "거부 안 함"과 "실제 유해 콘텐츠 생성"을 분리.

## 실험·결과(13개 모델, jailbreak)

- Qwen(1.8B/7B/14B/72B), Yi(6B/34B), Gemma(2B/7B), Llama-2(7B/13B/70B), Llama-3(8B/70B) 13개 모델, APO/AFT 두 정렬 방식 모두 포함.
- **Directional ablation** (JailbreakBench 100문항): 전 모델에서 개입 없을 때 거의 만점이던 refusal/safety score가 개입 후 크게 하락(Fig.1) — 72B까지 일관.
- **Activation addition** (Alpaca 무해 100문항): 개입 전 거의 거부 안 하던 모델이 개입 후 무해 요청까지 거부(Fig.3, Gemma 7B 예시 — 요가 효능 질문에 "위험할 수 있어 답 못함").
- **Weight orthogonalization(ORTHO) jailbreak vs HarmBench 159문항**: Qwen 계열에서 프롬프트별 최적화가 필요한 GCG(prompt-specific)와 맞먹는 ASR(예: Qwen 14B ORTHO 84.3, GCG 83.5). Llama-2는 system prompt 유무에 따라 ASR이 크게 갈림(7B: 22.6→79.9, system prompt 제거 시).
- **Coherence 유지(§4.3)**: MMLU/ARC/GSM8K는 orthogonalize 전후 거의 무변화(대부분 99% CI 이내), TruthfulQA만 일관되게 소폭 하락(안전 관련 카테고리와 겹치는 데이터셋 특성으로 저자는 해석) — "표적 능력만 훼손, 나머지 유지"라는 surgical 개입의 근거.
- **적대적 접미사 메커니즘(§5)**: Qwen 1.8B에서, 유해 지시에 적대적 접미사를 붙이면 마지막 토큰의 refusal-direction cosine similarity가 급격히 억제됨(무작위 접미사는 억제 안 함, Fig.5). Direct feature attribution으로 refusal 방향에 가장 크게 기여하는 top-8 attention head를 찾아보니, 적대적 접미사가 이 head들의 attention을 지시문 영역에서 접미사 영역으로 "탈취(hijack)"해 refusal 방향 기여를 억누른다는 것을 보임(Fig.6) — 즉 기존 jailbreak도 결국 같은 단일 방향의 표현을 억제하는 방식으로 작동한다는 통합적 설명.

## activation-steering 흐름 위치(projective/suppression 대표, erasure≠addition)

Burns(CCS, 비지도 탐지) → Turner(ActAdd) / Panickssery(CAA) / Zou(RepE) 계열의 "대조쌍 diff-of-means로 방향 추출" 기법을 그대로 계승하지만, 이 논문의 차별점은 **개입의 종류를 addition이 아니라 projective ablation(erasure)에 둔 대표 사례**라는 점이다. CAA/ActAdd/ITI는 전부 "방향을 더해서(add) 행동을 유도"하는 데 초점을 맞추지만, Arditi는 "방향을 지워서(project out) 행동을 억제"하는 쪽이 오히려 더 근본적인 인과관계(necessary condition)를 보여준다고 주장하고, addition은 보조 증거(sufficient condition)로만 쓴다. 이는 erasure(concept scrubbing, Belrose/INLP 계열 전통)와 addition(steering 전통)이 별개 축이며, 둘을 한 논문에서 필요-충분 양쪽으로 결합해 "단일 방향이 행동을 완전히 매개한다"는 강한 causal claim을 세운 최초 사례 중 하나다. weight orthogonalization은 이 ablation을 정적 가중치 편집(rank-1 edit)으로 등가 변환한 것으로, ROME류 weight-editing과 activation-steering 계열을 잇는 다리 역할도 한다.

## 우리 프로젝트 연결(¬C_failure 억제와 유비)

- 우리 C_steer = C_success ∧ ¬C_failure 중 ¬C_failure 항은 정확히 이 논문의 directional ablation과 같은 연산 구조다: "실패와 상관된 방향(들)을 활성화에서 투영-제거"하는 것이 곧 실패로 이어지는 표상을 억제한다는 발상. 다만 Arditi는 **1차원**(rank-1, r̂r̂ᵀ)이고 우리 conceptor는 **multi-dim subspace**(rank-k 연산자) — 이 논문은 "1차원으로도 충분한 경우가 실제로 존재한다"는 강한 증거이므로, 우리가 conceptor의 rank를 낮춰볼 때(ablation 실험) 하한 baseline으로 재사용 가능.
- 방향 선택 알고리즘(§C.1, bypass/induce/kl_score 3중 제약)은 우리가 "어느 층·어느 phase에서 개입할지"를 정할 때 쓸 수 있는 절차적 템플릿: (1) 개입 시 목표행동 변화폭(우리는 ΔSR), (2) 반대방향 개입 시 유도 효과, (3) 무관 상황에서 부작용(KL/task 성능 훼손) 최소화라는 3축 스크리닝은 우리 사다리식 ablation의 "각 단계 노이즈 fit 방지" 취지와 일치.
- necessary(ablation) + sufficient(addition) 양방향 인과 검증 설계는 우리 실패방향에도 그대로 적용 가능한 방법론: 실패 latent를 지웠을 때 성공률이 오르는지(necessary), 성공 latent에서 실패방향을 인위로 더했을 때 실제로 실패가 유도되는지(sufficient)를 둘 다 확인해야 "그 방향이 실패를 매개한다"는 주장이 강해진다 — 지금까지 우리 검증은 주로 steering-후-ΔSR(더하는 방향)에 치우쳐 있어, ablation-only 실험(빼기만 해서 인과 확인)을 추가할 근거가 된다.
- weight orthogonalization처럼 "추론 시 개입 대신 가중치를 영구 편집"하는 옵션은 우리가 백본 재학습 없음 제약(가중치 파인튜닝 금지)과는 다른 것 — 우리는 여전히 활성화 단(inference-time hook)에 머물러야 하지만, 이 논문은 동일 수학적 개입이 "매 forward마다 hook" 대 "1회 가중치 edit"로 등가 변환될 수 있음을 보여 배포 시 오버헤드 논의에 참고할 수 있다.

## 면접 포인트 (Q→A)

1. Q: "단일 방향이 거부를 매개한다는 게 왜 놀라운가?" A: "거부처럼 복잡해 보이는 안전 행동(다양한 유해성 판단, 맥락 이해, 언어 생성)이 72B 파라미터 모델에서도 결국 1차원 선형 부분공간 하나로 압축된다는 것은, 안전학습이 실제로는 매우 좁고 얕은 '스위치' 하나를 세우는 것에 불과할 수 있음을 시사한다. necessary(제거→거부 안 함)와 sufficient(추가→거부 유도)를 동시에 인과적으로 입증했기 때문에 상관관계가 아니라 실제 그 방향이 행동을 '매개(mediate)'한다고 말할 수 있다."
2. Q: "안전 함의는 무엇인가?" A: "이 발견으로 만든 rank-1 weight edit만으로 70B 모델을 5달러 이하 컴퓨팅으로 탈옥할 수 있음을 보여, 현재 오픈소스 안전 파인튜닝이 매우 취약(brittle)함을 실증했다. gradient 최적화도, 유해 completion 예시도 필요 없다는 점에서 기존 파인튜닝 기반 탈옥보다 문턱이 낮다 — 저자들은 이것이 오픈소스 배포 정책 논의에 기여한다고 프레이밍한다."
3. Q(우리 프로젝트 관점): "이 논문을 우리 VLA steering에 어떻게 참고하나?" A: "¬C_failure(실패 부분공간 억제)의 직접적 선례다. 다만 이 논문은 방향이 1차원이라는 강한 결과를 얻었지만, 우리 실패 latent는 실패 유형(goal vs motor)·phase에 따라 다차원적으로 갈릴 가능성이 커서 conceptor(multi-dim)로 일반화가 필요하다. 또한 우리는 아직 ablation-only causal 검증(빼기만 했을 때 SR이 오르는지)을 addition 실험만큼 체계적으로 하지 않았는데, 이 논문의 necessary+sufficient 이중검증 설계를 그대로 이식할 수 있다."

## 한계·비판

- 저자 스스로 "이 논문은 그런 방향이 존재한다는 존재증명(existence proof)이지, 최선의 추출법에 대한 연구가 아니다"라고 명시(§7) — diff-of-means·validation grid search라는 다소 휴리스틱한 절차이며 방향 자체의 의미론(semantic meaning)은 불명확("refusal"이 아니라 "harm/danger" 등 다른 개념일 수 있음).
- 검증된 모델이 전부 오픈소스 1.8B~72B 텍스트 전용 chat 모델 — 최신 SOTA 독점 모델·훨씬 큰 스케일·멀티모달(VLA 포함)에 일반화되는지는 미검증.
- TruthfulQA 성능이 orthogonalize 후 일관되게 하락하는데, "안전 관련 카테고리와 겹쳐서"라는 저자의 해석은 사후적(post-hoc)이며 완전히 surgical하지 않을 가능성을 시사(부작용이 완전히 0은 아님).
- 적대적 접미사의 메커니즘 분석(§5)은 단 하나의 모델(Qwen 1.8B)·단 하나의 접미사 사례에 국한 — 일반화 주장은 약함(저자도 "종합적 메커니즘 이해가 아니다"라고 인정).
- Panickssery et al.(CAA, 이 논문의 공저자이기도 함) 자신의 이전 연구는 "steering이 객관식에는 통하지만 장문 생성에는 잘 안 통한다"고 보고했는데, 본 논문은 장문 생성에서도 강한 효과를 보여 상충 — 데이터셋/모델/추출 절차 차이가 원인일 수 있으나 두 논문 결과의 정합성 논의는 부재.
- 우리 문제(VLA 실패)와 달리 refusal은 이산적 이지선다(거부 vs 응답)에 가까운 행동이라 방향이 잘 분리될 수 있음 — 연속적 동작 공간·다양한 실패 원인이 얽힌 VLA에는 "단일 방향으로 충분"이라는 결론이 그대로 이식되지 않을 가능성이 높다(우리가 검증해야 할 도메인 격차).
