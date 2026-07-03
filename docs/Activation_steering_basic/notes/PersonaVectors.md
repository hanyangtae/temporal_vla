# Persona Vectors: Monitoring and Controlling Character Traits in Language Models (Chen et al. 2025)

- 출처: arXiv:2507.21509v3(2025-09-05) · Anthropic Fellows Program(Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey/Anthropic) · PDF: `docs/Activation_steering_basic/PersonaVectors_2507.21509.pdf` · 정독 섹션: §2(자동추출 파이프라인)·§3(스티어링·모니터링)·§4~6(finetuning shift 예측·완화·데이터 스크리닝) 중심, 부록 A(프롬프트·토큰위치)·B(judge 검증)·J(CAFT 비교) 확인 · tier: must · 한줄 역할: 자연어 trait 설명만으로 대조프롬프트→활성화 평균차를 자동 추출해 "persona vector"를 만들고 이를 모니터링(projection)·제어(steering)·데이터 필터링(projection difference) 세 용도에 재사용하는 파이프라인 — 우리 succ/fail 방향 자동추출 + online 모니터링 설계의 직접 참조점.

## 문제·동기

LLM은 "Assistant" persona로 배포되지만 prompting/context에 따라(Bing Sydney, Grok 사례) 또는 finetuning 과정에서(GPT-4o 아첨 사건, Betley의 emergent misalignment) 예기치 않게 성격이 바뀐다. 기존 activation steering 연구(ActAdd, CAA, RepE)가 trait이 선형 방향으로 인코딩됨을 보였지만, trait마다 수작업 대조쌍 큐레이션이 필요했다. 저자들은 trait 이름+설명만 주면 자동으로 대조쌍을 생성해 벡터를 뽑고, 이를 배포 중 모니터링·제어뿐 아니라 학습 전 데이터 스크리닝까지 확장하는 통합 파이프라인을 만든다.

## 핵심 아이디어

"trait 설명 → LLM(Claude 3.7 Sonnet)이 (a)positive/negative 대조 system prompt 5쌍, (b)평가 질문 40개, (c)0-100 judge rubric을 자동 생성 → 대조 응답 생성 → response-token 평균 활성화의 평균차(mean-difference)를 persona vector로 추출"이라는 완전 자동 파이프라인이 핵심 novelty다. 벡터 하나로 (1) steering(제어), (2) 프롬프트 활성화 projection(모니터링), (3) 학습데이터 projection difference(사전 예측·필터링) 세 응용을 통일적으로 지원한다는 것이 CAA/ITI 대비 확장점.

## 방법(persona trait contrastive 추출, 모니터링·추론시 완화·학습데이터 예측)

- 추출: trait+설명 → 5쌍 대조 system prompt·40개 유도질문(추출 20/평가 20 분할)·평가 rubric 자동 생성(부록A). 각 질문에 positive/negative prompt로 10 rollout씩 생성, judge(GPT-4.1-mini, top-20 logit 가중합으로 0-100 스코어)로 필터링(양성 응답>50, 음성 응답<50만 채택) 후 층별 response-token 평균활성화 차 = persona vector. prompt-last/prompt-avg/response-avg 토큰 위치 비교(부록A.3) 결과 response-avg가 steering 효과 최고 → 채택. 레이어는 steering 효과 스윕으로 최적 층 1개 선택(부록B.4).
- 모니터링: 마지막 prompt 토큰 활성화를 persona vector에 projection → 이후 응답의 trait score와 상관(r=0.75-0.83, system-prompt 계열 변주). 텍스트 생성 전에 persona shift를 사전 감지.
- 추론시 완화: h←h−α·v (persona vector 방향으로 빼기)로 post-hoc steering. 강도 올릴수록 trait score↓ 하지만 MMLU도 같이 훼손(부작용).
- 예방적 스티어링(신규 기여): 학습 중에 h←h+α·v로 "원치 않는 방향"을 미리 밀어넣어, gradient가 그 방향으로 이동할 압력을 상쇄 — 사후 개입보다 capability 보존이 우수(coherence>80, MMLU 거의 무손실). 다중 레이어로 확장(층별 incremental vector v_l − v_(l-1)) 시 더 효과적.
- 학습데이터 예측: 데이터셋 응답의 평균 projection − base 모델 "자연" 응답의 평균 projection = projection difference ΔP. ΔP가 클수록 finetuning 후 trait shift가 큼(dataset-level, 강상관) → 개별 샘플 단위로도 trait-inducing 샘플이 분리 가능(Fig.9), LMSYS-CHAT-1M 실데이터에서도 재현되고 LLM 필터링을 통과한 샘플까지 잡아냄(상호보완).

## 실험·결과

- 모델: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct. Trait: evil/sycophancy/hallucination(본문), optimism/humor 등 4종 추가(부록G).
- steering이 층·강도에 따라 trait score를 단조적으로 조절(Fig.3, layer sweep). 모니터링 projection-trait score 상관 r=0.75-0.83.
- finetuning shift(활성화 이동량 projection) vs 실제 trait score 상관 r=0.76-0.97 — 의도적/비의도적 성격 변화 모두 persona vector로 설명됨(cross-trait baseline r=0.34-0.86보다 확연히 높음).
- 완화: post-hoc steering은 trait↓ 하지만 MMLU 저하 동반, preventative steering은 trait↓ 이면서 MMLU·coherence(>80) 거의 보존 — 다중 레이어 preventative는 어려운 데이터셋에서도 baseline 수준까지 억제.
- CAFT(zero-ablation, Casademunt 2025) 대비: evil/sycophancy는 CAFT도 되지만(base 활성화가 원래 negative 쪽이라 ablation이 사실상 positive steering으로 작동), hallucination처럼 base projection이 0 근처인 trait엔 CAFT 무력 — steering은 방향성 개입이라 더 일반적.
- human-LLM judge 합치율 94.7%(부록B)로 judge 신뢰성 검증, 외부 벤치마크(HaluEval 등)와도 강상관(r=0.855-0.964)으로 자체 평가질문 타당성 검증.

## activation-steering 흐름 위치(모니터링+제어 결합)

ActAdd/CAA(수동 대조쌍 mean-diff)·RepE/AxBench(일반 개념 추출 프레임)의 계보를 이어받아 "대조쌍 생성 자체를 자동화"한 지점에 있다. CAA가 "평균차 벡터를 어떻게 주입할까"에 집중했다면, 본 논문은 그 벡터를 모니터링(judge 없이 projection만으로 사전 감지)·제어(post-hoc/preventative)·데이터 필터링 세 축에 재사용하는 파이프라인 관점으로 확장한다 — 단일 활성화 방향이 탐지(monitoring)와 개입(control) 양쪽에 동시에 쓰인다는 것을 체계적으로 실증한 사례 중 하나(RepE도 유사 이원성을 주장했으나 본 논문은 finetuning shift 예측·데이터 스크리닝까지 붙여 완전한 파이프라인화).

## 우리 프로젝트 연결(방향 자동추출·online 모니터링)

- 우리 succ/fail 대조 방향추출(C_success ∧ ¬C_failure)은 본 논문의 "positive/negative 대조 프롬프트 → response-token 평균활성화 mean-diff" 파이프라인과 도메인만 다른 동형 구조다. 다만 우리는 conceptor(공분산 기반 multi-dim 연산자)를 쓰고 본 논문은 rank-1 mean-diff — CAA 노트와 같은 결로 "conceptor는 이 rank-1 벡터의 다차원 일반화"로 위치시킬 수 있다.
- "마지막 prompt 토큰 activation의 persona vector projection이 향후 trait score를 예측한다"(§3.3, r=0.75-0.83)는 결과는 우리의 핵심 미해결 문제("online phase/failure-type을 rollout 중 읽을 수 있는가")에 대한 가장 직접적인 선례다 — 생성 "전"에 projection만으로 향후 행동을 예측한다는 스킴은, DiT/VL activation을 phase 진행 중 projection해 실패 조짐을 사전 감지하려는 우리 시도와 동일한 구조다.
- preventative steering(학습 중 원치 않는 방향으로 미리 밀어 gradient 압력을 상쇄)은 우리의 "추론시 steering만" 스코프 밖(백본 재학습 없음 제약)이지만, 방법론적 유비로는 "개입을 언제 거는가"(추론시 vs 학습시)의 트레이드오프(capability 보존 vs 적용가능성)를 보여주는 참고 사례로 기록해둘 만하다.
- 이 논문의 개입은 매 디코딩 스텝 균일 적용(CAA와 동일한 한계) — 저자는 이를 "persona가 대화 전체에 걸쳐 지속되는 latent factor"(§9)라는 전제로 정당화한다. 우리 phase-matched 문제의식은 이 전제가 VLA rollout에서는 성립하지 않는다(phase마다 관련 신호가 바뀜)는 데서 출발한다는 점이 대조점.

## 면접 포인트(Q→A)

**Q1. Persona vector 추출이 CAA/ITI와 다른 점은?**
A. CAA/ITI는 trait마다 사람이 대조 데이터셋을 큐레이션해야 했다. 본 논문은 trait 이름+짧은 설명만 주면 LLM이 대조 system prompt·평가 질문·judge rubric까지 전부 자동 생성한다. 추출 연산 자체(response-token 평균차)는 CAA와 동일한 mean-difference 원리다.

**Q2. 왜 response 토큰 평균이 prompt 토큰보다 나은가?**
A. 부록A.3 ablation에서 prompt-last/prompt-avg/response-avg 세 위치를 비교했더니 response-avg로 뽑은 벡터가 steering 시 trait score를 가장 크게 움직였다(Fig.11) — persona가 "응답을 생성하는 동안"의 활성화에 더 강하게 인코딩된다는 의미다.

**Q3. Preventative steering이 post-hoc steering보다 나은 이유는?**
A. Post-hoc steering은 이미 trait 쪽으로 이동한 파라미터를 추론 시점에 활성화만 반대로 밀어내므로, 강하게 밀수록 capability(MMLU)까지 훼손된다. Preventative steering은 학습 중 미리 그 방향의 활성화를 인위적으로 채워줘서 loss가 "그 방향으로 이동하라"고 요구하는 압력 자체를 상쇄한다 — 파라미터가 애초에 덜 이동하므로 capability 손실이 적다.

**Q4(우리 프로젝트 관점). 이 논문의 "projection 기반 모니터링"을 우리 VLA에 어떻게 이식하나?**
A. 이 논문은 텍스트 생성 전 마지막 prompt 토큰의 persona-vector projection으로 향후 trait 발현을 예측한다(사전 판별). 우리는 VLA rollout 중 DiT/VL activation을 succ/fail 방향(혹은 conceptor)에 projection해 "이 rollout이 실패 쪽으로 가고 있는가"를 온라인으로 판별하려 한다 — 다만 판별 기준이 rollout phase(진행도)에 따라 달라져야 한다는 추가 난제(phase-matched)가 있어, 이 논문의 정적 단일벡터 판별보다 한 단계 더 복잡한 문제다.

## 한계·비판

- 지도적(supervised) 추출: trait을 미리 지정해야 하고 자연어 설명 품질에 결과가 좌우된다(저자 자인, §8) — 우리 문제에서 "실패 유형(goal vs motor)"을 사전에 정확히 언어로 정의할 수 있는지는 별도 검증이 필요하다.
- 대조 mean-diff는 coarse-grained 방향만 잡아 세밀한 행동 구분을 놓칠 수 있음 — 우리의 phase별로 세분화된 실패 신호에 그대로 적용하면 정보 손실 가능성이 있다.
- trait이 system-prompt로 유도 가능해야 한다는 전제(§8) — 안전정렬이 강한 모델은 트리거 자체가 안 걸려 파이프라인이 성립 안 할 수 있다. VLA에는 "system prompt로 유도"라는 개념이 없어 대조쌍 생성 방식 자체를 재설계해야 한다.
- 평가가 LLM judge(GPT-4.1-mini) 의존 — human-agreement 94.7%로 검증했지만 sycophancy/hallucination 경계 케이스에서 judge 편향 사례를 저자도 인정(부록B.2).
- 모니터링 상관(r=0.75-0.83)은 주로 "trait-encouraging vs suppressing" 프롬프트 유형 간 차이에서 나오고, 같은 유형 내부로 통제하면 상관이 약해진다(§3.3) — "명확한 프롬프트 유도"에는 잘 통하지만 미묘한 배포중 변화 탐지엔 약할 수 있고, 우리의 온라인 phase 판별에서도 "쉬운 케이스만 잡힐" 유사한 위험이 있다.
- 두 중형 오픈소스 모델(7B/8B)의 텍스트 전용 decoder-only LM에서만 검증 — continuous action을 생성하는 VLA(DiT)로의 구조적 전이는 미검증이다.
