# Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI (Maiya et al. 2025)

- 출처: arXiv:2511.01689v1 (2025-11-03, cs.CL) · Sharan Maiya(University of Cambridge, 1저자) + Henning Bartsch(MATS) + Nathan Lambert(Allen Institute for AI) + Evan Hubinger(Anthropic). 코드: github.com/maiush/OpenCharacterTraining
- PDF: docs/Activation_steering_basic/OpenCharacterTraining_2511.01689.pdf
- §5(산업) 파트: 장벽/현실 — persona 형성 과제에서 system prompt vs fine-tuning vs activation steering 3자 비교, fine-tuning이 강건성·일관성에서 steering을 능가함을 정량 실증 → 업계(Anthropic Constitutional AI, OpenAI Model Spec)가 steering 대신 fine-tuning으로 character를 굽는 이유의 직접 근거.
- 3축: **쓰기 아님, 학습**(논문 자체 기여는 steering이 아니라 DPO+SFT 파인튜닝 — 비교 baseline인 steering만 write/additive, 논문 방법은 weight에 영구히 "굽는" training-time 개입) / **연구, 그러나 산업 정조준**(저자 4인 중 2인이 산업 소속 — Nathan Lambert=Ai2, Evan Hubinger=Anthropic 현직, Anthropic Constitutional AI 자체를 비교 대상으로 명시하는 학술 논문) / **training-time**(character training=파인튜닝으로 고정 vs steering·system prompt=inference-time, 이 대조가 논문의 핵심 축).
- 한줄역할: 페르소나 형성이라는 동일 과제에서 fine-tuning·steering·prompting을 직접 맞대결시켜, steering이 "강도 튜닝이 모델마다 다르고(0.7~525) 일부 모델(Qwen)에서 아예 깨지며 응답이 과장·붕괴된다"는 정량적 약점을 보인 §5 반대 증거(counter-evidence) 논문.

## 문제·동기

Anthropic Constitutional AI, OpenAI Model Spec 등 frontier 랩은 이미 assistant persona를 post-training(character training)으로 형성해 실무에 쓰지만 구현 세부는 비공개이고, 학계엔 재현 가능한 오픈소스 구현·평가 기준이 없다. 반면 오픈소스 커뮤니티는 system prompt나 activation steering(repeng류)으로 persona를 손쉽게 바꾸지만, 이것이 산업의 fine-tuning 기반 character training과 실제로 얼마나 다른지(더 얕은지) 비교한 연구가 없었다. 저자들은 최초의 오픈 character training 구현을 공개하고 이를 두 대안(system prompt, activation steering)과 직접 대결시켜 이 공백을 메운다.

## 핵심 아이디어

Constitutional AI를 변형한다. 기존 CA 헌법은 "무엇을 골라야 하는가"를 페어와이즈 지시문으로 쓰지만, 이 논문의 헌법은 1인칭 성격 진술(~10개 assertion, "나는 ~하다")이다. 이를 (1) DPO distillation(교사 모델이 헌법을 system prompt로 embody해 chosen 응답 생성, 원 모델은 rejected) → (2) introspection SFT(post-distillation 체크포인트가 스스로 self-reflection/self-interaction 대화를 생성해 헌법 이상의 세부 뉘앙스를 학습)의 2단계 파이프라인으로 굽는다. 평가는 self-report 대신 revealed preference(두 trait 중 하나를 시스템프롬프트로 몰래 선택하게 하고 Elo로 집계)로 측정해 self-report 신뢰성 문제(Zou et al. 2024, Han et al. 2025)를 피한다.

## 방법(persona 형성 3방법 비교)

- **System prompt**: 헌법 텍스트를 그대로 system prompt에 넣어 embody 지시.
- **Activation steering**: Vogel(2024) repeng 오픈소스 구현 — "아무 얘기나 해봐" 응답과 헌법-embody 응답의 활성 차이에서 1st principal component를 스티어링 벡터로 추출, residual stream 12.5~87.5 백분위 layer 전체에 additive 상시 주입. 모델별 스티어링 상수를 완전히 다르게 수동 튜닝(Llama 0.7, Qwen 4.0, Gemma 525.0) — 논문이 이 자체를 fine-tuning 대비 단점으로 명시(범용 파이프라인 불가).
- **Fine-tuning(제안 방법, Character Training)**: distillation(DPO, LoRA rank64) + introspection(SFT, LoRA rank64) 2단계, LIMA+헌법 특화 프롬프트로 학습. Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B 3개 오픈 모델 × 11개 persona(sarcastic~misaligned)에 동일 파이프라인 적용.

## 실험·결과(steering vs finetuning 우열)

- **강건성(§3.2, adversarial "break character" 8종 프롬프트, Figure 5)**: MODERNBERT 분류기 F1 기준 system prompt가 가장 취약(쉽게 "helpful assistant"로 복귀). Steering은 Llama·Gemma에선 system prompt보다 강건하지만 Qwen에서는 성능이 붕괴(사실상 무작위 수준). Fine-tuning(distill+introspect)이 3모델 전부에서 최고 평균 F1.
- **Prefill 공격(§3.3, Table 2)**: distillation-only 대비 character training(distill+introspect) F1이 Llama 0.79→0.95, Qwen 0.66→0.86, Gemma 0.84→0.95 — introspective data가 robustness에 추가 기여.
- **일관성(§3.4, Table 3, LLM-judge 승률)**: character training이 steering 대비 승률 Llama 78.4%, Qwen 94.4%, Gemma 82.1%. Steering 응답은 "강제된" 저확률 토큰 샘플링으로 과장·붕괴되는 경향(Figure 6: 전부 대문자 SHOUT 예시).
- **일반능력(§3.5, Table 4)**: 5개 벤치마크(TruthfulQA/Winogrande/HellaSwag/ARC/MMLU)에서 flourishing·loving persona는 거의 저하 없음, misalignment persona만 유의 저하(헌법 자체가 "그럴듯하지만 오도하는 답"을 장려하도록 설계돼 의도된 결과로 해석). 선행연구(Chen et al. 2025 persona vectors, Durmus et al. 2024)가 보고한 "steering 강도 증가 → 능력 저하" 부작용이 fine-tuning 기반 방법에서는 거의 관찰되지 않음.

## §5(산업)에서의 위치

저자진에 Anthropic 현직 연구자(Evan Hubinger, Constitutional AI 원저자 그룹)와 Ai2의 Nathan Lambert(RLHF Book 저자, post-training practitioner)가 포함돼, 논문이 겨냥하는 비교 대상 자체가 "업계가 이미 쓰는 fine-tuning 기반 character training"이다. 즉 이 논문은 산업이 steering을 시도했다가 버린 사례가 아니라, 산업 practitioner가 "왜 fine-tuning을 쓰는지"를 사후적으로 정량 검증한 논문에 가깝다. §5에서 "안전 gate로 배포된 사례"(Circuit Breakers)나 "제품 UX로 실험된 사례"(GoogleSteerableChatbots)와 달리, 이 논문은 steering을 업계가 채택하지 않는 이유에 대한 반증(counter-evidence) 역할을 한다.

## 우리 프로젝트 연결(직접연결 약함 — 명시)

직접 연결은 약하다. 이 논문은 언어모델의 대화 persona(가치관·말투) 형성이 목표이고, steering baseline도 전 layer에 단일 방향을 상시 주입하는 blunt한 repeng 구현이라 우리의 pathway/phase-matched contrastive conceptor 방법과는 개입 정밀도가 근본적으로 다르다. 참고할 지점 두 가지:

- **경고**: steering 강도 튜닝이 모델마다 완전히 다르고(0.7 vs 4.0 vs 525.0) 과하면 응답이 원래 분포를 벗어나 붕괴한다는 관찰은, 우리 conceptor steering(h' = h·Mᵀ)에서도 강도가 체크포인트마다 캘리브레이션돼야 하고 과하면 action이 자연스러운 매니폴드를 벗어나는 유비적 위험 신호다.
- **차별점(우리에게 유리한 근거)**: 이 논문의 steering은 phase/pathway 구분 없이 전 layer·전 시점에 단일 방향을 상시 주입하는 가장 단순한 형태이며, 논문 스스로 이 blunt함을 fine-tuning 대비 약점으로 지목한다. 우리 방법은 정확히 그 지점(pathway 분리 + phase-matched, 특정 시점·pathway에만 개입)을 개선 축으로 삼는다 — 다만 이 논문의 "steering 실패 모드"가 우리 세팅에도 그대로 적용될지는 별도 검증이 필요하다.
- 이 논문은 fine-tuning(백본 재학습)을 승자로 결론짓는데, 우리 프로젝트는 "백본 재학습 없음"을 전제로 한다 — 이 논문의 결론을 그대로 받아들이면 우리 연구 전제와 충돌한다. §5에서는 이 긴장을 "그럼에도 왜 steering을 계속 연구하는가"(재학습 비용, latency, 안전 gate로서의 유연성 등 fine-tuning이 못 하는 것)에 대한 우리 쪽 반론 근거로 프레이밍해야 한다.

## 면접 포인트(Q→A)

Q1. 이 논문이 쓴 activation steering baseline은 최신 기법인가, 비교의 공정성에 어떤 영향을 주나.
A. 아니다. Vogel(2024)의 repeng — 두 대조군 활성 차이의 1st PC를 스티어링 벡터로 쓰는 비교적 단순한 구현이며, 전 layer(12.5~87.5 백분위)에 상시 additive 주입한다. Chen et al.(2025) persona vectors처럼 자연어 trait 기술에서 벡터를 추출하고 특정 layer만 타깃하는 더 정교한 기법과는 직접 비교하지 않았다 — "steering 일반"이 아니라 "이 특정 구현"이 fine-tuning에 졌다는 것이므로, 더 정교한 steering이 격차를 줄일 여지는 남아 있다.

Q2. Qwen 2.5 7B에서 steering이 유독 나빴던 이유는 무엇이라 보나.
A. 논문은 명시적 원인 분석을 하지 않지만, 모델별로 steering 상수를 0.7/4.0/525.0로 완전히 다르게 수동 튜닝해야 했다는 사실 자체가 동일 절차가 모델마다 다르게 반응함을 보여준다. Qwen은 residual stream 스케일이나 layer 민감도가 달라 최적 강도를 못 찾았을 가능성이 있으며, 이는 steering의 재현성/이식성 문제를 보여주는 사례로 저자들이 fine-tuning 대비 단점으로 명시한다.

Q3. 이 논문의 결론(fine-tuning이 steering보다 낫다)이 우리 VLA steering 프로젝트에 어떤 의미가 있나.
A. 도메인이 다르다(언어 persona vs 로봇 성공/실패). 우리는 "백본 재학습 없음"을 전제로 하므로 이 결론을 그대로 받아들이면 안 되고, steering을 쓰는 이유(재학습 비용, latency, 안전 gate로서의 유연성)를 별도로 방어해야 한다. 다만 이 논문이 보인 steering의 실패 모드(강도 튜닝 불안정, 과하면 output이 정상 분포를 벗어남)는 우리 conceptor steering의 강도 캘리브레이션에도 유효한 경고다.

Q4. 저자들이 "revealed preference" 방법을 새로 만든 이유는.
A. Self-report(모델에게 "너 지금 무슨 성격이야?" 직접 묻기)는 실제 행동과 괴리된다는 선행연구(Zou et al. 2024, Han et al. 2025)가 있다. 대신 두 trait 중 하나를 시스템프롬프트로 몰래 embody시키고 무작위 유저 프롬프트에 응답하게 한 뒤, LLM judge가 어느 trait이 발현됐는지 맞히게 해 Elo 랭킹을 낸다 — "내가 무슨 성격이라고 말하는가"가 아니라 "실제로 어떤 성격으로 행동하는가"를 측정한다.

## 한계·비판

- **모델 기반 평가의 순환성**: §3.1(revealed preference)과 §3.4(coherence) 판정 모두 LLM-as-a-Judge(GLM 4.5 Air)에 의존 — 저자 스스로 Discussion에서 bias·circularity 우려와 human rater 필요성을 인정한다.
- **steering baseline의 대표성**: repeng은 최신/최적 steering 구현이 아니라 상대적으로 단순한 baseline — "steering 일반"이 fine-tuning에 뒤진다는 결론의 일반화 범위는 제한적.
- **모델 규모**: 전부 10B 미만(8B/7B/4B) — 더 큰 모델에서 결론이 유지되는지 미검증.
- **misalignment persona의 능력 저하는 설계된 결과**: distillation 데이터 자체가 LIMA 질문에 "그럴듯하지만 틀린 답"을 담고 있어, 능력 저하가 character training 자체의 부작용인지 이 특정 헌법의 데이터 오염인지 완전히 분리되지 않는다.
- **다중 턴 평가는 초기 단계**: prefill attack 실험(§3.3)은 2턴짜리로 제한적이며, 저자도 다중 턴 LLM 평가가 "emerging area"임을 인정한다.
- **steering 강도 튜닝을 "단점"으로 프레이밍하지만 구현 선택의 문제일 수도 있음**: 모델별 상수(0.7/4.0/525.0)를 수동 반복 튜닝한 것은 repeng 구현의 관례이지, steering 자체의 근본적 한계인지는 더 나은 자동 캘리브레이션 방법이 있다면 재검토 여지가 있다.
