# ReFT: Representation Finetuning for Language Models (Wu & Arora et al. 2024, NeurIPS)

- 출처: arXiv:2404.03592 (v3, 2024-05-22) · NeurIPS 2024 · Zhengxuan Wu, Aryaman Arora(공동1저자), Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, Christopher Potts (Stanford)
- PDF: `docs/Activation_steering_basic/ReFT_2404.03592.pdf`
- 정독 섹션: §3 방법(3.1 동기/DAS, 3.2 LoReFT·DiReFT, 3.3 ReFT family 정의) 중심, §1·2·4·5 개관 확인
- tier: must
- 한줄 역할: activation steering을 "고정/통계적으로 뽑은 벡터를 더하는 것"에서 "저랭크 subspace 개입 함수(R,W,b)를 downstream task loss로 직접 gradient 학습하는 것"으로 끌어올린 논문 — steering vector류(ActAdd/CAA/RepE)를 자신의 프레임워크 안 rank-1 특수사례로 명시적으로 재정식화하고, LoRA류 weight-PEFT와 정면 비교해 파라미터 효율에서 압승 — 우리의 "백본 무학습" 원칙이 정확히 어디서 ReFT와 갈라지는지 시험하는 벤치마크 논문.

## 문제·동기

PEFT(LoRA, adapter, prefix-tuning 등)는 모델 파라미터의 일부를 적은 수만 업데이트해 적응 비용을 낮추지만, 공통적으로 "가중치"를 건드린다. 반면 해석가능성 연구는 representation이 이미 풍부한 의미 구조를 인코딩하고 있음을 계속 보여왔다(선형 표현 가설, 인과 개입 연구 등). 저자들은 "가중치 대신 표현을 편집하는 것이 더 강력한 대안일 수 있다"는 가설을 파고들어, activation steering(Turner ActAdd, Li ITI)과 representation engineering(Zou RepE)에서 쓰던 추론 시 개입을, task 성능을 위해 "학습"하는 방향으로 일반화한다.

## 핵심 아이디어

인과 해석가능성의 distributed interchange intervention(DII, Geiger et al. 2023b)을 제어(control) 도구로 전용한다. DII는 은닉표현의 저랭크 부분공간을 반사실적(counterfactual) 입력이 만들어냈을 값으로 덮어써 그 부분공간의 인과적 역할을 검증하는 해석 기법(distributed alignment search, DAS로 부분공간 R을 학습)인데, ReFT는 이 "부분공간을 찾아서 그 안의 값을 갈아끼운다"는 메커니즘을 그대로 가져오되, 반사실 소스 표현 대신 "task label 쪽으로 밀어주는 값"을 학습한다. 즉 해석가능성의 사후분석 도구를 inference-time steering 학습 도구로 뒤집은 것이 핵심.

## 방법(§3 — DII/DAS → LoReFT/DiReFT → ReFT family)

**DII (배경, 식 1)**: DII(b, s, R) = b + Rᵀ(Rs − Rb), R ∈ R^(r×d) orthonormal-row 저랭크 투영행렬. b(개입 대상 표현)를 s(반사실 소스 표현)가 부분공간 R 안에서 갖는 값으로 교체.

**LoReFT (식 2, 핵심)**: Φ_LoReFT(h) = h + Rᵀ(Wh + b − Rh). DII에서 "학습된 투영 소스" Rs = Wh+b 를 쓰는 변형 — 즉 반사실 입력 없이, 선형변환 Wh+b 로 만든 목표값을 r차원 부분공간에 주입한다. 학습 파라미터 ϕ = {R, W, b}(R∈R^(r×d) orthonormal, W∈R^(r×d), b∈R^r), 백본 파라미터는 전부 frozen.

**DiReFT (식 3, ablation)**: Φ_DiReFT(h) = h + W2ᵀ(W1h + b). orthogonality 제약과 차분(−Rh) 항을 제거 — 저자도 "은닉표현에 직접 적용한 LoRA"라고 명시(단, LoRA는 residual stream 자체엔 weight가 없어 직접 적용 불가하다는 차이는 있음). 학습은 더 빠르지만 성능은 LoReFT보다 소폭 낮음.

**ReFT family 정식 정의(§3.3)**: 개입 I = ⟨Φ, P, l⟩ — Φ: R^d→R^d(학습 파라미터 ϕ), P⊆{1,...,n} 개입 대상 토큰 위치 집합, l 개입 층. 순전파 중 h^(l) ← (Φ(h_p^(l)) if p∈P else h_p^(l))로 덮어쓰기. ReFT = 여러 개입의 집합이며 같은 층에 걸리는 개입끼리는 위치가 겹치지 않아야 함(정의 3.2). 훈련은 생성 task는 teacher-forcing 언어모델링 loss, 분류 task는 CLS 표현 위 별도 head loss로 ϕ(+θ)를 표준 gradient descent로 최적화한다.

**하이퍼파라미터**: prefix p개 + suffix s개 위치만 개입(tied/untied 선택 가능), 개입 층 집합 L, rank r. "prefix+suffix 소수 위치"만 개입하므로 개입 비용이 prompt 길이에 안 늘어난다는 게 LoRA 대비 장점으로 강조됨.

**기존 방법의 특수사례화(Appendix B)** — 저자가 직접 명시: RED(전 위치 element-wise scale+bias, adapter에 가까움), ActAdd(대조 프롬프트 차 벡터를 전 위치에 상수배 additive, I_ActAdd={⟨Φ,{1..n},l⟩}), RepE의 projection operator(Φ(h)=h − c·(a·h/‖a‖²)·a)는 **"스케일된 1차원 DII로 LoReFT의 특수사례"**라고 논문이 직접 적음. 즉 CAA/ActAdd/RepE 계열 전체가 ReFT의 rank-1·비학습(고정 계수)·전위치 버전으로 흡수된다.

## 실험·결과(LoRA 대비 효율·성능)

- **Commonsense reasoning(8개 task, LLaMA-7B/13B, Llama-2 7B, Llama-3 8B)**: LoReFT가 전 모델에서 SOTA. 7B: LoReFT 80.2 avg vs DoRA 78.1 vs LoRA 74.7(params 0.031% vs DoRA 0.838%, LoRA 0.826% → 약 27배 적음). 13B: LoReFT 83.3 vs DoRA 81.5. Llama-3 8B: LoReFT 86.6 vs DoRA 85.2. DiReFT는 LoReFT보다 소폭 낮지만 여전히 DoRA/LoRA를 상회.
- **Arithmetic reasoning(4개 CoT task)**: LoReFT/DiReFT가 LoRA/Adapter보다 **못함**(7B avg 42.6 vs LoRA 46.9) — 저자 스스로 "고정된 소수 위치 개입은 생성 길이가 길어질수록(다단계 CoT) 효과가 상대적으로 희석된다"고 원인 분석. 13B에서는 격차가 줄어 스케일 의존성을 시사.
- **Instruction-following(Llama-2 7B, Alpaca-Eval v1.0 win-rate)**: LoReFT 85.60 > 풀파인튜닝 80.93 > LoRA 81.48 > RED 81.69 — **풀파인튜닝조차 능가**. 파라미터 절반(half, 84.12)이나 학습데이터 1/64(1K예시, 18분 학습, 81.91)로도 다른 PEFT를 이긴다. 파라미터 비중 0.0039%.
- **GLUE(RoBERTa-base/large)**: LoReFT는 RED와 파라미터 매치 시 comparable, DiReFT는 대부분 PEFT보다 낮음 — 저자는 "작은 LM일수록 LoReFT의 orthogonality+차분 항이 중요하다"고 해석. Fig.1 캡션: "가장 큰 모델에서 가치가 가장 두드러짐" — 즉 이득이 스케일 의존적.
- **전체**: LoRA 대비 15×–65× 파라미터 효율, 4개 도메인(20+ dataset) 중 3개(commonsense, instruction-following, NLU)에서 최강 PEFT 상회, arithmetic만 열세.

## activation-steering 흐름 위치(고정 벡터 → 학습된 개입)

계보: Subramani(2022, 첫 단일벡터 추출) → ActAdd(Turner 2023, 대조 프롬프트 차 벡터, 고정 계수, 전위치) → CAA(Rimsky 2024, 대조쌍 평균으로 노이즈 감소) → RepE(Zou 2023, PCA reading/contrast vector + 3종 연산자) → ITI(Li 2023, head-sparse mean-diff) 까지는 전부 "고정되거나 통계적으로 1회 산출한 방향 + 튜닝된 스칼라 계수"를 추론 시에만 적용하는 **비학습·post-hoc analysis** 계열이다. **ReFT(본 논문)**는 이 개입 함수 자체(부분공간 R, 변환 W,b)를 downstream task loss로 gradient 학습해 "steering을 post-hoc 분석 단계가 아니라 학습 가능한 절차로 만든 것"(§5 결론에서 저자가 명시)이 결정적 전환점이다. Appendix B가 ActAdd/RepE를 자신의 프레임워크 안 특수사례로 명시적으로 봉합하면서, "고정 벡터 additive"는 "학습된 저랭크 subspace 개입"의 rank-1·비학습 극단으로 이론적으로 자리매김된다.

## 우리 프로젝트 연결(백본 무학습 원칙과의 긴장)

- **핵심 긴장**: 우리 원칙은 "VLA 백본 추가학습 없음"이고, conceptor C_steer = C_success ∧ ¬C_failure는 succ/fail 활성화 분포의 **상관행렬(closed-form 통계)**로 fit한다(gradient 학습 없음, downstream task loss로 backprop 안 함). ReFT는 백본 **가중치**는 정확히 0개 건드리지만, 개입 모듈(R,W,b)은 여러 epoch에 걸쳐 **gradient descent로 downstream task loss에 대해 명시적으로 학습**한다 — 논문 스스로 "기존 weight-PEFT의 drop-in replacement"라 규정하고 LoRA/DoRA/Adapter와 trainable-parameter% 대 accuracy Pareto로 정면 비교한다(Fig.1). 즉 ReFT는 "가중치는 안 건드리지만 사실상 PEFT"다.
- 이는 우리 원칙을 두 갈래로 나눠 재정의해야 함을 보여준다: (a) "백본 가중치 frozen"만 지키면 되는가 — 이러면 ReFT는 허용범위. (b) "어떤 모듈이든 gradient 학습을 아예 하지 않고 closed-form 통계로만 개입 방향을 구한다" — 이게 현재 우리가 실제로 지키는(더 엄격한) 제약이다. conceptor는 (b)를 만족하지만 ReFT류 학습된 저랭크 개입은 (b)를 위반한다.
- LoReFT가 instruction-following에서 **풀파인튜닝조차 능가**(85.60 vs 80.93)했다는 결과는, "gradient로 학습된 개입 방향"이 "통계로 고정한 방향"보다 실질적 헤드룸이 있을 수 있음을 시사한다 — 우리가 closed-form conceptor만 고집함으로써 잠재적 SR 이득을 포기하고 있는지 확인할 자연스러운 ablation 하나: 백본은 그대로 frozen한 채로 DAS 스타일 저랭크 R을 succ/fail 구분 task loss로 살짝만 gradient 학습해보고, 그 방향이 우리 closed-form conceptor 방향보다 나은지 비교(단, 이 실험 자체가 원칙 (b)를 넘는 것이므로 "예외적 ablation"으로 명시하고 본 method 라인과는 분리해야 함).
- Appendix B의 관찰("미래 ReFT는 schematic하게 — 이른 층에서는 첫 토큰, 늦은 층에서는 마지막 토큰처럼 — 개입 패턴을 바꿀 수 있어야 한다")은 정확히 우리의 phase-matched DiT steering 문제의식과 같은 방향을 가리키지만, 본 논문의 실제 실험은 이걸 구현하지 않았다(위치 p,s는 dev set 하이퍼파라미터 탐색으로 **고정**) — "온라인으로 무엇을·언제 개입할지 정하는 문제"는 ReFT에서도 미해결로 남아있어, 우리 "★ 중심 미해결 문제"와 정확히 같은 공백을 공유한다.
- arithmetic reasoning에서 LoReFT가 열세였던 원인("생성 길이가 길수록 고정 위치 개입 효과가 희석")은 우리 rollout 길이 confound 문제의식(길이가 길수록/짧을수록 라벨이 갈리는 현상, seen18에서 검증)과 구조적으로 유사 — VLA rollout처럼 생성 길이(=timestep 수)가 가변인 도메인에서는 "고정 위치/고정 층" 개입 설계가 근본적으로 불리할 수 있다는 방증.

## 면접 포인트(Q→A; ReFT vs LoRA vs steering vector)

**Q1. ReFT와 LoRA의 근본적 차이는?**
A. LoRA는 **가중치**에 저랭크 additive update를 학습해 추론 시 가중치에 merge할 수 있어(오버헤드 0) 강력한 PEFT다. ReFT는 **은닉표현(활성화)**의 특정 위치·층에서 저랭크 부분공간(R)을 찾아 그 부분공간의 값을 학습된 변환(Wh+b)으로 덮어쓴다 — residual stream은 "weight가 없는" 공간이라 merge가 불가능해 항상 nonzero 추론 오버헤드가 남는다(단 prompt의 소수 위치에만 개입하므로 prefill 단계에 집중). 대신 파라미터 수는 LoRA 대비 15×–65× 적고 여러 벤치마크에서 더 우수하다. DiReFT는 사실상 "LoRA를 은닉표현에 직접 적용한 것"이라는 형태적 유사성이 있다(논문이 각주로 명시).

**Q2. ReFT와 (CAA/ActAdd 같은) 고정 steering vector의 차이는?**
A. steering vector류는 대조 활성화의 평균차나 PCA 1주성분으로 방향을 **한 번 산출**하고, 추론 시 튜닝된 스칼라 계수로 더하기만 한다(gradient 학습 없음, post-hoc). ReFT는 개입 함수의 파라미터(R,W,b) 자체를 downstream task loss에 대해 **gradient descent로 학습**한다 — DAS(causal abstraction 해석기법)의 "부분공간을 찾고 그 안의 값을 반사실 값으로 바꾼다"는 메커니즘을 제어용으로 전용한 것. 논문 Appendix B는 RepE의 projection operator를 "LoReFT의 스케일된 1차원 특수사례"로 명시적으로 재정식화해, steering vector 계열 전체가 ReFT의 rank-1·비학습 극단임을 프레임워크 상으로 보여준다.

**Q3(우리 프로젝트 관점). ReFT를 우리 VLA steering에 그대로 쓸 수 있나?**
A. 그대로는 못 쓴다. 두 가지 장벽이 있다. (1) ReFT는 gradient 학습 가능한 downstream task loss(다음 토큰 cross-entropy, 분류 loss)를 전제하는데, VLA의 action head(diffusion/flow 기반 연속 액션)는 이런 명확한 이산 라벨이 없다 — "성공으로 이어지는 다음 액션이 정답"이라는 명시적 supervised 신호를 구성하기 어렵다. (2) 더 근본적으로 ReFT는 gradient로 개입 모듈을 학습하는 방법이라 우리의 "백본 추가학습 없음" 원칙(우리는 closed-form 통계 기반 conceptor만 허용)과 정확히 어긋난다 — 백본 가중치는 안 건드리지만 별도 모듈을 task loss로 학습한다는 점에서 본질적으로 PEFT다. 다만 LoReFT가 풀파인튜닝을 능가한 결과는, "약간의 학습된 개입"이 통계 기반 고정 방향보다 나을 수 있다는 참고 상한선(ceiling) 실험으로는 가치가 있다.

## 한계·비판

- **하이퍼파라미터 민감도**: 개입 위치(p,s), tied 여부, 층 집합 L, rank r, dropout, lr, weight decay, batch, warmup, epoch 등 탐색 공간이 매우 크다(Table 5-8). 저자도 §D.2에서 "ReFT는 PEFT/파인튜닝처럼 하이퍼파라미터에 민감할 수 있다"고 인정 — 실무 적용 비용이 상당하다.
- **긴 생성에 약함**: arithmetic reasoning(다단계 CoT)에서 LoRA/Adapter보다 열세 — 개입 위치가 고정된 소수 토큰이라 생성 길이가 길어질수록 효과가 상대적으로 희석된다는 저자 자평. 가변 길이 생성/rollout 도메인 전반에 대한 일반화 우려.
- **추론 오버헤드 0이 아님**: LoRA와 달리 residual stream(weightless)에 개입하므로 merge 불가 — rank·층 수·위치 수가 늘수록 오버헤드가 늘어난다(Appendix H, 10개 층·rank 8 개입 시 약 0.05초 추가).
- **기억(memorization) 능력이 해석가능성 주장과 긴장**: rank-1 개입 하나로 2048 토큰 시퀀스를 100% 복원하거나 256개 임의 입출력쌍을 기억할 수 있음(Appendix F) — 저랭크 개입이 "해석 가능한 개념 방향"이 아니라 단순 룩업/암기를 인코딩할 수 있다는 뜻이라, "학습된 개입=더 해석가능"이라는 저자 주장(§5)에 반례적 긴장을 준다.
- **개입 위치가 정적**: prefix p개+suffix s개는 dev set 하이퍼파라미터 탐색으로 한 번 고정되고 추론 내내 동일 — Appendix B가 스스로 제안한 "schematic(문맥별로 다른 위치/층 패턴)" 확장은 실현되지 않은 future work일 뿐, 온라인 라우팅은 다루지 않는다(우리 "★ 중심 미해결 문제"와 동일 공백).
- **스케일 의존적 이득**: Fig.1 캡션이 스스로 인정하듯 LoReFT의 이득은 "가장 큰 모델에서 가장 두드러짐" — GLUE(RoBERTa-base/large, <=350M)에서는 DiReFT가 대부분 PEFT보다 열세, 작은 모델·작은 표현 차원에서는 이점이 줄어든다.
- **task-specific 지도학습 데이터 필요**: 대조 활성화 통계(mean-diff/conceptor)만으로 충분한 steering vector류와 달리, ReFT는 각 task마다 별도의 (다량의) 라벨된 학습 데이터와 hyperparameter 탐색용 dev set이 필요 — 데이터·연산 비용이 steering vector류보다 훨씬 무겁다.
