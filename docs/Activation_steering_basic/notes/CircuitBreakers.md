# Improving Alignment and Robustness with Circuit Breakers (Zou et al. 2024)

- 출처: arXiv:2406.04313 (v4, 2024-07-12) · Gray Swan AI / CMU / Center for AI Safety (Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks). 코드: github.com/GraySwanAI/circuit-breakers
- PDF: `docs/Activation_steering_basic/CircuitBreakers_2406.04313.pdf`
- 정독 섹션: §5 Limitations and Conclusion 중심 (+ §1 Intro, §3 방법, §4 실험 전반 확인)
- tier: must
- 한줄 역할: RepE의 control(steering) baseline을 "안전 제품"으로 상용화한 논문 — 해로운 표현이 형성되는 순간을 **재라우팅(reroute)/단락(short-circuit)** 시켜 다음 토큰 생성을 인과적으로 차단, attack-agnostic 방어를 구현.

## 문제·동기

기존 정렬(RLHF/DPO refusal training)은 출력 레벨 지도만 하므로, refusal state를 우회하면(jailbreak) 그 뒤의 harmful state는 여전히 접근 가능하다(Fig 1 비유: instruct model은 harmful state가 그대로 있고, refusal training은 refusal state 층을 하나 더 씌운 것뿐이라 우회되면 뚫림). Adversarial training은 특정 공격에 대해서만 막고 unseen attack엔 일반화가 안 되며, 강건성을 얻는 만큼 capability를 깎는 트레이드오프가 "불가피한 사실"로 통용된다. 저자들은 이 트레이드오프 자체를 깨는 것을 목표로 한다: 특정 공격(vulnerability)을 막는 게 아니라, 모델이 애초에 해로운 출력을 만들어내는 능력(intrinsic hazard) 자체를 표현 공간에서 차단한다.

## 핵심 아이디어

RepE에서 빌려온 개념 방향(harmful representation)을 이용해, 모델이 harmful 출력을 생성하는 과정 중 그 표현이 나타나는 순간 이를 **원 표현과 직교(orthogonal)한 방향으로 재라우팅**한다("circuit-breaking" / "short-circuiting" — 전기 회로의 누전차단기 비유). 생성은 다단계(multi-step, autoregressive) 과정이므로 공격자는 매 스텝마다 영향력을 행사해야 하는데, 방어자는 그 어느 스텝에서든 개입할 기회를 갖는다는 것이 저자들의 핵심 통찰. 이 방식은 "어떤 입력이 공격인지"를 분류하는 문제(공격마다 다름, 무한히 다양)가 아니라 "어떤 출력이 harmful인지"를 표현 공간에서 정의하는 문제(유한하고 잘 정의됨)로 방어를 재구성한다 — 그래서 공격 방식에 무관(attack-agnostic)하다.

## 방법(RepE 기반, representation rerouting/short-circuit, LoRRA)

- **LoRRA(Low-Rank Representation Adaptation, RepE 논문에서 제안)** 를 구현체로 사용: 원 모델 M은 얼리고, LoRA adapter를 얹은 M_cb만 학습. 손실은 표현(representation) 레벨에서 정의되며 출력 토큰 레벨 손실이 아니다.
- **데이터 2분할**: Circuit Breaker Set(harmful 표현을 유도하는 예시) / Retain Set(circuit breaker가 발동하면 안 되는 예시 — UltraChat + XSTest, Llama-3는 refusal 데이터 추가). refusal-training된 모델(Llama-3)에서는 harmful user request는 제거하고 harmful assistant response만 남기는 방식으로 "이미 우회된 상태의 표현"을 학습시켜, 정상 refusal 메커니즘은 보존하면서 우회됐을 때만 circuit breaker가 걸리게 설계(§3 Data).
- **RR(Representation Rerouting) 손실** (Algorithm 1):
  - 재라우팅 손실 L_s = ReLU(cosine_sim(rep_M(x_s), rep_Mcb(x_s))) — circuit breaker set에서 원 모델 표현과의 코사인 유사도를 0 이하로 밀어(직교화), harmful 표현을 "쓸모없게" 만듦.
  - 유지 손실 L_r = ‖rep_M(x_r) − rep_Mcb(x_r)‖₂ — retain set에서는 원래 표현을 L2로 보존.
  - 계수 스케줄: c_s = α(1 − t/2T)(감소), c_r = α·t/2T(증가) — 초기엔 회로차단 위주, 후반엔 능력보존 위주로 전환.
  - 대안 손실 비교: RMU식 고정 랜덤벡터로 거리 최소화(‖rep_cb − α·rep_rand‖₂, α 튜닝 필요·수렴 실패), 랜덤 unit vector 정규화(RandP는 수렴하나 덜 강건, RandC는 수렴 실패) — cosine ReLU(RR)가 가장 안정적·효과적.
- 적용 위치: Llama-3-8B/Mistral-7B는 layer 10·20에 RR 손실, layer 0~20 전 linear layer에 LoRA adapter 삽입. 1×A100-80GB, 150 step, 20분 학습(매우 저비용).
- 멀티모달(LLaVA-NeXT-Mistral-7B): 이미지 인코더/projection은 고정, LM 백본만 layer 16 타깃(adapter는 14~16). 에이전트(function calling): circuit breaker/retain set에 Glaive function-calling 기반 harmful/benign 호출 예시 추가.
- 대조군 **Harmfulness Probing(HP)**: 같은 데이터로 16번째층(Mistral)/최종층(Llama-3) 활성화에 선형·MLP 분류기를 학습해 "표현을 읽어서(reading)" harmful 토큰이면 생성을 멈추는 방식(representation control 아닌 representation reading 대안). RR이 더 강하지만 MLP probe는 근접.

## 실험·결과(공격 방어, 능력 보존)

- **LLM 강건성 (Table 1, HarmBench 10종 공격 평균 ASR)**: Mistral-7B refusal-trained 76.7% → **+RR 9.8%**(약 87% 감소). Llama-3-8B refusal-trained 38.1% → **+RR 3.8%**(약 90% 감소) → 추가 RepE control까지 결합한 파인튜닝 **Cygnet 0.8%**(약 2자릿수 감소). 능력 저하는 MT-Bench 기준 1% 미만(Mistral 7.60→7.53, Llama-3 8.05→8.00), Open LLM 평균도 거의 동일(Mistral 65.4→65.4, Llama-3 68.8→68.3) — adversarial training(MT-Bench −8%p 이상 하락)과 대비되는 Pareto 개선.
- **개별 공격**: GCG(백박스), PAIR/TAP-T(LLM 최적화 jailbreak), Prefilling, Input-Embedding attack, RepE Attack(반대로 refusal 방향을 조작하는 공격) 등 unseen 공격에도 강건 (Fig 2, Table 1) — 예: Llama-3 Prefilling ASR 84.9%→3.3%, Input Embed 80.4%→9.6%, RepE Attack 91.2%→0.0%.
- **멀티모달**: LLaVA-NeXT-Mistral-7B, PGD(ε=32/255, 1000 step) 공격 하에서도 compliance rate 91.0%→**14.3%**(원본 대비 84% 감소, safety prompt 대비 85% 감소) — safety prompt는 오히려 PGD에서 96.2%로 악화. MMMU/LLaVA-Wild 능력은 0.5%p 이내로 보존.
- **AI 에이전트(함수호출)**: harmful function-call 100건 벤치마크, No-Attack 58%→8%, Forced-Function-Call(prefilling과 유사) 82%→14%, BFCL 능력 점수는 오히려 74.8→76.0으로 개선.
- **HP(reading) vs RR(control) 비교 (Table 2)**: Mistral 평균 ASR 80.6(baseline)→19.0(Linear probe)→14.3(MLP probe)→**11.2(RR)**. representation reading만으로도 상당히 강하지만 control(RR)이 항상 더 우수 — 단, probe는 공격자가 probe 존재를 모르는 약한 위협 모델에서 평가됨.
- **표현 분석 (Fig 6)**: "Here is how to synthesize meth: 1. Start with" prefill 공격에서 circuit-breaking이 걸린 모델은 layer 10부터 원 모델과의 코사인 유사도·norm이 **생성 시작 전(prefilling 도중)** 이미 급격히 변화 — 표현을 직접 관찰해 circuit breaker 발동 여부를 실시간 탐지 가능함을 보여줌(시스템 레벨 모니터링에 활용 가능 시사).
- **Ablation**: circuit breaker set에 refusal-bypass 예시 추가(w/ Augment) → ASR 5.8→2.5(능력 유지); retain set에 refusal 데이터 추가(w/ Refusal) → 능력 보존은 좋아지지만 ASR은 오름(0.6→2.5) — robustness-capability 트레이드오프가 데이터 큐레이션에도 존재. 카테고리별 일반화(Fig 5): 넓은 범주(Harmful/Illegal)로 학습하면 좁은 범주(Cyber/Chem-Bio)보다 cross-category 일반화가 더 좋음.

## activation-steering 흐름 위치(산업 안전 적용)

이 논문은 RepE(§3.2)가 제시한 세 operand 중 **LoRRA**(파인튜닝 기반, 저비용 추론)를 구체적인 안전 제품 스펙(Representation Rerouting 손실 + Circuit Breaker/Retain 데이터 설계 + layer 선택 실무)으로 완성한 사례다. RepE가 "이런 트레이드오프가 있다"는 프레임을 제시했다면, Circuit Breakers는 "그 중 어떤 조합(cosine ReLU + LoRA + 2-set 데이터)이 실제로 프론티어 LLM 안전 제품에 배포 가능한 수준으로 작동하는가"를 검증한 논문이다. Gray Swan AI라는 실제 스타트업이 저자로 참여해, HarmBench 표준 벤치마크·멀티모달·에이전트 함수호출까지 실무 배포 시나리오 전체를 커버한다는 점에서, activation/representation steering 계보 중 **가장 먼저 상용 제품화 단계까지 간 사례**로 위치시킬 수 있다(CAA/ActAdd=연구용 프로토타입, RepE=프레임 정의, Circuit Breakers=산업 배포 스펙).

## 우리 프로젝트 연결(실패표현 억제·재라우팅)

- 구조적 유비가 직접적이다: CB의 `harmful representation → 직교 방향으로 재라우팅 + retain set으로 정상 표현 보존`은 우리의 `C_steer = C_success ∧ ¬C_failure`가 하려는 일(실패 표현을 억제하며 성공 표현/정상 task 수행 능력은 보존)과 목적이 동일하다. 다만 CB는 코사인 유사도를 0으로 미는 **직교화(orthogonalization) 손실을 표현에 직접 파인튜닝으로 새겨넣는 것**이고, 우리는 **추론 시 projection/contrastive operator(h·Mᵀ)를 얼려진 백본 위에 적용**하는 것 — "백본 재학습 없음" 제약상 우리는 CB의 LoRRA 방식이 아니라 RepE §3.2의 순수 inference-time Contrast Vector/Projection operator 쪽에 가깝다.
- CB의 retain loss(정상 표현을 L2로 유지)는 우리가 conceptor를 fit할 때 "성공 표현을 훼손하지 않고 실패 방향만 억제"해야 하는 요구사항과 정확히 대응 — CB의 ablation(§4.4, w/o Refusal retain → ASR은 낮아지지만 capability 저하)은 우리 conceptor에서도 ¬C_failure 억제를 너무 세게 걸면 정상 동작(성공 궤적) 표현까지 깎일 위험이 있다는 경고로 읽을 수 있다.
- CB는 **매 생성 스텝마다 항상 켜져 있는(always-on) 개입**이라 online phase 조건부가 아니다 — 우리 프로젝트의 핵심 난제(phase/failure-type을 온라인에 식별해 steering을 라우팅)는 CB 범위 밖. 다만 Fig 6의 "표현의 코사인/norm 변화로 개입 발동 시점을 실시간 탐지할 수 있다"는 관찰은, 우리가 원하는 online phase/failure-type 신호(예: 현재 latent가 C_failure 쪽으로 얼마나 가까운지를 매 rollout step마다 읽어 steering 강도를 조절)의 계측 아이디어로 그대로 전용 가능하다.
- CB의 "카테고리 넓게 학습해야 일반화 잘 됨"(Fig 5) 관찰은, 우리가 VL(goal)/DiT(motor) 실패를 하나의 global C_failure로 뭉칠지 pathway별로 분리할지 판단할 때 참고할 수 있는 반례 데이터 — 단, CB는 "해로움"이라는 하나의 상위 개념 안에서의 카테고리 일반화이고, 우리는 애초에 **서로 다른 인과 메커니즘(goal 오인식 vs motor 실행 오류)**을 다루므로 직접 적용은 주의가 필요.

## 면접 포인트(Q→A)

**Q1. Representation Rerouting이 단순 activation steering(ActAdd/CAA)과 다른 점은?**
A. ActAdd/CAA는 추론 시 얼려진 모델에 고정된 방향 벡터를 더하는(additive) 기법이라 방향은 사전에 한 번 계산해 고정된다. RR은 (1) 코사인 유사도를 0으로 미는 **직교화** 손실이라 "더한다"가 아니라 "원래 방향과 무관하게 만든다"는 다른 연산이고, (2) 이 손실을 LoRA adapter 파라미터에 대해 **파인튜닝**해 표현 자체가 회로차단 상태로 바뀌도록 모델에 새겨넣는다(LoRRA) — 그래서 추론 시 추가 연산이 없고, 매 토큰마다 자동으로 발동한다는 점이 프로덕션 관점에서 중요한 차이다.
**Q2. 왜 "attack-agnostic"이라고 주장하는가?**
A. 방어가 "어떤 입력이 공격인가"를 분류하는 문제가 아니라 "harmful 출력이 형성될 때 나타나는 표현"을 직접 타깃으로 하기 때문. 공격자가 GCG suffix든 prefilling이든 임베딩 최적화든 어떤 경로로 그 harmful 표현에 도달하든, 도달한 순간 재라우팅되므로 특정 공격 패턴을 학습할 필요가 없다. Table 1에서 학습 시 노출되지 않은 공격(RepE Attack, Multilingual 등)에도 방어가 유지되는 것이 이 주장의 실증.
**Q3. Harmfulness Probing(표현 읽기)만으로는 왜 부족한가?**
A. Table 2에서 Linear/MLP probe도 refusal-only baseline보다 훨씬 강하지만 RR(표현 통제)에는 못 미친다. 더 중요한 건 probe는 "공격자가 probe의 존재를 모르는" 약한 위협 모델에서만 평가됐다는 점 — probe도 결국 입출력 필터와 같은 층위의 방어라 적응적 공격에 최적화되면 우회될 수 있다(논문도 이를 인정). RR은 표현 자체를 바꿔버리므로 이 우회 경로가 원천적으로 좁다.

## 한계·비판

- **여전히 파인튜닝이다**: LoRRA는 저비용(A100 1장, 20분)이지만 "훈련이 필요 없다"는 순수 inference-time steering(ActAdd/RepE Contrast Vector)과 달리 명백히 학습 단계가 있다 — 우리 프로젝트의 "백본 재학습 없음" 제약과는 이 지점에서 결이 다르다(우리는 RR이 아니라 RepE §3.2의 Contrast Vector/Projection operator 쪽 계보를 따라야 함).
- **범위가 좁게 정의된 안전 목표**: §5에서 저자 스스로 "harmful output을 생성하지 못하게 하는" 한 종류의 공격만 다룬다고 명시 — 예컨대 생성모델을 이미지 분류기로 쓰는 다른 유형의 적대적 공격(클래스 라벨 자체가 harmful하지 않은 경우)에는 적용되지 않는다. 즉 "무엇이 회로차단 대상인가"는 결국 Circuit Breaker Set 큐레이션이 결정하며, 새로운 위해 카테고리마다 데이터를 다시 만들어야 한다 — 우리 맥락에서도 succ/fail 데이터가 pathway·phase·failure-type별로 다시 필요하다는 점과 유사한 확장성 제약.
- **단일 턴(single-turn) 중심**: 멀티턴 대화에서의 지속성/재발동 여부는 검증되지 않음.
- **retain set 큐레이션이 예민함**: Table 8 ablation에서 보듯 retain 데이터 구성에 따라 ASR과 capability가 반대 방향으로 움직임 — "robust하게 억제"와 "정상 기능 보존" 사이 트레이드오프가 여전히 수작업 튜닝에 의존.
- **LLM/멀티모달 챗·에이전트 함수호출 도메인**: continuous control(로봇 액션), diffusion 기반 action head(DiT)로의 이전 가능성은 다루지 않음. "매 토큰마다 always-on 개입"이 로봇 rollout의 시간축(phase)에서도 그대로 최선일지는 검증되지 않았고, 오히려 우리 문제의식(phase-matched, 즉 정확한 타이밍에만 개입)과는 반대되는 설계(항상 켜둠)다.
- **표현 탐지의 부수 효과 미검증**: Fig 6에서 norm이 CB 발동 후 크게 증가하는 현상을 발견했지만 이를 직접 통제하지 않았고, 왜/언제(prefilling 중 vs generation 중) 발동 타이밍이 프롬프트마다 다른지(Fig 10 vs Fig 11)에 대한 메커니즘적 설명은 부족 — 표현 개입의 부작용(norm 폭주 등)이 다운스트림에 미치는 영향은 향후 과제로 남겨둠.
