# 로봇 조작 실패 온라인 검출 문헌 지형 — phase 조건부 여부 조사

- 조사일: 2026-08-10
- 범위: 2023~2026, 로봇 매니퓰레이션 실패/이상 검출(failure/anomaly detection) 및 인접 연구
- 방법: WebSearch로 후보 수집 → WebFetch로 arXiv abstract/HTML/PDF 원문 확인. **모든 사실 기재는 WebFetch로 실제 조회한 내용에 근거**하며, 검색 스니펫만으로 추측한 내용은 "미확인"으로 별도 표시.
- 핵심 질문: **기존 실패 검출기 중 task phase/subtask/조작 단계에 조건부(conditioned)로 동작하는 것이 있는가?**

---

## 0. 결론 요약 (TL;DR)

phase 조건부성은 **두 갈래로 완전히 분리**되어 존재한다.

| 유형 | 정의 | 존재 여부 | 대표 사례 |
|---|---|---|---|
| **Type A — 심볼릭/계획 기반 phase 조건부** | LLM/behavior-tree가 사전에 task를 subgoal·step으로 분해하고, **각 phase마다 다른 monitoring 대상(precondition/effect/constraint)을 명시적으로 지정** | **존재함 (다수)** | ConditionNET, Code-as-Monitor, DoReMi, Hierarchical Task Decomposition, (약하게) Behavior-Tree 통합 프레임워크 |
| **Type B — 학습된 내부표현 기반 phase 조건부** | 정책의 **internal activation/latent**에서 phase·progress를 온라인으로 읽어내고, **그 phase에 따라 검출기 자체(파라미터·threshold·서브모델)를 분기** | **미발견 (0건)** | — |
| **Type C — phase가 출력 레이블일 뿐** | phase는 실패를 사후 분류하는 카테고리이며, 검출 자체는 phase 무관 | 존재함 | FINO-Net |

우리 프로젝트가 겨냥하는 것은 **Type B**(VLA 내부 activation으로 phase를 읽고, phase별 실패 검출기를 두는 것)이며, 조사한 26편 중 이 유형에 해당하는 논문은 **없었다**. VLA internal-activation 기반 실패 검출기(SAFE, FAIL-Detect, VLA-FAIL, Tri-Info, SAFECAST, ActProbe, Foresight)는 전부 **단일·통일 검출기**를 전체 rollout에 균일 적용하며, phase 분기가 없음을 명시적으로 확인했다(Foresight는 특히 "continuous causal transformer, no phase boundaries" 명시). phase 조건부 monitoring이 존재하는 곳(Type A)은 전부 **VLM/LLM이 사전에 정의한 심볼릭 subgoal**을 조건으로 쓰지, 정책의 학습된 내부표현에서 phase를 읽지 않는다.

---

## 1. 지정 대표 연구 8편

| # | 논문 | arXiv/venue | 연도 | 입력 | phase 조건부? | 시간 구조 | baseline/지표 |
|---|---|---|---|---|---|---|---|
| 1 | **SAFE**: Multitask Failure Detection for VLA Models | [2506.09937](https://arxiv.org/abs/2506.09937), NeurIPS 2025 | 2025 | VLA internal features (hidden states) | **NO** | per-timestep 점수(MLP: 누적합 / LSTM: causal 시퀀스) + max-so-far 집계, functional conformal prediction으로 **시간(t)에 따른** threshold band(μ_t ± h_t) — phase 아닌 순수 시간축 | Token Uncertainty, Embedding Distance(Mahalanobis 등), OOD(RND, LogpZO), Action Consistency(STAC); AUROC, TPR/FPR, Bal-Acc, T-det |
| 2 | **FAIL-Detect**: Can We Detect Failures Without Failure Data? | [2503.08558](https://arxiv.org/abs/2503.08558), RSS 2025 | 2025 | policy input(관측) + policy output(action) 2-stage | **NO** (명시적 phase 조건부 근거 없음; "phase"는 waypoint 진행 맥락에서만 언급) | sequential OOD + conformal prediction, 시간(t)에 따라 변하는 threshold — phase 아닌 시간축 | Diffusion Policy, BC 변형, Energy-based OOD, ensemble uncertainty; AUROC, TPR@FPR, detection delay |
| 3 | **Sentinel** (원제: Unpacking Failure Modes of Generative Policies) | [2410.04640](https://arxiv.org/abs/2410.04640), CoRL 2024 | 2024 | action chunk 분포(erratic) + VLM video QA(task progression) | **NO** — 명시: *"VLM 판단은 전체 task instruction 대상이며 subtask/stage로 분해하지 않는다"*("does not decompose tasks into subtasks or stages... assesses overall task progression") | erratic=temporal action consistency(STAC), progression=sparse VLM 샘플링(주기 ν) | Diffusion recon/변형, DDPM loss, embedding 유사도, GPT-4o QA, STAC 변형; TPR/TNR/detection time/Bal-Acc (AUROC 없음) |
| 4 | **Code-as-Monitor** | [2412.04455](https://arxiv.org/abs/2412.04455), CVPR 2025 | 2025 | VLM(vision) + 코드 생성 | **PARTIAL(Type A)** — task를 subgoal(예: Approach/Grasp&Transfer/Place)로 분해, 각 subgoal마다 **다른 constraint 2종**(실행중 C_d / 완료시 C_u) 지정. 단 subgoal 자체는 GPT-4o가 사전 결정, 온라인 내부표현에서 읽지 않음 | 1Hz 프레임, event-driven(제약 위반 시 즉시 중단) | DoReMi, Inner Monologue, ReKep; 성공률, 실행시간, token 사용량, gIoU/cIoU |
| 5 | **AHA**: VLM for Detecting and Reasoning Over Failures | [2410.00371](https://arxiv.org/abs/2410.00371), arXiv 2024 | 2024 | VLM(이미지+텍스트, free-form reasoning) | **미확인** — abstract 수준에서 phase 조건부 근거 못 찾음(본문 미조회) | 미확인 | GPT-4o ICL 포함 5개 SOTA VLM 대비 +35.3%(구체 지표명 미확인) |
| 6 | **RoboMD** (원제: From Mystery to Mastery) | [2412.02818](https://arxiv.org/abs/2412.02818), arXiv 2024 | 2024 | vision-language embedding(RL 탐색 공간) | **NO** — vulnerability 탐색은 trial/config 단위이며 rollout 내부 timestep/phase 개념 없음 | 미확인(trial 단위로 보임) | vision-language 베이스라인 대비 vulnerability +23% 발견(구체 baseline명 미확인) |
| 7 | **REFLECT** | [2306.15724](https://arxiv.org/abs/2306.15724), CoRL 2023 | 2023 | multisensory(RGB-D/audio/state) → hierarchical summary → LLM | **미확인** — hierarchical summary가 사건(event) 단위로 구성되나 canonical manipulation phase(reach/grasp 등)와의 대응관계, 그리고 이게 "검출"(온라인 판정)인지 "사후 설명"인지 본문 확인 못함. 사후 실패 설명·복구 계획 프레임워크로 보임(실시간 검출기 아닐 가능성) | 미확인 | 미확인(RoboFail 데이터셋으로 평가한다는 것만 확인) |
| 8 | **DoReMi** | [2307.00329](https://arxiv.org/abs/2307.00329), IROS 2024 | 2023(IROS 2024) | VLM(VQA로 constraint violation 판정) | **PARTIAL(Type A)** — LLM이 plan step마다 **다른 constraint를 순차 생성**("continue generation starting with 'Constraint:'"), VLM은 Δt=0.2s마다 **현재 유효한 모든 constraint**를 질의. constraint 내용은 step마다 바뀌지만 검출 메커니즘(VQA) 자체는 phase 무관 동일 | 0.2초 주기 연속 모니터링 | SayCan, CLIPort, Inner Monologue, IM-Oracle; 성공률·실행시간만(정밀 detection 지표 없음) |

---

## 2. 확장 조사 — 후속·인접 연구 (18편)

### 2-A. VLA internal-activation 기반 실패 검출기 (SAFE 계열, 2025~2026)

| 논문 | ID/venue | phase 조건부? | 핵심 근거 |
|---|---|---|---|
| **Tri-Info**: Generalizable, Interpretable Failure Prediction for VLA via Information Theory | [2606.19998](https://arxiv.org/html/2606.19998v1), 2026 | **미확인** | 저자·요지만 확인, phase 언급 문장 못 찾음(정보이론 기반 해석가능성이 핵심) |
| **VLA-FAIL**: Efficient Task Failure Detection for Finetuned VLA | [2606.21386](https://arxiv.org/abs/2606.21386), 2026 | **NO(명시 없음)** | LLMD(last-layer Mahalanobis) + ACC(action chunk consistency), receding-horizon 겹침 활용. 새 지표 AUCPDT 제안. phase 조건부 근거 없음 |
| **SAFECAST**: Robust Failure Detection for VLA with Contrast-Set Training | [2608.04246](https://arxiv.org/abs/2608.04246), 2026 | **NO(명시 없음)** | "hidden-state risk probes + functional conformal prediction" — SAFE와 같은 패러다임(hidden state 사용)이나 phase 분기 없음. DROID/LIBERO에서 평가 |
| **I-FailSense**: General Robotic Failure Detection with VLM | [2509.16072](https://arxiv.org/abs/2509.16072), 2025 | **NO(명시 없음)** | semantic misalignment 검출에 학습 후 일반 실패로 일반화; phase 조건부 근거 없음 |
| **ActProbe**: Action-Space Probe for Early Failure Detection | [2606.08508](https://arxiv.org/html/2606.08508v1), 2026 | **NO — 단 "task-conditioned"라는 표현이 phase가 아니라 task-identity 조건부임을 확인 (용어 함정)** | TCE(temporal consistency error)+ACM(action chunk magnitude), causal LSTM으로 **per-step failure probability**. "task-conditioned LSTM"은 언어 instruction 임베딩으로 LSTM 초기 상태를 세팅하는 것 — **어떤 task인지**를 조건화하는 것이지 **task 내부 phase**를 조건화하는 게 아님. "critical moments"도 phase 개념과 무관("hesitate, drift off-task, commit to unrecoverable actions") |
| **Foresight**: Failure Detection with Action-Conditioned World Model Latents (long-horizon) | [2606.23085](https://arxiv.org/html/2606.23085v1), 2026 | **NO — 명시적으로 부정** | 직접 인용: world model latent가 "**continuously... without phase boundaries or specialized handling**"로 처리됨. Long-horizon(multi-subgoal) task를 다루지만 subgoal 경계와 무관하게 causal transformer가 전체를 하나로 처리, trajectory-level 라벨을 모든 timestep에 균일 상속. Functional conformal prediction도 시간축 조정이지 phase 조건부 아님 |

### 2-B. 심볼릭/계획 기반 phase 조건부 monitoring (Type A — 실제로 존재하는 갈래)

| 논문 | ID/venue | phase 조건부? | 핵심 근거 |
|---|---|---|---|
| **ConditionNET**: Learning Preconditions and Effects for Execution Monitoring | [2502.01167](https://arxiv.org/html/2502.01167v1), RA-L 2025 | **YES — 조사 전체에서 가장 강한 사례** | skill을 **pre-phase / core-phase / effect-phase** 3단계로 (수동 annotation) 분해. 직접 인용: *"During the pre phase, the expected state is the precondition, and any deviation is considered an anomaly"*, *"in the effect phase, the expected state is the effect"*, *"During the core phase... anomaly detection is suspended"*(core phase에서는 판정 자체를 끔). 입력은 vision-only(DINOv2+CLIP), per-frame 판정(8프레임 연속 시 확정). CLIP+MLP/FinoNET/TP-VQA 대비 Acc 0.97로 SOTA |
| **Code-as-Monitor** | 위 표 1 참조 | PARTIAL | subgoal마다 다른 constraint 세트 |
| **DoReMi** | 위 표 1 참조 | PARTIAL | plan step마다 다른 constraint 생성 |
| **Hierarchical Task Decomposition for Execution Monitoring and Error Recovery** | [2505.04565](https://arxiv.org/abs/2505.04565), IJRR 2025 | **YES(비지도 버전)** | 직접 인용: *"unsupervised task segmentation algorithm that combines intention recognition and feature clustering to infer the skills"* → *"leverage the inferred characteristic features of **each skill** in a novel unsupervised anomaly detection approach"*. 즉 skill(≈phase)마다 그 skill 고유의 특징 분포로 이상탐지 기준을 따로 세움. 단, VLA 내부 activation이 아니라 시연 데이터의 kinematic/feature clustering 기반으로 보임(본문 세부는 미확인) |
| **A Unified Framework for Real-Time Failure Handling (VLM + Reactive Planner + Behavior Trees)** | [2503.15202](https://arxiv.org/pdf/2503.15202), 2025 | **PARTIAL(약함)** | precondition 검증(스킬 실행 전) / postcondition 검증(실행 후)의 2단 구조는 있으나, 모든 BT 노드에 **동일한** 3단계(detect/identify/correct) 프로세스를 적용— 노드(phase)별로 별도 검출 로직을 두진 않음. VLM 프롬프트도 노드별 차등화 근거 못 찾음. baseline: pre-execution-only, reactive-only, combined(자기 방법) |
| **Guardian**: Detecting Robotic Planning and Execution Errors with VLM | [2512.01946](https://arxiv.org/abs/2512.01946), 2026 | **PARTIAL(약함, 미확인 다수)** | VQA 입력에 "proposed plans or subtasks"를 **문맥으로** 제공하지만, subtask별로 판정 로직/기준이 구조적으로 달라지는지는 본문 확인 못함. planning-error vs execution-error 구분은 phase 구분이라기보다 오류의 **원인 위치**(계획 vs 실행) 구분에 가까움 |

### 2-C. Stage/phase를 다루지만 실패 검출이 아닌 인접 연구 (참고용, 핵심 질문 범위 밖)

| 논문 | ID/venue | 비고 |
|---|---|---|
| **SARM**: Stage-Aware Reward Modeling for Long Horizon Manipulation | [2509.25358](https://arxiv.org/html/2509.25358v1), 2025 | stage를 예측하나 **reward shaping/behavior-cloning 필터링 전용**. 명시 확인: *"순수 reward shaping 논문. Failure detection 논문이 아님"* — 실패 조기중단·안전제어 목적 없음. stage 라벨도 사람이 수동 annotation. stage별 별도 판정 기준 없음(전체 progress 0~1 연속값 기반) |
| **Play to the Score**: Stage-Guided Dynamic Multi-Sensory Fusion | [2408.01366](https://arxiv.org/abs/2408.01366), CoRL 2024 | "task stage"를 예측해 **센서 모달리티 가중치**를 조절(청각/촉각/시각 우선순위). 실패 검출과 명시적 연결 근거 없음 |
| **KEMO**: Event-Driven Keyframe Memory | [2606.23589](https://arxiv.org/abs/2606.23589), 2026 | 명시 확인: *"이 논문은 failure detection을 다루지 않음"* — stage-critical event는 memory 압축(정책 context 절약)용 |
| **DAISS**: Phase-Aware Imitation Learning (Dual-Arm Ultrasound) | [2603.07663](https://arxiv.org/html/2603.07663), 2026 | phase-aware는 **제어 정책 구조**(양팔 비대칭 조율)를 위한 것, 실패 검출과 무관 |
| **A Physical Agentic Loop... Execution-State Monitoring** | [2604.07395](https://arxiv.org/abs/2604.07395), 2026 | "execution state"는 grasp **primitive 내부**의 결과 상태(empty/slip/stall/timeout) — manipulation phase(reach/grasp/place)가 아니라 grasp 자체의 내부 상태 머신. 명시 확인: *"이는 phase별 조건부 실패검출기가 아니다"* |

### 2-D. 기타 확인된 실패/이상 검출 (phase 조건부 근거 불충분)

| 논문 | ID/venue | 메모 |
|---|---|---|
| **RC-NF**: Robot-Conditioned Normalizing Flow | [2603.11106](https://arxiv.org/abs/2603.11106), CVPR 2026 | "robot-conditioned" = robot/object **state** 조건화이지 task phase 조건화인지 불명확(본문 미확인) |
| **Reliable Robotic Task Execution in the Face of Anomalies** | [2510.23121](https://arxiv.org/abs/2510.23121), RA-L 2025 | vision 기반, nominal-execution 대비 이상탐지. phase 세분화 근거 못 찾음. 3단계 **복구**(pause→perturb→reset) 구조는 있으나 이는 복구 절차지 검출 자체의 phase 조건화가 아님 |
| **FINO-Net** (Multimodal Detection and Identification of Robot Manipulation Failures, 후속판) | [2305.04639](https://arxiv.org/abs/2305.04639) (원조 FINO-Net: [2011.05817](https://arxiv.org/abs/2011.05817)), RA-L 2024 | **Type C 확인**: "manipulation phase failure" vs "post-manipulation phase failure"는 **충돌이 언제 일어났는지를 나타내는 출력 분류 축**("동일한 exteroception 셋업으로 두 phase의 실패를 검출·분류")이지, phase마다 다른 검출기/입력을 쓰는 게 아님. F1: detection 0.87 / classification 0.80 |
| **Critic in the Loop**: Tri-System VLA Framework | [2603.05185](https://arxiv.org/abs/2603.05185), 2026 | Critic이 "active subtask" 실행을 모니터링하며 VLA(routine subtask)와 VLM(replanning)간 권한을 분기 — subtask 인식 자체는 있으나, 이상탐지 로직이 subtask별로 구조적으로 다른지는 abstract 수준에서 확인 불가(본문 미조회) |

### 2-E. 조사 범위 밖(2023년 이전) 역사적 선행연구 — 참고

| 논문 | venue | 비고 |
|---|---|---|
| Multimodal Execution Monitoring for Anomaly Detection During Robot Manipulation (Park, Erickson, Bhattacharjee, Kemp) | ICRA 2016 | **2023~2026 범위 밖**이라 핵심 집계에서 제외하나, 역사적으로 흥미로운 선례: HMM 기반이며 *"detection threshold that changes based on the **execution progress**"* — progress(≈phase)에 따라 임계값이 달라지는 초기 사례. 다만 VLA/딥러닝 internal activation과 무관(고전적 멀티모달 센서 HMM) |

---

## 3. "Phase-conditioned"의 두 갈래 — 왜 구분이 중요한가

조사 결과 phase 관련 표현이 나오는 논문은 많지만, 실제로 뜻하는 바가 크게 다르다:

1. **task-identity 조건화 ≠ phase 조건화.** ActProbe의 "task-conditioned LSTM", SAFE의 multitask 학습 등은 "**어떤 task**인지"에 검출기를 맞추는 것이지, "**task 진행의 어느 시점(phase)**인지"에 맞추는 게 아니다. 이 둘을 혼동하면 안 된다.
2. **출력 레이블의 phase ≠ 입력·검출로직의 phase 조건화.** FINO-Net처럼 "이 실패가 manipulation phase에서 났는지 post-manipulation phase에서 났는지"를 분류하는 것은 phase를 **결과로 보고**하는 것이지, 검출기 자체가 phase별로 달라지는 게 아니다.
3. **심볼릭 사전계획 기반 phase(Type A) ≠ 학습된 내부표현 기반 phase(Type B).** ConditionNET·Code-as-Monitor·DoReMi·Hierarchical Task Decomposition은 모두 (i) LLM이 사전에 짠 plan/subgoal, 또는 (ii) 시연 데이터의 unsupervised skill segmentation으로 phase 경계를 얻고, 그 경계마다 다른 monitoring target(precondition/effect/constraint)을 명시적으로 배정한다. 이건 진짜 phase 조건부 monitoring이 맞지만, **VLA 정책의 internal activation에서 phase를 온라인으로 읽어내는 것과는 다른 정보원**이다 — 대부분 vision(관측)이나 사전정의된 심볼릭 plan을 쓰지, DiT/VLM backbone의 hidden state에서 phase 신호를 뽑지 않는다.
4. **VLA internal-activation 기반 실패 검출기(SAFE, FAIL-Detect, Foresight, VLA-FAIL, Tri-Info, SAFECAST, ActProbe)는 전부 phase 균일 처리.** 이들은 정확히 우리가 쓰려는 신호원(internal activation)을 쓰지만, 전부 전체 rollout에 단일 검출기를 (시간축 conformal band는 조정하되) 적용한다. Foresight는 이를 가장 명시적으로 확인해준다: multi-subgoal long-horizon task를 다루면서도 "phase boundaries나 특수 처리 없이" 연속 처리한다고 직접 서술.

---

## 4. 핵심 질문에 대한 최종 판정

**Q: 기존 실패 검출기 중 task phase/subtask에 조건부로 동작하는 것이 있는가?**

**A: 있다 — 그러나 전부 Type A(심볼릭/계획 기반)이며, Type B(정책 internal activation에서 phase를 읽어 검출기 자체를 분기하는 방식)는 이번 조사(26편, 2023~2026)에서 단 한 건도 발견하지 못했다.**

- Type A의 가장 강력한 예는 **ConditionNET**(RA-L 2025): behavior-tree가 정의한 pre/core/effect 3-phase마다 검증 대상(precondition/제외/effect)을 다르게 두고, core phase에서는 아예 판정을 끈다. Code-as-Monitor·DoReMi도 plan step마다 다른 constraint를 생성해 감시 대상을 바꾼다는 점에서 phase 조건부이나, "검출 알고리즘 자체"(VLM VQA 판정)는 phase 무관하게 동일하다.
- VLA internal-activation을 쓰는 최신 실패 검출기 계열(SAFE와 그 명시적 후속/인접 논문 6편: FAIL-Detect, Foresight, VLA-FAIL, Tri-Info, SAFECAST, ActProbe)은 **단 하나도 phase 조건부가 아니며**, 그 중 Foresight는 이를 명시적으로 부정하는 문장까지 확인했다.

**우리 아이디어("VLA 내부 activation으로 phase를 읽고, phase별 실패 검출기를 두는 것")의 니치 판정:**

**미점유(open) — 이번 조사 범위 내에서 선점 논문 없음.** 구성 요소별로 보면:
- "internal activation에서 succ/fail 분리"는 SAFE 계열이 이미 강하게 점유(단, phase 무관 통일 검출기로).
- "phase를 조건으로 monitoring 대상을 바꾸는 것" 자체는 Type A(심볼릭 계열)가 이미 하고 있음 — 이 조합이 "완전히 새로운 문제"라는 주장은 하면 안 됨(선행 개념은 있음).
- 그러나 "**internal activation에서 phase를 (사전 심볼릭 계획 없이) 온라인으로 읽어내고, 그 phase에 맞춰 학습된 검출기 자체를 분기**"하는 결합은 미발견. 이는 프로젝트 메모리의 기존 판단(`notall-online-failuretype-niche`: "내부 latent×online×실패TYPE×phase-matched steer" 니치가 NOTALL 저자도 안 함)과 일치하며, 이번 조사는 그 판단을 **실패-검출 문헌 전반으로 넓혀 재확인**한 결과다.

주의할 점: AHA·REFLECT·RC-NF·Guardian·Critic-in-the-Loop·Reliable Robotic Task Execution 등 5~6편은 abstract 이상 본문을 조회하지 못해 **phase 조건부 여부가 "미확인"**으로 남아있다. 이 니치 판정은 "확인된 26편 기준"이며, 완전한 반증은 아니다(특히 Guardian처럼 subtask를 VLM 입력 컨텍스트로 주는 논문들은 후속 조사 가치가 있음).
