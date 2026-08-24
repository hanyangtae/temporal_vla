# 참고문헌 목록 및 서론 인용 문장 초안

제7회 한국 인공지능 학술대회 투고(1~2쪽) `manuscript_draft.md`용.

**수록 원칙**: 레포 내 정독 노트(`docs/references/reading_notes/`)나 웹 조회(arXiv abstract)로
**서지가 실측 확인된 항목만** 수록했다. 각 항목 끝에 확인 경로를 `[확인: ...]`로 표기한다.
확인되지 않은 문헌은 넣지 않았다.

---

## 1. 대상 모델·벤치마크 (§2 방법에서 인용)

**[1] GR00T N1**
NVIDIA GEAR Team. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*
arXiv:2503.14734, 2025.
- 왜 인용: 본 연구의 분석 대상 모델. VL backbone(System 2) + diffusion transformer
  action head(System 1)의 이중 구조가 "DiT 잔차 활성화를 캡처한다"는 방법 서술의 전제다.
  우리가 쓰는 GR00T N1.5는 이 모델의 후속 릴리스판이다.
- 인용 위치: §2 방법 첫 문장("GR00T N1.5 rollout을 수집하고"), 초록의 모델명.
- ⚠ 주의: **N1.5 자체의 독립 논문은 확인되지 않았다.** N1 논문을 인용하고 본문에서
  "그 후속 릴리스인 N1.5"로 서술할 것. N1.5용 별도 arXiv 번호를 만들어 쓰지 말 것.
- [확인: 웹 — arxiv.org/abs/2503.14734]

**[2] RoboCasa**
S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, Y. Zhu.
*RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots.*
Robotics: Science and Systems (RSS), 2024. arXiv:2406.02523.
- 왜 인용: rollout 수집 환경. 9개 주방 조작 태스크의 출처이며, 다수 장면·물체 변이가
  "장면 정보 혼입(mi_scene)"을 논할 때의 배경이 된다.
- 인용 위치: §2 방법("RoboCasa 주방 조작 9개 태스크"), §3 장면 정보 문단.
- [확인: 웹 — arxiv.org/abs/2406.02523, RSS 2024]

---

## 2. VLA 내부 표현 해석 (§1 서론에서 인용)

**[3] Dr. VLA (SAE)**
A. Swann, L. McGranahan, H. Buurmeijer, M. Kennedy III, M. Schwager.
*Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models.*
arXiv:2603.19183, 2026.
- 왜 인용: VLA residual stream에 SAE를 학습시켜 **motion primitive에 대응하는 feature**
  (grasp/transport/pre-grasp 등)를 찾아낸 가장 직접적인 선행 연구. 우리와 같은 관심
  (활성화 안의 행동 단계 구조)을 다루지만, feature를 사람이 rollout 영상과 수동 정렬해
  이름 붙이며 **단계 분류 정확도를 정량화하지 않는다** — 우리의 MI·순도·시간 대조군
  margin이 채우는 빈자리다.
- 인용 위치: §1 서론 2번째 문장(선행 연구), §4 결론의 대비.
- [확인: 노트 `docs/references/reading_notes/dr_vla_sae.md` (서지·저자 명시)]

**[4] Observing and Controlling**
H. Buurmeijer*, C. Amo Alonso*, A. Swann, M. Pavone.
*Observing and Controlling Features in Vision-Language-Action Models.*
arXiv:2603.05487, 2026.
- 왜 인용: 선형 observer(probe)로 VLA 층별 상태를 **읽고**, 최소 노름 additive control로
  **쓰는** 것을 분리해 정식화(feature-observability / feature-controllability). "내부 단계를
  읽을 수 있으면 추론 시 제어의 기초가 된다"는 우리 서론 논지의 이론적 근거.
- 인용 위치: §1 서론 3번째 문장("단계 조건부 개입 … 추론 시 제어의 기초"), §4 결론.
- [확인: 노트 `docs/references/reading_notes/observing_controlling.md` (원문 대조 완료 기재)]

**[5] Event-Grounded SAE**
X. Jin, A. Chatterjee, P. Kumar, R. Paleja.
*Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies.*
arXiv:2605.17204, 2026.
- 왜 인용: SAE feature를 **rollout에서 추출한 행동 이벤트(kinematic keyframe cluster)**에
  정렬해 점수화 — "행동 단위로 활성화를 조건화한다"는 발상의 선례. 다만 이벤트를 활성화
  **바깥**(운동학)에서 정의하는 반면, 우리는 활성화 군집화 자체에서 단위를 얻는다는 점이
  대비된다.
- 인용 위치: §1 서론(선행 연구 나열), **§2 방법·§3 결과의 외부 대조군**(이 논문의
  keyframe-이벤트 파이프라인을 저자 코드로 재현해 같은 scene split 판독 프로토콜로 평가
  — `scripts/analysis/grid_phase/esae_{prepare,readout}.py`), 필요 시 §4 한계 문단.
- [확인: 노트 `docs/references/reading_notes/event_grounded_sae.md` (arXiv·저자·코드 URL 명시)]

**[10] AWE**
L. X. Shi, A. Sharma, T. Z. Zhao, C. Finn. *Waypoint-Based Imitation Learning for
Robotic Manipulation.* CoRL 2023. arXiv:2307.14326.
- 왜 인용: 행동 이벤트 대조군([5] 재현)의 waypoint 추출기. §2.3에서 이름 언급.
- [확인: 웹 — arxiv.org/abs/2307.14326, proceedings.mlr.press/v229/shi23b]

**[11] SigLIP**
X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer. *Sigmoid Loss for Language Image
Pre-Training.* ICCV 2023. arXiv:2303.15343.
- 왜 인용: 행동 이벤트 대조군의 시각 임베딩. §2.3에서 이름 언급.
- [확인: 웹 — arxiv.org/abs/2303.15343]

---

## 3. 활용처 근거 — steering / 실패 감지 (§1 도입부, §4 결론에서 인용)

**[6] Häon et al. (mechanistic steering)**
B. Häon*, K. Stocking*, I. Chuang, C. Tomlin.
*Mechanistic Interpretability for Steering Vision-Language-Action Models.*
Conference on Robot Learning (CoRL), 2025. arXiv:2509.00328.
- 왜 인용: VLA 내부 뉴런 override만으로 fast/slow 같은 행동 축을 zero-shot 제어할 수
  있음을 보인 연구. "내부 표현 판독 → 개입"이 실제로 작동한다는 활용처 근거.
- 인용 위치: §1 서론("단계 조건부 개입 … 의 기초"), §4 결론 마지막 문장.
- [확인: 노트 `docs/Activation_steering_basic/notes/MechInterpSteering.md` (저자·arXiv·venue 명시,
  레포 자체 재현 기록 `docs/steering/16_mechinterp_reproduction.md` 존재)]

**[7] SAFE (실패 감지)**
Q. Gu, Y. Ju, S. Sun, I. Gilitschenski, H. Nishimura, M. Itkina, F. Shkurti.
*SAFE: Multitask Failure Detection for Vision-Language-Action Models.*
Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2506.09937.
- 왜 인용: VLA 내부 hidden state로 실패를 온라인 검출하는 대표 연구. **단일 검출기를
  rollout 전체에 균일 적용**하고 단계 분기가 없으므로, "세밀 단계를 온라인 판독할 수
  있으면 단계 조건부 검출로 확장할 수 있다"는 우리 결론의 대비 대상이 된다.
- 인용 위치: §1 서론("실패 감지 등"), §4 결론 마지막 문장.
- [확인: 노트 `docs/references/reading_notes/phase_detection/A_failure_detection_landscape.md`
  + 웹 arxiv.org/abs/2506.09937 (저자 전체·NeurIPS 2025 확인)]

---

## 4. 비지도 분절 / 시간 추상화 맥락 (§1 서론에서 인용)

**[8] LOTUS**
W. Wan, Y. Zhu, R. Shah, Y. Zhu.
*LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill
Discovery.* IEEE International Conference on Robotics and Automation (ICRA), 2024.
arXiv:2311.02058.
- 왜 인용: 라벨 없는 시연 데이터를 계층적 군집화로 분절해 skill을 발견 — 우리와 같은
  "비지도 분절" 방법론이지만 **외부 관측(DINOv2 비전 feature)** 위에서 수행한다. 우리는
  같은 질문을 **정책 내부 활성화** 위에서 묻는다는 차별점을 세우는 데 쓴다.
- 인용 위치: §1 서론(비지도 분절 선행 연구 대비).
- [확인: 노트 `.../phase_detection/B_phase_from_internals.md` 표 #11 + 웹 arxiv.org/abs/2311.02058
  (제목·저자·ICRA 2024 확인)]

**[9] Options (시간 추상화 고전)**
R. S. Sutton, D. Precup, S. Singh.
*Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement
Learning.* Artificial Intelligence, 112(1-2):181–211, 1999.
- 왜 인용: 행동을 시간적으로 확장된 단위(option)로 묶는 고전적 정식화. "사람이 정의한
  primitive 단계"라는 우리 서론의 출발 개념이 어디서 왔는지 한 줄로 지시한다.
- 인용 위치: §1 서론 첫 문장("reach → grasp → transport 같은 primitive 단계로 기술된다").
- [확인: 웹 — Artificial Intelligence Vol.112 No.1-2, pp.181–211 (ACM DL)]

---

## 5. 선택 인용 (지면 여유 시)

**[10] PAMAE**
J. Yang, T. Yang, X. Chang, F. Chao, C. Shang, Q. Shen.
*PAMAE: Phase-Aware-MoE Action Experts Towards Reliable Flow-Matching Vision-Language-Action
Policies.* arXiv:2606.27144, 2026.
- 왜 인용: flow-matching action expert를 **단계 인지 라우터**로 분기시켜 성공률을 최대 9.2%p
  올린 연구. "단계를 읽으면 실제로 성능이 오른다"는 활용처 근거로는 가장 직접적이다.
  단 단계 라벨을 gripper·속도 임계값 휴리스틱으로 만들며 활성화에서 읽지 않는다.
- 인용 위치: §4 결론("단계 조건부 개입의 기초 신호") 보강, 또는 [6]과 묶어 서론.
- [확인: 노트 `.../phase_detection/B_phase_from_internals.md` 표 #1 + 웹 arxiv.org/abs/2606.27144]

---

## 6. 서론용 한글 문장 초안

1~2쪽 분량이라 관련연구 절을 따로 두지 않고 §1 서론에 녹이는 것을 전제로 한다.
현재 `manuscript_draft.md` §1의 2~3번째 문장을 아래로 교체·확장하는 안이다.

> **(A) 선행 연구 위치 잡기 — 서론 2번째 문장 자리)**
> VLA 내부 표현을 해석하려는 최근 시도들은 희소 오토인코더(SAE)로 residual stream에서
> 파지·이송 같은 운동 primitive에 대응하는 feature를 찾아내거나[3,5], 선형 probe로 층별
> 상태를 관측하고 그 방향으로 활성화를 밀어 행동을 제어할 수 있음을 보였다[4,6]. 그러나
> 이들은 발견한 feature를 사람이 rollout 영상과 수동으로 정렬해 이름 붙일 뿐, 그 표현이
> 사람이 정의한 행동 단계와 **어떤 해상도로** 대응하는지를 정량화하지 않는다.

> **(B) 비지도 분절과의 대비 — 서론 3번째 문장 자리)**
> 한편 라벨 없이 조작 궤적을 단위 행동으로 분절하려는 연구는 오래된 시간 추상화
> 정식화[9]에서 출발해 최근에는 시연 데이터의 시각 특징을 군집화하는 방식으로
> 이어지지만[8], 분절의 근거를 정책 **바깥**의 관측에서 찾는다. 정책이 스스로 무엇을
> 하고 있는지가 내부 활성화에 어떤 단위로 적혀 있는지는 별개의 질문이다.

> **(C) 활용처 — 서론 4번째 문장 자리)**
> 내부 단계를 추론 중에 읽을 수 있으면 단계 조건부 개입[6,10]이나 실패 감지[7]의 조건
> 신호로 쓸 수 있다. 특히 VLA 내부 표현 기반 실패 검출기는 지금까지 rollout 전체에 단일
> 검출기를 균일하게 적용해 왔다[7].

> **(D) 본 연구 — 서론 마지막 문장 자리, 기존 문장 유지·보강)**
> 본 연구는 GR00T N1.5[1]가 RoboCasa[2] 주방 조작을 수행하는 동안의 action head 활성화를
> 라벨 없는 군집화만으로 분석하고, 사람이 붙인 단계 라벨과의 관계를 시간 대조군 대비
> margin으로 정량화한다.

### 사용 시 주의

- [1]은 N1이고 우리 모델은 N1.5다. 본문에서 "GR00T N1[1]의 후속 릴리스인 N1.5" 형태로 쓸 것.
- (A)의 "정량화하지 않는다"는 [3]·[5] 노트에서 확인된 사실(단계 분류 정확도 미보고)에
  근거하나, 표현을 더 약하게 하려면 "체계적으로 정량화한 사례는 드물다"로 완화 가능.
- §3의 성공/실패 분리도 결과는 탐색적 관찰이므로 [7]과 나란히 놓되 우열 주장은 하지 말 것.
