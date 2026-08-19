# KAI2026 비교 베이스라인 후보 조사 — 로봇 조작 궤적의 action phase 분절·검출

조사일 2026-08-19. 대상 논문: `manuscript_draft.md` (제7회 한국 인공지능 학술대회, 최대 2쪽).

**우리 방법(비교 대상)**: GR00T N1.5의 action head(DiT) 잔차 활성화 → PCA-64(whitening) → KMeans로
스텝별 군집화. 학습형 라벨 불사용. 사람 GT phase 라벨은 **평가에만** 사용하고,
MI(군집; GT 단계)·순도(purity)·**시간 대조군 margin**·경계 정렬 z-score로 정량한다.
train split에서만 군집을 만들고 평가 split은 k-NN 전이 → 추론 중 스텝 단위 온라인 판독 가능.

**수록 규율**: 서지가 실측 확인된 항목만 수록한다. 각 항목에 확인 URL을 적었다.
확인 실패·오인용 위험 항목은 §6에 따로 모았다. **arXiv 번호·venue를 추측해서 붙이지 않는다.**

기존 레포 노트: [`docs/references/reading_notes/phase_detection/B_phase_from_internals.md`](../../references/reading_notes/phase_detection/B_phase_from_internals.md)
(내부표현 phase 판독 20편 표), [`A_failure_detection_landscape.md`](../../references/reading_notes/phase_detection/A_failure_detection_landscape.md)
(phase 조건부 실패검출 26편), [`event_grounded_sae.md`](../../references/reading_notes/event_grounded_sae.md).
본 문서는 그 표에서 **"phase 분절 baseline"** 각도로 필요한 계열(vision 군집·waypoint·고전 changepoint·
video TAS)을 새로 보강하고, 기존 후보(PAMAE·LOTUS 등)는 서지·주장 재검증만 했다.

---

## 0. 한눈에 — 분절 근거 신호 × 라벨 × 온라인

| 계보 | 대표 논문 | 분절 근거 신호 | 라벨 | 온라인 |
|---|---|---|---|---|
| 조작 skill discovery (비전 군집) | BUDS, LOTUS, XSkill | 외부 pretrained vision encoder feature | 불필요 | offline batch |
| 시연 분해 / subgoal·waypoint | UVD, AWE, keyframe heuristic | 외부 시각표현 임베딩 거리 / EEF 기하 / 관절속도·gripper 규칙 | 불필요 | offline 전처리(규칙은 원리상 online) |
| 고전 changepoint / 궤적 분절 | TSC, TSC-DL, BP-AR-HMM, CHAMP, BOCPD | 관측 kinematics(+vision) 시계열 | 불필요 | 대부분 offline, CHAMP·BOCPD만 online |
| Video temporal action segmentation | CTE, TW-FINCH / MS-TCN, ASFormer | 비디오 프레임 feature (+시간) | 불필요 / frame-level 라벨 필요 | offline batch |
| phase 라벨을 정의해 주입 (VLA) | PAMAE, Move-Then-Operate, SARM2 | gripper·속도 규칙 / MLLM 주석 / 사람 주석 | 휴리스틱·MLLM·사람 | online (라우터) |
| 정책 **출력**에서 경계 검출 | PACE | 예측 action chunk의 속도 골짜기 | 불필요 | online |
| 정책 **내부 표현** 해석 | SAE-VLA(2603.19183), Event-Grounded SAE, Observing&Controlling, LAR-MoE | 내부 활성화(단 분절 기준은 대부분 외부) | 육안·물리 event | 대체로 offline 분석 |
| **본 연구** | — | **정책 내부 DiT 활성화 자체** | 불필요(GT는 평가용) | **스텝 단위 가능** |

**핵심 판정 (§5에 근거)**: 정책 내부 활성화를 **라벨 없이 군집화해 사람 phase GT에 대해 정량 검증**한
선행 연구는 확인된 범위에 **없다**. 가장 가까운 LAR-MoE는 정성 비교(Figure 3)뿐이고,
PAMAE는 purity 계열 지표를 쓰지만 그 대상이 phase 판독 정확도가 아니라 expert 라우팅 상관이다.

---

## 1. 시각 feature 군집화 기반 skill discovery / 분절

### 1.1 LOTUS (기존 인용 [8], 재검증 완료)
- W. Wan, Y. Zhu, R. Shah, Y. Zhu. *LOTUS: Continual Imitation Learning for Robot Manipulation
  Through Unsupervised Skill Discovery.* ICRA 2024. arXiv:2311.02058.
  [확인: https://arxiv.org/abs/2311.02058 , 방법 세부 https://arxiv.org/html/2311.02058v3 ]
- 분절 신호: **vision only** — 프레임의 DINOv2 feature를 세그먼트 단위로 pooling,
  인접 세그먼트 cosine 유사도로 hierarchical agglomerative clustering → spectral clustering.
- 라벨 불필요 / **온라인 불가(offline batch, 학습 전처리)**.
- 비교 가치 **높음** — "군집화로 skill을 찾는다"의 최신 대표이며, 우리와 대비축(외부 vision encoder
  feature vs 정책 내부 DiT 활성화)이 가장 선명하다. 이미 초안 [8]로 인용 중.

### 1.2 BUDS
- Y. Zhu, P. Stone, Y. Zhu. *Bottom-Up Skill Discovery from Unsegmented Demonstrations for
  Long-Horizon Robot Manipulation.* IEEE RA-L 7(2):4126–4133, 2022. arXiv:2109.13841.
  [확인: https://arxiv.org/abs/2109.13841 (Comments에 RA-L 2022 명시) ]
- 분절 신호: vision + 데모 시계열의 계층적 agglomerative clustering, 다중 데모 간 반복 패턴으로 skill 정의.
- 라벨 불필요 / offline batch.
- 비교 가치 **중~높** — LOTUS의 직계 선행. 2쪽 지면에서는 LOTUS와 한 문장으로 묶는 편이 경제적.

### 1.3 XSkill
- M. Xu, Z. Xu, C. Chi, M. Veloso, S. Song. *XSkill: Cross Embodiment Skill Discovery.*
  CoRL 2023. arXiv:2307.09955. [확인: https://arxiv.org/abs/2307.09955 ]
- 분절 신호: vision only, self-supervised skill prototype(군집 중심에 해당) 학습.
- 라벨 불필요 / prototype 할당은 클립 단위 계산 가능하나 논문 세팅은 offline + prompt video.
- 비교 가치 **중** — 초점이 cross-embodiment 전이라 우리 축과 직접 대비는 약하다. 지면 여유 시에만.

### 1.4 Hierarchical Task Decomposition (Willibald & Lee)
- C. Willibald, D. Lee. *Hierarchical Task Decomposition for Execution Monitoring and Error
  Recovery: Understanding the Rationale Behind Task Demonstrations.* IJRR (accepted). arXiv:2505.04565.
  [확인: https://arxiv.org/abs/2505.04565 ]
- 분절 신호: intention recognition + **feature clustering**(상태·관계 feature, 순수 vision 아님).
- 라벨 불필요(비지도 분절) / 분절은 offline이나 결과 표현을 **실행 감시·이상탐지에 온라인 활용**.
- 비교 가치 **중~높** — "분절된 phase 단위마다 별도 감시 기준을 세운다"는 목적이 우리의 최종 동기
  (phase 조건부 개입·검출)와 겹친다. 한 줄 인용 가치 있음.

### 1.5 DexSkills (대조 칸 채우기용)
- X. Mao, G. Giudici, C. Coppola, K. Althoefer, I. Farkhatdinov, Z. Li, L. Jamone.
  *DexSkills: Skill Segmentation Using Haptic Data for Learning Autonomous Long-Horizon Robotic
  Manipulation Tasks.* arXiv:2405.03476. [확인: https://arxiv.org/abs/2405.03476 ]
- 분절 신호: haptic/tactile+proprio, **supervised**(사전정의 primitive 라벨), 학습된 분류기라 스텝 단위 적용 가능.
- 비교 가치 **낮~중** — §0 표의 "proprio·supervised" 칸을 채우는 용도.

---

## 2. 시연 분해 / waypoint·subgoal 추출

### 2.1 UVD — Universal Visual Decomposer
- Z. Zhang, Y. Li, O. Bastani, A. Gupta, D. Jayaraman, Y. J. Ma, L. Weihs.
  *Universal Visual Decomposer: Long-Horizon Manipulation Made Easy.* ICRA 2024. arXiv:2310.08581.
  [확인: https://arxiv.org/abs/2310.08581 ; venue https://zcczhang.github.io/UVD/ ]
- 분절 신호: **vision only** — 사전학습 시각표현(R3M 등) 임베딩 공간의 phase shift(임베딩 거리
  단조성 붕괴 지점) 검출. 군집화가 아니라 거리 기반 경계 검출이며 추가 학습 비용 0.
- 라벨 불필요 / 논문 용도는 데모 offline 분해(원리상 인과적 검출에 가까움).
- 비교 가치 **높음** — "외부 pretrained 표현만으로 subgoal 경계가 나온다"는 가장 강한 반대 서사.
  우리가 굳이 정책 내부를 봐야 하는 이유를 세우는 대비점.

### 2.2 AWE — Automatic Waypoint Extraction
- L. X. Shi, A. Sharma, T. Z. Zhao, C. Finn. *Waypoint-Based Imitation Learning for Robotic
  Manipulation.* CoRL 2023 (PMLR v229). arXiv:2307.14326.
  [확인: https://arxiv.org/abs/2307.14326 , https://proceedings.mlr.press/v229/shi23b.html ]
- 분절 신호: **kinematics만** — 선형 보간이 오차 임계 이내로 궤적을 근사하는 최소 waypoint 집합(DP).
- 라벨 불필요(임계값 하이퍼파라미터만) / offline 전처리(전 궤적 필요).
- 비교 가치 **중~높** — "순수 기하 분절"의 대표. 우리 군집이 단순 기하 분절로 환원되지 않음을
  주장할 때의 대조군. 참고로 Event-Grounded SAE(§5.4)가 event anchor 추출에 AWE를 그대로 쓴다.

### 2.3 Keyframe heuristic (Q-attention 계열, RLBench 표준)
- S. James, A. J. Davison. *Q-attention: Enabling Efficient Learning for Vision-based Robotic
  Manipulation.* arXiv:2105.14829 — keyframe 기준 명시: 관절속도 ≈0 + gripper open state 변화.
  [확인: https://arxiv.org/abs/2105.14829 , https://ar5iv.labs.arxiv.org/html/2105.14829 ]
  - 확장: S. James, K. Wada, T. Laidlow, A. J. Davison. *Coarse-to-Fine Q-attention.* CVPR 2022.
    arXiv:2106.12534. [확인: https://arxiv.org/abs/2106.12534 ]
  - 널리 인용되는 채택 사례: M. Shridhar, L. Manuelli, D. Fox. *Perceiver-Actor.* CoRL 2022.
    arXiv:2209.05451, §3.2에 동일 규칙. [확인: https://ar5iv.labs.arxiv.org/html/2209.05451 ]
- 분절 신호: proprio(관절속도) + gripper 상태 전이. 라벨 불필요, 규칙 자체는 스텝 단위 판정 가능.
- 비교 가치 **높음** — **우리 사람 GT phase 라벨이 사실상 이 계열 규칙의 수동판**임을 밝히고,
  "규칙 기반 분절 대비 무라벨 활성화 군집이 무엇을 더/덜 잡는가"로 서술하기 좋다.
- ⚠ arXiv:2105.14829의 저널 venue는 확인 실패 — "arXiv preprint"로만 쓰거나 CVPR 2022판을 인용할 것.

---

## 3. 고전 changepoint / trajectory segmentation

### 3.1 TSC — Transition State Clustering
- S. Krishnan, A. Garg, S. Patil, C. Lea, G. Hager, P. Abbeel, K. Goldberg.
  *Transition State Clustering: Unsupervised Surgical Trajectory Segmentation for Robot Learning.*
  ISRR 2015 (Springer SPAR, pp. 91–110); 저널판 IJRR 36(13-14):1595–1618, 2017.
  [확인: https://link.springer.com/chapter/10.1007/978-3-319-60916-4_6 ,
  https://journals.sagepub.com/doi/10.1177/0278364917743319 ] — **arXiv id 없음(붙이지 말 것)**.
- 분절 신호: 관측 kinematics 기반 전이점 후보 + 시각·시간 feature 상관을, 다수 시연을 가로질러 군집.
- 라벨 불필요(단 반복 시연 집합 전제) / offline(GMM·DP 계층 군집).
- 비교 가치 **높음** — 우리와 구조가 가장 닮은 고전 baseline. 발상("상태공간 군집으로 전이 발견")이
  동일하고 차이는 입력이 관측 kinematics냐 정책 내부 활성화냐 하나로 압축된다.

### 3.2 TSC-DL
- A. Murali, A. Garg, S. Krishnan, F. T. Pokorny, P. Abbeel, T. Darrell, K. Goldberg.
  *TSC-DL: Unsupervised Trajectory Segmentation of Multi-Modal Surgical Demonstrations with Deep
  Learning.* ICRA 2016. [확인: http://berkeleyautomation.github.io/tsc-dl/ ] — arXiv id 없음.
  (흔히 인용되는 "Transition State Clustering with Deep Learning"은 약칭이며 정식 제목은 위와 같다.)
- 분절 신호: vision(사전학습 CNN feature) + kinematics 멀티모달. 라벨 불필요, offline.
- 비교 가치 **중** — "학습된 심층 표현을 분절 신호로"의 선구지만 표현이 여전히 외부 인코더라
  UVD와 역할이 겹친다. 지면 부족 시 생략 가능.

### 3.3 BP-AR-HMM 시연 분절 (Niekum et al.)
- S. Niekum, S. Osentoski, G. D. Konidaris, A. G. Barto. *Learning and Generalization of Complex
  Tasks from Unstructured Demonstrations.* IROS 2012, pp. 5239–5246.
  [확인: https://ieeexplore.ieee.org/abstract/document/6386006/ ]
  - 확장 저널판: + S. Chitta, B. Marthi. *Learning Grounded Finite-State Representations from
    Unstructured Demonstrations.* IJRR 34(2):131–157, 2015.
    [확인: https://journals.sagepub.com/doi/abs/10.1177/0278364914554471 ] — arXiv id 없음.
- 분절 신호: proprio/kinematics 시계열의 autoregressive dynamics 전환. BP-AR-HMM으로 **mode 수를
  비모수 추론**하고 시연 간 mode 공유. 라벨 불필요 / offline(MCMC).
- 비교 가치 **높음** — "phase 개수를 데이터에서 비모수 추정"이 우리의 K 선택(k=24 / 태스크별 k=8)
  논의와 직접 맞물린다. 우리 KMeans 고정 K의 한계를 자인하며 인용하기 좋다.

### 3.4 CHAMP — 온라인 Bayesian changepoint
- S. Niekum, S. Osentoski, C. G. Atkeson, A. G. Barto. *Online Bayesian Changepoint Detection for
  Articulated Motion Models.* ICRA 2015, pp. 1468–1475.
  [확인: https://ieeexplore.ieee.org/document/7139383/ , https://people.cs.umass.edu/~sniekum/pubs/CPD15.pdf ]
- 관측 pose 시계열에 articulation 모델을 적합해 모델 전환점 검출. **온라인 가능**(이 목록에서
  로보틱스 쪽 유일). 비교 가치 **중** — 우리 "추론 중 스텝 단위 판독" 지향의 관측기반 선행.

### 3.5 BOCPD (고전 원전)
- R. P. Adams, D. J. C. MacKay. *Bayesian Online Changepoint Detection.* arXiv:0710.3742, 2007.
  [확인: https://arxiv.org/abs/0710.3742 ] — 도메인 무관 온라인 changepoint의 뿌리.
  비교 가치 **중**. 지면 부족 시 CHAMP만 남겨도 무방.

---

## 4. Video temporal action segmentation (TAS)

**로봇 조작 데이터에 직접 적용된 사례는 확인하지 못했다.** 조작 도메인은 별도 계보
(BUDS/LOTUS/UVD)로 발전했고, TAS는 Breakfast·50Salads·YouTube Instructions 같은 일반 비디오
벤치마크에 머문다. 이 **두 계보의 분리** 자체가 우리 논문의 위치 선정 문장으로 쓸 수 있다.

### 4.1 CTE (비지도)
- A. Kukleva, H. Kuehne, F. Sener, J. Gall. *Unsupervised Learning of Action Classes with
  Continuous Temporal Embedding.* CVPR 2019, pp. 12066–12074. arXiv:1904.04189.
  [확인: https://arxiv.org/abs/1904.04189 , CVF openaccess 페이지 ]
- 프레임 feature + **상대 시간(temporal embedding)** 을 잠재공간에 넣고 군집화. 비지도(클러스터 수 지정), offline.
- 비교 가치 **높음** — 우리 파이프라인(표현 → 차원축소 → KMeans → GT 대비 MI/purity)의 방법론적 조상이고,
  TAS의 표준 평가 관행(MoF/purity, Hungarian matching)을 차용했다고 쓸 수 있다.
  ★ **시간 임베딩을 넣어 성능을 얻는다는 사실 자체가, 우리 시간 대조군(clock) margin이 필요한 이유의
  직접 문헌 근거**다 — 리뷰어 방어에 특히 유용.

### 4.2 TW-FINCH (비지도, training-free)
- M. S. Sarfraz, N. Murray, V. Sharma, A. Diba, L. Van Gool, R. Stiefelhagen.
  *Temporally-Weighted Hierarchical Clustering for Unsupervised Action Segmentation.* CVPR 2021.
  arXiv:2103.11264. [확인: https://arxiv.org/abs/2103.11264 ]
- 프레임 feature + 시간 근접 가중 1-NN 그래프 계층 군집, **학습 불필요**, offline.
- 비교 가치 **높음** — 학습 없는 군집만으로 강한 성능이라는 점이 우리 "PCA-64→KMeans라는 단순 절차"
  선택의 정당화가 된다. 동시에 시간 가중이 성능의 큰 몫이라는 점도 clock 대조군 필요성의 근거.

### 4.3 MS-TCN / ASFormer (지도)
- Y. Abu Farha, J. Gall. *MS-TCN: Multi-Stage Temporal Convolutional Network for Action
  Segmentation.* CVPR 2019. arXiv:1903.01945. [확인: https://arxiv.org/abs/1903.01945 ]
- F. Yi, H. Wen, T. Jiang. *ASFormer: Transformer for Action Segmentation.* BMVC 2021.
  arXiv:2110.08568. [확인: https://arxiv.org/abs/2110.08568 ]
- 둘 다 vision, **frame-level 라벨 필요**, offline(전 시퀀스, 인과적 아님).
- 비교 가치 **중~낮** — "라벨이 있으면 잘 되지만 프레임 단위 주석이 필요하다"는 한 문장으로 묶어 처리.
- (후속 TAEC, arXiv:2303.05166, CVWW 2023 — CTE 개선판. 2쪽 논문에서는 CTE로 대표하면 충분.)

---

## 5. 정책 내부 표현 / 출력에서 phase를 읽는 직접 경쟁자

기존 노트 B의 후보 11편은 이번에 **arXiv id·저자·venue를 전부 재확인했고 id 오류는 0건**이었다
(단 표기 주의 3건은 §6). 여기서는 "phase 판독을 정량 보고하는가"를 기준으로 재정리한다.

### 5.1 LAR-MoE — **최근접 경쟁자**
- A. Rodriguez, C. Li, L. Mazza, R. Younis, O. Hellig, S. Bodenstedt, M. Wagner, S. Speidel.
  *LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning.*
  arXiv:2603.08476, 2026-03. [확인: https://arxiv.org/abs/2603.08476 , html v1 ]
- 분절 근거: **잠재 표현** — 단 동결된 정책 내부 활성화가 아니라, obs↔future-action을 잇도록 별도
  co-training한 student 인코더 출력 ẑ_t. 라우팅 p_t = softmax(T·MLP(ẑ_t)).
- phase 라벨: **획득하지 않음(label-free)**. 사람 주석 phase 분할과 expert 활성화를
  **Figure 3의 시각 비교로만** 제시, **정량 지표 없음**. 대신 supervised-MoE 대비 downstream SR.
- 라벨 불필요 / 온라인 가능(라우팅이 추론 시 동작).
- 비교 가치 **높음** — "phase 주석 없이도 phase 구조가 잠재공간에 뜬다"는 우리와 같은 주장을 하되
  정량화를 하지 않았다. 리뷰어가 가장 먼저 들이밀 논문이므로 **반드시 인용하고 차이를 명시**할 것.
- 축 차이: 새로 학습한 라우팅 전용 잠재 + 정성 비교 / 우리는 **동결 GR00T N1.5 DiT 활성화** +
  MI·purity·시간 대조군 margin 정량.

### 5.2 PAMAE — proprio/gripper 휴리스틱 phase의 정본
- J. Yang, T. Yang, X. Chang, F. Chao, C. Shang, Q. Shen. *PAMAE: Phase-Aware-MoE Action Experts
  Towards Reliable Flow-Matching Vision-Language-Action Policies.* arXiv:2606.27144, 2026-06.
  [확인: https://arxiv.org/abs/2606.27144 , html v1 ]
- 분절 근거: **gripper 개폐 + EEF 속도 임계 규칙**으로 pre-contact / contact-manipulation /
  post-contact-transport 3단계 pseudo-label 자동 생성. 라우터 입력에 backbone context가 들어가나
  phase 정의 자체는 물리 규칙.
- 정량: 보조 phase-prediction head(CE)는 있으나 **분류 정확도 미보고**. 유일한 관련 수치는
  **Phase-Conditioned Dominance Purity(PCP) 89.0%** — 이는 expert 라우팅이 phase와 얼마나
  상관되는지이지 phase 판독 정확도가 아니다. downstream 최대 +9.2%p SR.
- 라벨 자동(사람 불필요) / 온라인 가능.
- 비교 가치 **높음** — 휴리스틱 phase 라벨 대조군의 정본이자, **purity 계열 지표를 쓴다는 점에서
  우리 지표와 직접 대응**한다. 초안에서는 선택 인용 [10].
- 축 차이: 물리 임계값으로 phase를 **정의해 주입** / 우리는 내부표현에서 **발견**하고 GT로 검증.

### 5.3 PACE — 정책 **출력**에서 경계 검출
- J. Nie, J. Li, J. Zhang, J. Lao, C. Liu, T. Zhang, L. Lin, S. Huang. *PACE: Phase-Aware Chunk
  Execution for Robot Policies with Action Chunking.* arXiv:2606.00537 (v1 2026-05, v2 2026-07).
  [확인: https://arxiv.org/abs/2606.00537 , html v2 ]
- 분절 근거: 예측된 action chunk의 speed profile **저속 골짜기**를 경계 후보로. 내부 활성화 미사용.
- 정량: **GT 경계 대조 없음**, SR만(RoboTwin2 57.8→64.2, ALOHA 60.7→77.7, 실기 50.7→70.4).
  본문에서 "저속 골짜기로 표현되지 않는 결정점은 신호가 약하다"고 한계를 자인.
- label-free, training-free, **온라인 가능**.
- 비교 가치 **높음** — "phase를 라벨 없이 뽑는다"의 출력측 대표이자 우리와 정확히 대칭인 대안 신호원
  (내부 vs 출력). 게다가 우리처럼 GT 대조를 하지 않았다는 점이 우리 기여를 부각한다.

### 5.4 Event-Grounded SAE — 클러스터링하되 입력이 물리 상태
- X. Jin, A. Chatterjee, P. Kumar, R. Paleja. *Event-Grounded Sparse Autoencoders for
  Vision-Language-Action Policies.* arXiv:2605.17204, 2026-05.
  [확인: https://arxiv.org/abs/2605.17204 , html v1 ; 레포 정독 노트 `event_grounded_sae.md` ]
- 분절 근거: **혼합** — 군집 대상이 내부 활성화가 아니라 (visual embedding ⊕ EEF pose·gripper ⊕
  temporal progress) 기술자. AWE 키프레임 → agglomerative(cosine thr 0.18), suite당 36–61 클러스터.
  그 후 SAE feature를 이 event에 grounding.
- VLM이 pre_grasp/contact/release 등 라벨을 붙이지만 **"시각화 보조일 뿐 랭킹·개입 점수에 미사용"**
  이라 명시. **phase 정합 정량 지표 없음**, 평가는 closed-loop SR 하락.
- 비교 가치 **높음** — 우리와 가장 헷갈리기 쉬운 논문(둘 다 "클러스터링으로 phase 유사 구조").
  ⚠ 이미 초안 [5]로 인용 중이며, **"분절 자체는 외부 물리 신호로 하고 활성화는 설명 대상"**이라는
  차이를 정확히 써야 오인용을 피한다.

### 5.5 SAE-VLA (초안 [3])
- A. Swann, L. McGranahan, H. Buurmeijer, M. Kennedy III, M. Schwager. *Sparse Autoencoders Reveal
  Interpretable and Steerable Features in VLA Models.* arXiv:2603.19183 (v1 2026-03, v2 2026-06).
  [확인: https://arxiv.org/abs/2603.19183 , html v2 ]
- 내부 활성화(π0.5 PaliGemma 0/5/11/17 + action expert). motion-primitive/sub-task 전이에 반응하는
  feature 존재(F158 approach/grasp/move 전환 등)를 보이나 **라벨은 사람 육안 부여**, phase 정합
  정량 지표 없음. 보고 수치는 해석가능률 79.2%(95/120), ablation SR 97.5%→0%, 30개 feature LOO 100%.
- 비교 가치 **높음** — "내부에 primitive·전환 feature가 있다"는 **존재 주장**의 대표 선행연구.
- 축 차이: 단일 feature의 사후 육안 해석 / 우리는 다차원 상태 군집 + GT 대조 정량.
- ⚠ 제목에 "Dr.VLA"라는 표기는 없다(내부 별칭). 인용 시 정식 제목만 쓸 것.

### 5.6 그 밖 (내부표현이되 축이 다름 — 서술 배경용)
| 논문 | 서지·확인 | 신호 | 축 | 가치 |
|---|---|---|---|---|
| Observing and Controlling Features in VLA Models (초안 [4]) | Buurmeijer, Amo Alonso, Swann, Pavone. arXiv:2603.05487 [ https://arxiv.org/abs/2603.05487 ] | 내부 활성화 선형 프로브+개입 | **저수준 물리량**(EEF x/y/z·rpy·gripper aperture)만, "추상 semantic feature는 future work"라 명시 | 중 — 내부 판독 가능성 근거 |
| Move-Then-Operate | Xu, Lei, Gu, Tang, Chen, Wang. arXiv:2604.23620 [ https://arxiv.org/abs/2604.23620 ] | 학습 라벨=MLLM(vision), 추론=VLM hidden state MLP 라우터 | move/operate 2-phase **지도** 라우터, 라우터 정확도 미보고 | 중 — 지도 대조군 |
| SARM2 | Chen, Zheng, Yu, Huang, Sun, Goldberg, Wen, Abbeel, Shentu, Wu, Schwager. arXiv:2606.10305 [ https://arxiv.org/abs/2606.10305 ] | vision+proprio 외부 causal Transformer | 21 primitive **사람 주석** 200시간, stage 분류 정확도 미보고 | 중 — 완전지도 극단(라벨 비용 대비축) |
| SAFE (초안 [7]) | Gu, Ju, Sun, Gilitschenski, Nishimura, Itkina, Shkurti. NeurIPS 2025. arXiv:2506.09937 | 내부 feature | 실패확률 스칼라(지도), phase 없음 | 중 — 내부표현이 고수준 정보를 담는 근거 |
| What Frozen VLAs Already Know About Success | arXiv:2605.28527 [ https://arxiv.org/abs/2605.28527 ] | frozen 내부 feature 선형 프로브 | outcome/value 회귀, matched-pair 통제가 엄격 | 중 — confound 통제 방법론 참조 |
| Hide-and-Seek in Trajectories | Park, Li, Oh, Yeh, Kira, Hagenow, Li. arXiv:2605.30834 | 내부 action embedding | trajectory 라벨만으로 **실패 시점 국소화**(약지도) | 중 — 시간구조 유도 방법론 근접, 축은 실패 |
| ProgVLA | Kim, Choi, Baek, Renders. arXiv:2605.28231 [ https://arxiv.org/abs/2605.28231 ] | 내부표현 위 auxiliary progress head | **연속 progress**(지도), 이산 phase 아님, 판독 정량 미보고 | 중 — "내부에서 진행도를 읽는" 지도 대조군 |
| ProbeAct | arXiv:2606.09740 | 내부 hidden state 프로브 | 대상이 **3D 물체 위치** | 낮 |
| PALM (CVPR 2026) | Liu et al. arXiv:2601.07060 | vision affordance | 연속 progress 지도 예측 | 낮 |
| SMP / Abstracting Robot Manipulation Skills via MoE Diffusion Policies | Hao, Zhai, Liu, Soh. arXiv:2601.21251 | orthogonal skill basis + sticky routing | GT subtask 정합 정량 **없음** | 낮~중 |
| S²-VLA (IJCAI 2026) | Xie et al. arXiv:2606.27872 | 명시적 belief-state 모듈 | 진행도 추적 gating, 내부 활성화 판독 아님 | 낮 |

### 5.7 판정 — 직접 경쟁자 유무
**없다.** "내부표현 × label-free 군집화 × 사람 phase GT 정량 대조" 세 조건을 모두 만족하는 논문은
이번 검증(12편 이상 원문 확인) 범위에서 0편이다. 기존 노트 B §5(항목 5)의 결론과 일치하며, 이번에는
분절(segmentation) 계열까지 넓혀 재확인했다.

선행 연구는 다음 네 갈래로 갈린다:
1. phase 라벨을 **정의해 주입** — PAMAE(gripper·속도 규칙), Move-Then-Operate(MLLM), SARM2(사람 주석)
2. **정책 출력**의 운동학에서 경계 추정 — PACE(속도 골짜기), AWE·keyframe heuristic
3. **정책 외부 관측**을 군집화 — LOTUS·BUDS·UVD(vision), TSC·BP-AR-HMM(kinematics), CTE·TW-FINCH(video)
4. **물리 event로 군집화한 뒤 내부 feature를 설명** — Event-Grounded SAE; 또는 내부 feature를 육안
   해석 — SAE-VLA(2603.19183); 또는 별도 학습 잠재를 정성 비교 — LAR-MoE

---

## 6. 확인 실패 / 오인용 위험 (수록 금지 목록)

- **VLA-Trace (arXiv:2605.30117)** — 원문 확인 결과 hidden-state phase probing이 **없다**.
  CKA 표현기하 + attention knockout뿐이고 "phase"는 rollout 전·후반 절반의 attention IoU 프록시(§3.3.1).
  검색 스니펫의 "hidden states encode action states/transitions"는 본문 근거 없음 → **인용 금지**.
- **Move-Then-Operate의 "ICML 2026"** — abs comments는 "15 pages, 10 figures"뿐. venue 근거 없음.
  기존 노트 B 표 #3의 ICML 2026 표기는 **정정 필요**. arXiv preprint로만 인용할 것.
- **arXiv:2603.19183의 "Dr.VLA"** — 제목·초록에 없는 내부 별칭. 정식 제목으로만 인용.
- **arXiv:2105.14829의 저널 venue(RA-L 여부)** — 확인 실패. preprint로 쓰거나 CVPR 2022판 인용.
- **TSC / TSC-DL / Niekum 3편의 arXiv 번호** — 존재 확인 실패. venue만 표기하고 arXiv id를 붙이지 말 것.
- **"SPOT"(조작 skill segmentation)** — 동명 약어 다수로 단일 논문 특정 실패. 수록하지 않음.
- **"DINOv2-feature clustering"의 독립 대표 논문** — 확인 실패. 실질적 사례가 LOTUS이므로 별도 인용 불필요.
- **MimicPlay (arXiv:2302.12422, CoRL 2023)** — 존재는 확인되나 군집 기반 분절이 아님(latent plan 학습).
  우리 비교축과 무관 → 인용 비권장.
- **"DiffusionSeeder", "Learning to Discover Subgoals", "Universal Manipulation Interface"** —
  해당 계열의 분절 문헌으로 확인되지 않음 → 인용 금지.
- **TAS 계열의 로봇 조작 직접 적용 사례** — 확인 실패(§4 서두 참조). "TAS를 로봇에 적용한 선행"이라고
  쓰지 말고, "두 계보가 분리되어 있다"로 서술할 것.

---

## 7. 결론 — 2쪽 논문의 "서술 비교 표"에 넣을 4~5편 추천

1~2쪽 국내 논문에서 baseline **재실행(re-run)은 비현실적**이다(각 후보가 서로 다른 입력 modality·
데이터 전처리·평가 프로토콜을 요구하고, 우리 GT는 태스크당 3~6단계 수동 주석 91개 경계뿐).
따라서 **서술 비교 표 한 개 + 서론 한 문단**으로 처리하는 것을 권한다.

| # | 후보 | 우리와의 축 차이 (한 줄) | 비고 |
|---|---|---|---|
| 1 | **LOTUS** (ICRA 2024) | 같은 "무라벨 군집 분절"이지만 대상이 **외부 vision encoder feature**이고 데모 전처리(offline)다. | 이미 초안 [8] |
| 2 | **PAMAE** (arXiv:2606.27144) | phase를 **gripper·속도 규칙으로 정의해 주입**하고, 보고 지표(PCP 89.0%)는 라우팅 상관이지 phase 판독 정확도가 아니다. | 이미 초안 [10] |
| 3 | **PACE** (arXiv:2606.00537) | 같은 label-free·online이지만 신호원이 **정책 출력(action chunk 속도 골짜기)**이고 GT 경계 대조 없이 SR로만 검증한다. | 신규 추천 |
| 4 | **LAR-MoE** (arXiv:2603.08476) | "phase 주석 없이 잠재에 phase가 뜬다"는 주장은 같으나, 대상이 **새로 학습한 라우팅 잠재**이고 검증이 **정성 Figure 비교**뿐이다. | 신규 추천 (리뷰어 방어 필수) |
| 5 | **TSC** (ISRR 2015 / IJRR 2017) | 동일한 "상태 군집으로 전이 발견" 발상의 고전이지만 입력이 **관측 kinematics**이고 다수 시연 offline 군집이다. | 지면 여유 시 |

**대안 5번**: 지면이 더 빠듯하면 TSC 대신 **UVD**(외부 사전학습 표현의 임베딩 거리로 subgoal 경계)를
넣는 편이 최신 대비축으로 더 강하다. 반대로 방법론적 정직성을 강조하려면 **CTE 또는 TW-FINCH**를
한 줄 넣어, "TAS 계열은 시간 정보를 군집에 주입해 성능을 얻는다 → 그래서 우리는 시간 대조군(clock)
margin으로 그 몫을 빼고 판정한다"고 쓰는 것이 리뷰어 방어에 가장 효과적이다.

**권장 서론 문장(초안)**:
> 조작 궤적을 단위 행동으로 분절하려는 연구는 (i) gripper·속도 규칙이나 사람·MLLM 주석으로 단계를
> **정의해 주입**하거나[PAMAE, SARM2], (ii) 정책이 낸 **행동 출력**의 운동학에서 경계를
> 추정하거나[PACE], (iii) 정책 **바깥의 관측**(영상·기구학)을 군집화한다[LOTUS, TSC].
> 정책 자신의 내부 활성화를 라벨 없이 군집화해 사람의 단계 정의와 어떤 해상도로 대응하는지
> 정량한 사례는 확인되지 않는다.

**§ 인용 시 주의 재확인**: §6의 수록 금지 목록(특히 VLA-Trace, Move-Then-Operate의 venue,
TSC 계열 arXiv id)을 `references.md`에 반영할 때 그대로 지킬 것. 기존 노트 B 표 #3의
"ICML 2026"과 표 #13(LAR-MoE) "미확보" 항목은 본 문서 기준으로 갱신되었다.
