# exp5-4 Phase E — 후보 선택(selection) 계열 문헌 대조

작성 2026-07-27. 조사 도구 = WebSearch/WebFetch. **근거 수준 표기**: 초록·논문 페이지까지 확인한
것은 [abs], 본문 일부 확인은 [본문], 검색 스니펫만인 것은 [스니펫]. 스니펫만인 항목의 세부
수치는 확정 사실로 인용하지 말 것.

## 우리 방법(대조 기준)

GR00T N1.5(flow-matching DiT). **rollout 시작 전**, 후보 denoise 노이즈 draw k개를 각각
**첫 inference 1회만** 돌려 t=0 DiT residual 활성을 뽑고, 성공/실패 라벨의 **offline
within-scene mean-diff 축**에 사영해 1등 draw만 실제로 실행.

차별점 후보:
- ① 외부 보상모델·가치함수 없이 **정책 내부 활성**만 점수로 사용
- ② succ/fail 라벨의 **offline within-scene 대조**로 축을 fit (scene 통제)
- ③ 후보당 **forward 1회**, rollout 0회, **실행 전 1회** 선택 (에피소드 단위 게이팅)

---

## 1. Value-guided action selection (외부 가치함수 계열)

### V-GPS — Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance
arXiv:2410.13816, CoRL 2024 (Nakamoto, Mees, Kumar, Levine). [abs]
- 점수: **외부** 언어조건부 Q함수(오프라인 RL로 별도 학습). 정책 가중치 접근 불필요.
- 개입: rollout **중 매 스텝**, 정책이 뽑은 여러 action 후보를 re-rank해 최고를 실행.
- 비용: 후보당 정책 forward 1회 + value forward 1회. rollout 불필요.
- 침해: ③(후보당 forward 1회)은 이미 표준. ①은 **정면 반례가 아님**(외부 모델 사용 = 우리와 반대).
  ②도 아님(RL 보상, scene 통제 없음). → 우리 니치의 "외부 모델 불필요" 축은 여기서 살아남음.

### FM-Steer — Enhance Generalist Policies with Value-Guided Cascaded Denoising
CVPR 2026 (Song et al., AgiBot/Shanghai AI Lab 계열). [스니펫 + 목차]
- 점수: **외부 value**로 denoising 궤적 중간의 flow point를 평가, "가장 값어치 있는 flow point"를
  골라 Lite-Flow denoiser로 남은 Euler step을 마무리.
- 개입: **denoising 내부**(생성 도중). 노이즈 draw 선택이 아니라 중간 상태 선택.
- 비용: cascaded — 여러 후보 flow point에 대한 부분 denoise.
- 침해: ③에 가장 가까운 형태(생성 전/중 개입). 다만 외부 value 의존이라 ①은 유지.
- ⚠️ CVPR openaccess PDF는 403으로 직접 확인 실패. 세부는 검색 스니펫 기반.

### VLS — Steering Pretrained Robot Policies via Vision–Language Models
arXiv:2602.03973 (2026). [스니펫]
- 점수: **외부 VLM**이 후보 행동/서브골을 평가. 내부 활성 무관.
- 침해: 없음(①의 정반대 사례로 인용 가치).

---

## 2. Best-of-N + verifier (로봇/VLA)

### RoVer — Robot Reward Model as Test-Time Verifier for VLA
arXiv:2510.10975 (2025). [abs]
- 점수: **외부 process reward model(PRM)**. 후보 action을 스칼라로 채점 + 후보를 PRM이 예측한
  방향으로 확장(refine)까지 함.
- 개입: rollout **중** 매 inference. perception feature 캐싱으로 후보당 비용 절감.
- 비용: 후보당 PRM forward 1회(지각은 공유). rollout 불필요.
- 침해: ③ 침해(실행 전 값싼 채점은 표준). ① 미침해(외부 PRM). ② 미침해(scene 통제 없음).

### MG-Select — Verifier-free Test-Time Sampling for VLA Models
arXiv:2510.05681 (2025 / ICLR 2026). [abs] ★ ①에 대한 주요 위협
- 점수: **verifier 없음**. 같은 VLA에 state·language를 랜덤 마스킹해 만든 reference 분포와의
  **KL divergence**를 confidence로 써서 N개 후보 중 선택.
- 개입: test-time, action chunk 생성 후 선택(실행 전이긴 하나 **매 chunk마다**).
- 비용: 후보당 forward + reference 분포용 추가 forward. rollout 0회.
- 침해: **①을 부분 침해** — "외부 모델 없이 내부 신호로 고른다"는 주장은 선점됨. 단 신호가
  *출력 분포의 불확실성*이지 *내부 활성*이 아니고, **succ/fail 라벨을 전혀 안 씀**(②는 무사).
  또 conditional/unconditional 분포 학습을 위한 **joint training(드롭아웃)** 이 필요 → 우리의
  "백본 무학습"과는 다름.

### ForesightFlow / Potential-Guided Flow Matching for VLA Policy Improvement
arXiv:2606.04968 (2026). [스니펫 + HTML 발췌] ★ ②에 대한 위협
- 점수: **정책 내부에 심은 success-potential 좌표**(별도 critic 네트워크 아님). flow endpoint에
  추가 좌표를 붙여 "이 구간이 성공 궤적 위에 있을 확률"을 추정 → **self-guided best-of-K**.
- 개입: 실행 전, chunk 단위 K개 후보 중 선택. one-step boundary estimator로 forward 1회.
- 라벨: **혼합 품질 데이터의 성공/실패 라벨로 fine-tuning**(stage-level success-potential target).
- 침해: ①(외부 critic 없음) + ②(succ/fail 대조) + ③(후보당 forward 1회) **셋 다 형식적으로 저촉**.
  결정적 차이 = **정책 자체를 fine-tune 함**(우리는 백본 동결·offline 선형축만 fit), scene 통제
  없음, 초기 노이즈가 아니라 완성 chunk 선택.

---

## 3. Initial-noise selection / noise optimization

### Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps
arXiv:2501.09732 (2025, Ma et al.). [abs]
- 이미지 생성. "더 좋은 **노이즈**를 찾는 탐색 문제"로 정식화, 축 2개 = verifier(무엇으로 채점)
  × search algorithm. → **초기 노이즈 선택이라는 문제 설정 자체의 원조**.
- 점수: 외부 verifier(미적 점수, CLIP 등). 비용: 후보당 **전체 denoise 1회 이상**.
- 침해: ③의 "노이즈를 고른다"는 아이디어는 선점. 단 로봇/실행 전 게이팅 아님, 내부 활성 아님.

### You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector
arXiv:2603.15757 (2026-03, v최종 2026-06). [abs] ★ ③에 대한 주요 위협
- 대상: 동결된 diffusion/flow matching **로봇 정책**. 초기 노이즈를 prior 샘플링 대신 잘 고른
  **상수 노이즈 벡터**로 교체.
- 점수: **외부 task reward + Monte-Carlo policy evaluation** = 후보 노이즈당 **실제 rollout 필요**
  (실물 태스크 ~60 search episode).
- 산출: 태스크당 **고정 1개** 노이즈(에피소드마다 재선택 아님), 다른 태스크로 전이도 관찰.
- 침해: ③의 "초기 노이즈가 성공률을 좌우한다"는 **핵심 전제를 선점**. 그러나 비용 축(rollout
  필요 vs 우리는 forward 1회)과 조건화 축(태스크 고정 상수 vs 우리는 **scene/에피소드별 재선택**),
  신호 축(외부 보상 vs 내부 활성)에서 우리와 분리됨. **가장 직접적인 baseline이자 비교 대상**.

### PAINT — Start Right, Arrive Right: Asynchronous Execution via Initial Noise Selection
arXiv:2606.19774 (2026-06). [본문]
- 대상: flow-matching VLA의 action chunk. flow ODE를 역행(inversion)해 **이미 실행된 prefix와
  이어지는 초기 노이즈**를 고름. training-free, 라벨 0.
- 개입: rollout **중 매 chunk**. 비용 3N model evaluation(rollout 0회).
- 목적: 성공률이 아니라 **chunk 간 연속성**(비동기 실행 지연 대응).
- 침해: "VLA에서 초기 노이즈를 실행 전에 고른다"는 **메커니즘**은 선점. 목적함수(연속성 vs 성공
  확률)와 신호(ODE inversion vs 활성 사영)가 달라 ①②는 무사.

### Model Already Knows the Best Noise: Bayesian Active Noise Selection via Attention in Video Diffusion
arXiv:2505.17561 (2025). [스니펫]
- 비디오 diffusion. **모델 내부 attention 신호**로 초기 노이즈 후보를 사전 채점(생성 완료 전).
- 침해: ①+③의 조합(=내부 신호로 노이즈를 값싸게 고름)이 **이미지/비디오 도메인에는 존재**함을
  보여주는 사례. 로봇·succ/fail 라벨은 없음 → 우리의 기여는 "이식 + 결과 라벨 대조"로 축소.

---

## 4. VLA 내부 표현으로 성공/실패 예측 → 재샘플링

### What Frozen VLAs Already Know About Success: A Probing Study of Value-Like Structure
arXiv:2605.28527 (2026-05-27, Peking Univ. 외). [abs] ★★ **최대 위협**
- 신호: **동결된 VLA(π0/π0.5, GR00T, OpenVLA)의 내부 feature**에 **success/failure 라벨로 학습한
  선형 probe**. 정책 재학습 없음.
- 사용: LIBERO-Goal에서 probe를 **test-time selector**로 사용 — π0.5의 action prefix 후보 중 선택
  → push-plate 성공률 26.7% → **44.3%**.
- 비용: forward만(역전파 없음).
- 침해: **①②③ 모두 형식적으로 저촉**. 남는 차이 = (a) 선택 대상이 **action prefix**이지
  **초기 노이즈 draw**가 아님, (b) **디코딩 중 반복 선택**이지 rollout 전 1회 게이팅 아님,
  (c) **within-scene 대조/scene 통제 없음**(pooled probe → scene 암기 confound 미통제),
  (d) LIBERO 시뮬, GR00T DiT residual t=0 특정 아님.

### VLAConf — Calibrated Task-Success Confidence for VLA Models
arXiv:2605.29605 (2026). [abs+PDF 일부]
- 신호: **VLA 내부 표현 공간**에서 confidence 추정(visual/language 토큰 풀링), 성공 라벨로 캘리브레이션.
- 사용: 주로 **모니터링/abstain**; 후보 선택은 부차적(재샘플 트리거 언급 수준).
- 침해: ①② 저촉(내부 활성 + 성공 라벨). ③ 미저촉(선택 루프 아님).

### ActProbe — Action-Space Probe for Early Failure Detection of Generative Robot Policies
arXiv:2606.08508 (2026-06-07). [본문]
- 신호: **행동 공간만**(chunk 간 temporal consistency error + magnitude), 내부 접근 없음, 재샘플 없음.
- 개입: rollout 중 조기 경보.
- 침해: 없음. 오히려 "내부 접근 없이도 된다"는 대조 주장 → 우리 ①의 필요성을 방어해야 할 상대.

### Tri-Info — Generalizable, Interpretable Failure Prediction for VLA via Information Theory
arXiv:2606.19998 (2026). [abs]
- 신호: 행동 다양성·시간 일관성·상태전이 결합의 정보이론 지표. 재샘플/선택 용도 아님.
- 논문이 명시: "embedding 기반 failure detection은 in-domain 정확하지만 **아키텍처 특이적**" →
  우리 접근에 대한 **표준 비판 문구**로 인용될 것. 반박 준비 필요.

---

## niche 판정표

| 차별점 | 선점 여부 | 가장 가까운 선행 | 남은 여지 |
|---|---|---|---|
| ① 외부 모델 없이 **내부 활성**만으로 채점 | **선점됨** | 2605.28527(선형 probe selector), 2510.05681(내부 KL), 2505.17561(attention) | "활성 기반"은 새 것 아님. 새로울 수 있는 건 *어느* 활성(DiT residual t=0)뿐 |
| ② succ/fail **offline within-scene 대조**로 축 fit | **부분 선점** | 2605.28527(succ/fail 선형 probe, pooled), 2606.04968(success-potential, fine-tune) | **within-scene 통제**를 한 사례는 확인 안 됨 → 여기가 가장 방어 가능한 축 |
| ③ 후보당 forward 1회 · rollout 0회 · **실행 전 1회** 노이즈 선택 | **부분 선점** | 2603.15757(노이즈 선택, 단 rollout 필요), 2606.19774(노이즈 선택, 단 목적=연속성), 2510.05681/2510.10975(값싼 best-of-N, 단 대상=action) | "**초기 노이즈** × **성공 확률** × **rollout 없이**" 삼중 교집합은 아직 빈칸 |

### 한 줄 판정 (니치 수위 조정)

세 차별점이 개별로는 전부 선행에 존재하므로 **"외부 보상 없이 내부 활성으로 고른다"는 헤드라인
주장은 폐기**하고, 니치를 **"동결 VLA의 t=0 DiT 활성 × within-scene 통제 succ/fail 축 × rollout
없이 실행 전 초기-노이즈 1회 게이팅"이라는 삼중 교집합의 첫 사례**로 좁혀 주장할 것 —
즉 기여를 *신호의 새로움*이 아니라 **개입 지점(초기 노이즈)·비용(rollout 0)·통제(scene-matched)의
조합**에 두고, Golden Ticket(2603.15757)을 rollout-비용 baseline, 2605.28527을 활성-probe baseline으로
명시 비교하는 형태가 안전하다.

### 후속 확인 필요(미해결)

- FM-Steer 본문(CVPR openaccess 403) — 초기 노이즈까지 건드리는지 재확인 필요.
- 2605.28527이 selector를 **몇 스텝마다** 적용하는지, scene 분할(train/test scene 겹침) 여부 —
  초록만으로는 불명. exp5-4 결과 작성 전 본문 정독 권장.
