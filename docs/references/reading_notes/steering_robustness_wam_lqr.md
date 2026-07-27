# Steering Robustness into World Action Models via MI and Optimal Control (WA-LQR)

- arXiv:2607.14943 (2026-07-16), Hong*·Skifstad*·Dai·Chan·Chou (전원 Georgia Tech, Trustworthy Robotics Lab)
- **출처/venue**: Robot World Models Workshop @ RSS 2026 **워크샵 채택** (메인 컨퍼런스 아님).
  PI Glen Chou = 2024-11 부임 신임 조교수(UMich PhD'22 안전 제어, MIT CSAIL 포닥, RSS Pioneer'22)
  — 거장 랩 아니라 제어이론 배경의 신생 라이징 랩. 위약·유의성 부재는 워크샵 수준과 정합,
  CoRL/ICRA 확장판 가능성 높음.
- 코드: https://github.com/trustworthyrobotics/steering_robustness_WAMs (공개, 2026-07-21 클론·분석)
- 우리 exp4-2(perturbation conceptor)의 **직접 선행연구**. 분석일 2026-07-21.

## 한 줄 요약

World Action Model(비디오 예측 + action을 같은 DiT에서 denoise하는 모델)의 DiT residual stream에서
clean-vs-perturbed 대조 방향을 뽑아, (a) ActAdd(open-loop 벡터 더하기) 또는 (b) WA-LQR(층간 선형
동역학 위 closed-loop LQR 피드백)로 steer → **교란된 입력 하에서의 SR을 회복**시키는 논문.

## 셋업

- 모델 3종 (모두 LIBERO-10): Cosmos-Policy 2B(비디오+action 통합 DiT 28블록),
  DiT4DiT(video DiT와 별도의 action DiT), LingBot-VA(같은 transformer가 video 모드→action 모드 2회 루프).
- 교란 3종 (robustness = 이 교란에 대한 내성):
  1. **카메라 pose/FOV** — MuJoCo `cam_pos/cam_quat/cam_fovy` 직접 변경 (시뮬레이터 수준).
  2. **gripper 초기 위치** — reset 후 OSC 액션으로 EE를 δxyz(σ=10cm) 이동시킨 뒤 policy 시작.
  3. **Gaussian 이미지 노이즈** — 관측 이미지에 σ=90 픽셀 노이즈.
- 평가: 교란 하에서 steered vs unsteered SR 비교 (같은 seed·같은 init). collect task 1개에서
  방향을 뽑아 다른 task로 전이하는 Task i→Task j 프로토콜. 30 trial/task.

## 방향 추출 (코드 확인 사실)

- "샘플" = rollout에서 action-chunk 추론마다 저장한 관측 1개를 **denoising 1-step만 open-loop
  재통과**시켜 얻은 DiT block output (mean-pool → 층당 D 벡터, per-timestep 옵션 있음).
  rollout 전체 활성이 아님.
- **대조쌍 의미가 교란별로 다름 (중요)**:
  - 노이즈·카메라: **state-matched pair** (같은 MuJoCo state, 교란 유무만 다름) → 순수 교란 방향.
  - **gripper: outcome 기반** — 성공 rollout→positive, 실패 rollout→negative로 버킷팅 후
    proprio 최근접이웃으로 사후 재짝짓기 (`pair_inputs_by_similarity.py`). 즉 gripper arm은
    사실상 **succ/fail 대조**이며 교란 대조와 혼입됨.
- ActAdd 방향: 단순 diff-of-means `pos.mean − neg.mean`, 층별 1벡터, 정규화 없음.
- LQR용: (layer-partition × denoising-t)별로 signed 활성의 randomized SVD → k=64 부분공간 V +
  사영된 대조 평균 c_means. 방향 v = c_means/‖c_means‖, 목표 크기 μ = ‖c_means‖.
  **Cosmos에서는 action 토큰 슬롯을 기본 제외** — world(미래관측/value) 슬롯만 steer하고
  action은 따라오게 함. **제외 이유는 논문·코드 어디에도 명시 없음** (action 포함 원본
  스크립트 `run_partition_svd_pairs.py`도 존재하나 파이프라인은 제외 버전 사용, ablation 없음).

## WA-LQR (코드 확인 사실)

- **개입 지점 = 전 층**: 마지막 layer가 아니라 28개 DiT 블록 각각의 출력(residual stream)에
  layer별 hook. SVD/제어는 층을 3 partition(0-9/10-19/20-27)으로 묶어 관리. **layer 선택
  근거·ablation 없음** (LQR의 "층 진행=시간" 프레이밍상 전 층 개입이 자연스러운 구조).
- WA-LQR 조합 자체는 이 논문의 신규 제안 ("first open- and closed-loop activation steering
  methods for WAMs"). chunk-decay(R 지수 증가) 스케줄도 이 논문 설계 — 사실상 rollout 시간축
  조건부 개입의 원시형이나 무조건 단조 감쇠일 뿐, phase-matched 아님.

- 상태 = 현재 블록 출력의 부분공간 사영 z = Vᵀx. 동역학 = 층→층 Jacobian A를 autograd(jvp/vjp)로
  계산해 사영 (Ã = V_{l+1}ᵀ J V_l). Cosmos는 B(denoise step 간) 생략, DiT4DiT만 B 포함.
- 제어 u = residual stream에 더하는 벡터. 피드백 법칙: α = λμ − v·z (setpoint 대비 오차),
  u = V_out K (α·v). **α self-gating** — 사영이 이미 setpoint에 도달하면 u→0.
- Q_SCALE=1e4, R은 action-chunk가 진행될수록 exp(c/τ), τ=3으로 **증가** → steering이 첫 chunk에
  가장 세고 지수 감쇠. 선택된 denoising timestep에서만 발화(ActAdd는 전 step).

## 결과 (Table 1, Cosmos 기준)

- 카메라: 46.0→59.3 (WA-LQR, +13.3pp). gripper: 61.3→72.7 (+11.4pp).
- 노이즈: 26.7→ ActAdd 67.3 (+40.6pp) / WA-LQR 58.7. — 노이즈는 open-loop ActAdd가 더 좋음.
- DiT4DiT gripper 65.7→71.7. **LingBot-VA는 거의 무효과** — 사전 SVM 분리도(hinge loss)와
  steering 이득이 상관 r≈−0.7 → "선형 분리 안 되면 steer 안 된다"를 예측 도구로 제시.
- 한계(저자 명시): task·모델 간 전이 제한, per-setting 분석 필요, 아키텍처 의존 큼.

## 증거 품질 (우리 기준 감사)

- **위약(random-direction/permutation) 대조 전무** — 코드 전체 grep으로 확인. "random"은 교란
  seed 변경뿐. 우리 exp2/exp3에서 위약이 진짜 효과를 기각시킨 전례에 비추면 최대 증거 공백.
- 유의성 검정 없음 (mean±std over 30 trials만).
- gripper arm은 대조축 혼입(교란×outcome)이라 세 교란의 결과를 같은 방법의 성과로 합산하기 어려움.
- 이득은 전부 **교란된 입력에서** 측정 — nominal 입력 SR 개선 증거 아님.

## 우리 연구와의 관계

| 축 | 이 논문 | 우리 (temporal_vla) |
|---|---|---|
| 개입 지점 | DiT block residual (동일) | DiT block residual |
| 대조축 | clean vs **유도된** perturbation (gripper만 outcome 혼입) | 자연 발생 succ vs fail |
| 연산자 | diff-of-means 벡터 + LQR 피드백 | conceptor (2차모멘트 부분공간) |
| 시간 조건 | denoising-t 조건부 O, **rollout-phase 조건부 X** | rollout phase-matched (우리 고유) |
| 평가 | 교란 하 SR 회복 | nominal SR 개선 (더 어려운 타깃) |
| 위약 대조 | 없음 | permutation/gated placebo 표준 |

- 우리 자연 succ/fail 대조 null vs 저들의 clean-vs-perturbed +13~40pp → 병목이 연산자(conceptor
  수학)가 아니라 **대조축의 신호 강도**라는 해석 지지 (conceptor 포화 소견과 정합).
- exp4-2 설계에 주는 것: 교란 대조는 신호가 실재함을 3자 검증. 다리 실험(자연 실패 활성을 교란
  축에 사영) 가치 재확인.
- 우리 신규성 방어선: ① 자연 outcome 대조 ② rollout-phase 조건부 ③ 위약 대조 포함 엄밀 평가
  ④ nominal SR 타깃. 단, "gripper arm이 사실상 succ/fail 대조로 +11pp"라는 점은 outcome 대조
  니치 0건 주장에 부분 반례가 될 수 있어 인용 시 정확히 기술할 것 (단 그쪽은 교란 유도 실패이고
  state-matched가 아니어서 여전히 자연 실패 대조와는 다름).
- 가져올 만한 부품: α self-gating(setpoint 도달 시 자동 감쇠 — 우리 gated steering과 유사 발상),
  chunk-decay 스케줄, SVM hinge loss를 "steer 가능성 사전 예측자"로 쓰는 프로토콜.
