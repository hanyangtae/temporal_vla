# GR00T N1.6 RoboCasa — Inference Datapoint Semantics

이 문서는 우리가 latent 분석에 사용하는 SAFE rollout pkl의 한 datapoint (이전 표현 "frame") 가 실제로 GR00T N1.6과 RoboCasa 환경에서 어떤 단위에 대응하는지 정리한다. "latent space 점 1개가 무엇인가?"를 명확히 하지 않고 분석/시각화 결과를 해석하면 시간 단위가 혼동되기 쉬워 작성.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - [02 Evaluation](n16_02_eval.md)
> - [03 SAFE Overview](n16_03_safe_overview.md)
> - [04 SAFE Collection](n16_04_safe_collection.md)
> - [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)
> - **06 Inference Datapoint Semantics (이 문서)**
> - [07 SAFE Detector](n16_07_safe_detector.md)
> - [08 SAFE Visualization](n16_08_safe_visualization.md)
> - [09 SAFE Parity](n16_09_safe_parity.md)
> - [10 SAFE Report](n16_10_safe_report.md)

> **표기 원칙**: 본 문서와 코드 옵션에서 "frame" 이라는 용어는 사용하지 않는다.
> 한 datapoint는 1회의 **GR00T inference** 출력이고, rollout 길이는 inference
> 횟수(`n_inferences` 또는 `T_inf`)로 표기한다.

기본 수치는 `ah16` collection 기준이다. `ah8` ablation에서는 같은 inference 1회를
SAFE point 1개로 유지하되, SAFE feature 원천은 `[K=4, H=8, D=1024]`이고
`n_action_steps=8`로 RoboCasa에 8 raw sim step만 실행한다.

## TL;DR — 한 datapoint의 정의

> **한 datapoint = 현재 이미지 3장 + 로봇 state + language instruction +
> 미래 action-token 정보를 GR00T 가 1회 처리한 결과**.
> rollout 당 **11 ~ 45개** 생성되고, sim_timestep 환산 **176 ~ 720 step**,
> 시뮬레이션 내부 시간 **8.8 초 ~ 36 초**.

| 항목 | 값 |
|---|---|
| 1 datapoint (= `hidden_states[t]`) | **GR00T inference 1회** 의 DiT action-token latent. 기본 `ah16`: `[K=4, H=16, D=1024]`; `ah8`: `[K=4, H=8, D=1024]` |
| 1 rollout 내 datapoint 수 (`n_inferences`) | `ah16`: **11 ~ 45** (기존 success 평균 17.6, failure 항상 45); `ah8`: 최대 **90** |
| 1 datapoint 사이 시간 | `ah16`: **16 raw sim step = 0.8 sec**; `ah8`: **8 raw sim step = 0.4 sec** simulated time |
| 1 rollout 시간 | 최대 **36 sec** simulated time (`max_episode_steps=720`); success 시간은 collection별로 측정 |
| wall-clock 시간 | 보장되지 않음 (GPU/시뮬레이션 부하에 따라 다름) |

## GR00T N1.6 추론 사이클 1회

`scripts/safe/groot_n16/robocasa/collect/collect_rollout.py:333-340, 372-374`,
`scripts/safe/groot_n16/robocasa/serve/feature_server.py:96-117` 기준.

```
loop until terminated/truncated:
    obs    ← env (현재 시점 단일 observation)
    actions, hidden_state ← policy.get_action(obs)          # 1 inference
    env.step(actions)                                        # ah16: 16 raw sim step; ah8: 8 raw sim step
```

### 입력 (1 inference 당)

- 카메라 3장 (`agentview_left`, `agentview_right`, `eye_in_hand`)
  — **현재 시점 1장씩** (`video_delta_indices = [0]`, 과거 history 없음)
- robot state (`base_position`, `base_rotation`, `eef_position_relative`,
  `eef_rotation_relative`, `gripper_qpos`) — **현재 시점 단일 값**
  (`state_delta_indices = [0]`)
- language instruction (task_description)

→ Model 입력에는 **temporal history가 들어가지 않음.** "현재 1 obs → 미래 plan" 구조.

### 출력 (1 inference 당)

- `actions`: **미래 16 step짜리 decoded action chunk** (`valid_action_horizon = 16`,
  RoboCasa PandaOmron policy output은 GR00T의 native `model_action_horizon = 50` 중 앞 16개만 decode)
- pkl의 `actions`에는 decoded action chunk 전체 16 step을 저장한다.
- `ah8` ablation은 저장된 decoded action chunk 중 앞 8개만 실행하고, SAFE feature도 앞 8 action-token만 저장한다.
- 각 step은 7-D (`action.end_effector_position[3] + .end_effector_rotation[3] + .gripper_close[1]`)
  로 환경에 전달됨
- 내부 표현: **DiT action-token latent `[K=4, H=16, D=1024]`** — 이게 우리가 pkl에 저장하는 feature

### 실행

- `MultiStepWrapper`가 decoded action chunk의 leading action들을 한꺼번에 받아 실행
  (`ah16`: `n_action_steps=16`, `ah8`: `n_action_steps=8`)
- 지정된 action step 실행이 끝나야 다음 inference

## 시간 단위

RoboCasa Kitchen env는 `control_freq = 20` Hz (default,
`src/benchmarks/robocasa/robocasa/environments/kitchen/kitchen.py:381`).

| 단위 | 환산 |
|---|---|
| 1 raw env step | **50 ms** simulated time |
| 1 inference cycle | `ah16`: **16 step = 0.8 sec**; `ah8`: **8 step = 0.4 sec** simulated time |
| Inference frequency | `ah16`: **1.25 Hz**; `ah8`: **2.5 Hz** |
| `max_episode_steps` (rollout 상한) | **720 raw step = 36 sec** |
| Max inferences per rollout | `ah16`: 720 / 16 = **45**; `ah8`: 720 / 8 = **90** |

따라서:

- 기존 `ah16` failure rollout이 모두 `n_inferences = 45` 인 이유 = **timeout (36 sec)** 까지 도달
- 기존 `ah16` success rollout 평균 `n_inferences ≈ 17.6` = 평균 **14 sec simulated time** 만에 성공
- 기존 `ah16` success min `n_inferences = 11` = **8.8 sec** (가장 빠른 성공)

**중요한 구분**: 위 시간은 모두 **simulated time** (kitchen 안 시계).
Wall-clock은 GPU 추론 + MuJoCo step + rendering 부하에 따라 보통 분 단위로 더 김.

## Captured feature `[K, H, D]`의 의미

`feature_kind = "groot_n16_dit_valid_action_tokens_pre_velocity"`,
`feature_axes = ['denoising_step', 'valid_action_step', 'feature_dim']`.

| 축 | 의미 | 크기 |
|---|---|---|
| K | flow-matching denoising step | 4 |
| H | exported action-token horizon (미래 step) | 기본 `ah16`: 16; `ah8`: 8 |
| D | DiT output feature dimension | 1024 |

- **K**: GR00T-N1.6은 flow-matching이라 한 번에 plan을 안 풀고 4번 반복 적분.
  K=0은 거의 noise, K=3은 final action에 가까움.
- **H**: export된 미래 action step마다 해당 action을 표현하는 token.
  - `token 0` = 지금 실행할 action — state-dominant
  - 기본 `ah16`의 `token 15` = 0.75 sec 뒤 action — goal-dominant
  - `ah8`의 마지막 token은 `token 7` = 0.35 sec 뒤 action
  - 정의상 token `i` 는 `i * 50ms` 뒤 action 의 representation
- **D**: 각 token의 1024-D feature. action decoder가 받기 직전 (pre-velocity)
  의 hidden — SAFE 논문에서 detector 학습용으로 권장하는 위치

## 분석 시 aggregation 선택

원본 `[T_inf, K, H, D]` → 분석용 `[T_inf, D]` 또는 `[D]` 로 압축할 때
어디서 무엇을 잃는지.

### K mean (`diff_idx_rel = mean`)

- 잃음: "denoising 단계별로 plan이 어떻게 refine 되었나" (모델의 확신도/수정 패턴)
- 보존: 4 단계의 평균적 representation
- 대안: K=3 (final) 만 사용 — 더 "확정된" 표현

### H mean (`horizon_idx_rel = mean`)

- 잃음: **단기 motion plan의 phase 정보**.
  같은 평균을 갖는 두 plan이 실제론 다른 dynamics (예: "직선으로 가기" vs "좌우 흔들기")
  를 갖는 경우 구분 불가
- 보존: 16 token의 평균적 의도 ("이 inference에서 평균적으로 어디로 가려나")
- 대안 1: `H[0]` (현재 step) — state-dominant
- 대안 2: `H[15]` (목표 직전) — goal-dominant
- 대안 3: H 전체 flatten → 16384-D
- 우리가 쓴 K·H mean = SAFE 논문의 pi0 계열 분석과 동일한 convention

### Inference-축 mean (`z_mean`, t = 1..n_inferences 평균)

- 잃음: rollout 진행에 따른 plan 변화 (시간 dynamics)
- 보존: rollout 전체에 걸친 "평균 의도" 의 위치
- 이게 우리 7-label rollout-level 시각화의 입력

### 의미적 해석

| latent 점 | 의미 |
|---|---|
| 단일 inference `z[t]` | "1.25 Hz로 찍은 model의 현재 plan summary" |
| `z_mean` (rollout-level, inference축 평균) | "이 rollout 동안 model이 평균적으로 어디를 향해 plan 했는가" |
| `h_T_inf` (LSTM 마지막 hidden) | "model의 11~45개 plan summary 시퀀스 전체를 LSTM이 비선형 압축한 256-D 요약" |

## LSTM detector 와 cadence

SAFE-LSTM detector는 GR00T와 **같은 cadence (1.25 Hz)** 로 작동:

```
매 GR00T inference 마다:
    z_t  ← K·H mean-pool of action-token latent (1024-D)
    h_t  ← LSTM_cell(z_t, h_{t-1})       # online accumulation
    p_t  ← sigmoid(fc(h_t))               # failure probability now
    if p_t > τ: raise alarm
```

- input은 GR00T feature와 동일 형식 (1024-D × inference 시퀀스)
- 시퀀스 길이는 rollout마다 11~45 (sparse)
- LSTM hidden state가 시간적 history를 누적 — model 입력에 history가 없는 한계를 보완
- rollout 시작 시 hidden state zero로 reset

## 발표/문서/코드 표기 원칙

- "frame" 이라는 단어는 **사용하지 않음**. 한 datapoint는 **inference 1회의 결과**
- rollout 길이는 **`n_inferences`** 또는 **`T_inf`** 로 표기 (옛 `T`, `T_env` 대체)
- 시간 표기 시 항상 **"simulated time"** 임을 명시
- 단위 명시: inference 개수 vs sim step 개수
- 예: "rollout 평균 길이 = 17.6 inference (≈14 sec simulated) / 281 sim step"

코드 옵션 명칭 (analyses/cluster_static.py 등):

| 옵션 | 의미 |
|---|---|
| `--feature z_inference` | rollout 안 inference 1회당 1점 (옛 `z_frame`) |
| `--feature z_mean` | rollout 내 inference축 평균 (default: 전체 inference) |
| `--feature z_last` | rollout 마지막 inference |
| `--feature h_T_inf` | LSTM hidden after all inferences |
| `--truncate-t N` | 처음 N inference만 사용 (rollout-level feature에 직교) |

## 참고 source

| 항목 | 경로 |
|---|---|
| MultiStepConfig / WrapperConfigs defaults | `src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py:49-67` |
| Kitchen env default control_freq=20 | `src/benchmarks/robocasa/robocasa/environments/kitchen/kitchen.py:381` |
| Feature capture hook (DiT forward) | `scripts/safe/groot_n16/robocasa/serve/feature_server.py:96-117` |
| Rollout 수집 루프 | `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py:344-408` |
| SAFE detector 학습 config | `outputs/eval/robocasa/groot_n16/safe_train_logs/.../config.yaml` |
| SAFE detector 모델 정의 | `failure_prob/model/lstm.py` in `/home/dongkyu/pdk_ws/SAFE` |
