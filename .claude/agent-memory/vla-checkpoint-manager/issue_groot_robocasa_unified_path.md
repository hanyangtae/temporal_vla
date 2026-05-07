---
name: groot × RoboCasa 통일 HTTP API path 정확도 0% 이슈
description: GR00T 자체는 RoboCasa 에서 100% 작동 (native eval 검증) 이지만 통일 HTTP API path (groot serve + RoboCasaActionProcessor 또는 GrootRoboCasaEnv 도입 시도) 에서는 0% 성공. native 와 우리 path 의 obs/action 매핑 차이가 핵심. R3 commit 으로 GrootRoboCasaEnv 우리 fork 에 이식했으나 mujoco 의존성 호환 + robosuite robot 초기화 실패로 통일 path 미완성.
type: project
---

# groot × RoboCasa 통일 HTTP API path 0% — 별건 진단/해결

## 한 줄 요약
GR00T native eval (`scripts/eval/groot_robocasa.sh local`) 은 CloseDrawer **1/1 = 100%** 정상 동작 — 모델/체크포인트 자체 문제 아님. 하지만 우리 통일 HTTP API path (`scripts/eval/robocasa_eval.py` + `scripts/serve/groot.py`) 는 CloseFridge/OpenDrawer/CloseCabinet 모두 0%. **obs/action 매핑이 native 와 다름**. R3 commit (사용자 fork 에 GrootRoboCasaEnv + KeyConverter 이식) 후 `--use-groot-env` flag 시도했으나 mujoco 의존성 호환성 + robosuite robot 초기화 실패로 검증 미완.

**Why:** RoboCasa 평가 통일성 + 향후 finetuning 한 GR00T 체크포인트도 동일 path 로 평가 가능하게 하려면 통일 HTTP API path 가 native 와 동일 schema 보장 필요. 현재 ObsProcessor / ActionProcessor 재구현이 부정확.

**How to apply:** GR00T 만의 문제이므로 다른 모델 (xvla, dreamvla, lerobot) 의 통일 API path 는 영향 없음. 분기 별건 작업으로 처리.

## 현재 검증 결과

| 시나리오 | 결과 | 비고 |
|---|---|---|
| **native local eval** (groot 컨테이너 + GrootRoboCasaEnv + GR00T model 직접) | CloseDrawer **1/1 = 100%**, latency 60ms, 27s rollout | 모델 + env + 학습 schema 모두 정상 |
| 통일 API + RoboCasaObsProcessor + RoboCasaActionProcessor (기존 path) | CloseFridge/OpenDrawer/CloseCabinet 모두 **0/1 (500 step)**. action.eef_pos.x mean=0.82, max=1.0 — 거의 max delta 만 출력 (garbage) | obs 잘못 매핑 (state zero fallback, camera 키 의미 어긋남) |
| 통일 API + camera 키 fix (left/right/wrist) | 여전히 0%. action mean 0.8 (큰 변화 없음) | state zero fallback 영향 큼 |
| 통일 API + `--use-groot-env` (GrootRoboCasaEnv 사용) | robocasa 컨테이너: `composite_controller None` AttributeError. groot 컨테이너 (mujoco 3.2.6): 우리 fork 의 mujoco==3.3.1 assert fail | 의존성 환경 호환 미해결 |

## 핵심 진단

### native vs 우리 path 의 차이

| 단계 | native (GrootRoboCasaEnv) | 우리 path |
|---|---|---|
| robocasa env 생성 | `gym.make("robocasa_panda_omron/...")` → GrootRoboCasaEnv → `RoboCasaEnv.create_env_robosuite` (robot-specific composite controller) | `robosuite.make(...)` (controllers="OSC_POSE", camera 128x128 hardcode) |
| obs 변환 | `key_converter.map_obs(raw_obs)` → state.end_effector_position_relative 등 | RoboCasaObsProcessor 재구현 (state zero fallback, key 매핑 어긋남) |
| action 변환 | `key_converter.unmap_action(action_dict)` → robosuite step 입력 | RoboCasaActionProcessor 의 12D vector (재구현, 부정확) |

### action.eef_pos 통계 (CloseCabinet 500 step rollout, 8000 chunk steps)

```
eef_pos x: mean=0.82, std=0.23, min=-0.24, max=1.00   ← 거의 max 만
eef_pos y: mean=0.005, std=0.22
eef_pos z: mean=0.13, std=0.21
eef_axisangle 0: mean=-0.23, std=0.09
gripper: mean=0.002 (거의 닫힘 default)
base_motion / control_mode: 모두 0 (zero fallback)
```

OSC_POSE controller (input_min/max=[-1,1] → output ±0.05m) 가 0.82 받음 → 매 step 0.04m delta 한 방향 → 누적 발산 (사용자가 영상에서 본 "한쪽 방향으로 누적, 과한 꺾임").

### 원인

**obs 가 학습 분포 밖** → 모델이 garbage action 출력. 두 핵심 잘못된 매핑:
1. **카메라 매핑 의미 어긋남** (R3 직전 ObsProcessor 의 3-camera 모드: static=left, wrist=right, wrist2=eye_in_hand 라는 가짜 키. GR00T 의 wrist_0 (= eye_in_hand) 자리에 right 가 들어가고 진짜 wrist 는 zero).
   → 이번 세션에서 1차 fix 했으나 std 만 약간 늘고 mean 은 그대로 → 이게 1차 원인은 아님.
2. **state 절반이 zero fallback** — state.end_effector_position_relative, state.end_effector_rotation_relative 등 base frame 상대 ee 정보가 모델에 0 으로 들어감.
   → robocasa env raw obs (`robot0_eef_pos`, `robot0_eef_quat`, `robot0_base_pos`) 에서 계산해서 emit 해야 하는데 우리 ObsProcessor 가 안 함. 이게 핵심 원인 가능성 큼.

## R3 commit 의 의도와 한계

**R3 (`src/benchmarks/robocasa` submodule 의 c74c1c8 commit)** — `hanyangtae/robocasa` fork 에 다음 NVIDIA squarefk fork 의 파일 이식:
- `robocasa/models/robots/__init__.py` — PandaOmronKeyConverter, RobotKeyConverter, gather_robot_observations, make_key_converter 등 KeyConverter 정의 (761 lines)
- `robocasa/utils/gym_utils/gymnasium_basic.py` — RoboCasaEnv (general gym wrapper, 309 lines)
- `robocasa/utils/gym_utils/gymnasium_groot.py` — GrootRoboCasaEnv (GR00T schema 변환 layer, 172 lines)
- `robocasa/utils/gym_utils/__init__.py` — export

**의도**: 통일 API path 에서 `gym.make("robocasa_panda_omron/...")` 만 호출하면 GrootRoboCasaEnv 가 KeyConverter schema 로 obs/action 자동 변환. ObsProcessor / ActionProcessor 재구현 부담 제거. finetuning data process / training / eval 모두 동일 schema 사용 보장.

**미완성:** 의존성 환경 충돌
- 우리 fork (`hanyangtae/robocasa`) — `__init__.py` 에 `assert mujoco.__version__ == "3.3.1"`
- GR00T external fork (`squarefk/robocasa`) — `setup.py` 에 mujoco==3.2.6
- robocasa 컨테이너: mujoco 3.3.1 (우리 fork 와 호환), 그러나 **`gym.make` 시 robot 초기화 실패** (`composite_controller None`).
- groot 컨테이너 (mujoco 3.2.6 임시 install): 우리 fork 의 3.3.1 assert fail.

## 다음 시도 방향 (우선순위 순)

### 1. `composite_controller None` 진단 (robocasa 컨테이너에서)
- 가장 핵심. GrootRoboCasaEnv 의 RoboCasaEnv.create_env_robosuite 가 만든 robosuite env 의 robots 첫 element 가 None. 왜?
- 가능 원인:
  - `robosuite_models` 패키지 미설치 — Panda omron robot 등록 누락
  - `mink==0.0.13` (우리) vs `mink==0.0.5` (robosuite 1.5.2 요구) 호환성
  - controller_configs 의 HYBRID_MOBILE_BASE 가 mujoco 3.3.1 에서 init fail
- 진단 step:
  ```python
  import robosuite
  env = robosuite.make("CloseDrawer", robots="PandaOmron", controller_configs=...)
  print(env.robots, env.robots[0].composite_controller if env.robots else "no robots")
  ```

### 2. 의존성 정렬
- `src/benchmarks/robocasa` 의 mujoco assert 를 `>= 3.2.6` 로 완화 (또는 제거) 후 추가 commit
- 또는 robocasa 컨테이너에 `robosuite_models`, `mink==0.0.5` 추가 install 검토

### 3. state.end_effector_position_relative 등 정확한 계산
- 만약 `composite_controller None` 해결 후에도 state zero fallback 이면 → 옵션 B (상대 ee 직접 계산해서 ObsProcessor 가 emit) 진행
- 또는 GrootRoboCasaEnv 의 KeyConverter.map_obs 를 RoboCasaObsProcessor 가 직접 호출 (재구현 회피)

### 4. action chunk 길이 조정
- robocasa_eval.py 가 매 step `vla_client.predict` 호출 + chunk 첫 step 만 적용 (16 step chunk → 1 step 사용 → 16배 비효율).
- native eval 은 chunk 8 step 적용 후 다시 predict.
- chunk-N-step apply 패턴 추가 (n_action_steps 가 8 또는 16 일 때).

## 미커밋 / 커밋 상태

이 세션의 GR00T-RoboCasa 관련 작업:

**Submodule commit (`src/benchmarks/robocasa`):**
- `c74c1c8` feat: GR00T 호환 RoboCasa env wrapper 추가 (R3)

**Main repo 미커밋 (이 세션, GR00T-RoboCasa 관련):**
- `M docker-compose.yml` (groot service: HF ext4, scripts/utils, GPU deploy, HF_TOKEN)
- `M docker/groot/Dockerfile` (fastapi/uvicorn/opencv-python-headless/pyyaml 추가)
- `M scripts/serve/groot.py` (프로파일 기반 풀 리팩터, GR00T native dict → 통일 sub-key 매핑, fallback zero image/state)
- `M scripts/eval/robocasa_eval.py` (`--use-groot-env` flag + `run_vla_rollouts_groot` + GR00T schema rename 매핑)
- `M src/processor/action/robocasa.py` (eef_axisangle / base_motion / torso 옵션 키 처리, missing 경고)
- `M src/processor/obs/robocasa.py` (3-camera 모드 키 정정: static/left/right/wrist)
- `M src/benchmarks/robocasa` (submodule pointer → c74c1c8)
- `?? configs/checkpoints/groot__robocasa_panda_omron.yaml`

## 참조 파일

- 진단 영상: `outputs/eval/robocasa/groot/260506144254/CloseFridge.mp4`, `260506145144_OpenDrawer/OpenDrawer.mp4`, `260506150049_CloseCabinet/CloseCabinet.mp4`, `260506150849_CloseCabinet_dump/CloseCabinet.mp4`, `260506153014_CloseCabinet_camfix/CloseCabinet.mp4`
- action dump: `outputs/eval/robocasa/groot/_dump_CloseCabinet.jsonl`, `_dump_CloseCabinet_camfix.jsonl`
- native eval log (성공): `/tmp/groot_native.log` 의 `Episodes: 100%|██████████| 1/1 [00:27<00:00]  results: ('robocasa_panda_omron/CloseDrawer_PandaOmron_Env', [True], {})`
- GR00T external fork: `src/policies/Isaac-GR00T/external_dependencies/robocasa` (squarefk/robocasa, mujoco==3.2.6)
- 사용자 fork: `src/benchmarks/robocasa` (hanyangtae/robocasa, mujoco==3.3.1)

## 중요 — R3 후 결정 사항

- **GR00T external 의 robocasa 는 더 이상 wrapper 코드 출처로 필요 없음** (사용자 fork 에 동일 코드 commit 됨).
- 하지만 의존성 호환성 (mujoco 3.2.6 vs 3.3.1) 으로 인해 native eval 이 GR00T external 의존하는 형태로 남아있음.
- 정통 path: 사용자 fork 만 사용 + mujoco 3.3.1 환경 + robosuite 1.5.2 → composite_controller 문제 진단/해결.
