---
name: pi05 × Calvin rollout 0% 디버깅 이슈
description: lerobot serve(pi05) + calvin eval 조합에서 smoke 통과하지만 모든 sequence 가 0% 성공률로 실패. serve 레이어는 정상이며 체크포인트/프로파일 fallback config 측 디버깅이 필요한 별건 이슈.
type: project
---

# pi05 × Calvin rollout 0% 이슈 (별건 추적)

## 한 줄 요약
`lerobot_pi05__calvin_sft` 프로파일로 lerobot serve 띄우고 Calvin eval 돌리면 smoke test 와 health 는 정상이지만 **rollout 10 sequence × 0% 성공률**. serve/통일 API 계약은 깨끗하고, 원인은 체크포인트 자체 또는 프로파일의 `external_config` fallback 설정에 있음.

**Why:** 체크포인트 온보딩 작업의 다른 선들 (xvla 5/5, openvla_oft smoke) 을 막지 않기 위해 별건으로 분리. 나중에 단독으로 디버깅.

**How to apply:** lerobot serve 리팩터(5f54660) 자체는 회귀 없음. 다시 손대야 할 곳은 **체크포인트 + Calvin spec 매칭**이지 serve 통신 레이어가 아님. 디버깅 시 serve 코드부터 의심하지 말 것.

## 환경 정보

- **브랜치**: `feat/vla-checkpoint-manager`
- **프로파일**: `configs/checkpoints/lerobot_pi05__calvin_sft.yaml`
- **체크포인트 (로컬)**: `/temporal_vla/checkpoints/pi05-calvin-sft`
  - lerobot 표준 `config.json` **없음** → 프로파일 `model_specific.external_config` fallback 사용 중
  - `model.safetensors` + 하위 `norm_stats.json` (RLinf/InternRobotics 구조)
- **벤치**: Calvin (`scripts/eval/calvin.py`, `compatible_benchmarks: [calvin]`)
- **norm stats**: `InternRobotics/InternData-Calvin_ABC/norm_stats.json` (lerobot 내부 processor 가 적용)
- **n_action_steps**: 50 (chunk size, health 에서 자동 감지 정상)

## 증상

- **Smoke test**: pass. `/health` → `{model: pi05, n_action_steps: 50, action_dim: 7}` 반환 OK.
- **Rollout 결과** (`outputs/calvin_pi05_eval.log`, 2026-04-08 17:43, 10 seq):
  ```
  Average sequence length: 0.000
  1/5 ~ 5/5 success rate: 모두 0.0%
  ```
- **Rollout 결과** (최신, `outputs/eval/calvin/lerobot/260424081902/seq0000_result0.mp4`): 마찬가지로 0%
- 모든 sequence 가 첫 sub-task 부터 실패 (avg seq length 0). action 은 정상 형식으로 반환되지만 환경에서 진척 없음.

## 추정 원인 (우선순위 순)

### 1. `external_config.state_shape` 가 학습 시 dim 과 다름
- 현재 프로파일: `state_shape: [7]` ← Calvin native robot state 7D 가정
- pi05 학습 데이터에서 state 가 다른 차원으로 normalize 되어 들어갔을 가능성
- norm_stats.json 의 `observation.state` 키 dim 을 직접 확인 필요

### 2. `norm_stats` 키 매핑 어긋남
- `key_selection: []` 로 비워둠 → lerobot 내부 default 매핑에 의존
- 실제 체크포인트가 학습 시 사용한 키 (예: `observation.state`, `action`, `observation.images.top` 등) 와 매칭 안 될 수 있음
- 증상: 정규화 잘못 적용되면 action 이 다른 스케일로 나감 → 환경에서 진척 0

### 3. `camera_key: observation.images.top` 오매칭
- Calvin 의 통일 API key 는 `observation.images.static`, `observation.images.gripper`
- lerobot serve 가 어떤 key 로 매핑해 모델에 넣는지 확인 필요
- 학습 시 사용된 정확한 키와 어긋나면 시각 입력 자체가 무의미

### 4. `policy_type: pi05` 와 가중치 매칭
- `model.safetensors` 만 있는 RLinf 외부 체크포인트 → lerobot policy class 로 그대로 로드 가능한지 미검증
- pi05 vs pi0 vs lerobot 자체 변형 (configuration_pi05) 사이 weight key prefix 차이 가능

### 5. Action `gripper_encoding`
- 프로파일: `range: "[-1,1]"`, `binarize: false`, `threshold: 0.0`, `sign_flip: false`
- pi05 가 실제로 [-1, 1] 연속값을 내는지, 또는 [0, 1] / 이진값인지 학습 코드 재확인 필요
- xvla 디버깅 시 (project_xvla_calvin_eval 메모리) gripper sign_flip 이슈가 있었음 — 동일 함정 가능

## 디버깅 액션 후보

재개 시 아래 순으로:

1. **norm_stats.json 직접 열어 키/dim 확인**
   ```bash
   cat /temporal_vla/checkpoints/pi05-calvin-sft/norm_stats.json | python -m json.tool | head -50
   ```
   `observation.state`, `action`, image 키들의 mean/std shape 와 `external_config` 비교.

2. **RLinf 의 CALVIN inference reference 코드 찾기**
   - HF repo: `RLinf/...` 류 모델 카드의 example
   - GitHub `RLinf/RLinf` repo 의 `examples/calvin/` 또는 `eval_calvin.py`
   - 그 코드가 어떤 image key, state shape, gripper convention 을 쓰는지가 정답.

3. **lerobot serve 안에서 매 step 의 입력/출력을 dump**
   - `scripts/serve/lerobot.py` 에 임시 print 추가 (action raw, normalized, post-processed) → log 비교
   - smoke test 의 dummy action 과 rollout 의 action 분포가 다른지 확인

4. **xvla (5/5) 와 obs/action 비교**
   - 같은 Calvin env 에서 xvla 가 정상 작동 → ObsProcessor 출력은 정상 보장
   - lerobot serve 가 받는 obs 가 xvla 가 받는 것과 동일한지, 모델 입력으로 변환되는 단계에서 어디가 다른지

5. **policy_type 변경 실험**
   - `model_specific.policy_type` 를 `pi0` 로 바꿔서 로딩되는지 (가중치 prefix 매칭 sanity check)

## 참조 파일

- 프로파일: `configs/checkpoints/lerobot_pi05__calvin_sft.yaml`
- serve: `scripts/serve/lerobot.py` (커밋 5f54660 에서 프로파일 기반으로 전환됨)
- Calvin eval: `scripts/eval/calvin.py`
- 비교 baseline: `scripts/serve/xvla.py` + `configs/checkpoints/xvla__calvin_abc_d.yaml` (동일 Calvin 에서 5/5 성공)
- 과거 log: `outputs/calvin_pi05_eval.log`, `outputs/eval/calvin/lerobot/260424081902/`
- xvla 학습/디버깅 메모리 (gripper convention 함정): `project_xvla_calvin_eval` (사용자 auto-memory)

## 중요 — 건드리지 말 것

- **lerobot serve 의 통일 API 통신 레이어** (action_layout, sub-key emit 로직). xvla 5/5 가 동일 패턴으로 동작하므로 회귀 가능성 낮음.
- **lerobot torch ABI / flash-attn 제거 fix** (5f54660). 이미 수렴된 환경 픽스.
