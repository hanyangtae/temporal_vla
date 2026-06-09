# GR00T N1.6 RoboCasa — SAFE Collection

ZMQ SAFE feature server 또는 HTTP `/act_with_features`를 통해 RoboCasa rollout pkl/mp4/csv triplet을 수집한다. ah8/ah16 action-horizon mode와 task-set 정의도 함께 둔다.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - [02 Evaluation](n16_02_eval.md)
> - [03 SAFE Overview](n16_03_safe_overview.md)
> - **04 SAFE Collection (이 문서)**
> - [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)
> - [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md)
> - [07 SAFE Detector](n16_07_safe_detector.md)
> - [08 SAFE Visualization](n16_08_safe_visualization.md)
> - [09 SAFE Parity](n16_09_safe_parity.md)
> - [10 SAFE Report](n16_10_safe_report.md)

## ZMQ SAFE Feature Collection

목적: official RoboCasa eval 환경을 유지하면서 SAFE 학습용 pkl schema를 만든다.

`feature_server.py`는 DiT pre-velocity capture 로직을 `src/policies/groot/safe_features.py` 의 `capture_dit_features` 에 위임한다. HTTP `/act_with_features` ([n16_11](n16_11_http_act_changes.md)) 도 같은 함수를 호출하므로, 두 경로가 같은 hidden state 정의를 보장한다.

`collect_rollout.py --inference-seed`는 ZMQ와 HTTP transport 양쪽에 적용된다. ZMQ에서는 request `options.inference_seed`, HTTP에서는 request payload `inference_seed`로 전달된다. Transport parity run은 base seed에 policy-step index를 더한 schedule을 사용한다.

서버:

```bash
cd /temporal_vla/src/policies/Isaac-GR00T

UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/temporal_vla/data/huggingface \
HF_MODULES_CACHE=/tmp/hf_modules \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python /temporal_vla/scripts/safe/groot_n16/robocasa/serve/feature_server.py \
  --profile /temporal_vla/configs/checkpoints/groot__robocasa365_ckpt120000.yaml \
  --host '*' \
  --port 5557 \
  --device cuda \
  --feature-dtype float16 \
  --feature-slice valid
```

수집:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

RUN_ID=n16_task_set_official_uv_smoke \
EPISODES_PER_TASK=1 \
SEED_START=241 \
EPISODE_START_IDX=0 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_official_uv_host.sh
```

RoboCasa365 target atomic-seen 18-task 수집:

1ep/task smoke:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

ROBOCASA_SAFE_RUN_ID=target_atomic_seen18_ckpt120000_robocasa365_1ep \
TASK_SET=target_atomic_seen18 \
EPISODES_PER_TASK=1 \
SEED_START=100020 \
EPISODE_START_IDX=0 \
HOST=127.0.0.1 \
PORT=5557 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
```

100ep/task collection:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

ROBOCASA_SAFE_RUN_ID=target_atomic_seen18_ckpt120000_robocasa365_100ep \
TASK_SET=target_atomic_seen18 \
EPISODES_PER_TASK=100 \
SEED_START=100000 \
EPISODE_START_IDX=0 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
```

Action-horizon paired ablation:

```text
canonical scenario manifests:
outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099

ah8 run:
outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_ah8_100ep/raw_rollouts

ah16 paired run:
outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_ah16_100ep/raw_rollouts
```

`ah8`는 GR00T model-level action tokens 50개 중 leading 8개를 SAFE feature로 저장하고, decoded action chunk 16개 중 leading 8개만 RoboCasa에 실행한다. pkl의 `actions`에는 decoded action chunk 전체 16개를 유지한다. SAFE point는 inference 1회당 1점으로 유지한다. `ah16`은 같은 manifest root를 import해서 같은 scene composition에서 기존 16-step cadence를 재수집하는 paired baseline이다. 이전 `target_atomic_seen18_ckpt120000_robocasa365_100ep` collection은 `scenario_seed`와 `ep_meta` manifest가 없으므로 historical reference로만 사용한다.

지원 mode:

| mode | server export | collector execution | pkl `hidden_states` | pkl `actions` |
|---|---|---|---|---|
| `ah8` | `--feature-action-horizon 8` | `N_ACTION_STEPS=8` | `[4,8,1024]` | decoded 16-step chunk 전체 |
| `ah16` | omit `--feature-action-horizon` or set `16` | `N_ACTION_STEPS=16` | `[4,16,1024]` | decoded 16-step chunk 전체 |

`exported_action_token_count`와 `n_action_steps`가 다르면 collector가 pkl 쓰기 전에 실패한다. 따라서 ah8/ah16 모두 server export horizon과 collector execution horizon을 같은 값으로 맞춘다.

## HTTP `/act_with_features` SAFE Feature Collection

HTTP transport는 같은 collector에서 `--policy-transport http`로 선택한다. 이 경로는 `scripts/safe/groot_n16/robocasa/collect/collect_policy_clients.py`가 `make_groot_robocasa_processors(action_mode="chunk")`를 통해 FastAPI `/act_with_features` 응답을 unified action sub-key에서 GR00T native action key로 되돌리고, `scripts/safe/groot_n16/robocasa/collect/collect_schema.py` helper를 통해 `features.hidden_states` blob을 기존 SAFE pkl의 `hidden_states` list로 저장한다. GR00T key mapping은 `src/policies/groot/schema.py`와 `src/policies/groot/robocasa_io.py`를 source of truth로 둔다.

Smoke command:

```bash
# 1) groot container/process에서 HTTP server를 먼저 띄운다.
cd /temporal_vla
python scripts/serve/groot.py \
  --profile configs/checkpoints/groot__robocasa365_ckpt120000.yaml \
  --host 127.0.0.1 \
  --port 8500 \
  --feature-dtype float32 \
  --feature-action-horizon 16

# 2) robocasa env에서 HTTP feature collection smoke.
cd /temporal_vla
python scripts/safe/groot_n16/robocasa/collect/collect_rollout.py \
  --policy-transport http \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 8500 \
  --env-name robocasa_panda_omron/CloseFridge_PandaOmron_Env \
  --robocasa-env-source robocasa365 \
  --output-dir outputs/tmp/groot_http_act_features_safe_collect_20260529/http_rollout \
  --task-id 0 \
  --n-episodes 1 \
  --n_action_steps 16 \
  --max-episode-steps 1 \
  --seed 100000 \
  --inference-seed 424242 \
  --ep-meta-dir outputs/tmp/groot_http_act_features_safe_collect_20260529/ep_meta
```

Smoke artifact와 SAFE loader 결과는 [10 SAFE Report](n16_10_safe_report.md#http-act_with_features-safe-collection-smoke-2026-05-29)에 기록한다. HTTP/ZMQ transport parity 결과는 [09 SAFE Parity](n16_09_safe_parity.md#closed-loop-safe-transport-smoke)에 둔다.

`ah8` 서버:

```bash
cd /temporal_vla/src/policies/Isaac-GR00T

UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/temporal_vla/data/huggingface \
HF_MODULES_CACHE=/tmp/hf_modules \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python /temporal_vla/scripts/safe/groot_n16/robocasa/serve/feature_server.py \
  --profile /temporal_vla/configs/checkpoints/groot__robocasa365_ckpt120000.yaml \
  --host '*' \
  --port 5557 \
  --device cuda \
  --feature-dtype float16 \
  --feature-slice valid \
  --feature-action-horizon 8
```

`ah8` smoke는 `CloseFridge` 3 episodes로 실행한다. 같은 canonical manifest root에 seed `100000..100002`가 먼저 export된다.

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

ROBOCASA_SAFE_RUN_ID=target_atomic_seen18_ckpt120000_robocasa365_ah8_smoke \
TASK_SET=target_atomic_seen18 \
TASKS_OVERRIDE=CloseFridge \
EPISODES_PER_TASK=3 \
SEED_START=100000 \
EPISODE_START_IDX=0 \
N_ACTION_STEPS=8 \
EP_META_ROOT=/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099 \
EP_META_ROOT_CONTAINER=/temporal_vla/outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099 \
HOST=127.0.0.1 \
PORT=5557 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
```

`ah8` smoke 검증:

```bash
python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
  outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_ah8_smoke/raw_rollouts \
  --tasks-override CloseFridge \
  --episodes-per-task 3 \
  --allow-partial \
  --expected-hidden-shape 4,8,1024 \
  --expected-feature-action-horizon 8 \
  --expected-n-action-steps 8
```

`ah8` 100ep/task collection:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

ROBOCASA_SAFE_RUN_ID=target_atomic_seen18_ckpt120000_robocasa365_ah8_100ep \
TASK_SET=target_atomic_seen18 \
EPISODES_PER_TASK=100 \
SEED_START=100000 \
EPISODE_START_IDX=0 \
N_ACTION_STEPS=8 \
EP_META_ROOT=/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099 \
EP_META_ROOT_CONTAINER=/temporal_vla/outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099 \
HOST=127.0.0.1 \
PORT=5557 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
```

### Headless Render Failure Recovery

`Default framebuffer is not complete` 또는 `Current sensor for observable ..._image is invalid`는 GR00T policy/SAFE feature server가 아니라 RoboCasa/robosuite의 MuJoCo offscreen camera render context 실패다. Collector는 done 이후 `SyncVectorEnv`의 자동 reset을 사용하지 않는다. 이 reset은 다음 observation을 만들기 위한 카메라 render를 추가로 발생시키지만, SAFE collection은 episode 종료 후 그 observation을 사용하지 않는다. `collect_task_set_via_docker_exec.sh`는 Docker collector에 `MUJOCO_GL=egl`과 `PYOPENGL_PLATFORM=egl`을 함께 전달하고, episode마다 기본 2회 시도한다.

같은 run을 이어갈 때는 원래 collection command를 그대로 다시 실행한다. 이미 pkl/mp4가 완성된 episode는 skip되고, 실패한 episode부터 다시 수집된다. 시도 횟수만 늘릴 때는 wrapper env로 조정한다.

```bash
COLLECTION_MAX_ATTEMPTS=3 \
COLLECTION_RETRY_SLEEP_SEC=10 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
```

`ah8` 최종 검증:

```bash
python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
  outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_ah8_100ep/raw_rollouts \
  --expected-hidden-shape 4,8,1024 \
  --expected-feature-action-horizon 8 \
  --expected-n-action-steps 8
```

`ah16` paired collection은 같은 canonical manifest root를 import한다. 서버는 `--feature-action-horizon`을 생략하면 decoded valid horizon 16을 export한다.

`ah16` 서버:

```bash
cd /temporal_vla/src/policies/Isaac-GR00T

UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/temporal_vla/data/huggingface \
HF_MODULES_CACHE=/tmp/hf_modules \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python /temporal_vla/scripts/safe/groot_n16/robocasa/serve/feature_server.py \
  --profile /temporal_vla/configs/checkpoints/groot__robocasa365_ckpt120000.yaml \
  --host '*' \
  --port 5557 \
  --device cuda \
  --feature-dtype float16 \
  --feature-slice valid
```

`ah16` 100ep/task collection:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

ROBOCASA_SAFE_RUN_ID=target_atomic_seen18_ckpt120000_robocasa365_ah16_100ep \
TASK_SET=target_atomic_seen18 \
EPISODES_PER_TASK=100 \
SEED_START=100000 \
EPISODE_START_IDX=0 \
N_ACTION_STEPS=16 \
EP_META_ROOT=/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099 \
EP_META_ROOT_CONTAINER=/temporal_vla/outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099 \
HOST=127.0.0.1 \
PORT=5557 \
bash scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh
```

`ah16` 최종 검증:

```bash
python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
  outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_ah16_100ep/raw_rollouts \
  --expected-hidden-shape 4,16,1024 \
  --expected-feature-action-horizon 16 \
  --expected-n-action-steps 16
```

Smoke validation (2026-05-28):

```text
ah8 smoke:
outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_ah8_smoke/raw_rollouts/CloseFridge

ah16 smoke:
outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_ah16_smoke/raw_rollouts/CloseFridge

shared scenario manifests:
outputs/eval/robocasa/groot_n16/scenario_manifests/target_atomic_seen18_seed100000_100099/CloseFridge
```

검증 결과:

| mode | task | seeds | result | pkl `hidden_states` | pkl `actions` | ep_meta mode |
|---|---|---|---|---|---|---|
| `ah8` | `CloseFridge` | `100000..100002` | 3/3 success, verifier `status=ok` | `[4,8,1024]` | decoded 16-step chunk 전체 | exported |
| `ah16` | `CloseFridge` | `100000..100002` | 3/3 success, verifier `status=ok` | `[4,16,1024]` | decoded 16-step chunk 전체 | imported |

ah8/ah16 GR00T 서버를 동시에 올리면 16GB GPU에서 CUDA OOM이 날 수 있다. Smoke와 full collection은 하나의 GR00T feature server만 띄운 상태에서 ah8, ah16을 순차 실행한다.

이 mode는 RoboCasa365 (`src/benchmarks/robocasa`)를 사용한다. RoboCasa v0.2 (`robocasa_v02`) 기준 6-task SAFE split과 달리 `CloseBlenderLid`, `NavigateKitchen`, `OpenStandMixerHead` 같은 RoboCasa365 task가 포함되므로 `ROBOCASA_ENV_SOURCE=robocasa365`가 자동으로 선택된다. 기본 저장 위치는 아래다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/raw_rollouts
```

`target_atomic_seen18_ckpt120000_100ep`가 이미 존재하면 pre-fix PandaOmron-profile 산출물과 섞일 수 있으므로 새 collection에는 `target_atomic_seen18_ckpt120000_robocasa365_100ep`를 사용한다.

진행 중 partial 검증:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
  outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/raw_rollouts \
  --allow-partial
```

완료 후 최종 검증:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
  outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/raw_rollouts
```

완료 상태:

- completed at: `2026-05-27 04:02:09 KST`
- verified at: `2026-05-27 09:25 KST`
- branch: `pdk/0526/groot_rollout`
- seed start: `100000`
- seed range per task: `100000..100099`
- seed formula: `seed = 100000 + episode_idx`
- artifacts: `1800` pkl, `1800` mp4, `1800` csv
- output size: `7.9G`
- verifier: `status=ok`
- total SR: `967/1800 = 53.7%`

Verifier summary:

```text
root=outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/raw_rollouts
tasks=18 episodes_per_task=100
completed=1800 expected=1800
summary_seeds=100000..100099 unique=100 rows=1800 formula=seed_start+episode_idx
status=ok
```

Per-task SR은 [SAFE reproduction report](n16_10_safe_report.md#robocasa365-18-task-collection-결과)에 둔다. 이 collection은 raw rollout/latent-feature producer artifact이며, 기존 `safe_seen4_unseen2_100ep` 6-task detector split과는 별도 run이다.

저장 위치:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/experiments/collection_smoke/rollouts_${RUN_ID}
```

SAFE feature contract:

- transport: ZMQ `PolicyServer` endpoint, not FastAPI HTTP
- endpoint: `get_action_with_features`
- default feature kind: `groot_n16_dit_valid_action_tokens_pre_velocity`
- feature source: DiT output immediately before velocity/action projection
- serialized rollout feature shape per step: `[K, H_valid, D]`
- expected default shape: `[4, 16, 1024]`
- dtype: `float16`

GR00T N1.6 checkpoint config의 model-level `action_horizon` / processor `max_action_horizon` 값은 50이다. checkpoint-120000 profile에서 선택한 embodiment modality config의 decoded action delta는 16개다. GR00T 내부 action head는 50개 action-token trajectory를 만들고, RoboCasa output은 앞 16 step을 decode한다.

따라서 SAFE 기본 export는 다음처럼 잡는다.

```python
all_action_tokens = model_output[:, -50:, :]
safe_tokens = all_action_tokens[:, :16, :]
```

원하면 feature server를 `--feature-slice all`로 띄워 `[4, 50, 1024]` 전체 action-token feature를 저장할 수 있다. 이 설정은 RoboCasa에서 직접 decode/execution되는 앞 16 step과 뒤 34 model-level token을 함께 넣으므로 detector input을 희석할 수 있다.

SAFE 논문식 flow-matching feature에 맞춰 collector는 feature 축을 보존한다. horizon/diffusion aggregation은 detector train/eval 단계에서 선택한다.

영상 저장은 GR00T upstream `VideoRecordingWrapper`에 맡긴다. SAFE collector는 `VideoConfig(video_dir=...)`만 설정해서 wrapper를 켜고, `fps`, `steps_per_render`, `max_episode_steps` 등은 upstream 기본값을 그대로 둔다. Upstream RoboCasa observation은 3개 camera의 `res256`/`res512` key를 모두 포함하므로, recorder 앞단에서 영상용 observation만 `video.res256_image_side_0`, `video.res256_image_side_1`, `video.res256_image_wrist_0`로 제한한다. Episode 종료 후 wrapper가 만든 mp4를 SAFE 산출물 이름(`task{id}--ep{idx}--succ{0|1}.mp4`)으로 이동한다.

검증된 action 동등성:

- official direct policy action과 SAFE feature path action 비교
- `max_abs=0.0`
- 비교 대상 action keys:
  - `action.end_effector_position`
  - `action.end_effector_rotation`
  - `action.gripper_close`
  - `action.base_motion`
  - `action.control_mode`

## Common Candidate 9 Tasks

GR00T official RoboCasa v0.2 eval task와 robocasa365 v1.0 atomic dataset이 의미적으로 겹치는 후보는 아래 9개로 둔다.

| RoboCasa v0.2 task | robocasa365 v1.0 task | official SR |
|---|---|---:|
| `CoffeeSetupMug` | `CoffeeSetupMug` | 31.0% |
| `OpenSingleDoor` | `OpenCabinet` | 81.5% |
| `OpenDrawer` | `OpenDrawer` | 81.1% |
| `PnPCounterToCab` | `PickPlaceCounterToCabinet` | 47.5% |
| `PnPSinkToCounter` | `PickPlaceSinkToCounter` | 50.0% |
| `PnPCounterToStove` | `PickPlaceCounterToStove` | 63.2% |
| `TurnOffStove` | `TurnOffStove` | 31.0% |
| `TurnOnMicrowave` | `TurnOnMicrowave` | 91.5% |
| `TurnOnSinkFaucet` | `TurnOnSinkFaucet` | 89.0% |

## Selected Seen 6 Tasks

최종 seen-task set은 robocasa365 data에도 있고, GR00T official RoboCasa eval에도 대응되는 task로 둔다.

| task id | RoboCasa v0.2 task | robocasa365 v1.0 task | official SR |
|---:|---|---|---:|
| 0 | `CoffeeSetupMug` | `CoffeeSetupMug` | 31.0% |
| 1 | `OpenSingleDoor` | `OpenCabinet` | 81.5% |
| 2 | `PnPCounterToCab` | `PickPlaceCounterToCabinet` | 47.5% |
| 3 | `PnPSinkToCounter` | `PickPlaceSinkToCounter` | 50.0% |
| 4 | `PnPCounterToStove` | `PickPlaceCounterToStove` | 63.2% |
| 5 | `OpenDrawer` | `OpenDrawer` | 81.1% |

제외한 후보:

- `TurnOffStove`: official SR `31.0%`, 이번 6-task set에서는 제외.
- `TurnOnMicrowave`, `TurnOnSinkFaucet`: 성공률이 높아 failure detector 데이터 균형 관점에서 이번 6-task set 우선순위 밖에 둔다.
