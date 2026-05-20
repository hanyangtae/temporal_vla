# SAFE x GR00T N1.6 RoboCasa Wiring

이 문서는 GR00T N1.6 RoboCasa와 SAFE detector를 연결할 때 사용할 serving/eval 경로를 고정한다. 결론은 HTTP와 ZMQ를 모두 유지하되, 현재 실험 기준선은 ZMQ로 둔다.

## 현재 결론

- `ZMQ official eval`: GR00T N1.6 RoboCasa 성공률을 판단하는 기준 경로.
- `ZMQ SAFE feature server`: SAFE rollout 수집 기준 경로. official RoboCasa 클라이언트 환경을 쓰면서 action과 feature를 함께 저장한다.
- `HTTP /act`: 프로젝트 공통 serving API로 유지한다. 다만 현재는 GR00T N1.6 RoboCasa 성공률 기준선으로 쓰지 않는다.

HTTP 경로는 폐기하지 않는다. 다만 official ZMQ에서는 OpenDrawer가 정상 성공률을 보였고, HTTP 연동에서는 성공률이 낮았으므로 먼저 동일 observation/action chunk 기준 동등성 검증이 필요하다.

## Checkpoint And Env

체크포인트:

- host: `/home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B`
- container: `/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B`
- profile: `/home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml`

RoboCasa 환경:

- 기준 env: `/home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python`
- 기준 RoboCasa 코드: `/home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T/external_dependencies/robocasa`
- local robocasa365: `/home/dongkyu/pdk_ws/temporal_vla/src/benchmarks/robocasa`

GR00T official eval과 SAFE collection은 GR00T fork의 RoboCasa v0.2 task name을 쓴다. local data는 robocasa365 v1.0 이름을 쓰므로 task mapping을 명시적으로 유지한다.

## ZMQ Official Eval

목적: GR00T N1.6 checkpoint 자체가 RoboCasa PandaOmron에서 정상 동작하는지 확인한다.

서버:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T

UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/home/dongkyu/pdk_ws/temporal_vla/data/huggingface \
HF_MODULES_CACHE=/tmp/hf_modules \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path /home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper \
  --port 5555
```

클라이언트 예시:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
  --n_episodes 10 \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 5555 \
  --max_episode_steps 720 \
  --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n_action_steps 8 \
  --n_envs 1
```

확인된 결과:

- `OpenDrawer`, 10 episodes, `n_envs=1`, `n_action_steps=8`
- success list: `[True, True, False, False, True, True, True, True, True, True]`
- SR: `0.8`
- official README의 OpenDrawer `81.1%`와 같은 수준

따라서 현재 checkpoint 자체가 깨졌다는 증거는 없다.

## ZMQ SAFE Feature Collection

목적: official RoboCasa eval 환경을 유지하면서 SAFE 학습용 pkl schema를 만든다.

서버:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T

UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/home/dongkyu/pdk_ws/temporal_vla/data/huggingface \
HF_MODULES_CACHE=/tmp/hf_modules \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python /home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/serve/feature_server.py \
  --profile /home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml \
  --model-path-override /home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B \
  --host '*' \
  --port 5557 \
  --device cuda \
  --feature-dtype float16 \
  --feature-slice valid
```

수집:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

RUN_ID=n16_six_task_official_uv_smoke \
EPISODES_PER_TASK=1 \
SEED_START=241 \
EPISODE_START_IDX=0 \
bash scripts/safe/groot_n16/robocasa/collect/collect_six_task_official_uv_host.sh
```

저장 위치:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/rollouts_${RUN_ID}
```

SAFE feature contract:

- endpoint: `get_action_with_features`
- default feature kind: `groot_n16_dit_valid_action_tokens_pre_velocity`
- feature source: DiT output immediately before velocity/action projection
- serialized rollout feature shape per step: `[K, H_valid, D]`
- expected default shape: `[4, 16, 1024]`
- dtype: `float16`

`50`은 우리가 임의로 정한 SAFE horizon이 아니다. GR00T N1.6 checkpoint config의 model-level `action_horizon` / processor `max_action_horizon` 값이다. 반면 RoboCasa PandaOmron modality config의 decoded action delta는 16개다. GR00T 내부 action head는 50개 action-token trajectory를 만들지만, RoboCasa output으로 decode되는 것은 앞 16 step이다.

따라서 SAFE 기본 export는 다음처럼 잡는다.

```python
all_action_tokens = model_output[:, -50:, :]
safe_tokens = all_action_tokens[:, :16, :]
```

원하면 feature server를 `--feature-slice all`로 띄워 `[4, 50, 1024]` 전체 action-token feature를 저장할 수 있다. 하지만 이 경우 뒤 34 token은 RoboCasa에서 직접 decode/execution되는 action step이 아니므로 detector input을 희석할 수 있다.

SAFE 논문식 flow-matching feature에 맞추기 위해 collector는 feature를 pooling하지 않는다. horizon/diffusion aggregation은 detector train/eval 단계에서 선택해야 한다.

영상 저장은 GR00T upstream `VideoRecordingWrapper`에 맡긴다. SAFE collector는 `VideoConfig(video_dir=...)`만 설정해서 wrapper를 켜고, `fps`, `steps_per_render`, `max_episode_steps` 등은 upstream 기본값을 그대로 둔다. 다만 upstream RoboCasa observation은 3개 camera의 `res256`/`res512` key를 모두 포함하므로, recorder 앞단에서 영상용 observation만 `video.res256_image_side_0`, `video.res256_image_side_1`, `video.res256_image_wrist_0`로 제한한다. Episode 종료 후 wrapper가 만든 mp4를 SAFE 산출물 이름(`task{id}--ep{idx}--succ{0|1}.mp4`)으로 이동한다.

검증된 action 동등성:

- official direct policy action과 SAFE feature path action 비교
- `max_abs=0.0`
- 비교 대상 action keys:
  - `action.end_effector_position`
  - `action.end_effector_rotation`
  - `action.gripper_close`
  - `action.base_motion`
  - `action.control_mode`

## HTTP Path

목적: 프로젝트 공통 VLA serving/eval API를 유지한다.

서버:

```bash
docker compose run --rm groot \
  python /temporal_vla/scripts/serve/groot.py \
  --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml
```

API:

- `POST /act`
- `POST /reset`
- `GET /health`

현재 판단:

- HTTP는 heterogeneous VLA serving을 위한 경로로 계속 유지한다.
- GR00T N1.6 RoboCasa 성공률 기준선은 아직 ZMQ official eval이다.
- HTTP에서도 최종적으로는 대응 task의 SR이 높게 나와야 한다. 낮은 SR이 나온다면 checkpoint 성능 문제가 아니라, 우선 wiring/runner/schema mismatch를 의심한다.
- HTTP를 다시 SR eval에 쓰려면, 같은 RoboCasa observation에 대해 HTTP action과 ZMQ official action의 chunk shape, key mapping, action scale, reset timing, camera/state schema를 먼저 비교해야 한다.

현재 의심 지점:

- observation key conversion 차이
- action key mapping 차이
- chunk length 또는 action step 소비 방식 차이
- reset/stateful policy timing 차이
- project robocasa365와 GR00T fork RoboCasa env 차이

### TODO: HTTP SR Recovery

현재 HTTP와 ZMQ의 차이는 단순 transport 차이가 아니다.

- HTTP 경로는 보통 project runner를 타며 `src/benchmarks/robocasa`의 robocasa365 v1.0 환경을 사용한다.
- ZMQ official 경로는 `src/policies/Isaac-GR00T/external_dependencies/robocasa`의 GR00T fork RoboCasa v0.2 환경을 사용한다.
- 따라서 현재 비교에는 transport, env version, task class name, observation schema, action application 방식 차이가 함께 섞여 있다.

그래도 기대값은 명확하다. 공통 task로 고른 5개는 v0.2와 v1.0 사이에서 의미적으로 대응되는 task이므로, HTTP path에서도 충분히 높은 SR이 나와야 한다. 특히 `OpenSingleDoor`/`OpenCabinet`처럼 official SR이 높은 task에서 HTTP SR이 크게 낮다면 detector 문제가 아니라 policy wiring 문제로 본다.

HTTP SR 회복을 위한 검증 순서:

1. 같은 checkpoint, 같은 seed, 같은 initial env state에 최대한 가깝게 맞춘다.
2. ZMQ official client가 만든 raw observation을 저장한다.
3. 같은 observation을 HTTP `/act` 입력 payload로 변환한다.
4. HTTP output action을 GR00T native action key로 되돌려 ZMQ official action과 비교한다.
5. action key별 shape, first action, chunk horizon, scale, gripper sign, rotation convention을 비교한다.
6. action이 같거나 충분히 가까우면, 같은 env에서 HTTP action application loop만 교체해서 SR을 본다.
7. action은 같은데 SR만 낮으면 action consumption, reset, termination, wrapper, env version 차이를 본다.
8. action부터 다르면 observation conversion 또는 output mapping을 먼저 고친다.

우선순위가 높은 체크포인트:

- image keys: `side_0`, `side_1`, `wrist_0`가 GR00T의 `res256_image_*` 계열과 동일 의미인지 확인한다.
- state keys: `eef_pos`, `eef_quat`, `gripper_qpos`, `joint_pos`의 순서와 batch/time dimension이 official wrapper와 같은지 확인한다.
- action keys: `end_effector_position`, `end_effector_rotation`, `gripper_close`, `base_motion`, `control_mode`가 HTTP의 `action.eef_pos`, `action.eef_axisangle`, `action.gripper`로 손실 없이 매핑되는지 확인한다.
- action chunk: GR00T N1.6의 chunk horizon과 RoboCasa action repeat/consume 방식이 HTTP runner와 ZMQ runner에서 같은지 확인한다.
- reset: episode 시작 시 `/reset`이 반드시 호출되고, server-side policy state가 official ZMQ path와 같은 시점에 초기화되는지 확인한다.
- env version: robocasa365 v1.0에서 task semantics가 v0.2와 충분히 같은지 task별로 확인한다.

이 TODO가 끝나기 전까지 HTTP SR 결과는 “모델 성능”으로 해석하지 않는다. HTTP SR은 wiring parity가 확인된 뒤에만 SAFE/GR00T N1.6 성능 지표로 쓴다.

## Common Candidate 9 Tasks

GR00T official RoboCasa v0.2 eval task와 local robocasa365 v1.0 atomic dataset이 의미적으로 겹치는 후보는 아래 9개로 둔다.

| GR00T fork v0.2 task | local robocasa365 v1.0 task | official SR |
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

최종 seen-task set은 local robocasa365 data에도 있고, GR00T official RoboCasa eval에도 대응되는 task로 둔다.

| task id | GR00T fork v0.2 task | local robocasa365 v1.0 task | official SR |
|---:|---|---|---:|
| 0 | `CoffeeSetupMug` | `CoffeeSetupMug` | 31.0% |
| 1 | `OpenSingleDoor` | `OpenCabinet` | 81.5% |
| 2 | `PnPCounterToCab` | `PickPlaceCounterToCabinet` | 47.5% |
| 3 | `PnPSinkToCounter` | `PickPlaceSinkToCounter` | 50.0% |
| 4 | `PnPCounterToStove` | `PickPlaceCounterToStove` | 63.2% |
| 5 | `OpenDrawer` | `OpenDrawer` | 81.1% |

제외한 후보:

- `TurnOffStove`: official SR `31.0%`, 이번 6-task set에서는 제외.
- `TurnOnMicrowave`, `TurnOnSinkFaucet`: 성공률이 높아 failure detector 데이터 균형 관점에서는 우선순위를 낮춘다.

## Paper-Faithful SAFE Split

SAFE 논문/레포 방식에 맞출 때는 별도 `train / validation / CP / test` 4-way episode split을 만들지 않는다.

- raw rollout cap: `max_rollouts_per_task: 100`
- task-level split: `unseen_task_ratio: 0.25`
- seen-task episode split: `seen_train_ratio: 0.75`
- detector train: seen task의 train rollout
- threshold / conformal calibration: `val_seen`
- final evaluation: `val_unseen`

따라서 6개 task에서는 `round(0.25 * 6) = 2`개 task가 unseen이 되고, 나머지 4개 task가 seen이 된다. 각 task를 100 rollout으로 맞추면 전체 600 rollout이며, split은 대략 다음 크기가 된다. SAFE 레포 DROID 설정의 `60/task`보다 큰 cap이지만, task별 SR이 낮아 성공 rollout이 부족할 수 있으므로 N1.6 RoboCasa에서는 `100/task`를 사용한다.

이번 small reproduction에서는 완전 랜덤 task split 대신 taxonomy constraint를 둔다. unseen task는 Open 계열 1개와 PnP 계열 1개로 고정한다. 실제 unseen task는 `OpenDrawer`와 `PnPCounterToCab`이다. 이 선택은 `OpenSingleDoor`를 seen 쪽에 남겨 local robocasa365의 `OpenCabinet` 대응 경로를 계속 점검할 수 있게 하고, `val_unseen` 성공/실패 비율도 `114/86`으로 과도하게 치우치지 않는다.

| split | source | count |
|---|---|---:|
| `train` | 4 seen tasks × 75 rollout | 300 |
| `val_seen` | 4 seen tasks × 25 rollout | 100 |
| `val_unseen` | 2 unseen tasks × 100 rollout | 200 |

`val_seen`은 validation과 conformal calibration 역할을 함께 한다. 별도 CP-only split이나 seen-task test split을 만들면 논문식 재현에서 벗어난다.

생성된 split:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_split_seen4_unseen2_openDrawer_pnpCab_100ep
```

Split summary:

| split | total | success | failure | SR |
|---|---:|---:|---:|---:|
| `train` | 300 | 141 | 159 | 47.0% |
| `val_seen` | 100 | 58 | 42 | 58.0% |
| `val_unseen` | 200 | 114 | 86 | 57.0% |

`manifest.tsv`와 `summary.tsv`를 함께 저장해 이후 학습 seed와 split seed가 섞이지 않게 한다.

## SAFE LSTM Training Result

SAFE repo에는 GR00T N1.6용 dataset loader/config를 추가했다.

- `/home/dongkyu/pdk_ws/SAFE/failure_prob/data/groot_n16.py`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/dataset/groot_n16.yaml`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/__init__.py`

Loader contract:

- split directory는 `train`, `val_seen`, `val_unseen`을 물리적으로 유지한다.
- per-step hidden feature `[4, 16, 1024]`를 읽는다.
- detector input은 `horizon_idx_rel=mean`, `diff_idx_rel=mean`으로 aggregation해 `[T, 1024]`가 된다.
- `val_seen`은 validation과 conformal calibration에 쓰고, `val_unseen`은 held-out unseen-task 평가에 쓴다.

학습 script:

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_mean_mean.sh
```

실행한 설정:

- model: SAFE `lstm`
- seeds: `0`, `1`, `2`
- epochs: `1000`
- batch size: `64`
- lr: `3e-4`
- lambda_reg: `1e-2`
- W&B project: `vla-safe`
- timing plots: disabled, because current data has episode-level success/failure only and frame-level failure-onset label이 없다.

완료 checkpoint:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_train_logs/groot_n16-robocasa_seen4_unseen2_openDrawer_pnpCab_100ep-lstm-seed0_mean_mean/20260520/110404/model_final.ckpt
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_train_logs/groot_n16-robocasa_seen4_unseen2_openDrawer_pnpCab_100ep-lstm-seed1_mean_mean/20260520/110559/model_final.ckpt
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_train_logs/groot_n16-robocasa_seen4_unseen2_openDrawer_pnpCab_100ep-lstm-seed2_mean_mean/20260520/110644/model_final.ckpt
```

`20260520/110237`의 seed0 run은 timing plot label assertion으로 끝난 실패 attempt라서 결과로 쓰지 않는다. `smoke_mean_mean`은 1 epoch wiring smoke라서 결과로 쓰지 않는다.

3-seed 평균 성능:

| metric | mean ± std |
|---|---:|
| `val_seen` final-step ROC-AUC | `0.748 ± 0.096` |
| `val_seen` final-step PRC-AUC | `0.674 ± 0.124` |
| `val_unseen` final-step ROC-AUC | `0.700 ± 0.056` |
| `val_unseen` final-step PRC-AUC | `0.676 ± 0.026` |
| `val_seen` early max-so-far ROC-AUC | `0.768 ± 0.062` |
| `val_seen` early max-so-far PRC-AUC | `0.683 ± 0.112` |
| `val_unseen` early max-so-far ROC-AUC | `0.701 ± 0.047` |
| `val_unseen` early max-so-far PRC-AUC | `0.677 ± 0.065` |
| `val_seen` end max-so-far ROC-AUC | `0.950 ± 0.043` |
| `val_seen` end max-so-far PRC-AUC | `0.950 ± 0.042` |
| `val_unseen` end max-so-far ROC-AUC | `0.849 ± 0.121` |
| `val_unseen` end max-so-far PRC-AUC | `0.864 ± 0.107` |

Threshold / CP 결과:

| metric | mean ± std |
|---|---:|
| `val_seen` best threshold bal-acc | `0.854 ± 0.047` |
| `val_seen` best threshold T-det | `0.653 ± 0.049` |
| `val_unseen` best threshold bal-acc | `0.754 ± 0.031` |
| `val_unseen` best threshold T-det | `0.702 ± 0.019` |
| split CP alpha `0.2`, by earliest stop, bal-acc | `0.634 ± 0.027` |
| split CP alpha `0.2`, by earliest stop, TPR | `0.535 ± 0.131` |
| split CP alpha `0.2`, by earliest stop, TNR | `0.734 ± 0.097` |
| functional CP alpha `0.2`, by earliest stop, bal-acc | `0.586 ± 0.037` |
| functional CP alpha `0.2`, by earliest stop, TPR | `0.209 ± 0.107` |
| functional CP alpha `0.2`, by earliest stop, TNR | `0.962 ± 0.035` |

해석:

- wiring은 닫혔다. GR00T N1.6 rollout feature가 SAFE loader를 통과하고, LSTM 학습/validation/CP table 생성/checkpoint 저장까지 완료됐다.
- trajectory 후반 또는 전체 max-so-far 기준의 success/failure separability는 의미 있게 나온다.
- earliest-stop 기준 CP intervention은 아직 보수적이다. 특히 functional CP는 TNR은 높지만 TPR이 낮아 failure를 놓치는 쪽이다.
- taskwise로는 unseen `PnPCounterToCab`은 early에도 어느 정도 분리되지만, unseen `OpenDrawer`는 early separability가 약하다. seed0/1에서는 end 기준도 약하고, seed2는 end 기준은 강하지만 early 기준은 여전히 약하다.
- 이 상태는 “N1.6 SAFE detector baseline”으로는 쓸 수 있지만, 강한 proactive intervention claim에는 부족하다.

## SAFE Feature Visualization

SAFE 논문 Figure 1류의 latent-space 진단은 detector score나 CP threshold가 아니라, SAFE loader가 만든 per-timestep detector input feature를 대상으로 한다. 현재 N1.6 설정에서는 rollout의 `[T, 4, 16, 1024]` feature를 `horizon_idx_rel=mean`, `diff_idx_rel=mean`으로 aggregation한 `[T, 1024]` vector를 t-SNE에 넣는다.

Visualization 산출물은 `notebooks/`가 아니라 GR00T N1.6 eval output tree 아래에 둔다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep
```

각 visualization directory는 `feats_projected_skip1.pkl`, `feats_vis_skip1-succ.png`, `feats_vis_skip1-taskid.png`, `manifest.json`을 가진다. task structure와 success/failure signal을 한 그림에서 보기 위해 후처리 overlay도 저장한다.

- `feats_vis_skip1-taskid_failred.png`: 기존 task-id t-SNE 좌표를 그대로 쓰고, success frame은 task id 색, failure rollout frame은 단색 빨강으로 칠한다.
- `feats_vis_skip1-taskid_failure_overlay.png`: task id 색상 위에 실패 rollout frame을 검은 테두리로 겹친다.
- `feats_vis_skip1-task_success_facets.png`: task별 subplot 안에서 success rollout frame은 파란색, failure rollout frame은 episode 내 상대 시간에 따라 붉게 표시한다.

`manifest.json`에는 source split/task, projector, aggregation, rollout count, feature count, 생성된 output 파일명을 기록한다.

현재 생성된 t-SNE artifacts:

| scope | rollout | timestep feature | path |
|---|---:|---:|---|
| all splits | 600 | 18,428 | `safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep/all/tsne_mean_mean` |
| `val_unseen` | 200 | 5,660 | `safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen/tsne_mean_mean` |
| `val_unseen/OpenDrawer` | 100 | 2,041 | `safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen_OpenDrawer/tsne_mean_mean` |
| `val_unseen/PnPCounterToCab` | 100 | 3,619 | `safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen_PnPCounterToCab/tsne_mean_mean` |

재생성 runner:

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py
```

Overlay runner:

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py
```

예시:

```bash
/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py \
  --scope val_unseen \
  --task PnPCounterToCab \
  --projector tsne

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py \
  outputs/eval/robocasa/groot_n16/safe_feature_vis/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen/tsne_mean_mean
```

초기 관찰:

- `val_unseen` 전체로 보면 task structure와 success/failure signal이 함께 섞인다.
- `OpenDrawer` 단독 t-SNE는 success/failure separation이 약하다.
- `PnPCounterToCab` 단독 t-SNE는 실패 rollout 후반부로 보이는 red/orange region이 더 뚜렷하다.
- overlay 기준으로도 `PnPCounterToCab`은 late-failure frame이 특정 영역에 비교적 많이 몰리지만, `OpenDrawer`는 success/failure가 더 강하게 섞인다.
- 따라서 현재 latent-space evidence는 task-dependent하다. 전역 t-SNE 한 장만으로 논문 Figure 1 수준의 강한 claim을 하기는 어렵고, taskwise/trajectory-line view가 다음 진단 대상이다.

## Files

ZMQ SAFE:

- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/serve/feature_server.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_six_task_official_uv_host.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_mean_mean.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py`

HTTP:

- `/home/dongkyu/pdk_ws/temporal_vla/scripts/serve/groot.py`
- `/home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml`

Validation utilities:

- `/home/dongkyu/pdk_ws/temporal_vla/docs/robocasa_task_name_mapping.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/adr/0001-dedicated-safe-groot-n16-zmq-server.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/safe/groot_n16_robocasa_wiring.md`

## Next Steps

1. Pick the default checkpoint among seed0/1/2. Seed1 has the best `val_unseen` threshold bal-acc, while seed0 has the best early unseen ROC/PRC.
2. Inspect detector behavior by task and by success/failure trajectory, especially `OpenDrawer` unseen behavior.
3. Run feature aggregation ablation: `horizon_idx_rel` and `diff_idx_rel` in `{first,last,mean}`.
4. Separately run HTTP-vs-ZMQ action equivalence before using HTTP for SR evaluation.
