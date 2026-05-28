# SAFE x GR00T N1.6 RoboCasa Wiring

이 문서는 GR00T N1.6 RoboCasa와 SAFE detector를 연결할 때 사용할 serving/eval 경로를 고정한다. 결론은 HTTP와 ZMQ를 모두 유지하되, 현재 실험 기준선은 ZMQ로 둔다.

## 현재 결론

- `ZMQ official eval`: GR00T N1.6 RoboCasa 성공률을 판단하는 기준 경로.
- `ZMQ SAFE feature server`: SAFE rollout 수집 기준 경로. official RoboCasa 클라이언트 환경을 쓰면서 action과 feature를 함께 저장한다.
- `HTTP /act`: 프로젝트 공통 serving API로 유지한다. GR00T N1.6 RoboCasa 성공률 기준선은 ZMQ official eval로 둔다.
- RoboCasa365 `target_atomic_seen18` 100ep/task collection은 완료됐다. `target_atomic_seen18_ckpt120000_robocasa365_100ep`는 `18 x 100 = 1800` episode triplet을 포함하고, seed range는 task별 `100000..100099`, verifier 기준 `status=ok`, total SR `967/1800 = 53.7%`다.

HTTP 경로는 프로젝트 공통 serving API로 유지한다. official ZMQ OpenDrawer baseline은 README 수준이고, HTTP 연동은 observation/action chunk 동등성 검증을 거쳐 SR 지표로 편입한다.

## Checkpoint And Env

체크포인트:

- host: `/home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000`
- container: `/temporal_vla/outputs/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000`
- source: `Abhi03/grootn16_robocasa365_multitask_learning/checkpoint-120000`
- profile: `/home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa365_ckpt120000.yaml`

`groot__robocasa365_ckpt120000.yaml`은 사용자-facing 이름을 RoboCasa365 checkpoint 기준으로 둔다. 내부 `model_specific.embodiment_tag`는 checkpoint metadata에 맞춰 `NEW_EMBODIMENT`를 사용한다.

RoboCasa 환경:

- 기준 env: `/home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python`
- `robocasa_v02`: `/home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T/external_dependencies/robocasa`
- `robocasa365`: `/home/dongkyu/pdk_ws/temporal_vla/src/benchmarks/robocasa`

GR00T official eval과 기본 SAFE collection은 RoboCasa v0.2 (`robocasa_v02`) task name을 쓴다. RoboCasa365 수집은 `robocasa365` task name을 쓰므로 task mapping을 명시적으로 유지한다.

## Shared Run Config

SAFE N1.6 RoboCasa scripts는 per-script hardcoded path 대신 공통 run identity를 아래 두 파일에서 가져온다.

| file | consumer | role |
|---|---|---|
| `scripts/safe/groot_n16/robocasa/run_config.py` | Python scripts | `Path` 객체와 final detector selection의 canonical config |
| `scripts/safe/groot_n16/robocasa/run_config.sh` | Bash wrappers | shell 환경에서 쓰는 adapter config |

Python script는 `run_config.py`를 import하고, Bash wrapper는 `run_config.sh`를 source한다. 새 run/task set을 만들 때 개별 script 안의 output path를 직접 고치지 말고 아래 값을 override한다.

| variable | scope | meaning |
|---|---|---|
| `ROBOCASA_SAFE_RUN_ID` | Python/Bash shared run root | top-level SAFE run directory, default `safe_seen4_unseen2_100ep` |
| `ROBOCASA_SAFE_EXPERIMENT_ID` | feature visualization | visualization experiment directory name |
| `ROBOCASA_SAFE_FINAL_HORIZON_IDX_REL` | detector train/eval | final detector horizon aggregation |
| `ROBOCASA_SAFE_FINAL_DIFF_IDX_REL` | detector train/eval | final detector diffusion aggregation |
| `ROBOCASA_SAFE_HPARAM_SWEEP_ID` | detector train/eval | hparam sweep experiment directory |
| `RUN_ID` | collection wrappers only | per-collection output suffix under `experiments/collection_smoke/rollouts_${RUN_ID}` |

`ROBOCASA_SAFE_RUN_ID`는 전체 SAFE run tree를 바꾸고, `RUN_ID`는 한 번의 rollout collection 산출물 이름만 바꾼다.

현재 default run config 확인:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

python -c "import sys; sys.path.insert(0, 'scripts/safe/groot_n16/robocasa'); import run_config as c; print(c.RUN_ROOT); print(c.SPLIT_ROOT); print(c.FINAL_LSTM_RUN_DIR); print(c.FINAL_DETECTOR_DIR)"
```

## Script Pipeline Order

이 디렉터리의 파일은 아래 순서로 읽으면 된다. 각 step은 앞 step의 artifact를 입력으로 삼는다.

| order | file | purpose | main input -> output |
|---:|---|---|---|
| 0 | `run_config.py` | Python-side shared run identity, paths, final detector selection | constants -> Python defaults |
| 0 | `run_config.sh` | Bash-side shared run identity adapter | env/defaults -> shell variables |
| 0 | `safe_feature_vectors.py` | `[K,H,D]` Flow-matching SAFE feature를 timestep-level SAFE feature vector로 aggregate | rollout pkl + aggregation command -> `[T,D]` features |
| 1 | `serve/feature_server.py` | ZMQ feature server exposing `get_action_with_features` | GR00T checkpoint -> action + unpooled feature |
| 2 | `collect/collect_rollout.py` | one-task/one-range rollout collector | ZMQ feature server + RoboCasa env -> SAFE-readable pkl/mp4/csv |
| 2 | `collect/collect_task_set_official_uv_host.sh` | preferred host-side task-set collection wrapper using official `robocasa_uv` env | task set + seeds -> raw rollout directories |
| 2 | `collect/collect_task_set_in_container.sh` | collection wrapper for already-running container shell context | task set + seeds -> raw rollout directories |
| 2 | `collect/collect_task_set_via_docker_exec.sh` | host wrapper that enters the Docker container for collection | task set + seeds -> raw rollout directories |
| 3 | `split/prepare_seen4_unseen2_split.py` | paper-faithful SAFE split construction | raw rollouts -> `train` / `val_seen` / `val_unseen` split tree |
| 4 | `train/train_lstm_mean_mean.sh` | legacy mean/mean SAFE-LSTM baseline | split tree -> baseline train logs |
| 4 | `train/train_lstm_aggregation_ablation.sh` | aggregation ablation over horizon/diffusion axes | split tree -> aggregation train logs |
| 5 | `analyze/summarize_lstm_aggregation_ablation.py` | select candidate aggregation from ablation logs | aggregation train logs -> json/md summary |
| 6 | `train/train_lstm_hparam_sweep.sh` | hparam sweep with selected final aggregation defaults | split tree -> hparam train logs |
| 7 | `analyze/summarize_lstm_hparam_sweep.py` | summarize hparam sweep and selection rule | hparam train logs -> json/md summary |
| 8 | `analyze/finalize_lstm_detector.py` | pin final checkpoint and generate fixed/CP/functional-CP artifacts | selected run dir -> `final_detector/` |
| 9 | `vis/plot_safe_conformal_curves.py` | SAFE Figure-8-style CP operating curves | `final_detector/*.csv` -> CP figures |
| 10 | `vis/run_feature_visualization.py` | adapter around SAFE's original feature visualizer | split tree -> SAFE-style t-SNE/UMAP artifacts |
| 10 | `vis/plot_safe_style_feature_space.py` | native SAFE-style feature-space plotter | split tree -> feature-space plots |
| 11 | `vis/plot_task_success_overlay.py` | overlay task/success labels on projected feature artifacts | feature visualization pkl -> overlay images |
| 12 | `vis/compute_feature_silhouette.py` | static feature-space separability diagnostics | split tree / projection pkl -> silhouette tables |
| 13 | `analyze/diagnose_rollout_mean_feature_separability.py` | rollout-mean aggregation separability diagnostic | split tree -> rollout-level separability tables/plots |

Default visualization and silhouette scripts now use the final detector aggregation from `run_config.py`. To reproduce the early `mean/mean` artifacts, pass `--horizon-idx-rel mean --diff-idx-rel mean` explicitly.

## ZMQ Official Eval

목적: pretrained GR00T N1.6 PandaOmron baseline이 RoboCasa v0.2에서 정상 동작하는지 확인한다.

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

이 결과를 pretrained PandaOmron baseline 정상 동작 기준선으로 둔다.

## ZMQ SAFE Feature Collection

목적: official RoboCasa eval 환경을 유지하면서 SAFE 학습용 pkl schema를 만든다.

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

> **재현 주의:** seed는 기록 라벨일 뿐, env 생성 시 전달되지 않아 scene을 고정하지 못한다 — 이 collection은 seed로 재현 불가. 메커니즘·fix는 [robocasa_env_reproducibility.md](../robocasa_env_reproducibility.md) §11 참조.

Verifier summary:

```text
root=outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/raw_rollouts
tasks=18 episodes_per_task=100
completed=1800 expected=1800
summary_seeds=100000..100099 unique=100 rows=1800 formula=seed_start+episode_idx
status=ok
```

Per-task SR은 [SAFE reproduction report](groot_n16_robocasa_safe_report.md#robocasa365-18-task-collection-결과)에 둔다. 이 collection은 raw rollout/latent-feature producer artifact이며, 기존 `safe_seen4_unseen2_100ep` 6-task detector split과는 별도 run이다.

저장 위치:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/experiments/collection_smoke/rollouts_${RUN_ID}
```

SAFE feature contract:

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

## Paper-Faithful SAFE Split

SAFE 논문/레포 방식에 맞춰 task-level split과 seen-task episode split을 사용한다.

- raw rollout cap: `max_rollouts_per_task: 100`
- task-level split: `unseen_task_ratio: 0.25`
- seen-task episode split: `seen_train_ratio: 0.75`
- detector train: seen task의 train rollout
- threshold / conformal calibration: `val_seen`
- final evaluation: `val_unseen`

따라서 6개 task에서는 `round(0.25 * 6) = 2`개 task가 unseen이 되고, 나머지 4개 task가 seen이 된다. 각 task를 100 rollout으로 맞추면 전체 600 rollout이며, split은 대략 다음 크기가 된다. SAFE 레포 DROID 설정의 `60/task`보다 큰 cap이지만, task별 SR이 낮아 성공 rollout이 부족할 수 있으므로 N1.6 RoboCasa에서는 `100/task`를 사용한다.

이번 small reproduction에서는 taxonomy constraint를 둔다. unseen task는 Open 계열 1개와 PnP 계열 1개로 고정한다. 실제 unseen task는 `OpenDrawer`와 `PnPCounterToCab`이다. 이 선택은 `OpenSingleDoor`를 seen 쪽에 남겨 robocasa365의 `OpenCabinet` 대응 경로를 계속 점검할 수 있게 하고, `val_unseen` 성공/실패 비율도 `114/86`으로 균형을 유지한다.

| split | source | count |
|---|---|---:|
| `train` | 4 seen tasks × 75 rollout | 300 |
| `val_seen` | 4 seen tasks × 25 rollout | 100 |
| `val_unseen` | 2 unseen tasks × 100 rollout | 200 |

`val_seen`은 validation과 conformal calibration 역할을 함께 한다. 별도 CP-only split이나 seen-task test split을 만들면 논문식 재현에서 벗어난다.

Split 생성 전 count 확인:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py \
  --source-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/raw_rollouts \
  --split-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/split \
  --dry-run
```

Split 생성은 새 run에서 한 번 수행한다. 현재 run의 source of truth는 아래 `split` directory다. 기존 `split` directory가 있으면 script가 중단되므로, 재생성은 새 `ROBOCASA_SAFE_RUN_ID`에서 수행한다.

Split 최초 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py \
  --source-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/raw_rollouts \
  --split-root outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/split
```

생성된 split:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/split
```

Split summary:

| split | total | success | failure | SR |
|---|---:|---:|---:|---:|
| `train` | 300 | 141 | 159 | 47.0% |
| `val_seen` | 100 | 58 | 42 | 58.0% |
| `val_unseen` | 200 | 114 | 86 | 57.0% |

`manifest.tsv`와 `summary.tsv`를 함께 저장해 이후 학습 seed와 split seed를 분리한다.

## SAFE LSTM Final Detector

SAFE repo에는 GR00T N1.6용 dataset loader/config를 추가했다.

- `/home/dongkyu/pdk_ws/SAFE/failure_prob/data/groot_n16.py`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/dataset/groot_n16.yaml`
- `/home/dongkyu/pdk_ws/SAFE/failure_prob/conf/__init__.py`

Loader contract:

- split directory는 `train`, `val_seen`, `val_unseen`을 물리적으로 유지한다.
- per-step hidden feature `[4, 16, 1024]`를 읽는다.
- detector input은 train/eval config의 aggregation에 따라 만든다.
- 최종 SAFE-LSTM은 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`를 사용하므로 detector input은 `[T, 2048]`이다.
- `val_seen`은 validation과 conformal calibration에 쓰고, `val_unseen`은 held-out unseen-task 평가에 쓴다.

관련 script:

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.sh
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/safe_feature_vectors.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_aggregation_ablation.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_hparam_sweep.py
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py
```

`run_config.py` / `run_config.sh`가 run id, output root, 최종 aggregation, hparam sweep directory의 단일 출처다. `safe_feature_vectors.py`가 `[K,H,D]` Flow-matching SAFE feature를 timestep-level SAFE feature vector로 aggregation하는 공용 Module이다.

Aggregation ablation 실행:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

WANDB_MODE=online \
bash scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh
```

Aggregation ablation 요약:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_aggregation_ablation.py
```

최종 aggregation 기준 hparam sweep 실행:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

WANDB_MODE=online \
bash scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh
```

Hparam sweep 요약:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_hparam_sweep.py
```

Final detector 고정 및 CP artifact 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py
```

최종 선택:

- model: SAFE `lstm`
- epochs: `1000`
- batch size: `64`
- lr: `3e-4`
- lambda_reg: `1`
- aggregation: `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`
- selected checkpoint seed: `2`
- W&B project: `vla-safe`
- timing plots: disabled, because current data has episode-level success/failure only and inference-step-level failure-onset label이 없다.

최종 checkpoint:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/model_final.ckpt
```

최종 산출물:

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector
```

이 directory에는 `model_final.ckpt`, `config.yaml`, `manifest.json`, `final_operating_point.json`, `fixed_threshold_eval.csv`, `split_cp_eval.csv`, `functional_cp_eval.csv`, `functional_cp_bands.npz`, `per_rollout_scores.csv`, `README.md`가 있다.

초기 aggregation ablation 결과:

| rank | horizon | diff | dim | val_seen bal-acc | val_seen T-det | val_seen ROC-AUC | val_unseen bal-acc | val_unseen T-det | val_unseen ROC-AUC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `concat-2` | `0.0` | 2048 | `0.932 ± 0.011` | `0.574 ± 0.026` | `0.922 ± 0.034` | `0.785 ± 0.021` | `0.694 ± 0.015` | `0.749 ± 0.053` |
| 10 | `mean` | `mean` | 1024 | `0.854 ± 0.039` | `0.653 ± 0.040` | `0.854 ± 0.042` | `0.754 ± 0.025` | `0.702 ± 0.015` | `0.779 ± 0.015` |

이후 SAFE-style feature visualization과 timestep-level separability 진단에서 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`를 최종 후보로 고정하고 hparam sweep을 다시 수행했다.

Hyperparameter sweep 결과:

| metric | mean ± std |
|---|---:|
| best hparam | `lr=3e-4`, `lambda_reg=1` |
| `val_seen` bal-acc | `0.985 ± 0.012` |
| `val_seen` T-det | `0.539 ± 0.130` |
| `val_seen` ROC-AUC | `0.995 ± 0.006` |
| `val_unseen` bal-acc | `0.981 ± 0.028` |
| `val_unseen` T-det | `0.642 ± 0.052` |
| `val_unseen` ROC-AUC | `0.994 ± 0.008` |

Final pinned detector 결과:

| item | value |
|---|---:|
| selected checkpoint | `seed2` |
| fixed threshold baseline | `0.5487` |
| fixed threshold `val_unseen` bal-acc | `1.0000` |
| fixed threshold `val_unseen` TPR/TNR | `1.0000 / 1.0000` |
| fixed threshold `val_unseen` mean T-det | `0.8194` |

최종 운영점은 split conformal prediction으로 고정한다. Fixed threshold는 baseline으로 함께 기록한다.

| item | value |
|---|---:|
| method | `split_cp` |
| alpha | `0.2` |
| eval time | `by final end` |
| calibration label | `neg_success` |
| threshold | `0.5301596522331238` |
| `val_unseen` bal-acc | `0.9518` |
| `val_unseen` TPR/TNR | `1.0000 / 0.9035` |
| `val_unseen` acc/F1 | `0.9450 / 0.9399` |
| `val_unseen` mean T-det | `0.4114` |

해석:

- wiring은 닫혔다. GR00T N1.6 rollout feature가 SAFE loader를 통과하고, LSTM 학습/validation/CP table 생성/checkpoint 저장까지 완료됐다.
- 논문식 feature aggregation ablation과 LSTM hyperparameter sweep을 수행했고, 최종 detector/checkpoint/threshold를 별도 산출물로 고정했다.
- `val_unseen`에서도 failure monitoring 성능은 강하다. 최종 CP 운영점의 mean T-det는 `0.4114`로 이전 운영점보다 앞당겨졌다. 현재 label scope는 rollout-level success/failure이며, proactive intervention 평가는 inference-step-level onset/intervention label을 추가한 뒤 다룬다.
- CP alpha sweep은 최종 선택된 aggregation/hparam/seed2 checkpoint의 score 위에서 수행했다.
- Functional CP band도 SAFE repo 구현 그대로 계산했다. `alpha=0.2`, `by final end`, success-calibrated functional CP는 `val_unseen` bal-acc `0.9605`, TPR/TNR `1.0000 / 0.9211`, mean T-det `0.4251`이다. Best by-final-end functional point는 `alpha=0.05`에서 bal-acc `1.0000`, mean T-det `0.6982`다.
- static latent-space failure zone 근거는 약하다. detector 성능은 정적 cluster 분리보다 LSTM score trajectory와 threshold crossing으로 해석한다.

SAFE 논문 Figure 8류의 CP 시각화는 다음 위치에 생성한다. 이 그림은 CP operating point curve다.

```text
/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/conformal_figure/by_final_end
```

기본 산출물은 `cp_balacc_tdet.{png,pdf}`와 `cp_alpha_{fpr,fnr,tpr,tnr,bal_acc}.{png,pdf}`다. 입력은 `final_detector/split_cp_eval.csv`와 `final_detector/functional_cp_eval.csv`이며, 로컬 CSV를 source of truth로 사용한다.

CP curve 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py \
  --eval-time "by final end"
```

## SAFE Feature Visualization

SAFE 논문 Figure 1류의 latent-space 진단은 SAFE loader가 만든 per-timestep detector input feature를 대상으로 한다. 초기 t-SNE artifact는 `mean/mean` aggregation으로 만들었고, 최종 detector의 aggregation은 `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`이다. 현재 visualization/silhouette script의 기본값은 최종 detector aggregation이며, 초기 artifact를 재생성할 때만 `--horizon-idx-rel mean --diff-idx-rel mean`을 명시한다.

최종 aggregation 기준 detector input:

```text
[T, 4, 16, 1024] -> horizon mean, diff concat(first,last) -> [T, 2048]
```

Visualization 산출물은 GR00T N1.6 eval output tree 아래에 둔다.

```text
/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep
```

각 visualization directory는 `feats_projected_skip1.pkl`, `feats_vis_skip1-succ.png`, `feats_vis_skip1-taskid.png`, `manifest.json`을 가진다. task structure와 success/failure signal을 한 그림에서 보기 위해 후처리 overlay도 저장한다.

- `feats_vis_skip1-taskid_failred.png`: 기존 task-id t-SNE 좌표를 그대로 쓰고, success datapoint(=inference)은 task id 색, failure rollout의 datapoint(=inference)은 단색 빨강으로 칠한다.
- `feats_vis_skip1-taskid_failure_overlay.png`: task id 색상 위에 실패 rollout의 datapoint을 검은 테두리로 겹친다.
- `feats_vis_skip1-task_success_facets.png`: task별 subplot 안에서 success rollout의 datapoint(=inference)은 파란색, failure rollout의 datapoint(=inference)은 episode 내 상대 시간에 따라 붉게 표시한다.

`manifest.json`에는 source split/task, projector, aggregation, rollout count, feature count, 생성된 output 파일명을 기록한다.

현재 생성된 t-SNE artifacts:

| scope | rollout | timestep feature | path |
|---|---:|---:|---|
| all splits | 600 | 18,428 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/all/tsne_mean_mean` |
| `val_unseen` | 200 | 5,660 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen/tsne_mean_mean` |
| `val_unseen/OpenDrawer` | 100 | 2,041 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen_OpenDrawer/tsne_mean_mean` |
| `val_unseen/PnPCounterToCab` | 100 | 3,619 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen_PnPCounterToCab/tsne_mean_mean` |
| all splits, SAFE-style | 600 | 18,428 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/safe_style_visualize_features/all_hmean_dconcat_2-tsne` |
| `val_unseen`, SAFE-style | 200 | 5,660 | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/safe_style_visualize_features/val_unseen_hmean_dconcat_2-tsne` |

Silhouette 산출물:

| aggregation | path | conclusion |
|---|---|---|
| `mean/mean` | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/silhouette_mean_mean` | success/failure silhouette near zero |
| `concat-2/0.0` | `safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/silhouette_hconcat2_d0p0` | 초기 detector-metric 후보에서도 static failure zone은 약함 |

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
  outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/visualizations/feature_space/seen4_unseen2_openDrawer_pnpCab_100ep/val_unseen/tsne_mean_mean
```

SAFE-style feature plot 생성:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/plot_safe_style_feature_space.py \
  --scope val_unseen \
  --projector tsne
```

Silhouette 진단:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py
```

Rollout-mean separability 진단:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla

/home/dongkyu/miniforge3/envs/vla-safe/bin/python \
  scripts/safe/groot_n16/robocasa/analyze/diagnose_rollout_mean_feature_separability.py
```

초기 관찰:

- `val_unseen` 전체로 보면 task structure와 success/failure signal이 함께 섞인다.
- `OpenDrawer` 단독 t-SNE는 success/failure separation이 약하다.
- `PnPCounterToCab` 단독 t-SNE는 실패 rollout 후반부로 보이는 red/orange region이 더 뚜렷하다.
- overlay 기준으로도 `PnPCounterToCab`은 late-failure datapoint(=inference)이 특정 영역에 비교적 많이 몰리지만, `OpenDrawer`는 success/failure가 더 강하게 섞인다.
- 최종 aggregation의 original 2048-D silhouette에서도 `val_unseen` success/failure Mahalanobis score는 `-0.0027`이고, task+failure도 음수다. static failure zone 근거는 약하고, detector score trajectory 중심으로 해석한다.

## HTTP Path And SR Recovery

목적: 프로젝트 공통 VLA serving/eval API를 유지하고, ZMQ official 기준선과 action parity를 맞춘다.

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

Health check:

```bash
curl http://127.0.0.1:8000/health
```

역할:

- HTTP는 heterogeneous VLA serving 경로다.
- GR00T N1.6 RoboCasa SR 기준선은 ZMQ official eval이다.
- HTTP SR은 observation/action parity 확인 후 SAFE/GR00T N1.6 성능 지표에 편입한다.
- 낮은 HTTP SR에서는 wiring/runner/schema mismatch를 우선 점검한다.

현재 HTTP와 ZMQ 비교에는 transport 이외의 차이가 함께 있다.

- HTTP 경로는 보통 project runner를 타며 `src/benchmarks/robocasa`의 robocasa365 v1.0 환경을 사용한다.
- ZMQ official 경로는 `src/policies/Isaac-GR00T/external_dependencies/robocasa`의 RoboCasa v0.2 (`robocasa_v02`) 환경을 사용한다.
- transport, env version, task class name, observation schema, action application 방식 차이가 함께 섞여 있다.

공통 task로 고른 5개는 v0.2와 v1.0 사이에서 의미적으로 대응되는 task다. `OpenSingleDoor`/`OpenCabinet`처럼 official SR이 높은 task에서 HTTP SR이 크게 낮으면 policy wiring을 먼저 점검한다.

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

## Files

ZMQ SAFE:

- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/run_config.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/safe_feature_vectors.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/serve/feature_server.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_task_set_in_container.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_task_set_official_uv_host.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/collect/collect_task_set_via_docker_exec.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/split/prepare_seen4_unseen2_split.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_mean_mean.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/train/train_lstm_hparam_sweep.sh`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_aggregation_ablation.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/summarize_lstm_hparam_sweep.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/finalize_lstm_detector.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/analyze/diagnose_rollout_mean_feature_separability.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_safe_style_feature_space.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_task_success_overlay.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/plot_safe_conformal_curves.py`
- `/home/dongkyu/pdk_ws/temporal_vla/scripts/safe/groot_n16/robocasa/vis/compute_feature_silhouette.py`

Final detector artifacts:

- `/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/README.md`
- `/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/model_final.ckpt`
- `/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector/final_operating_point.json`

HTTP:

- `/home/dongkyu/pdk_ws/temporal_vla/scripts/serve/groot.py`
- `/home/dongkyu/pdk_ws/temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml`

Validation utilities:

- `/home/dongkyu/pdk_ws/temporal_vla/docs/robocasa_task_name_mapping.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/adr/0001-dedicated-safe-groot-n16-zmq-server.md`
- `/home/dongkyu/pdk_ws/temporal_vla/docs/safe/groot_n16_robocasa_wiring.md`

## Next Steps

1. Separately run HTTP-vs-ZMQ action equivalence before using HTTP for SR evaluation.
2. If proactive intervention is the goal, define or collect inference-step-level failure onset/intervention labels.
3. Optionally compare `--feature-slice all` (`H=50`) against the current valid-horizon (`H=16`) export.
