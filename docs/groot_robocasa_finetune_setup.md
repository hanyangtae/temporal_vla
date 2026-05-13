# GR00T N1.6 RoboCasa fine-tuning setup

Isaac-GR00T `n1.6-release` 기반으로 RoboCasa PandaOmron per-task LeRobot v2.1 mixture 를 학습하고, GR00T fine-tuning을 host 또는 Docker에서 실행하기 위한 runbook이다.

이 문서는 두 학습 scope를 구분한다.

1. `pretrain/atomic` 10-task baseline: legacy pretrain baseline.
2. `target/atomic` 15-task: target atomic-seen fine-tuning scope.

## 요약

- 학습 entrypoint는 local mirror `scripts/train/launch_finetune_ttt.py`를 사용한다. TTT 인자를 비워 baseline GR00T fine-tuning으로 실행한다.
- 로컬 wrapper `scripts/train/groot_robocasa_finetune.sh`는 repo 경로, checkpoint, dataset, modality config, 주요 hyperparameter를 정리해서 넘기는 역할만 한다.
- target atomic-seen 15-task 학습은 `scripts/train/groot_robocasa_target_finetune.sh`를 사용한다. 이 wrapper가 15개 target dataset path를 만든 뒤 baseline wrapper에 위임한다.
- `launch_finetune.py`, `finetune_config.py`, model setup 등 upstream GR00T 핵심 학습 코드는 수정하지 않는다.
- 여러 RoboCasa task는 `--dataset-path` 하나에 `:`로 join해서 넘긴다. mirror entry가 이를 task별 dataset path로 split해서 mixture를 만든다.
- `ROBOCASA_PANDA_OMRON` enum 존재만으로는 충분하지 않다. 학습 시 `configs/policies/groot_robocasa_panda_omron_config.py`를 `--modality_config_path`로 import해서 modality config를 등록해야 한다.
- 학습 직후 확인할 eval quickstart는 이 문서에 포함한다. 자세한 eval 조건, 로그 저장, SR 계산, troubleshooting은 `docs/groot_robocasa_eval_setup.md`를 기준으로 한다.
- Full DiT fine-tuning은 16GB GPU에서 OOM이 난다. 관찰된 실행에서는 약 36GB VRAM을 사용했고, 20,000 step은 약 2시간 걸렸다.

## 필요한 파일

서버에서 학습을 돌리려면 최소한 아래가 있어야 한다.

```text
temporal_vla/
├── scripts/train/groot_robocasa_finetune.sh
├── scripts/train/groot_robocasa_target_finetune.sh
├── scripts/train/launch_finetune_ttt.py
├── configs/policies/groot_robocasa_panda_omron_config.py
├── checkpoints/nvidia/GR00T-N1.6-3B/
├── data/robocasa/v1.0/pretrain/atomic/<Task>/20250819/lerobot/
├── data/robocasa/v1.0/target/atomic/<Task>/<date>/lerobot/
└── src/policies/Isaac-GR00T/
```

복사하지 않아도 되는 것:

```text
.venv/
outputs/
wandb/
__pycache__/
.cache/
```

## 서버 사전 준비

권장 환경:

- Linux + NVIDIA GPU + CUDA 12.x
- 직접 Python/uv 환경 또는 Docker `groot` container
- 디스크 여유:
  - checkpoint와 extracted dataset만 둘 경우 50GB 이상 권장
  - HF cache/raw task archive까지 보관할 경우 200GB 이상 권장
- VRAM:
  - 16GB: full fine-tuning 불가
  - 24GB: 매우 빡빡함
  - 32GB: 최소 실험선
  - 40GB/48GB 이상: 현실적

서버에서 repo를 새로 받는 경우:

```bash
git clone https://github.com/hanyangtae/temporal_vla.git
cd temporal_vla
git submodule update --init --recursive
```

Host 명령 예시는 `temporal_vla` git checkout 내부에서 실행하는 것을 전제로 `REPO_ROOT="$(git rev-parse --show-toplevel)"`를 사용한다. repo 밖에서 실행할 때는 `REPO_ROOT=/path/to/temporal_vla`처럼 직접 지정한다. Docker 내부 경로는 compose mount 기준으로 `/temporal_vla`를 사용한다.

GR00T uv 환경:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}/src/policies/Isaac-GR00T"

# uv가 없으면 먼저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# venv + dependency 설치
uv sync

# import 확인
uv run python -c "import gr00t; from gr00t.data.embodiment_tags import EmbodimentTag; print(EmbodimentTag.ROBOCASA_PANDA_OMRON)"
```

Modality config 등록 확인:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}/src/policies/Isaac-GR00T"

uv run python -c "import os, sys; sys.path.append(os.path.expanduser(os.environ['REPO_ROOT']) + '/configs/policies'); import groot_robocasa_panda_omron_config; from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS; print('robocasa_panda_omron' in MODALITY_CONFIGS)"
```

`True`가 나와야 한다.

## Task Scope

### Pretrain 10-task baseline

`scripts/train/groot_robocasa_finetune.sh` default는 아래 10개 `pretrain/atomic` task를 mixture로 학습한다.

1. `OpenDrawer`
2. `CloseDrawer`
3. `OpenCabinet`
4. `CloseCabinet`
5. `OpenFridge`
6. `CloseFridge`
7. `OpenMicrowave`
8. `CloseMicrowave`
9. `PickPlaceCounterToStove`
10. `PickPlaceCounterToSink`

`20250819` pretrain/human LeRobot v2.1 기준 task별 크기:

| Task | Episodes | Frames |
| --- | ---: | ---: |
| `OpenDrawer` | 102 | 20,488 |
| `CloseDrawer` | 110 | 15,670 |
| `OpenCabinet` | 107 | 37,492 |
| `CloseCabinet` | 105 | 27,754 |
| `OpenFridge` | 105 | 33,138 |
| `CloseFridge` | 106 | 26,888 |
| `OpenMicrowave` | 105 | 26,017 |
| `CloseMicrowave` | 105 | 20,075 |
| `PickPlaceCounterToStove` | 108 | 24,039 |
| `PickPlaceCounterToSink` | 108 | 22,410 |
| **Total** | **1,061** | **253,971** |

### Target atomic-seen 15-task

Target atomic-seen fine-tuning scope는 `scripts/train/groot_robocasa_target_finetune.sh`의 15개 task/date pair다.

| # | Task | Date |
| ---: | --- | --- |
| 1 | `CloseFridge` | `20250816` |
| 2 | `CloseToasterOvenDoor` | `20250818` |
| 3 | `CoffeeSetupMug` | `20250813` |
| 4 | `OpenCabinet` | `20250813` |
| 5 | `OpenDrawer` | `20250816` |
| 6 | `PickPlaceCounterToCabinet` | `20250811` |
| 7 | `PickPlaceCounterToStove` | `20250818` |
| 8 | `PickPlaceDrawerToCounter` | `20250820` |
| 9 | `PickPlaceSinkToCounter` | `20250813` |
| 10 | `PickPlaceToasterToCounter` | `20250817` |
| 11 | `SlideDishwasherRack` | `20250820` |
| 12 | `TurnOffStove` | `20250812` |
| 13 | `TurnOnElectricKettle` | `20250817` |
| 14 | `TurnOnMicrowave` | `20250813` |
| 15 | `TurnOnSinkFaucet` | `20250812` |

이 15개는 target atomic-seen 18개 중 `CloseBlenderLid`, `NavigateKitchen`, `OpenStandMixerHead`를 제외한 scope다.

## 데이터 origin

per-task LeRobot v2.1 dataset 을 다음 경로에 둔다 (TTT 트랙과 동일 경로).

```bash
data/robocasa/v1.0/pretrain/atomic/<Task>/20250819/lerobot/
data/robocasa/v1.0/target/atomic/<Task>/<date>/lerobot/
```

다운로드:

```bash
bash scripts/utils/download_robocasa_pretrain_human.sh
bash scripts/utils/download_robocasa_target_human.sh
# 또는 task 선택: bash ... OpenDrawer CloseDrawer
```

→ `data/robocasa/v1.0/{pretrain,target}/atomic/<Task>/<date>/lerobot/` 로 task별 dataset이 받아진다. UTexas Box 에서 tar 받아 압축 해제까지 한 번에.

각 task dataset 구조:

```text
data/chunk-000/episode_*.parquet
videos/chunk-000/observation.images.robot0_agentview_left/*.mp4
videos/chunk-000/observation.images.robot0_agentview_right/*.mp4
videos/chunk-000/observation.images.robot0_eye_in_hand/*.mp4
meta/{episodes.jsonl, episodes_stats.jsonl, info.json, modality.json, stats.json, tasks.jsonl}
```

`meta/modality.json` 의 video key 는 RoboCasa raw 카메라 이름 (`robot0_agentview_left/right`, `robot0_eye_in_hand`) 그대로이고, state/action key 는 PandaOmron 표준 (`base_position`, `base_rotation`, `end_effector_position_relative` 등) 이다. 즉 단일 modality config 가 pretrain 10-task와 target 15-task에 그대로 적용된다.

(참고: TTT 트랙은 추가로 `scripts/extract/prepare_robocasa_dataset.py` 로 `progress` 컬럼을 in-place 로 붙인다. baseline finetune 은 `progress` 를 참조하지 않으므로 이 단계 없이도 학습 가능. 이미 TTT 용으로 한 번 돌려놓은 dataset 이라면 그대로 baseline 에 써도 무해.)

### Workflow Notes

- Dataset은 task별 LeRobot directory를 유지하고, 학습 시 `DATASET_PATH`를 `:`로 join해서 mixture로 넘긴다.
- Upstream N1.6 `launch_finetune.py`의 `--dataset_path`는 단일 문자열이므로 여러 path를 그대로 나열하지 않는다. Multi-path 처리는 `scripts/train/launch_finetune_ttt.py`의 `:` split로 한다.
- Fine-tuned checkpoint 평가는 `docs/groot_robocasa_eval_setup.md`의 ZMQ server/client workflow를 기준으로 한다. `scripts/serve/groot.py` + `scripts/eval/robocasa_eval.py` 조합은 legacy workflow다.

## modality config

파일:

```text
configs/policies/groot_robocasa_panda_omron_config.py
```

`ROBOCASA_PANDA_OMRON` enum 은 upstream 에 등록되어 있지만 그 embodiment 의 modality config 자체는 등록되어 있지 않다. 학습 시 `--modality_config_path` 로 위 파일을 import 해야 `MODALITY_CONFIGS["robocasa_panda_omron"]` 이 채워진다. 안 넘기면 `KeyError: 'robocasa_panda_omron'`.

이 config 는 upstream `Isaac-GR00T` 코드를 직접 수정하지 않기 위해 repo 의 `configs/` 아래에 둔다.

등록 tag:

```python
EmbodimentTag.ROBOCASA_PANDA_OMRON
```

Video keys:

```text
robot0_agentview_left
robot0_agentview_right
robot0_eye_in_hand
```

State keys:

```text
end_effector_position_relative
end_effector_rotation_relative
gripper_qpos
base_position
base_rotation
```

Action keys:

```text
end_effector_position
end_effector_rotation
gripper_close
base_motion
control_mode
```

Language key:

```text
annotation.human.task_description
```

Action config 는 모두 `ActionRepresentation.ABSOLUTE` 로 설정했다. RoboCasa PandaOmron 의 action 이 모두 absolute 이기 때문이며, dataset 의 `meta/relative_stats.json` 도 `{}` 로 두면 된다.

## Train wrapper

파일:

```text
scripts/train/groot_robocasa_finetune.sh
```

이 wrapper는 자기 자신의 위치에서 repo root를 계산한다. 그래서 host의 어떤 `temporal_vla` checkout 경로와 Docker mount의 `/temporal_vla`에서 같은 스크립트를 사용할 수 있다. 필요하면 `REPO_ROOT` 또는 `ISAAC_GR00T_DIR`로 override한다.

- `gr00t/experiment/launch_finetune.py`, `finetune_config.py`, model setup 등 upstream 핵심 학습 코드는 수정하지 않는다.
- multi-path mixture 가 필요해서 mirror entry 를 따로 두었다 (`scripts/train/launch_finetune_ttt.py`). upstream 을 fork 하지 않고 mirror 한 줄 추가로 처리.
- 로컬에서 추가한 것은 RoboCasa PandaOmron modality config, mirror entry, train wrapper 다.
- custom modality config 는 `src/policies/Isaac-GR00T/examples` 가 아니라 `configs/policies` 아래에 둔다.

| 변수 | 기본값 |
| --- | --- |
| `BASE_MODEL_PATH` | `${REPO_ROOT}/checkpoints/nvidia/GR00T-N1.6-3B`가 있으면 사용, 없으면 `nvidia/GR00T-N1.6-3B` |
| `DATASET_PATH` | `${REPO_ROOT}/data/robocasa/v1.0/pretrain/atomic/<Task>/20250819/lerobot` 10개를 `:` join |
| `MODALITY_CONFIG_PATH` | `${REPO_ROOT}/configs/policies/groot_robocasa_panda_omron_config.py` |
| `OUTPUT_DIR` | `${REPO_ROOT}/outputs/groot_robocasa_baseline_10tasks` |
| `MAX_STEPS` | `20000` |
| `SAVE_STEPS` | `5000` |
| `SAVE_TOTAL_LIMIT` | `4` |
| `GLOBAL_BATCH_SIZE` | `64` |
| `DATALOADER_NUM_WORKERS` | `2` |
| `SHARD_SIZE` | `1024` |
| `NUM_SHARDS_PER_EPOCH` | `100000` |
| `EPISODE_SAMPLING_RATE` | `0.1` |
| `LEARNING_RATE` | `1e-4` |
| `WEIGHT_DECAY` | `1e-5` |
| `WARMUP_RATIO` | `0.05` |
| `TUNE_PROJECTOR` | `1` |
| `TUNE_DIFFUSION_MODEL` | `1` |

`groot_robocasa_target_finetune.sh`는 이 wrapper를 재사용하되 `DATASET_PATH`를 target 15-task 경로로, `OUTPUT_DIR`를 `${REPO_ROOT}/outputs/groot_robocasa_target_15tasks`로 override한다.

- `groot` container에는 `torchcodec`, `decord`, `ffmpeg`가 없고 `cv2`는 있다.
- upstream `video_utils.py` 는 수정하지 않는다.
- 장기 학습 전에는 container 에 `decord` (가벼움) 또는 `torchcodec` 를 설치해서 upstream video path 가 정상 동작하도록 맞추는 것이 좋다. mirror entry 에서는 `video_backend="decord"` 가 default.

## entry: `launch_finetune_ttt.py` (baseline mode)

파일:

```bash
scripts/train/launch_finetune_ttt.py
```

upstream `launch_finetune.py` 의 mirror. 두 가지 확장:

1. **multi-path**: `--dataset_path` 에 `:` 가 있으면 split 해서 task 별로 `dataset_paths=[p]` 로 mixture 구성 (line 112-117).
2. **TTT 인자**: `--ttt_predictor_path`, `--ttt_eagle_cache_root`, `--ttt_update_in_train`. 비우면 baseline GR00T finetune (line 54 의 명시: *“Empty `ttt_predictor_path` → baseline GR00T finetune (no TTT)”*).

따라서 dataset merge 없이 multi-path mixture 로 baseline fine-tuning 을 돌리는 정식 진입점은 이 entry 다.

## LLM tuning note

N1.6 코드 기준으로 `tune_llm=False`라도 model config/checkpoint의 `tune_top_llm_layers=4`가 적용될 수 있다. 이 값을 바꾸려면 upstream 코드 변경 또는 별도 실험 스크립트가 필요하므로 기본 wrapper에는 넣지 않는다.

## Smoke test

Host에서 smoke test:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=1 \
OUTPUT_DIR="${REPO_ROOT}/outputs/groot_robocasa_smoke" \
bash scripts/train/groot_robocasa_finetune.sh
```

Docker에서 smoke test:

```bash
docker compose build groot
docker compose up -d groot

docker compose exec groot bash -lc '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=1 \
OUTPUT_DIR=/temporal_vla/outputs/groot_robocasa_smoke \
bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
'
```

ckpt 1 개 ≈ 22GB (3B fp32 + AdamW state). `SAVE_STEPS=5000 × SAVE_TOTAL_LIMIT=4` 설정은 checkpoint 누적 사용량을 약 88GB로 제한하기 위한 보수적 default다.

`groot_robocasa_finetune.sh` 와 `groot_ttt_robocasa_finetune.sh` 는 mirror entry 와 hyperparameter 를 공유하지만, baseline / TTT 인자 분기를 분리하기 위해 별도 wrapper 로 둔다. 같은 entry 의 두 호출 형태 정도로 보면 된다.

### Hyperparameter 출처

Wrapper default는 하나의 RoboCasa 전용 공식 script에서 복사한 값이 아니라, 아래 출처를 조합한 것이다.


| 값                        | 현재 설정              | 출처                                                                     | 비고                                                                       |
| ------------------------ | ------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `LEARNING_RATE`          | `1e-4`             | `gr00t/configs/finetune_config.py`, GR00T paper Table 6                | 코드 default 와 논문 일치                                                       |
| `WEIGHT_DECAY`           | `1e-5`             | `gr00t/configs/finetune_config.py`, GR00T paper Table 6                | 코드 default 와 논문 일치                                                       |
| `WARMUP_RATIO`           | `0.05`             | `gr00t/configs/finetune_config.py`, GR00T paper Table 6                | 코드 default 와 논문 일치                                                       |
| optimizer                | `adamw_torch`      | `gr00t/experiment/launch_finetune.py`; paper 는 AdamW                   | wrapper 에서 직접 제어하지 않음                                                    |
| `TUNE_PROJECTOR`         | `1`                | `gr00t/configs/finetune_config.py`                                     | upstream default `True`                                                  |
| `TUNE_DIFFUSION_MODEL`   | `1`                | `gr00t/configs/finetune_config.py`                                     | upstream default `True`                                                  |
| `DATALOADER_NUM_WORKERS` | `2`                | `gr00t/configs/finetune_config.py`                                     | official examples 는 4 또는 6 도 사용                                          |
| `SHARD_SIZE`             | `1024`             | `gr00t/configs/finetune_config.py`, `examples/SO100/finetune_so100.sh` | 코드 default                                                               |
| `NUM_SHARDS_PER_EPOCH`   | `100000`           | `gr00t/configs/finetune_config.py`, `examples/SO100/finetune_so100.sh` | 코드 default                                                               |
| `EPISODE_SAMPLING_RATE`  | `0.1`              | `gr00t/configs/finetune_config.py`, `examples/SO100/finetune_so100.sh` | 코드 default                                                               |
| `MAX_STEPS`              | `20000`            | GR00T paper Table 6, official LIBERO/SimplerEnv scripts                | paper post-training range `20k-60k` 의 lower bound                        |
| `GLOBAL_BATCH_SIZE`      | `64`               | `gr00t/configs/finetune_config.py`                                     | code default. paper 는 `128 / 1024`, 서버 메모리에 따라 상향 조정 가능                  |
| `SAVE_STEPS`             | `5000`             | checkpoint disk cap                                                      | official example scripts 는 `1000` 이지만 ckpt 1 개 22GB 누적 위험으로 보수 운용         |
| `SAVE_TOTAL_LIMIT`       | `4`                | checkpoint disk cap                                                      | step_05000 ~ step_20000 4 개 보관, downstream eval 에 충분                     |
| `color_jitter_params`    | `0.3/0.4/0.5/0.08` | official fine-tuning example scripts                                   | `finetune_config.py` default 는 아님                                        |


현재 default 는 `공식 코드 default + 공식 예제 관례 + 운영 제약`의 조합이다. 코드베이스 default 만 따르려면 `SAVE_STEPS=1000`, paper post-training 에 가까이 가려면 `GLOBAL_BATCH_SIZE=128` 정도가 후보.

## Full run

Pretrain 10-task baseline fine-tune:

```bash
docker compose exec groot bash -lc '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
'
```

Target atomic-seen 15-task fine-tune:

```bash
docker compose exec groot bash -lc '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /temporal_vla/scripts/train/groot_robocasa_target_finetune.sh
'
```

다중 GPU:

```bash
NUM_GPUS=4 MASTER_PORT=29500 bash scripts/train/groot_robocasa_target_finetune.sh
```

## 메모리 관찰값

- Full DiT fine-tuning 기준 약 36GB VRAM 사용
- `MAX_STEPS=20000` 기준 약 2시간
- 16GB GPU에서는 full fine-tuning이 OOM

메모리 절감용으로 가능한 wrapper 옵션:

```bash
TUNE_DIFFUSION_MODEL=0 bash scripts/train/groot_robocasa_target_finetune.sh
```

이 경우 DiT full fine-tuning이 아니므로 최종 recipe 검증이 아니라 축소 실험으로만 본다.

## 학습 결과 연결

학습이 끝나면 output 아래에 checkpoint가 생긴다.

```text
outputs/groot_robocasa_target_15tasks/checkpoint-<step>/       # current target atomic-seen scope
outputs/groot_robocasa_baseline_10tasks/checkpoint-<step>/     # legacy pretrain 10-task baseline
```

평가용 profile을 따로 만들 때는 base profile을 복사한 뒤 checkpoint path만 바꾼다.

```bash
cp configs/checkpoints/groot__robocasa_panda_omron.yaml \
   configs/checkpoints/groot__robocasa_panda_omron_finetuned.yaml
```

예시:

```yaml
name: groot__robocasa_panda_omron_finetuned
checkpoint_source:
  type: local
  id: /temporal_vla/outputs/groot_robocasa_target_15tasks/checkpoint-20000
```

Docker 평가는 `docs/groot_robocasa_eval_setup.md`의 ZMQ workflow를 사용한다. 즉 `groot` container에서 `MODEL_PATH=/temporal_vla/outputs/groot_robocasa_target_15tasks/checkpoint-20000`로 server를 띄우고, `robocasa` container에서 `client-target15`를 실행한다. 기존 pretrain 10-task baseline을 평가할 때만 `client-train10`을 쓴다.

## 학습 후 Eval Quickstart

상세한 평가 조건, 로그 저장, SR 계산, observation key alias 설명은 `docs/groot_robocasa_eval_setup.md`를 본다. 여기서는 학습 직후 base와 fine-tuned checkpoint를 같은 target atomic-seen 15-task 조건으로 비교하는 최소 명령만 둔다.

최초 1회 준비:

```bash
docker exec -it groot bash /temporal_vla/scripts/eval/groot_robocasa.sh setup-server
docker exec -it robocasa bash /temporal_vla/scripts/eval/groot_robocasa.sh setup-client
```

Base checkpoint server:

```bash
docker exec -it groot bash -lc '
MODEL_PATH=/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B \
PORT=5556 \
bash /temporal_vla/scripts/eval/groot_robocasa.sh server
'
```

Base checkpoint client, target 15-task:

```bash
docker exec -it robocasa bash -lc '
POLICY_CLIENT_HOST=127.0.0.1 \
PORT=5556 \
EVAL_RUN_ID=base_target15_50ep_env1_$(date +%Y%m%d_%H%M%S) \
bash /temporal_vla/scripts/eval/groot_robocasa.sh client-target15 50 1 8 720
'
```

Fine-tuned checkpoint server:

```bash
docker exec -it groot bash -lc '
MODEL_PATH=/temporal_vla/outputs/groot_robocasa_target_15tasks/checkpoint-20000 \
PORT=5556 \
bash /temporal_vla/scripts/eval/groot_robocasa.sh server
'
```

Fine-tuned checkpoint client, target 15-task:

```bash
docker exec -it robocasa bash -lc '
POLICY_CLIENT_HOST=127.0.0.1 \
PORT=5556 \
EVAL_RUN_ID=ft20000_target15_50ep_env1_$(date +%Y%m%d_%H%M%S) \
bash /temporal_vla/scripts/eval/groot_robocasa.sh client-target15 50 1 8 720
'
```

메모:

- server 명령은 foreground로 떠 있으므로 base와 fine-tuned를 평가할 때 각각 server terminal을 하나씩 잡거나 기존 server를 종료한 뒤 다시 띄운다.
- `client-target15`는 target atomic-seen 15-task scope다. `client-train10`은 legacy pretrain 10-task baseline을 볼 때만 사용한다.
- 공식 비교값은 우선 `50 episodes`, `n_envs=1`, `n_action_steps=8`, `max_steps=720`으로 맞춘다.

## 트러블슈팅

| 증상 | 원인/해결 |
| --- | --- |
| `KeyError: 'robocasa_panda_omron'` | `--modality_config_path`가 빠졌거나 경로가 잘못되어 modality config가 import되지 않은 상태다. |
| `unrecognized arguments: --tune_top_llm_layers` | 현재 upstream `FinetuneConfig`에는 해당 CLI가 없다. 기본 wrapper에는 넣지 않는다. |
| `embodiment_id.json not found` | checkpoint path가 잘못됐다. `checkpoints/nvidia/GR00T-N1.6-3B/embodiment_id.json` 존재를 확인한다. |
| LeRobot dataset load 실패 | `meta/modality.json`, `meta/info.json`, `meta/episodes.jsonl` 등 필수 meta 파일 누락 여부를 확인한다. |
| 학습 중 OOM | 더 큰 GPU에서 실행하거나, smoke test는 `GLOBAL_BATCH_SIZE=1`, 축소 실험은 `TUNE_DIFFUSION_MODEL=0`을 사용한다. |
| step/sec가 너무 느림 | `DATALOADER_NUM_WORKERS`를 늘리거나 video backend 병목을 확인한다. |
| video backend 오류 | 장기 학습 전 Docker image 또는 uv 환경에 `torchcodec`/`decord`/`ffmpeg` 중 필요한 backend를 정리한다. |

## 참고

- `src/policies/Isaac-GR00T/getting_started/finetune_new_embodiment.md`
- `src/policies/Isaac-GR00T/getting_started/data_preparation.md`
- `src/policies/Isaac-GR00T/examples/finetune.sh`
- `src/policies/Isaac-GR00T/examples/robocasa/README.md`
- `src/policies/Isaac-GR00T/examples/robocasa-gr1-tabletop-tasks`
- `configs/checkpoints/groot__robocasa_panda_omron.yaml`
- `scripts/train/groot_robocasa_finetune.sh`
