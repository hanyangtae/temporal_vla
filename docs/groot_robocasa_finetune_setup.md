# GR00T N1.6 RoboCasa 10-task fine-tuning setup

Isaac-GR00T `n1.6-release` 기반으로 RoboCasa PandaOmron 10개 task를 하나의 LeRobot v2.1 dataset으로 병합하고, GR00T fine-tuning을 host 또는 Docker에서 실행하기 위한 runbook이다.

## 요약

- 학습 entrypoint는 upstream `gr00t/experiment/launch_finetune.py`를 그대로 사용한다.
- 로컬 wrapper `scripts/train/groot_robocasa_finetune.sh`는 repo 경로, checkpoint, dataset, modality config, 주요 hyperparameter를 정리해서 넘기는 역할만 한다.
- `launch_finetune.py`, `finetune_config.py`, model setup 등 upstream GR00T 핵심 학습 코드는 수정하지 않는다.
- 10개 RoboCasa task는 `--dataset-path` 하나로 넘기기 위해 `data/datasets/robocasa_10tasks_lerobot_v21` 아래에 병합했다.
- `ROBOCASA_PANDA_OMRON` enum 존재만으로는 충분하지 않다. 학습 시 `configs/policies/groot_robocasa_panda_omron_config.py`를 `--modality_config_path`로 import해서 modality config를 등록해야 한다.
- Full DiT fine-tuning은 16GB GPU에서 OOM이 난다. 서버 관찰값 기준 약 36GB VRAM을 사용했고, 20,000 step은 약 2시간 걸렸다.

## 필요한 파일

서버에서 학습을 돌리려면 최소한 아래가 있어야 한다.

```text
temporal_vla/
├── scripts/train/groot_robocasa_finetune.sh
├── configs/policies/groot_robocasa_panda_omron_config.py
├── checkpoints/nvidia/GR00T-N1.6-3B/
├── data/datasets/robocasa_10tasks_lerobot_v21/
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
  - checkpoint와 병합 dataset만 둘 경우 50GB 이상 권장
  - HF cache/raw task archive까지 다룰 경우 200GB 이상 권장
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

GR00T uv 환경:

```bash
cd ~/temporal_vla/src/policies/Isaac-GR00T

# uv가 없으면 먼저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# venv + dependency 설치
uv sync

# import 확인
uv run python -c "import gr00t; from gr00t.data.embodiment_tags import EmbodimentTag; print(EmbodimentTag.ROBOCASA_PANDA_OMRON)"
```

Modality config 등록 확인:

```bash
export REPO_ROOT=~/temporal_vla
cd "${REPO_ROOT}/src/policies/Isaac-GR00T"

uv run python -c "import os, sys; sys.path.append(os.path.expanduser(os.environ['REPO_ROOT']) + '/configs/policies'); import groot_robocasa_panda_omron_config; from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS; print('robocasa_panda_omron' in MODALITY_CONFIGS)"
```

`True`가 나와야 한다.

## Checkpoint 준비

기본 위치:

```text
checkpoints/nvidia/GR00T-N1.6-3B
```

서버에서 새로 다운로드:

```bash
cd ~/temporal_vla
mkdir -p checkpoints/nvidia

huggingface-cli download nvidia/GR00T-N1.6-3B \
  --local-dir checkpoints/nvidia/GR00T-N1.6-3B
```

기존 머신에서 옮기는 경우:

```bash
rsync -avhP \
  /path/to/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B/ \
  user@training-server:/home/dongkyu/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B/
```

확인:

```bash
ls checkpoints/nvidia/GR00T-N1.6-3B/config.json
ls checkpoints/nvidia/GR00T-N1.6-3B/embodiment_id.json
```

## Dataset 준비

대상 task:

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

원본은 RoboCasa atomic PandaOmron dataset이고, 현재 학습에는 병합된 LeRobot v2.1 dataset을 사용한다.

```text
data/datasets/robocasa_10tasks_lerobot_v21
```

검증된 병합 dataset 통계:

| 항목 | 값 |
| --- | --- |
| episodes | 1061 |
| frames | 253971 |
| unique language tasks | 99 |
| video files | 3183 |
| data chunks | 2 |
| disk usage | 733M |

필수 메타 파일:

```text
meta/embodiment.json
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/info.json
meta/modality.json
meta/relative_stats.json
meta/stats.json
meta/tasks.jsonl
```

`relative_stats.json`은 `{}`로 둔다. 현재 RoboCasa PandaOmron action config는 모두 `ABSOLUTE`라서 relative action statistics가 필요하지 않다.

서버로 zip을 옮겨 푸는 경우:

```bash
mkdir -p ~/temporal_vla/data/datasets
unzip ~/temporal_vla/data/datasets/robocasa_10tasks_lerobot_v21.zip \
  -d ~/temporal_vla/data/datasets
```

zip 내부가 `robocasa_10tasks_lerobot_v21/` 폴더를 포함하지 않고 `data/`, `meta/`, `videos/`를 바로 담고 있으면 다음처럼 푼다.

```bash
mkdir -p ~/temporal_vla/data/datasets/robocasa_10tasks_lerobot_v21
unzip ~/temporal_vla/data/datasets/robocasa_10tasks_lerobot_v21.zip \
  -d ~/temporal_vla/data/datasets/robocasa_10tasks_lerobot_v21
```

## Dataset merge script

파일:

```text
scripts/data/merge_robocasa_lerobot_v21.py
```

기능:

- 10개 task dataset을 하나의 LeRobot v2.1 dataset root로 merge한다.
- `episode_index`를 전역 index로 다시 쓴다.
- frame-level `index`를 전역 frame index로 다시 쓴다.
- `task_index`, `annotation.human.task_description`, `annotation.human.task_name`을 merged `tasks.jsonl` 기준으로 다시 매핑한다.
- `episodes.jsonl`, `episodes_stats.jsonl`, `info.json`, `stats.json`을 다시 생성한다.
- video 파일은 기본적으로 hardlink로 연결한다. 같은 filesystem이면 디스크를 중복 사용하지 않는다.

실행:

```bash
python scripts/data/merge_robocasa_lerobot_v21.py
```

예상 출력:

```text
Wrote merged dataset: data/datasets/robocasa_10tasks_lerobot_v21
  episodes: 1061
  frames: 253971
  tasks: 99
  chunks: 2
```

## Modality config

파일:

```text
configs/policies/groot_robocasa_panda_omron_config.py
```

이 파일은 upstream `Isaac-GR00T`를 직접 수정하지 않고 `ROBOCASA_PANDA_OMRON`에 대한 modality config를 등록하기 위해 둔다. `launch_finetune.py` 실행 시 `--modality_config_path`로 전달해야 한다.

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

Action config는 모두 `ActionRepresentation.ABSOLUTE`로 설정했다.

## Train wrapper

파일:

```text
scripts/train/groot_robocasa_finetune.sh
```

이 wrapper는 자기 자신의 위치에서 repo root를 계산한다. 그래서 host의 `~/temporal_vla`와 Docker mount의 `/temporal_vla`에서 같은 스크립트를 사용할 수 있다. 필요하면 `REPO_ROOT` 또는 `ISAAC_GR00T_DIR`로 override한다.

주요 기본값:

| 변수 | 기본값 |
| --- | --- |
| `BASE_MODEL_PATH` | `${REPO_ROOT}/checkpoints/nvidia/GR00T-N1.6-3B`가 있으면 사용, 없으면 `nvidia/GR00T-N1.6-3B` |
| `DATASET_PATH` | `${REPO_ROOT}/data/datasets/robocasa_10tasks_lerobot_v21` |
| `MODALITY_CONFIG_PATH` | `${REPO_ROOT}/configs/policies/groot_robocasa_panda_omron_config.py` |
| `OUTPUT_DIR` | `${REPO_ROOT}/outputs/groot_robocasa_10tasks_full` |
| `MAX_STEPS` | `20000` |
| `SAVE_STEPS` | `500` |
| `SAVE_TOTAL_LIMIT` | `2` |
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

Wrapper가 직접 제어하지 않는 값:

- optimizer: upstream `launch_finetune.py`의 `adamw_torch`
- VLLN: upstream model config/checkpoint default
- top LLM layers: upstream model config/checkpoint default

현재 N1.6 코드 기준으로 `tune_llm=False`라도 model config/checkpoint의 `tune_top_llm_layers=4`가 적용될 수 있다. 이 값을 바꾸려면 upstream 코드 변경 또는 별도 실험 스크립트가 필요하므로 기본 wrapper에는 넣지 않는다.

## 실행

Host에서 smoke test:

```bash
cd ~/temporal_vla

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=1 \
OUTPUT_DIR=~/temporal_vla/outputs/groot_robocasa_smoke \
bash scripts/train/groot_robocasa_finetune.sh
```

Docker에서 smoke test:

```bash
docker compose exec groot bash -lc '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=1 \
OUTPUT_DIR=/temporal_vla/outputs/groot_robocasa_smoke \
bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
'
```

Host에서 full run:

```bash
cd ~/temporal_vla

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/train/groot_robocasa_finetune.sh
```

Docker에서 full run:

```bash
docker compose exec groot bash -lc '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
'
```

다중 GPU:

```bash
NUM_GPUS=4 MASTER_PORT=29500 bash scripts/train/groot_robocasa_finetune.sh
```

## Hyperparameter 출처와 메모리

현재 wrapper의 값은 하나의 RoboCasa 전용 공식 script에서 복사한 것이 아니라, GR00T 코드 default, 공식 예제 관례, GR00T paper post-training 범위를 조합한 값이다.

| 값 | 현재 설정 | 출처/근거 |
| --- | --- | --- |
| `LEARNING_RATE` | `1e-4` | `gr00t/configs/finetune_config.py`, GR00T paper Table 6 |
| `WEIGHT_DECAY` | `1e-5` | `gr00t/configs/finetune_config.py`, GR00T paper Table 6 |
| `WARMUP_RATIO` | `0.05` | `gr00t/configs/finetune_config.py`, GR00T paper Table 6 |
| optimizer | `adamw_torch` | `gr00t/experiment/launch_finetune.py` |
| `TUNE_PROJECTOR` | `1` | upstream default `True` |
| `TUNE_DIFFUSION_MODEL` | `1` | upstream default `True` |
| `SAVE_TOTAL_LIMIT` | `2` | 서버 디스크 사용량을 줄이기 위한 wrapper default |
| `DATALOADER_NUM_WORKERS` | `2` | `gr00t/configs/finetune_config.py` |
| `SHARD_SIZE` | `1024` | `gr00t/configs/finetune_config.py` |
| `NUM_SHARDS_PER_EPOCH` | `100000` | `gr00t/configs/finetune_config.py` |
| `EPISODE_SAMPLING_RATE` | `0.1` | `gr00t/configs/finetune_config.py` |
| `MAX_STEPS` | `20000` | paper post-training range `20k-60k`의 lower bound |
| `GLOBAL_BATCH_SIZE` | `64` | 코드 default에 가까운 서버용 conservative setting |
| `SAVE_STEPS` | `500` | 중간 checkpoint 확인을 위한 wrapper setting |
| `color_jitter_params` | `0.3/0.4/0.5/0.08` | official fine-tuning example scripts 관례 |

관찰값:

- Full DiT fine-tuning 기준 약 36GB VRAM 사용
- `MAX_STEPS=20000` 기준 약 2시간
- 16GB GPU에서는 full fine-tuning이 OOM

메모리 절감용으로 가능한 wrapper 옵션:

```bash
TUNE_DIFFUSION_MODEL=0 bash scripts/train/groot_robocasa_finetune.sh
```

이 경우 DiT full fine-tuning이 아니므로 최종 recipe 검증이 아니라 축소 실험으로만 본다.

## 학습 결과 연결

학습이 끝나면 output 아래에 checkpoint가 생긴다.

```text
outputs/groot_robocasa_10tasks_full/checkpoint-<step>/
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
  id: /temporal_vla/outputs/groot_robocasa_10tasks_full/checkpoint-20000
```

Docker 평가 예시:

```bash
docker exec groot python /temporal_vla/scripts/serve/groot.py \
  --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron_finetuned.yaml

docker exec robocasa python /temporal_vla/scripts/eval/robocasa_eval.py \
  --task-set robocasa_eval_25 \
  --vla-server http://localhost:8500 \
  --use-groot-env \
  --num-rollouts 10 \
  --num-steps 720 \
  --output-dir outputs/eval/robocasa/groot/finetuned_<DATE>
```

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
