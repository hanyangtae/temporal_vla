# GR00T N1.6 RoboCasa 10-task fine-tuning setup

Isaac-GR00T `n1.6-release` 공식 코드베이스 기준으로 RoboCasa PandaOmron 10개 atomic task를 **per-task mixture** 로 fine-tuning 하기 위한 기록. dataset 은 합치지 않는다 (TTT 트랙과 데이터 경로/entry 를 일원화).

## 결론

- 공식 GR00T N1.6 fine-tuning 경로를 사용한다.
- 입력은 GR00T 가 기대하는 LeRobot v2.1 형식.
- upstream `gr00t/experiment/launch_finetune.py` 의 `--dataset_path` 는 단일 `str` 이라 multi-path mixture 를 그대로 지원하지 않는다. 이 repo 의 mirror entry **`scripts/train/launch_finetune_ttt.py`** 가 `:`-split 으로 multi-path 를 확장하고, TTT 인자를 비우면 baseline GR00T finetune 으로 동작한다 (해당 파일 line 54 명시).
- 10 task 를 merge 하지 않고 `data/robocasa/v1.0/pretrain/atomic/<Task>/20250819/lerobot` 10 개 경로를 그대로 mixture 로 넘긴다.
- 합쳐진 single LeRobot dataset 이 꼭 필요한 다른 사용처가 있을 때만 `scripts/data/merge_robocasa_lerobot_v21.py` 를 별도로 사용 (현재 fine-tuning 경로에서는 불사용).
- 16GB GPU 에서 full fine-tuning 은 OOM 가능성이 높다.
- 짧은 run 은 실제 성능 검증이 아니라, 데이터 로딩, modality mapping, forward/backward, checkpoint 저장이 되는지 확인하는 절차다.

참고한 공식 문서:

- [https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/getting_started/finetune_new_embodiment.md](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/getting_started/finetune_new_embodiment.md)
- [https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/examples/finetune.sh](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/examples/finetune.sh)
- [https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa)
- [https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa-gr1-tabletop-tasks](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa-gr1-tabletop-tasks)

## 대상 task

아래 10개 RoboCasa atomic task 를 mixture 로 학습한다.

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

## 데이터 origin

per-task LeRobot v2.1 dataset 을 다음 경로에 둔다 (TTT 트랙과 동일 경로).

```bash
data/robocasa/v1.0/pretrain/atomic/<Task>/20250819/lerobot/
```

다운로드:

```bash
bash scripts/utils/download_robocasa_pretrain_human.sh
# 또는 task 선택: bash ... OpenDrawer CloseDrawer
```

→ `data/robocasa/v1.0/pretrain/atomic/<Task>/<date>/lerobot/` 로 10 task 전체가 받아진다. UTexas Box 에서 tar 받아 압축 해제까지 한 번에.

각 task dataset 구조:

```text
data/chunk-000/episode_*.parquet
videos/chunk-000/observation.images.robot0_agentview_left/*.mp4
videos/chunk-000/observation.images.robot0_agentview_right/*.mp4
videos/chunk-000/observation.images.robot0_eye_in_hand/*.mp4
meta/{episodes.jsonl, episodes_stats.jsonl, info.json, modality.json, stats.json, tasks.jsonl}
```

`meta/modality.json` 의 video key 는 RoboCasa raw 카메라 이름 (`robot0_agentview_left/right`, `robot0_eye_in_hand`) 그대로이고, state/action key 는 PandaOmron 표준 (`base_position`, `base_rotation`, `end_effector_position_relative` 등) 이다. 즉 단일 modality config 가 10 task 에 그대로 적용된다.

(참고: TTT 트랙은 추가로 `scripts/extract/prepare_robocasa_dataset.py` 로 `progress` 컬럼을 in-place 로 붙인다. baseline finetune 은 `progress` 를 참조하지 않으므로 이 단계 없이도 학습 가능. 이미 TTT 용으로 한 번 돌려놓은 dataset 이라면 그대로 baseline 에 써도 무해.)

## modality config

파일:

```bash
configs/policies/groot_robocasa_panda_omron_config.py
```

`ROBOCASA_PANDA_OMRON` enum 은 upstream 에 등록되어 있지만 그 embodiment 의 modality config 자체는 등록되어 있지 않다. 학습 시 `--modality_config_path` 로 위 파일을 import 해야 `MODALITY_CONFIGS["robocasa_panda_omron"]` 이 채워진다. 안 넘기면 `KeyError: 'robocasa_panda_omron'`.

이 config 는 upstream `Isaac-GR00T` 코드를 직접 수정하지 않기 위해 repo 의 `configs/` 아래에 둔다.

등록 tag:

```python
EmbodimentTag.ROBOCASA_PANDA_OMRON
```

Video keys:

```python
robot0_agentview_left
robot0_agentview_right
robot0_eye_in_hand
```

State keys:

```python
end_effector_position_relative
end_effector_rotation_relative
gripper_qpos
base_position
base_rotation
```

Action keys:

```python
end_effector_position
end_effector_rotation
gripper_close
base_motion
control_mode
```

Language key:

```python
annotation.human.task_description
```

Action config 는 모두 `ActionRepresentation.ABSOLUTE` 로 설정했다. RoboCasa PandaOmron 의 action 이 모두 absolute 이기 때문이며, dataset 의 `meta/relative_stats.json` 도 `{}` 로 두면 된다.

## GR00T 코드베이스 유지 방침

핵심 방침:

- `gr00t/experiment/launch_finetune.py`, `finetune_config.py`, model setup 등 upstream 핵심 학습 코드는 수정하지 않는다.
- multi-path mixture 가 필요해서 mirror entry 를 따로 두었다 (`scripts/train/launch_finetune_ttt.py`). upstream 을 fork 하지 않고 mirror 한 줄 추가로 처리.
- 로컬에서 추가한 것은 RoboCasa PandaOmron modality config, mirror entry, train wrapper 다.
- custom modality config 는 `src/policies/Isaac-GR00T/examples` 가 아니라 `configs/policies` 아래에 둔다.

Video backend 관련 주의:

- 현재 `groot` container 에는 `torchcodec`, `decord`, `ffmpeg` 가 없고 `cv2` 는 있다.
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

## train wrapper

파일:

```bash
scripts/train/groot_robocasa_finetune.sh
```

baseline 전용 wrapper. mirror entry `scripts/train/launch_finetune_ttt.py` 를 TTT 인자 없이 호출한다. `DATASET_PATH` 가 10 atomic task 경로를 `:` 로 join 한 형태로 default 설정되어 있고, modality config 도 `configs/policies/groot_robocasa_panda_omron_config.py` 가 default. TTT 학습은 `scripts/train/groot_ttt_robocasa_finetune.sh` 별도.

기본값:

```bash
BASE_MODEL_PATH=${REPO_ROOT}/checkpoints/nvidia/GR00T-N1.6-3B   # 없으면 nvidia/GR00T-N1.6-3B (HF) fallback
ATOMIC_ROOT=${REPO_ROOT}/data/robocasa/v1.0/pretrain/atomic
DATE_TAG=20250819
DATASET_PATH=<10 atomic task 경로 ":" join, ATOMIC_ROOT/<Task>/<DATE_TAG>/lerobot>
MODALITY_CONFIG_PATH=${REPO_ROOT}/configs/policies/groot_robocasa_panda_omron_config.py
OUTPUT_DIR=${REPO_ROOT}/outputs/groot_robocasa_baseline_10tasks
ENTRY_SCRIPT=${REPO_ROOT}/scripts/train/launch_finetune_ttt.py
MAX_STEPS=20000
SAVE_STEPS=5000
SAVE_TOTAL_LIMIT=4
GLOBAL_BATCH_SIZE=64
DATALOADER_NUM_WORKERS=2
SHARD_SIZE=1024
NUM_SHARDS_PER_EPOCH=100000
EPISODE_SAMPLING_RATE=0.1
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05
USE_WANDB=1
TUNE_PROJECTOR=1
TUNE_DIFFUSION_MODEL=1
```

optimizer, VLLN, top LLM layer 제어는 upstream `launch_finetune.py` 기본값을 따른다.

ckpt 1 개 ≈ 22GB (3B fp32 + AdamW state). `SAVE_STEPS=5000 × SAVE_TOTAL_LIMIT=4 = 88GB` 디스크 사용을 가정. 이전에 `SAVE_STEPS=1000 × SAVE_TOTAL_LIMIT=20` (= 440GB) 로 돌리다가 step ~9000 에서 디스크 풀로 학습이 사망한 사례가 있어서 default 를 보수적으로 잡았다.

`groot_robocasa_finetune.sh` 와 `groot_ttt_robocasa_finetune.sh` 는 mirror entry 와 hyperparameter 를 공유하지만, baseline / TTT 인자 분기를 분리하기 위해 별도 wrapper 로 둔다. 같은 entry 의 두 호출 형태 정도로 보면 된다.

### Hyperparameter 출처

현재 wrapper 의 값은 하나의 RoboCasa 전용 공식 script 에서 복사한 것이 아니라, 아래 출처를 조합한 것이다.


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
| `SAVE_STEPS`             | `5000`             | 디스크 풀 사고 이후 보수 설정                                                      | official example scripts 는 `1000` 이지만 ckpt 1 개 22GB 누적 위험으로 보수 운용         |
| `SAVE_TOTAL_LIMIT`       | `4`                | 디스크 풀 사고 이후 보수 설정                                                      | step_05000 ~ step_20000 4 개 보관, downstream eval 에 충분                     |
| `color_jitter_params`    | `0.3/0.4/0.5/0.08` | official fine-tuning example scripts                                   | `finetune_config.py` default 는 아님                                        |


엄밀히 말하면 현재 default 는 `공식 코드 default + 공식 예제 관례 + 운영 사고 경험` 의 조합이다. 코드베이스 default 만 따르려면 `SAVE_STEPS=1000`, paper post-training 에 가까이 가려면 `GLOBAL_BATCH_SIZE=128` 정도가 후보.

## 실행

Full baseline fine-tune (10 atomic task per-task mixture):

```bash
docker compose exec groot bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
```

짧은 syntax/data-path 확인:

```bash
docker compose exec -T groot bash -lc \
  'MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=1 OUTPUT_DIR=/temporal_vla/outputs/groot_robocasa_baseline_check bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh'
```

주의: 16GB GPU 에서는 upstream 기본 optimizer/action-head 설정 때문에 짧은 실행도 OOM 이 날 수 있다. 공식 recipe 검증은 48GB 급 이상 GPU 에서 돌리는 쪽이 현실적이다.

## 현재 상태에서 가능한 것

가능:

- 10 task per-task LeRobot v2.1 을 mixture loader 로 읽기
- N1.6 base checkpoint 로딩
- PandaOmron modality config 적용
- `launch_finetune_ttt.py` baseline mode 경로로 fine-tuning 명령 구성

아직 아닌 것:

- 16GB GPU 에서 full GR00T N1.6 fine-tuning
- video backend dependency 정리 (`decord` 권장, container 에 미설치 시 mirror entry 가 강제하는 `video_backend="decord"` 가 실패)

## 실제 fine-tuning 으로 넘어갈 때

메모리가 충분한 GPU 에서는 기본값 그대로 실행한다.

```bash
docker compose exec groot bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
```

optimizer, VLLN, top LLM layer 는 upstream 기본값을 사용한다. 현재 코드베이스 기본값 기준으로 `tune_llm=False`, `tune_visual=False`, `tune_projector=True`, `tune_diffusion_model=True` 이며, model config 의 기본 top LLM/VLLN 설정도 적용된다.

현재 16GB GPU 에서 위 설정은 OOM 가능성이 높다.

현실적인 다음 선택지는 아래 중 하나다.

- 더 큰 GPU 에서 full fine-tuning 을 돌린다.
- `decord` 또는 `torchcodec` 를 container 에 설치해서 video decoding 병목을 줄인다.
- `bitsandbytes` 8-bit optimizer, LoRA, 더 작은 trainable subset 등 메모리 절감 방법을 추가로 검토한다.
- upstream 을 건드리지 않을 경우, 16GB 용 축소 실험은 별도 스크립트로 분리한다.
