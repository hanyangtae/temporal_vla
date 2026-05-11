# GR00T N1.6 RoboCasa 10-task fine-tuning setup

Isaac-GR00T `n1.6-release` 공식 코드베이스 기준으로 RoboCasa PandaOmron 10개 task를 하나의 LeRobot v2.1 dataset으로 합치고, GR00T fine-tuning entrypoint까지 준비한 기록.

## 결론

- 공식 GR00T N1.6 fine-tuning 경로를 사용한다.
- 입력 dataset은 GR00T가 기대하는 LeRobot v2.1 형식이어야 한다.
- 공식 `launch_finetune.py` CLI는 `--dataset-path` 하나만 받으므로, 10개 task는 하나의 dataset root로 merge했다.
- upstream `launch_finetune.py`는 유지한다. 로컬 wrapper는 공식 entrypoint에 dataset/model/config 인자를 넘기는 역할만 한다.
- 현재 16GB GPU에서는 full fine-tuning이 OOM 가능성이 높다.
- 짧은 run은 실제 성능 검증이 아니라, 데이터 로딩, modality mapping, forward/backward, checkpoint 저장이 되는지 확인하는 절차다.

참고한 공식 문서:

- [https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/getting_started/finetune_new_embodiment.md](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/getting_started/finetune_new_embodiment.md)
- [https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/examples/finetune.sh](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/examples/finetune.sh)
- [https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa)
- [https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa-gr1-tabletop-tasks](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release/examples/robocasa-gr1-tabletop-tasks)

## 대상 task

아래 10개 RoboCasa atomic task를 하나의 dataset으로 병합했다.

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

원본 위치:

```bash
src/benchmarks/robocasa/datasets/v1.0/pretrain/atomic/<Task>/20250819/lerobot
```

위 경로의 각 task dataset은 이미 LeRobot v3.0에서 v2.1로 변환한 결과를 사용했다.

## 생성한 merged dataset

출력 위치:

```bash
data/datasets/robocasa_10tasks_lerobot_v21
```

검증된 크기:


| 항목                    | 값      |
| --------------------- | ------ |
| episodes              | 1061   |
| frames                | 253971 |
| unique language tasks | 99     |
| video files           | 3183   |
| data chunks           | 2      |
| disk usage            | 733M   |


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

`relative_stats.json`은 `{}`로 둔다. RoboCasa PandaOmron N1.6 checkpoint의 action config가 모두 `ABSOLUTE`라서 relative action statistics가 필요하지 않다.

## 추가한 merge script

파일:

```bash
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

출력:

```text
Wrote merged dataset: data/datasets/robocasa_10tasks_lerobot_v21
  episodes: 1061
  frames: 253971
  tasks: 99
  chunks: 2
```

## 추가한 modality config

파일:

```bash
configs/policies/groot_robocasa_panda_omron_config.py
```

이 config는 upstream `Isaac-GR00T` 코드를 직접 수정하지 않기 위해 repo의 `configs/` 아래에 둔다. `launch_finetune.py`에는 `--modality_config_path`로 전달한다.

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

Action config는 모두 `ActionRepresentation.ABSOLUTE`로 설정했다.

## GR00T 코드베이스 유지 방침

핵심 방침:

- `launch_finetune.py`, `finetune_config.py`, model setup 등 upstream 핵심 학습 코드는 수정하지 않는다.
- 학습 진입은 upstream `gr00t/experiment/launch_finetune.py`를 그대로 사용한다.
- 로컬에서 추가한 것은 dataset merge script, RoboCasa PandaOmron modality config, train wrapper다.
- custom modality config는 `src/policies/Isaac-GR00T/examples`가 아니라 `configs/policies` 아래에 둔다.

Video backend 관련 주의:

- 현재 `groot` container에는 `torchcodec`, `decord`, `ffmpeg`가 없고 `cv2`는 있다.
- upstream `video_utils.py`는 수정하지 않는 방향이다.
- 장기 학습 전에는 Docker image에 `torchcodec` 또는 `decord`를 설치해서 upstream video path가 정상 동작하도록 맞추는 것이 좋다.
- 16GB GPU에서 짧은 확인을 위해 `optim`, `tune_top_llm_layers`, `tune_vlln` 같은 비공식 제어 옵션을 추가할 수는 있지만, 기본 fine-tuning 경로에는 섞지 않는다.

## 추가한 train wrapper

파일:

```bash
scripts/train/groot_robocasa_finetune.sh
```

기본값:

```bash
BASE_MODEL_PATH=/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B
DATASET_PATH=/temporal_vla/data/datasets/robocasa_10tasks_lerobot_v21
MODALITY_CONFIG_PATH=/temporal_vla/configs/policies/groot_robocasa_panda_omron_config.py
OUTPUT_DIR=/temporal_vla/outputs/groot_robocasa_10tasks_full
MAX_STEPS=20000
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=5
GLOBAL_BATCH_SIZE=128
DATALOADER_NUM_WORKERS=2
SHARD_SIZE=1024
NUM_SHARDS_PER_EPOCH=100000
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05
TUNE_PROJECTOR=1
TUNE_DIFFUSION_MODEL=1
```

optimizer, VLLN, top LLM layer 제어는 upstream `launch_finetune.py` 기본값을 따른다.

### Hyperparameter 출처

현재 wrapper의 값은 하나의 RoboCasa 전용 공식 script에서 복사한 것이 아니라, 아래 출처를 조합한 것이다.


| 값                        | 현재 설정              | 출처                                                                     | 비고                                                                       |
| ------------------------ | ------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `LEARNING_RATE`          | `1e-4`             | `gr00t/configs/finetune_config.py`, GR00T paper Table 6                | 코드 default와 논문 일치                                                        |
| `WEIGHT_DECAY`           | `1e-5`             | `gr00t/configs/finetune_config.py`, GR00T paper Table 6                | 코드 default와 논문 일치                                                        |
| `WARMUP_RATIO`           | `0.05`             | `gr00t/configs/finetune_config.py`, GR00T paper Table 6                | 코드 default와 논문 일치                                                        |
| optimizer                | `adamw_torch`      | `gr00t/experiment/launch_finetune.py`; paper는 AdamW                    | wrapper에서 직접 제어하지 않음                                                     |
| `TUNE_PROJECTOR`         | `1`                | `gr00t/configs/finetune_config.py`                                     | upstream default `True`                                                  |
| `TUNE_DIFFUSION_MODEL`   | `1`                | `gr00t/configs/finetune_config.py`                                     | upstream default `True`                                                  |
| `SAVE_TOTAL_LIMIT`       | `5`                | `gr00t/configs/finetune_config.py`, official example scripts           | 코드 default                                                               |
| `DATALOADER_NUM_WORKERS` | `2`                | `gr00t/configs/finetune_config.py`                                     | official examples는 4 또는 6도 사용                                            |
| `SHARD_SIZE`             | `1024`             | `gr00t/configs/finetune_config.py`, `examples/SO100/finetune_so100.sh` | 코드 default                                                               |
| `NUM_SHARDS_PER_EPOCH`   | `100000`           | `gr00t/configs/finetune_config.py`, `examples/SO100/finetune_so100.sh` | 코드 default                                                               |
| `EPISODE_SAMPLING_RATE`  | `0.1`              | `gr00t/configs/finetune_config.py`, `examples/SO100/finetune_so100.sh` | 코드 default                                                               |
| `MAX_STEPS`              | `20000`            | GR00T paper Table 6, official LIBERO/SimplerEnv scripts                | paper post-training range `20k-60k`의 lower bound                         |
| `GLOBAL_BATCH_SIZE`      | `128`              | GR00T paper Table 6                                                    | paper post-training value `128 or 1024` 중 memory-conservative choice     |
| `SAVE_STEPS`             | `500`              | GR00T paper simulation eval protocol                                   | official example scripts는 보통 `1000`; strict code-example 기준이면 `1000`도 가능 |
| `color_jitter_params`    | `0.3/0.4/0.5/0.08` | official fine-tuning example scripts                                   | `finetune_config.py` default는 아님                                         |


엄밀히 말하면 현재 default는 `공식 코드 default + 공식 예제 관례 + 논문 post-training lower-bound`이다. “코드베이스 default만” 따르려면 `MAX_STEPS=10000`, `SAVE_STEPS=1000`, `GLOBAL_BATCH_SIZE=64`가 더 가깝다. 반대로 paper post-training 설정에 더 맞추려면 현재처럼 `MAX_STEPS=20000`, `GLOBAL_BATCH_SIZE=128`을 lower-bound로 쓰는 것이 합리적이다.

Fine-tuning 실행:

```bash
docker compose exec groot bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
```

짧은 syntax/data-path 확인:

```bash
docker compose exec -T groot bash -lc \
  'MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=1 OUTPUT_DIR=/temporal_vla/outputs/groot_robocasa_10tasks_check bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh'
```

주의: 16GB GPU에서는 upstream 기본 optimizer/action-head 설정 때문에 짧은 실행도 OOM이 날 수 있다. 공식 recipe 검증은 48GB급 이상 GPU에서 돌리는 쪽이 현실적이다.

## 현재 상태에서 가능한 것

가능:

- merged 10-task dataset을 GR00T loader로 읽기
- N1.6 base checkpoint 로딩
- PandaOmron modality config 적용
- upstream `launch_finetune.py` 경로로 fine-tuning 명령 구성

아직 아닌 것:

- 16GB GPU에서 full GR00T N1.6 fine-tuning
- video backend dependency 정리 (`torchcodec` 또는 `decord` 설치)

## 실제 fine-tuning으로 넘어갈 때

메모리가 충분한 GPU에서는 기본값 그대로 실행한다.

```bash
docker compose exec groot bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
```

optimizer, VLLN, top LLM layer는 upstream 기본값을 사용한다. 현재 코드베이스 기본값 기준으로 `tune_llm=False`, `tune_visual=False`, `tune_projector=True`, `tune_diffusion_model=True`이며, model config의 기본 top LLM/VLLN 설정도 적용된다.

현재 16GB GPU에서 위 설정은 OOM 가능성이 높다.

현실적인 다음 선택지는 아래 중 하나다.

- 더 큰 GPU에서 full fine-tuning을 돌린다.
- `torchcodec` 또는 `decord`를 container에 설치해서 video decoding 병목을 줄인다.
- `bitsandbytes` 8-bit optimizer, LoRA, 더 작은 trainable subset 등 메모리 절감 방법을 추가로 검토한다.
- upstream을 건드리지 않을 경우, 16GB용 축소 실험은 별도 스크립트로 분리한다.

