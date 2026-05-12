# GR00T N1.5 RoboCasa fine-tuning setup

Isaac-GR00T `n1.5-release` 기준으로 RoboCasa GR1 tabletop benchmark를 fine-tuning 하는 방법을 정리한다. 이 문서는 `src/policies/Isaac-GR00T-N1.5` submodule의 공식 문서를 기준으로 작성한다.

이 문서는 N1.6 PandaOmron 10-task fine-tuning 문서와 의도적으로 분리한다.

- N1.6 문서: `docs/groot_robocasa_finetune_setup.md`
- N1.6 코드 경로: `src/policies/Isaac-GR00T`
- N1.5 문서: `docs/groot_n1_5_robocasa_finetune_setup.md`
- N1.5 코드 경로: `src/policies/Isaac-GR00T-N1.5`

## 결론

- N1.5 RoboCasa 공식 recipe는 `examples/RoboCasa/README.md`에 있다.
- 학습 entrypoint는 `scripts/gr00t_finetune.py`다.
- 대상은 `ROBOCASA_PANDA_OMRON`이 아니라 `gr1` embodiment다.
- data config는 `fourier_gr1_arms_waist`를 사용한다.
- 공식 RoboCasa recipe는 24개 GR1 tabletop task의 `_1000` dataset을 multi-dataset 입력으로 넘긴다.
- 공식 reproduce command는 `8 GPU`, `batch-size 48`, `gradient-accumulation-steps 4`, `learning_rate 3e-5`, `max-steps 60000`, `tune-visual`이다.
- 공식 문서의 reported average success rate는 task당 50 rollouts 기준 `0.48`이다.
- `n_envs > 1` eval은 success rate를 낮출 수 있다고 공식 README가 경고한다.

## 공식 문서 위치

주요 근거 파일:

```text
src/policies/Isaac-GR00T-N1.5/README.md
src/policies/Isaac-GR00T-N1.5/examples/RoboCasa/README.md
src/policies/Isaac-GR00T-N1.5/scripts/gr00t_finetune.py
src/policies/Isaac-GR00T-N1.5/getting_started/4_deeper_understanding.md
src/policies/Isaac-GR00T-N1.5/gr00t/experiment/data_config.py
```

공식 RoboCasa README:

```text
src/policies/Isaac-GR00T-N1.5/examples/RoboCasa/README.md
```

이 파일이 N1.5 RoboCasa tabletop fine-tuning과 evaluation의 중심 문서다.

## N1.6 RoboCasa 세팅과 다른 점

N1.6 PandaOmron fine-tuning과 N1.5 GR1 tabletop fine-tuning은 같은 RoboCasa 이름을 쓰지만 학습 stack이 다르다.

| 항목 | N1.6 PandaOmron | N1.5 GR1 tabletop |
| --- | --- | --- |
| 코드 경로 | `src/policies/Isaac-GR00T` | `src/policies/Isaac-GR00T-N1.5` |
| 학습 entrypoint | `gr00t/experiment/launch_finetune.py` 계열 | `scripts/gr00t_finetune.py` |
| 대상 embodiment | `ROBOCASA_PANDA_OMRON` | `gr1` |
| data config | custom modality config 필요 | `fourier_gr1_arms_waist` |
| task scope | RoboCasa PandaOmron atomic 10 tasks | GR1 tabletop 24 tasks |
| modality config | repo의 `configs/policies/...`를 import | dataset/config에 사전 준비 |
| 목적 | 우리 N1.6 PandaOmron 실험 | N1.5 공식 RoboCasa tabletop recipe 재현 |

따라서 N1.5를 돌릴 때 `scripts/train/groot_robocasa_finetune.sh`를 그대로 쓰면 안 된다. N1.5는 `src/policies/Isaac-GR00T-N1.5` 아래에서 `scripts/gr00t_finetune.py`를 호출해야 한다.

## PandaOmron 10-task를 N1.5로 돌리는 경우

우리 RoboCasa PandaOmron dataset을 N1.5에서 돌릴 수는 있지만, 이 경우는 공식 GR1 RoboCasa recipe가 아니다. N1.5에는 `ROBOCASA_PANDA_OMRON` pretrained embodiment tag가 없으므로 `new_embodiment`로 학습한다.

사용하는 custom data config:

```text
configs/policies/robocasa_n15_panda_omron_data_config.py  →  RobocasaPandaOmron10TaskDataConfig
```

dataset schema:

```text
dataset (per-task, target × human × atomic, 15 task):
data/robocasa/v1.0/target/atomic/<Task>/<date>/lerobot

다운로드 스크립트:
scripts/utils/download_robocasa_target_human.sh
(box id 출처: src/benchmarks/robocasa/.../box_links_ds.json 의 target/atomic/<Task>/<date>/lerobot.tar)

15 task (RoboCasa atomic-seen 18개 중 CloseBlenderLid / NavigateKitchen / OpenStandMixerHead 3개 제외):
TurnOnElectricKettle, CloseToasterOvenDoor, OpenCabinet, SlideDishwasherRack,
PickPlaceToasterToCounter, TurnOnMicrowave, OpenDrawer, PickPlaceSinkToCounter,
PickPlaceCounterToStove, CloseFridge, TurnOnSinkFaucet, PickPlaceCounterToCabinet,
CoffeeSetupMug, PickPlaceDrawerToCounter, TurnOffStove

video:
video.robot0_agentview_left
video.robot0_agentview_right
video.robot0_eye_in_hand

state:
state.base_position
state.base_rotation
state.end_effector_position_relative
state.end_effector_rotation_relative
state.gripper_qpos

action:
action.base_motion
action.control_mode
action.end_effector_position
action.end_effector_rotation
action.gripper_close

language:
annotation.human.task_description
```

주의:

- 이 경로는 N1.6 `ROBOCASA_PANDA_OMRON` pretrained head를 쓰지 않는다.
- `new_embodiment` action head를 현재 PandaOmron 데이터로 학습하는 실험이다.
- 이 실험 결과와 N1.6 PandaOmron fine-tuning 결과는 같은 의미로 비교하면 안 된다.
- `gripper_close`, `control_mode`는 현재 dataset의 `-1/1` 범위를 보존하기 위해 N1.5 `binary`가 아니라 `min_max` normalization을 사용한다.
- target dataset 의 `meta/modality.json` / `meta/embodiment.json` 은 pretrain split과 identical 하므로 `RobocasaPandaOmron10TaskDataConfig` 를 그대로 재사용한다.
- task당 500~543 episode (info.json `total_episodes` 기준), 합계 ~7,620 episode / ~84k frame.

### 공통 환경변수

```bash
cd /home/junhyeong/pkt_ws/temporal_vla
export PYTHONPATH=$PWD/configs/policies
```

### Smoke 1 — data config import

```bash
conda run -n gr00t --no-capture-output python -c \
"from robocasa_n15_panda_omron_data_config import RobocasaPandaOmron10TaskDataConfig; \
c=RobocasaPandaOmron10TaskDataConfig(); print(c.action_keys)"
```

`action_keys: ['action.base_motion', 'action.control_mode', 'action.end_effector_position', 'action.end_effector_rotation', 'action.gripper_close']` 가 출력되면 OK.

### Smoke 2 — `load_dataset.py` (단일 task)

upstream `load_dataset.py` 는 `--data-config` 옵션이 없고 dataset `meta/modality.json` 만 본다.

```bash
conda run -n gr00t --no-capture-output python \
  src/policies/Isaac-GR00T-N1.5/scripts/load_dataset.py \
  --dataset-path $PWD/data/robocasa/v1.0/target/atomic/CloseFridge/20250816/lerobot \
  --embodiment-tag new_embodiment
```

### Smoke 3 — data config + transform (CPU only, GPU 사용 X)

`RobocasaPandaOmron10TaskDataConfig.transform()` 까지 실제 데이터에 적용되는지 검증. 모델 로드 안 함.

```bash
CUDA_VISIBLE_DEVICES="" conda run -n gr00t --no-capture-output python /tmp/groot_dataconfig_smoke.py
```

스크립트 본체는 `LeRobotSingleDataset` 인스턴스화 → `ds[0]` 1샘플 가져와 `state (1,64)`, `action (16,32)`, `eagle_content` 까지 shape 확인 (`embodiment_id=31`).

### Fine-tuning (target atomic-seen 15 task)

이 프로젝트 PandaOmron 학습 기본 설정. `scripts/gr00t_finetune.py` `ArgsConfig` dataclass default에서 batch / max-steps / save-steps 만 override 한다. dataset 은 위 다운로드 스크립트로 받은 target × atomic 15 task 를 glob 으로 전달 (task별 `<date>` 가 달라서 `*/*` 두 단계 와일드카드).

```bash
TS=$(date +%y%m%d%H%M%S)
OUT=$PWD/outputs/train/groot_n1_5/$TS

conda run -n gr00t --no-capture-output python \
  src/policies/Isaac-GR00T-N1.5/scripts/gr00t_finetune.py \
  --dataset-path $PWD/data/robocasa/v1.0/target/atomic/*/*/lerobot \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --output-dir $OUT \
  --batch-size 64 \
  --max-steps 50000 \
  --save-steps 10000
```

| 항목 | 값 | 비고 |
| --- | --- | --- |
| batch-size | **64** / GPU | override (dataclass default 32) |
| max-steps | **50000** | override (dataclass default 10000) |
| save-steps | **10000** | override (dataclass default 1000) |
| num-gpus | 1 | dataclass default 그대로 |
| gradient-accumulation-steps | 1 | dataclass default 그대로 |
| base-model | `nvidia/GR00T-N1.5-3B` | dataclass default 그대로 |
| tune-llm / tune-visual | False | dataclass default 그대로 |
| tune-projector / tune-diffusion-model | True | dataclass default 그대로 |
| lr / wd / warmup | 1e-4 / 1e-5 / 0.05 | dataclass default 그대로 |
| lora-rank | 0 (LoRA off) | dataclass default 그대로 |
| embodiment-tag | `new_embodiment` | dataclass default 그대로 |
| video-backend | `torchcodec` | dataclass default 그대로 |
| report-to | `wandb` | dataclass default 그대로 |
| dataloader-num-workers / prefetch | 12 / 4 | dataclass default 그대로 |

참고로 공식 RoboCasa GR1 recipe (`examples/RoboCasa/README.md`) 는 또 다른 값 사용 (batch 48 / steps 60000 / 8 GPU / lr 3e-5 / grad-accum 4 / `--tune-visual`).

## 환경 준비

공식 N1.5 README는 Python 3.10 conda 환경과 editable install을 기준으로 설명한다. 이 프로젝트도 동일하게 conda + pip 조합을 쓴다 (uv 미사용).

### 1. conda env 생성

```bash
cd /home/junhyeong/pkt_ws/temporal_vla/src/policies/Isaac-GR00T-N1.5

conda create -n gr00t python=3.10 -y
conda activate gr00t

pip install --upgrade setuptools
pip install -e .[base]
```

### 2. CUDA toolkit (nvcc) 준비 — flash-attn 빌드용

시스템에 `nvcc`가 없으면 conda env 내부로 toolkit을 가져온다. `cuda-toolkit=12.4` 메타패키지는 컴포넌트를 12.9로 끌어오지만 (`cuda-toolkit` 메타는 nvcc 마이너 버전을 핀하지 않음), CUDA 12.x 마이너 버전 호환성 덕분에 torch `+cu124` 빌드와 같이 동작한다.

```bash
conda install -n gr00t -c nvidia cuda-toolkit=12.4 -y
```

### 3. flash-attn 설치 (env-local CUDA_HOME 기준)

```bash
conda run -n gr00t --no-capture-output bash -c \
  'export CUDA_HOME=$CONDA_PREFIX && export PATH=$CUDA_HOME/bin:$PATH && \
   pip install --no-build-isolation flash-attn==2.7.1.post4'
```

### 4. 시스템 의존성 (관리자 권한 필요)

```bash
sudo apt-get install -y ffmpeg libsm6 libxext6
```

### 5. 설치 확인

```bash
conda run -n gr00t --no-capture-output python -c \
"import torch, flash_attn, gr00t; \
print('torch:', torch.__version__, 'cuda:', torch.version.cuda); \
print('flash_attn:', flash_attn.__version__); \
print('cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count()); \
print('gr00t:', gr00t.__file__)"
```

기대 출력:

```text
torch: 2.5.1+cu124 cuda: 12.4
flash_attn: 2.7.1.post4
cuda available: True devices: <N>
gr00t: .../src/policies/Isaac-GR00T-N1.5/gr00t/__init__.py
```

## Dataset

공식 RoboCasa README는 Hugging Face dataset `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`을 사용한다.

사용 subset:

```text
Humanoid robot tabletop manipulation - downsampled: 24k trajectories
```

공식 문서는 `_1000`으로 끝나는 dataset folder만 sparse checkout하라고 안내한다.

```bash
data_root=/tmp/robocasa_finetune_data

git clone --filter=blob:none --no-checkout \
  https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  "${data_root}"

cd "${data_root}"
git sparse-checkout init --cone
git sparse-checkout set "**/*_1000/"
```

학습 command에서는 `data_root`가 실제 dataset folder들의 상위 경로를 가리켜야 한다.

```bash
data_root=/tmp/robocasa_finetune_data
```

공식 README는 modality config가 dataset 안에 이미 준비되어 있다고 설명한다. N1.6 PandaOmron처럼 별도 `--modality_config_path`를 넘기는 구조가 아니다.

## 공식 RoboCasa task list

공식 N1.5 RoboCasa recipe는 아래 24개 `_1000` dataset을 multi-dataset으로 넣는다.

```bash
data_root=/tmp/robocasa_finetune_data

ALL_DATASET_PATHS=(
  "${data_root}/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "${data_root}/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
)
```

경로 검증:

```bash
for path in "${ALL_DATASET_PATHS[@]}"; do
  test -d "$path" || echo "missing: $path"
done
```

## Fine-tuning command

공식 RoboCasa README의 reproduce command:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T-N1.5

python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 8 \
  --batch-size 48 \
  --learning_rate 3e-5 \
  --output-dir /mnt/amlfs-02/shared/checkpoints/xiaoweij/0910/robocasa-checkpoints-60K/ \
  --data-config fourier_gr1_arms_waist \
  --embodiment_tag gr1 \
  --tune-visual \
  --max-steps 60000 \
  --save-steps 5000 \
  --gradient-accumulation-steps 4
```

우리 서버에서 output 경로는 별도로 잡는다.

```bash
OUTPUT_DIR=/home/dongkyu/pdk_ws/temporal_vla/outputs/groot_n1_5_robocasa_tabletop
```

예시:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T-N1.5

python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 8 \
  --batch-size 48 \
  --learning_rate 3e-5 \
  --output-dir /home/dongkyu/pdk_ws/temporal_vla/outputs/groot_n1_5_robocasa_tabletop \
  --data-config fourier_gr1_arms_waist \
  --embodiment_tag gr1 \
  --tune-visual \
  --max-steps 60000 \
  --save-steps 5000 \
  --gradient-accumulation-steps 4
```

Effective global batch size:

```text
8 GPUs * 48 per-GPU batch * 4 gradient accumulation = 1536
```

## Smoke test

전체 학습 전에 데이터 로딩과 forward/backward만 확인하려면 step을 줄인다.

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T-N1.5

python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 1 \
  --batch-size 1 \
  --learning_rate 3e-5 \
  --output-dir /tmp/groot_n1_5_robocasa_smoke \
  --data-config fourier_gr1_arms_waist \
  --embodiment_tag gr1 \
  --max-steps 2 \
  --save-steps 2 \
  --gradient-accumulation-steps 1
```

4090급 GPU에서 OOM이 나면 공식 README의 조언대로 DiT fine-tuning을 끈다.

```bash
python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 1 \
  --batch-size 1 \
  --learning_rate 3e-5 \
  --output-dir /tmp/groot_n1_5_robocasa_smoke \
  --data-config fourier_gr1_arms_waist \
  --embodiment_tag gr1 \
  --max-steps 2 \
  --save-steps 2 \
  --gradient-accumulation-steps 1 \
  --no-tune_diffusion_model
```

이 smoke test는 성능 검증이 아니다. dataset path, modality/data config, model loading, backward, checkpoint 저장이 가능한지만 확인하는 절차다.

## 중요한 fine-tuning config

`scripts/gr00t_finetune.py`의 `ArgsConfig` 기본값:

| config field | 기본값 | 의미 |
| --- | --- | --- |
| `base_model_path` | `nvidia/GR00T-N1.5-3B` | 시작 checkpoint |
| `batch_size` | `32` | GPU당 batch size |
| `max_steps` | `10000` | 총 optimizer step |
| `save_steps` | `1000` | checkpoint 저장 간격 |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `warmup_ratio` | `0.05` | cosine schedule warmup ratio |
| `tune_llm` | `False` | language model backbone tuning |
| `tune_visual` | `False` | vision tower tuning |
| `tune_projector` | `True` | action head projector tuning |
| `tune_diffusion_model` | `True` | action head DiT tuning |
| `lora_rank` | `0` | 0이면 LoRA 미사용 |
| `gradient_accumulation_steps` | `1` | gradient accumulation |
| `video_backend` | `torchcodec` | video decoding backend |
| `balance_dataset_weights` | `True` | multi-dataset weight balancing |
| `balance_trajectory_weights` | `True` | trajectory length 기반 sampling |

주의할 점:

- 공식 RoboCasa recipe는 script 기본값 `1e-4` 대신 `3e-5`를 쓴다.
- 공식 RoboCasa recipe는 script 기본값 `10000` steps 대신 `60000` steps를 쓴다.
- 공식 RoboCasa recipe는 기본 `tune_visual=False`를 `--tune-visual`로 켠다.
- `--dataset-path`는 `List[str]`이므로 여러 dataset path를 그대로 나열할 수 있다.
- dataset이 여러 개면 `LeRobotMixtureDataset`을 사용하고, 기본적으로 dataset/trajectory balancing이 켜진다.

## LoRA

N1.5 `scripts/gr00t_finetune.py`는 LoRA를 지원한다.

```bash
python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 2 \
  --batch-size 16 \
  --learning_rate 3e-5 \
  --output-dir /home/dongkyu/pdk_ws/temporal_vla/outputs/groot_n1_5_robocasa_lora \
  --data-config fourier_gr1_arms_waist \
  --embodiment_tag gr1 \
  --max-steps 20000 \
  --save-steps 5000 \
  --lora-rank 64 \
  --lora-alpha 128
```

공식 README는 LoRA fine-tuning이 가능하다고 설명하지만, 성능 관점에서는 full fine-tuning을 권장한다.

## Evaluation

공식 RoboCasa README는 먼저 inference server를 열고, RoboCasa simulation client를 실행하는 방식을 사용한다.

공식 fine-tuned checkpoint server:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T-N1.5

python3 scripts/inference_service.py --server \
  --model_path youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain \
  --data_config fourier_gr1_arms_waist
```

내가 학습한 checkpoint를 평가할 때:

```bash
python3 scripts/inference_service.py --server \
  --model_path /home/dongkyu/pdk_ws/temporal_vla/outputs/groot_n1_5_robocasa_tabletop/checkpoint-60000 \
  --data_config fourier_gr1_arms_waist
```

simulation client:

```bash
python3 scripts/simulation_service.py --client \
  --env_name <TASK_NAME> \
  --video_dir ./videos \
  --max_episode_steps 720 \
  --n_episodes 50
```

공식 README의 reported SR은 task당 50 rollouts 기준이다. `n_envs > 1`은 success rate를 낮출 수 있다는 경고가 있으므로, 공식 재현 기준으로는 `n_envs=1`을 우선 사용한다.

## 공식 reported performance

공식 N1.5 RoboCasa README는 fine-tuned model을 다음 Hugging Face repo에 올려두었다고 설명한다.

```text
youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain
```

reported average success rate:

```text
0.48
```

평가 조건:

```text
24 RoboCasa GR1 tabletop tasks
50 rollouts per task
```

따라서 N1.5 recipe 재현 여부를 볼 때는 단일 task 한두 개보다 24개 task 전체의 50-episode SR을 기준으로 비교하는 편이 맞다.

## 운영 메모

권장 실험 순서:

1. N1.5 환경 설치 확인
2. `_1000` dataset 24개 경로 확인
3. `max-steps 2`, `batch-size 1` smoke test
4. 공식 checkpoint `youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain` eval로 simulation 환경 확인
5. 공식 recipe와 같은 hyperparameter로 full fine-tuning
6. 동일 eval 조건, task당 50 episodes, `n_envs=1`로 SR 비교

N1.5 recipe를 N1.6 PandaOmron 실패 분석에 참고할 수는 있지만, 두 실험의 success rate를 직접 같은 의미로 비교하면 안 된다. embodiment, data config, task set, action space가 다르다.
