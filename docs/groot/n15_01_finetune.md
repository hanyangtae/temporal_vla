# GR00T N1.5 RoboCasa fine-tuning setup

Isaac-GR00T `n1.5-release` 기준으로 RoboCasa GR1 tabletop benchmark를 fine-tuning 하는 방법을 정리한다. 이 문서는 `src/policies/Isaac-GR00T-N1.5` submodule의 공식 문서를 기준으로 작성한다.

이 문서는 N1.6 PandaOmron fine-tuning 문서와 의도적으로 분리한다.

- N1.6 문서: `n16_01_finetune.md`
- N1.6 코드 경로: `src/policies/Isaac-GR00T`
- N1.5 문서: `n15_01_finetune.md`
- N1.5 eval 문서: `n15_02_eval.md`
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
- PandaOmron dataset을 N1.5 `new_embodiment`로 학습/eval하는 경로도 정리한다. Task scope는 target atomic-seen 15 task다.

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


| 항목              | N1.6 PandaOmron                          | N1.5 GR1 tabletop                   |
| --------------- | ---------------------------------------- | ----------------------------------- |
| 코드 경로           | `src/policies/Isaac-GR00T`               | `src/policies/Isaac-GR00T-N1.5`     |
| 학습 entrypoint   | `gr00t/experiment/launch_finetune.py` 계열 | `scripts/gr00t_finetune.py`         |
| 대상 embodiment   | `ROBOCASA_PANDA_OMRON`                   | `gr1`                               |
| data config     | custom modality config 필요                | `fourier_gr1_arms_waist`            |
| task scope      | PandaOmron pretrain 10-task 또는 target atomic-seen 15-task | GR1 tabletop 24 tasks               |
| modality config | repo의 `configs/policies/...`를 import     | dataset/config에 사전 준비               |
| 목적              | N1.6 PandaOmron 실험                    | N1.5 공식 RoboCasa tabletop recipe 재현 |


따라서 N1.5를 돌릴 때 `scripts/train/groot_robocasa_finetune.sh`를 그대로 쓰면 안 된다. N1.5는 `src/policies/Isaac-GR00T-N1.5` 아래에서 `scripts/gr00t_finetune.py`를 호출해야 한다.

## PandaOmron target atomic-seen 15-task를 N1.5로 돌리는 경우

RoboCasa PandaOmron dataset을 N1.5에서 돌릴 수는 있지만, 이 경우는 공식 GR1 RoboCasa recipe가 아니다. N1.5에는 `ROBOCASA_PANDA_OMRON` pretrained embodiment tag가 없으므로 `new_embodiment`로 학습한다.

사용하는 custom data config:

```text
configs/policies/robocasa_n15_panda_omron_data_config.py  →  RobocasaPandaOmron10TaskDataConfig
```

`RobocasaPandaOmron10TaskDataConfig` 이름은 pretrain 10-task sanity check 때 만든 historical name이다. state/action/video/language schema가 같으므로 target atomic-seen 15-task 학습/eval에도 같은 config를 사용한다.

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
- `new_embodiment` action head를 PandaOmron 데이터로 학습하는 실험이다.
- 이 실험 결과와 N1.6 PandaOmron fine-tuning 결과는 같은 의미로 비교하면 안 된다.
- `gripper_close`, `control_mode`는 dataset의 `-1/1` 범위를 보존하기 위해 N1.5 `binary`가 아니라 `min_max` normalization을 사용한다.
- target dataset 의 `meta/modality.json` / `meta/embodiment.json` 은 pretrain split과 identical 하므로 `RobocasaPandaOmron10TaskDataConfig` 를 재사용한다.
- task당 500~543 episode, episode당 평균 ~252 frame (info.json `total_frames` / `total_episodes` 기준). 합계 **7,622 episode / 1,917,362 frame**.
- 학습 sample 수 = `max_steps × batch_size = 50000 × 64 = 3,200,000` → 데이터셋 1 epoch ≈ 1,917,362 frame 기준 **약 1.67 epoch**.

### 공통 환경변수

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
export HF_HOME="${REPO_ROOT}/data/huggingface"
export PYTHONPATH="${REPO_ROOT}/configs/policies:${PYTHONPATH:-}"
```

위 명령은 `temporal_vla` git checkout 내부에서 실행하는 것을 전제로 한다. repo 밖에서 실행할 때는 `REPO_ROOT=/path/to/temporal_vla`처럼 직접 지정한다.

N1.5 base model cache:

```text
data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
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

학습 시작 전:

1. repo root 에서 실행, `PYTHONPATH` 에 `configs/policies` 추가 (`--data-config <module>:<Class>` 형식이 외부 모듈을 import 함)
2. `nvidia-smi` 로 비어있는 GPU index 확인 → `CUDA_VISIBLE_DEVICES=<idx>` 로 핀 (`--num-gpus 1` default 그대로 두고 visible 만 1개로 제한)
3. wandb API key 는 `read -s` 로 1회 입력 → 같은 셸 안에서 export 유지

#### 1단계 — wandb key 1회 입력 (셸 1회만)

```bash
read -s WANDB_API_KEY; export WANDB_API_KEY
```

커서가 깜빡이면 키 paste + Enter. 같은 터미널 셸 동안 환경변수로 유지된다. 셸 닫으면 사라짐. (이 줄을 학습 명령 블록과 합치면 paste 시 `read -s` 가 다음 줄을 키 입력으로 먹어버리므로 분리.)

#### 2단계 — 학습 명령 (한 블록 paste 가능)

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
export HF_HOME="${REPO_ROOT}/data/huggingface"
export PYTHONPATH="${REPO_ROOT}/configs/policies:${PYTHONPATH:-}"
TS=$(date +%y%m%d%H%M%S)
OUT="${REPO_ROOT}/outputs/train/groot_n1_5/${TS}"
N15_BASE_MODEL="${REPO_ROOT}/data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"

CUDA_VISIBLE_DEVICES=2 \
WANDB_ENTITY=rnlgksclsrn9868-hanyang-university \
WANDB_PROJECT=finetune-gr00t-n1d5 \
WANDB_NAME=target15_$TS \
conda run -n gr00t --no-capture-output python \
  src/policies/Isaac-GR00T-N1.5/scripts/gr00t_finetune.py \
  --dataset-path "${REPO_ROOT}"/data/robocasa/v1.0/target/atomic/*/*/lerobot \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --base-model-path "$N15_BASE_MODEL" \
  --output-dir $OUT \
  --batch-size 64 \
  --max-steps 50000 \
  --save-steps 10000
```

메모:

- `CUDA_VISIBLE_DEVICES=3` 은 예시 — 실제 비어있는 GPU index 로 바꿔 사용. 프로세스는 device 0 으로 인식하지만 실제 사용 물리 GPU 는 그 index.
- `WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_PROJECT` / `WANDB_NAME` 은 HF Trainer + wandb integration 이 환경변수만 읽는다 (`gr00t_finetune.py` 자체엔 wandb CLI 인자 없음).
- entity/project 는 `rnlgksclsrn9868-hanyang-university` / `finetune-gr00t-n1d5` 고정.
- env 변수는 그 터미널 셸 process 한정 — 셸 닫으면 자동으로 사라짐.


| 항목                                    | 값                      | 비고                                 |
| ------------------------------------- | ---------------------- | ---------------------------------- |
| batch-size                            | **64** / GPU           | override (dataclass default 32)    |
| max-steps                             | **50000**              | override (dataclass default 10000) |
| save-steps                            | **10000**              | override (dataclass default 1000)  |
| num-gpus                              | 1                      | dataclass default 그대로              |
| gradient-accumulation-steps           | 1                      | dataclass default 그대로              |
| base-model                            | `nvidia/GR00T-N1.5-3B` | dataclass default 그대로              |
| tune-llm / tune-visual                | False                  | dataclass default 그대로              |
| tune-projector / tune-diffusion-model | True                   | dataclass default 그대로              |
| lr / wd / warmup                      | 1e-4 / 1e-5 / 0.05     | dataclass default 그대로              |
| lora-rank                             | 0 (LoRA off)           | dataclass default 그대로              |
| embodiment-tag                        | `new_embodiment`       | dataclass default 그대로              |
| video-backend                         | `torchcodec`           | dataclass default 그대로              |
| report-to                             | `wandb`                | dataclass default 그대로              |
| dataloader-num-workers / prefetch     | 12 / 4                 | dataclass default 그대로              |


참고로 공식 RoboCasa GR1 recipe (`examples/RoboCasa/README.md`) 는 또 다른 값 사용 (batch 48 / steps 60000 / 8 GPU / lr 3e-5 / grad-accum 4 / `--tune-visual`).

### Fine-tuning subset (각 task 앞 200 episode, ~5h sanity check)

전체 dataset (~1.9M frame, 7,622 episode) 으로 50k step 돌리면 ~25h 걸린다. sanity check 또는 빠른 iteration 용으로 task당 앞 200 episode 만 사용하고 step 도 줄이는 모드.

핵심:

- N1.5 `LeRobotSingleDataset` 는 episode subset 인자가 없다. submodule 수정 없이 처리하기 위해 wrapper script `scripts/train/gr00t_n15_finetune_subset.py` 가 `_get_trajectories` 를 monkey-patch 한다.
- `MAX_EPISODES_PER_DATASET=<N>` env var 로 상한 지정 (`0` 또는 unset 이면 patch 미적용 = upstream `gr00t_finetune.py` 와 동일 동작).
- 학습 시간은 dataset 크기가 아니라 `max_steps` 에 비례하므로 step 도 같이 줄여야 시간 단축이 의미 있음.

예상치 (task당 200 episode, `episodes.jsonl` 의 episode length 합계 기준):

| 항목 | 값 |
| --- | --- |
| 사용 episodes | 15 × 200 = **3,000** |
| 사용 frames (정확) | **754,530** (전체 1,917,362 의 **39.4%**) |
| **1 epoch (batch 64)** | **11,790 step** |

step → epoch / 시간 환산 (per-step ≈ 1.8 s 기준):

| `max-steps` | epoch | 시간 |
| --- | --- | --- |
| 10,000 | 0.85 | ~5 h |
| 11,790 | 1.00 | ~5.9 h |
| **23,580** | **2.00** | **~12 h** ← 이 프로젝트 기본 |

> `save-steps` 는 `11790` 으로 두면 1 epoch / 2 epoch 두 지점에서 정확히 checkpoint 가 떨어져 의미 단위가 깔끔 (총 2 ckpt, `save_total_limit=3` 한도 안). 더 자주 보고 싶으면 `7860` (≈ 0.67 epoch 간격, 3 ckpt) 등으로 줄이면 됨.

#### 명령 (2단계 paste 가능, 1단계 wandb key 는 위와 동일)

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
export HF_HOME="${REPO_ROOT}/data/huggingface"
export PYTHONPATH="${REPO_ROOT}/configs/policies:${PYTHONPATH:-}"
TS=$(date +%y%m%d%H%M%S)
OUT="${REPO_ROOT}/outputs/train/groot_n1_5/${TS}-subset200"
N15_BASE_MODEL="${REPO_ROOT}/data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"

CUDA_VISIBLE_DEVICES=2 \
MAX_EPISODES_PER_DATASET=200 \
WANDB_ENTITY=rnlgksclsrn9868-hanyang-university \
WANDB_PROJECT=finetune-gr00t-n1d5 \
WANDB_NAME=target15_subset200_$TS \
conda run -n gr00t --no-capture-output python \
  scripts/train/gr00t_n15_finetune_subset.py \
  --dataset-path "${REPO_ROOT}"/data/robocasa/v1.0/target/atomic/*/*/lerobot \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --base-model-path "$N15_BASE_MODEL" \
  --output-dir $OUT \
  --batch-size 64 \
  --max-steps 23580 \
  --save-steps 11790
```

메모:

- entrypoint 가 wrapper (`gr00t_n15_finetune_subset.py`) 로 바뀐 것 외에는 인자 동일. wrapper 가 `MAX_EPISODES_PER_DATASET` 처리 후 upstream `gr00t_finetune.py` 를 `runpy` 로 실행.
- `max-steps 23580` = 정확히 2 epoch, `save-steps 11790` = 1 epoch / 2 epoch 지점에서 checkpoint 2개 (`save_total_limit=3` 한도 안, 둘 다 보존).
- stdout 에 `[episode-subset] <task>: using 200/513 episodes ...` 형식 line 이 task 마다 한 줄씩 찍히면 patch 적용 확인.

## 환경 준비

공식 N1.5 README는 Python 3.10 conda 환경과 editable install을 기준으로 설명한다. 이 프로젝트도 동일하게 conda + pip 조합을 쓴다 (uv 미사용).

### 옵션 A — Docker 환경

서버에 conda 환경을 직접 만들지 않을 때는 compose service `groot_n15`를 쓴다. 기존 `groot` service는 N1.6 전용이므로 N1.5와 섞지 않는다.

```bash
cd "$(git rev-parse --show-toplevel)"

docker compose build groot_n15
docker compose up -d groot_n15
docker compose exec groot_n15 bash
```

service 정의:

```text
service:        groot_n15
container_name: groot_n15
image:          temporal-vla:groot-n15
Dockerfile:     src/policies/Isaac-GR00T-N1.5/Dockerfile
working_dir:    /temporal_vla
PYTHONPATH:     /temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/configs/policies:/temporal_vla
HF_HOME:        /temporal_vla/data/huggingface
```

일회성 실행으로 쓰고 싶으면:

```bash
docker compose run --rm groot_n15 bash
```

메모:

- `groot_n15`는 N1.5 submodule의 공식 Dockerfile을 build context로 쓴다.
- compose service는 repo root를 `/temporal_vla`로 mount하고, HF cache는 repository-local `/temporal_vla/data/huggingface`를 사용한다.
- `user: "${USER_ID}:${GROUP_ID}"`로 실행하므로 output file owner가 host 사용자로 남는다. 컨테이너 안에서 추가 `pip install`이 필요하면 별도 image를 다시 빌드하는 편이 낫다.
- GPU pinning이 필요하면 `docker-compose.override.yml`에서 `groot_n15`의 `device_ids`를 지정한다. 기본 service는 모든 GPU를 visible로 둔다.

Docker 안에서 설치 확인:

```bash
python -c "import torch, flash_attn, gr00t; print(torch.__version__, torch.version.cuda); print(flash_attn.__version__); print(gr00t.__file__)"
```

Docker smoke test:

```bash
export PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/configs/policies:/temporal_vla

CUDA_VISIBLE_DEVICES=0 python \
  src/policies/Isaac-GR00T-N1.5/scripts/gr00t_finetune.py \
  --dataset-path /temporal_vla/data/robocasa/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --embodiment_tag new_embodiment \
  --output-dir /temporal_vla/outputs/train/groot_n1_5/docker_smoke \
  --batch-size 1 \
  --max-steps 2 \
  --save-steps 2
```

공식 GR1 tabletop 24-task recipe도 같은 컨테이너에서 실행한다. 이 경우 `--data-config fourier_gr1_arms_waist`, `--embodiment_tag gr1`, `_1000` dataset 24개 path를 사용한다.

### 옵션 B-1 — conda env 생성

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}/src/policies/Isaac-GR00T-N1.5"

conda create -n gr00t python=3.10 -y
conda activate gr00t

pip install --upgrade setuptools
pip install -e .[base]
```

### 옵션 B-2 — CUDA toolkit (nvcc) 준비 — flash-attn 빌드용

시스템에 `nvcc`가 없으면 conda env 내부로 toolkit을 가져온다. `cuda-toolkit=12.4` 메타패키지는 컴포넌트를 12.9로 끌어오지만 (`cuda-toolkit` 메타는 nvcc 마이너 버전을 핀하지 않음), CUDA 12.x 마이너 버전 호환성 덕분에 torch `+cu124` 빌드와 같이 동작한다.

```bash
conda install -n gr00t -c nvidia cuda-toolkit=12.4 -y
```

### 옵션 B-3 — flash-attn 설치 (env-local CUDA_HOME 기준)

```bash
conda run -n gr00t --no-capture-output bash -c \
  'export CUDA_HOME=$CONDA_PREFIX && export PATH=$CUDA_HOME/bin:$PATH && \
   pip install --no-build-isolation flash-attn==2.7.1.post4'
```

### 옵션 B-4 — 시스템 의존성 (관리자 권한 필요)

```bash
sudo apt-get install -y ffmpeg libsm6 libxext6
```

### 옵션 B-5 — 설치 확인

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
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}/src/policies/Isaac-GR00T-N1.5"

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

Output 경로는 repository 아래로 잡는다.

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
OUTPUT_DIR="${REPO_ROOT}/outputs/groot_n1_5_robocasa_tabletop"
```

예시:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}/src/policies/Isaac-GR00T-N1.5"

python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 8 \
  --batch-size 48 \
  --learning_rate 3e-5 \
  --output-dir "${REPO_ROOT}/outputs/groot_n1_5_robocasa_tabletop" \
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
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}/src/policies/Isaac-GR00T-N1.5"

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
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}/src/policies/Isaac-GR00T-N1.5"

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


| config field                  | 기본값                    | 의미                             |
| ----------------------------- | ---------------------- | ------------------------------ |
| `base_model_path`             | `nvidia/GR00T-N1.5-3B` | 시작 checkpoint                  |
| `batch_size`                  | `32`                   | GPU당 batch size                |
| `max_steps`                   | `10000`                | 총 optimizer step               |
| `save_steps`                  | `1000`                 | checkpoint 저장 간격               |
| `learning_rate`               | `1e-4`                 | AdamW learning rate            |
| `weight_decay`                | `1e-5`                 | AdamW weight decay             |
| `warmup_ratio`                | `0.05`                 | cosine schedule warmup ratio   |
| `tune_llm`                    | `False`                | language model backbone tuning |
| `tune_visual`                 | `False`                | vision tower tuning            |
| `tune_projector`              | `True`                 | action head projector tuning   |
| `tune_diffusion_model`        | `True`                 | action head DiT tuning         |
| `lora_rank`                   | `0`                    | 0이면 LoRA 미사용                   |
| `gradient_accumulation_steps` | `1`                    | gradient accumulation          |
| `video_backend`               | `torchcodec`           | video decoding backend         |
| `balance_dataset_weights`     | `True`                 | multi-dataset weight balancing |
| `balance_trajectory_weights`  | `True`                 | trajectory length 기반 sampling  |


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
  --output-dir "${REPO_ROOT}/outputs/groot_n1_5_robocasa_lora" \
  --data-config fourier_gr1_arms_waist \
  --embodiment_tag gr1 \
  --max-steps 20000 \
  --save-steps 5000 \
  --lora-rank 64 \
  --lora-alpha 128
```

공식 README는 LoRA fine-tuning이 가능하다고 설명하지만, 성능 관점에서는 full fine-tuning을 권장한다.

## Evaluation Quickstart

N1.5 eval은 N1.6 eval과 별도 경로를 사용한다. N1.6의 `n16_02_eval.md`는 `groot`/`robocasa` ZMQ workflow 기준이고, N1.5는 `groot_n15` service와 `src/policies/Isaac-GR00T-N1.5/scripts` 아래 entrypoint를 사용한다.

상세 명령은 `n15_02_eval.md`를 기준으로 한다. 해당 문서에서 single server-client eval(`--mode tuned` / `--mode base`)과 base/tuned pair comparison(`--mode both`)을 분리해서 안내한다.

권장 확인 순서:

1. `docker compose build groot_n15`
2. PandaOmron target 15-task는 `new_embodiment` + `RobocasaPandaOmron10TaskDataConfig`로 offline dataset MSE부터 확인
3. official GR1 tabletop 재현은 `gr1` + `fourier_gr1_arms_waist` + `inference_service.py`/`simulation_service.py`로 확인

PandaOmron fine-tuned checkpoint의 첫 offline MSE 확인:

```bash
docker compose exec groot_n15 bash -lc '
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/eval_policy.py \
  --model-path /temporal_vla/outputs/train/groot_n1_5/<RUN>/checkpoint-<STEP> \
  --dataset-path /temporal_vla/data/robocasa/v1.0/target/atomic/OpenCabinet/20250813/lerobot \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --embodiment-tag new_embodiment \
  --modality-keys base_motion control_mode end_effector_position end_effector_rotation gripper_close \
  --steps 150 \
  --trajs 3
'
```

## 운영 메모

권장 실험 순서:

1. N1.5 환경 설치 확인
2. `_1000` dataset 24개 경로 확인
3. `max-steps 2`, `batch-size 1` smoke test
4. 공식 checkpoint eval로 simulation 환경 확인
5. 공식 recipe와 같은 hyperparameter로 full fine-tuning
6. `n15_02_eval.md` 기준으로 SR 비교

공식 reported checkpoint, SR, 평가 조건은 `n15_02_eval.md`의 `Reported Reference`를 기준으로 한다. N1.5 recipe를 N1.6 PandaOmron 실패 분석에 참고할 수는 있지만, 두 실험의 success rate를 직접 같은 의미로 비교하면 안 된다. embodiment, data config, task set, action space가 다르다.
