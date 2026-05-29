# GR00T × RoboCasa Finetune 학습 서버 셋업

다른 서버에서 GR00T-N1.6-3B 를 RoboCasa atomic task 데이터로 finetune 하기 위한 준비 절차.
현재 시점 기준 코드/데이터 상태를 그대로 옮긴다는 가정.

> **TLDR**
> 1. repo clone + submodule init + Isaac-GR00T uv 환경 셋업
> 2. HF cache 의 `nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos` blob 들 (이미 ~61GB 받아져 있음) 을 그대로 옮겨서 disk 공간 절약
> 3. 우리가 고른 10개 atomic task 만 tar 추출 → GR00T LeRobot v2 디렉토리 트리로 정리
> 4. `gr00t/experiment/launch_finetune.py` 호출 (`embodiment_tag=ROBOCASA_PANDA_OMRON`)

## 사전 가정

- Linux + NVIDIA GPU (A100 / H100 / RTX6000Ada 등 24GB+ VRAM) + CUDA 12.x
- Docker 또는 직접 Python 환경 (Isaac-GR00T 가 `uv` 사용 권장)
- 디스크: 최소 200GB 여유 (체크포인트 ~7GB + 데이터 ~70GB + 학습 출력 + 스왑)

## 학습 대상 task (10개)

모두 RoboCasa atomic dataset 의 PandaOmron embodiment 데이터.

| # | task | episodes (cache) |
|---|---|---|
| 1 | OpenDrawer | 519 |
| 2 | CloseDrawer | 110 |
| 3 | OpenCabinet | 1262 |
| 4 | CloseCabinet | 1267 |
| 5 | OpenFridge | 105 |
| 6 | CloseFridge | 731 |
| 7 | OpenMicrowave | 105 |
| 8 | CloseMicrowave | 634 |
| 9 | PickPlaceCounterToStove | 307 |
| 10 | PickPlaceCounterToSink | 108 |

총 ~5,148 episodes. base 가 거의 정지하는 task 위주로 선정 (PnP 도 인접 fixture 만).

선정 근거: `examples/robocasa/README.md` baseline + 우리 `outputs/eval/robocasa/groot/` 검증 결과.

---

## 1. Repository + submodule

```bash
# 1) repo clone
git clone https://github.com/hanyangtae/temporal_vla.git
cd temporal_vla

# 2) 학습 브랜치 체크아웃 (또는 dev 머지 후 dev)
git checkout refactor/groot-helpers-for-finetune  # 또는 dev

# 3) submodule init (Isaac-GR00T, robocasa, robosuite, ... 모두 받아짐)
git submodule update --init --recursive
```

submodule 중 학습에 필요한 것:
- `src/policies/Isaac-GR00T` — GR00T 학습/추론 코드
- `src/benchmarks/robocasa` (`hanyangtae/robocasa` fork, `temporal_vla` 브랜치) — 평가용. 학습엔 직접 의존 X 지만 모달리티 정합 검증할 때 필요.

## 2. Isaac-GR00T 환경 (uv)

GR00T 는 `uv` 기반 dependency 관리. 직접 `pip` 로 깔지 않기를 권장.

```bash
cd /path/to/temporal_vla/src/policies/Isaac-GR00T

# uv 미설치 시 (Astral 공식 installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# venv + dependency 설치 (몇 분 걸림)
uv sync

# 활성화
source .venv/bin/activate

# 검증
uv run python -c "import gr00t; from gr00t.data.embodiment_tags import EmbodimentTag; print(EmbodimentTag.ROBOCASA_PANDA_OMRON)"
```

> 만약 Docker 컨테이너로 가는 쪽이면 `docker/groot/Dockerfile` 그대로 쓰고 컨테이너 안에서 위 절차 진행.

## 3. 체크포인트 (GR00T-N1.6-3B base) 준비

> Legacy note: 이 문서는 초기 fine-tuning runbook이다. 현재 N1.6 RoboCasa fine-tuning 기준 문서는 `docs/groot/n16_01_finetune.md`다. 체크포인트 위치는 현재 repo 규칙에 맞춰 `outputs/checkpoints/GR00T-N1.6-3B`를 사용한다.

```bash
# Hugging Face 토큰 (gated repo 아니지만 다운로드 속도 ↑)
export HF_TOKEN=<token>

# cache 루트에 <cache>/checkpoints/nvidia/ 디렉토리
mkdir -p <cache>/checkpoints/nvidia
huggingface-cli download nvidia/GR00T-N1.6-3B \
    --local-dir <cache>/checkpoints/nvidia/GR00T-N1.6-3B
```

용량: ~7 GB. 우리 프로파일 `configs/checkpoints/groot__robocasa_panda_omron.yaml` 의 `checkpoint_source.id` 가 이 경로를 가리킴.

## 4. 학습 데이터 (Kitchen-Sim-Demos) 준비

NVIDIA 가 RoboCasa atomic + composite task 의 GR00T LeRobot v2 데이터를 HF 에 공개.

### 4.1 HF cache 가져오기 (옵션 A — 기존 환경 그대로 옮기기)

현재 머신의 cache 가 약 61GB blobs 받아진 상태. **이걸 통째로 rsync 하는 게 가장 빠름**.

```bash
# (현재 머신에서) 학습 서버로 복사
rsync -avhP \
    <cache>/datasets/huggingface/hub/datasets--nvidia--PhysicalAI-Robotics-Kitchen-Sim-Demos/ \
    user@training-server:<cache>/datasets/huggingface/hub/datasets--nvidia--PhysicalAI-Robotics-Kitchen-Sim-Demos/
```

> ⚠️ snapshot 트리는 비어 있고 blob 만 있는 상태일 수 있음 (이전 다운로드가 symlink 까지 안 만들어준 케이스). 다음 4.3 단계에서 어차피 tar 를 직접 풀 거라 무관.

### 4.2 옵션 B — 학습 서버에서 새로 다운로드

cache 옮기는 게 어려우면, 우리 10 task 만 골라 받기:

```bash
mkdir -p ${VLA_DATASETS_ROOT}/huggingface
export HF_HOME=${VLA_DATASETS_ROOT}/huggingface

python - <<'PY'
from huggingface_hub import snapshot_download
# 우리 10 task 의 atomic 데이터만 받기 (allow_patterns)
TASKS = [
    "OpenDrawer", "CloseDrawer", "OpenCabinet", "CloseCabinet",
    "OpenFridge", "CloseFridge", "OpenMicrowave", "CloseMicrowave",
    "PickPlaceCounterToStove", "PickPlaceCounterToSink",
]
patterns = [f"pretrain/atomic/{t}/*" for t in TASKS]
snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir_use_symlinks=False,
)
PY
```

### 4.3 tar 추출 → GR00T LeRobot v2 디렉토리 정리

cache 의 blob 들은 POSIX tar archive 들 (`lerobot/data/...`, `lerobot/meta/...`). 학습 진입점에 넘기려면 디스크에 **task 별 디렉토리** 로 추출해야 함.

```bash
# 추출 대상 루트
mkdir -p <cache>/datasets/groot_robocasa_train

# (다음 절에 추가될) 추출 스크립트 — TODO: scripts/data/extract_kitchen_sim_demos.py
python scripts/data/extract_kitchen_sim_demos.py \
    --hf-cache <cache>/datasets/huggingface/hub/datasets--nvidia--PhysicalAI-Robotics-Kitchen-Sim-Demos \
    --output   <cache>/datasets/groot_robocasa_train \
    --tasks    OpenDrawer CloseDrawer OpenCabinet CloseCabinet \
               OpenFridge CloseFridge OpenMicrowave CloseMicrowave \
               PickPlaceCounterToStove PickPlaceCounterToSink
```

추출 후 디렉토리 구조 (예: OpenDrawer):
```
<cache>/datasets/groot_robocasa_train/OpenDrawer/
├── data/chunk-000/episode_000000.parquet ...
├── videos/chunk-000/observation.images.<view>/episode_000000.mp4 ...
└── meta/{episodes.jsonl, modality.json, info.json, tasks.jsonl, stats.json, ...}
```

> ⚠️ **이 추출 스크립트는 아직 작성 안 됨.** 첫 학습 직전에 쓰면 됨. `tar -xf <blob> -C <task_dir>` 한 줄 + blob → task 매핑 정도.

## 5. 학습 실행

GR00T 의 native finetune 진입점을 그대로 쓴다. ROBOCASA_PANDA_OMRON 은 이미 등록된 embodiment 라 modality config 별도 작성 불필요.

### 5.1 단일 task smoke train

```bash
cd src/policies/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path <cache>/checkpoints/nvidia/GR00T-N1.6-3B \
    --dataset-path    <cache>/datasets/groot_robocasa_train/OpenDrawer \
    --embodiment-tag  ROBOCASA_PANDA_OMRON \
    --num-gpus        1 \
    --output-dir      /path/to/temporal_vla/outputs/train/groot/smoke_opendrawer \
    --max-steps       100 \
    --save-steps      100 \
    --save-total-limit 2 \
    --global-batch-size 16 \
    --dataloader-num-workers 4
```

검증할 것:
- `loss` 가 step 30~50 안에 떨어지기 시작 (보통 `0.5 → 0.3` 정도)
- VRAM 24GB 안 넘는지
- step/sec 측정 (이걸로 전체 학습 시간 추정)

### 5.2 10 task multitask train

GR00T finetune 가 multi-dataset 입력을 받는 방식 두 가지:
- (a) `--dataset-path` 에 여러 경로 전달 (CLI 가 list 지원)
- (b) 모든 task 를 단일 디렉토리로 합쳐서 `episodes.jsonl` 통합

대부분의 GR00T 예제는 (a). `launch_finetune.py --help` 로 확인:
```bash
uv run python gr00t/experiment/launch_finetune.py --help | grep -A 1 dataset-path
```

본격 실행 (예시 — 인자/스텝수는 H/W 와 데이터 양에 따라 조정):
```bash
NUM_GPUS=4
torchrun --nproc-per-node=$NUM_GPUS \
    gr00t/experiment/launch_finetune.py \
    --base-model-path <cache>/checkpoints/nvidia/GR00T-N1.6-3B \
    --dataset-path \
        <cache>/datasets/groot_robocasa_train/OpenDrawer \
        <cache>/datasets/groot_robocasa_train/CloseDrawer \
        <cache>/datasets/groot_robocasa_train/OpenCabinet \
        <cache>/datasets/groot_robocasa_train/CloseCabinet \
        <cache>/datasets/groot_robocasa_train/OpenFridge \
        <cache>/datasets/groot_robocasa_train/CloseFridge \
        <cache>/datasets/groot_robocasa_train/OpenMicrowave \
        <cache>/datasets/groot_robocasa_train/CloseMicrowave \
        <cache>/datasets/groot_robocasa_train/PickPlaceCounterToStove \
        <cache>/datasets/groot_robocasa_train/PickPlaceCounterToSink \
    --embodiment-tag    ROBOCASA_PANDA_OMRON \
    --num-gpus          $NUM_GPUS \
    --output-dir        /path/to/outputs/train/groot/n16_robocasa_10task_$(date +%Y%m%d) \
    --max-steps         20000 \
    --save-steps        2000 \
    --save-total-limit  5 \
    --global-batch-size 64 \
    --use-wandb \
    --tune-projector    True \
    --tune-diffusion-model True \
    --tune-llm          False \
    --tune-visual       False \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 8
```

기본 fine-tune 정책 (`gr00t/configs/finetune_config.py`): projector + diffusion model 만 unfreeze. LLM/visual 은 frozen 유지.

## 6. 학습 결과 → 평가 path 연결

학습 끝나면 `outputs/train/groot/.../checkpoint-<N>/` 디렉토리 생성. 이걸 우리 통일 HTTP API path 로 평가하려면 별도 프로파일 작성:

```bash
cp configs/checkpoints/groot__robocasa_panda_omron.yaml \
   configs/checkpoints/groot__robocasa_panda_omron_finetuned.yaml
```

수정할 곳:
```yaml
# configs/checkpoints/groot__robocasa_panda_omron_finetuned.yaml
name: groot__robocasa_panda_omron_finetuned
checkpoint_source:
  type: local
  id: /temporal_vla/outputs/train/groot/n16_robocasa_10task_<DATE>/checkpoint-20000
# 나머지는 base 와 동일 (action_layout, n_action_steps 등)
```

평가:
```bash
# groot 컨테이너에서 serve
docker exec groot python /temporal_vla/scripts/serve/groot.py \
    --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron_finetuned.yaml

# robocasa 컨테이너에서 eval (각 task 5~10 rollout 권장)
docker exec robocasa python /temporal_vla/scripts/eval/robocasa_eval.py \
    --task-set robocasa_eval_25 \
    --vla-server http://localhost:8500 \
    --use-groot-env \
    --num-rollouts 10 \
    --num-steps 720 \
    --output-dir outputs/eval/robocasa/groot/finetuned_<DATE>
```

base 모델 기준 (`examples/robocasa/README.md`) average 66.22% 와 비교해서 우리가 학습한 10 task 가 얼마나 올라갔는지가 핵심 metric.

## 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `gym.make("robocasa_panda_omron/...")` 시 `composite_controller None` | `src/benchmarks/robocasa` submodule pointer 가 `fda28d0` 이상인지 확인. 우리 fork (`hanyangtae/robocasa`, `temporal_vla` 브랜치) 의 fix 필요. |
| 학습 중 OOM | `--global-batch-size` 줄이기 / `--gradient-accumulation-steps` 늘리기 / `--bf16` (이미 default) 유지. |
| `embodiment_id.json` not found | 체크포인트 경로 잘못. `<cache>/checkpoints/nvidia/GR00T-N1.6-3B/embodiment_id.json` 존재 확인. |
| LeRobot dataset load 실패 | `meta/modality.json` 누락 가능성. NVIDIA Kitchen-Sim-Demos 는 GR00T-flavored 라 들어 있어야 함. tar 추출이 incomplete 한지 확인. |
| 학습 step/sec 너무 느림 | `--dataloader-num-workers` ↑, video decode bottleneck 이면 미리 frame 추출하는 옵션 검토. |

## 참조

- 본 repo `CLAUDE.md` (프로젝트 컨벤션)
- `src/policies/Isaac-GR00T/getting_started/finetune_new_embodiment.md` (NVIDIA 공식 finetune 가이드)
- `src/policies/Isaac-GR00T/getting_started/data_preparation.md` (LeRobot v2 + GR00T flavor 설명)
- `src/policies/Isaac-GR00T/examples/robocasa/README.md` (RoboCasa baseline 결과표)
- `configs/checkpoints/groot__robocasa_panda_omron.yaml` (현재 base 프로파일)
- `src/policies/groot/` (학습 adapter 와 serve 가 공유하는 helper)
