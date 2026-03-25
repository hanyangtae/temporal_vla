# temporal_vla

VLA(Vision-Language-Action) 모델의 **실패 루프 탈출** 문제를 연구하는 프로젝트입니다.
성공 데이터로만 학습된 VLA 모델이 실패 시 같은 trajectory를 반복하는 문제를, 외부 모듈(TTA 기반 progress predictor)을 통해 VLA 백본 추가학습 없이 해결하는 것을 목표로 합니다.

RoboCasa/Calvin 시뮬레이션 환경에서 다양한 VLA 모델(pi0, groot, DreamVLA 등)을 Docker 기반 통일 API로 서빙·평가하는 인프라를 갖추고 있습니다.

## Research Direction

### 문제
성공 데이터로만 학습된 VLA 모델이 실패 시 같은 trajectory를 반복(Loop)하며 실패를 반복하는 현상.

### 가설
외부 모듈(TTA progress predictor)을 사용해 **VLA 백본 추가학습 없이, 실패 데이터 없이**, 기존 VLA 모델의 action output에 변형을 줘서 실패 루프 탈출 → **Success Rate 개선**.

### 접근 방식
1. **실패 감지**: VITA(ICLR 2026) 기반 TTA adaptation module로 task 진행률(0~1) 예측. 단조증가에서 벗어나면 실패로 판단.
2. **Action 변형** (실험 후보):
   - LLM 출력 head 직전에 TTA hidden state projection add
   - VLA 출력 Logit shifting (이산화 출력 모델)
   - Diffusion action expert에 FiLM condition 입력
   - Input 토큰 추가 (가장 간단)

### Baseline
- VLA 모델: pi0, groot
- 진행률/실패 추정: VITA

## Architecture

각 모델과 시뮬레이션 환경의 의존성 충돌을 방지하기 위해 Docker 컨테이너를 분리합니다.
모든 컨테이너는 `network_mode: host`를 사용하여 `localhost`로 통신합니다.

모델 서버와 벤치마크 스크립트는 **통일 API**를 사용하여 모델-벤치마크 조합을 자유롭게 변경할 수 있습니다.

```
┌─────────────────────────────┐       통일 API (HTTP)
│  robocasa / calvin          │       POST /act, /reset
│  벤치마크 + 평가             │──────────────────────────────┐
│                             │─────────────┐                │
│  ObsProcessor → VLAClient   │             │                │
│  ActionProcessor ← 응답     │             │                │
└─────────────────────────────┘             ▼                ▼
                                    ┌──────────┐  ┌───────────┐  ┌──────────┐
                                    │   xvla   │  │ dreamvla  │  │  upvla   │
                                    │  :8100   │  │  :8200    │  │  :8300   │
                                    └──────────┘  └───────────┘  └──────────┘

모든 컨테이너: ./  →  /temporal_vla (볼륨 마운트)
```

### 통일 API 규격 (LeRobot 컨벤션)

모든 모델 서버(`serve_*.py`)가 동일한 엔드포인트와 요청/응답 형식을 따릅니다.
키 네이밍은 [LeRobot](https://github.com/huggingface/lerobot) 컨벤션을 따릅니다:

| Endpoint | 설명 |
|----------|------|
| `POST /act` | 액션 예측. 요청: `{"observation.images.static": b64png, "observation.images.wrist": b64png, "observation.state": [...], "task": "..."}`<br>응답: `{"action": [[floats], ...], "latency_ms": float}` |
| `POST /reset` | 에피소드 시작 시 히스토리 초기화 (필요 없는 모델은 no-op) |
| `GET /health` | 서버 상태 + feature 정보 (`input_features`, `output_features`, `n_action_steps`) |

벤치마크 스크립트는 `VLAClient`(`scripts/utils/vla_client.py`)와 `ProcessorPipeline`(`src/processor/`)을 사용합니다.
`--vla-server` URL만 바꾸면 어떤 모델이든 평가할 수 있으며, obs/action 변환은 벤치마크별 Processor가 처리합니다.

### Containers

| Container | 역할 | Python | Port |
|-----------|------|--------|------|
| robocasa | RoboCasa 시뮬레이션 + 평가 + GUI (KasmVNC/X11) | 3.11 | 8444 (VNC) |
| calvin | Calvin 벤치마크 + 평가 (headless EGL) | 3.8 | - |
| xvla | X-VLA 학습/추론 서버 (LeRobot) | 3.10 | 8100 |
| dreamvla | DreamVLA 학습/추론 서버 | 3.10 | 8200 |
| upvla | UP-VLA 추론 서버 | 3.10 | 8300 |

## Prerequisites

- Ubuntu 22.04+
- NVIDIA GPU (CUDA 12.1 호환 드라이버, `nvidia-smi`로 확인)
- Docker Engine 24+ with Compose V2 (`docker compose` 명령어 사용)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Git
- 디스크 약 50GB 이상 (Docker 이미지 + 데이터셋 + 모델 캐시)

## Quick Start

### 1. Clone (서브모듈 포함)

```bash
git clone --recursive <repo-url> temporal_vla
cd temporal_vla
```

이미 clone한 경우:

```bash
git submodule update --init --recursive
```

### 2. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 항목을 수정합니다:

```bash
USER_NAME=<호스트 사용자 이름>   # whoami
USER_ID=<호스트 UID>            # id -u
GROUP_ID=<호스트 GID>           # id -g
HF_TOKEN=<HuggingFace 토큰>    # huggingface.co/settings/tokens
```

### 3. 디스플레이 모드 선택

사용 환경에 맞는 override 파일을 복사합니다:

**SSH 원격 접속 (VNC 모드):**

```bash
cp docker-compose.override.example.yml docker-compose.override.yml
```

`.env`에서 `VNC_PW` 설정 → 브라우저로 `https://<서버IP>:8444` 접속

**로컬 우분투 PC (X11 모드):**

```bash
cp docker-compose.override.local.example.yml docker-compose.override.yml
```

`.env`에서 `DISPLAY` 설정 (보통 `:0`, `echo $DISPLAY`로 확인) → 호스트에서 `xhost +local:docker` 실행

### 4. 이미지 빌드

```bash
# robocasa (필수)
docker compose build robocasa

# 모델 컨테이너 (필요 시)
docker compose build xvla
docker compose build dreamvla
```

> 첫 빌드는 PyTorch, Flash Attention 등 빌드로 30분 이상 소요될 수 있습니다.

### 5. 컨테이너 시작

```bash
docker compose up -d robocasa
```

### 6. GPU 확인

```bash
docker compose exec robocasa nvidia-smi
```

## Project Structure

```
temporal_vla/
├── docker-compose.yml                        # 서비스 정의 (3개 컨테이너)
├── docker-compose.override*.yml              # GPU + 디스플레이 설정 템플릿
├── .env.example                              # 환경변수 템플릿
├── docker/
│   ├── robocasa/                             # 시뮬레이션 컨테이너 (Python 3.11, KasmVNC)
│   ├── xvla/                                 # X-VLA 컨테이너 (Python 3.10, LeRobot)
│   └── dreamvla/                             # DreamVLA 컨테이너 (Python 3.10)
├── scripts/
│   ├── serve_xvla.py                         # X-VLA 추론 서버 (:8100, 통일 API)
│   ├── serve_dreamvla.py                     # DreamVLA 추론 서버 (:8200, 통일 API)
│   ├── serve_upvla.py                        # UP-VLA 추론 서버 (:8300, 통일 API)
│   ├── train_xvla.sh                         # X-VLA LoRA fine-tuning
│   ├── train_dreamvla.sh                     # DreamVLA fine-tuning (Calvin)
│   ├── train_dreamvla_robocasa.py            # DreamVLA fine-tuning (RoboCasa) 스크립트
│   ├── train_dreamvla_robocasa.sh            # DreamVLA fine-tuning (RoboCasa) 실행
│   ├── extract/
│   │   ├── extract_sam_robocasa.py           # SAM feature 추출 (LeRobot → .pt)
│   │   └── extract_cotrack_robocasa.py       # CoTracker trajectory 추출 (LeRobot → .npz)
│   ├── robocasa_vla_eval.py                  # RoboCasa closed-loop 평가 (모델 무관)
│   ├── calvin_eval.py                        # Calvin 평가 (모델 무관)
│   ├── robocasa_playback_eval.py             # 녹화 데이터 재생 평가 (상태 체크 / open-loop)
│   ├── robocasa_render_failures.py           # 실패 에피소드 영상 렌더링
│   ├── start_vnc.sh                          # KasmVNC 시작 스크립트
│   └── utils/
│       ├── vla_client.py                     # 통일 VLA HTTP 클라이언트 (VLAClient)
│       └── robocasa_eval.py                  # playback 평가 유틸리티
├── lerobot/                                  # Git submodule (LeRobot)
├── data/
│   ├── datasets/                             # RoboCasa 데이터 (LeRobot v2.1, 원본)
│   └── huggingface/                          # HuggingFace 모델 캐시
├── outputs/                                  # 로그, 평가 결과, 영상
├── src/
│   ├── benchmarks/
│   │   ├── calvin/                           # Git submodule (CALVIN benchmark)
│   │   ├── robocasa/                         # Git submodule (주방 벤치마크)
│   │   └── robosuite/                        # Git submodule (로봇 시뮬레이션)
│   ├── datasets/                             # 학습용 데이터 파이프라인
│   │   └── adapters/                         # 모델별 adapter (LeRobotDataset wrapping + 변환 + collator)
│   │       └── dreamvla.py                   # DreamVLA adapter (SAM/track feature 지원)
│   ├── processor/                            # 추론용 Processor Pipeline (LeRobot 인터페이스 호환)
│   │   ├── base.py                           # ProcessorStep, DataProcessorPipeline
│   │   ├── types.py                          # FeatureType, PolicyFeature, Transition
│   │   ├── factory.py                        # make_calvin_processors(), make_robocasa_processors()
│   │   ├── obs/                              # 벤치마크별 ObservationProcessorStep
│   │   │   ├── calvin.py                     # CalvinObsProcessor
│   │   │   └── robocasa.py                   # RoboCasaObsProcessor
│   │   └── action/                           # 벤치마크별 ActionProcessorStep
│   │       ├── calvin.py                     # CalvinActionProcessor
│   │       └── robocasa.py                   # RoboCasaActionProcessor
│   └── utils/
├── src/utils/common/logger.py                # 공용 로깅 모듈
```

## Common Commands

### 컨테이너 관리

```bash
docker compose up -d robocasa                              # robocasa 시작
docker compose exec robocasa bash                          # robocasa 셸 접속
docker compose up -d dreamvla                              # dreamvla 시작
docker compose down                                        # 전체 중지
docker compose ps                                          # 상태 확인
docker compose logs -f robocasa                            # 실시간 로그
```

### Training

```bash
# X-VLA LoRA fine-tuning
docker compose run --rm xvla \
  bash /temporal_vla/scripts/train_xvla.sh

# DreamVLA fine-tuning (Calvin)
docker compose run --rm dreamvla \
  bash /temporal_vla/scripts/train_dreamvla.sh

# DreamVLA fine-tuning (RoboCasa)
docker compose run --rm dreamvla \
  bash /temporal_vla/scripts/train_dreamvla_robocasa.sh
```

### Inference Server (통일 API)

모든 서버는 동일한 API를 제공합니다. 벤치마크에서 `--vla-server` URL만 바꾸면 됩니다.

```bash
# X-VLA 서버 (port 8100)
docker compose run --rm xvla \
  python /temporal_vla/scripts/serve_xvla.py --model-path lerobot/xvla-base

# DreamVLA 서버 (port 8200)
docker compose run --rm dreamvla \
  python /temporal_vla/scripts/serve_dreamvla.py \
    --checkpoint /temporal_vla/checkpoints/dreamvla/checkpoint.pt \
    --precision bf16

# UP-VLA 서버 (port 8300)
docker compose run --rm upvla \
  python /temporal_vla/scripts/serve_upvla.py \
    --model-config /temporal_vla/src/policies/UP-VLA/policy_rollout/upvla_model.yaml
```

### Evaluation

```bash
# 녹화 데이터 재생 평가 (빠른 상태 체크, 기본값)
docker compose exec robocasa python /temporal_vla/scripts/robocasa_playback_eval.py \
  --dataset /temporal_vla/data/datasets/v1.0/pretrain/atomic/TurnOnToaster/20250820/lerobot

# 녹화 데이터 재생 평가 (open-loop 액션 재생, 궤적 발산 확인)
docker compose exec robocasa python /temporal_vla/scripts/robocasa_playback_eval.py \
  --dataset <path> --use-actions

# RoboCasa closed-loop 평가 (모델 무관, 서버 먼저 실행 필요)
# --vla-server URL만 바꾸면 DreamVLA, X-VLA, UP-VLA 등 어떤 모델이든 평가 가능
docker compose exec robocasa python /temporal_vla/scripts/robocasa_vla_eval.py \
  --task TurnOnMicrowave --vla-server http://localhost:8200

# 태스크셋 평가 (pretrain50, target50, all_tasks 등)
docker compose exec robocasa python /temporal_vla/scripts/robocasa_vla_eval.py \
  --task-set pretrain50 --vla-server http://localhost:8200 \
  --output-dir /temporal_vla/outputs/vla_eval

# Calvin 평가 (모델 무관)
docker compose exec calvin python /temporal_vla/scripts/calvin_eval.py \
  --dataset-path /temporal_vla/data/calvin/task_ABC_D \
  --server-url http://localhost:8300 --act-step 10

# 전체 pretrain 데이터셋 재생 평가 (데이터 품질 확인용)
docker compose exec robocasa python /temporal_vla/scripts/robocasa_playback_eval.py \
  --all --split pretrain --output-dir /temporal_vla/outputs/eval

# 실패 에피소드 영상 렌더링
docker compose exec robocasa python /temporal_vla/scripts/robocasa_render_failures.py \
  --result /temporal_vla/outputs/eval_result.json
```

## GPU Configuration

`docker-compose.override.yml`에서 GPU 할당을 관리합니다. 기본값은 GPU 0입니다.

```yaml
# GPU 번호 변경 예시
device_ids: ['1']              # GPU 1로 변경
NVIDIA_VISIBLE_DEVICES=1       # 환경변수도 함께 변경
```

Multi-GPU 서버에서는 각 컨테이너에 서로 다른 GPU를 할당할 수 있습니다.

## Documentation

- [Docker 사용 가이드](docs/docker_guide.md) — Docker에 익숙하지 않은 팀원을 위한 상세 가이드
