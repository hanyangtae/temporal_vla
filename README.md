# temporal_vla

Vision-Language-Action(VLA) 모델을 RoboCasa 주방 시뮬레이션 환경에서 fine-tuning하고 평가하는 프로젝트입니다.
현재 X-VLA와 DreamVLA를 지원하며, 향후 커스텀 모델 개발 및 실물 실험으로 확장할 예정입니다.

## Architecture

각 모델과 시뮬레이션 환경의 의존성 충돌을 방지하기 위해 Docker 컨테이너를 분리합니다.
모든 컨테이너는 `network_mode: host`를 사용하여 `localhost`로 통신합니다.

```
┌──────────────────────────┐
│     robocasa container   │         HTTP localhost
│  시뮬레이션 + 평가 + GUI  │───────────────────────────┐
│  Python 3.11             │                           │
│  KasmVNC / X11           │──────────────┐            │
└──────────────────────────┘              │            │
                                          ▼            ▼
                                  ┌──────────┐  ┌───────────┐
                                  │   xvla   │  │ dreamvla  │
                                  │  :8100   │  │  :8200    │
                                  │ LeRobot  │  │ DreamVLA  │
                                  └──────────┘  └───────────┘

모든 컨테이너: ./  →  /temporal_vla (볼륨 마운트)
```

| Container | 역할 | Python | Port |
|-----------|------|--------|------|
| robocasa | 시뮬레이션 환경, 데이터셋 평가, GUI (KasmVNC/X11) | 3.11 | 8444 (VNC) |
| xvla | X-VLA 학습/추론 서버 (LeRobot) | 3.10 | 8100 |
| dreamvla | DreamVLA 학습/추론 서버 | 3.10 | 8200 |

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
│   ├── serve_xvla.py                         # X-VLA 추론 서버 (:8100)
│   ├── serve_dreamvla.py                     # DreamVLA 추론 서버 (:8200)
│   ├── train_xvla.sh                         # X-VLA LoRA fine-tuning
│   ├── train_dreamvla.sh                     # DreamVLA fine-tuning
│   ├── robocasa_playback_eval.py             # 녹화 데이터 재생 평가 (상태 체크 / open-loop)
│   ├── robocasa_vla_eval.py                  # DreamVLA closed-loop 평가
│   ├── robocasa_render_failures.py           # 실패 에피소드 영상 렌더링
│   ├── convert_v21_to_v30.py                 # LeRobot v2.1 → v3.0 변환 (X-VLA용)
│   ├── start_vnc.sh                          # KasmVNC 시작 스크립트
│   └── utils/
│       ├── vla_client.py                     # 모델 서버 HTTP 클라이언트 (XVLAClient, DreamVLAClient)
│       └── robocasa_eval.py                  # playback 평가 유틸리티
├── robosuite/                                # Git submodule (로봇 시뮬레이션)
├── robocasa/                                 # Git submodule (주방 벤치마크)
├── lerobot/                                  # Git submodule (LeRobot)
├── data/
│   ├── datasets/                             # RoboCasa 데이터 (LeRobot v2.1, 원본)
│   ├── datasets_v3/                          # X-VLA용 변환 데이터 (LeRobot v3.0)
│   └── huggingface/                          # HuggingFace 모델 캐시
├── outputs/                                  # 로그, 평가 결과, 영상
├── src/utils/common/logger.py                # 공용 로깅 모듈
├── configs/                                  # 모델 설정 (예정)
├── models/                                   # 커스텀 모델 코드 (예정)
└── experiments/                              # 실험 기록 (예정)
```

## Common Commands

### 컨테이너 관리

```bash
docker compose up -d robocasa                              # robocasa 시작
docker compose exec robocasa bash                          # robocasa 셸 접속
docker compose up -d xvla                                  # xvla 시작
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

# DreamVLA fine-tuning (사전에 dreamvla 레포 clone 필요)
# git clone https://github.com/Zhangwenyao1/DreamVLA dreamvla
docker compose run --rm dreamvla \
  bash /temporal_vla/scripts/train_dreamvla.sh
```

### Inference Server

```bash
# X-VLA 서버 (port 8100)
docker compose run --rm xvla \
  python /temporal_vla/scripts/serve_xvla.py --model-path lerobot/xvla-base

# DreamVLA 서버 (port 8200)
# --checkpoint: 학습된 체크포인트 경로 (필수)
# --precision: bf16 권장 (GPU 메모리 절약)
docker compose run --rm dreamvla \
  python /temporal_vla/scripts/serve_dreamvla.py \
    --checkpoint /temporal_vla/checkpoints/dreamvla/checkpoint.pt \
    --precision bf16
```

### Data Conversion

```bash
# LeRobot v2.1 → v3.0 (X-VLA용)
docker compose run --rm xvla \
  python /temporal_vla/scripts/convert_v21_to_v30.py
```

### Evaluation

```bash
# 녹화 데이터 재생 평가 (빠른 상태 체크, 기본값)
docker compose exec robocasa python /temporal_vla/scripts/robocasa_playback_eval.py \
  --dataset /temporal_vla/data/datasets/v1.0/pretrain/atomic/TurnOnToaster/20250820/lerobot

# 녹화 데이터 재생 평가 (open-loop 액션 재생, 궤적 발산 확인)
docker compose exec robocasa python /temporal_vla/scripts/robocasa_playback_eval.py \
  --dataset <path> --use-actions

# DreamVLA closed-loop 평가 (서버 먼저 실행 필요)
docker compose exec robocasa python /temporal_vla/scripts/robocasa_vla_eval.py \
  --dataset <path> --vla-server http://localhost:8200

# 전체 pretrain 데이터셋 평가
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
