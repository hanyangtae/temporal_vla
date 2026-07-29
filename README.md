# temporal_vla

VLA(Vision-Language-Action) 모델의 **latent activation steering** 을 연구하는 프로젝트입니다.
실패하는 VLA의 내부 활성화를 추론 시 성공 분포 쪽으로 **steer** 하여, 백본 재학습 없이 Success Rate를 올리는 것이 목표입니다.

RoboCasa/Calvin 시뮬레이션 환경에서 다양한 VLA 모델(pi0.5, groot, UP-VLA, X-VLA 등)을 Docker 기반 통일 API로 서빙·평가하는 인프라를 갖추고 있습니다.

## Research Direction

문제 설정은 **pathway-resolved + phase-matched activation steering** 입니다. 성공/실패 활성화의
차이를 연산자로 만들어(예: contrastive conceptor `C_steer = C_success ∧ ¬C_failure`, `h' = h·Mᵀ`)
추론 중 활성화를 성공 쪽으로 밀고, VL(goal "what")과 DiT(motor "how") pathway를 나눠서,
DiT는 rollout phase에 조건부로 개입합니다. 백본 재학습은 없습니다.

**현재 상태 — steering이 SR을 올리는지 자체를 아직 검증 중입니다.** 지금까지 확인된 것:

- **COAST(대조 conceptor) 재현 실패.** 논문의 평균 ΔSR +0.16이 우리 환경에서 재현되지 않았고,
  여러 라운드에 걸쳐 위약 대조까지 붙인 뒤에도 scene 일관된 개선이 나오지 않았습니다.
- **읽을 수는 있으나 쓰지는 못합니다(read ≠ write).** scene·길이·dwell·seed를 통제하면 성공/실패
  분리는 실재합니다(일부 cell AUROC 0.84~0.91). 그런데 같은 방향으로 개입하면 SR이 안 움직입니다.
- **겉보기 분리의 상당수는 confound**였습니다 — 길이(실패는 항상 timeout), task 정체성, scene 암기.
- **핵심 난제는 그대로**: 추론 중(online)에 어느 pathway·어느 phase에서 실패하는지 식별 가능한가.
- **다음 후보**: 연산자를 바꿔보는 축(WA-LQR 계열 diff-of-means + LQR 재현 검토), scene 성분을
  SAE로 분리한 뒤 conceptor, phase 앵커 재정의. 아직 어느 것도 확정된 방법이 아닙니다.

배경: 이 방향은 이전 "실패 루프 탈출(loop) / 메모리 부재" 및 "TTA progress predictor" 프레이밍을 대체합니다 — 실패 데이터를 직접 분석한 결과 loop는 실패의 표면 현상일 뿐이었고, 문제를 latent steering 관점으로 재정의했습니다. 라운드별 상세는 [`docs/steering/`](docs/steering/README.md), 문제 설정 단일 출처는 [`14_pathway_phase_online_steering.md`](docs/steering/14_pathway_phase_online_steering.md).

### Baseline
- VLA 모델: pi0.5, groot
- World action모델: cosmos policy
- 평가: Success Rate, steering 후 ΔSR 인과 재측정 (EVAL_SEED=100000 표준)

## Architecture

각 모델과 시뮬레이션 환경의 의존성 충돌을 방지하기 위해 Docker 컨테이너를 분리합니다.
모든 컨테이너는 `network_mode: host`를 사용하여 `localhost`로 통신합니다.

모델 서버와 벤치마크 스크립트는 **통일 API**를 사용하여 모델-벤치마크 조합을 자유롭게 변경할 수 있습니다.

```
┌──────────────────────────────────┐    통일 API (HTTP)
│  robocasa / calvin (/ libero)    │    POST /act, /act_with_features,
│  벤치마크 + 평가                  │         /reset, GET /health
│                                  │────────────────────────────────┐
│  ObsProcessor → VLAClient        │                                │
│  ActionProcessor ← response      │                                │
└──────────────────────────────────┘                                ▼
        ┌─────────┐ ┌──────────┐ ┌─────────────┐ ┌─────────┐ ┌─────────┐
        │  xvla   │ │  upvla   │ │ openvla_oft │ │ lerobot │ │  groot  │
        │  :8100  │ │  :8300   │ │   :8400     │ │  :8400  │ │  :8500  │
        └─────────┘ └──────────┘ └─────────────┘ └─────────┘ └─────────┘

모든 컨테이너: ./ → /temporal_vla (볼륨 마운트), network_mode: host
```

세부 사항은 [`docs/01_serving_interface.md`](docs/01_serving_interface.md) 참조.

### 통일 API 규격

모든 모델 서버(`scripts/serve/*.py`)와 벤치마크 평가 스크립트가 같은 HTTP 계약을 따릅니다. 엔드포인트(`/act`, `/act_with_features`, `/reset`, `/health`), 요청 payload 의 sub-key 네임스페이스, 응답 sub-key 표준, 모델 × 벤치마크 호환 매트릭스, 운영 패턴은 [`docs/01_serving_interface.md`](docs/01_serving_interface.md) 단일 문서를 단일 출처(single source of truth)로 두고 정리합니다.

요약:

- `/act` 는 sub-keyed action dict 를 반환합니다. 모델은 자신의 native 출력을 표준 sub-key (`action.eef_pos`, `action.eef_euler` / `action.eef_rot6d` / ..., `action.gripper` 등) 로 분리해 보내고, 벤치마크 측 ActionProcessor 가 env 포맷으로 합쳐 `env.step()` 에 넘깁니다.
- `/act_with_features` 는 `/act` 와 같은 응답에 `features.*` namespace (`features.hidden_states` feature blob + 메타) 를 더해 반환합니다 (모델이 features 를 지원할 때만).
- 벤치마크 스크립트는 `VLAClient` (`scripts/utils/vla_client.py`) 와 generic `ProcessorPipeline` (`src/processor/`) 만 사용하므로, `--vla-server` URL 만 바꾸면 같은 벤치에 다른 모델을 붙일 수 있습니다. GR00T `GrootRoboCasaEnv` native-key 경로는 예외적으로 `src/policies/groot/robocasa/io.py` adapter를 사용해 upstream parity와 SAFE wiring을 맞춥니다.
- 새 체크포인트/모델/벤치를 붙이는 절차는 [`docs/03_adding_checkpoint.md`](docs/03_adding_checkpoint.md) 와 [`configs/checkpoints/README.md`](configs/checkpoints/README.md) 를 참조합니다.

### Containers

| Container | 역할 | Python | Port |
|-----------|------|--------|------|
| robocasa | RoboCasa 시뮬레이션 + 평가 + GUI (KasmVNC/X11) | 3.11 | 8444 (VNC) |
| calvin | Calvin 벤치마크 + 평가 (headless EGL) | 3.8 | - |
| xvla | X-VLA 학습/추론 서버 | 3.10 | 8100 |
| upvla | UP-VLA 추론 서버 | 3.10 | 8300 |
| openvla_oft | OpenVLA-OFT 추론 서버 (LIBERO/Calvin) | 3.10 | 8400 |
| lerobot | pi0 / pi05 추론 서버 (LeRobot stack) | 3.10 | 8400 |
| groot | GR00T N1.6 학습/추론 서버 + SAFE features | 3.10 | 8500 |
| groot_n15 | GR00T N1.5 fine-tuning | 3.10 | - |

`openvla_oft` 와 `lerobot` 은 다른 컨테이너이므로 같은 호스트 포트 `8400` 을 동시에 띄우진 않습니다. 모델 × 벤치마크 호환 매트릭스는 [`docs/01_serving_interface.md`](docs/01_serving_interface.md#model--benchmark-호환-매트릭스) 참조.

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
├── docker-compose.yml                        # 서비스 정의 (8개 컨테이너)
├── docker-compose.override*.yml              # GPU + 디스플레이 설정 템플릿
├── .env.example                              # 환경변수 템플릿
├── configs/
│   └── checkpoints/                          # 체크포인트 프로파일 YAML (serve 분기 기준)
├── docker/                                   # 컨테이너별 Dockerfile
├── scripts/
│   ├── path_setup.py                         # 경로 단일 출처 (CHECKPOINTS_ROOT, DATA_ROOT)
│   ├── serve/                                # 모델 추론 서버 — HTTP 통일 API (Docker 내 실행)
│   │   ├── lerobot.py                        # pi0 / pi0.5 / GR00T N1.5 (+ steering 배선)
│   │   ├── groot.py                          # GR00T
│   │   ├── openvla_oft.py  xvla.py  upvla.py # 기타 policy
│   │   └── steering_hooks.py  safe_hooks.py  # activation hook (steer / capture)
│   ├── eval/                                 # 벤치마크 평가 (모델 무관)
│   │   ├── robocasa_eval.py                  # RoboCasa closed-loop
│   │   └── calvin.py                         # Calvin
│   ├── safe/                                 # ★ 실험 파이프라인 본체 (exp2~exp5)
│   │   ├── groot_n15/robocasa/               # steering 본류
│   │   │   ├── collect/ split/ analyze/      # 수집 · 분할 · 분석
│   │   │   └── steer/                        # 라운드별(exp2~exp5) + queue/ 양-머신 러너
│   │   └── groot_n16/robocasa/               # N1.6 — collect/ serve/(ZMQ) steer/ analyze/ vis/
│   ├── scene_sae/                            # scene 성분 분리 SAE (G1~G3)
│   ├── train/  analysis/  extract/           # 학습 · 분석 · 데이터셋 준비
│   ├── review/                               # 레포 검토 판정 UI
│   └── utils/
│       ├── vla_client.py                     # 통일 VLA HTTP 클라이언트 (VLAClient)
│       ├── cache_env.sh                      # 셸 경로 단일 출처
│       └── remote_compute.sh                 # 원격 노드 오케스트레이션
├── src/
│   ├── benchmarks/                           # submodule — robocasa / robosuite / calvin
│   ├── policies/                             # submodule — Isaac-GR00T, UP-VLA + groot adapter
│   ├── processor/                            # 추론용 Processor Pipeline (obs/ action/ factory)
│   ├── datasets/                             # 학습용 데이터 파이프라인 (+ adapters/)
│   ├── sae/                                  # SAE 코어 (models, train, metrics, pca, cluster)
│   ├── phase_online/                         # online phase 신호
│   ├── ttt/                                  # 구 TTA/progress predictor (보존)
│   └── utils/
├── lerobot/                                  # Git submodule (LeRobot)
├── docs/                                     # 문서 — 지도는 docs/00_docs_map.md
├── tests/
└── outputs/                                  # 로그, 평가 결과, 영상
```

체크포인트·데이터셋은 repo 트리 밖 `~/.cache/temporal_vla/` 에 두고 컨테이너에 `/cache` 로
마운트합니다. 경로는 `scripts/path_setup.py` / `scripts/utils/cache_env.sh` 를 통해서만 참조합니다.

## Common Commands

### 컨테이너 관리

```bash
docker compose up -d robocasa                              # robocasa 시작
docker compose exec robocasa bash                          # robocasa 셸 접속
docker compose down                                        # 전체 중지
docker compose ps                                          # 상태 확인
docker compose logs -f robocasa                            # 실시간 로그
```

### Training


### Inference Server (통일 API)

모든 서버는 동일한 API를 제공합니다. 벤치마크에서 `--vla-server` URL만 바꾸면 됩니다.

```bash
# X-VLA 서버 (port 8100)
docker compose run --rm xvla \
  python /temporal_vla/scripts/serve/xvla.py --model-path lerobot/xvla-base

# UP-VLA 서버 (port 8300)
docker compose run --rm upvla \
  python /temporal_vla/scripts/serve/upvla.py \
    --model-config /temporal_vla/src/policies/UP-VLA/policy_rollout/upvla_model.yaml
```

### Evaluation

```bash
# RoboCasa closed-loop 평가 (모델 무관, 서버 먼저 실행 필요)
# --vla-server URL만 바꾸면 X-VLA, UP-VLA 등 어떤 모델이든 평가 가능
docker compose exec robocasa python /temporal_vla/scripts/eval/robocasa_eval.py \
  --task TurnOnMicrowave --vla-server http://localhost:8300

# 태스크셋 평가 (pretrain50, target50, all_tasks 등)
docker compose exec robocasa python /temporal_vla/scripts/eval/robocasa_eval.py \
  --task-set pretrain50 --vla-server http://localhost:8300 \
  --output-dir /temporal_vla/outputs/vla_eval

# Calvin 평가 (모델 무관)
docker compose exec calvin python /temporal_vla/scripts/eval/calvin.py \
  --dataset-path /cache/datasets/calvin/task_ABC_D \
  --server-url http://localhost:8300 --act-step 10

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

문서 지도(어떤 문서가 어디에 있는지)는 [`docs/00_docs_map.md`](docs/00_docs_map.md) 하나로 모읍니다.

자주 쓰는 진입점:

- [통일 API 규격](docs/01_serving_interface.md) — 엔드포인트·sub-key·호환 매트릭스 단일 출처
- [Docker 사용 가이드](docs/02_docker_guide.md) — Docker에 익숙하지 않은 팀원을 위한 상세 가이드
- [새 체크포인트 추가](docs/03_adding_checkpoint.md) — profile/serve/eval 배선 체크리스트
- [연구 문서](docs/steering/README.md) — steering 라운드별 계획·결과
