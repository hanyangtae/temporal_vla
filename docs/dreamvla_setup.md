# DreamVLA 설치 및 실행 가이드

DreamVLA는 Seer 기반의 Vision-Language-Action 모델로, world knowledge forecasting을 통한 inverse dynamics modeling으로 perception-prediction-action 루프를 구현합니다.

- 논문: [NeurIPS 2025] DreamVLA
- 원본 repo: https://github.com/Zhangwenyao1/DreamVLA
- 프로젝트 내 위치: `src/policies/dreamvla` (git submodule)

## 아키텍처 개요

```
robocasa 컨테이너 (Python 3.11)          dreamvla 컨테이너 (Python 3.10)
┌─────────────────────────┐              ┌──────────────────────────────┐
│ RoboCasa 시뮬레이션      │  HTTP:8200   │ DreamVLA 추론/학습 서버       │
│ + LeRobot v2.1 데이터셋  │ ──────────> │ src/policies/dreamvla        │
│ + 평가 스크립트           │              │ + LeRobot 도구 활용           │
└─────────────────────────┘              └──────────────────────────────┘
```

## 1. 서브모듈 초기화

```bash
git submodule update --init src/policies/dreamvla
```

## 2. Docker 컨테이너 빌드

```bash
docker compose build dreamvla
```

dreamvla 컨테이너에 설치되는 핵심 의존성:
- Python 3.10 + PyTorch 2.5.1 (CUDA 12.1)
- Flash Attention 2
- `flamingo_pytorch`, `CLIP`, `timm==0.9.16`
- `transformers==4.40.2`, `omegaconf==2.1.2`
- `einops_exts`, `h5py==3.11.0`

## 3. 전체 파이프라인

### 3-1. 데이터 변환: RoboCasa LeRobot → DreamVLA npz

DreamVLA는 Calvin 스타일 npz 포맷을 사용합니다. RoboCasa의 LeRobot v2.1 데이터셋을 변환합니다.

```bash
docker compose run --rm dreamvla python /temporal_vla/scripts/convert_robocasa_to_dreamvla.py \
    --input-base /temporal_vla/data/datasets/v1.0 \
    --output-base /temporal_vla/data/datasets_dreamvla/v1.0 \
    --split pretrain \
    --image-size 224
```

변환 결과: `data/datasets_dreamvla/v1.0/pretrain/{task_name}/`
- `episode_XXXXXXX.npz` (rgb_static, rgb_gripper, action, proprio)
- `lang_annotations/auto_lang_ann.npy` (언어 instruction 매핑)

dry-run으로 미리 확인:
```bash
docker compose run --rm dreamvla python /temporal_vla/scripts/convert_robocasa_to_dreamvla.py --dry-run
```

### 3-2. 학습 (Fine-tuning)

```bash
docker compose run --rm dreamvla bash /temporal_vla/scripts/train_dreamvla.sh
```

환경변수로 학습 파라미터 조정 가능:
```bash
DATA_DIR=/temporal_vla/data/datasets_dreamvla \
OUTPUT_DIR=/temporal_vla/outputs/dreamvla_robocasa \
BATCH_SIZE=2 \
LR=1e-4 \
EPOCHS=50 \
docker compose run --rm dreamvla bash /temporal_vla/scripts/train_dreamvla.sh
```

DreamVLA `train.py` 주요 옵션:
- `--finetune_type`: calvin, libero_pretrain, libero_finetune, droid, oxe 등
- `--lr_scheduler`: linear, cosine, cosine_restart, constant
- `--gradient_checkpointing`: 메모리 절약
- `--resume_from_checkpoint`: 학습 재개
- Mixed precision: bf16/fp16/fp32 지원

### 3-3. 추론 서버 실행

```bash
docker compose run --rm dreamvla python /temporal_vla/scripts/serve_dreamvla.py \
    --model-path /temporal_vla/outputs/dreamvla_robocasa/checkpoint_best \
    --port 8200
```

### 3-4. RoboCasa 벤치마크 평가

robocasa 컨테이너에서 DreamVLA 서버에 HTTP 요청으로 평가:
```bash
# robocasa 컨테이너 안에서
python /temporal_vla/scripts/robocasa_playback_eval.py \
    --model-url http://localhost:8200
```

## 4. LeRobot 버전 정리

| 컨테이너  | LeRobot 버전      | 데이터 포맷 | 용도                  |
|----------|-------------------|------------|----------------------|
| robocasa | `lerobot==0.3.3`  | v2.1       | 데이터 로딩 + 시뮬레이션 평가 |
| xvla     | `lerobot[xvla]`   | v3.0       | X-VLA 학습/추론        |
| dreamvla | 직접 사용 안함       | npz (Calvin) | 자체 데이터 로더 사용    |

> **참고:** LeRobot 도구는 주로 robocasa/xvla 컨테이너에서 데이터셋 관리와 포맷 변환에 활용합니다. DreamVLA 컨테이너에서는 변환된 npz 데이터를 자체 데이터 로더로 처리합니다.

## 5. 디렉토리 구조

```
src/policies/
└── dreamvla/          ← git submodule (Zhangwenyao1/DreamVLA)
    ├── models/        ← DreamVLA 모델 코드 (dreamvla_model.py 등)
    ├── utils/         ← 학습 유틸리티 (train_utils.py, data_utils.py)
    ├── data_process/  ← 데이터 전처리
    ├── data_info/     ← annotation 파일 (auto_lang_ann.npy 등)
    ├── scripts/       ← 원본 학습/평가 스크립트
    ├── train.py       ← 학습 진입점
    ├── eval_calvin.py ← CALVIN 벤치마크 평가
    ├── eval_libero.py ← LIBERO 벤치마크 평가
    └── requirements.txt
```

## 6. 트러블슈팅

### Flash Attention 빌드 실패
```bash
# 컨테이너 안에서 수동 설치
pip install psutil packaging ninja
pip install flash-attn --no-build-isolation
```

### CLIP 설치 실패
```bash
pip install git+https://github.com/openai/CLIP.git
```

### numpy 버전 충돌
DreamVLA는 `numpy==1.23.1`을 요구하지만, 다른 패키지와 충돌할 수 있습니다. Docker 컨테이너 격리로 해결됩니다.

### 모델 weights 다운로드
```bash
# HuggingFace에서 pretrained weights 다운로드
python -c "from huggingface_hub import snapshot_download; snapshot_download('WenyaoZhang/DreamVLA', local_dir='/temporal_vla/data/huggingface/dreamvla')"
```
