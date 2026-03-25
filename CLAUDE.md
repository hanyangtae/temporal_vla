# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 참고하는 지침입니다.

## 프로젝트 개요

VLA 모델의 **실패 루프 탈출** 문제를 연구하는 프로젝트.
성공 데이터로만 학습된 VLA 모델이 실패 시 같은 trajectory를 반복하는 문제를,
외부 모듈(TTA 기반 progress predictor)을 통해 VLA 백본 추가학습 없이 해결하는 것이 목표.

인프라: Docker 컨테이너로 모델과 벤치마크(RoboCasa, Calvin)를 분리하고, 통일 HTTP API로 통신.

## 연구 방향

- **실패 감지**: VITA(ICLR 2026) 기반 TTA adaptation module → task 진행률(0~1) 예측, 단조증가 이탈 시 실패 판단
- **Action 변형** (실험 후보):
  - LLM 출력 head 직전에 TTA hidden state projection add
  - VLA 출력 Logit shifting (이산화 출력 모델)
  - Diffusion action expert에 FiLM condition 입력
  - Input 토큰 추가
- **Baseline 모델**: pi0, groot
- **Metric**: Success Rate 상승

## 핵심 아키텍처

- **모델 서버** (`scripts/serve_*.py`): FastAPI + uvicorn. 통일 API (`/act`, `/reset`, `/health`).
- **벤치마크 평가** (`scripts/*_eval.py`): 모델 무관. `VLAClient`로 통신.
- **Processor Pipeline** (`src/processor/`): **추론(eval)용**. 벤치마크별 env↔통일API obs/action 변환.
- **Dataset + Adapter** (`src/datasets/`): **학습(train)용**. 벤치마크별 generic dataset + 모델별 adapter.
- **통일 클라이언트** (`scripts/utils/vla_client.py`): `VLAClient` 클래스 1개.
- **Docker**: `docker-compose.yml`에 5개 서비스. 모두 `network_mode: host`.

## 통일 API 규격

`scripts/utils/vla_client.py:1-24` 참고. 핵심 규칙:
- state sub-keys는 벤치마크마다 존재하는 키가 다름. 모델 서버는 필요한 키만 꺼내 쓰고, 없으면 변환(quat→euler 등)을 자체 수행.
- 이미지는 base64 PNG. action 응답은 항상 2D array.

## 주요 파일 경로

- 모델 서버: `scripts/serve_dreamvla.py` (:8200), `scripts/serve_upvla.py` (:8300), `scripts/serve_xvla.py` (:8100)
- 벤치마크 평가: `scripts/robocasa_vla_eval.py`, `scripts/calvin_eval.py`
- 학습: `scripts/train_dreamvla_robocasa.py` + `.sh`
- Feature 추출: `scripts/extract_sam_robocasa.py`, `scripts/extract_cotrack_robocasa.py`
- Processor (추론용): `src/processor/` — `base.py`, `types.py`, `factory.py`, `obs/`, `action/`
- Dataset (학습용): `src/datasets/adapters/dreamvla.py` (adapter, LeRobotDataset 직접 사용)
- 경로 설정: `scripts/path_setup.py`
- 모델 소스 (submodule): `src/policies/dreamvla/`, `src/policies/UP-VLA/`
- 벤치마크 소스 (submodule): `src/benchmarks/robocasa/`, `src/benchmarks/robosuite/`, `src/benchmarks/calvin/`, `lerobot/`

## 개발 컨벤션

- **브랜치**: `feat/`, `fix/`, `exp/`, `refactor/` 접두사. `dev`에서 분기, PR은 `dev`로.
- **커밋 메시지**: 한글. `feat:`, `fix:`, `refactor:`, `docs:`, `config:`, `script:` 접두사.
- **Python**: robocasa=3.11, calvin=3.8, 나머지=3.10. calvin 관련 코드는 3.8 호환 필수.
- **체크포인트**: git에 커밋하지 않음. `.gitignore`에 포함.

## Processor Pipeline (추론용)

`src/processor/` 참고. 추론 시 데이터 흐름:

```
env obs → ObsProcessor (통일 키 변환) → VLAClient (HTTP) → 모델 서버 → VLAClient → ActionProcessor (env 포맷) → env.step
```

## Dataset + Adapter (학습용)

`src/datasets/` 참고. 학습 시 데이터 흐름:

```
LeRobotDataset (v3.0) → Model Adapter (차원 변환 + SAM/track feature 로딩 + collator) → 학습 루프
```

- **Model Adapter** (`src/datasets/adapters/<model>.py`): 모델별 1개. LeRobotDataset을 직접 wrapping. state/action 변환 + collator.
- DreamVLA adapter는 SAM feature, CoTracker trajectory label도 로딩 지원 (`sam_features_path`, `track_label_path`).

## Feature 추출 (학습 보조 데이터)

SAM/CoTracker feature는 학습 전에 오프라인으로 추출:

```
LeRobotDataset → extract_sam_robocasa.py → {save_path}/rgb_static/training/{frame_idx}.pt
LeRobotDataset → extract_cotrack_robocasa.py → {save_path}/rgb_static/training/{frame_idx}.npz
```

- SAM: `segment-anything` ViT-B encoder → avg_pool → `[C, 256]` per frame.
- CoTracker: frame pair (frame_gap=5) → optical flow delta → `{tracks: [784, 2], visibility: [784]}` per frame.
- 체크포인트: `src/policies/dreamvla/segment-anything/ckpts/`, `src/policies/dreamvla/co-tracker/checkpoints/`.

## 확장 가이드

### 새 모델 추가

1. `docker/`에 Dockerfile, `docker-compose.yml`에 서비스 추가
2. `scripts/serve_<model>.py` 작성 (통일 API: `/act`, `/reset`, `/health`)
3. (학습) `src/datasets/adapters/<model>.py` — adapter + factory

### 새 벤치마크 추가

1. (평가) `src/processor/obs/`, `action/`에 ProcessorStep 구현 + `factory.py`에 등록
2. (학습) `src/datasets/<benchmark>_lerobot.py` — generic dataset
3. `scripts/<benchmark>_eval.py` 작성

## 주의사항

- serve 스크립트는 Docker 컨테이너 내에서 실행됨. 경로는 `/temporal_vla/...` 기준.
- action 차원은 모델마다 다를 수 있음 (7, 14 등). 벤치마크에서 환경에 맞게 매핑.
