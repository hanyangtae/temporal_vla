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

- **모델 서버** (`scripts/serve/*.py`): FastAPI + uvicorn. 통일 API (`/act`, `/reset`, `/health`).
- **벤치마크 평가** (`scripts/eval/*.py`): 모델 무관. `VLAClient`로 통신.
- **Processor Pipeline** (`src/processor/`): **추론(eval)용**. 벤치마크별 env↔통일API obs/action 변환.
- **Dataset + Adapter** (`src/datasets/`): **학습(train)용**. 벤치마크별 generic dataset + 모델별 adapter.
- **통일 클라이언트** (`scripts/utils/vla_client.py`): `VLAClient` 클래스 1개.
- **Docker**: `docker-compose.yml`에 5개 서비스. 모두 `network_mode: host`.

## 통일 API 규격

`scripts/utils/vla_client.py` 참고. 핵심 규칙:

### Observation (요청)
- state sub-keys는 벤치마크마다 존재하는 키가 다름. 모델 서버는 필요한 키만 꺼내 쓰고, 없으면 변환(quat→euler 등)을 자체 수행.
- 이미지는 base64 PNG.
- 예: `observation.images.static`, `observation.state.eef_pos`, `observation.state.eef_euler`, `task`

### Action (응답) — Sub-key 방식
- 모델 서버는 자신의 native format을 **action sub-key**로 분리하여 반환.
- 벤치마크 측 ActionProcessor가 자신에게 필요한 포맷으로 변환.
- action sub-key 값은 항상 2D list `[n_steps, dim]`.

```
// relative 모델 (DreamVLA, UP-VLA 등)
{"action.eef_pos": [[dx,dy,dz],...], "action.eef_euler": [[r,p,y],...], "action.gripper": [[g],...]}

// absolute 모델 (X-VLA 등)
{"action.eef_pos": [[x,y,z],...], "action.eef_rot6d": [[6D],...], "action.gripper": [[g],...]}
```

표준 action sub-key: `action.eef_pos`, `action.eef_euler`, `action.eef_rot6d`, `action.eef_quat`, `action.gripper`, `action.joint_pos`

### /health 응답
- `action_type`: `"relative"` | `"absolute"` — 모든 서버 필수.
- `action_keys`: 서버가 반환하는 action sub-key 목록.
- `n_action_steps`: 예측당 반환 action 수.

## 주요 파일 경로

- 모델 서버: `scripts/serve/dreamvla.py` (:8200), `scripts/serve/upvla.py` (:8300), `scripts/serve/xvla.py` (:8100)
- 벤치마크 평가: `scripts/eval/robocasa_eval.py`, `scripts/eval/calvin.py`
- 학습: `scripts/train/dreamvla_robocasa.py` + `.sh`
- 분석: `scripts/analysis/` — 루프 패턴 분석, CLIP 캐시, trajectory 수집 등
- Feature 추출: `scripts/extract/extract_sam_robocasa.py`, `scripts/extract/extract_cotrack_robocasa.py`
- Processor (추론용): `src/processor/` — `base.py`, `types.py`, `factory.py`, `obs/`, `action/`
- Dataset (학습용): `src/datasets/adapters/dreamvla.py` (adapter, LeRobotDataset 직접 사용)
- 경로 설정: `scripts/path_setup.py`
- 모델 소스 (submodule): `src/policies/dreamvla/`, `src/policies/UP-VLA/`
- 벤치마크 소스 (submodule): `src/benchmarks/robocasa/`, `src/benchmarks/robosuite/`, `src/benchmarks/calvin/`, `lerobot/`
-출력: `outputs`
-모델 추론 결과 출력: `outputs/eval/{benchmark}/{model}/{yymmddhhmmss}/`

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
LeRobotDataset → scripts/extract/extract_sam_robocasa.py → {save_path}/rgb_static/training/{frame_idx}.pt
LeRobotDataset → scripts/extract/extract_cotrack_robocasa.py → {save_path}/rgb_static/training/{frame_idx}.npz
```

- SAM: `segment-anything` ViT-B encoder → avg_pool → `[C, 256]` per frame.
- CoTracker: frame pair (frame_gap=5) → optical flow delta → `{tracks: [784, 2], visibility: [784]}` per frame.
- 체크포인트: `src/policies/dreamvla/segment-anything/ckpts/`, `src/policies/dreamvla/co-tracker/checkpoints/`.

## 확장 가이드

### 새 모델 추가

1. `docker/`에 Dockerfile, `docker-compose.yml`에 서비스 추가
2. `scripts/serve/<model>.py` 작성 (통일 API: `/act`, `/reset`, `/health`)
3. (학습) `src/datasets/adapters/<model>.py` — adapter + factory

### 새 체크포인트 추가

기존 모델 아키텍처에 새 체크포인트를 얹거나, 같은 체크포인트의 벤치마크 변형을 추가할 때.

1. `configs/checkpoints/<base_model>__<variant>.yaml` 프로파일 작성 (스키마: `configs/checkpoints/README.md`).
2. 해당 serve 스크립트에 프로파일 기반 분기 추가 (기존 하드코딩을 프로파일 필드로 치환).
3. `python scripts/utils/checkpoint_profile.py configs/checkpoints/<name>.yaml` 로 로드 검증.
4. eval 스크립트에서 serve 기동 시 `--profile` 경로 지정.

상세 절차: `docs/adding_checkpoint.md`. 반복 작업 시 `vla-checkpoint-manager` 에이전트 호출.

### 새 벤치마크 추가

1. (평가) `src/processor/obs/`, `action/`에 ProcessorStep 구현 + `factory.py`에 등록
2. (학습) `src/datasets/<benchmark>_lerobot.py` — generic dataset
3. `scripts/eval/<benchmark>.py` 작성

## 주의사항

- serve 스크립트는 Docker 컨테이너 내에서 실행됨. 경로는 `/temporal_vla/...` 기준.
- action 차원은 모델마다 다를 수 있음 (7, 14 등). 벤치마크에서 환경에 맞게 매핑.
