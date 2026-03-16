# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 참고하는 지침입니다.

## 프로젝트 개요

VLA(Vision-Language-Action) 모델을 RoboCasa/Calvin 시뮬레이션 환경에서 평가하는 프로젝트.
Docker 컨테이너로 모델(DreamVLA, X-VLA, UP-VLA)과 벤치마크(RoboCasa, Calvin)를 분리하고,
통일 HTTP API로 통신한다.

## 핵심 아키텍처

- **모델 서버** (`scripts/serve_*.py`): FastAPI + uvicorn. 통일 API 엔드포인트 `/act`, `/reset`, `/health`.
- **벤치마크 스크립트** (`scripts/robocasa_vla_eval.py`, `scripts/calvin_eval.py`): 모델 무관. `VLAClient`로 통신.
- **Processor Pipeline** (`src/processor/`): LeRobot `ProcessorStep` 인터페이스 호환. 벤치마크별 obs/action 변환을 파이프라인으로 분리.
- **통일 클라이언트** (`scripts/utils/vla_client.py`): `VLAClient` 클래스 1개. 이미지는 base64 PNG, 응답 action은 항상 2D. LeRobot 키 네이밍 컨벤션 사용.
- **Docker**: `docker-compose.yml`에 5개 서비스. 모두 `network_mode: host` (localhost 통신).

## 통일 API 규격 (LeRobot 컨벤션)

```
POST /act
  요청: {
    "observation.images.static": b64png,
    "observation.images.wrist": b64png,
    "observation.state": [...],
    "task": "..."
  }
  응답: {"action": [[float...], ...], "latency_ms": float}
POST /reset  ← 히스토리 초기화 (필요 없으면 no-op)
GET  /health ← {"status": "ok"|"not_loaded", "model": "...",
                "input_features": {...}, "output_features": {...}, "n_action_steps": int}
```

## 주요 파일 경로

- 모델 서버: `scripts/serve_dreamvla.py` (:8200), `scripts/serve_upvla.py` (:8300), `scripts/serve_xvla.py` (:8100)
- 벤치마크: `scripts/robocasa_vla_eval.py` (RoboCasa), `scripts/calvin_eval.py` (Calvin)
- 클라이언트: `scripts/utils/vla_client.py`
- Processor: `src/processor/` (base, types, factory, obs/, action/)
- Docker: `docker-compose.yml`, `docker/` 디렉토리
- 모델 소스 (git submodule): `src/policies/dreamvla/`, `src/policies/UP-VLA/`
- 경로 설정: `scripts/path_setup.py` (PYTHONPATH 헬퍼, 스크립트 상단에서 import)
- 벤치마크 소스 (git submodule): `src/benchmarks/robocasa/`, `src/benchmarks/robosuite/`, `src/benchmarks/calvin/`, `lerobot/`

## 개발 컨벤션

- **브랜치**: `feat/`, `fix/`, `exp/`, `refactor/` 접두사. `dev` 브랜치에서 분기, PR은 `dev`로.
- **커밋 메시지**: 한글. `feat:`, `fix:`, `refactor:`, `docs:`, `config:`, `script:` 접두사 사용.
- **Python**: 컨테이너마다 버전이 다름 (robocasa=3.11, calvin=3.8, 나머지=3.10). calvin 관련 코드는 3.8 호환 필수.
- **체크포인트**: git에 커밋하지 않음. `.gitignore`에 포함.
- **git submodule**: `src/benchmarks/robosuite`, `src/benchmarks/robocasa`, `lerobot`, `src/policies/dreamvla`, `src/benchmarks/calvin`, `src/policies/UP-VLA`

## 새 모델 추가 시

1. `docker/` 에 Dockerfile 추가
2. `docker-compose.yml`에 서비스 추가
3. `scripts/serve_<model>.py` 작성 (통일 API 준수: `/act`, `/reset`, `/health`)
4. 기존 벤치마크 스크립트에서 `--vla-server` URL만 바꾸면 평가 가능

## Processor Pipeline (벤치마크별 obs/action 변환)

LeRobot의 `ProcessorStep`/`DataProcessorPipeline` 인터페이스를 따르는 경량 자체 구현.
numpy 기반, Python 3.8 호환. 벤치마크별 obs/action 변환을 선언적으로 분리한다.

```
env.step(action)
  → env obs (env-specific dict)
  → ObsProcessor (키 리매핑 + 포맷 통일)
    → 통일 키: observation.images.*, observation.state
  → VLAClient (base64 인코딩 + HTTP 전송)
  → 모델 서버 (모델별 전처리 + 추론)
  → VLAClient (action 수신)
  → ActionProcessor (통일 action → env-specific action)
env.step(action)
```

- `ObsProcessor`: env obs dict → 통일 키 (`observation.images.*`, `observation.state`) 변환. 이미지 전처리(resize, normalize 등)는 모델 서버 내부에서 수행.
- `ActionProcessor`: 모델 출력 action → env에 맞는 차원/포맷 변환.
- `DataProcessorPipeline`: step 체인 + feature 추적 + save/load 지원.
- `factory.py`: `make_calvin_processors()`, `make_robocasa_processors()` 등 벤치마크별 팩토리.

## 새 벤치마크 추가 시

1. `src/processor/obs/<benchmark>.py` — `ObservationProcessorStep` 구현 (env obs → 통일 키)
2. `src/processor/action/<benchmark>.py` — `ActionProcessorStep` 구현 (통일 action → env action)
3. `src/processor/factory.py`에 `make_<benchmark>_processors()` 추가
4. `scripts/<benchmark>_eval.py` 작성, `VLAClient` + processor pipeline 사용

## 주의사항

- serve 스크립트는 Docker 컨테이너 내에서 실행됨. 경로는 `/temporal_vla/...` 기준.
- 이미지 전송은 base64 PNG. numpy list 직접 전송은 사용하지 않음.
- action 차원은 모델마다 다를 수 있음 (7, 14 등). 벤치마크에서 환경에 맞게 매핑.
