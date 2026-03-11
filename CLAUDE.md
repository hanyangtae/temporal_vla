# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 참고하는 지침입니다.

## 프로젝트 개요

VLA(Vision-Language-Action) 모델을 RoboCasa/Calvin 시뮬레이션 환경에서 평가하는 프로젝트.
Docker 컨테이너로 모델(DreamVLA, X-VLA, UP-VLA)과 벤치마크(RoboCasa, Calvin)를 분리하고,
통일 HTTP API로 통신한다.

## 핵심 아키텍처

- **모델 서버** (`scripts/serve_*.py`): FastAPI + uvicorn. 통일 API 엔드포인트 `/act`, `/reset`, `/health`.
- **벤치마크 스크립트** (`scripts/robocasa_vla_eval.py`, `scripts/calvin_eval.py`): 모델 무관. `VLAClient`로 통신.
- **통일 클라이언트** (`scripts/utils/vla_client.py`): `VLAClient` 클래스 1개. 이미지는 base64 PNG, 응답 actions는 항상 2D.
- **Docker**: `docker-compose.yml`에 5개 서비스. 모두 `network_mode: host` (localhost 통신).

## 통일 API 규격

```
POST /act
  요청: {"images": {"static": b64png, "wrist": b64png}, "state": [...], "instruction": "..."}
  응답: {"actions": [[float...], ...], "latency_ms": float}
POST /reset  ← 히스토리 초기화 (필요 없으면 no-op)
GET  /health ← {"status": "ok"|"not_loaded", "model": "..."}
```

## 주요 파일 경로

- 모델 서버: `scripts/serve_dreamvla.py` (:8200), `scripts/serve_upvla.py` (:8300), `scripts/serve_xvla.py` (:8100)
- 벤치마크: `scripts/robocasa_vla_eval.py` (RoboCasa), `scripts/calvin_eval.py` (Calvin)
- 클라이언트: `scripts/utils/vla_client.py`
- Docker: `docker-compose.yml`, `docker/` 디렉토리
- 모델 소스 (git submodule): `src/policies/dreamvla/`, `src/policies/UP-VLA/`
- 벤치마크 소스 (git submodule): `robocasa/`, `robosuite/`, `src/benchmarks/calvin/`, `lerobot/`

## 개발 컨벤션

- **브랜치**: `feat/`, `fix/`, `exp/`, `refactor/` 접두사. `dev` 브랜치에서 분기, PR은 `dev`로.
- **커밋 메시지**: 한글. `feat:`, `fix:`, `refactor:`, `docs:`, `config:`, `script:` 접두사 사용.
- **Python**: 컨테이너마다 버전이 다름 (robocasa=3.11, calvin=3.8, 나머지=3.10). calvin 관련 코드는 3.8 호환 필수.
- **체크포인트**: git에 커밋하지 않음. `.gitignore`에 포함.
- **git submodule**: `robosuite`, `robocasa`, `lerobot`, `src/policies/dreamvla`, `src/benchmarks/calvin`, `src/policies/UP-VLA`

## 새 모델 추가 시

1. `docker/` 에 Dockerfile 추가
2. `docker-compose.yml`에 서비스 추가
3. `scripts/serve_<model>.py` 작성 (통일 API 준수: `/act`, `/reset`, `/health`)
4. 기존 벤치마크 스크립트에서 `--vla-server` URL만 바꾸면 평가 가능

## 새 벤치마크 추가 시

1. `scripts/<benchmark>_eval.py` 작성
2. `VLAClient`(`scripts/utils/vla_client.py`)를 사용하여 모델 서버와 통신
3. 환경 obs → `{"static": img, "wrist": img}` 매핑, 모델 action → 환경 action 매핑은 벤치마크 스크립트에서 처리

## 주의사항

- serve 스크립트는 Docker 컨테이너 내에서 실행됨. 경로는 `/temporal_vla/...` 기준.
- 이미지 전송은 base64 PNG. numpy list 직접 전송은 사용하지 않음.
- action 차원은 모델마다 다를 수 있음 (7, 14 등). 벤치마크에서 환경에 맞게 매핑.
