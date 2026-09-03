# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 참고하는 지침입니다.
**원칙: 이 파일에는 코드에서 유도할 수 없는 사실과 단일 출처 포인터만 둔다. 절차·수치의
정본은 각 docs/skill 파일이며, 여기 요약이 정본과 어긋나면 정본이 이긴다.**

## Agent 운영 규칙

- `.agents/agent_spec.md` = agent 작업 방식·검증·git/PR 절차의 단일 기준 (repo-local).
- `AGENTS.md`는 Codex 계열 entrypoint. 지침 충돌 시 더 구체적 작업 문맥 우선.
- Codex(GPT) 협업은 `.claude/skills/codex-collab/SKILL.md` 규약. review-lane 호출은
  반드시 `scripts/utils/codex_ask.sh` 경유.

## 프로젝트 개요

VLA 모델의 **latent steering** 연구. VLA 잠재공간에서 실패/성공 latent를 구분하고 추론 시
활성화를 성공 쪽으로 steer해 SR을 올린다 (백본 추가학습 없음). 메인 method =
**pathway-resolved(VL/DiT 분리) + phase-matched steering**, 핵심 난제 = **online phase/failure-type 식별**.

방향·가설·검증 설계·현재 판정의 단일 출처: [`docs/steering/RESEARCH_DIRECTION.md`](docs/steering/RESEARCH_DIRECTION.md)
(라운드별 실측은 `docs/steering/RESULTS.md`). 요지만: 분리(C1)는 부분 확립, 조종(C2)은
위약 대비 부정 우세(read ≠ write), 유형(C3)·라우팅(C4)이 열린 축. **길이·instruction·scene
confound 통제 없는 분리 주장 금지** (실패=항상 timeout이라 time-pooled 분리는 아티팩트).

## 평가 표준

정본: `.claude/skills/robocasa-steer-eval/SKILL.md` (launcher·pre-flight 게이트 포함).
핵심 고정값: **EVAL_SEED=100000**, N_ENVS=2·N_EP=20 per condition, per-episode tsv 필수,
fit-seed와 eval-seed 분리(held-out) 필수. **GPU 서버(kanu·srv48·srv50) 운영·세션 간 예약
규약 정본: [`docs/05_gpu_server_rules.md`](docs/05_gpu_server_rules.md)** — 발사 전
`scripts/utils/gpu_lease.sh claim` 필수, 타 세션이 잡고 있으면 대기 또는 사용자 보고.
요지: 빈 GPU만(타인 프로세스 있으면 금지)·kanu 최대 3장·kanu serve 2/GPU·srv 6/GPU.

## 핵심 아키텍처

- **모델 서버** (`scripts/serve/*.py`): FastAPI **HTTP**, 통일 API (`/act`, `/reset`, `/health`).
  - ⚠ 예외: N1.6 SAFE 수집·eval 주 경로 `scripts/safe/groot_n16/robocasa/serve/feature_server.py`는
    GR00T upstream `PolicyServer`(**ZMQ** + msgpack) — 통일 HTTP API를 타지 않는다.
- **벤치마크 평가** (`scripts/eval/*.py`): 모델 무관, `VLAClient`로 통신.
- **Processor Pipeline** (`src/processor/`): 추론용 env↔통일API 변환.
  단, GR00T `--use-groot-env` 평가와 SAFE wiring은 이를 우회하고
  `src/policies/groot/robocasa/io.py`(native key 유지, upstream parity)를 쓴다.
- **Dataset + Adapter** (`src/datasets/`): 학습용. generic LeRobot dataset + 모델별 adapter
  (`src/datasets/adapters/<model>.py`).
- **통일 클라이언트**: `scripts/utils/vla_client.py`의 `VLAClient` 하나 (`predict`, `predict_with_features`).
- **Docker**: `docker-compose.yml` 8개 서비스, 모두 `network_mode: host`.

## 통일 API 규격

단일 출처: [`docs/01_serving_interface.md`](docs/01_serving_interface.md) (endpoint,
sub-key 네임스페이스, `/act_with_features`, 모델×벤치마크 매트릭스). `/health` 필수 필드와
표준 action sub-key도 해당 문서 기준.

## 주요 파일 경로

- 모델 서버 (HTTP): `scripts/serve/lerobot.py` (pi0/pi0.5·N1.5, steering 배선), `scripts/serve/groot.py`,
  `scripts/serve/openvla_oft.py` (:8400), `scripts/serve/upvla.py` (:8300), `scripts/serve/xvla.py` (:8100)
- 모델 서버 (ZMQ): `scripts/safe/groot_n16/robocasa/serve/feature_server.py`
- steering hook: `scripts/serve/steering_hooks.py` (+ `safe_hooks.py`, `patching_hooks.py`, `attn_hooks.py`)
- 벤치마크 평가: `scripts/eval/robocasa_eval.py`, `scripts/eval/calvin.py`
- 실험 파이프라인: `scripts/safe/groot_n15/robocasa/` (exp2~5 본류), `scripts/safe/groot_n16/robocasa/`
- SAE: `scripts/scene_sae/`, `src/sae/` · 분석: `scripts/analysis/`
- 레포 검토·정리: `docs/review/` (`LEDGER.tsv`), `scripts/review/ledger_ui.py`
- 경로 설정: `scripts/path_setup.py`
- 모델 소스 (submodule): `src/policies/UP-VLA/`, `src/policies/Isaac-GR00T/` (fork: do-dong-park, EVAL_SEED 커밋 포함)
- 벤치마크 소스 (submodule): `src/benchmarks/robocasa/`, `robosuite/`, `calvin/`, `lerobot/`
- 출력: `outputs/` · 추론 결과: `outputs/eval/{benchmark}/{model}/{yymmddhhmmss}/`

## ★ Activation·연산자 저장 규약

**rollout activation·steering 연산자(conceptor/setM)를 저장·이관·삭제하는 모든 작업은
[`docs/04_data_storage_convention.md`](docs/04_data_storage_convention.md)를 먼저 읽고 따른다.**
요지: 식별자는 내용 지문 `sig`/`opsig` (경로 아님), 산출물 안 절대경로 기록 금지, 수집
rollout(pkl 有)과 평가 rollout(pkl 無) 저장 위치 분리, 캡처 밀도 5열은 문서 §4·§6.
grid 좌표는 **scene(주방)·jitter j·noise 3축 폴더층** `s<sid>/j<jid>/n<nid>` (2026-09-03 v6) — 규약은 문서 §3.1.1, 격자·수집 계약은 `docs/collab_within_claude/handoff_20260903_grid_v6_scene_jitter.md`.

## 체크포인트·데이터셋 경로 (cache)

체크포인트·데이터셋은 repo 밖 `~/.cache/temporal_vla/` (`checkpoints/`, `datasets/`).
컨테이너에는 `/cache`로 bind-mount (`VLA_CACHE_ROOT=/cache`). 학습 산출물(파인튜닝 ckpt,
rollout)은 `outputs/`에 그대로.

경로 참조는 단일 소스만 사용 (하드코딩 금지):
- Python: `from scripts.path_setup import CHECKPOINTS_ROOT, DATA_ROOT`
- Shell: `source "${REPO_ROOT}/scripts/utils/cache_env.sh"` 후 `${VLA_CHECKPOINTS_ROOT}` / `${VLA_DATASETS_ROOT}`.
  컨테이너 전용 스크립트는 `/cache/...` 리터럴 가능. `configs/checkpoints/*.yaml`도 `/cache/...` 기준.

## 원격 compute 노드

대용량 rollout은 원격 노드에 있고, CPU·numpy 분석/fit은 원격에서 돌려 소용량 결과만 회수한다.
정본: `.claude/agents/remote-compute.md` (env 제약 포함) + 헬퍼 `scripts/utils/remote_compute.sh`.
코드 동기화는 git 브랜치 (scp 금지). SR eval·수집은 로컬 전용.

## 개발 컨벤션

- **브랜치**: `feat/`, `fix/`, `exp/`, `refactor/` 접두사. `dev`에서 분기, PR은 `dev`로.
- **커밋 메시지**: 한글. `feat:`, `fix:`, `refactor:`, `docs:`, `config:`, `script:` 접두사.
- **Python**: robocasa=3.11, calvin=3.8, 나머지=3.10. calvin 관련 코드는 3.8 호환 필수.
- **체크포인트**: git에 커밋하지 않음.

## 확장 가이드

- 새 모델: Dockerfile + compose 서비스 → `scripts/serve/<model>.py` (통일 API) → (학습) adapter.
- 새 체크포인트: `configs/checkpoints/*.yaml` 프로파일 → serve 분기 → 검증. 절차 정본:
  [`docs/03_adding_checkpoint.md`](docs/03_adding_checkpoint.md) (스키마: `configs/checkpoints/README.md`).
- 새 벤치마크: `src/processor/` ProcessorStep + factory 등록 → (학습) `src/datasets/<benchmark>_lerobot.py`
  → `scripts/eval/<benchmark>.py`.

## 주의사항

- serve 스크립트는 Docker 컨테이너 내 실행: repo 코드는 `/temporal_vla/...`, 체크포인트·데이터셋은 `/cache/...`.
- action 차원은 모델마다 다름 (7, 14 등). 벤치마크에서 환경에 맞게 매핑.
