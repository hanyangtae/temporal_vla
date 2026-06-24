# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 참고하는 지침입니다.

## Agent 운영 규칙

Claude Code 세션에서는 `.agents/agent_spec.md`를 repo-local 운영 규칙으로 한 번 확인하고 따른다.

- 이 파일의 나머지 섹션은 프로젝트 구조, 개발 컨벤션, 실행 경로를 설명한다.
- `.agents/agent_spec.md`는 agent의 작업 방식, 검증 기준, 문서화 방식, git/PR 절차의 단일 기준이다.
- `AGENTS.md`는 Codex 계열 agent entrypoint다. Claude Code의 instruction discovery는 `.agents/agent_spec.md`에서 멈추고, `AGENTS.md`는 참고 문맥으로 사용한다.
- 두 문서의 지침이 충돌하면 더 구체적인 작업 문맥과 현재 repo-local instruction을 우선한다.

## 프로젝트 개요

VLA 모델의 **latent steering**을 연구하는 프로젝트.
VLA 잠재공간에서 **실패 latent와 성공 latent를 구분**하고, 추론 시 활성화를 성공 쪽으로
**steer**하여 Success Rate를 올리는 것이 목표 (VLA 백본 추가학습 없음). 현재 main stream은
**pathway(VL/DiT) 분리 + phase-matched steering**, 핵심 난제는 **online phase/failure-type
식별** (아래 "연구 방향").

배경 동기: 성공 데이터로만 학습된 VLA가 실패 시 같은 trajectory를 반복하는 문제
(이전 "실패 루프 탈출" 프레이밍). 현재는 이를 latent steering 관점에서 접근한다.

인프라: Docker 컨테이너로 모델과 벤치마크(RoboCasa, Calvin)를 분리하고, 통일 HTTP API로 통신.

## 연구 방향

메인 method = **pathway-resolved + phase-matched activation steering** (백본 재학습 없음).
succ/fail 활성화 분포에서 contrastive conceptor `C_steer = C_success ∧ ¬C_failure` 를 fit하고
추론 시 `h' = h·Mᵀ` 로 성공 부분공간 쪽으로 steer (COAST 계열, 단일벡터 additive 아니라
multi-dim contrastive 연산자). 여기에 두 축을 더한다. 단일 출처:
[`docs/steering/14_pathway_phase_online_steering.md`](docs/steering/14_pathway_phase_online_steering.md).

- **(1) Pathway 분리 steering**: VL(goal "what")과 DiT(motor "how") pathway를 **각각 따로** steer.
  근거 = NOTALL 기능 분리 + 우리 데이터(VL 이른 신호 t≤8, DiT 늦은 신호 t≥12). 주의:
  Eagle→VL-SA→DiT 직렬이라 "따로"가 진짜 독립이 아님(downstream 결합 고려).
- **(2) Phase-matched DiT steering**: DiT는 **rollout phase(시간)에 조건부로** steer. COAST는 전
  timestep을 한 공간(클래스별 R=E[hhᵀ])에 pool해 길이/phase confound를 섞음 — 우리는 phase별
  성공 분포를 타깃.
- **★ 중심 미해결 문제 (타당성 판정)**: 추론 중(**online**)에 어느 pathway가 실패했고
  (goal vs motor) 어느 task-phase인지 식별 가능한가? 안 되면 steering을 라우팅할 수 없다 →
  다음 실험의 첫 질문은 "steer가 듣나"보다 "phase/type을 온라인에 읽을 수 있나".
- **검증**: 사다리식 ablation (global → pathway-split → +phase-bin), 각 단계 ΔSR 비교 — 이전
  단계가 신호를 보일 때만 복잡도 추가(약한 신호에서 noise fit 방지).
- **표현 분석**: succ/fail latent 분리 검증 (SAFE식 feature-space). **길이 confound 통제 필수**
  — 실패는 항상 timeout이라 시간-pooled 분리는 아티팩트.
- **TTA (VITA 기반 progress predictor)**: 메인 아님(구 "loop escape" 방향). 단 phase-matched
  steering이 online phase/progress 신호를 요구하므로 **보조 부품(phase 신호 공급원)으로 복귀 검토 가능**.
- **Baseline 모델**: pi0, groot. **Metric**: SR 상승, steering 후 ΔSR 인과 재측정 (EVAL_SEED=100000 표준).

## 평가 표준 (2026-06-05 확정)

이후 모든 robocasa eval 은 아래 표준으로 통일한다 (조건 간 baseline noise 제거 → 같은
condition pair 끼리 ΔSR 비교 가능).

- **고정 EVAL_SEED = 100000** (= 동료(do-dong-park) collection seed 표준,
  `scripts/safe/groot_n16/robocasa/collect/task_sets.sh` 의 `ROBOCASA_SEED_START_FOR_TASK_SET`).
  `eval_steer_compare.sh` default, `EVAL_SEED` 환경변수로 override.
  - `gym.make(env_name, seed=EVAL_SEED + env_idx)` 로 robocasa kitchen `env.rng` 고정.
  - `env.reset(seed=[EVAL_SEED, EVAL_SEED+1, ...])` 로 첫 reset 결정적.
  - 같은 (env, EVAL_SEED) → 같은 episode 시리즈 (layout/style/object/instruction).
  - 동료 collection (seed 100000..100099) 의 episode 와 매칭 → eval/collection 일관성 유지.
- **GPU 양보 default**: `GPUS="4 5 6 4 5 6"` (GPU 0-3 동료용). 3 GPU × 2 server = 6 worker.
- **per-episode logging**: `groot_robocasa_zmq_eval.py` 가 video-dir/per_episode.tsv 출력
  (`episode_idx`, `success`, `language` — instruction variant 별 SR 분석용).
- **N_ENVS=2, N_EP=20 per condition** — wall-time 과 binomial noise 의 균형점.

## 핵심 아키텍처

- **모델 서버** (`scripts/serve/*.py`): FastAPI + uvicorn. 통일 API (`/act`, `/reset`, `/health`).
- **벤치마크 평가** (`scripts/eval/*.py`): 모델 무관. `VLAClient`로 통신.
- **Processor Pipeline** (`src/processor/`): **추론(eval)용**. generic 벤치마크 env↔통일API obs/action 변환.
- **GR00T RoboCasa IO adapter** (`src/policies/groot/robocasa/io.py`): `GrootRoboCasaEnv` native key↔HTTP GR00T 변환. GR00T upstream parity / SAFE wiring 경로는 `src/processor/`를 우회.
- **Dataset + Adapter** (`src/datasets/`): **학습(train)용**. 벤치마크별 generic dataset + 모델별 adapter.
- **통일 클라이언트** (`scripts/utils/vla_client.py`): `VLAClient` 클래스 1개.
- **Docker**: `docker-compose.yml`에 5개 서비스. 모두 `network_mode: host`.

## 통일 API 규격

통일 HTTP API (endpoint, sub-key 네임스페이스, `/act_with_features`, 모델 × 벤치마크 호환 매트릭스, 운영 패턴) 의 단일 출처는 [`docs/01_serving_interface.md`](docs/01_serving_interface.md). 아래는 자주 쓰는 사실만 요약.

- 클라이언트는 `scripts/utils/vla_client.py` 의 `VLAClient` 한 개. `predict()` 와 `predict_with_features()`.
- 요청: `observation.images.<view>` (base64 PNG), `observation.state.<key>` (float list), `task` (str).
- 응답: `action.<subkey>` (2D list `[n_steps, dim]`) + `latency_ms`. `/act_with_features` 는 추가로 `features.hidden_states` (feature blob), `features.kind/axes/...` 노출.
- 표준 action sub-key: `action.eef_pos`, `action.eef_euler` | `action.eef_rot6d` | `action.eef_quat` | `action.eef_axisangle`, `action.gripper`, `action.joint_pos`. GR00T 전용: `action.base_motion`, `action.control_mode`.
- `/health` 필수 필드: `status`, `action_type` (`"relative"` | `"absolute"`), `action_keys`, `n_action_steps`. `/act_with_features` 지원 시 `supports_features`, `feature_kind`, `feature_axes` 도 노출.

## 주요 파일 경로

- 모델 서버: `scripts/serve/upvla.py` (:8300), `scripts/serve/xvla.py` (:8100)
- 벤치마크 평가: `scripts/eval/robocasa_eval.py`, `scripts/eval/calvin.py`
- 학습: `scripts/train/launch_finetune.py` (GR00T N1.6 baseline fine-tune) + `.sh`
- 분석: `scripts/analysis/` — CLIP 캐시, dataset 디버그 등
- Processor (추론용): `src/processor/` — `base.py`, `types.py`, `factory.py`, `obs/`, `action/`
- Dataset (학습용): `src/datasets/` (generic LeRobot dataset + adapter)
- 경로 설정: `scripts/path_setup.py`
- 모델 소스 (submodule): `src/policies/UP-VLA/`, `src/policies/Isaac-GR00T/`
- 벤치마크 소스 (submodule): `src/benchmarks/robocasa/`, `src/benchmarks/robosuite/`, `src/benchmarks/calvin/`, `lerobot/`
-출력: `outputs`
-모델 추론 결과 출력: `outputs/eval/{benchmark}/{model}/{yymmddhhmmss}/`

## 체크포인트·데이터셋 경로 (cache)

체크포인트와 데이터셋은 repo 트리 밖 cache 에 둔다 (git 추적 안 함).

- 호스트: `~/.cache/temporal_vla/` 아래 `checkpoints/`, `datasets/`.
- 컨테이너: docker-compose 가 위 cache 를 `/cache` 로 bind-mount + `VLA_CACHE_ROOT=/cache` 주입 → `/cache/checkpoints/...`, `/cache/datasets/...`.
- 베이스 모델: `checkpoints/nvidia/GR00T-N1.6-3B`. 데이터셋: `datasets/robocasa/...` 등 (구 repo `data/` 내용을 그대로 옮김).
- 학습 산출물(파인튜닝 ckpt, rollout 등)은 이동 대상이 아니라 `outputs/` 에 그대로 둔다.

경로 참조 규칙 (단일 소스 — 하드코딩 금지):
- Python: `from scripts.path_setup import CHECKPOINTS_ROOT, DATA_ROOT` (repo root 가 sys.path 에 있을 때).
- Shell: `source "${REPO_ROOT}/scripts/utils/cache_env.sh"` 후 `${VLA_CHECKPOINTS_ROOT}` / `${VLA_DATASETS_ROOT}`. 컨테이너 전용 스크립트는 `/cache/...` 리터럴 사용 가능.
- `configs/checkpoints/*.yaml` 의 로컬 체크포인트 경로도 `/cache/...` 기준.

## 원격 compute 노드

대용량 rollout(`raw_rollouts`)은 원격 노드에 쌓여 있고, 분석·conceptor fit 같은 순수 CPU·numpy
작업은 데이터가 있는 원격에서 돌리고 결과(NPZ/plot/JSON, 소용량)만 회수한다(34GB 재전송 회피).
코드 동기화는 git 브랜치로 한다(scp 금지). 단일 출처 헬퍼: `scripts/utils/remote_compute.sh`.

- 워크플로우: 로컬 브랜치 작업 → `sync-code`(push + 원격 checkout) → 원격에서 `run`/`run-bg` →
  `pull-results`로 산출물만 회수. 로컬 데이터를 올릴 땐 `push-data`.
- 원격 노드 (기본값, env로 override): `kimseungjun@166.104.146.37:11112`, repo `~/workspace/temporal_vla`.
- 전용 오케스트레이터 agent: **`remote-compute`** (`.claude/agents/remote-compute.md`).
- **원격 env 제약**: rollout pkl 은 torch 텐서를 담아 unpickle 에 torch 필요 →
  분석·fit 은 **`~/anaconda3/bin/python`** (torch+numpy+matplotlib 보유)으로 돌린다
  (helper `REMOTE_PYTHON` 기본값). base `python3` 는 torch 없음. **scipy 는 어느 python 에도 없음.**
  무거운 numpy 는 `OMP/OPENBLAS_NUM_THREADS` cap(공유 노드).
- SR eval(robocasa Docker)·수집은 원격이 아니라 **로컬 전용**. 원격은 분석·fit 까지만.

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

GR00T `--use-groot-env` 평가와 SAFE wiring은 위 generic pipeline이 아니라
`src/policies/groot/robocasa/io.py`를 사용한다. 이 경로는 `GrootRoboCasaEnv`
native observation/action key를 유지해 upstream GR00T eval과 맞춘다.

## Dataset + Adapter (학습용)

`src/datasets/` 참고. 학습 시 데이터 흐름:

```
LeRobotDataset (v3.0) → Model Adapter (차원 변환 + collator) → 학습 루프
```

- **Model Adapter** (`src/datasets/adapters/<model>.py`): 모델별 1개. LeRobotDataset을 직접 wrapping. state/action 변환 + collator.

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

상세 절차: `docs/03_adding_checkpoint.md`. 반복 작업도 해당 체크리스트를 기준으로 수행하고, 별도 에이전트 호출은 사용자가 명시적으로 요청한 경우에만 수행한다.

### 새 벤치마크 추가

1. (평가) `src/processor/obs/`, `action/`에 ProcessorStep 구현 + `factory.py`에 등록
2. (학습) `src/datasets/<benchmark>_lerobot.py` — generic dataset
3. `scripts/eval/<benchmark>.py` 작성

## 주의사항

- serve 스크립트는 Docker 컨테이너 내에서 실행됨. repo 코드 경로는 `/temporal_vla/...`, 체크포인트·데이터셋은 `/cache/...` 기준.
- action 차원은 모델마다 다를 수 있음 (7, 14 등). 벤치마크에서 환경에 맞게 매핑.
