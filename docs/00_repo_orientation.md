# 레포 오리엔테이션 가이드 — 코드 지도 · 상태 점검 · 탐색법

> 작성 2026-06-24. 이 레포(`temporal_vla`)에 다시 들어왔을 때 **무엇이 무슨 역할인지**,
> **현재 구조가 정리돼 있는지**, **모르는 부분을 어떻게 추적하는지**를 한 곳에서 잡기 위한
> 문서. 연구 서사·runbook은 다루지 않는다(그건 [`docs/README.md`](README.md) →
> `steering/`·`groot/`). 본 문서는 **코드 관점 지도 + 위생 점검 + 탐색 절차**다.
> 사실은 작성 시점에 `git ls-files`/`grep`/소스 직접 확인으로 검증함.

---

## 0. 30초 요약 (먼저 읽기)

레포는 머릿속에서 **두 덩어리**로 나누면 길을 잃지 않는다.

1. **인프라 (모델-벤치마크 통일 서빙·평가)** — 어떤 VLA든 같은 HTTP `/act` 계약으로 띄우고
   같은 evaluator에 붙인다. 단일 출처: [`docs/01_serving_interface.md`](01_serving_interface.md).
   코드: `scripts/serve/`, `scripts/eval/`, `src/processor/`, `scripts/utils/vla_client.py`.
2. **연구 (latent steering 파이프라인)** — 성공/실패 활성화 수집 → split → conceptor fit →
   steer/eval → 분석/시각화. 단일 출처: [`docs/steering/14_pathway_phase_online_steering.md`](steering/14_pathway_phase_online_steering.md).
   코드: `scripts/safe/groot_n16/robocasa/**`, `src/conceptor/`, `src/policies/groot/safe/`.

**처음 4개 문서만 순서대로 읽으면 전체 그림이 잡힌다**:
[`docs/README.md`](README.md) → [`CLAUDE.md`](../CLAUDE.md) →
[`steering/14`](steering/14_pathway_phase_online_steering.md)(연구 방향) →
[`01_serving_interface.md`](01_serving_interface.md)(인프라 계약).

---

## 1. 코드 지도 — 무엇이 무슨 역할인가

### 1.1 최상위 레이아웃

| 경로 | 역할 | 비고 |
|---|---|---|
| `scripts/` | 사람이 직접 실행하는 진입점 전부 (serve/eval/train/safe/utils) | 아래 1.4 |
| `src/` | 우리가 작성한 라이브러리 코드 (conceptor/processor/datasets/policies/utils) | 아래 1.3 |
| `src/benchmarks/`, `src/policies/<model>/`, `lerobot/` | **upstream submodule** (robocasa, robosuite, calvin, Isaac-GR00T, UP-VLA, openvla-oft) | 내부는 우리 코드 아님 |
| `docs/` | 실행 절차·연구 결정·결과·논문 reference. entrypoint = `docs/README.md` | well-maintained |
| `configs/` | `checkpoints/`(체크포인트 프로파일 YAML), `policies/`, `robocasa/`(task tsv) | 새 ckpt 추가 시 |
| `docker/`, `docker-compose.yml` | 서비스별 컨테이너 정의 (현재 **7개 서비스**) | 아래 1.2 |
| `outputs/` | eval/rollout/detector 산출물 (git ignore) | `outputs/eval/{bench}/{model}/{ts}/` |
| `data/` | (사실상 빈 껍데기) 데이터는 `~/.cache/temporal_vla/`로 이전됨 | 2.2 참고 |
| `tests/`, `scripts/test/`, `conftest.py` | 단위/스모크 테스트 | |
| `.agents/agent_spec.md`, `AGENTS.md` | agent 운영 규칙·entrypoint | CLAUDE.md가 가리킴 |

### 1.2 통일 서빙 인프라 (모델 ↔ 벤치마크)

같은 HTTP 계약(`/act`, `/act_with_features`, `/reset`, `/health`)으로 어떤 모델이든
어떤 벤치마크에든 붙인다. 벤치마크는 `--vla-server` URL만 바꾸면 모델이 교체된다.

| 서버 (`scripts/serve/`) | 모델 | 포트 |
|---|---|---|
| `xvla.py` | X-VLA | 8100 |
| `upvla.py` | UP-VLA | 8300 |
| `openvla_oft.py` | OpenVLA-OFT | 8400 |
| `lerobot.py` | pi0/pi05 (LeRobot stack), GR00T N1.5 | 8400 |
| `groot.py` | GR00T N1.6 (+ SAFE feature) | 8500 |

데이터 흐름: `env obs → src/processor (통일 키 변환) → VLAClient(HTTP) → 서버 → VLAClient →
src/processor (env 포맷) → env.step`. **단, GR00T `--use-groot-env` 경로는 예외** — generic
processor를 우회하고 `src/policies/groot/robocasa/io.py`(`GrootRoboCasaEnv`)를 써서 upstream
parity와 SAFE wiring을 맞춘다.

### 1.3 `src/` — 우리가 작성한 코드

| 패키지 | 역할 | 상태 |
|---|---|---|
| `src/conceptor/` | **메인 method 수학**. COAST 계열 contrastive conceptor (`C_steer = C_success ∧ ¬C_failure`) + steering 게이트 `h' = h·Mᵀ`. → 1.5 | LIVE |
| `src/processor/` | **추론용** generic obs/action 변환 파이프라인 (`base/types/factory` + `obs/`, `action/`). 벤치마크별 ProcessorStep. | LIVE |
| `src/datasets/` | **학습용** LeRobot dataset 확장 슬롯 — 현재 구체 구현 없음(phase1/TTT 로더 제거). `adapters/` 계획만. | 스캐폴드 |
| `src/policies/groot/` | GR00T wrapper: `core/`(loader·service·schema·preprocess, serve/groot.py가 사용), `robocasa/`(native IO adapter), `safe/`(DiT feature capture/직렬화). | LIVE |
| `src/policies/safe_capture.py`, `safe_metadata.py` | cross-policy SAFE hook lifecycle + feature 메타데이터 naming 계약 | LIVE |
| `src/utils/common/` | 공유 인프라: `logger.py`, `image.py`(b64↔numpy), `feature_blob.py`, `serving.py`(서버 부트스트랩) | LIVE |
### 1.4 `scripts/` — 진입점

| 디렉터리 | 역할 | 상태 |
|---|---|---|
| `serve/` | 모델 추론 서버 (1.2). `lerobot_adapters/` 서브디렉터리 포함 | LIVE |
| `eval/` | 벤치마크 평가 (모델 무관). `robocasa_eval.py`(메인), `calvin.py`, `libero.py`, `groot_robocasa_zmq_eval.py`, phase1 predictor eval | LIVE |
| `train/` | 학습. `launch_finetune.py`(GR00T N1.6 baseline fine-tune, multi-path mixture) + `.sh` 래퍼 | LIVE |
| `utils/` | **공유 인프라**: `vla_client.py`(VLAClient), `checkpoint_profile.py`, `cache_env.sh`, `remote_compute.sh`, `path_setup.py`(repo root) | LIVE |
| `safe/` | **연구 파이프라인 본체** (131 파일). → 1.4.1 | LIVE(n16) / LEGACY(n15) |
| `analysis/`, `data/` | 일회성/저빈도: CLIP 캐시·dataset 디버그·v2.1 merge. 산출물 캐시됨 (`extract/`=TTT Stage0, 제거됨) | DORMANT || `deprecated/` | **비어있음** (COAST-faithful 스크립트 삭제 흔적) | EMPTY |

#### 1.4.1 `scripts/safe/` 해부 (131 파일 — 가장 큰 미스터리)

버전(n15/n16)이 아니라 **기능 단계**로 읽어야 한다. 메인은 `groot_n16/robocasa/`.

```
safe/
├── _common/         split_lib.py(rollout 파싱), feature_value_summary.py  ← 공유, LIVE
├── groot_n16/robocasa/   ← 메인 파이프라인 (단계 순서대로)
│   ├── collect/     rollout 수집 (collect_rollout.py + task_sets.sh + 컨테이너 런처)
│   ├── split/       train/val/test 분할 (prepare_seen4_unseen2_split.py 가 active)
│   ├── steer/       conceptor fit + steering eval (fit_conceptor_steering.py, eval_steer_compare.sh)
│   ├── analyze/     검출기·표현 분석 (pathway_online_detection.py, pathway_lstm_detector.py,
│   │                pathway_step_attribution.py, pathway_separation.py 등 ~22)
│   ├── train/       SAFE LSTM 검출기 학습 sweep (정책 학습 아님)
│   ├── vis/         시각화 프레임워크 (core/ 라이브러리 + analyses/ 드라이버 + run_*.py, ~45)
│   └── serve/       feature_server.py (SAFE feature 전용 ZMQ, ADR-0002)
├── groot_n15/robocasa/   ← N1.5 instruction-fixed 참조/parity 트랙 (LEGACY, 26 파일)
└── lerobot/         멀티벤치 SAFE 보조 (2 파일)
```

**파이프라인 한 줄**: `collect → split → steer(fit conceptor) → analyze(검출기/분리) → vis`.
사람이 직접 돌리는 건 각 단계의 `*.sh` 런처와 `fit_conceptor_steering.py`,
`eval_steer_compare.sh`, `run_feature_visualization.py`. 나머지는 import되는 헬퍼다.

### 1.5 conceptor (메인 method 코드 — `src/conceptor/`)

연구 문서 [`steering/14`](steering/14_pathway_phase_online_steering.md)의 수식이 여기 산다.
자세한 설명은 `src/conceptor/README.md`.

- `core.py` — `compute_correlation`, `compute_conceptor`(R(R+α⁻²I)⁻¹), boolean
  `not/and/or_conceptor`, `contrastive_conceptor`(= C_success ∧ ¬C_failure).
- `steering.py` — `build_steering_matrix`(β 게이트), `apply_steering`(h' = h·Mᵀ).
- `analysis.py` — `conceptor_quota`, `conceptor_overlap`, `eigenvalue_spectrum`,
  `failure_containment` (진단 지표).

수학(generic numpy)과 **pathway 선택(VL vs DiT)·phase-bin 적용은 호출자 책임**이다 — conceptor
모듈 자체에는 pathway 개념이 없고, `scripts/safe/.../steer/`·`scripts/serve/`의 hook이 어느
활성화에 적용할지 정한다.

### 1.6 live vs legacy 한눈에

- **LIVE 메인**: serve/eval/utils 인프라, `src/processor`·`conceptor`·`policies/groot`,
  `safe/groot_n16/robocasa/**`.
- **LEGACY(보존)**: `safe/groot_n15/`(N1.5 parity 참조), `analysis/`·`data/`(일회성, 산출물
  캐시됨). TTT 파이프라인(`src/ttt/`·phase1·`extract/`) 제거 — 설계만 `docs/ttt/README.md`.
- **EMPTY/스텁**: `scripts/deprecated/`, `src/datasets/adapters/`.

---

## 2. 현재 상태 점검 — "최적인가"

결론: 기능은 동작하지만 **위생·문서 정합성에 정리 빚이 쌓여 있다**. 아래는 실제 확인한 것만.

### 2.1 Git 위생 (검증됨)

| 발견 | 근거 | 심각도 | 권고 |
|---|---|---|---|
| **23MB `랩미팅0623.pptx`가 git 추적됨** | `git ls-files`에 존재, `.gitignore`에 `*.pptx` 규칙 없음 | 높음 | `git rm --cached` + `.gitignore`에 `*.pptx`(또는 `*.pptx`/`*.pdf` 산출물 규칙). 이력에 영구 잔존하므로 빠를수록 좋음 |
| **`CLAUDE.md`가 .gitignore에 있는데도 추적됨** | `.gitignore:21`에 `CLAUDE.md`, 그러나 `git ls-files CLAUDE.md` 매치 | 중간 | CLAUDE.md는 팀 공유(체크인)가 맞으므로 **.gitignore에서 `CLAUDE.md` 라인 제거**(오해 유발). AGENTS.md도 추적 중 |
| **`src/benchmarks/calvin/`가 submodule인데 .gitignore에도 있음** | `.gitmodules`에 calvin 등록 + `.gitignore:47`에 `src/benchmarks/calvin/` | 중간 | submodule 경로를 ignore하는 건 모순 → submodule 상태 꼬임 원인 가능. `git submodule status` 확인 후 ignore 라인 정리 |
| `LIBERO`는 의도적 local checkout(ignore됨) | `.gitignore:46` `src/benchmarks/LIBERO/`, `.gitmodules` 미등재 | 낮음 | "local-only 벤치"임을 문서화하면 충분 (떠돌이 아님) |
| `data/`가 root 소유 + 빈 `huggingface/`만 | `ls -la data/`→root:root, `git ls-files data/`=∅(ignore됨) | 낮음 | `chown` 후 빈 디렉터리 삭제 가능 (캐시로 이전 완료) |
| `__pycache__`/`temp/`/`logs/`/`nohup.out` 클러터 | 디스크엔 있으나 **모두 git 미추적**(`.gitignore` 커버) | 낮음 | 추적 문제 없음. 디스크 정리만 선택적 |
| GR00T submodule 2개가 같은 URL | `Isaac-GR00T`(n1.5-release-34-g…) vs `Isaac-GR00T-N1.5`(n1.5-release) | 낮음 | 의도(N1.6 작업본 vs N1.5 고정)면 `.gitmodules`에 ref/용도 주석. `Isaac-GR00T-N1.5`는 현재 Dockerfile modified 상태 → commit/discard 결정 |

### 2.2 구조·문서 정합성

| 발견 | 근거 | 심각도 | 권고 |
|---|---|---|---|
| 최상위 `README.md`의 "Project Structure"가 stale | docker/ 하위를 robocasa·xvla 둘만 나열(실제 7), `scripts/eval/robocasa.py`(실제 `robocasa_eval.py`), "3개 컨테이너"(실제 7), 빈 "Training" 섹션 | 중간 | README는 설치/소개로 축소하고 구조 상세는 본 문서·`docs/README.md`로 위임 |
| `groot_n15` 서비스 표기가 compose에 없음 | README/CLAUDE의 컨테이너 표엔 `groot_n15`, `docker-compose.yml` 서비스엔 없음 | 낮음 | 표 갱신 또는 override 파일 여부 확인 |
| `scripts/safe/`에 README 없음 | 131 파일인데 단계 설명 문서 부재 | 중간 | `scripts/safe/README.md`(또는 본 문서 1.4.1로 링크)로 collect→…→vis 단계 1장 정리 |
| `SAFE_GROOT_N16_DATA_BUNDLE_README.md`가 최상위 | 루트에 단독 위치 | 낮음 | `docs/groot/` 아래로 이동 검토 |

### 2.3 정리 우선순위 (권고 순서)

1. **(높음)** pptx git에서 제거 + `.gitignore` 규칙 추가.
2. **(중간)** `.gitignore` 모순 정리: `CLAUDE.md` 라인 제거, submodule 경로(calvin) ignore 검토.
3. **(중간)** `scripts/safe/README.md` 추가(본 문서 1.4.1 재사용) — 131 파일 미스터리 해소.
4. **(중간)** 최상위 `README.md` stale 섹션 정정/축소.
5. **(낮음)** `data/` 소유권·빈 디렉터리, GR00T submodule ref 주석, 루트 단독 md 이동.

> 위는 **권고**다. 실제 적용은 별도 작업으로 진행한다(이 문서는 진단까지).

---

## 3. 어떻게 살펴봐야 하나 — 탐색 플레이북

### 3.1 진입 순서 (문서 → 코드)

1. [`docs/README.md`](README.md) — 문서 전체 지도·reading order.
2. [`CLAUDE.md`](../CLAUDE.md) — 프로젝트 개요·연구 방향·경로 규칙·평가 표준.
3. 작업이 **연구**면 [`steering/14`](steering/14_pathway_phase_online_steering.md)·
   [`steering/15`](steering/15_research_structure.md), **인프라**면
   [`01_serving_interface.md`](01_serving_interface.md).
4. GR00T 흐름은 [`groot/00_groot_flow_map.md`](groot/00_groot_flow_map.md)(call chain).

### 3.2 코드 경로 추적법

- **진입점부터**: 사람이 실행하는 건 `scripts/`의 `*.py main`·`*.sh`. 거기서 import를 따라
  `src/`로 내려간다.
- **경로 해석은 헬퍼 한 곳**: Python은 `from scripts.path_setup import CHECKPOINTS_ROOT,
  DATA_ROOT`, shell은 `source scripts/utils/cache_env.sh`. 하드코딩 경로를 찾지 말 것.
- **누가 쓰는지**: `grep -rn "from src.<pkg>" scripts/ src/`로 호출처를 센다(live 판별과 동일).

### 3.3 live / dead 판별법

한 모듈이 살아있는지는 세 가지로 본다: (1) `grep`로 import 참조가 있나, (2) `docs/`·
`CLAUDE.md`가 언급하나, (3) `scripts/deprecated/`나 `_legacy/`에 있나. 셋 다 음성이면 dead
후보. (주의: 문서 라벨과 실제 wiring을 함께 볼 것 — 이번 정리에서 `src/ttt/`를 제거할 때 문서상
demoted였지만 baseline finetune launcher가 실제로 의존하고 있어 먼저 분리해야 했다.)

### 3.4 단일 출처(SSOT) 맵 — "이건 어디가 기준?"

| 주제 | 단일 출처 |
|---|---|
| 통일 HTTP API 계약 | `docs/01_serving_interface.md` |
| 메인 연구 방향·가설·ablation | `docs/steering/14_pathway_phase_online_steering.md` |
| 경로(체크포인트/데이터셋) | `scripts/path_setup.py` / `scripts/utils/cache_env.sh` |
| 평가 표준(seed·GPU·로깅) | `CLAUDE.md` "평가 표준" 섹션 (EVAL_SEED=100000) |
| conceptor 수학 | `src/conceptor/` (+ `README.md`) |
| SAFE feature 메타데이터 계약 | `src/policies/safe_metadata.py` |
| 원격 compute 워크플로우 | `scripts/utils/remote_compute.sh`, CLAUDE.md "원격 compute 노드" |

### 3.5 실행 환경 주의

- **serve 스크립트는 Docker 안에서** 실행 (repo 경로 `/temporal_vla/...`, 캐시 `/cache/...`).
  로컬 host에서의 import 실패는 정적 신호로만 취급.
- **robocasa eval/수집은 로컬 전용**, 대용량 rollout **분석·conceptor fit은 원격 노드**
  (`~/anaconda3/bin/python`, torch 보유; base python3엔 torch/scipy 없음).
- Python 버전 분기: robocasa 3.11 / calvin 3.8 / 그 외 3.10.

---

### 갱신 규칙

이 문서는 **코드 구조가 바뀌면** 갱신한다(연구 결과는 `steering/`에 둔다). 사실은 추측하지 말고
`git ls-files`/`grep`/소스로 재확인한 것만 적는다.
