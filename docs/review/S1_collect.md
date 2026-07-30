# S1 — 수집 (rollout + activation 캡처)

> 스테이지 카드. **판정 열은 사용자가 채운다** (UI: `python3 scripts/review/ledger_ui.py`).
> 작성일 2026-07-30 · 기준 커밋 `5f320ab` · 기계 판독분 = `S1_files.tsv`(36건)

## 0. 대상

| 위치 | 파일 | LOC |
|---|---|---|
| `scripts/safe/groot_n15/robocasa/collect/` | 24 | 5,091 |
| `scripts/safe/groot_n16/robocasa/collect/` | 12 | 3,070 |
| **합계** | **36** | **8,161** |

두 계열이 **전송 방식으로 갈린다**: n15 = HTTP(`scripts/serve/lerobot.py`), n16 = ZMQ
(`feature_server.py`). 같은 "수집"이지만 코드를 공유하지 않는다.

## 1. 기계 검사 결과 (`scripts/review/dup_scan.py`)

### 1.1 함수/클래스 완전 중복 — **0군**

8줄 이상 def/class 중 본문 해시가 일치하는 것이 **파일 간에 하나도 없다.** 복사-붙여넣기로
같은 로직이 흩어져 있을 것이라는 가정은 이 스테이지에서는 틀렸다.

### 1.2 파일 쌍 유사도 — 4쌍

| ratio | A | B |
|---|---|---|
| **0.85** | `n16/collect_task_set_in_container.sh` (86) | `n16/collect_task_set_official_uv_host.sh` (92) |
| 0.66 | `n16/collect_task_set_in_container.sh` | `n16/collect_task_set_via_docker_exec.sh` (122) |
| 0.65 | `n16/collect_task_set_official_uv_host.sh` | `n16/collect_task_set_via_docker_exec.sh` |
| 0.57 | `n15/collect_fit_6phase.sh` (62) | `n15/collect_strict_cells.sh` (60) |

**`collect_task_set_*.sh` 3종이 서로 0.65~0.85** — 같은 수집 절차의 실행 환경 변형
(컨테이너 내부 / 호스트 uv env / docker exec)이다. 셋 다 05-27~29 작성으로 **이 스테이지에서
가장 오래된 파일**이고 **셋 다 고아**다.

### 1.3 고아 파일 — 13개 (36개 중 36%)

아무 코드에서도 import·호출되지 않는다. 다만 셸 러너는 사람이 직접 실행하는 것이라
"고아 = 죽었다"가 아니다. **판정은 "그 라운드가 끝났는지"로 해야 한다.**

| 성격 | 파일 |
|---|---|
| n16 수집 러너 3종 (최고령·상호 유사) | `collect_task_set_{in_container,official_uv_host,via_docker_exec}.sh` |
| 라운드 전용 수집 러너 | `collect_phase_event_4cell.sh` · `collect_bread_strict.sh` · `collect_potato_apple_seed2.sh` · `collect_strict_cells.sh` |
| 운영 스크립트 | `watch_phase_event_cleanup.sh` · `archive_on_done.sh` |
| 일회성 진단 | `_diag_wrong_grasp.py` · `repro_check.py` |
| 기타 | `classify_instructions.py` · `phase_live_rollout.py` · `phase_render_from_real.py` |

## 2. 읽기 순서 (`읽기01`~`읽기07`)

허브부터 의존을 타고 내려가는 순서. TSV 행 순서와 같다.

| # | 파일 | 줄 | 왜 |
|---|---|---|---|
| 1 | `n15/http_feature_collect.py` | **1261** | exp2~exp5 전 라운드가 이걸로 수집했다. 단일 최대 파일이고 S1의 실질 본체 |
| 2 | `n16/collect_rollout.py` | 276 | n16(ZMQ) 계열 진입점 |
| 3 | `n16/collect_env.py` | 200 | env 생성 + 1 episode 실행. **EVAL_SEED·`gym.make(seed=)` 규약이 여기** |
| 4 | `n16/collect_policy_clients.py` | 282 | HTTP/ZMQ transport |
| 5 | `n16/collect_artifacts.py` | 179 | pkl/mp4/사이드카 기록 |
| 6 | `n16/collect_schema.py` | 75 | 산출물 스키마 |
| 7 | `n16/verify_rollout_collection.py` | 528 | 수집 검증 게이트 |

## 3. 스테이지 경계 문제 — 두 파일이 S1에 있으나 S2 소속

- **`n16/collect/robocasa_event_labeler.py` (1014줄)** — phase 분할 본체다. 위치는 `collect/`인데
  역할은 S2(라벨링)다. 게다가 `TASK_EVENTS` 등록부라서 **미등록 task는 수집이 KeyError로 죽는다**
  ([`../steering/PITFALLS.md`](../steering/PITFALLS.md) §8). 수집과 라벨링이 코드 수준에서 결합돼 있다.
- **`n15/collect/env_step_phase.py` (115줄)** — env-step 해상도 phase GT.

두 파일은 S2 카드에서 함께 판정하는 것이 맞다. 여기서는 `S2소속` 플래그만 달았다.

## 4. 확인이 필요한 중복 의심 (기계로는 안 잡힌 것)

유사도는 낮지만 **역할이 겹칠 가능성**이 있는 쌍이다. 기계 탐지가 0군이라 이건 사람이 봐야 한다.

1. **`http_feature_collect.py`(1261) ↔ `collect_instruction_fixed_http_features.py`(268)**
   — 둘 다 N1.5 HTTP 수집이다. 후자가 전자의 manifest 기반 래퍼인지, 별도 경로인지 확인 필요.
2. **`verify_rollout_collection.py`(528) ↔ [`DATA_HANDLING.md`](../steering/DATA_HANDLING.md) §1**
   — 문서에 적은 보존 검증 절차(실물 개수·용량·평균 크기)가 이 스크립트에 이미 있는지,
   아니면 문서와 코드가 서로 다른 검증을 하는지.
3. **`archive_on_done.sh`(32)** — 원격 HDD rsync 백업을 한다. **DATA_HANDLING §2의 규약
   (`-L`/상대 심링크)을 지키는지 확인해야 한다.** exp2 fit 유실 사고가 정확히 이 지점에서 났다.

## 5. instruction 계열 6개 — 묶어서 판정

`select_instruction_seeds`(573) · `launch_instruction_seed_shards`(257) ·
`merge_instruction_seed_shards`(166) · `materialize_selected_ep_meta`(154) ·
`backfill_instruction_ep_meta`(182) · `classify_instructions`(127) = **1,459줄**.

instruction-fixed 라운드(구 `11_phase4`, archive됨)의 도구 세트다. 라운드는 종결됐지만
**seed마다 task variant가 바뀌는 문제는 여전히 유효**하므로(`select_instruction_seeds`의
존재 이유), 도구를 버릴지 남길지는 "앞으로 instruction을 고정한 수집을 또 할 것인가"에 달렸다.

## 6. 살아있는 것 — 섭동 계열 3개

`perturbation.py`(524) · `build_perturb_grid.py`(323) · `collect_perturb_grid.sh`(183) = 1,030줄.
exp5-2의 **유일한 위약-분리 양성**이 나온 무대([`../steering/RESULTS.md`](../steering/RESULTS.md) §1)라
현재 가장 살아있는 경로다.

## 7. 판정 시 갈리는 지점

1. **`collect_task_set_*.sh` 3종** — 0.85 유사에 셋 다 고아. 하나로 합칠지, 실제로 쓰는 하나만
   남길지, 셋 다 archive할지. n16 수집을 앞으로 할 것인가에 달렸다.
2. **`http_feature_collect.py` 1261줄** — S1의 본체인데 단일 파일로 크다. 쪼갤 가치가 있는지는
   내부를 읽어봐야 한다(n16은 collect_env/artifacts/schema/clients로 나뉘어 있는데 n15는 한 덩어리).
3. **라운드 전용 러너 4개** — 전부 07-06 작성, 전부 고아. 재수집 재현성 때문에 남길지.
4. **시각화 3개**(`phase_live_rollout`·`phase_live_render`·`phase_render_from_real`) — S8로 이동 후보.
