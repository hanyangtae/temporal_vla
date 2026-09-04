# 원장 — 2026-09-02 grid 데이터 전량 폐기 (껍데기 보존)

## 왜 지웠나

사용자 판정: **replay 시 데이터와 수집 시 데이터가 다르다** (v4r 재수집 라운드에서 수집
라벨과 replay 라벨의 반전 59% 실측 — `docs/collab_within_claude/handoff_20260902_v4r_round.md`).
원인 미규명. 기존 activation 으로 fit 한 연산자·분석물은 replay 세계와 어긋나므로 전부
폐기하고, **아래 시나리오에 맞춰 같은 파이프라인으로 처음부터 다시 모은다.** 연산자는
activation 이 있으면 재생성 가능하므로 보존 가치가 없다(docs/04 §1).

## 시나리오 (재수집의 목적)

VLA 는 작업장마다 finetune 이 필요한데, 작업장에 **약간의 변화**가 생길 때마다 finetune
하기엔 데이터·주기 부담이 크다. → 같은 작업장에서 변화로 SR 이 떨어졌을 때 finetune 보다
적은 데이터로 **activation 기반 감지 → steering** 으로 회복을 시도한다.

전제 데이터: ① finetune 에 쓰인 expert 데이터, ② 과거 같은 scene 의 rollout(물건 배치만
약간 다름), ③ 현재 scene 의 실패 rollout(구제 가능한 case 만). unseen scene 은 대상 외.

| 축 | 값 |
|---|---|
| instruction | 10 (v2 와 동일) |
| scene | 5 (v2 s0–4, base env_seed 는 plan 참조) |
| noise (denoise seed) | 5 (v2 n0–4 = 1300000–1300004) |
| **물체 재배치 k** | **5 — 전부 신규**, base 재사용 없음 (ep_meta 고정+연속 reset, docs/04 §3.1.1) |
| 합계 | **1,250판 (instruction 당 125)** |

## 무엇을 남겼나 (이 디렉토리)

| 파일 | 내용 |
|---|---|
| `rollouts_all_3282.tsv` | 폐기 직전 아카이브 전체 인덱스 — 행 = rollout, 열에 plan_id·machine·instruction·scene_idx·noise_idx·**env_seed·inference_seed**·success·steps·ckpt·capture_layers. "어떤 scene 을 어떤 instruction 으로 어떤 seed 로 어디서 모았고 결과가 뭐였나"의 전부 |
| `index_rollouts_v4_1250.tsv` | v4 k-grid 뷰(지터 1,000 + base 250, `cell_si`·`jitter_reset_idx`) |
| `index_rollouts_v1_dedup_820.tsv` | v1 dedup 인덱스(마시멜로 kanu 정본화 후) |
| `grid_meta_json_3282.tgz` | 셀별 `meta.json` 전부(좌표·seed·k·캡처밀도 5열·sig). 아카이브에도 **제자리 보존** |
| `ep_meta/<task>/<env>--seed<es>.json` | v4 지터 재현용 ep_meta 50개 (task 10 × scene 5) |
| `kscan_v4/*.tsv` | k-스캔 원본 50개(scene당 N=12, `k \t instruction`) — v5 채택 k 5개 산출 근거, 50/50 scene 충족 확인 |
| (레포) `configs/collect/n15_grid_v{1,2,3,4}/collection_plan.json` | plan 정본 — instruction 별 scene seed 목록·noise seed·ckpt·capture_layers·denoise_k·token_mode |
| (레포) `configs/collect/n15_grid_v{3,4}/kscan_adopted.json` | k-스캔 채택/기각 k (drawer 는 방향 재추첨 때문에 필터 필수) |

수집 절차(어떻게 모았나)는 `docs/04` §5.1–5.2, 좌표 규약은 §3.1·§3.1.1,
N1.5 러너 = `scripts/safe/groot_n15/robocasa/collect/collect_grid.sh`
(플래그: n_action_steps 5, max 720, capture_layers 0,2,4,8,10,12,15, all_token_full, denoise_k 4).

## 무엇을 지웠나 (승준 HDD `temporal_vla_store/groot/n15/`)

| 대상 | 용량 | 처리 |
|---|---|---|
| `grid/*/rollout.pkl` 3,282 · `video.mp4` 3,282 · `traj.csv` 3,282 | **1,287.8GB** (pkl 1,284.3 + mp4 3.5) | 삭제 완료 — `meta.json` 3,282·`ep_meta/` 50 제자리 유지 확인 |
| `analysis/` (grid_phase·grid_phase_v2·grid_phase_v4: 연산자 npz·segA shard·v4r 진단 캡처, 1,280 파일) | **438.0GB** | 전부 삭제 완료 |
| `runs/` | 19GB | **유지** (legacy 집계·로그, 수집 데이터 아님) |
| `index/` | 소량 | 유지 |

삭제 실행 원장: `DELETED_20260902.tsv` (경로·개수·바이트). 삭제 후 HDD 여유 **1.7TB**.
재수집 계약·착수 순서: `docs/collab_within_claude/handoff_20260902_grid_recollect_v5.md` §0.

## 2026-09-04 legacy 삭제 원장 (승준 `groot/n15/grid/`)

사용자 지시("이전 데이터는 지워"). 분석 정본은 v6(plan a81f07b86371, 주방 목록이 달라 v5 와 섞이지 않음). 사전 공지 후 action phase·포크 세션 이의 없음 확인. 레포의 v5 기록(`n15_grid_v5_scenario/{collection_plan.json,index_rollouts_v5.tsv,kscan_adopted.json}`)과 이 원장은 유지.

| 삭제 시각(KST) | 디렉토리 | 바이트 | meta.json | rollout.pkl | 비고 |
|---|---|---|---|---|---|
| 2026-09-04 13:46:57 | e82e99cb666b | 480,657,109,721 | 1,250 | 1,250 | v5 정본(k-층 재배치·coffee/drawer_right 교체 완료본) |
| 2026-09-04 13:47:22 | 8daefeabf020, e6b316053d1c | 8,643 | 0 | 0 | v5 plan_id 이력 README 만 |
| 2026-09-04 13:47:22 | 3134e339de4c, 46ea62d53e09, 8ae74723a29e, 979d4833a7db, b8054b5e7258 | 33,091,712 | 3,282 | 0 | v1~v4 meta 껍데기(내용은 이 디렉토리의 `grid_meta_json_3282.tgz` 에 보존) |

삭제 후 잔여: `a81f07b86371/`(v6, 895셀+ep_meta) · `375e3c46c962/README.txt`(v6 plan_id 이력). 승준 여유 909G → 1.4T.
