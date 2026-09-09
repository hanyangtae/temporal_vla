# 핸드오프 — grid 데이터 수집 (2026-09-04, 세션 "데이터 추가 수집")

이전 판: `history/handoff_20260902_grid_recollect_v5.md` (v5 계약·N1.6 HTTP 절차·함정 목록 — 그대로 유효한 것은 여기 §6 에 요약).
v6 **설계 계약**의 정본은 `handoff_20260903_grid_v6_scene_jitter.md`(포크 세션 작성) + `docs/04_data_storage_convention.md` §3.1.1.
이 문서는 **현재 데이터 상태·미결 작업·운영 규칙**을 넘긴다. 읽는 순서: §1 → §2 → §3.

---

## 1. 현재 상태 — v6 라운드 완결 (2026-09-05 02:15 KST)

- **아카이브 정본**: 승준 `temporal_vla_store/groot/n15/grid/08f1c9df8207/` — 1,800/1,800 셀(12키 × scene 3 × j 5 × noise 10), pkl 1,800, 결손 0. 인덱스 `configs/collect/n15_grid_v6_scene_jitter/index_rollouts_v6.tsv`(1,800행, `--strict` 통과). 구 루트 `77e745c37b0f/`·`a81f07b86371/`·`375e3c46c962/` 는 README.txt 만 남은 껍데기.
- **QA 완료**: VL 상수화 1,800/1,800 OK(`qa_vl_check.tsv`), 영상 FREEZE 0 / NOISE 50(`qa_video_scan.tsv`) — NOISE 는 전부 `PPCC/marshmallow s1`(흑백 타일 바닥 오탐). 그 scene 은 75/75 전패지만 관측·VL·활성화(action phase 3중 확인) 정상 = **초기조건형 전패 scene**(스폰이 캐비닛 벽을 향해 목표가 시야 밖). 인덱스에서 빼지 않는다; 성공 0 이라 연산자·detector 셀로는 못 쓴다.
- **키 개정 적용됨(2026-09-05 rebase)**: oven/washer left/right = 로봇 기준 target fixture 쪽. 매핑(구 plan 77e745c37b0f → 신 08f1c9df8207; pkl·sig·s/j/n 불변, meta `grid_instruction`·`side` 패치 + `key_renamed_from`):

  | 구 키 (77e745c37b0f) | 신 키 (08f1c9df8207) |
  |---|---|
  | OvenRack/out-left | OvenRack/out-right |
  | OvenRack/out-right | OvenRack/out-left |
  | DishwasherRack/out-left | DishwasherRack/out-right |
  | DishwasherRack/out-right | DishwasherRack/out-left |

  drawer/PPCC/coffee 불변. 지문 동결본(action phase): 승준 `~/workspace/temporal_vla/outputs/analysis/grid_phase_v6_frozen/prerebase_{ovenwasher,ALL}_FINAL.tsv`.
- **SR(1,800)**: 전체 0.537 — apple 1.00 / bread .81 / oven-right(신) .77 / candle .72 / drawer-right .68 / marshmallow .63 / oven-left(신) .60 / drawer-left .55 / dish-left(신) .49 / jug .15 / dish-right(신) .03 / coffee .01.
- 남은 일: 없음(GPU lease 0, 3머신 staging 비움). 재수집이 필요해지면 §3·§5 절차 그대로.

## 2. 데이터 상태 (2026-09-04 기준)

| 데이터 | plan_id | 셀 | 위치(승준 HDD `temporal_vla_store/groot/n15/grid/`) | 상태 |
|---|---|---|---|---|
| **v6** (정본) | `08f1c9df8207` (← 77e745c37b0f ← a81f07b86371 ← 375e3c46c962 ← b3dbe412d190; n10 확대 → out-left/right 키 의미 개정) | 1,800/1,800 | `08f1c9df8207/{kanu,worker1,worker2}/<key>/s<sid>/j<jid>/n<nid>/base/` + `ep_meta/` | QA 완료, 인덱스 1,800행 |
| v5 (legacy) | `e82e99cb666b` (← e6b316053d1c ← 8daefeabf020) | 1,250 | `e82e99cb666b/…/s<i>/k<r>/n<j>/base/` | QA 완료(무효 10 교체됨), 인덱스 `configs/collect/n15_grid_v5_scenario/index_rollouts_v5.tsv` |
| v1~v4 껍데기 | 3134…, 46ea…, 8ae7…, 979d…, b805… | meta.json 만 | 동일 루트 | 원장 `configs/collect/ledger_20260902_purge/` |

- v6 격자(2026-09-04 n10 확대 후): noise 10(seed 1300000~1300009). instruction 12키(OvenRack/out-left·right, DishwasherRack/out-left·right, OpenDrawer/left·right, PPCC 5종, CoffeeSetupMug) × scene 3 × j 5 × noise 10. 홈: srv48(worker1)=oven/washer 4키, kanu=drawer 2+coffee, srv50(worker2)=PPCC 5.
- **2026-09-04 키 의미 개정(사용자 결정)**: oven·washer 의 `out-left`/`out-right` 는 이제 **로봇 기준 target fixture 가 있는 쪽**이다(개정 전 = 로봇 스폰 쪽 → 뜻이 정반대였다. 영상 프레임 실측으로 구 `out-left` 셀에서 fixture 가 로봇 오른쪽에 보임을 확인). 두 키를 서로 맞바꾸고 `side` 를 전부 반전 → plan `77e745c37b0f` → **`08f1c9df8207`**. 물리 수집은 불변(같은 seed·같은 base 오프셋; 수집기 lat 부호 규칙도 함께 뒤집혀 상쇄). 아카이브는 `rebase_plan_id.py --rename-key/--flip-side` 로 폴더 rename + meta 패치(pkl 불변), QA TSV 도 같이 교체. drawer left/right·PPCC·coffee 는 무관.
- v6 j 축의 물리량은 계열마다 다르다(핸드오프 0903 §3): PPCC·coffee = ep_meta 고정+연속 reset(reset_idx=j), oven·washer = base 오프셋(lat,back)만(reset 0), drawer = 채택 reset 목록[j] **+** back 오프셋 혼합(교차설계 아님 → 요인 분해 불가). 인덱스 좌표 열은 `jitter_idx`(j); `jitter_reset_idx`·`base_lat`·`base_back` 은 출처 기록.
- v6 SR(900 시점) 0.556: apple 1.00 / bread .87 / oven-left .81 / drawer_right .71 / candle·marshmallow .67 / oven-right .60 / drawer_left .56 / washer-right .55 / jug .17 / washer-left .05 / coffee .01.
- 캡처 규격(v5·v6 동일): N1.5 `lerobot_groot_n15__robocasa365_ckpt120000`, capture_layers 0,2,4,8,10,12,15, all_token_full, denoise_k 4, n_action_steps 5, record_shape [7,4,49,1536].

## 3. 이번 세션에서 확정된 사실·규칙 (놓치면 다시 사고 남)

1. **replay≠수집 반전(v4r 59%)의 원인 = ep_meta JSON 사전 주입.** `reset(seed)` 전에 JSON ep_meta 를 주입하면 k/j 번째 상태가 수집과 달라진다. collector 가 지터+`--ep-meta-load-env-name` 조합을 거부(fail-loud). replay/eval 은 seed reset 재획득 경로(EP_META_DIR 없이) — 3머신 게이트 A=B=D bit 동일 실증.
2. **좌표 = 폴더층.** v5 `s<i>/k<r>/n<j>`, v6 `s<sid>/j<jid>/n<nid>`. 평탄 si 폐지. plan 수정 → plan_id 변경 → 아카이브 rename(`rebase_plan_id.py`/migrate) + meta 패치 필요. **아카이브 rename·삭제는 소비 세션(action phase 등) 러너를 죽이므로 실행 전 사전 공지.**
3. **관측 고정(렌더 정지) QA 표준**: 신규 셀은 `scripts/collect/scan_video_integrity.py`(FREEZE 플래그) + pkl VL Δnorm 상수화(vl[t]==vl[t-2]) 통과해야 아카이브에 넣는다. NOISE 플래그는 격자무늬 바닥 같은 텍스처에 오탐(marshmallow v6 s1) — FREEZE·VL 이 판정 기준. 정지 현상은 scene 결정적(재시작·재수집·타 머신에서 재현) → 해당 (scene,j) 폐기·교체가 유일한 대응.
4. **pull task 지터의 실체(v5)**: 주입 ep_meta 가 `init_robot_base_pos/ori`·`object_cfgs` 를 고정 → base Δ 0.000m, 팔 관절만 재추첨. 이 실측이 v6 base-오프셋 설계의 근거.
5. **첫 셀 게이트**(머신·파이프라인마다 1회): `scripts/collect/v5_first_cell_gate.sh`(v5) / `first_cell_gate.sh`(v6) — A 수집 / B 재실행 / C JSON 주입(거부돼야 정상) / D eval 경로 → A=B=D bit 동일이어야 본수집.
6. **오케스트레이터 재기동 규칙**: 워커 전원 "이관 대기" 시점에만; `docker exec` 클라이언트만 죽으면 컨테이너 안 `http_feature_collect` 가 고아로 남아 같은 셀을 두 번 써 pkl 지문 충돌(지문은 serve_boot_id 포함이라 재수집이면 항상 다름) — 컨테이너 안 프로세스도 확인. `pkill -f`/`pgrep -f` 패턴은 자기 ssh 명령줄에 자기-매칭(exit 144) — **PID/PGID 로 죽일 것**(setsid 이면 pgid=pid).
7. **이관 병목**: 승준 sshd MaxStartups 10 → 3머신 shipper 스트림 합산 ~22 이하(kanu 6·srv 8~9). kanu→승준 ~3MB/s, srv ~8MB/s, 총 ~9MB/s 상한. `STAGING_WAIT_GB` backpressure 는 디스크 여유(srv48 ~20G!)에 맞춰(srv48 12~24, kanu 45, srv50 60~150). GPU util 0 은 대개 이관 대기.
8. **GPU 규칙**(docs/05): 빈 GPU만·lease 필수. srv50 GPU0(kdw4537 공유)·kanu(junhyeong 공유)는 **사용자가 명시 예외를 준 경우에만** 발사 — 예외는 그때그때 다시 받는다.
9. **side / 키의 left·right = 로봇 기준 target fixture 가 있는 쪽**(2026-09-04 개정, oven·washer 만 해당). 스폰 쪽이 아니다 — 선택표의 `spawn_lat`(>0 = 로봇이 fixture 왼쪽에 스폰)·키 항목 `spawn_side` 만 스폰 기준이며 `side` 의 반대다. 수집기 base 오프셋 규칙도 `sign = +1 if side=="left"`(= 항상 fixture 쪽으로 민다)로 함께 뒤집혔으므로 **개정 전후 base 위치는 동일**하다. 개정 시 아카이브 이관: `rebase_plan_id.py --rename-key OvenRack/out-left=OvenRack/out-right ... --flip-side` (키 rename 은 임시 이름을 거쳐 swap 을 지원, meta 의 `grid_instruction`·`side` 패치 + `key_renamed_from` 기록, pkl 불변). **rename 은 소비 세션 러너를 죽이므로 실행 전 사전 공지**(규칙 2 와 동일).

## 4. QA 명령 (승준, anaconda python)

```bash
# 영상 스캔 (cv2) — 결과 TSV flag OK/FREEZE/NOISE
OMP_NUM_THREADS=1 ~/anaconda3/bin/python scripts/collect/scan_video_integrity.py --grid-root <grid>/<plan_id> --out /tmp/scan.tsv --procs 6
# VL 상수화 (pkl, torch) — 세션 임시 스크립트 /tmp/vl_check.py 와 같은 로직: vl_hidden_states Δnorm 이 6 record 연속 상수면 VL_FROZEN
```
v6 결과 파일: `configs/collect/n15_grid_v6_scene_jitter/{qa_video_scan.tsv,qa_vl_check.tsv,qa_invalid_cells.txt}`.

## 5. 도구·경로

- 수집 러너 `scripts/safe/groot_n15/robocasa/collect/collect_grid.sh`(`MAX_CELLS`, `DONE_LIST`, `STAGING_WAIT_GB`, `SERVE_MODE=host` for srv), shipper `ship_to_archive.sh`(`PARALLEL`, `ONESHOT`), 인덱서 `scripts/collect/build_grid_index.py`(stdlib, 승준에서 실행), 검증 `verify_grid.py`, plan_id 재지정 `rebase_plan_id.py`, k-층 이관 `migrate_grid_k_layer.py`, seed→fixture 그룹 스캔 `scan_fixture_groups.py`(산출 `outputs/analysis/seed_scan/fixture_groups/`).
- 런처(비추적): kanu `outputs/collect/logs/launch_v6_kanu.sh`(워크트리 plan·COLLECTOR_PY 주입), srv `~/pkt_ws/temporal_vla/outputs/collect/logs/launch_v6_srv.sh`(MODE=gate|collect).
- kanu 메인 트리 dev 는 다른 세션의 로컬 논문 커밋으로 origin/dev 와 갈라져 있음 → 수집은 워크트리(`.claude/worktrees/grid-v6`)에서.

## 6. 이전 판에서 승계되는 규약 (요약)

좌표 없는 수집 금지(plan 밖 셀 RuntimeError), sig 는 무결성 열, 캡처 밀도 5열 meta 기록, 수집/평가 rollout 위치 분리, 이관은 승준 HDD 로만·대조 통과분만 로컬 삭제, 머신 매칭(replay 는 수집 머신에서), 데스크탑 pdk 영구 배제, srv 런처 `SERVE_MODE=host`+`SERVE_PY`+`SERVE_PYTHONPATH` 필수, 컨테이너 NVML 상실 시 `docker restart`. N1.6 HTTP full 수집 절차·실측은 `history/handoff_20260902_grid_recollect_v5.md` §3.
