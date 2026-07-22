# A100(worker2) 오프로드 플랜 — apple phase 분산 + 데이터 환류

> 2026-07-07 작성. 전제: A100 스모크(serve+수집 1판) 통과. 대상: **지금 돌고 있는
> scene-seed 매트릭스의 apple phase(A-S3)** 일부를 worker2로 분산.
> 데이터 흐름 = worker2 → (arm 단위 즉시) → 로컬(집계·정리) → 승준 서버(아카이브).

## 0. 요약

- **오프로드 단위 = cell 통째** (`ppcs_apple_s100084`, `ppcs_apple_s100104`의 armset 전부, ho_base 포함).
  arm 단위 분할 금지 — **같은 cell의 base/steered가 다른 하드웨어에서 돌면 hardware confound**.
- 수집(A-S1)·fit(A-S2)은 **전부 로컬 유지** (xa/gx pool fit이 bread+apple 원본을 한곳에 요구).
  worker2에는 **fit 산출 NPZ만 push** (수백 MB).
- worker2 실행 = `heldout_round_cell.sh` 재사용 + **host-serve 래퍼 신규 파일** (실행 중 스크립트 무수정).
- 환류 = 로컬 pull-watcher가 **arm 완료 단위로 rsync pull → 검증 → worker2 원본 삭제** (디스크 49GB 제약).
  로컬 경로가 표준 경로 그대로라 `aggregate_final_scene.py` 무수정 동작.
- 정리 후 승준 서버(166.104.146.37:11112)로 cell 단위 rsync 아카이브(검증 후 로컬 대용량 삭제) — 기존 정책.

## 1. 왜 이 분할인가

| 후보 | 판정 | 이유 |
|---|---|---|
| A-S1 수집 오프로드 | ✗ | fit(A-S2)이 원본 요구 → 800MB/cell 왕복만 추가, 수집은 ~3h로 병목 아님 |
| arm 부분집합 오프로드 | ✗ | base/steered 하드웨어 불일치 → ΔSR에 confound |
| **apple cell 2개 통째(A-S3)** | ✓ | A-S3가 전체 벽시계의 ~90%; NPZ+코드+ckpt만 있으면 자립 실행 |
| bread gx armset 오프로드 | ✗ | bread baseline이 로컬 하드웨어 산출물 — 같은 이유로 로컬 유지 |

## 2. 실행 배선

### worker2 (GPU 2·3만, 0·1 불가침)
- **신규** `scripts/safe/groot_n15/robocasa/steer/master_apple_worker2.sh`:
  - cell 2행(s100084, s100104), `APPLE_SETS` 9개 armset 동일 정의.
  - **GPU 정책(2026-07-07 사용자 지시)**: serve는 **GPU 한 개에만 몰아 싣기** (GPU2 단독, GPU3 비움).
    개수 한도 = VRAM이 아니라 **worker2 총 CPU 60% 이하** (serve+collector 쌍당 ~2~2.5코어;
    calibration 실측 6쌍+타 사용자 = 55~56% → 6쌍이 상한 부근). serve:collector = 1:1 유지
    (gated phase POST 간섭 금지). 발사 후 top으로 총 CPU 확인, 60% 초과 시 축소.
  - serve = **host conda** 기동(스모크 검증 커맨드: `CUDA_VISIBLE_DEVICES=… PYTHONPATH=$HOME/pkt_ws/temporal_vla/lerobot/src ~/miniconda3/envs/lerobot_050_groot/bin/python scripts/serve/lerobot.py …`),
    collector = robocasa 컨테이너 `docker exec`(스모크와 동일). `heldout_round_cell.sh`의 serve 기동이
    컨테이너 결합이면 **호스트-serve 변형을 신규 파일로**(`heldout_round_cell_host.sh`) — 로컬 원본 무수정.
  - NPZ 경로는 host 절대경로로 치환(컨테이너 `/temporal_vla` 경로 사용 금지 — serve가 host에서 돎).
  - **N_WORKERS 상향**: A100 VRAM 여유(serve 5.8GB)로 cell당 4 serve → cell 소요 ~35h → **~17h**.
    CPU는 공유(load~5에서 시작) — 8 serve 기준 load 관찰 후 필요시 축소.
- 시작 조건: 로컬 A-S2 완료 후 NPZ push (`rsync final_{ps,xa,gx}{15,30,60}/{s100084,s100104,all}` — 심링크는 상대경로라 안전).

### 로컬 master 개입 (유일한 개입 지점)
- **FINAL2_BREAD_DONE 생성 직후** master kill (bread 집계는 sentinel 전에 이미 실행됨) →
  `master_final_scene2.sh` 수정 → 재발사. §2.4-6 절차(kill→수정→재발사) 준수.
- 수정 내용: A-S3 lane 재배치 —
  - GPU4+5 = `ppcs_apple` 4-worker, GPU6+7 = `ppcs_apple_s100050` 4-worker (freed lane 활용, ~23h),
  - bread gx armset 4개는 apple 뒤 순차(기존과 동일하게 각 lane 후속),
  - s100084/s100104 행 제거(worker2 담당).
- 재발사 시 `.S1B_DONE` 등 stage sentinel로 bread 구간 자동 스킵, 완료 arm은 체인 dedup으로 fast-skip.

## 3. 데이터 환류 (3-hop)

```
worker2(생산) --arm 완료마다--> 로컬(정본·집계) --cell 완료마다--> 승준(아카이브)
```

1. **worker2→로컬** — **신규** `scripts/safe/groot_n15/robocasa/steer/pull_worker2_arms.sh` (로컬, setsid detached):
   - 10분 폴링: worker2 `final_chain.log`의 `DONE ->` arm 검출 → 해당 arm 디렉토리 rsync pull
     (표준 경로 `steer_eval/<cell>/<arm>/` 동일 유지) → **검증(rc=0 + pkl 60개 + tsv 행수)** →
     worker2 쪽 pkl/mp4 삭제(tsv·log 유지). worker2 상주 사용량 ≤ ~2GB/lane (49GB 제약 대응).
   - 미검증 삭제 금지. 실패 시 재시도 큐, 3회 실패 시 보고만.
2. **로컬 처리**: 정본이 로컬에 모이므로 기존 흐름 그대로 — per-arm `sr_result_heldout.tsv` 누적,
   전 arm 완료 시 `aggregate_final_scene.py`(8-cell 최종 + Notion) 실행. **worker2산 cell은 집계
   보고에 "A100 실행" 각주** (hardware 각주, within-cell 비교는 동일 하드웨어라 유효).
3. **로컬→승준**: cell 완료+집계 반영 후 `remote_compute.sh push-data`(또는 기존 rsync 라인)로
   `outputs/eval/robocasa/groot_n15/steer_eval/<cell>/` 아카이브 → **rc=0 + du 대조 검증 후**
   로컬 pkl/mp4 삭제, tsv/json/NPZ/문서 유지. (fit용 `phase_event_6p/raw_rollouts`는 SAE 재사용
   후보라 아카이브만 하고 로컬 유지 여부는 SAE 착수 시 결정.)

## 4. 순서·타임라인

| 시점 | 액션 |
|---|---|
| 지금~bread 완료(~12h) | 스모크 마무리 → worker2 래퍼·pull-watcher 작성 → (승인 시) 기존 bread NPZ로 worker2 1-arm 드라이런 |
| FINAL2_BREAD_DONE | master kill → lane 재배치 수정 → 재발사 (apple A-S1·A-S2 로컬 진행 ~4h) |
| A-S2 완료 | NPZ push → worker2 `master_apple_worker2.sh` 발사, 로컬 pull-watcher 발사 |
| 이후 | worker2 ~17h / 로컬 ~23h+gx — arm 단위 환류 상시 |
| 전 arm 완료 | 최종 집계+Notion → cell 단위 승준 아카이브 → 로컬 정리 |

기대 효과: apple phase 벽시계 **~46h → ~23h** (worker2 17h + 로컬 재배치 23h, 병렬).

## 5. 함정·안전

- 실행 중 로컬 스크립트 수정 금지 → **모든 어댑터는 신규 파일**, master 수정은 kill 후에만.
- worker2 디스크 99% — pull-후-삭제 엄수, 발사 전 `df` 확인, watcher 죽으면 lane도 세울 것(디스크 폭주 방지 위해 worker2 스크립트에 여유 <10GB 시 pause 가드).
- stale sentinel(`FINAL2_*_DONE`) 재발사 전 정리; waiter는 세션 소멸 주의(PID 기반 재설치).
- serve 종료 시 GPU 반납 확인(cleanup-policy), worker2 GPU0·1 절대 미사용.
- 검증 전 삭제 금지(양쪽 hop 공통): rc=0 + 개수/용량 대조.

## 6. hardware confound 보강 (2026-07-07 스모크 실증 후 추가)

**실증**: 같은 (scenario_seed=100084, inference_seed=0)이 로컬=succ0(wrong-grasp, 144 rec) vs
A100=succ1(정상, 72 rec) — **(scene, ep) 결정성은 머신-로컬**(bf16 커널/아키텍처 차이로 trajectory 발산).

confound가 걸리는 곳의 분해:

| 비교 축 | hardware 혼입 | 상태 |
|---|---|---|
| conceptor fit (ps/xa/xb/gx) | 없음 — fit 데이터 전부 로컬 수집 | ✅ 설계로 회피 |
| per-cell ΔSR (base vs steered) | 없음 — cell 통째 co-location | ✅ |
| per-seed flip 분해 | **머신 간 episode 매칭 불가**(위 실증) — cell 내(같은 머신)에서만 유효 | ✅ cell 내 한정 |
| cell 간 scene-일관성 비교 | **A100 cell vs 로컬 cell 비교에 machine 혼입** | ⚠️ 아래 처방 |
| 로컬-fit conceptor를 A100 활성화에 주입 | 분포 shift 이론상 가능(가중치 동일, bf16 라운딩) | ⚠️ calibration이 겸사 검증 |

**처방**:
1. **calibration = bread를 A100에서 재측정** (로컬 추가 실행 불필요 — bread 로컬 완주 결과가 이미 존재):
   `ppcc_bread`(s100084)의 `ho_base` + 대표 arm(`ho_permps60`, `ho_gatedps15`)을 worker2에서 동일
   설정(ep60–119, inference_seed=ep×1000, PROX=1)으로 실행 → 로컬 수치와 직접 대조.
   판정: base SR 차이·ΔSR 차이가 binomial noise(n=60, SE_diff≈0.075) 내 → 머신 간 비교 허용;
   초과 → 결론은 머신-내 비교로 한정하고 보고서에 명시.
   산출 경로는 `steer_eval_a100_calib/ppcc_bread/`로 분리(로컬 표준 경로와 충돌 방지).
   **결과(2026-07-07 완료)**: base 로컬 .783 vs A100 .750 (z≈0.43) / gatedps15 .917 vs .933 (z≈0.35) /
   ΔSR +.133 vs +.183 (차이 z≈0.5) — **전 지표 noise 범위, 머신 간 비교 허용**(단 n=60 검출한계 ±~0.15,
   bread84 1개 cell 기준 외삽임을 각주). gatedps15의 bread84 국소 +효과는 독립 episode 재표본에서 재현
   (pooled 120판 Δ+.158, z≈3.5) — 단 scene-국소(같은 arm이 s300028에선 −.12), scene-일관성 주장 불가.
2. apple 4 cell을 2(A100)+2(로컬)로 **instruction 내 교차 배치**(현 설계) — instruction=machine 완전
   정렬(예: apple 전부 A100)은 cross-instruction 비교를 hardware와 100% confound시키므로 금지.
3. 집계 JSON·Notion 테이블에 cell별 `machine` 필드 명기, 헤드라인 판정은 머신-내 비교 우선.
