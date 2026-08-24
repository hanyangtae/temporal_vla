# 45 — n15_grid_v2 확장 수집 결산 (s+5 · n+5)

2026-08-19. branch `feat/online-gated-pipe`. 42 §7(pdk 렌더 비결정) 후속으로 base grid를
10×10 → **15 scene × 15 noise**로 확장한 수집 라운드의 결산.

## 1. 계획·정본

- plan: `configs/collect/n15_grid_v2/collection_plan.json` (plan_id **3134e339de4c**).
  신규 scene 5개/task는 미사용 seed 대역 100742~101741을 reset-스캔(`scan_seed_instructions.py`)
  해 canonical instruction 일치로 선별. noise n10–14 = 기존 noise_seeds 1300010–14.
- 기수집 v1 유효분 819셀은 `done_prefill.txt`로 스킵. **pdk(dongkyu-MS-7D43) 수집분은
  전량 삭제·재수집**(42 §7; OvenRack 225 전체 + marshmallow 결손, 사용자 지시).
- 수집 머신 = v1 머신 매칭 (사용자 지시): kanu=drawer_left·apple·marshmallow,
  worker1(srv48)=coffee·dish, worker2(srv50)=drawer_right·candle·jug.
  예외: **OvenRack 새 홈=kanu**(구 홈 pdk 소멸), **bread는 kanu**(재편 전 kanu가 완집 —
  v2 bread의 replay 홈은 kanu다).

## 2. 결산 (아카이브 실물, index 재생성 완료)

| task | 셀 (목표 225) | base SR | 수집 머신(v2 신규분) |
|---|---|---|---|
| OpenDrawer/left | 225 | 0.66 | kanu |
| OpenDrawer/right | 225 | 0.69 | worker2 |
| PPCC/apple | 225 | 1.00 | kanu |
| PPCC/bread | 225 | 0.71 | kanu(v2)·worker1(v1) |
| PPCC/candle | 225 | 0.66 | worker2 |
| PPCC/jug | 222 | 0.16 | worker2 |
| PPCC/marshmallow | 221 | 0.81 | kanu |
| CoffeeSetupMug | 223 | 0.09 | worker1 |
| DishwasherRack/out | 225 | 0.33 | worker1 |
| OvenRack/out | 220 | 0.50 | **kanu (전량 v2 재수집)** |

- 합계 **2,236 rollout** (v1 819 + v2 1,417), 인덱스 무결성 위반 0.
- **수집실패 14셀**(1%): OvenRack s14×{n2,3,6,12,13}, marshmallow s11n2·s11n8·s12n11·
  s14n4, jug 3, coffee 2 — 재시도 2회에도 재현 실패 = seed feasibility성(mixer 100010
  전례). 분석·eval에서 해당 셀 제외가 정본.
- 인덱스: `outputs/steer/online_pipe/manifests/index_rollouts_v2.tsv` (2236행; 구
  v1 전용 index_rollouts.tsv는 dedup판 820행으로 보존). 아카이브 =
  승준 HDD `~/datasets/temporal_vla_store/groot/n15/grid/3134e339de4c/`.

## 3. 사고·패치 기록 (재발 방지)

1. **이관 병목 사고**: 승준 노드행 링크가 스트림당 ~1MB/s 셰이핑 + 수신 총량 ~9MB/s.
   실시간 이관이 수집(6 serve ≈ 15MB/s 생산)을 못 따라가 3머신 staging 폭주(합 274GB)
   → 디스크 포화(공용 루트 여유 4–7GB) → 수집기 전멸. 패치(커밋):
   - `ship_to_archive.sh` **PARALLEL N-스트림**(기본 8, 라운드로빈 셀 분배).
   - `collect_grid.sh` **STAGING_WAIT_GB backpressure**(staging 초과 시 수집 대기).
2. **CPU 폭식**: serve 스레드 무제한(개당 ~460%) → 공유 서버 load 163. 패치:
   docker serve에 `SERVE_OMP_THREADS`(기본 4) 주입 + host serve는 launcher에서 cap.
3. kanu 병렬 러너 GPU 중복(42 §6 함정)과 별개로, **원격 pkill 자기-매칭 함정**: ssh 원격
   명령·harness 명령 문자열이 pkill -f 패턴에 걸려 자기 자신이 죽는다 — 정지·재발사는
   스크립트 파일을 만들어 setsid로 실행할 것.
4. srv50 잔존 serve 정리 누락(candle eval 잔재) — 종료 보고 전 `pgrep+nvidia-smi` 확인
   원칙 재확인.

## 4. 후속

- fit/분석은 index_rollouts_v2.tsv 기준으로 확장 사분면(seen 5×5 fit → unseen scene
  s10–14 × unseen noise n10–14)이 열림 — 기존 s0-9×{0,1,5,6} eval 사분면의 상위집합.
- pdk는 이후 수집·replay에서 영구 배제 (42 §7).

## 5. 오염영상(caption burn-in) 재생성 (2026-08-20)

kanu v2 수집분은 당시 메인레포(dev)의 collector 기본 overlay_text=True 로 **영상만 오염**
(pkl·메타·판정 무결; srv48/50분은 False 강제 브랜치 코드 = 클린 — 표본 40/40 실증).
조치: 국내투고 세션의 action 번들(actions+scenario_seed)로 kanu에서 replay 재생성 —
**797/797 완주, 전판 eef 편차 0(비트 일치)·실패 0**, 아카이브 video.mp4 교체 완료
(무작위 25 표본 FLAG 0). 우선군(OvenRack 220 + marshmallow 132)은 국내투고 논문
일정에 맞춰 선처리·통보. 도구: replay_orchestrate_v2.sh·replay_batch_runner.py(v2 번들
규약·컨테이너 EGL default 경로)·make_v2_manifest.py·push_clean_videos.sh.
함정 기록: robocasa 컨테이너 EGL 은 default 1대만 열거(NVML 복구와 무관, GPU 지정 불가
— eef 게이트로 무해)·robosuite 는 CVD 설정 시 EGL id∈CVD 를 강요.

## 6. v3 지터 축 (k-grid) 정식 채택 (2026-08-21)

§4 의 "재발 변주 3축" 중 배치·로봇자세 축이 **n15_grid_v3 계열로 정식 채택**됐다.
정본: 좌표·인덱스·replay 규약 = `docs/04` §3.1.1, 검증 실측·설계 이력 =
`docs/collab/collect_request_v3_jitter.md` §6–7. 요지: ep_meta 고정+연속 reset 으로
물건·target 고정·배치/관절만 재추첨, (base_es, ep_meta, k) bit 결정적, pkl 좌표는 평탄
si=scene*100+k, 인덱스는 명시 3축, OpenDrawer 는 방향 재추첨 때문에 k-사전 스캔 필수.
1차 수집 = task 2(drawer-L kanu·candle srv50) × scene 3 × 채택 k 20 × inf 2 = 240판.

## 7. v4 지터 확대 수집 결산 (2026-08-21)

사용자 재지시로 §6 파일럿(240판)을 접고 **전 task 확대**로 전환 — 신규 1,000판 수집 완료.

- plan **46ea62d53e09** (`configs/collect/n15_grid_v4/`): 10 task × v2 scene 0-4 ×
  v2 noise 0-4(1300000-4) × 신규 k 4개. base 상태는 **기존 v2/v1 셀 재사용**(재수집 없음)
  → task·scene·noise 당 **5상태**(base + k4).
- 좌표(exp6 합의): pkl/디렉토리는 평탄 `s{scene*100+k}` 2축 유지, **base 슬롯 = scene*100+99**
  (채택 k 가 0-3 고정이 아니라 스캔 통과분이라 +4 는 충돌 — 실측 최대 채택 k=9).
- **index_rollouts_v4.tsv = 1,250행** (지터 1,000 + base 250, 결손 0, 무결성 위반 0):
  scene_idx=base scene(0-4), `cell_si`=평탄, `jitter_reset_idx`= k 또는 `"base"`.
  base 는 plan_id 가 아니라 **좌표 기준**으로 선별해야 한다 (v2 plan 아래 s0-4×n0-4 는
  35개뿐 — 나머지는 v1 수집분 979d/b805; 좌표 중복은 v2>v1b>v1 우선).
- SR(5상태 합산): apple 1.00, oven 0.82, drawer-R 0.77, marshmallow 0.78, bread 0.74,
  candle 0.69, drawer-L 0.41, dish 0.34, jug 0.18, coffee 0.07.
- ep_meta JSON 50개(10 task × 5 scene) 아카이브 동봉: `46ea62d53e09/ep_meta/<task>/`.
- 수집 머신 = v2 매칭 (kanu 5task 500 · srv50 3task 300 · srv48 2task 200).
- 함정: srv50/srv48 런처에 **SERVE_MODE=host·SERVE_PY·SERVE_PYTHONPATH 누락 시 전멸**
  (`No such container: lerobot` → serve TIMEOUT → ABORT). v3 파일럿에서 실제 발생.
  또 신규 plan 은 **전용 staging** 을 써야 한다 (평탄 si 가 기존 plan 셀 키와 문자열
  충돌해 DONE_LIST 오판).

