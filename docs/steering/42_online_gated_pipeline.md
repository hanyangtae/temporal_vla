# 42 — Online failure-gated phase-matched steering 파이프 (exp6)

2026-08-13. branch `feat/online-gated-pipe`. 교수님 방침(되는 case로 전체 파이프 완성)에
따라 **"SAFE 감지 → GT phase 판정 → 그 phase의 setM 개입"** closed-loop를 구축·검증한
라운드. 결과 한 줄: **감지·phase판정·스위칭 사슬은 전부 검증 완료. 병목은 개입 연산자 —
현 setM(β0.5/1.0)은 셀-paired replay에서 구제 0.**

## 1. 파이프 구성 (재사용 좌표)

```
[client http_feature_collect]                  [serve lerobot.py]
 매 get_action:                                  /act_with_features:
  labeler.step() → 현재 GT phase                  DiT L12 캡처 → LSTM (h,c) 캐리
  응답 features.failure_fired 수신                → failure_score, CP δ_t 초과 판정
  발화 전: POST "off" / 발화(latch) 후:            /steering_phase: phase별 setM_seg
    현재 phase POST (phase-follow)                 스위칭 (미등록 phase=identity)
```

- **detector**: per-task SAFE-LSTM (hidden256, 입력 = DiT L12 · 마지막 denoise · 49토큰
  mean [1536] = grid shard 좌표 X[:,5,3,3,:]). functional-CP 밴드(train μ/σ + calib bw),
  α=0.1. 학습·시뮬 `scripts/analysis/grid_phase/failure_detector_sim.py`, 런타임
  `src/failure_online/online_failure.py` (1-step == 배치 forward 수치 일치 검증).
  **혼합 vs per-task**: 5-seed split 회전 실측으로 per-task 채택 (SAFE 논문 Table 6의
  혼합 우위는 task당 30판 희소+zero-shot 조건 얘기 — 우리는 task당 100판, 혼합은
  OvenRack FPR 1.0 붕괴).
- **연산자**: phase별 setM seg 포맷 (`fit_setm.py --phase-groups auto --npz-format seg`),
  위약 = 라벨순열 dose-match. future-only 변형(`make_seg_mask_variant.py`, seg_mask
  [0,1,0]) — 전토큰 적용 시 eef 진동(사용자 관측) 완화용.
- **fit/eval 분할 (사용자 확정)**: fit = scene 0–4 × noise 0–4 (클래스당 episode ≥3 보장;
  OvenRack만 scenes {0,1,2,3,7} — s0–4에 실패 0), eval = **수집 셀 replay**
  (10 scene × noise {0,1,5,6}, index의 env/inference seed 재생) → 사분면
  (seen/unseen scene × noise)별 **구제율**(수집실패→성공)·**파손율**(수집성공→실패).
- **eval 머신 = 수집 머신** (각 instruction이 단일 머신 수집): drawer=kanu,
  Dish=srv48(GPU0×serve6), candle=srv50(GPU2×serve6), OvenRack=dongkyu-MS-7D43(미실행).
- arm: base(detector만, steering 미등록=identity — failure_scores 기록용) / online /
  online_fut / online_pl / oracle_always(상시 phase-gating) (+ oracle_always_pl,
  online_fut_pl).

## 2. 방법론 검증 — 전부 통과

| 검증 | 결과 |
|---|---|
| **replay 결정성** | base arm이 수집 결과를 **3개 머신 모두 40/40 셀 재현** (구제 0·파손 0). +부수 확인 1회(모드 버그로 gating 없이 돈 arm도 40/40) → 이후 구제/파손은 순수 개입 인과 |
| **detector 실전 = 시뮬 예측** | drawer: 실패 17/17 발화 + FPR 0.087 (시뮬 예측 TPR 1.0/FPR 0.09와 일치). Dish: 실패 전부 + FPR 0.125. candle: FPR ~0.14 (시뮬 0.40보다 양호) |
| **배선 무결성** | off ≡ no-op(409→identity 처리), `[steer-registered]` preflight, phase 어휘 대조, sidecar trigger_step/phase_at_trigger/failure_scores |

## 3. 본 결과 — 셀-paired 구제/파손 (α0.1, 40셀/arm)

### OpenDrawer/left (kanu, β1.0)
| arm | SR | 구제 | 파손 |
|---|---|---|---|
| base | 0.575 | — | — |
| online | 0.525 | 0/17 | 2/23 |
| online_pl (위약) | 0.525 | 0/17 | 2/23 |
| online_fut | 0.575 | 2/17 | 2/23 |
| oracle_always | **0.200** | 0/17 | **15/23** |

β0.5: online = pl = fut 전부 0.525 (구제 0, 파손 2 — 세 arm이 동일 셀을 뒤집음).

### DishwasherRack/out (srv48, β1.0)
| arm | SR | 구제 | 파손 |
|---|---|---|---|
| base | 0.400 | — | — |
| online | 0.325 | 0/24 | 3/16 |
| online_pl | 0.350 | 0/24 | 2/16 |
| online_fut | 0.375 | 0/24 | 1/16 |
| oracle_always | 0.275 | 1/24 | 6/16 |

### PPCC/candle (srv50, β1.0)
| arm | SR | 구제 | 파손 |
|---|---|---|---|
| base | 0.725 | — | — |
| online | 0.675 | 1/11 | 3/29 |
| online_pl | 0.625 | 1/11 | 5/29 |
| online_fut | 0.675 | 0/11 | 2/29 |
| oracle_always | 0.450 | **3/11** | 14/29 |

## 4. 판정

1. **개입(현 setM)은 구제를 만들지 못한다.** 전 task에서 online 처치 ≈ 위약 (drawer는
   수치까지 동일). 구제 합계: 처치 1~2 vs 위약 1 — 위약 수준. 파손은 개입 강도에 비례
   (oracle 상시 β1.0 = drawer −37.5pp 파괴, exp5-3 재확인).
2. **β 민감도 없음(0.5~1.0)**: drawer β0.5에서 처치·위약·future가 동일 결과 — 뒤집힌
   셀은 연산자 내용이 아니라 "개입이 있었다는 사실"(타이밍 섭동)로 갈리는 경계 셀로 해석.
3. **candle oracle의 구제 3/11**은 개입이 원리적으로 실패를 뒤집을 수 있음을 보여주나
   (exp5-2 P1 ppcc 회복과 방향 일치), 같은 arm이 성공 14/29를 파괴 — 현 연산자는
   방향/게이팅 선택성이 없다.
4. **OvenRack(초기조건형)은 감지-후-개입 프레임 부적합**: s5b detector FPR 0.39(α0.05)
   ~0.94(α0.3) — CP 밴드가 succ/fail이 아니라 "위험 scene 소속"을 읽음 (41 라운드
   within-scene 분리와 정합). 처방은 oracle_always vs oracle_always_pl 대조(데스크탑
   접근 확보 시 실행).

**종합: "언제·어디에 개입할지"는 풀렸고, "무엇을 주입할지"가 미해결.** 다음 후보:
(a) setM 방향 재설계 — scene-잔차화 후 fit / phase 내 성공 궤적으로의 projection
(conceptor 계열 재검토), (b) β·발화창 미세 sweep 대신 개입 형태 변경(중단·재샘플 —
41 §8.5 실행표류형 처방), (c) candle oracle 구제 3판 해부(어느 phase·어떤 이동이
뒤집었나 — sidecar/영상).

## 6. 증보 (2026-08-14) — 평균/분산 진단 · 수축 연산자 3종 · OvenRack 대조 · 영상

### 6.1 평균 vs 분산 진단 (`mean_var_sep.py`, fit 셀 in-sample)

setM 실패의 원인 후보를 형상으로 진단: 평균 분리는 전 phase 존재(0.72–0.99, scene 미통제
in-sample)하나, **reach 계열은 실패 산포가 성공의 1.3–1.8배**이고 "성공 중심으로부터의
거리"만으로 AUROC 0.81–0.92 — 실패는 한 방향으로 밀린 게 아니라 **사방으로 퍼진**
구조(성공=좁은 다양체, 실패=여집합; seen18 task-간 관찰의 task-내부 확장판). candle만
분산 분리가 없고 평균만 약함(0.72). → 단일 방향 setpoint가 원리적으로 부적합한 형상.

### 6.2 수축 연산자 3종 (`fit_contraction_ops.py`, h' = h·Mᵀ)

- fit 진단: **sconceptor**(순수 C_s, α0.1)만 전면 비퇴화(rel-move 3–7%). **대조
  conceptor(C_s∧¬C_f)는 β=1에서 거의 전 셀 활성화 소거(rel-move >0.9)로 퇴화** — exp3
  포화 소견의 재확인이자 "실패 방향이 없다"(6.1)와 정합. varc(분산-가중 수축)는 예측대로
  산포-지배 phase에서만 작동(6–9%).
- eval (같은 replay 40셀, 감지-후 개입, arm별 구제/파손):

| task(base) | op | online | 위약 | future-only | fut 위약 |
|---|---|---|---|---|---|
| Dish(0.400) | sconceptor | 0.325 (0/3) | 0.350 (0/2) | **0.425 (1/0)** | 0.400 (0/0) |
| Dish | varc | 0.350 (1/3) | 0.325 (0/3) | 0.400 (0/0) | 0.400 (0/0) |
| Dish | sconceptor β0.5 fut | — | — | 0.350 (0/2) | 0.375 (0/1) |
| Dish | sconceptor fut **상시** | 0.225 (1/8) | | | |
| candle(0.725) | sconceptor | 0.550 (0/7) | 0.550 (0/7) | 0.600 (0/5) | — |
| candle | varc | 0.550 (0/7) | 0.550 (0/7) | 0.625 (0/4) | — |
| candle | conceptor(1 phase) | 0.575 (0/6) | 0.675 (0/2) | 0.625 (0/4) | — |
| drawer(0.575) | sconceptor | 0.525 (0/2) | 0.525 (0/2) | 0.525 (0/2) | — |
| drawer | varc | 0.525 (0/2) | 0.525 (0/2) | 0.525 (0/2) | — |
| drawer | conceptor(1 phase) | 0.550 (0/1) | 0.550 (0/1) | 0.600 (2/1) | — |

판정: **수축 계열도 전면 null~미약.** 유일한 순양 셀(Dish sconceptor fut +1/0, 위약 0/0)은
β0.5에서 비재현·candle 미재현이라 노이즈 우세로 해석. 처치=위약 동일 수치가 반복되는 것
(sconceptor≡varc≡위약인 cell 다수)은 뒤집힌 셀이 연산자 내용이 아니라 **개입 존재의
타이밍 섭동**으로 갈리는 경계 셀임을 시사.

### 6.3 OvenRack (데스크탑 pdk_external, 수집 머신 매칭, α0.05)

| arm | SR | 구제 | 파손 |
|---|---|---|---|
| base | 0.625 | 3/17 | 1/23 |
| online(감지→setM) | 0.650 | 6/17 | 3/23 |
| oracle_always(상시 setM) | 0.575 | 9/17 | 9/23 |
| **oracle_always_pl(상시 위약)** | **0.800** | **14/17 (0.82)** | 5/23 |

두 발견: ① **데스크탑 replay는 비결정**(base가 수집을 4/40 못 재현 — kanu·srv48·srv50은
40/40) → 이 표는 그 요동 폭 위에서 읽어야 함. ② **상시 무작위-방향 개입이 최고 성적**
(+0.175, 전실패 scene s5/8/9의 unseen 사분면까지 6/6 구제) — OvenRack 초기조건형 실패는
방향적 교정 대상이 아니라 **아무 섭동이든 흔들면 깨지는 불안정 평형**이며, setM 방향은
오히려 위약보다 해로움. 41 라운드의 "전실패 scene"도 기하 불가가 아니라 정책 초기 습관.

### 6.4 종합 재판정 (exp6 최종)

1. 감지→phase판정→스위칭 파이프는 완성·검증 (셀-paired replay, detector 실전=시뮬 일치).
2. **선형 활성화 개입은 형태(방향 setpoint/수축/분산-가중) 불문 rescue를 만들지 못한다**
   — 처치≈위약 반복, 구제 0–2판 노이즈, 상시는 파괴적.
3. 예외 신호는 **OvenRack 상시 위약**의 대량 구제 = "개입 내용"이 아니라 "섭동 자체"의
   효과. → 다음 라운드 1순위는 **감지 후 재샘플**(denoise seed 재추첨, 방향 가정 없음)
   — 42 §4 후보 (b)가 이 데이터로 승격.
4. 영상 증거: `outputs/eval/robocasa/groot_n15/video_gallery/` (4 task, 셀×arm 격자,
   주석판 — 실패예측값/발화/phase/개입상태, http://166.104.35.33:8898).

인프라 추가 유산: replay 사분면 러너(EP_MODE=replay)·연산자 3종 fit·post-hoc 영상
주석기·갤러리·A100 serve6 규칙·GR00T 캡션 상단 배너(fork 108f338, gitlink 고정).
함정 기록: kanu 병렬 러너는 GPU 목록을 분리할 것(같은 목록 공유 시 GPU당 4 serve OOM),
worker 기동 500 오류는 resume 반복으로 수렴, 데스크탑 replay 비결정.

## 5. 재현 좌표

- 러너: `scripts/steer/online_gated/run_online_gated_eval.sh` (EP_MODE=replay 기본,
  DRY_RUN=1 검증, worktree 실행 시 NPZ/ckpt/OUT **절대경로** 필수 — abspath는 main-tree
  기준). A100 GPU당 serve 6, kanu 2.
- detector 시뮬·ckpt: `outputs/analysis/grid_phase/detector_sim_s5m5/` (+`_s5b_oven`),
  승준 원본 동일 경로.
- 연산자: `outputs/steer/online_pipe/<slug>/setM_s5m5_seg{,_pl,_fut,_fut_pl}` (Oven은
  `setM_s5b_seg*`). fit manifest = `manifests/<slug>_s5m5.tsv`.
- 결과: kanu `outputs/eval/robocasa/groot_n15/online_gated_replay{,_b05}/`,
  srv48·srv50 `~/pkt_ws/temporal_vla/outputs/eval/robocasa/groot_n15/online_gated/`.
  집계는 `collect_results.py --cells-tsv ... --fit-scenes 0,1,2,3,4 --fit-noises 0,1,2,3,4`.
- 함정 기록: worktree lerobot 서브모듈 init 필수(빈 dir가 패키지 가림), 컨테이너 NVML
  상실 시 `docker restart lerobot`, base arm 409 처리(71a71f0), online_fut_pl 모드
  분기 버그(수정됨 — β0.5 fut_pl 산출물은 base 재현본이니 위약으로 읽지 말 것).
