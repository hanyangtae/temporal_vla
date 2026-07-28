# 31. exp5-3 — within-scene setM steering (drawer) + mixer scene-matched 확장

> 2026-07-27. 단일 출처: 이 문서 + 승준 `~/sm_*{drawer,mixer}*.json`·`~/exp53_npz/fit_report.json`.
> 스크립트: `scripts/safe/groot_n15/robocasa/steer/exp5_3/`. 실행 머신 = rudxo_home(4090, serve 3).
> 배경: scene-matched 진단(같은 scenario_seed 에 inference_seed 8종 변주 → scene 안 succ/fail
> 혼재)에서 drawer 분리 실재(L12 0.837, t=0에 0.71) + **cross-scene fit 은 scene 구조 학습**
> 판명. exp2(scene 고정·conceptor·bread/apple)의 drawer·setM·within-scene 재수행.

## 1. Fit — within-scene 방향, LOO-by-seed, per-scene setpoint

- 방향 = 혼재 13 scene 별 (μ_fail−μ_succ) 세그먼트별(state/future/action) 평균 → 정규화.
  fold = leave-one-seed-out 8종 (평가 seed 의 episode 는 fit 미기여 — in-sample 차단).
- ★설계 교정(게이트가 잡음): **전역 setpoint 는 scene offset 을 못 따라감** — seed-out
  비중심 read-out 0.59 로 붕괴, scene-중심은 0.868. → permanent 배포를 **per-scene setpoint
  registry**(`scene{S}/dit_L12/conceptors.npz`, client `--steer-phase-name scene{S}`)로 변경.
  exp2 의 per-scene fit 정신과 일치. fold 에 그 scene 성공이 없으면 전역 fallback(무음 off 방지).
- 게이트: scene-중심 read-out 0.868 ✓ · move/gap 2.92≤3(within-scene gap 기준) ✓ ·
  fold-cos ≥0.907 ✓. gated 채택 phase = reach-to-handle·grasp-handle (succ/fail 각 ≥50 rec).
- 배선 함정: `setpoint_seg` 는 serve 에 **`--steering-token-select all` 필수** (누락 시 기동 실패).

## 2. drawer steering 결과 — β=1.0 파괴적 해악 (사용자 중단)

grid = scene 20 × inference_seed 8 (수집과 동일; baseline = srv50 수집 320판).
A0 앵커 40판(home)으로 머신 이동 확인: 0.350 vs srv50 동일칸 0.375 → **이동 없음**.

| arm | n | SR | base(동일칸) | Δ판 | 구제/해악 | McNemar p† |
|---|---|---|---|---|---|---|
| setM_within_permanent | 160 | **0.025** | 0.344 | **−51** | 1 / 52 | <1e-4 |
| setM_within_gated | 27(부분) | 0.222 | 0.519 | −8 | 1 / 9 | 0.02 |

† cross-machine (srv50 baseline vs home eval) — trajectory 짝 아닌 조건 짝, 각주 필수.
단 A0 앵커가 이동 없음을 보여 방향성 판정에는 충분.

- **판정: β=1.0 은 성공하던 episode 까지 파괴** (해악 52 vs 구제 1). 읽히는 방향
  (진단 0.84)이라도 이 dose 로 밀면 행동이 무너진다 — read≠write 의 정량 실증.

### 2.1 β·세그먼트·위약 sweep (2026-07-28 완결 — A0 라틴 40셀 paired, 결정론 40/40)

| arm | SR | 보존율 | 구제율 | jerk(A0비) |
|---|---|---|---|---|
| A0 (=A0_kin 재실행) | 0.350 | 1.000 | — | 1.00× |
| full β=0.5 / β=0.2 | 0.225 / 0.300 | 0.571 / 0.786 | 0.038 / 0.038 | 1.19× / 1.05× |
| future-only β=1.0 / β=0.5 | **0.350** / 0.325 | 0.786 | 0.115 / 0.077 | **0.94×** / 1.00× |
| **위약 fut β=1.0** (순열·준직교·dose-match) | 0.300 | 0.786 | 0.038 | 1.08× |
| gated×fut β=1.0 (reach/grasp 한정) | 0.325 | 0.786 | 0.077 | 1.03× |

- **해악 = 용량 문제 확정**(β 단조 회복) + **떨림·파괴의 주범 = action 토큰 개입**
  (같은 β=1.0 에서 full 0.025 vs fut 0.350·jerk 0.94×).
- **방향 이득은 위약 미초과**: 구제 fut 3/26 vs 위약 1/26 vs gated_fut 2/26 — 우연 범위.
  해악(성공 3/14 손실)은 처치·위약 동일 = dose 요동. exp2 의 "위약 미초과"가
  within-scene 방향·future-only·phase-gated 까지 확장 반복.
- **최종 판정: 이 축은 read 전용** — 밀기(steering) 계열은 용량·세그먼트·phase·위약
  대조 전부에서 이득 없음. 활용 경로는 selection(exp5-4 계획)으로 이동.
- 상세 회신: `exp5-3_to_exp5-1_beta_sweep_reply.txt` + `exp5-3_bsweep_paired.tsv` (repo 루트).

## 3. mixer scene-matched 수집 + 진단 (Phase 2)

- 수집: feasible 20 scene(BLOCKED 100010 제외) × seed 8 = **160/160**, capture ON(full-token),
  **SR 0.69 (succ 110 / fail 50)**. 구 SR 0.33 은 n=6 스모크였고 16-실행 0.75 와 정합.
  env 판정(head>0.99)은 결과가 이봉분포라 경계 애매성 없음(문서 27) — 재판정 불요.
  영상: **상단 여백 배너**(instruction+phase+step) 주석본 160개 병행 생성 (하단 가림 캡션 대체,
  2026-07-27 표준). 데이터: 승준 `exp5_3_mixer_sm`(160 pkl) + **home 원본 보존**(삭제 금지 지시).
- 진단 (drawer 파이프 동일, 창 [0,17), 혼재 scene 15/20 상당 — n_eval 120):

| 검정 (최적층) | drawer L12 | **mixer L8** | beer L12 |
|---|---|---|---|
| within-scene, scene 홀드아웃 | 0.847 | **0.728** (z 3.2) | 0.623 |
| + seed 홀드아웃 | 0.837 | **0.613** (z 1.7, p .04) | 0.622 |
| t=0 | 0.712 | 0.610 | 0.542(우연) |
| 전역(cross-scene) 방향 | 0.627 | 0.526 | 0.669 |

- **within-scene 방향의 우월이 2번째 cell 에서 재현** (0.526 vs 0.728).
- mixer 는 **중간 강도**: seed 홀드아웃에서 0.73→0.61 — 신호의 상당 부분이 노이즈 draw
  주효과("그 draw 가 원래 나쁨"). drawer(초기조건형·seed-robust 0.84 유지)와 다른 체제.
- 국소: **disengage phase AUROC 0.900** (n=31, dwell 0.430 — 체류 착시 아님) — "부분 개방 후
  놓침" 실패 유형이 활성에서 뚜렷. phase-gated 개입 후보 지점이나 소표본.
- mixer steer 는 보류(사용자: 수집 먼저). 우선순위 drawer β 재조정 > mixer.

## 4. 종합 정리 (2026-07-28 라운드 종결 — 시도 / 결과 / 미시도 / 데이터)

### 4.1 시도한 것과 결과

| # | 시도 | 결과 |
|---|---|---|
| 1 | **scene-matched 수집** — 같은 scenario_seed 에 inference_seed 8종 변주 | drawer 320판(srv50)·mixer 160판(home)·beer 320판 내 포함. scene 안 succ/fail 혼재 성립 (drawer 13/20·mixer 15/20 혼재) |
| 2 | **within-scene 분리 진단** (scene·길이·dwell·seed 통제) | **분리 실재**: drawer 0.84(z4.4, t=0에 0.71=초기조건형)·mixer 0.73/seed-out 0.61(중간)·beer 0.62(보류). cross-scene 방향(0.53~0.67)은 열세 = **기존 cross-scene fit 은 scene 구조 학습** |
| 3 | **setM steering, β=1.0 full** (LOO-by-seed fit·per-scene setpoint) | **파괴적**: SR 0.344→0.025 (−51/160), 해악 52/구제 1. read≠write 정량 실증 |
| 4 | **β sweep** (0.2/0.5, exp5-1 요청) | 해악 β-단조(100%→43%→21%) = **용량 문제**. 구제는 전 용량 ~0 |
| 5 | **future-only mask [0,1,0]** (β 1.0/0.5) | β=1.0 에서 **무해**(SR 0.350=A0)·jerk 0.94× → **떨림·파괴 주범 = action 토큰 개입** 확정. 구제 3/26 최고지만 순Δ 0 |
| 6 | **위약** (scene내 라벨순열 전파이프·준직교·dose-match) | 해악 3/14 처치와 동일, 구제 1/26 — **방향 이득은 위약 미초과**. exp2 패턴 반복 |
| 7 | **gated×future** (reach/grasp phase 한정) | SR 0.325, 구제 2/26 — 위약과 구별 불가 |
| 8 | 부수 검증 | **머신 결정론 40/40**(A0 재실행 성패 완전일치)·jerk 로깅 상시화·상단배너 annot 표준화·seed→instruction registry(drawer 좌우/beer 1%) |

**한 줄 결론: within-scene succ/fail 축은 잘 읽히지만(0.84), 어떤 밀기(β×세그먼트×phase)로도
위약을 못 이긴다 — "read 전용" 확정.** conceptor(exp2/3)에 이어 setM 까지, raw 활성 대조
기반 밀기 연산자 두 계열이 모두 닫힘.

### 4.2 안 해본 것 (닫은 이유 포함)

- **성분 제거(projection-out, LEACE류)** · **SAE feature clamp**: G3 메뉴의 잔여 2 연산자.
  raw 대조 방향 재사용인 한 같은 벽(방향에 이득 정보 없음) 예상 — 시도하려면 위약 내장 +
  scene 잔차화(SAE) 선행 조건부로만.
- **WA-LQR 게인(W)**: exp4 조건부 항목, 게이트 미통과 상태로 미시도.
- **fut β≥1.5 증량**: 위약 미초과 확인 후 기대값 하락으로 미실행.
- **beer·mixer steering**: 진단 약(0.62)/중간(0.61) — drawer 결론이 음성이라 착수 안 함.
- **selection(고르기)**: exp5-4 로 이관 — 별도 세션에서 Phase A 게이트 발동, **13/13 은
  seed 주효과(암기) 판정으로 중단**([[exp5-4-noise-selection-verdict]]). 본 세션의
  seed-공유 부풀림 경고가 실증된 셈.
- 떨림 보강 재실행(구 permanent 160 전량 kin) — β sweep 의 A0_kin 40판으로 대체, 전량은 미실행.

### 4.3 데이터·인프라 자산 (위치)

| 자산 | 위치 | 규모 |
|---|---|---|
| drawer/beer scene-matched 원본 pkl | 승준 HDD `datasets/.../scene_matched_exp41/` | 320판 143GB |
| mixer scene-matched 원본 pkl | 승준 `exp5_3_mixer_sm/` + **home 원본 보존** | 160판 97GB |
| 축약 NPZ (진단용, record별 [L,D]) | 승준 `~/sm_npz/`·`~/sm_npz_mixer/` | ~2.8GB |
| fit cache (토큰 해상도) | 승준 `~/exp53_npz/fit_cache_L12.npz` | 재fit 수초 |
| steering NPZ registry (LOO 8fold) | 승준 `~/exp53_npz/`·home `~/exp53_npz/deploy/` — permanent(per-scene setpoint)/gated/permanent_fut/gated_fut/placebo_fut | 189+18+… 개 |
| eval 산출 (9 arm, 487판, json+mp4) | home `outputs/eval/robocasa/groot_n15/exp5_3/` | jerk 는 β sweep 이후 arm 만 |
| 셀 단위 paired 원자료 | repo 루트 `exp5-3_bsweep_paired.tsv` | 40행 |
| 진단/게이트 JSON | 승준 `~/sm_*.json`·`~/exp53_npz/fit_report.json`·`placebo_report.json` | — |
| 스크립트 | `scripts/safe/groot_n15/robocasa/steer/exp5_3/` (fit/eval/sweep/위약/mixer수집/집계) | 커밋됨 |

## 5. Confound audit

| # | 게이트 | 판정 | 근거 |
|---|---|---|---|
| 1 | 길이 | 통과 | 진단 = 공통 창(cap=min len); steer 판정 = SR(길이 무관) |
| 2 | task | 통과 | cell 내 분석 |
| 3 | instruction | 통과 | drawer=right 검증 seed·mixer=1종 |
| 4 | in-sample | 통과 | LOO-by-seed fit 실물(fold별 NPZ sha 상이·fit_report 기록) |
| 5 | pooling | 부분 | 진단 창 평균 병용 — fixed-t 곡선으로 보강(문서 30절 아님, drawer 선행 분석) |
| 6 | phase/dwell | 통과(국소) | mixer disengage dwell 0.43; contact-head 는 dwell 0.72 로 **교란 잔존** |
| 7 | 관측≠인과 | **실증** | 진단 0.84 인 방향이 β=1.0 개입에선 −51판 — read≠write |
| 8 | scene 국소 | 해당 | permanent 해악은 13/20 scene 전부 음(−) — scene-일관 해악 |

주장 강도: drawer steer = **intervention effect (해악 방향, β=1.0 한정)**. mixer 진단 =
diagnostic evidence.
