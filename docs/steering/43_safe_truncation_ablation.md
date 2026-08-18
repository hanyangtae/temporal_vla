# 43. SAFE detector 학습 길이 절제 ablation (rollout vs phase 단위)

2026-08-13, branch `exp/safe-length-ablation` (feat/online-gated-pipe에서 분기),
스크립트 `scripts/analysis/grid_phase/failure_detector_sim.py` (`--truncate-train`).
데이터 = grid 930판 segA shard (승준), 3 seed(scene split 회전) × 3 절제 조건 × LSTM/MLP.

## 질문

failure detector 학습 시퀀스를 길이 절제하면 성능이 어떻게 변하나?
(full 학습은 "길면 실패"만 배울 위험 — seen18 검증에서 length-only AUROC 0.996 동급 사고)

- `none`: full 시퀀스 학습 (기준)
- `rollout`: task별 W = 성공 ep 길이 ceil(μ+1σ)로 train·calib 절단
- `phase-gt`: GT phase별 dwell cap(성공 dwell ceil(μ+1σ), fit 규약과 동일)으로 절단
- **test는 항상 full** (온라인 현실 모사). 지표: TPR/FPR(functional-CP 발화),
  `tpr_before_W`(W 이전 발화율 = timer가 원리상 모르는 시점의 조기 검출), 고정 판정시각
  AUROC(td, 종료판 제외), timer 기준선(t≥W 발화).

## 결과

### 1) 총괄(pooled pertask, α=0.2, 3-seed 평균) — 절제는 검출률을 바꾸지 않는다

| 조건 | TPR | FPR | td10 |
|---|---|---|---|
| none | 0.92 | 0.31 | 0.77 |
| rollout | 0.91 | 0.32 | 0.78 |
| phase-gt | 0.91 | 0.38 | 0.74 |
| timer(길이만) | 1.00 | 0.12 | — |

seed0 단독에서 보였던 FPR 개선(0.60→0.42)은 **3-seed에서 비재현** — pooled 검출 성능은
절제 무관. (pooled 행은 task 혼합이라 참고치.)

### 2) 일관 효과는 조기성(preW) — 절제가 발화 시점을 앞당긴다

3-seed 평균, pertask lstm best-α (TPR/FPR | preW):

| task | none | rollout | phase-gt |
|---|---|---|---|
| OpenDrawer/left | 0.93/0.10 · pW **0.17** | 1.00/0.13 · pW **0.83** | 1.00/0.11 · pW **1.00** |
| PPCC/bread | 0.89/0.35 · pW 0.48 | 0.96/0.31 · pW 0.85 | 0.92/**0.23** · pW **0.92** |
| PPCC/marshmallow | 1.00/0.22 · pW 0.94 | 1.00/0.23 · pW 1.00 | 1.00/**0.11** · pW 1.00 |
| PPCC/candle | 1.00/0.14 · pW 0.33 | 1.00/0.12 · pW 0.58 | 1.00/0.17 · pW 0.58 |
| DishwasherRack/out | 0.67/0.00 · pW 0.33 | 0.67/0.00 · pW 0.48 | 0.55/0.00 · pW 0.48 |
| OvenRack/out | 0.97/**0.00** · pW 0.62 | 1.00/0.25 · pW 1.00 | 0.94/0.00 · pW **0.24** |
| OpenDrawer/right | 1.00/0.49 | 0.67/0.47 | 1.00/0.47 — 전 조건 불량 |
| PPCC/jug | FPR 1.00 (succ 2판) — 판정 불가 |

- **실행표류형 task(drawer-left·PPCC)에서 절제가 preW를 크게 올림** (drawer-left
  0.17→1.00, bread 0.48→0.92) — TPR/FPR 유지한 채 발화만 이른 시점으로. 기전: full 학습
  score는 길이에 따라 서서히 오르는 성분을 포함 → CP 밴드도 같이 부풀어 발화가 늦음;
  절제하면 score 상승이 feature 이벤트에 묶임.
- **phase-gt ≥ rollout**: preW 동급~우세 + 일부 task FPR 추가 감소(marshmallow 0.11,
  bread 0.23). phase-dwell 초과 성분까지 지워도 검출이 안 죽음 = 남는 신호는 feature 기반.
- **초기조건형 OvenRack엔 절제가 해로움** (rollout FPR 0→0.25, phase-gt preW 0.62→0.24)
  — 41 라운드의 초기조건형/실행표류형 이분과 정합: 초기조건형 신호는 절제로 지워지는
  구간(초반 궤적 전체)에 있음.

### 3) OvenRack detector는 scene에 따라 방향이 뒤집힌다 — online 후보 제외

조기 td AUROC(lstm): seed0(test scene 7,9) td5–td20 **0.02~0.10(역전)**,
seed2(test scene 0,5) td5 **1.00(정방향)**. held-out scene에 따라 부호가 뒤집힘 =
cross-scene 방향 일반화 실패(scene-특이 feature 학습). within-scene 0.86–0.90(41 §)과
모순 아님 — scene 잔차화 없이는 detector로 못 씀.

### 4) timer가 강한 기준선

실패=timeout이라 "W 넘으면 발화"만으로 TPR 1.00/FPR 0.12. detector의 존재 이유는
**W보다 60~85 record 이른 발화**뿐이며, timer FPR(0.12)를 이기면서 이른 task는
dishwasher(0.00)·drawer-left(0.10~0.13)·marshmallow(phase-gt 0.11) 정도.

### 5) 기전 분해 — 조기 발화는 dwell 초과가 아니라 내용 신호 (3-seed 합산)

phase-gt 학습 모델은 "phase에 오래 머무는 구간"을 학습에서 본 적이 없으므로, 조기 발화가
실은 "dwell cap 초과 = 학습 지지집합 이탈"이라는 dwell 신호 재활용일 가능성을 분해
(`fire_phase_decomp.py`: 발화 순간의 현재-phase 누적 dwell vs cap, lstm pertask α=0.2):

- pooled: 발화한 실패판 173개 중 **cap 초과 후 발화는 31%뿐**, median dwell/cap **0.57**
  (성공 dwell 예산의 절반 시점에 발화) — none과 동일 수준(27%/0.57). → **dwell-OOD 기각,
  발화의 ~70%는 dwell이 정상 범위일 때 내용으로 우는 것.**
- 조기성 이득이 컸던 task일수록 내용 기전이 깨끗함: drawer-left over-cap **0%**(none은
  62%) r0.36, bread 0% r0.55, marshmallow 0% r0.14 — 절제가 발화를 dwell 경계가 아니라
  **정상-dwell 구간 안쪽으로** 당김.
- OvenRack은 phase-gt에서 over-cap 44%·r0.93로 dwell 경계에 걸림 — 초기조건형에서
  절제가 해로운 것과 정합.
- 한계: dwell은 GT phase 기준 측정이며, 더 미세한 반복-패턴 신호 사용 가능성까지
  배제하진 않음.

### 6) 학습 데이터량·영상 확인 (`trunc_budget.py`, `export_fire_scores.py`+`render_fire_overlay.py`)

- **데이터량**: 절제로 버려진 episode 0 (판 수 불변, 길이만 감소). 학습 record는 none 100%
  → rollout 68.7% → phase-gt **57.6%** (task별 4~62% 감소, 실패 timeout 비율에 비례).
  즉 phase-gt는 **가장 적은 데이터로 같은 검출률 + 더 나은 조기성** — 성능 차가 데이터량
  이득이라는 해석은 성립 안 함 (단 mode 간 미세 차이와 데이터량의 완전 분리는 불가).
- **오버레이 영상 27개** (9 ep × 3 mode, seed0 test, pertask lstm α=0.2):
  `outputs/analysis/grid_phase/fire_videos/` — 상단 instruction·좌측 score/밴드·발화 후
  빨간 테두리. frame↔record 정렬은 수집 규약(5 step/record, 2 step/frame)으로 검증
  (27개 전부 기대 frame 수 일치).
  - 관찰: 성공판 오발화가 none은 3/3 발화 vs rollout/phase-gt는 대체로 억제(표본 3판 —
    집계 FPR은 mode 무관이므로 과일반화 금지). drawer-left s2n0은 mode 간 발화가
    15↔42로 갈림(판별 변동 큼).
  - **OvenRack phase-gt는 t_fire=1(첫 step 상시 발화)** — §3의 scene 역전과 함께,
    OvenRack에서 절제 detector는 밴드가 사실상 붕괴. 제외 판정 재확인.

## Confound 감사

| 게이트 | 판정 | 근거 |
|---|---|---|
| 길이 | 통과(설계) | test full-seq, 판정은 fixed-td(종료판 제외)·preW·timer 병기 |
| task identity | 통과 | pertask 행만 판정, pooled는 참고 |
| instruction | N/A | slug당 canonical 1종 |
| in-sample | 통과 | scene 단위 train/calib/test 분리(TSV 기록), 3 seed 회전 |
| pooling | 통과 | per-record 시퀀스 |
| phase/dwell | 실험 대상 | phase-gt 조건이 dwell 통제 자체 |
| 관찰≠인과 | 라벨 | detector performance(시뮬) — 개입 효과 아님 |
| scene-국소 | 부분 | test 2 scene×3 seed; drawer-left 실패 2~4판/seed 소표본, jug·coffee·apple 판정 불가 |

## 파이프 함의 (exp6)

1. detector 학습은 **절제 기본값 채택** 권고 — phase-gt(fit과 동일 dwell 규약) 우선,
   rollout이 차선. 이유는 검출률이 아니라 **조기성**(개입 시간 예산).
2. 단 OvenRack(초기조건형)은 절제 없이 + scene 잔차화 검토, 또는 detector 대신
   phase-gating 상시 arm으로.
3. "phase별 감지"는 불필요 확인 — phase는 개입 라우팅에서만. detector는 task당 1개.
4. eval에서 timer arm(t≥W 개입)을 반드시 대조로 — detector 개입의 부가가치는
   "이른 개입이 SR을 더 살리는가"로만 증명됨.

### 7) LOTO task 전이 (zero-shot 9-fold × 3절제 × 3seed) — 전이는 절제로도 살아나지 않음

각 task를 완전히 배제(학습·표준화·CP 보정에 0판)하고 나머지 task로 학습 → held-out task
전 episode 평가 (`--arm loto`, CP 밴드는 train-task 성공 풀링). 3-seed 평균:

- **pooled 조기 AUROC(td10) = 0.43~0.47, 전 절제 조건 동일 = chance.** phase 단위
  길이 통제(phase-gt)도 task 전이를 만들지 못한다. (td20도 0.49~0.55.)
- **방향이 task마다 반전**: drawer-right td10 0.29~0.41(역방향) vs marshmallow 0.72~0.78
  (정방향) — 실패 표현의 방향이 task-특이적. 41 라운드의 task별 최적 셀 상이·과거
  cross-task conceptor 공유 음성과 정합.
- **유일한 생존 = 같은 env family 내 instruction 공유**: 전이가 chance를 넘는 fold는
  PPCC 형제(bread/candle/jug/apple)를 학습에 포함한 PPCC held-out들(marshmallow td10
  0.75+, jug/bread td20 0.7~0.8)뿐. → detector는 **task(env) 단위 분리 + task 내
  instruction 공유**가 근거 있는 구성 (family-pooled vs per-variant head-to-head는 미실행).
- 문헌 대조: SAFE(LIBERO)·RL²(SIMPLER, 우리 풀 재현 docs/steering/38)의 공유 detector
  성립은 근접-도메인 task 가족 + seen 한정 — RL²는 플랫폼만 바뀌어도 번들 detector가
  실패 36.5% 미검출(재학습으로만 복원). 우리 RoboCasa는 task마다 부엌 자체가 다름
  (같은 seed도 layout/style 상이 실측)이라 공유 전제가 애초에 없음. seen18 재현
  (unseen 0.434)의 재확인.

## 후속 후보 (미실행)

- **진행도-사분위 절제**: 연산자-설계 세션 실측(2026-08-14, Notion 3bc63918…)에 따르면 GT
  phase 내부에도 1차원 진행 경로가 있음(경로/관반경 1.3~3.5배, held-out 슬로프 +1.2,
  실패=경로 위 정체). phase-gt보다 fine한 절제 입도로 "경로 위 진행도 bin"(`--truncate-train
  progress-quantile` 류)이 unsupervised cluster보다 순서·단조 제약이 성립하는 후보.
  41의 intrinsic k8 우위의 실체일 가능성.

## 산출물

- 로컬: `outputs/analysis/grid_phase/detector_trunc{,_s1,_s2}/<mode>/sim_summary.tsv`
  (+sim_detail.json, ckpt .pt)
- 승준: `~/workspace/temporal_vla_safeablate/outputs/analysis/grid_phase/` (동일)
- 판독 스크립트: 세션 tmp (일회성) — 표는 본 문서에 고정
