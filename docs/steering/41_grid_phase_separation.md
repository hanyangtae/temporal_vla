# 41 — 새 base grid 의 phase 별 succ/fail 분리도 지도 (GT vs intrinsic phase)

2026-08-12. 새 N1.5 grid(930판, 9 instruction × scene 10 × noise 10 + apple 30)에서
phase 조건부 성공/실패 분리가 **어느 layer·어느 토큰에서** 보이는지, phase 정의를
**GT(event labeler)** 와 **activation 자체(KMeans 클러스터)** 로 나눠 측정한 결과.
분석 코드 `scripts/analysis/grid_phase/` 4종 + `seed_memo_probe.py`, 실행은 전부 승준
(데이터 소재지). 산출물: `outputs/analysis/grid_phase/{rung0,rung1,rung2,rung3,seed_probe}/`.

**탐색 라운드다** — 다중비교 보정 없음, n_perm 100~500, 위약 없음. 단 길이 통제와
scene-LOSO 는 전 셀 공통.

## 0. 프로토콜 (통계량 단일화)

- 통계량은 하나: **풀링 + held-out scene 내부 중심화 LOSO AUROC** (`g2_residual_read.loso`)
  + **scene 내 라벨 순열 z** (docs/32 §통계량 혼용 경고 준수. "scene별 AUROC 평균"은 안 씀).
  혼재 scene(성공·실패 공존)만 평가에 들어간다.
- 분석 단위 = (episode, phase) 1행. **equal-budget**: (task, phase)별
  B = 그 phase 도달 episode 들의 phase 내 record 수 최소값, 각 episode 는 그 phase
  **앞쪽 B개** record 의 평균. B<3 이면 skip. dwell(실패가 오래 머묾)이 값에 못 들어온다.
- 모든 셀에 `length_auroc`(phase record 수만의 AUROC) 병기 — 길이 혼입 자가감사.
- feature: Tier A = [7 DiT layer × 4 denoise × 4 seg(state/future/action/all토큰평균) × 1536] + VL(2048).
  Tier B = 49토큰 개별 (상위 조합만).

## 1. 배관 게이트 (Rung 0)

이전 라운드 앵커(scene-matched drawer 0.84) 프로토콜(창 38 풀링)로 새 데이터의
drawer-left L12 future = **0.779 (z 2.83)**, 길이 AUROC 0.33. scene seed·fold 축이 다른
새 수집에서 같은 대역 → 추출·통계 배관 정상 판정.

## 2. GT phase 조건부 지도 (Rung 1)

task 별 최고 셀 (seg future/all, denoise mean):

| task | 무phase(창38) 최고 | GT phase 최고 | 판정 |
|---|---|---|---|
| OvenRack/out | L2 future 0.94 / z3.5 | **reach-to-rack L2 future 0.92 / z3.4 (lenA 0.06, B14)**, contact-rack 0.92~0.95 / z3.5 (lenA 0.62 ⚠) | ★ phase 특정까지 성공. 접근 단계에서 이미 갈림 |
| PPCC/bread | L4 all 0.85 / z4.1 | reach-to-object L15 action 0.69 / z2.3 (lenA 0.01) | 신호는 실재하나 phase로 자르면 약해짐(전반 분산형) |
| PPCC/candle | L8 future 0.77 / z4.0 | reach-to-object 0.62 / z1.8 | 〃 |
| OpenDrawer/left | L0 future 0.82 / z3.1 | disengage 0.94 / z2.3 (B3, mix2 ⚠) | 탐색 수준 |
| OpenDrawer/right | L12 state 0.74 / z2.4 | reach-to-handle 0.67 / z1.6 | 약함 |
| PPCC/jug | L8 state 0.82 / z3.2 | reach-to-object 0.60 / z1.3 | 약함 (SR 0.18, 혼재 6) |
| PPCC/marshmallow | L8 state 0.71 / z2.4 | reach-to-object 0.63 / z1.6 | 약함 |
| DishwasherRack/out | L15 future 0.77 / z2.3 | reach-to-rack 0.67 / z1.5 | 약함 |
| CoffeeSetupMug | 0.82 / z1.8 | 무신호 | 혼재 scene 2개(SR 0.09) — 판정 불가 |

- **layer 축은 평평하다**: OvenRack 은 L0~L12 전부 z 3.2~3.6. 특정 layer 국소화 없음.
- **denoise 축도 평평** (1b): K-mean ≈ k3 ≈ k0 (z 차이 ≤0.3). denoise step 선택은 중요치 않음.
- PPCC 계열 공통: 남는 phase 신호는 **reach(초반)** 에 몰림 — 초기조건형 관찰과 정합.
- phase 로 자르면 z 가 내려가는 task 다수: (a) 신호가 특정 phase 에 안 몰려 있고
  (b) phase 도달 episode·창 축소로 검정력 손실이 겹친 결과.

## 3. "실패 잦은 seed 암기" 진단 (seed_memo_probe)

구 drawer 사례(scene SR 암기)·exp5-4(noise seed 암기) 재발 여부를 명시 검증:

| 셀 | act LOSO (z) | scene-SR 베이스라인* | within-scene AUROC (z, p) | t=0 단독 |
|---|---|---|---|---|
| OvenRack L2 future reach | 0.90 (3.1) | **0.49** | **0.86** (2.5, .002) | 0.76 |
| OvenRack L8 state contact | 0.92 (3.2) | **0.49** | **0.90** (2.9, <.001) | 0.86 |
| drawer-left L12 grasp | 0.61 (1.2) | 0.54 | 0.62 (1.1, .16) | 0.64 |
| bread L15 action reach | 0.69 (2.3) | 0.39 | **0.71** (2.0, .006) | 0.53 |

*평가 대상(혼재 scene) 안에서 "그 scene 의 자기 제외 실패율"만으로 예측한 AUROC.

- **scene 암기 아님**: 평가 집합 내 scene 정체성 판별력 0.49(무작위)인데 activation 은
  0.90; 같은 scene 안 쌍 비교(within-scene)에서도 0.86~0.90 유지 (scene 암기만 심은
  합성에서는 0.58 로 붕괴함을 확인).
- **noise seed 주효과 없음**: instruction×scene 블록 내 순열 1000회 p=0.37.
- 유보: OvenRack 혼재 scene 4개 — "암기 아님"은 견고하나 **0.90 이라는 크기는 4-scene
  추정치**. 조기성은 셀마다 갈림 — OvenRack 은 t=0 부터(초기조건형), bread 는 t=0 무신호
  (실행표류형, phase 내 누적).

## 4. 토큰 해상도 (Rung 2, 상위 task×layer 만)

- **국소 마법 토큰은 없다.** OvenRack reach L2: future 토큰 32개 평균 z +3.25 (max 3.47),
  action 토큰 +2.41, state +3.42 — future 세그먼트 전반이 고르게 나름.
- bread 의 reach 신호는 **action horizon 끝 토큰(43~48)** 에서 최대 (z 2.4) — 먼 미래
  action 예측 토큰이 실패를 먼저 반영하는 모양새. 후속 관찰 가치.
- drawer-left disengage 는 토큰 10/30 에서 0.96~0.98 (z 2.4, B3) — 표본 얇음.

## 5. GT vs intrinsic phase (Rung 3) — 핵심 질문

같은 좌표(task × layer × seg)에서 phase 정의만 교체:

| task | GT | intrinsic k8(per-task) | intrinsic k24(global) | tq4 q0(초반) | shuffle |
|---|---|---|---|---|---|
| OvenRack/out | 0.95/z3.5 | 0.96/z3.4 | 0.96/z3.6 | 0.95/z3.5 | 0.85/z3.2 |
| OpenDrawer/left | 0.94/z2.3 | **0.93/z4.5** | 0.65/z1.5 | 0.63/z1.4 | 0.67/z1.9 |
| PPCC/bread | 0.67/z2.3 | **0.77/z3.4** | 0.72/z2.4 | 0.66/z2.1 | 0.77/z3.1 |
| PPCC/candle | 0.62/z1.8 | **0.73/z2.6** | 0.66/z2.5 | 0.62/z1.8 | 0.67/z2.3 |
| PPCC/marshmallow | 0.92/z1.5 | **0.75/z2.9** | 0.91/z2.0 | 0.64/z1.6 | 0.71/z2.5 |
| OpenDrawer/right | 0.67/z1.6 | 0.40/z−0.9 | 0.54/z0.3 | 0.72/z2.0 | 0.72/z2.1 |
| 나머지(jug·dish·coffee) | 약 | 약 | 약 | 약 | 약 |

- **intrinsic(per-task k8)이 GT 를 이기는 task 4/9** (drawer-left, bread, candle,
  marshmallow), 동급 1(OvenRack), 지는 것 1(drawer-right). activation 이 스스로 정의한
  상태 단위가 라벨러의 이벤트 단위보다 succ/fail 을 더 잘 층화하는 task 가 있다 —
  phase-matched steering 의 gating 신호로 intrinsic 클러스터를 쓸 근거.
- **global k24 는 per-task k8 보다 대체로 못하다** — phase 구조가 task 특이적.
- ⚠ **time-quantile 의 겉보기 압승은 사후 판독이다**: tq4 전체 최고 z 4~7 은 전부
  **q3(종반)** 에서 나온다 (task 전부 q3 0.92~1.00). 종반 상태는 결말의 흔적이라
  분리가 자명하다(성공=닫힘 직전, 실패=늦은 표류). 예측용 공정 비교는 q0 인데 q0 는
  GT/intrinsic 을 못 이긴다 (표의 tq4 열). 종반 셀은 검출기가 아니라 판정기 성능이다.
- shuffle(구간 순서 셔플)이 z 2~3 을 유지하는 task 들 = 신호가 phase 귀속 없이 에피소드
  전반에 퍼져 있다는 뜻 — §2 의 "무phase 기준선이 세다"와 같은 현상.

### intrinsic ↔ GT 정렬 (per-task k8)

- MI(cluster;GT) 0.25~0.76 bits, purity 0.54~0.93 — **중간 정렬**: GT 를 재발견하지도,
  무관하지도 않음. margin(vs clock) 0.15~0.51 bits 양수 — 시간 등분보다 정보 있음
  (동료 결과 재현, apple 은 −0.63 = 성공 30판뿐이라 퇴화).
- boundary F1 0.08~0.22 (z −2~+11, task 편차 큼) — 경계 시점은 GT 이벤트와 약하게만 일치.
- ⚠ mi_scene 0.2~1.0 bits — 클러스터에 scene 성분 혼입(특히 dishwasher·drawer-right).
  intrinsic 을 gating 에 쓸 땐 scene 오염 잔차화 필요.

## 6. 종합 판정

1. **phase 조건부 분리가 실재하고 scene/seed 암기가 아니다** (probe 3종 통과).
   가장 강한 것은 **OvenRack reach 단계** (0.86~0.90 within-scene, t=0 부터, 길이무관).
2. **phase 정의는 task 마다 최적이 다르다**: OvenRack 은 GT=intrinsic 동급,
   drawer-left/bread/candle/marshmallow 는 **intrinsic 이 우세**, 신호가 전반 분산형인
   task 는 어느 정의로도 조건화 이득이 없다.
3. layer·denoise·토큰 축은 대체로 평평 — "어디서"보다 "언제(phase)"가 지배 축.
4. 종반 분리(z 4~7)는 사후 판독이라 steering 트리거로는 무용, 예측용은 q0/reach 대역.

## 7. 다음 단계 후보

- steering 연결 1순위: **OvenRack reach-phase 조건부** (GT 라벨러가 온라인 판독 가능) +
  drawer-left/bread 의 **intrinsic k8 gating** (kNN transfer 로 온라인화, scene 잔차화 선행).
- 혼재 scene 을 늘리려면 noise 연장(m10→m40, plan_id 불변)이 곧 검정력 — 특히 OvenRack
  (혼재 4) 우선.
- bread 의 "먼 미래 action 토큰이 먼저 반영" 관찰은 onset-relative 분석으로 추적 가능
  (wrong-grasp 이벤트 라벨 있는 task 한정).

## 부록 — 재현 커맨드

승준에서 (코드 sync 후):
```
extract:  extract_grid_matrix.py --grid-root <store>/grid --index-tsv <store>/index/rollouts.tsv \
          --out-dir <store>/analysis/grid_phase --instructions all --tier segA --workers 4
engine:   phase_sep_matrix.py --shards <segA> --layers all --segs state,future,action,all --vl \
          --denoise mean --phase-def gt|labels:<npz>|time-quantile:K|shuffle --n-perm 100
intrinsic: intrinsic_phase.py --shards-dir <segA> --out <intrinsic> --scope per-task --k 6,8,12 \
          --align-out align.json
probe:    seed_memo_probe.py --shard <npz> --layer L --seg S --phase P / --all-noise-check <segA>
```
중간 산출물(shard NPZ 30GB + tokB 15GB + intrinsic)은 승준
`<store>/analysis/grid_phase/` 존치, JSON/TSV/PNG 만 로컬 회수.

---

## 8. rung4 추가 검증 (2026-08-12 오후): K sweep · grasp 시점 · 시점 분해

### 8.1 시간 구조가 지배 변수다 (quantile 분해)

time-quantile K등분(4/8/12)의 quantile별 mean z:

- **q0(시작 구간) ≈ 0** (tq4 +0.3, tq8 +0.0, tq12 −0.2) — OvenRack 제외 시작 시점 분리 없음.
- **q1 이후 전 quantile +3.2~4.1** — 상대위치 어디를 잘라도 강한 분리.

해석: 실패 신호는 대부분 task 에서 **시작 시점엔 없고 episode 전개와 함께 급격히
형성**된다(실행표류형 지배). 상대위치 매칭은 절대시각 매칭이 아니므로(실패 판의 q1 은
이미 env-step 90+), q1+ 의 분리는 상당 부분 진행 중인 실패 상태의 판독이다.
**OvenRack 만 t=0 부터 분리되는 초기조건형** — 이 구분이 steering 트리거 설계의 핵심 축.
따라서 tqK 는 "칸수 매칭 공정 대조" 역할을 못 한다(q0 외 전부 사후 오염).

### 8.2 intrinsic 검증 — 조기 셀 한정 비교 (relpos 중앙값 < 0.5)

클러스터/phase 별 episode 내 상대시점을 실측해 조기 셀만 비교:

| | GT | intr k6 | intr k8 | intr k12 |
|---|---|---|---|---|
| 조기 mean z | +0.99 | +0.80 | +1.06 | +1.08 |
| 후기 mean z | −0.10 | +0.46 | +1.02 | +0.14 |

- **조기 한정 시 intrinsic k8 과 GT 는 사실상 동급** (+1.06 vs +0.99). §5 의 "intrinsic
  우위 4/9"는 상당 부분 후기(사후) 클러스터 셀의 기여였다.
- 예외 = **drawer-left: 조기에서도 intrinsic 우세** (+2.1/max 4.5 vs GT +0.5/max 1.7)
  — intrinsic gating 의 실질 후보는 drawer-left 로 좁혀 읽는 것이 정직하다.
- k8 의 후기 mean +1.02 가 유일하게 조기와 같은 수준 = k8 클러스터의 우위분에 후기
  상태 판독이 섞여 있었다는 방증.

### 8.3 K sweep: k8 근방이 최적

전 셀 mean z: k6 +0.69 / **k8 +1.05** / k12 +0.75 / k16 +0.61 / k24 −0.30 / k32 +0.28.
k≥16 은 과분할로 클러스터당 record 부족(B<3 skip 급증)·상태 쪼개짐 → 퇴화.

### 8.4 grasp 시점 (--align end)

- PPCC reach 끝(=grasp 시도 구간) 정렬: bread 0.98/z5.6, candle 0.94/z5.2,
  marshmallow 0.90/z4.3, jug 0.98/z4.2 (first 정렬 대비 z +3 안팎 상승, lenA 0.00, 같은 B).
  ⚠ 단 PPCC 실패 다수는 grasp 미도달이라 reach 가 timeout 까지 이어짐 → end 정렬은
  "성공의 grasp 순간 vs 실패의 늦은 표류" 비교가 섞인다(절대시각 불일치). 액면 채택 금지,
  "신호가 reach 진행에 따라 누적 상승"으로만 읽을 것.
- drawer grasp-handle 끝: left 0.94/z4.7, right 0.94/z3.5 — **양쪽 다 잡은 판끼리의
  비교**라 더 의미 있음: 잡은 뒤 당기기 직전 상태가 이미 다르다. (시각 불일치는 잔존)

### 8.5 종합 갱신

- 예측적(조기) 신호로 확립: **OvenRack (t=0부터)** 하나. drawer-left 는 intrinsic 조기
  셀에서 중간 강도. 나머지는 실행표류형 — 신호는 실재하나 "실패가 전개된 뒤" 강해짐.
- 실행표류형 task 의 개입 여지는 "일찍 감지"가 아니라 **전개 중 감지→중단/재샘플**
  (grasp 직전 대역이 가장 읽기 좋은 창).
- intrinsic gating 후보: drawer-left 한정 (k8, scene 잔차화 선행).
