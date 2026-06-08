# Phase 3: VL(goal) vs DiT(motor) pathway 분리력 비교 결과

작성: 2026-06-03. 데이터: `target_atomic_moderate10_pathway_pertoken_100ep` (1000 ep, VL+DiT-7layer 정렬).
비교 기준: `target_atomic_moderate10_multilayer_perT_100ep` (DiT-32layer).

## 1. 핵심 발견: VL이 더 이른 시점에 신호를 냄

| t | VL(goal) | DiT-b31(motor) | delta(VL-DiT) |
|---|---|---|---|
| **t=4** | **0.677** | 0.648 | **+0.029 (VL 우위)** |
| **t=8** | **0.713** | 0.701 | **+0.013 (VL 우위)** |
| t=12 | 0.745 | 0.752 | -0.007 (동등) |
| t=16 | 0.724 | **0.754** | -0.030 (DiT 우위) |
| t=20 | 0.741 | 0.743 | -0.002 (동등) |

**결론**: VL은 t≤8(step 8까지)에서 DiT보다 먼저 실패를 감지. DiT는 t≥12에서 따라잡음.
→ **온라인 early steering에서 VL pathway가 개입 여지 더 큼** (motor commitment 이전 신호).

## 2. DiT 내 layer 패턴 (7 captured layers, t=12 avg)

| local idx | DiT block | AUROC |
|---|---|---|
| L0 | 0 | 0.631 |
| L1 | 2 | 0.632 |
| L2 | 4 | 0.636 |
| L3 | 8 | 0.655 |
| L4 | 16 | 0.724 |
| L5 | 24 | 0.752 |
| **L6** | **31** | **0.752** |

단조 증가 — **후반층(block 24, 31)이 best**. NOTALL Table 15 (DiT early-window ablation이 가장 파괴적, 후기 불필요)와 일치. COAST prior 결과(q̄ monotone 감소, 중간층 peak 없음)도 재확인.

## 3. Task별 비교 (t=12, permutation null95 포함)

| task | VL | DiT-b31 | null95 | VL sig | DiT sig | 판정 |
|---|---|---|---|---|---|---|
| CloseToasterOvenDoor | **0.800** | 0.761 | [0.30,0.64] | * | * | VL 우위 |
| NavigateKitchen | 0.591 | 0.492 | [0.39,0.64] | — | — | **둘 다 미유의** |
| OpenCabinet | 0.883 | **0.912** | [0.34,0.64] | * | * | DiT 우위 |
| OpenDrawer | 0.731 | **0.888** | [0.37,0.64] | * | * | DiT 우위 |
| PickPlaceCounterToCabinet | 0.861 | **0.907** | [0.37,0.62] | * | * | DiT 우위 |
| PickPlaceCounterToStove | 0.734 | **0.790** | [0.33,0.67] | * | * | DiT 우위 |
| PickPlaceDrawerToCounter | 0.597 | 0.626 | [0.32,0.64] | — | — | **둘 다 미유의** |
| SlideDishwasherRack | **0.931** | 0.829 | [0.27,0.63] | * | * | **VL 압도 (+0.10)** |
| TurnOnMicrowave | 0.765 | 0.774 | [0.33,0.65] | * | * | 동등 |
| TurnOnSinkFaucet | 0.560 | 0.540 | [0.31,0.68] | — | — | **둘 다 미유의** |

**VL 우위 (goal-type failure)**: CloseToasterOvenDoor, SlideDishwasherRack
**DiT 우위 (motor-type failure)**: OpenCabinet, OpenDrawer, PnPCounterToCabinet, PnPCounterToStove
**예측 불가 (3 task)**: NavigateKitchen, PickPlaceDrawerToCounter, TurnOnSinkFaucet

## 4. NOTALL 가설 검증

NOTALL: "VLM pathways encode goal semantics('what'), expert pathways encode motor programs('how')"
→ **재현됨**: VL 신호가 먼저 뜨는 task = goal failure(문 열기/닫기 목표 오인), DiT 신호가 강한 task = motor failure(정밀 manipulation). SlideDishwasherRack(목표 zone으로 가는 방향 오류)이 VL 0.931로 압도적인 것이 특히 시사적.

## 5. Phase 4 steering 타깃 선택

**권장 전략**:

1. **Primary: VL pathway (action_head.vlln), early intervention (t≤8)**
   - 이유: earliest signal, upstream of motor commitment, 1 forward에서 K=4 denoising 전파
   - 대상 task: CloseToasterOvenDoor, SlideDishwasherRack (VL dominant)

2. **Secondary: DiT block 24 or 31, t=12 이후**
   - 대상 task: OpenCabinet, OpenDrawer, PnP tasks (DiT dominant)

3. **Type-matched vs always-on 비교**:
   - always-on(VL, β small): 전체 평균 ΔSR 측정
   - online(t≤8에서 VL anomaly 감지 후 steer 시작): 개입 정밀도 ↑
   - task별: VL-dominant task에서 VL steer, DiT-dominant task에서 DiT steer

4. **제외 고려 (steering 효과 기대 낮음)**: NavigateKitchen, PickPlaceDrawerToCounter, TurnOnSinkFaucet
   — 두 pathway 모두 유의 신호 없음. 실패 메커니즘이 latent에서 선형으로 분리 안 됨.

## 6. 남은 작업

- Phase 4: VL conceptor fit (W=success mean step, truncated) + SR eval
  - `fit_conceptor_steering.py --force-layer pathway=vl`  확장 필요
  - eval matrix: VL β∈{0.1,0.3} × always-on/online × task-7(미유의 3개 제외)
- 아카이브: 두 raw_rollouts → kimseungjun@166.104.146.37:11112 (task #7)
