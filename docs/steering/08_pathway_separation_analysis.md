# Phase 3: Pathway 분리력 분석 (DiT 32-layer + VL vs DiT)

> **⚠ 반증됨 (2026-07-29) — §2.1의 "VL이 먼저 실패를 감지한다" 결론은 폐기한다.**
> 근거 두 가지. ① 효과 크기가 없다: t=8에서 VL−DiT AUROC 차이가 **+0.013**(0.713 vs 0.701),
> t=12에서 −0.007로 사실상 동등하다. ② 창이 불일치한다: 이 비교는 VL을 t≤8 풀링으로,
> DiT를 다른 창에서 재고 있어 **DiT를 이른 창에서 과소측정**한다
> ([`15_research_structure.md`](15_research_structure.md) C3 경고 참조).
> 이 문서는 "그때 이렇게 측정했다"는 **기록으로만 보존**한다. 수치·방법은 유효하나
> §2.1 결론과 그로부터 파생된 "VL early / DiT late" 서술은 인용하지 말 것.

> Phase 3 결과 통합 문서. §1 = DiT 32-layer pre-failure 분리력(구 `08_phase3_dit32_separation`),
> §2 = VL(goal) vs DiT(motor) 정렬 비교(구 `09_phase3_vl_dit_comparison`), §3 = 종합
> (goal-vs-motor task 분열), §4 = caveat. 방향 단일 출처는
> [`14_pathway_phase_online_steering.md`](14_pathway_phase_online_steering.md).
>
> ⚠ instruction confound caveat → 구 `11_instruction_confound`(아카이브)
> (헤드라인 VL AUROC가 instruction 쏠림 아티팩트일 수 있음 — §4 참조).

---

## §1. DiT 32-layer pre-failure 분리력

작성: 2026-06-02. 대상 run: `target_atomic_moderate10_multilayer_perT_100ep` (1000 ep, 32-layer per-token).
분석 스크립트: `scripts/safe/groot_n16/robocasa/analyze/pathway_separation.py`
산출: `…/analysis/pathway_separation/pathway_separation.json`

### 방법

- **길이통제**: 고정 t(4,8,12,16,20) step까지만 feature pool, 길이>=t 인 rollout만 사용 ([[truncation-length-standard]]).
- **표상**: DiT transformer_blocks[L] valid-16 action token mean-pool → `[D=1536]` per-step → 첫 t step mean.
- **분리력**: PCA(SVD, n=30)→LDA 방향 투영→Mann-Whitney AUROC, 5-fold CV, task 내 succ/fail.
- **기준선**: length-only AUROC (step count 단독).

### 결과 요약

| t | length-only | DiT best layer | task-avg AUROC |
|---|---|---|---|
| 4 | 0.999 | L24 | 0.592 |
| 8 | 0.999 | L26 | 0.635 |
| **12** | 0.999 | **L23~L25** | **0.751** |
| 16 | 0.998 | L23 | 0.711 |
| 20 | 0.997 | L31 | 0.765 |

**best 종합**: t=12~20, L23~L31 구간이 peak. t=12 선택 (실시간 steer 개입 여지 확보).

### Task별 (t=12, best DiT layer)

| task | n | fail | best layer | AUROC |
|---|---|---|---|---|
| CloseToasterOvenDoor | 95 | 50 | L28 | 0.709 |
| NavigateKitchen | 93 | 59 | L30 | 0.598 |
| OpenCabinet | 100 | 51 | L26 | **0.937** |
| OpenDrawer | 98 | 66 | L24 | **0.884** |
| PickPlaceCounterToCabinet | 94 | 33 | L31 | **0.872** |
| PickPlaceCounterToStove | 99 | 34 | L25 | **0.843** |
| PickPlaceDrawerToCounter | 99 | 61 | L23 | 0.668 |
| SlideDishwasherRack | 92 | 43 | L23 | 0.731 |
| TurnOnMicrowave | 81 | 45 | L31 | 0.673 |
| TurnOnSinkFaucet | 98 | 73 | L25 | 0.706 |

### 해석

1. **신호 실재**: 길이 통제 후에도 task-avg 0.75 (t=12). SAFE 공정 metric val_seen 0.683보다 높음.
   LDA 프로브 직접 접근이 LSTM보다 신호를 더 잘 포착.
2. **후반층 집중**: 신호는 L20+ 집중. 초기 단계(t=4)에선 약(0.59), t=12에서 peak.
   → NOTALL의 "motor program은 trajectory 초기에 commit, 후기 DiT 불필요"와 일치 (Table 15).
3. **Task 이질성 큼**: OpenCabinet 0.937 vs NavigateKitchen 0.598. DiT만으로는 unseen 일반화 어려움.
   → SAFE unseen chance 결과와 일관. **VL(goal pathway) 추가로 이 이질성이 줄어드는지가 핵심** (→ §2).
4. **길이 confound 완전 통제됨**: length-only 0.997~0.999이지만 AUROC가 그보다 낮음 → 고정-t 방법 정상 동작.

---

## §2. VL(goal) vs DiT(motor) pathway 분리력 비교

작성: 2026-06-03. 데이터: `target_atomic_moderate10_pathway_pertoken_100ep` (1000 ep, VL+DiT-7layer 정렬).
비교 기준: `target_atomic_moderate10_multilayer_perT_100ep` (DiT-32layer, §1).

### 2.1 핵심 발견: VL이 더 이른 시점에 신호를 냄

| t | VL(goal) | DiT-b31(motor) | delta(VL-DiT) |
|---|---|---|---|
| **t=4** | **0.677** | 0.648 | **+0.029 (VL 우위)** |
| **t=8** | **0.713** | 0.701 | **+0.013 (VL 우위)** |
| t=12 | 0.745 | 0.752 | -0.007 (동등) |
| t=16 | 0.724 | **0.754** | -0.030 (DiT 우위) |
| t=20 | 0.741 | 0.743 | -0.002 (동등) |

**~~결론~~ (반증됨 — 문서 상단 참조)**: ~~VL은 t≤8에서 DiT보다 먼저 실패를 감지. DiT는
t≥12에서 따라잡음. → 온라인 early steering에서 VL pathway가 개입 여지 더 큼.~~

**현재 판정**: 차이 +0.013은 효과라 부를 수 없고, 두 pathway를 서로 다른 창에서 잰
비교라 방향성 자체가 성립하지 않는다. **VL/DiT 사이에 감지 시점 차이가 있다고 말할 근거 없음.**

### 2.2 DiT 내 layer 패턴 (7 captured layers, t=12 avg)

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

### 2.3 Task별 비교 (t=12, permutation null95 포함)

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

### 2.4 NOTALL 가설 검증

NOTALL: "VLM pathways encode goal semantics('what'), expert pathways encode motor programs('how')"
→ **재현됨**: VL 신호가 먼저 뜨는 task = goal failure(문 열기/닫기 목표 오인), DiT 신호가 강한 task = motor failure(정밀 manipulation). SlideDishwasherRack(목표 zone으로 가는 방향 오류)이 VL 0.931로 압도적인 것이 특히 시사적. (⚠ 단 이 SlideDishwasherRack 0.931은 instruction confound 의심 — §4.)

---

## §3. 종합 — 타이밍·pathway·steering 타깃 선택

### 3.1 한 줄 종합

- **~~타이밍~~ (반증됨)**: ~~VL(goal)은 이른 t≤8에서 먼저 감지, DiT는 늦은 t≥12에서 따라잡음.~~
  → 문서 상단 참조. **pathway 간 감지 시점 차이는 근거 없음.** 유효하게 남는 것은 DiT 내부에서
  신호가 후반층(block 24, 31)에 집중된다는 층 방향 관찰뿐이다.
- **goal-vs-motor task 분열**: 실패 메커니즘이 task에 따라 갈린다. goal-type(방향/목표 오인) task는
  VL 우위, motor-type(정밀 조작) task는 DiT 우위, 일부 task(navigate/일부 PnP/sink)는 두 pathway
  모두 latent에서 선형으로 분리 안 됨. **VL/DiT를 따로 봐야 하는 근거는 타이밍이 아니라 이것** —
  실패 원인·case에 따라 어느 pathway를 써야 하는지가 달라진다.
- **VL을 써야 하는 대표 case** (원인 기반, 타이밍 기반 아님):
  - **카메라 섭동** — exp5-2에서 C1(카메라)은 DiT setM이 해악이고 VL 평균이동(`setpoint_vl`)이
    정합했다 (ppcc 약양성 3:0). 지각 입력이 흔들린 실패는 VL 쪽에서 잡아야 한다.
  - **wrong-grasp** — 다른 물체를 잡은 뒤 재탐색 구간에서 VL activation이 확연히 갈린다
    ([`22_wrong_grasp_vl_separation.md`](22_wrong_grasp_vl_separation.md)).
- NOTALL의 pathway 기능 분리(VL=goal, DiT=motor)는 RoboCasa rollout에서도 task 분열 형태로 재현됨.

### 3.2 Phase 4 steering 타깃 선택 (권장 전략)

> ⚠ 아래 권장안은 "VL이 t≤8에서 먼저 감지한다"는 **반증된 전제** 위에 작성됐다.
> 타이밍 근거(t≤8 / t≥12) 부분은 무효다. 실제 이후 라운드가 무엇을 했고 어떻게 됐는지는
> [`RESULTS.md`](RESULTS.md)를 본다. 아래는 당시 계획 기록으로만 남긴다.

1. **~~Primary: VL pathway (`action_head.vlln`), early intervention (t≤8)~~**
   - ~~이유: earliest signal, upstream of motor commitment~~ → **타이밍 근거 무효**
   - 대상 task: CloseToasterOvenDoor, SlideDishwasherRack (VL dominant) — task 분열 근거는 유효

2. **Secondary: DiT block 24 or 31, ~~t=12 이후~~**
   - 대상 task: OpenCabinet, OpenDrawer, PnP tasks (DiT dominant)

3. **Type-matched vs always-on 비교**:
   - always-on(VL, β small): 전체 평균 ΔSR 측정
   - ~~online(t≤8에서 VL anomaly 감지 후 steer 시작)~~ → 개입 시점은 phase/원인 기준으로 잡아야 한다
   - task별: VL-dominant task에서 VL steer, DiT-dominant task에서 DiT steer

4. **제외 고려 (steering 효과 기대 낮음)**: NavigateKitchen, PickPlaceDrawerToCounter, TurnOnSinkFaucet
   — 두 pathway 모두 유의 신호 없음. 실패 메커니즘이 latent에서 선형으로 분리 안 됨.

---

## §4. Caveats

- **⚠ instruction confound**: §2.3 헤드라인의 VL-우위 task(특히 SlideDishwasherRack VL 0.931)는
  failure 전조가 아니라 VL goal 토큰이 instruction(slide in/out)을 인코딩하고 그 instruction이
  성공/실패와 거의 1:1로 상관된 **아티팩트**일 가능성이 크다. 신뢰 가능한 신호는 instruction-balanced
  task(OpenDrawer left/right ~45% 균형, DiT 0.888)와 단일 instruction task(CloseToasterOvenDoor
  VL 0.800)에 한정. 상세 판정 → 구 `11_instruction_confound`(아카이브).
  fixed-instruction 재수집으로 재검증하는 계획 → 구 `11_phase4`(아카이브).
- **길이 confound**: 모든 분리력은 고정-t 길이통제에서만 유효(실패=항상 timeout이라 time-pooled
  분리는 길이 아티팩트). 근거 → [`01_seen18_latent_analysis.md`](01_seen18_latent_analysis.md).
- **직렬 pathway**: Eagle→VL-SA→DiT는 직렬 → VL/DiT "따로"가 진짜 독립이 아님. VL-OOD는 거의 항상
  DiT도 OOD로 만든다 → 진짜 질문은 "VL로 설명되는 것 이상의 DiT-OOD가 있나". 라우팅 타당성 판정은
  steering crossover 실험으로만 (→ `14`, `15`).
- **소표본 chance≠0.5**: AUROC 유의 판정은 permutation null95(§2.3 null95 열) 기준. CV+shuffle
  baseline 없이 in-sample 분리를 과신하지 말 것.
- **unseen 일반화 미확인**: task 이질성이 커(OpenCabinet 0.937 vs NavigateKitchen 0.598) DiT만으로는
  unseen 일반화 어려움(SAFE unseen chance와 일관).

---

## 남은 작업 (Phase 4)

- VL conceptor fit (W=success mean step, truncated) + SR eval
  - `fit_conceptor_steering.py --force-layer pathway=vl` 확장
  - eval matrix: VL β∈{0.1,0.3} × always-on/online × task-7(미유의 3개 제외)
  - 상세 실행 계획 → 구 `11_phase4`(아카이브)
- 아카이브: 두 raw_rollouts → kimseungjun@166.104.146.37:11112
