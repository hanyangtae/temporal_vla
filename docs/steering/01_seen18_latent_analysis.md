# seen18 GR00T-N1.6 RoboCasa 잠재공간 분석 정리

> **이 문서에서 살아남는 것은 두 가지다** (2026-07-30 정리).
> ① **길이 confound** — 실패는 항상 timeout이라 time-pooled 분리(AUROC 0.998)는 아티팩트다.
> 이후 모든 분리 주장의 전제이고, 이 문서가 그 원출처다.
> ② **도구·데이터 지도** — 어떤 run을 어떤 스크립트로 분석해 어디에 산출했는지.
> 나머지 개별 관찰(공유 실패 zone, onset regime 등)은 이후 라운드에서 scene·seed 통제가
> 들어가며 조건이 바뀌었으므로, 인용 전 [`RESULTS.md`](RESULTS.md)·[`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md) C1과 대조할 것.

작성: 2026-05-28 · 대상 데이터: `outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep`
도구: `scripts/safe/groot_n16/robocasa/vis/seen18.py` (단일 CLI) · 산출: `…/analysis/visualizations/`

> **목적**: 성공/실패 latent이 구분되는지, 어떤 조건에서 드러나는지 검증해 **latent steering**(succ/fail
> 구분 → steer로 SR↑)의 표현 측 근거를 마련. (연구 방향 = latent steering, TTA 무기한 연기.)

---

## 0. 데이터 개요
- **1800 rollout** = 18 atomic task × 100 ep. 전체 SR **0.537** (성공 967 / 실패 833) — 균형.
- task별 SR 0.15(CloseBlenderLid) ~ 0.82(CloseFridge·OpenStandMixerHead).
- feature: 각 env-step당 DiT pre-velocity action-token latent `[K=4,H=16,D=1024]` → K·H mean-pool → step당 1024-d.
- rollout 길이 5~45 step. 캐시: `analysis/feature_cache/pooled_all_hmean_dmean.npz`.

---

## 1. 길이 confound — 모든 해석의 전제 ⚠
- **실패는 100%가 45-step time limit까지 감(timeout), 성공은 전부 그보다 일찍 종료**(평균 17.7±8.4).
- **에피소드 길이만으로 실패 예측 AUROC = 0.998** (task별 0.98~1.00).
- ⇒ episode-mean / 시간-pooled feature의 높은 succ/fail 분리(예: Mahalanobis-centroid CV AUROC 0.96, episode-mean within-task ~0.92~0.98)는 **VLA가 실패를 "안다"는 증거가 아니라 길이/조기종료 아티팩트**.
- SAFE 논문이 T를 min 길이로 cap하는 이유와 동일. **이 데이터로 실패 인코딩을 물을 땐 반드시 길이 통제 필수.**
- 도구: `seen18.py length` → `length_confound/`.

---

## 2. 진짜 succ/fail 신호 (길이 통제 후) — 약하지만 실재
- **고정 절대 frame**(성공·실패 같은 frame, 살아있는 rollout만)에서 per-task CV logistic AUROC:
  - mean ~**0.6** 전 구간, t=10~12 부근 **0.69**로 약간 상승. (`seen18.py length` / `auroc --` abs)
- **permutation test(n=200)**: 이 약한 값(~0.6~0.8)이 **다수 task·시점에서 chance보다 유의**(abs 37셀, mean 45셀 p<.05). 즉 effect size는 작아도 **신호는 실재**(잡음 아님).
- 단 **불균형 task**(소표본, 예 CloseBlenderLid SR0.15)는 비유의 — 진짜 신호 없음.
- 신호는 **다차원·확산적**(PCA 1~2D로 안 모이고 ~20D) → COAST의 "rank-1 아닌 multi-dim"과 일치, 단일벡터 additive보다 **contrastive conceptor**가 맞는 연산자.
- 시간 방향과 **직교**: `cos(u_fail, u_time)=0.04`, time 방향 제거해도 AUROC 불변 → 길이/timestep 반응이 아님. (`seen18.py faildir` → `failure_direction/`)

---

## 3. 두 실패 레짐 — 언제부터 분리되나 (abs + permutation)
최초 유의(p<.05) frame 기준:

| 최초 유의 | 레짐 | task (AUROC) |
|---|---|---|
| **frame 0** | **초기조건형** (~8개) | OpenDrawer(.70)·OpenCabinet(.70)·SlideDishwasherRack(.76)·PnP-Cabinet(.74)·PnP-Sink(.66)·CloseToasterOvenDoor(.67)·TurnOnMicrowave(.70)·OpenStandMixerHead(.73) |
| f2~f4 | (이른 실행) | NavigateKitchen·CloseFridge·CoffeeSetupMug·PnP-DrawerToCounter |
| **f10~f14** | **실행표류형** | TurnOffStove·TurnOnSinkFaucet·PnP-CounterToStove·PnP-ToasterToCounter |
| never | 표본 부족 | CloseBlenderLid(SR.15)·TurnOnElectricKettle(fail n=18) |

- **frame 0(행동 전)부터 분리** = 결과가 **초기 scene·물체배치·instruction 난이도**로 상당 부분 결정(절반 task). 정책 드리프트가 아님.
- **실행표류형**(늦은 onset) = 초반 구분 안 되다 실행 중 발산 → **mid-rollout steering 개입 여지가 큼**. 초기조건형은 시작부터 결정돼 개입 효과 제한적일 수 있음.

---

## 4. 공유 실패 zone (SAFE 재현) — 후반에 약하게 실재
- **비순환 검증**(task-whitening, metric이 outcome 모름): task별 succ/fail centroid의 cross-task 퍼짐 비율 fail/succ:

  | progress p | 0.0 | 0.5 | 0.8 | 1.0 |
  |---|---|---|---|---|
  | fail/succ spread 비율 | 0.98 | 0.92 | 0.85 | **0.75** |

  실패 centroid 퍼짐 17.8→9.3(수렴), 성공은 후반 다시 task별로 벌어짐 10.5→12.5.
- ⇒ **실패는 진행할수록 task 무관 공유 "정체" zone으로 표류, 성공은 task별 목표로 갈림.** 효과는 후반에 강함.
- ⚠ **circularity 주의**: `task_succ_fail` whitening(실패를 한 그룹으로)은 cross-task fail 분산을 줄여 **실패가 tight하게 뭉쳐 보이게 유도**(tautological). 옛 7-label의 "fail 깔끔히 뭉침"은 상당 부분 이 효과. 정직한 신호는 task-whitening centroid-spread(위 표)로만.
- 도구: `seen18.py evolution --whiten-label task_succ_fail --full-dim-whiten` (시각), 정량은 task-whitening.

---

## 4b. Cross-task 실패 공유의 구조 (COAST Sec 4.4 재현 시도)

§4의 "약한 공유 zone"을 conceptor containment 관점에서 더 판 결과. 결론: **COAST Sec 4.4의
"cross-task 공유 실패 subspace"는 재현되지 않으며**, 표면적으로 보이는 fail>succ containment은
길이/rank 아티팩트다. 분석 산출은 `…/analysis/cross_task_failure_analysis/`(약칭 `ANALYSIS`).

| # | 결론 | 근거(수치) | 생산 스크립트 |
|---|---|---|---|
| 2.1 | **COAST Sec4.4 cross-task 공유 실패 subspace 미재현** | rollout-mean: fail containment 0.505 < succ 0.539; per-step all: fail 0.92 > succ 0.87이나 ↓ | `cross_task_failure_analysis.py`, `cross_task_failure_perstep.py` |
| 2.2 | **그 fail>succ는 길이/rank 아티팩트** (확정) | 단일 5-step window containment 전구간 ~0.65 평평; 벡터수 cap 맞추면 gap 소멸(cap200 fail0.652≈succ0.653) | `failure_stuck_state_test.py` |
| 2.3 | **실패 공유는 family 무구조**(=공통 stuck), 성공은 family 구조 | fail within≈cross(gap~0), succ within0.893>cross0.876 | `cross_task_containment_structure.py` |
| 2.4 | **실패는 cross-task 위치 공유, 성공은 task-specific** | global K8: fail high-entropy 4/8·task-pure1; succ task-pure 5/8 | `cross_task_mode_sharing.py` |
| 2.5 | **within-task 진짜 실패모드는 드묾**(성공대조) | 진짜 fail-특이 2/9 (PnPCabinet 0.65/succ0.11, StandMixer 0.64/0.15); CloseFridge는 scene | `within_task_failure_modes.py`, `within_task_succ_vs_fail.py` |
| 2.6 | **실패는 이산 모드가 아니라 연속체**(near-miss/stuck은 연속적 정도) | per-step region silhouette 0.09–0.13, 모든 region meanT≈0.5(시간 비국소), fail/succ 패턴 유사 | `failure_trajectory_modes.py` |

**한 줄 종합**: 성공은 task별 목표 영역으로 구조적 수렴 / 실패는 task도 안 가리고 이산 모드도 없이
"목표 미도달" 연속 영역에 흩어짐. COAST식 "공유 실패 방향"은 길이 아티팩트.

**steering 설계 함의**: ❌ 단일 cross-task 실패 방향/전이 근거 약함(2.1–2.4) → global 한 방향
steering·cross-task 전이는 기대 낮음. ✅ 성공이 task-구조적(2.4) → **per-task** 또는 **성공-유인
(C⁺ 쪽)** conceptor가 데이터에 부합. `fit_conceptor_steering.py --per-task`부터 시작.

⚠ **방법 caveat**: conceptor containment는 **표본수/rank에 민감** → 클래스 간 벡터 수를 맞춰 비교해야
한다(맞추지 않으면 fail>succ가 가짜로 뜸). circularity 금지: whitening은 outcome-agnostic한 `task`로만.

---

## 5. task 정체성 vs outcome
- **pairwise task AUROC**(task A vs B 분류): rollout 1점 ~**0.99**, 전체 timestep group-aware(td16) mean **0.998**/min **0.975**, progress 전 구간 ~1.0(worst pair mid에서 0.94).
- ⇒ **latent을 task 정체성이 지배** (frame 0부터 ~1.0 = scene/instruction 인코딩, 행동 전부터). 단 AUROC는 고차원에서 saturate.
- **task 구분 난이도**(taskdist, truncated z_mean t_d=17): 가장 뚜렷 = **TurnOnMicrowave·NavigateKitchen·SlideDishwasherRack**, 가장 헷갈림 = **PnP 계열·Close 계열**. 단 pairwise AUROC상 *진짜 못 구분되는* 쌍은 없음(전부 ≥0.94) → "유사도로 솎기"엔 Mahalanobis centroid 거리 dendrogram이 변별력.
- ⇒ succ/fail이 약한 이유: **task/scene 축이 지배 → outcome 신호는 그에 거의 직교하는 작은 부분공간**. task 조건화(whitening) 후에야 드러남.
- 도구: `seen18.py pairwise`, `seen18.py taskdist`.

---

## 6. 시간 집계 방식 비교 (failure-zone이 가장 잘 보이는 단위)
스킴별 fail/succ centroid-spread 비율(낮을수록 강함, full-dim, task-whitening):

| 스킴 | 성격 | failure-zone 강도 |
|---|---|---|
| truncated-progress/frame (t_d=16) | 앞 16 step 누적 | 거의 없음(ratio ~0.97) — 후반 수렴 구간을 잘라서 |
| padded (45 freeze) | 누적, 성공 freeze | **0.86, 생존편향 없는 가장 정직한 view** |
| window W=1~4 (비누적) | 시점 구간만 | 가장 강함(0.79~0.89) — 단 후반 window는 성공 소진(survivorship) |

- **누적/episode-mean의 높은 분리는 길이 누설** 포함. 길이 통제하려고 truncate하면 정작 후반 신호가 사라짐 → **padded가 트레이드오프 우위**.
- 도구: `seen18.py temporal`, `seen18.py snapshots`, `seen18.py windows`.

---

## 7. COAST geometry (baseline only)
- **Fig 3A**(conceptor 스펙트럼): C⁺/C⁻ eigenvalue 급감(low-rank), C_steer = C⁺∧¬C⁻ 유효 차원 ~소수(%) — succ/fail 판별 부분공간은 저차원·multi-dim. (`seen18.py coast --fig spectrum`)
- **Fig 4A**(top-2 eigvec of C_steer): 대부분 task에서 succ/fail 2σ 타원이 **거의 동심원**(centroid gap 작음, 실패 타원이 더 퍼짐) → 약한 분리.
- **Fig 4B**(v1 over normalized time): succ/fail 밴드 전 구간 크게 겹침, 후반 약간 divergence.
- ⇒ COAST의 π0.5 LIBERO 대비 **GR00T-N1.6 RoboCasa는 succ/fail 대비 geometry가 약함**(작은 centroid gap, 겹치는 밴드). steering이 증폭할 contrastive signal 자체가 약할 가능성 — 실제 효과는 intervention 실험 전엔 단정 불가.
- 단 baseline만(steered 없음), 축은 label 유도(in-sample).

---

## 8. 방법론적 교훈
- **held-out vs in-sample**: 분리 판정은 **held-out CV AUROC**로. mean-diff/LDA 축의 in-sample 분리는 소표본에서 과적합(예 CloseBlenderLid f4: in-sample AUROC 0.91 vs **held-out 0.35 ≈ shuffle 0.34** = 신호 0). t-SNE 2D 겹침 ≠ 분리 불가(선형 분리 방향을 2D가 못 보존).
- **shuffle/permutation baseline 필수**: 소표본은 chance가 0.5가 아님. AUROC를 자기 shuffle과 비교해야 함.
- **whitening 라벨 선택**이 결론을 바꿈: outcome을 라벨에 넣으면(task_succ_fail/task_failure) 분리가 부분 보장(circular). 정직한 건 **task-only whitening**.

---

## 9. latent steering 함의 (종합)
- ✅ succ/fail latent 분리는 **실재**(길이통제·permutation 유의) — steering 전제의 근거.
- ⚠ 단 **약하고(~0.6~0.76) 다차원** → 단일벡터 additive 아닌 **COAST식 contrastive conceptor**가 맞는 연산자.
- 신호는 **후반·누적에서 강해짐** → mid-rollout 개입 + 시간 누적 detector 유리, 조기 검출은 어려움.
- **두 레짐**: 실행표류형 task = steering 최적 타깃 / 초기조건형 = 시작부터 결정, 효과 제한적일 수 있음.
- **다음 본론**: succ/fail로 `C_steer` fit → serve 활성화 주입(`h'=h·Mᵀ`) → **재롤아웃 ΔSR 측정**(인과 검증), 레짐별 효과 분리.

---

## 10. 재현 (단일 CLI)
```bash
PY=~/miniconda3/envs/hyundai_aigs/bin/python
cd scripts/safe/groot_n16/robocasa/vis
$PY seen18.py length                       # 길이 confound + early-detection
$PY seen18.py auroc                         # abs/mean/window × task × time, permutation 유의
$PY seen18.py faildir                       # 실패 방향 vs 시간 방향
$PY seen18.py evolution --whiten-label task_succ_fail --full-dim-whiten   # 공유 실패 zone 진화
$PY seen18.py temporal                      # 시간집계 스킴 비교
$PY seen18.py pairwise --mode all_timesteps # task pairwise 분리
$PY seen18.py taskdist                      # task 거리/난이도/grouping
$PY seen18.py coast                         # Fig 3A/4A/4B
$PY seen18.py html3d                        # 인터랙티브 3D (vendor 없으면 CDN)
$PY seen18.py <cmd> --help                  # 각 옵션
```

- **feature cache**: `…/analysis/feature_cache/pooled_all_hmean_dmean.npz` (per-env-step DiT
  pre-velocity action-token latent `[K=4,H=16,D=1024]` → K·H mean-pool → step당 1024-d).
  로더: `vis/core/io.py` → `reconstruct_rollouts(cache)`. conceptor 모듈은 `src/conceptor/`.
- ⚠ **§4b cross-task 스크립트 재실행 주의**: `analyze/cross_task_*.py` 등 다수가 삭제된
  `temporal_agg.py`를 import해 **깨져 있다**. 재실행 전 `from core.io import reconstruct_rollouts`로
  교체 필요(`within_task_succ_vs_fail.py`는 이미 수정됨). 기초 시각화는 위 `seen18.py` CLI로 재현 가능.

관련 메모리: `seen18-rollout-length-confound`, `seen18-genuine-failure-direction`,
`seen18-shared-failure-zone`, `seen18-failure-onset-regimes`, `project-direction-latent-steering`.
