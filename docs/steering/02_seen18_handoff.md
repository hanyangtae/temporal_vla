# seen18 → COAST steering & SAFE detector 핸드오프

작성: 2026-05-28 · 대상: `outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep`

> **용도**: 두 후속 세션 — (A) **COAST steering**(steer 후 SR↑), (B) **SAFE detector**(조기 failure 검출) —
> 이 각각 실험을 돌릴 때의 *근거·주의·재사용 인프라·실행 스펙*을 모은 단일 진입 문서.
> 기초 잠재공간 분석은 `01_seen18_latent_analysis.md`(섹션 0–10) 참조. 본 문서는 **cross-task 실패 심화**(아래 §2)와
> **두 세션 브리핑**(§3,§4)에 집중. 모든 분석 산출은 `…/analysis/cross_task_failure_analysis/`(아래 ANALYSIS) 한곳에 모음.

약칭: `RUN = outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep`,
`ANALYSIS = RUN/analysis/cross_task_failure_analysis`, `VIS = RUN/analysis/visualizations`.

---

## 1. 인프라 (재사용)
- **데이터/캐시**: `RUN/analysis/feature_cache/pooled_all_hmean_dmean.npz`. per-env-step DiT pre-velocity action-token latent `[K=4,H=16,D=1024]`를 K·H mean-pool → step당 1024-d. 1800 rollout(18 task×100), SR 0.537, 실패 전부 45-step / 성공 mean 17.7±8.4.
- **로더**: `scripts/safe/groot_n16/robocasa/vis/core/io.py` → `reconstruct_rollouts(cache)` → `(rollouts[{z[T,1024],succ,task}], names)`. `load_feature_cache`. 캐시에 `ep_task_id/ep_episode_idx/ep_success`(rollout 순서 정렬).
- **rollout→mp4**: `RUN/raw_rollouts/<TaskName>/task{tid}--ep{ep}--succ{0/1}.mp4`. manifest `RUN/analysis/split/manifest.tsv`(split,task_id,task,category,episode_idx,success,source_path; (task_id,ep) 정렬 = reconstruct 순서).
- **conceptor 모듈**: `src/conceptor/` — `compute_correlation/compute_conceptor/not_/and_/or_/contrastive_conceptor`, `conceptor_overlap/quota/eigenvalue_spectrum/failure_containment`, `build_steering_matrix/apply_steering`. (relative ridge 직접 쓰려면 `C=R@inv(R+λ·tr(R)/d·I)`.)
- **steering fit**: `scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py` — succ/fail 분포에서 `C_steer=C⁺∧¬C⁻` fit, aperture sweep + overlap band(0.85–0.95) 선택, global/per-task 지원.
- **SAFE detector 자산(구 seen4)**: `train/train_lstm_*.sh`, `analyze/finalize_lstm_detector.py`, `analyze/summarize_lstm_*`; SAFE repo `../SAFE`. conformal 곡선 플로팅은 `vis/plot_safe_conformal_curves.py`(또는 seen18 CLI).
- **distance/metrics**: `core/distance.py`(`pooled_within_cov`,`whiten`), `core/metrics.py`(`silhouette_safe`,`cv_auroc`,`per_point_ab`).
- **env**: `~/miniconda3/envs/hyundai_aigs/bin/python` (분석). RoboCasa 재롤아웃은 Docker(py3.11). 
- ⚠ **리팩토링 주의**: `vis/`가 `core/` 패키지 + `vis/seen18.py` CLI로 마이그레이션됨. **`temporal_agg.py` 삭제됨** → `analyze/*.py` 다수가 `from temporal_agg import ...`로 import 깨짐. 재실행 시 `from core.io import reconstruct_rollouts`로 교체 필요(`within_task_succ_vs_fail.py`는 이미 수정됨). `coast_conceptor_spectrum.py`/`coast_fig4a/4b.py`는 삭제됨(출력만 `VIS/coast_fig4a,coast_fig4b`에 잔존).

---

## 2. 확립된 결론 (근거 수치 + 생산 파일)
기초(길이 confound·약한 진짜 신호·onset 레짐·temporal 집계)는 `01_seen18_latent_analysis.md` 참조. 아래는 **cross-task 실패 심화**(이 세션 작업):

| # | 결론 | 근거(수치) | 생산 스크립트 | 출력 |
|---|---|---|---|---|
| 2.1 | **COAST Sec4.4 cross-task 공유 실패 subspace 미재현** | rollout-mean: fail containment **0.505 < succ 0.539**; per-step all: fail 0.92 > succ 0.87이나 ↓ | `analyze/cross_task_failure_analysis.py`, `cross_task_failure_perstep.py` | `ANALYSIS/{results.json, containment_*_{z_episode,z_trunc16}.{png,tsv}, perstep/}` |
| 2.2 | **그 fail>succ는 길이/rank 아티팩트** (확정) | 단일 5-step window containment 전구간 ~0.65 평평; 벡터수 cap 맞추면 gap 소멸(cap200 fail0.652≈succ0.653) | `failure_stuck_state_test.py` + 인라인 equalize | `ANALYSIS/stuck_test/` |
| 2.3 | **실패 공유는 family 무구조**(=공통 stuck), 성공은 family 구조 | fail within≈cross(gap~0), succ within0.893>cross0.876 | `cross_task_containment_structure.py` | `ANALYSIS/structure/` |
| 2.4 | **A6: 실패는 cross-task 위치 공유, 성공은 task-specific** | global K8: fail high-entropy 4/8·task-pure1; succ task-pure 5/8 | `cross_task_mode_sharing.py` | `ANALYSIS/mode_sharing/` |
| 2.5 | **within-task 진짜 실패모드는 드묾**(성공대조) | 진짜 fail-특이 2/9 (PnPCabinet 0.65/succ0.11, StandMixer 0.64/0.15); CloseFridge는 scene(succ0.70>fail0.58) | `within_task_failure_modes.py`, `within_task_succ_vs_fail.py` | `ANALYSIS/within_task/` |
| 2.6 | **실패는 이산 모드가 아니라 연속체**(near-miss/stuck은 연속적 정도) | per-step region silhouette 0.09–0.13, 모든 region meanT≈0.5(시간 비국소), goal region 불명확, fail/succ 패턴 유사 | `failure_trajectory_modes.py` | `ANALYSIS/trajectory_modes/` |
| 2.7 | **shared 실패 zone ≈ "목표 미도달 영역"**(비순환): 후반 약하게 수렴 | task-whiten centroid-spread 비율 fail/succ p=0 0.98 → p=1.0 0.75 | (seen18 CLI / 인라인) | `VIS/evolution/failure_zone_centroid_spread.png` |

**한 줄 종합**: 성공은 task별 목표 영역으로 구조적 수렴 / 실패는 task도 안 가리고 이산 모드도 없이 "목표 미도달" 연속 영역에 흩어짐. succ/fail 분리는 길이통제 시 약함(~0.6, 후반 ↑). COAST식 "공유 실패 방향"은 길이 아티팩트.

---

## 3. COAST 세션 브리핑 (steer 후 SR↑)
**목표**: `C_steer=C⁺∧¬C⁻`를 action-expert residual에 적용(`h'=h·Mᵀ`, β∈{0.1,0.3})하고 **재롤아웃해 ΔSR 측정**. (지금까지 ΔSR은 전혀 미측정.)

**우리 데이터가 시사하는 것 (실험 설계에 반영)**:
- ❌ **단일 cross-task 실패 방향/전이 근거 약함**(§2.1–2.4) → global 한 방향 steering·cross-task 전이는 기대 낮음.
- ✅ **성공이 task-구조적**(§2.4) → **per-task** 또는 **성공-유인(C⁺ 쪽)** steering이 데이터 부합. 우선 **per-task `C_steer` fit**(이미 `fit_conceptor_steering.py --per-task` 지원)으로 시작.
- ⚠ succ/fail 대비 geometry 약함(Fig4A/4B 타원 겹침, `VIS/coast_fig4a,4b`) → 작은 β부터, 효과 미미 가능성 염두.
- ❓ **미검증**: Fig3A 저랭크 subspace(quota·spectrum). steer 전에 `eigenvalue_spectrum(C_steer)`로 effective rank 확인 권장(스크립트 삭제됨 → `src/conceptor.eigenvalue_spectrum`로 재작성).

**실행 골격**: (1) per-task succ/fail latent로 `fit_conceptor_steering.py` → `C_steer`+α 선택, (2) serve 측(`steer/serve/feature_server.py` 또는 serve 스크립트)에 `apply_steering` 훅, (3) RoboCasa Docker 재롤아웃 N ep, (4) steered vs unsteered SR 비교(task별·전체, z-test).
**검증**: ΔSR>0 유의? overlap sim(C⁺,C⁻)와 ΔSR 상관(COAST Fig3B). 단일벡터(CAA) baseline 대비.

---

## 4. SAFE / detector 세션 브리핑 (조기 failure 검출)
**목표**: per-step latent로 **temporal detector**(SAFE-MLP 누적 Σσ(g) / SAFE-LSTM) 학습 + **functional conformal** threshold → ROC-AUC(max-so-far), TPR/FPR/bal-acc, **detection-time**, **seen/unseen** task 일반화.

**반드시 지킬 것 (우리 데이터 핵심 교훈)**:
- ❗ **길이 confound**: 실패 전부 45-step/성공 조기종료라 **시간-pooled/episode-level 분리(~0.9)는 아티팩트**(길이 AUROC 0.998). SAFE처럼 **T를 task별 min 길이로 cap**(또는 matched-frame)해야 함. 길이통제 시 진짜 per-frame 신호 **~0.6, 후반 ~0.7**(`01_seen18_latent_analysis.md` §2, `VIS/length_confound/`, `VIS/auroc_tables/`).
- **onset 레짐**(§seen18_latent_analysis §3): ~8 task는 frame0부터 유의, 4 task는 f10+에야 → detection-time이 task별로 크게 다름. unseen 일반화 평가 시 고려.
- **유의성**: permutation null로 판정(소표본 chance≠0.5). `VIS/auroc_tables/`(별표·red 테두리)와 동일 기준 사용.

**재사용**: SAFE repo `../SAFE`; 구 seen4 detector 파이프라인 `train/train_lstm_*.sh` + `analyze/finalize_lstm_detector.py` + `vis/plot_safe_conformal_curves.py`(seen18로 split·경로만 교체). per-step feature는 캐시. **seen18용 split 필요**: 현재 `RUN/analysis/split/manifest.tsv`는 전체 1800 manifest이고, seen/unseen train/cal split은 아직 없음(구 split prepare는 seen4 전용) → seen18 split 생성부터.
**검증**: max-so-far ROC-AUC(T-cap), seen vs unseen, bal-acc vs detection-time 곡선(α sweep). 학습 없는 baseline(Mahalanobis/kNN, `core.metrics`)도 비교.

---

## 5. 파일 인덱스 & 재실행 주의
- **분석 스크립트**: `scripts/safe/groot_n16/robocasa/analyze/` (cross_task_*, failure_*, within_task_*). **대부분 `temporal_agg` import 깨짐** → 재실행 전 `from core.io import reconstruct_rollouts as reconstruct` + `DEFAULT_CACHE` 하드코딩으로 교체(예: `within_task_succ_vs_fail.py` 참고). 출력 `OUT`은 모두 `ANALYSIS/` 안으로 수정됨.
- **CLI(권장 경로)**: `vis/seen18.py` (cache/embed/separation/pertask_tsne/pairwise/taskdist/faildir/windows/temporal/evolution). 기초 시각화는 이걸로 재현.
- **출력**: `ANALYSIS/`(cross-task 심화 전부) + `VIS/`(기초 시각화: auroc_tables, succ_fail_3d(3D html), evolution, length_confound, step_windows, per_task_*, coast_fig4a/4b 등).

## 6. 방법론 규칙 (두 세션 공통)
1. **길이통제 필수** — 실패=timeout이라 어떤 시간-aggregate/progress도 길이 누설. matched-frame 또는 T-cap.
2. **circularity 금지** — 측정 대상(succ/fail) 라벨로 metric(whitening 등)을 만들지 말 것. whitening은 `task`(outcome-agnostic)로.
3. **permutation/held-out** — 소표본 chance≠0.5. AUROC는 CV+shuffle baseline, 시각 분리(t-SNE/타원)는 in-sample이라 과신 금지(분리 판정은 held-out AUROC).
4. **conceptor containment는 표본수/rank에 민감** — 벡터수 맞춰 비교.
