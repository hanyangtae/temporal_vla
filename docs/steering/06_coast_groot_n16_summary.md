# COAST Conceptor Steering on GR00T N1.6 RoboCasa — 종합 요약

작성: 2026-05-29 (다른 세션 cold-pickup 용).
관련 메모리: `project-direction-latent-steering`, `seen18-*`, `truncation-length-standard`,
`groot-robocasa-serve-path`, `cpu-budget-cap`, `no-latex-dollar`.

## 0. 한 줄

GR00T N1.6 robocasa365 체크포인트에서 COAST 식 contrastive conceptor
(`C_steer = C_success ∧ ¬C_failure`) 로 latent steering 을 재현하려 했고,
**DiT 32 layer 전체에서 q̄(ℓ) 가 monotone 감소·peak 없음, 절대값 약함(0.09~0.12)** 으로
SR 개선 가능성이 낮다는 결과가 layer-select 단계에서 드러났다. SR eval 진행 중.

## 1. 배경 / 동기

- 메인 연구방향: VLA latent steering (success/failure 분리 → 추론 시 steer → SR↑).
  TTA(VITA progress predictor) 방향은 무기한 연기됨.
- COAST 원본 (`docs/references/COAST.pdf`) 는 GR00T **N1.5** (16-layer DiT, d=1536) 에서
  Stage1 layer sweep → ℓ=10 선택. 우리는 **N1.6** (32-layer DiT, inner_dim=1536, output_dim=1024)
  와 robocasa365 multitask 체크포인트로 시도.
- 사전 단일-layer(ℓ=31 의 post-projection, 1024-dim) steering 실험:
  baseline SR 0.65 → steered 0.45~0.55 (SR 하락). 그래서 적절한 layer 를 찾아야 한다는
  결론으로 multi-layer 수집·sweep 으로 진행.

## 2. 데이터: 1000-rollout multilayer 수집

- 위치: `outputs/eval/robocasa/groot_n16/target_atomic_moderate10_multilayer_100ep/raw_rollouts/`
- 구성: **10 moderate-SR task × 100 ep = 1000 pkl**. SR 31~67% 구간 (성공·실패 모두 ≥31,
  conceptor fit 충분).
- 캡처: 매 env step 마다 **DiT transformer_blocks 32 개 전부**의 residual stream
  (1536-dim) 을 server-side 에서 action token H · denoising step K 평균 pool →
  per-step `[L=32, D=1536]` fp16.
- task 별 succ/fail 분포 (왼쪽=task, succ/fail):

| task | succ | fail |
|---|---|---|
| CloseToasterOvenDoor | 48 | 52 |
| NavigateKitchen | 45 | 55 |
| OpenCabinet | 52 | 48 |
| OpenDrawer | 40 | 60 |
| PickPlaceCounterToCabinet | 67 | 33 |
| PickPlaceCounterToStove | 63 | 37 |
| PickPlaceDrawerToCounter | 32 | 68 |
| SlideDishwasherRack | 55 | 45 |
| TurnOnMicrowave | 47 | 53 |
| TurnOnSinkFaucet | 31 | 69 |

- 선정 이유: seen18 (18 task) 중 SR 양극단(거의 100% 성공/실패) 제외 → moderate 10 task.
  user 명시적 결정. 100 ep 으로 늘려 길이 통제 후 task 당 fit 통계 확보.
- 18 task 로 확장하려면 추가 8 task × 100 ep 수집 (~3-4h, 6 GPU). 미수행.
- 수집 launcher: `scripts/safe/groot_n16/robocasa/steer/collect_multilayer_parallel.sh`.
- 사용된 endpoint: `get_action_with_multilayer_features` (feature_server.py 신규).
- 길이 confound 경고: 실패=항상 timeout(=45 step), 성공=조기종료 (mean ≈ 17~19). 길이 단독으로
  AUROC≈0.998 → 시간-pooled feature 분리는 아티팩트. fit/select 에서 W 절단 필수
  ([[truncation-length-standard]]).

## 3. Layer-select 비교: COAST 방식 vs balanced 방식

스크립트: `scripts/safe/groot_n16/robocasa/steer/layer_select_compare.py`
결과 JSON: `.../conceptor_steering/layer_selection/stage1_quota_compare.json`
α₀ = 10 (COAST A.10.2 Stage1 표준).

### 3.1 방식 정의

- **coast**: 절단·균형 없음. task 마다 success/failure step 전부 사용. 길이 confound 그대로.
- **balanced**: 길이/표본수 통제. task t 마다:
  1. W_t = p10(success 길이) 반올림
  2. 길이 < W_t 성공 episode 제거(drop)
  3. 남은 성공 + 실패 episode 를 첫 W_t step 으로 절단
  4. 실패 episode 수 = 남은 성공 episode 수 (subsample) → N_succ_step = N_fail_step.

### 3.2 결과 — 두 방식 모두 monotone 감소, 중간층 peak 없음

|  | coast ℓ* | balanced ℓ* |
|---|---|---|
| ℓ* | **4** | **0** |
| q̄ | 0.1206 | 0.0916 |
| ov̄ | 0.838 | 0.702 |
| 전체 q̄ 패턴 | L0=0.116 → L4=0.121 → L31=0.018 | L0=0.092 → L31=0.025 |

두 방식 모두:
- 초기층(L0~L5) 이 quota 최고, 깊을수록 단조 감소.
- COAST N1.5 (16-layer DiT, ℓ=10/16 가 peak) 의 **중간층 peak 패턴이 부재**.
- 절대값 매우 낮음: 0.09~0.12 (COAST 보고 peak 통상 0.3~0.5 대비 ~1/3).

### 3.3 ℓ* 에서 task 별 quota

**coast ℓ=4** (높음 → 낮음)

| task | q |
|---|---|
| OpenCabinet | 0.1589 |
| PickPlaceCounterToCabinet | 0.1513 |
| CloseToasterOvenDoor | 0.1346 |
| PickPlaceCounterToStove | 0.1344 |
| TurnOnMicrowave | 0.1261 |
| NavigateKitchen | 0.1241 |
| PickPlaceDrawerToCounter | 0.1165 |
| SlideDishwasherRack | 0.1056 |
| TurnOnSinkFaucet | 0.0794 |
| OpenDrawer | 0.0752 |

**balanced ℓ=0** (높음 → 낮음)

| task | q |
|---|---|
| OpenCabinet | 0.1202 |
| CloseToasterOvenDoor | 0.1006 |
| SlideDishwasherRack | 0.0991 |
| PickPlaceDrawerToCounter | 0.0972 |
| PickPlaceCounterToStove | 0.0907 |
| OpenDrawer | 0.0881 |
| TurnOnMicrowave | 0.0833 |
| NavigateKitchen | 0.0820 |
| PickPlaceCounterToCabinet | 0.0798 |
| TurnOnSinkFaucet | 0.0746 |

두 방식의 top task 분포가 다른 점이 시사적 — coast 의 PickPlaceCounterToCabinet (top2)
가 balanced 에선 하위 (success/fail 67/33 으로 SR 가장 높음 → 길이 confound 가 quota 를
부풀린 task). balanced 가 confound 제거 효과.

## 4. Fit (Stage2 α-band + Stage3 β 미정)

- 스크립트: `scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py`
  - `--force-layer N` 옵션 추가 (Stage1 sweep skip, 지정 layer 에서 직접 fit).
- 두 layer 각각에서 per-task fit:
  - `.../conceptor_steering/layer0/truncated_w19/{global, task_0_..., ..., task_9_...}/conceptors.npz`
  - `.../conceptor_steering/layer4/truncated_w19/{global, task_0_..., ..., task_9_...}/conceptors.npz`
- W = 19 = success len mean (COAST 표준 truncation, [[truncation-length-standard]]).
- Stage2 α-band 정상 작동: task 별 α=1~5 범위 선택 (overlap band 안에 있는 alpha 유지).

## 5. SR eval — 완료 (Phase 1, 16-action-token pool)

- 오케스트레이션: `scripts/safe/groot_n16/robocasa/steer/eval_steer_compare.sh`
- 매트릭스: **10 task × 5 condition × N_EP=20 = 1000 rollout**.
- 결과 TSV: `outputs/eval/robocasa/groot_n16/target_atomic_moderate10_multilayer_100ep/steer_eval/steer_compare_full/results.tsv`
- 실행 형태: 3 GPU × 2 server (GPUS="0 1 2 0 1 2", N_ENVS=2 로 안착).

### 5.1 task × condition SR (괄호 = baseline 대비 ΔSR, ★ = COAST 논문 atomic-seen 7 중복)

| task | baseline | L0 β=0.1 | L0 β=0.3 | L4 β=0.1 | L4 β=0.3 |
|---|---|---|---|---|---|
| CloseToasterOvenDoor | 0.45 | 0.35 (−.10) | 0.55 (+.10) | 0.20 (−.25) | 0.45 (.00) |
| NavigateKitchen | 0.15 | 0.25 (+.10) | 0.10 (−.05) | 0.05 (−.10) | 0.30 (+.15) |
| OpenCabinet | 0.75 | 0.70 (−.05) | 0.60 (−.15) | 0.50 (−.25) | 0.65 (−.10) |
| ★ OpenDrawer | 0.45 | 0.40 (−.05) | 0.30 (−.15) | 0.30 (−.15) | 0.30 (−.15) |
| ★ PickPlaceCounterToCabinet | 0.70 | 0.75 (+.05) | 0.65 (−.05) | 0.50 (−.20) | 0.55 (−.15) |
| ★ PickPlaceCounterToStove | 0.65 | 0.65 (.00) | 0.75 (+.10) | **0.90 (+.25)** | 0.65 (.00) |
| PickPlaceDrawerToCounter | 0.30 | 0.15 (−.15) | 0.35 (+.05) | 0.25 (−.05) | 0.30 (.00) |
| SlideDishwasherRack | 0.55 | 0.45 (−.10) | **0.80 (+.25)** | 0.65 (+.10) | 0.52 (−.03) |
| TurnOnMicrowave | 0.55 | 0.65 (+.10) | 0.30 (−.25) | 0.55 (.00) | 0.45 (−.10) |
| TurnOnSinkFaucet | 0.35 | 0.40 (+.05) | 0.40 (+.05) | 0.35 (.00) | 0.40 (+.05) |

### 5.2 평균 ΔSR

| condition | 전체 10 task | COAST ★ 3 task (OpenDrawer / PnPCab / PnPStove) |
|---|---|---|
| L0 β=0.1 | **−0.015** | **+0.000** |
| L0 β=0.3 | **−0.010** | −0.033 |
| L4 β=0.1 | **−0.065** | −0.033 |
| L4 β=0.3 | −0.033 | **−0.100** |

### 5.3 핵심 패턴

- **평균 양 상승 없음**: 모든 condition 평균 ΔSR ≤ 0. 가장 좋은 것도 L0 β=0.1 의 −0.015 (사실상 0).
- **L4 > L0 손해**: coast ℓ*=4 가 balanced ℓ*=0 보다 SR 더 떨어뜨림 (특히 β=0.1: -0.065 vs -0.015).
- **단일 task 큰 양수**: SlideDishwasherRack L0_b03 (+0.25), PickPlaceCounterToStove L4_b01 (+0.25) — N=20 에서 binomial std ≈ 0.11 → ~2.3σ, 가능성은 있지만 task별 cherry-pick.
- **COAST ★ 3 task subset**: 모두 음 또는 0 (논문 직접 비교군에서 효과 없음).

→ **N1.6 의 action-16-token pool 로는 COAST 재현 불가**. 16 token 한정이 원인인지 (Phase 2: T-full
재수집 후 49-token mean 비교로 검증) 또는 N1.6 architecture 자체의 한계인지 (fallback: N1.5 ckpt)
가 다음 단계.

## 6. NOTALL (ICLR 2026) 연결고리

`docs/references/NOTALL.txt` (NOT ALL FEATURES ARE CREATED EQUAL) 의 GR00T N1.5 분석과
우리 N1.6 결과의 대응:

| NOTALL (N1.5) | 우리 (N1.6) |
|---|---|
| 32 layer = 12 Eagle LM + 4 VL-SA + 16 DiT | 32 layer DiT-only sweep (Eagle/VL-SA 미탐색) |
| DiT 가 가장 ablation-sensitive (40~80% SR drop) | DiT 32 layer 전부에서 quota 약함 |
| **DiT L0 가장 파괴적 (21% destruction)** | **balanced ℓ*=0** (DiT L0 ≈ NOTALL kill-switch) |
| Eagle moderate, VL-SA 가장 resilient | (해당 pathway 분석 안 함) |
| VL-SA per-token EV 83~89%, mean-pool 시 99% | (해당 pathway 분석 안 함) |
| 70% kill-switch 는 object concept (early-stage binding) | balanced top task = OpenCabinet (object) ⊕ |
| expert pathway: motor 인코딩, VLM: goal | DiT = expert pathway → motor 신호 위주 가능성 |

NOTALL 시사점:
- 우리가 sweep 한 DiT 만으로는 succ/fail 분리가 약할 수 있음 — **VL-SA 또는 Eagle pathway
  에서의 contrastive structure 미탐색**.
- DiT L0 의 강한 영향력은 motor-binding 의 early-layer 특성과 일치 — 우리 balanced ℓ*=0
  과 부합. 다만 NOTALL 은 ablation 결과, 우리는 success/failure contrastive subspace.
- per-token vs mean-pooled: 우리 collection 은 server-side 에서 K·H mean-pool 했음. NOTALL
  은 VL-SA 에서 mean-pool 이 EV/fidelity 를 boost. **우리도 per-token vs pooled 비교 안 함**.
- pathway-specific steering (expert vs VLM 분리 hook 등록) 미시도.

## 7. 핵심 파일·산출물

### 코드
- `src/conceptor/` — conceptor 수학 모듈 (`compute_conceptor`, `and/not/or_conceptor`,
  `contrastive_conceptor`, `conceptor_quota`, `conceptor_overlap`, `build_steering_matrix`,
  `apply_steering`).
- `tests/test_conceptor.py`, `tests/test_steering_hook.py` — 18 unit test (전부 통과).
- `scripts/serve/steering_hooks.py` — `ConceptorSteering(model, M, layer=None|int)`,
  `load_steering_matrix(npz, beta, alpha, key)`. layer=None → action_head.model 출력
  (pre-velocity), layer=i → transformer_blocks[i].
- `scripts/safe/groot_n16/robocasa/serve/feature_server.py` — ZMQ feature server.
  endpoints: `get_action_with_features` (단일 layer SAFE), `get_action_with_multilayer_features`
  (32 layer 전부). args: `--steering-npz`, `--steering-beta`, `--steering-alpha`,
  `--steering-key`, `--steering-layer`, `--capture-layers`. dev merge 후 콜리그 리팩터
  (`SafeFeatureExtractor` 클래스)에 맞춰 재배선됨.
- `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py` — ZMQ client, multilayer
  endpoint 지원. multilayer pooled feature 는 SAFE per-token 불변식 우회 (write_safe_triplet
  에 multilayer kind 가드).
- `scripts/safe/groot_n16/robocasa/steer/`
  - `collect_multilayer_parallel.sh` — N GPU × (server+collect client) 병렬 수집.
  - `layer_select_compare.py` — coast vs balanced 양 방식 quota sweep + 비교 JSON.
  - `fit_conceptor_steering.py` — Stage1 (sweep) + Stage2 (α-band) + Stage3 fit. 신규
    `--force-layer` 로 sweep skip.
  - `eval_steer_compare.sh` — 10 task × 5 condition × N_EP steer eval, multi-GPU 병렬.

### 산출물 (지속 보존)
- 수집: `outputs/eval/robocasa/groot_n16/target_atomic_moderate10_multilayer_100ep/raw_rollouts/<task>/*.pkl`
  (1000 pkl, ~3.8 GB)
- layer-select JSON: `.../conceptor_steering/layer_selection/stage1_quota_compare.json`
- fit NPZ: `.../conceptor_steering/layer{0,4}/truncated_w19/<task>/conceptors.npz` × 22 개
  (각 layer 의 global 1 + task 10 = 11)
- SR eval TSV: `.../steer_eval/steer_compare_full/results.tsv` (진행 중)

### 메모리 (관련, 다른 세션에서도 자동 로드됨)
- `project-direction-latent-steering` — 메인 방향, TTA 연기
- `seen18-rollout-length-confound` — 길이 confound 경고
- `seen18-genuine-failure-direction` — 시간방향 직교 신호 존재
- `seen18-failure-onset-regimes` — 초기조건형 vs 실행표류형
- `truncation-length-standard` — W = success len [mean, mean+1σ]
- `groot-robocasa-serve-path` — ZMQ feature_server.py 경로
- `cpu-budget-cap` — CPU ≤40% (≤~25 코어), OMP/OPENBLAS ≤16
- `no-latex-dollar` — 답변에 $ 금지

## 8. 시사점 / Open

- **현 결과 요약**: GR00T N1.6 DiT subspace 에서 COAST conceptor 의 contrastive structure
  는 약하고 layer 별 peak 가 없다. 이전 단일-layer steering SR 하락(0.65→0.45~0.55) 과
  일관. SR eval 결과가 이 예측을 정량적으로 확인할 것.
- **NOTALL 기반 새 방향 후보**:
  1. **Pathway-specific steering** — Eagle LM / VL-SA 에서 contrastive conceptor 시도
     (현재 DiT-only). VL-SA 가 pooled 에서 EV↑ 라는 NOTALL 결과 활용.
  2. **SAE-based concept identification** — frequency-weighted contrastive selection 으로
     manipulation concept 추출, concept ablation/steering. COAST 와 직교한 접근.
  3. **per-token vs pooled 비교** — 우리 collection 은 server-side pooled. per-token raw
     vector 도 capture 해 SAE/conceptor 둘 다에서 fidelity 측정.
  4. **Visual pathway override 측정** — null prompt + 활성화 injection 으로 visual
     dominance 확인 (NOTALL 의 핵심 metric, override rate).
  5. **Cross-task injection** — failure 시 success rollout 의 activation 으로 steer
     하는 직접적 실험 (conceptor 매개 X). NOTALL 의 spatially-bound motor program 가설을
     N1.6 robocasa365 에서 검증.
- **N1.6 DiT 가 N1.5 처럼 작동 않는 가능성**:
  - N1.6 는 32-layer (N1.5 는 16-layer) — 깊이 차이가 representation 분포에 영향.
  - robocasa365 multitask SFT 가 contrastive structure 를 평탄화했을 가능성.
  - 18 task → 10 task scope 축소가 average q̄ 를 떨어뜨렸을 가능성 (다른 8 task 추가
    확인 가치 있음).

## 9. 진행 중 작업 / 인계

- 백그라운드 SR eval (`eval_steer_compare.sh`, 이전 세션 task ID `bj0toof0q`): 결과는
  `results.tsv` 로 누적. **이 세션과 독립적으로 계속 돌아감.** docker 컨테이너의
  feature_server 프로세스가 살아 있는지 `docker exec groot pgrep -fc feature_server.py` 로
  확인 가능. 완료 시 결과 TSV 로 task × condition matrix 표 구성하면 됨.
- `eval_steer_compare.sh` 의 cleanup 은 worker 별 마지막에 자기 port 의 server 만 kill →
  정상 종료 시 모두 정리.
