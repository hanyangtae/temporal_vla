# exp4-1 실행 계획 — oracle-timing 실패 구제 (확정판)

작성 2026-07-21, **확정 2026-07-22** (Codex Gate1 반영 + 사용자 최종 결정). 공유 배경·연산자 정의·게이트는 [`24_exp4_shared_plan.md`](24_exp4_shared_plan.md)(이하 "공유문서") — 반드시 먼저 읽을 것. Gate1 원장: `docs/collab/2026-07-22-exp4-plan-gate1.md`.

## 확정 스코프 (2026-07-22 사용자 결정)

- **연산자: setpoint형(Ms) + 기존 conceptor(A)만.** 제거형(ablation-to-zero)은 arm에서 제외 (Ms의 s≈0 특수경우로 흡수됨; fit 후 s 값 보고로 갈음).
- **WA-LQR(W): 타당성 게이트 통과 시 추가 시도** (§5). 불가 판정이면 근거 기록 후 생략.
- **축: within-instruction + cross-scene만. cross-instruction은 유예.**
- **Task 4종: OpenStandMixerHead, OpenDrawer, PickPlaceCounterToCabinet(bread), PickPlaceCounterToCabinet(beer).** (07-22 재결정: CloseFridge 탈락 — 실행5/예측16 SR 0/14 chunk-길이 함정. mixer 근거·라벨러 = docs/steering/26, fridge 기록 = 25.)
- **Scene 실현가능성 필터 필수 선행** (공유문서 §5, NOTICE_scene_feasibility_for_exp4.txt): 기하적으로 성공 불가능한 seed는 fit·eval 양쪽에서 동일 제외. rescue 실험에서 특히 치명 — 불가 scene은 어떤 steering으로도 구제 불가라 ITT 분모를 오염시킴.

## 목표

결정적으로 재현되는 실패 episode에 대해, **사용자가 라벨 영상을 보고 지정한 개입 시점 t0**부터 steering을 켜 구제율을 측정한다. 온라인 검출기 없는 oracle 타이밍 — 주장 등급: **"hindsight timing을 포함한 oracle 상한"** (온라인 검출 가능성과 별개).

## 1. Episode set — task별 준비 상태

| task/cell | 결정적 실패 풀 | 영상+사이드카 | 준비 작업 |
|---|---|---|---|
| ppcc_bread (s300033/s400020) | patchceil 77판 (77/77 bitwise 재현 검증) | **완비** (`patchceil/*/rollouts/nopatch/.../task5--ep*--succ0.{mp4,json}`, env_step GT 인라인) | 없음 — 즉시 주석 가능 |
| ppcc_beer, OpenDrawer | exp3 fit15/fit30 수집 실패분 — (scenario_seed, inference_seed, **머신**) 기록 보유 | mp4 있음, env-step 사이드카 없음 | baseline 1회 재실행(수집 머신에서)으로 사이드카 생성 → 주석 팩 |
| OpenStandMixerHead | **신규 cell** — 없음 (스모크 SR 0.33·이봉판정·instruction 1종·fixture 1종, docs/steering/26) | 없음 | ① **scene feasibility 스캔** seed 100000-100099 (`analyze/mixer_scene_feasibility.py` 기성, 실측 100010 BLOCKED) ② feasible seed에서 C0 스캔(exp3 `c0_scan` 패턴)으로 결정적 실패 확보 → baseline + 사이드카 + 재현 검증 |

- **feasibility 필터는 4 task 공통 선행**: mixer는 스크립트 기성, drawer/ppcc는 4-파라미터 이식 후 스캔(오염 여부 미확인 — 기존 patchceil 77판·exp3 실패분도 스캔해서 불가 seed 발견 시 분모에서 제외·기록). 제외 seed·q_max는 annotation manifest에 컬럼으로 기록.
- 결정론은 **머신 단위** — 각 episode는 수집한 머신에서 재현·eval (로컬/srv50(구 w2) 층화, cross-machine 비교는 각주 규칙).
- **ITT 분모 (Codex 반영)**: task별 headline rescue rate의 분모는 "**feasibility 통과한** 결정적 실패 전체"로 고정. 사용자가 주석을 생략한 episode는 **비구제로 계상**(별도로 주석 부분집합 rate 병기). 분모 선별 편향 방지 — 단 기하 불가 seed 제외는 편향이 아님(정책·arm 무관, seed만의 함수·전 arm 동일 적용).
- 1차 착수는 ppcc_bread(준비 완료, feasibility 사후 스캔 병행), 나머지 3 task는 준비 작업과 병행.

## 2. t0 지정 — 사용자 수동 주석

### 2.1 주석 팩 (구현 ~20 LOC)

- 기반: `scripts/safe/groot_n15/robocasa/eval/annotate_phase_video.py` (json 사이드카 → instruction + `step <env_step>` + phase명 오버레이, 기존 동작).
- 추가: **이벤트 마커** — `env_step_grasp_steps/drop_steps/wrong_grasp_steps` 통과 시 라벨 틱 (~15–25 LOC).
- ⚠️ 영상 frame ≠ env-step (steps_per_render=2). t0는 **화면에 구워진 step 숫자**로 읽을 것.

### 2.2 주석 manifest — `annotation_t0.tsv`

```
cell	episode_idx	scenario_seed	inference_seed	t0_env_step	t0_record	note
```

- 사용자는 `t0_env_step`만 기입. **`t0_record = ceil(t0_env_step / 5)` (Codex 반영 — floor는 t0보다 최대 4 env-step 먼저 개입하는 look-ahead가 됨).** 변환은 `build_t0_manifest.py` 한 곳에서만. floor/ceil 민감도는 부차 분석으로 사전 등록.
- **주석 동결**: eval 시작 전 1회 확정. 결과를 본 후의 재주석은 별도 라운드로 표기 (oracle 과대평가 방지).

### 2.3 K와 arm 간 동일성

- **K = t0_record번째 inference부터 episode 끝까지 latch** (record r ⇔ env-step [5r, 5r+5)).
- K는 episode마다 다르지만 **같은 episode에는 모든 arm이 같은 K** — arm 간 차이는 연산자뿐.

## 3. Latch 구현 — client-side

- serve: 기존 gated-phase 경로. phase `steer` 하나 등록, 미등록("off") → identity. eval serve `--collect` 금지, client `--no-features`.
- client(`http_feature_collect.py`): 신규 `--steer-from-record K`, inference 직전 `progress_before >= K ? "steer" : "off"` POST.
- **적용 무결성 감사**: `phase_gated_flags`로 `sum == n_inferences − K` && `first_true == K` 사후 대조, 불일치 rollout 폐기. `[steer-norms]` preflight.
- **GPU id·serve slot을 rollout 사이드카에 기록**하고 arm을 slot 간 순환 배정 (arm×GPU confound 통제, Codex 반영).

## 4. 연산자 준비

### 4.1 Ms — setpoint형 mean-diff (primary treatment)

- **fit**: r̂ = normalize(μ_fail − μ_succ) (비중심화, per-phase × layer), **s = μ_succ·r̂** (성공 평균의 r̂ 좌표 = setpoint).
- **적용**: h′ = h − β[(h·r̂) − s]·r̂ = (I − βr̂r̂ᵀ)h + βs·r̂ — 오차 비례 개입, s 도달 시 개입량 0(자기 소멸), β≤1이면 목표 초과 불가.
- **선행**: ACE(arXiv:2411.09003 — ablation-to-zero의 붕괴 위험 지적, affine+bias로 교정), LEACE 계열(affine erasure), WA-LQR(setpoint 피드백 α=λμ−v·z). novelty 주장 금지, 인용 대상.
- **serve 확장 필요 (~120 LOC)**: 현 hook은 곱셈 전용 → affine(bias 항) 지원 추가 — 신규 `SetpointSteering` hook(벡터 r̂·스칼라 s·β 보유, `steered += −β((h·r̂)−s)r̂`), NPZ에 벡터 키(`alpha0_v_steer`, `alpha0_s`) + 로더 분기, gated registry의 `set_matrices` 대응 `set_vector`. 기존 conceptor 경로와 공존.
- **fit 데이터**(실행 세션 확인 후 택1, task별): ① 같은 cell full-token/기존 수집에 succ·fail 양쪽 있으면 within-cell fit ② 없으면 결정적 재수집(캡처 ON) ③ cross-scene fit은 명시 각주. **구제 대상 ep가 fit fail 클래스에 들어가면 leave-target-out.**
- 진단: r̂ 부트스트랩 각도 분산 + s 값 보고 (s≈0이면 제거형과 동치임을 명기).

### 4.2 Pr — 위약 (label-permutation, Codex 반영)

- 무작위 방향(구안)은 dose-matched가 아님(데이터 분산 축과 안 겹쳐 실제 변화량 과소) → **succ/fail 라벨 순열로 fit한 r̂_perm 여러 개** 사용, held-out ‖Δh‖/‖h‖ 분포가 Ms와 겹치는지 확인 후 동결.

### 4.3 A — 기존 exp3 conceptor (legacy 기준선)

- exp3 배포 NPZ 그대로 (재fit 없음). **"감쇠 단독 대조" 해석은 하지 않는다** (정확한 (1−β)I가 아니고 dose도 Ms와 다름 — Codex 반영). 역할: 사용자 지정 legacy 비교선.

### 4.4 B — exp4-2 산출 conceptor (도착 시 slot-in, 공유문서 §4 계약)

## 5. W — WA-LQR (조건부 arm, 타당성 게이트)

참고 문서: [`24c_walqr_reference.md`](24c_walqr_reference.md) + reading note. 우리 구조는 저들의 DiT4DiT(별도 action DiT)와 가장 유사 — 단 **world 슬롯 제외 설계는 이식 불가**(GR00T action head엔 world 슬롯이 없음) → action 경로 직접 steer.

- **F1 타당성 게이트 (eval 0판, 순서대로 통과 못 하면 중단·기록)**:
  1. 방향/부분공간 fit: succ/fail 대조로 (layer-partition × denoise-t) SVD k≤64 + c_means — CPU, 기존 수집 데이터로 가능한가.
  2. Jacobian: GR00T action-head 블록 간 jvp/vjp가 offline으로 계산 가능한가 (DiT4DiT `compute_jacobians_full.py` 레시피 이식).
  3. serve 주입: §4.1의 affine hook 확장 위에 층별 u = V_out·K·(α·v) 가산기 구현 가능한가 (LingBot `lqr_injector.py:91-227` 참조).
- 통과 시: W arm을 Ms와 동일 프로토콜로 추가 (latch K 동일). α self-gating·chunk-decay는 저들 기본값에서 시작.
- 주의: 저들의 이득은 **교란 하** 측정 — 우리는 nominal 실패 구제라 전이가 자동이 아님을 결과 해석에 명기.

## 6. Arms + 통계

동일 episode·동일 K:

| arm | 연산자 | 역할 |
|---|---|---|
| A0 | 없음 | 결정론 재확인 (sentinel 우선, §8) |
| A | 기존 conceptor (legacy) | 사용자 지정 기준선 |
| Ms | setpoint mean-diff | **primary treatment** |
| Pr | label-permutation setpoint | Ms의 위약 |
| W | WA-LQR (F1 통과 시) | closed-loop 비교 |
| B | exp4-2 conceptor (도착 시) | 유도실패 연산자 |

- Metric: rescue rate (ITT 분모, §1). **Primary contrast: Ms vs Pr, paired McNemar** (task별·사용자 t0에서만). W·B는 각각 vs Pr 동일 프로토콜.
- 참조선: ppcc_bread는 patchceil의 direct action replay 12/77 (15.6%)을 결과표에 병기.
- 계층 게이트: primary 유의 시에만 layer sweep·t0 민감도 확장. Pr>0 자체도 보고(방향 무관 개입만으로 구제되는 실패군).
- 검정 family: task 4 × primary 1 = 4건 Holm 보정. 보고는 confound-audit 경유.

## 7. 러너

- `heldout_round_cell.sh` base → `steer/exp4_1/run_oracle_rescue.sh`: per-episode K를 manifest에서 조회, arm별 serve(연산자 NPZ 교체), fresh process/ep, 빈 GPU × serve 2, GPU id 기록·arm 순환.
- `steer/exp4_1/build_t0_manifest.py`: 주석 검증 + ceil 변환의 유일 지점.

## 8. Smoke (본 eval 전 필수)

1. K=0 → flags 전부 True.
2. K=10^9 → 전부 False + A0와 **bitwise 동일** (identity fallback ≡ 무개입).
3. K=중간 → 첫 True == 정확히 K.
4. **affine hook 항등 smoke**: r̂ 임의·β=0, 그리고 r̂=0 벡터에서 출력 bitwise 불변 (신규 hook 배선 검증).
5. **A0 sentinel (Codex 반영)**: 전량 재실행 대신 cell별 경계 사례 포함 무작위 12판 재현 → 1판이라도 불일치 시에만 전량 확장.

## 9. 예산 (4 task)

- 준비: ppcc_bread 즉시 / beer·drawer 사이드카 재실행 각 수 시간 / mixer feasibility 스캔(CPU, seed당 수초)+C0 스캔+baseline ~1일.
- 주석: task당 실패 수십 편 (사용자 시간; batch 분할).
- 1차 eval: task당 실패 N × arm 4(A0 sentinel 제외 시 실질 3.2) ≈ ppcc_bread 기준 ~250 rollout, 빈 GPU 2장 반나절. 4 task 전체 ≈ 2–3일 (병렬·머신 층화 포함). W·B는 도착 시 +1 arm씩.
- fit·진단: CPU (원격 가능).

## 10. 리스크

- R1 주석 동결 위반(oracle 과대) → §2.2. R2 off-by-one → smoke 3. R3 identity≠no-hook → smoke 2·4. R4 fit-eval 겹침 → leave-target-out. R5 무음 미적용 → flags 감사. R6 머신 층화 위반 → manifest에 machine 컬럼. R7 affine hook 신규 배선 → smoke 4 + Gate2 코드 리뷰. R8 mixer 결정적 실패 풀 부족 가능성(스모크 SR 0.33이라 실패는 많을 것이나 feasibility 제외 후 규모 미정) → 스캔 결과로 사용자 보고. R9 feasibility 스크립트의 한계(정적·관절만, 그리퍼 걸림 미검출·StandMixer 전용이라 타 task는 4-파라미터 이식 필요) → NOTICE §5 숙지, 이식분은 실측 교차검증(rollout 최대 도달값 대조) 후 사용.

## 11. 새 세션 시작 프롬프트 (복붙용)

```
exp4-1(oracle-timing 실패 구제)을 실행한다.
계획: docs/steering/24a_exp4-1_oracle_rescue_plan.md (+ 공유 24_exp4_shared_plan.md) — 먼저 정독.
브랜치: dev에서 exp/exp4-1-oracle-rescue 분기.
순서: ① annotate_phase_video.py 이벤트 마커 확장 → ppcc_bread 77판 주석 팩 생성 후 경로 보고
(내가 annotation_t0.tsv를 채운다) ∥ beer·drawer 사이드카 재실행, mixer feasibility 스캔(100000-100099,
NOTICE_scene_feasibility_for_exp4.txt)+feasible seed C0 스캔, 기존 실패 풀(ppcc·drawer)도 feasibility 사후 스캔
② setpoint affine hook(~120 LOC) + fit_mean_diff.py + Pr(라벨 순열) + client latch + 러너 구현,
구현 완료 시 Codex Gate2 리뷰 ③ WA-LQR F1 타당성 게이트(eval 0판) — 결과 보고 후 W arm 여부 결정
④ 내 주석 도착 후 smoke 5종 → A0 sentinel → 본 eval(A/Ms/Pr[/W]).
GPU는 비어있는 것 확인 후 사용(GPU당 serve 2), exp4-2 세션과 자원 겹침 확인.
문제·불확실성 발생 시 중단하고 보고.
```
