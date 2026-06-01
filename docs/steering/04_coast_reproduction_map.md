# COAST 재현 매핑 (파일·함수 ↔ 논문)

작성일: 2026-05-29
목적: 우리가 구현한 코드가 COAST(`docs/references/COAST.txt`) 의 어느 식/섹션을 **어떻게**
구현했는지 한눈에 검증할 수 있게 한다. 충실 재현(✓) / 의도적 변경(⚠) / 미구현(✗) 을 표시한다.

대상 논문: *Contrastive Conceptor Activation Steering (COAST)*. 우리 대상 모델은 **GR00T N1.6**
(d=1024, 32-layer DiT) — COAST 의 GR00T 는 **N1.5** (d=1536, 16-layer). 체크포인트 번호는
동일(robocasa365 `checkpoint-120000`).

---

## 0. 한눈에 — 무엇이 재현됐나

| 구성요소 | 상태 | 비고 |
|---|---|---|
| Conceptor 계산 (C=R(R+α⁻²I)⁻¹) | ✓ 충실 | 단위테스트로 PSD·고유값∈[0,1) 검증 |
| Boolean algebra (NOT/AND/OR) | ✓ (AND은 ⚠ 수치 변경) | AND은 pinv 대신 고유값 clamp |
| Contrastive conceptor (Eq.4) | ✓ 충실 | |
| Multiplicative gating (Eq.5, h'=hMᵀ) | ✓ 충실 | |
| 하이퍼파라미터 선택 (A.10.2) | ✓ Stage2/3, ✗ Stage1 | layer 선택은 단일 layer라 skip |
| Steering strategies (Sec.3.2) | ⚠ global만 | per-step/positive-only 미실행 |
| Activation 추출 (A.7.2) | ✓ + ⚠ 확장 | 길이 confound 통제 모드 추가(COAST엔 없음) |
| Inference-time hook 적용 | ✓ 충실 | 단, **단일 layer(pre-velocity)** — COAST는 mid-DiT ℓ=10 |
| SR 개선 재현 (Sec.4.2) | ✗ **미재현** | 우리 layer에선 개선 없음(아래 §6) |

**요약**: conceptor **수학(algebra·gating·하이퍼선택)은 충실히 재현**됐고 단위테스트로 검증됨.
하지만 **실험적 SR 개선은 재현 안 됨** — 핵심 원인은 (a) layer 선택(Stage1)을 못 함(단일
pre-velocity layer만 수집, COAST는 16개 중 ℓ=10 선택), (b) 우리 데이터의 극단적 길이 confound.

---

## 1. 파일 목록

| 파일 | 역할 | 논문 대응 |
|---|---|---|
| `src/conceptor/core.py` | conceptor 계산 + Boolean algebra + contrastive | Sec.3.1, App.A.9.1–9.4, Eq.4 |
| `src/conceptor/steering.py` | multiplicative gate + 적용 | Eq.5, A.9.2 |
| `src/conceptor/analysis.py` | quota/overlap/spectrum/containment | Eq.10/11, A.10.2, Fig.3A, Sec.4.3/4.4 |
| `src/conceptor/__init__.py` | 공개 API | — |
| `tests/test_conceptor.py` | conceptor 단위테스트 (15) | 수학 속성 검증 |
| `tests/test_steering_hook.py` | hook 단위테스트 (3) | A.9.2 적용 검증 |
| `scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py` | 활성화→conceptor fit + A.10.2 선택 | A.7.2, Sec.3.1, A.10.2 |
| `scripts/serve/steering_hooks.py` | inference forward hook (ConceptorSteering) | A.9.2 적용 |
| `scripts/safe/groot_n16/robocasa/serve/feature_server.py` (수정) | serve에 steering 배선 | A.9.2 배포 |
| `scripts/safe/groot_n16/robocasa/steer/run_steer_eval.sh` | baseline vs steered SR eval | A.8 / Sec.4.2 |
| `configs/checkpoints/groot__robocasa365_ckpt120000.yaml` (수정) | ckpt 경로 cache 정정 | — (인프라) |

참고: `vis/coast_conceptor_spectrum.py`, `coast_fig4a_subspace.py`, `coast_fig4b_v1_trajectory.py`
는 `src.conceptor` 를 소비하는 **분석/시각화**(Fig.3A·Fig.4 재현)이며 본 모듈 위에서 동작한다.

---

## 2. 상세 매핑 — `src/conceptor/core.py`

| 함수 | 구현 | 논문 위치 | 충실도 |
|---|---|---|---|
| `compute_correlation(X)` | X̃=X−mean(X); R=X̃ᵀX̃/N | A.9.1 step1–2 ("mean-center → R=(1/N)X̃ᵀX̃") | ✓ |
| `compute_conceptor(X,α)` | C=R(R+α⁻²I)⁻¹, `np.linalg.solve(R+α⁻²I, R)` (대칭) | Sec.3.1 / A.9.1 step3. 고유값 λ=σ/(σ+α⁻²)∈[0,1) | ✓ |
| `not_conceptor(C)` | I−C | A.9.2 (Jaeger NOT) | ✓ |
| `and_conceptor(A,B)` | 고유값 [ε,1−ε] **clamp** 후 `inv(inv(A)+inv(B)−I)` | A.9.2 / Sec.3 AND `(A⁻¹+B⁻¹−I)⁻¹` | ⚠ (§4-1) |
| `or_conceptor(R_A,R_B,α)` | (R_A+R_B)(R_A+R_B+α⁻²I)⁻¹ | A.9.2 (Jaeger OR) | ✓ (미사용) |
| `contrastive_conceptor(Xs,Xf,α)` | `and(C_s, not(C_f))` = C_s∧¬C_f | **Eq.4** | ✓ |
| 정밀도 | 계산 float64, `as_float32`로 저장 | A.9.4 | ✓ (AND은 ⚠) |

검증: `tests/test_conceptor.py` — PSD, 고유값∈[0,1), n≪d 작동, α-단조(quota), NOT-idempotence,
AND-commutativity, **near-singular AND validity**(clamp 회귀테스트), overlap self=1 등 15개 통과.

## 3. 상세 매핑 — `src/conceptor/steering.py`

| 함수 | 구현 | 논문 위치 | 충실도 |
|---|---|---|---|
| `build_steering_matrix(C,β)` | M=(1−β)I+β·C_steer | **Eq.5** / A.9.2 | ✓ |
| `apply_steering(h,M)` | h'=h·Mᵀ ((B,S,d)/(B,d)) | A.9.2 "h'=hM⊤ before remaining layers" | ✓ |

## 4. 상세 매핑 — `src/conceptor/analysis.py`

| 함수 | 구현 | 논문 위치 | 충실도 |
|---|---|---|---|
| `conceptor_quota(C)` | (1/d)tr(C) | **Eq.10** / A.10.2 Stage1 / Table18 | ✓ |
| `conceptor_overlap(A,B)` | tr(AB)/√(tr(A²)tr(B²)) | **Eq.11** / A.10.2 Stage2 / Sec.4.3 | ✓ |
| `eigenvalue_spectrum(C)` | 내림차순 고유값 | Fig.3A | ✓ |
| `failure_containment(s,t)` | tr(C_s·C_t)/tr(C_s²) | Sec.4.4 cross-task | ✓ (미사용) |

## 5. 상세 매핑 — `fit_conceptor_steering.py` (활성화→conceptor + 하이퍼선택)

| 단계 | 구현 | 논문 위치 | 충실도 |
|---|---|---|---|
| 활성화 로드 | success/fail rollout의 pooled z (DiT pre-velocity, K·H mean) | A.7.2 "residual stream at layer ℓ, mean-pool action tokens, one h per denoising step" | ✓ (단일 layer, §4-2) |
| per-task fit | task별 C_s/C_f/C_steer | Sec.3.1, Sec.4.2(per-task) | ✓ |
| Stage1 (layer, quota) | quota(C_steer)@α=10 **기록만**, 선택 안 함 | A.10.2 Stage1, Eq.10 | ✗ (§5-skip) |
| Stage2 (aperture, overlap) | α grid sweep → overlap∈[0.85,0.95] band 선택, 없으면 최근접 | **A.10.2 Stage2**, Eq.11 | ✓ |
| α grid | {0.1,0.3,0.5,0.8,1,1.5,2,3,5,10} | Table14 GR00T RoboCasa | ✓ |
| Stage3 (β) | {0.1,0.3} 권장 기록(0.5 drop), rollout 미실행 | A.10.2 Stage3 | ✓ |
| time-agg 모드 | `coast`(전 timestep) / `episode_mean` / `truncated_wW` | coast=A.7.2 충실; 나머지는 **우리 확장** | ⚠ (§4-3) |
| strategy | `global`(K·H mean) | Sec.3.2 global | ⚠ per-step/positive-only ✗ |

## 6. 상세 매핑 — inference hook & eval

| 항목 | 구현 | 논문 위치 | 충실도 |
|---|---|---|---|
| `steering_hooks.ConceptorSteering` | `action_head.model`(DiT) forward hook, action token에 h'=hMᵀ, per-task M, β보간 | A.9.2 "forward hook on selected action-expert layer's output" | ✓ (단, layer=pre-velocity) |
| `feature_server.py` 배선 | `--steering-npz/beta/alpha/key` → DiT에 영구 hook 등록 | A.9.2 배포 | ✓ |
| `run_steer_eval.sh` | baseline(β=0) vs β=0.3 모드별 SR, 같은 eval 파라미터 | A.8 eval 프로토콜 / Sec.4.2 | ✓ (프로토콜), ✗ (결과) |

---

## 7. 의도적 변경(⚠) 과 이유

**4-1. AND 연산 — pinv 대신 고유값 clamp.**
COAST A.9.4 는 near-singular conceptor를 Moore-Penrose `pinv`로 처리한다고 명시. 우리 데이터
(n≪d + 큰 α에서 C_f가 1에 saturate → I−C_f near-singular)에서는 default `pinv`가 0근처 고유값을
거대값으로 역변환해 **결과가 발산**(quota 수천)했다. 그래서 입력 conceptor 고유값을 `[ε,1−ε]`
(ε=1e-3)로 clamp 후 plain `inv`. 두 valid conceptor의 AND는 valid해야 한다는 불변식을 보장하고,
healthy 케이스 값은 4자리까지 보존(회귀테스트 `test_and_near_singular_stays_valid_conceptor`).
→ **의미는 동일(robust AND), clamp라는 정규화가 추가됨.** aperture를 위/아래로 capping하는 것과 동치.

**4-2. 단일 layer (pre-velocity) — COAST의 layer 선택과 다름.**
우리 활성화는 SAFE 수집물인 **DiT 출력 직전(pre-velocity) 1개 지점**만 있다. COAST는 action-expert
residual stream을 **여러 layer**(GR00T N1.5는 16개)에서 뽑아 Stage1 quota로 **ℓ=10**을 선택해 steer
한다. 우리 지점은 사실상 마지막(decode 직전)이라 COAST가 고른 중간 layer와 다르다. **이것이 SR
미개선의 가장 유력한 원인.**

**4-3. 길이 confound 통제 모드 — COAST에 없는 우리 확장.**
seen18은 실패=timeout(45 step)/성공=조기종료(mean 17.7)로 길이가 라벨을 결정(AUROC 0.998).
COAST는 이 confound를 다루지 않고 전 timestep을 stack한다(=우리 `coast` 모드). 우리는 추가로
`episode_mean`(최대 confound, 아티팩트 천장)·`truncated_wW`(첫 W step만, 길이 통제)를 만들어 비교.
결과(아래): 길이 통제 시 overlap이 0.86→0.95→0.97로 올라 **분리도 대부분이 길이 아티팩트**임이 드러남.

## 8. 미구현 / Skip (✗)

- **Stage1 layer 선택**: 단일 layer라 불가. quota는 진단값으로만 기록. → multi-layer 재수집 필요.
- **per-step / positive-only strategy** (Sec.3.2): global만 fit·eval. per-step은 기존 pkl의 K축으로
  추가 fit 가능(데이터 있음), positive-only는 C_success로 즉시 가능 — 미실행.
- **β=0.1, 다중 task, 검정력 있는 episode 수**: D는 1 task(PickPlaceCounterToStove)·β=0.3·20ep만.
- **cross-task transfer (Sec.4.4)**: `failure_containment` 구현했으나 실험 미수행.

---

## 9. 검증 상태

**수학 (단위테스트, host `hyundai_aigs`)**: `pytest tests/test_conceptor.py tests/test_steering_hook.py`
→ 18 passed. conceptor PSD·고유값∈[0,1)·n≪d·NOT idempotence·AND commutativity·near-singular
validity·β=0 identity·action-token만 steer 등 검증.

**Geometry (seen18 fit, §4-3)**: overlap@selected (task 평균)
- episode_mean 0.86 / coast 0.95 / truncated_w18 0.97 → **길이 통제 시 genuine 분리 ≈ 없음**.
- C_steer quota 0.002~0.005 (거의 빈 부분공간).

**Eval (D, PickPlaceCounterToStove, 20ep, GR00T N1.6 robocasa365 ckpt-120000)**:

| config | β | SR | vs baseline |
|---|---|---|---|
| baseline | 0 | 0.65 (13/20) | — |
| coast | 0.3 | 0.55 | −0.10 |
| episode_mean | 0.3 | 0.55 | −0.10 |
| truncated_w18 | 0.3 | 0.45 | −0.20 |

- **SR 개선 0건** (모두 baseline 이하). n=20이라 통계적 유의 없음(2-prop z≈1.3, p≈0.2) → "개선
  신호 없음 + 약한 하락 경향"이 정확. baseline 0.65 ≈ collection 0.61(노이즈 내).
- geometry(분리≈0)와 일관. C_steer가 거의 비어 M≈(1−β)I(균일 감쇠)에 가까워 directional 개선 없음.

---

## 10. 재현 충실도 종합 (정직한 평가)

- **알고리즘 재현: 충실.** conceptor/Boolean/contrastive/gating/하이퍼선택(Stage2,3)이 식 그대로
  구현·검증됨. 유일한 수학적 변경은 AND의 수치 안정화(clamp)이며 의미 보존.
- **실험 재현: 미달.** COAST의 +0.16 SR(GR00T N1.5 RoboCasa) 같은 개선을 **재현 못 함**. 이유는
  버그가 아니라 **조건 차이**: (1) 단일 pre-velocity layer(COAST는 ℓ 선택), (2) 극단적 길이 confound,
  (3) global·단일 β·1 task의 제한된 sweep.
- **판정**: "COAST 메커니즘을 우리 코드로 정확히 재현했고, 그것을 우리 단일 layer에 적용했더니
  이 layer에선 효과가 없다"가 현재 상태. COAST 결과 자체의 재현 여부를 가르려면 **multi-layer
  수집 후 Stage1 layer 선택**(보류했던 단계)이 필요하다 — 그게 다음 결정 포인트.

## 11. 재검증 방법 (사용자용)

```bash
conda activate hyundai_aigs
pytest tests/test_conceptor.py tests/test_steering_hook.py -v     # 수학 18개
# fit 재현 (3모드, 캐시 입력):
python scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py \
  --cache outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/feature_cache/pooled_all_hmean_dmean.npz --per-task
# steered eval (docker, ~2.5h): scripts/safe/groot_n16/robocasa/steer/run_steer_eval.sh
```
