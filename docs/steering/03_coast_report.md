# COAST Conceptor Steering — GR00T N1.6 RoboCasa 적용 진행 보고서

작성일: 2026-05-27

COAST(Contrastive Conceptor Activation Steering)의 conceptor algebra를 구현하고,
GR00T N1.6 RoboCasa rollout feature에 적용해 "성공/실패 부분공간을 대비한
inference-time steering이 가능한가"를 점검한 기록. **결론부터: 모듈/스크립트는
완성·검증됐으나, 현재 가진 단일 layer 데이터에서는 성공/실패 분리도가 낮아(overlap
~0.95) steering 신호가 약하다. 재현의 관건은 layer 다양성(현재 32 layer 중 1개만
수집됨)과 길이 confound 통제다.**

---

## 1. 산출물 (코드)

| 경로 | 내용 | 검증 |
|---|---|---|
| `src/conceptor/{core,steering,analysis,__init__}.py` | COAST conceptor algebra (correlation/conceptor, NOT/AND/OR, contrastive, steering gate, quota/overlap/spectrum) | `tests/test_conceptor.py` 15 passed |
| `src/conceptor/README.md` | 수식·API·GR00T 통합·paper section 매핑·precision·hyperparam 가이드 | — |
| `tests/test_conceptor.py`, `conftest.py` | 단위 테스트 + `from src.conceptor` 절대 import용 sys.path | host `hyundai_aigs`(py3.10) |
| `scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py` | rollout feature → contrastive conceptor fit + COAST A.10.2 α/β 선택 + steering matrix(NPZ) 저장 | smoke + 전체 train run |

검증 수준: 전부 **host smoke**(py_compile, pytest, CLI 실행). Docker robocasa eval(runtime SR)은 미수행.

---

## 2. COAST 절차 매핑 (App. A.9 / A.10.2)

- **conceptor / Boolean algebra / contrastive (Eq. 4)**: `src/conceptor/core.py`. 내부 float64, 저장 float32.
- **steering gate (A.9.2)**: `M = (1-β)I + β·C_steer`, `h' = h·Mᵀ` — `src/conceptor/steering.py`.
- **A.10.2 3-stage 하이퍼파라미터 선택** (`fit_conceptor_steering.py`):
  - Stage 1 (layer, quota): **skip** — 단일 layer(pre-velocity)만 수집돼 layer 선택 불가. quota(C_steer @ α=10)는 진단값으로 기록.
  - Stage 2 (aperture, overlap): COAST GR00T RoboCasa α 그리드 `{0.1,0.3,0.5,0.8,1,1.5,2,3,5,10}`(Table 14)에서 mean overlap이 sweet-spot band `[0.85,0.95]`에 드는 α retain, 없으면 가장 가까운 것.
  - Stage 3 (β): rollout 필요 단계라 굽지 않고 `{0.1,0.3}` 권장값만 기록(0.5는 harmful, drop).

---

## 3. 수치 이슈와 해결 (중요)

**증상**: contrastive AND가 near-singular 입력에서 발산 — C_steer quota가 4231 등
비정상값(conceptor면 [0,1)이어야 함).

**원인**: `and_conceptor = pinv(pinv(A)+pinv(B)-I)`에서
- 작은 α: `C_s`가 near-empty(0 근처 고유값 다수) → `pinv(C_s)` 발산
- 큰 α: `C_f`가 1에 saturate → `NOT(C_f)=I-C_f`가 near-singular → `pinv` 발산

기본 `np.linalg.pinv` rcond(1e-15)는 0 근처 고유값을 거대값으로 역변환. 상대 rcond
상향(1e-6)은 한쪽만 잡혀 불충분.

**해결**: AND 전 conceptor 고유값을 `[ε, 1-ε]`(ε=1e-3)로 clamp 후 plain `inv`.
양쪽 inverse 조건수를 ≤1/ε로 bound → 결과 validity 보장. healthy case 값은 4자리까지
보존. `src/conceptor/core.py`의 `_clamp_spectrum` + `_AND_EIG_CLAMP`, regression test
`test_and_near_singular_stays_valid_conceptor` 추가.

---

## 4. Fit 결과 (seen4 train split, mean/mean pooling, D=1024)

데이터: `safe_split_seen4_unseen2_openDrawer_pnpCab_100ep` train (성공 z 2695 / 실패 z 7155).
산출물: `outputs/.../conceptor_steering/train_hmean_dmean/` (gitignore, 미커밋).

| 그룹 | 선택 α | overlap(C_s,C_f) | quota(C_steer) |
|---|---|---|---|
| global | 0.1 (closest) | 0.956 | **0.001** (C_steer≈0 → steering 신호 없음) |
| task0 CoffeeSetupMug | {5,10} (band) | 0.93~0.95 | 0.015~0.024 |
| task1 OpenSingleDoor | {0.1,0.3,0.5,5,10} (band) | 0.89~0.94 | 0.001~0.029 |
| task3 PnPSinkToCounter | {5,10} (band) | 0.93~0.95 | 0.017~0.026 |
| task4 PnPCounterToStove | 10 (closest) | 0.963 | 0.031 |

---

## 5. 해석 / 한계

1. **mode-conditional이 필수.** global conceptor는 quota≈0.001(모든 task 섞으면 성공/실패
   방향이 상쇄). task별로 나눠야 quota가 산다. (PPGuide의 single global guidance 대비
   우리 차별점인 mode-conditional과 데이터로 일치.)

2. **분리도가 낮다.** healthy α(5,10)에서도 overlap 0.93~0.96. COAST pi0.5 LIBERO는
   α=10에서 0.67까지 떨어졌다(Table 19). 즉 이 pre-velocity layer에서 성공/실패가
   거의 같은 부분공간에 깔려 있다(euclidean silhouette ≈0.008과 일치).

3. **시간축은 COAST와의 차이가 아니다.** COAST도 timestep을 순서 없이 모은 정적 분포로
   conceptor를 fit한다(우리와 동일). 따라서 "LSTM은 시간을 쓰는데 conceptor는 못 쓴다"는
   COAST 재현 실패의 설명이 아니다. 진짜 변수는 **그 정적 표현(layer)에 순간 분리가
   있느냐**이고, COAST는 layer 선택(GR00T ℓ=10)으로 그걸 확보했다.

4. **band 선택이 degenerate α를 잘못 고른다.** COAST는 overlap이 α에 monotone 감소함을
   가정하나, 우리 데이터는 비monotone(small α에서 C가 비어 overlap이 *낮게* 나오는
   허위 진입). COAST는 Stage 1(high-quota layer)이 이를 걸러주지만 우리는 단일 layer라
   보호가 없다. → band 선택에 **quota floor guard**(C_steer가 거의 0인 α 제외) 필요.
   (미적용, 결정 대기)

5. **seen18 데이터의 길이 confound.** 새 데이터
   `target_atomic_seen18_ckpt120000_robocasa365_100ep`는 **실패=항상 timeout(45step) /
   성공=조기종료**라 길이만으로 라벨이 갈린다(AUROC 0.998). conceptor는 timestep을
   샘플로 모으므로 실패 분포가 late-timestep activation에 지배 → "늦은 timestep" 방향을
   "실패"로 학습하는 **길이 아티팩트**. time-pooled 분리는 신뢰 불가. **길이 통제 필수**
   (공통 horizon truncate / step 위치 매칭·subsample, step index 저장).

---

## 6. 데이터 / 용량 견적 (multi-layer 수집)

- **GR00T N1.6 action head = 32-layer diffusion DiT** (N1.5는 16; `gr00t_n1d6.py`), D=1024.
  현재 수집은 맨 끝 출력 1점(pre-velocity)뿐.
- 현재 1 layer raw(`[K=4,H=16,D=1024]` fp16) = **7.2GB**/1800 rollout. timestep당 128KB.
  conceptor는 K·H 평균 `[1024]`만 필요 → **÷64**.

| 한 pass 저장 포맷 | 32 layer 전부 | 가능 분석 |
|---|---|---|
| pooled `[1024]` | **3.6GB** | global conceptor (Stage 1 layer 선택 OK) |
| `[K=4,1024]` (denoise step 유지) | **14GB** | global + per-step |
| raw `[4,16,1024]` | 230GB | 전부 (H별까지, 과함) |

- 최종 선택 steering matrix만 영구 저장: ~150~400MB (sweep 행렬은 pooled z에서 즉시 계산).
- **rollout은 1회면 충분.** forward 1번에 32 layer가 다 계산되므로 hook만 32곳 + 저장
  포맷만 정하면 된다. seed non-repro라 2-pass는 scene이 달라져 오히려 불가.
- 진짜 비용은 용량이 아니라 **재수집 GPU 시간**(현재 pkl은 1 layer만 + obs 미저장이라
  offline 재forward 불가).

---

## 7. 다음 단계 (결정 대기)

1. **band 선택 quota guard 적용 여부** — 적용 시 task별 선택 α가 {5,10}로 정리, Step 3
   config = task별 {5,10}×{0.1,0.3}.
2. **multi-layer 재수집(1 pass)** — `safe_hooks.py`에 32-layer hook 추가, pooled `[1024]`
   또는 `[K,1024]` + step index 저장. **길이 통제 설계 동반.**
3. 재수집 후: COAST Stage 1(quota로 layer 선택, 길이 통제 상태) → 분리되는 layer 탐색.
   - 분리 layer 발견 → Step 2(steering hook) + Step 3(steering on/off eval).
   - 어느 layer에도 순간 분리 없음 → "이 모델/태스크엔 conceptor steering이 약한 도구"
     라는 결론, pivot.

---

## 참고
- COAST 논문: `docs/COAST.pdf` (A.9 구현 세부, A.10.2 하이퍼파라미터 선택).
- 관련 메모: seen18 길이 confound, collection seed non-repro, RoboCasa runtime env.
- SAFE feature 정의/로더: `scripts/safe/groot_n16/robocasa/safe_feature_vectors.py`.
