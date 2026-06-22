# 16. COAST 재현 (π0.5 × LIBERO-10, GLOBAL) — Positive Control

## 목적 (왜 이 트랙인가)

GR00T-N1.5 × RoboCasa COAST 재현은 충실 네이티브 스택에서 mean ΔSR ≈ +0.014 (≈0)로
**positive control 실패** (n=30, 2026-06-22). COAST 논문 자체는 RoboCasa에서도 N1.5 +0.16을
보고하므로, 우리 스택과 논문 스택 사이 어딘가에 차이가 있다.

이 트랙은 **다른 (model, benchmark) 페어로 control을 다시 잡는다**:

- COAST가 **모든 벤치마크 중 가장 큰 절대 gain**을 보고한 곳 = **π0.5 × LIBERO-10**
  (Base 0.43 → Global 0.76, **ΔSR +0.33**; COAST.txt:366).
- 즉 "재현 코드/파이프라인이 옳다면 신호가 가장 크게 떠야 하는" 조건.
- 여기서도 ΔSR이 0이면 우리 conceptor fit/steer 배선 버그를 의심, 신호가 뜨면
  RoboCasa/N1.5 트랙의 0은 **스택 차이(해상도·base SR·denoise step 등)** 쪽으로 좁혀진다.
- 전략은 **GLOBAL 만** (C_steer = C_success ∧ ¬C_failure). Per-Step / Pos.-Only는 범위 밖.

출처: COAST arXiv 2605.17144, `docs/references/COAST.txt`. 모든 수치에 줄번호 인용.

---

## 충실 spec (faithful reproduction table)

| 항목 | 값 | COAST.txt 인용 |
|---|---|---|
| 모델 | π0.5 (frozen PaLI-Gemma2 prefix + trainable Gemma2 action expert) | :1401-1406 |
| action expert | 18 decoder layer, hidden d=1024, AdaRMS | :1402-1406 |
| 체크포인트 | openpi-libero **step-2000** (early; 대부분 task에서 처음으로 nonzero success) | :1551, :1918 |
| denoising | 10 Euler step, 전부 실행 (early stop 없음) | :1408-1411 |
| action horizon | 10 | :1919 |
| hook 지점 | `gemma_expert.model.layers[{0,5,11,17}]` residual stream | :1413-1416 |
| **steer layer ℓ** | **11** (geometric 선택) | :1856-1857 |
| token pooling | suffix **S=10** action token을 token축으로 **mean-pool** → layer·denoise step당 1 벡터 | :1442-1446 |
| 집계 (global) | 모든 denoising step의 활성화를 class당 1 conceptor로 pool | :284 |
| α (global) | {0.5, 1.0} | :1856 |
| β (global) | {0.1, 0.3} | :1856 |
| 전략 | global only | :1856 |
| fit episodes | **15 ep/task** (`--num_episodes 15`) | :1542-1543 |
| test episodes | **30 ep/task**, fit과 **disjoint seed** | :1544-1545 |
| success label | LIBERO native `info["is_success"]` (마지막 step 기준) | :1548-1549 |
| task 제외 | success < 3 **또는** failure < 3 인 task는 conceptor fit에서 제외 | :1553-1554 |
| conceptor | **per-task** (task별 따로 fit), per-task oracle α/β/layer는 Table 17 | :1862-1875 |

per-task oracle 하이퍼파라미터는 `scripts/safe/pi05/libero/steer/coast_pi05_libero_hyperparams.json`
의 `per_task` 맵 (Table 17, :1865-1875)에 전사. global 기본값은 같은 파일 `_global_config`.

---

## 기대 수치 (Table 1 / Table 4)

π0.5 × LIBERO-10, mean SR (COAST.txt Table 1 mean row, :366; Table 4 Global mean :1037과 일치):

| 조건 | mean SR | ΔSR |
|---|---|---|
| Base (unsteered) | **0.43** | — |
| +CAA (linear) | 0.47 | +0.04 |
| **COAST Global** | **0.76** | **+0.33** |
| COAST Per-Step | 0.80 | +0.37 |
| COAST Pos.-Only | 0.63 | +0.20 |

per-task Global SR (Table 4, :1027-1036) — 우리 ΔSR 비교의 task별 target:

| task (abbrev) | Global SR | ℓ/α/β (oracle) |
|---|---|---|
| KITCHEN_SCENE3 (stove + moka) | 0.93 | L5 / 0.5 / 0.1 |
| KITCHEN_SCENE4 (bowl in drawer) | 0.73 | L5 / 1.0 / 0.1 |
| KITCHEN_SCENE6 (mug in microwave) | 0.40 | L11 / 0.1 / 0.3 |
| KITCHEN_SCENE8 (two moka pots) | 0.47 | L5 / 1.0 / 0.1 |
| LIVING_ROOM1 (soup + cheese) | 0.93 | L11 / 0.5 / 0.3 |
| LIVING_ROOM2a (soup + tomato) | 0.80 | L11 / 0.5 / 0.3 |
| LIVING_ROOM2b (cheese + butter) | 0.93 | L5 / 0.5 / 0.1 |
| LIVING_ROOM5 (two mugs on plates) | 0.60 | L11 / 0.5 / 0.5 |
| LIVING_ROOM6 (mug + pudding) | 0.87 | L5 / 0.5 / 0.1 |
| STUDY_SCENE1 (book in caddy) | 0.93 | L5 / 0.5 / 0.1 |

참고: geometric 선택 ℓ=11은 10개 중 5개 task의 oracle layer와 일치, 나머지 5개는 oracle이
layer 5를 선호하나 quota가 비슷해 gap은 보통 <5pt (:1876-1882).

진단 (Table 19, :1886): layer 11 success–failure overlap이 α에 따라 감소
(α0.1→0.955, 0.5→0.937, 1.0→0.882, 2.0→0.808, 10→0.670) — overlap 큰 task에서 gain 큼.

---

## RoboCasa/GR00T-N1.5 트랙과의 차이

| 축 | groot_n15 × RoboCasa (control 실패) | **이 트랙: π0.5 × LIBERO-10** |
|---|---|---|
| 모델 | GR00T N1.5 (flow-matching DiT, self-attn) | π0.5 (Gemma2 action expert) |
| 벤치마크 | RoboCasa 7 atomic-seen | LIBERO-10 (10 task) |
| 체크포인트 | 충실 native 스택 (base SR 0.443) | openpi-libero **step-2000 (early)** |
| steer layer | ℓ=10 | **ℓ=11** |
| α/β | α=0.1, β{0.1,0.3} (uniform) | α{0.5,1.0}, β{0.1,0.3} (per-task oracle) |
| denoise | 4 step | **10 step** |
| token pool | S=49 (1 state +32 future +16 action) | **S=10** action token |
| 논문 ΔSR | +0.16 (우리 재현 ≈+0.014) | **+0.33** (가장 큰 gain) |
| 우리 상태 | control **실패** | control **재시도** |

핵심: 이 트랙은 base SR이 낮은 early checkpoint(0.43)라 **succ/fail mix가 충분**해
contrastive conceptor fit이 잘 되고, 논문 ΔSR도 가장 크다 → control 신호가 가장 잘 떠야 함.

---

## 파이프라인 단계

1. **collect** — openpi-libero step-2000 체크포인트로 LIBERO-10 각 task **15 fit ep** rollout,
   ℓ∈{0,5,11,17} residual stream 활성화(S=10 mean-pool) + `info["is_success"]` 라벨 저장.
2. **per-task conceptor fit** — task별로 succ<3 또는 fail<3이면 제외, 나머지는
   C_success ∧ ¬C_failure (global, denoise step 전부 pool) fit. α/β는 `per_task` 맵(oracle)
   또는 global {0.5,1.0}×{0.1,0.3}.
3. **serve baseline/steered** — π0.5 serve에 ℓ=11 hook 배선. baseline = steer off,
   steered = h' = h·Mᵀ (global conceptor 적용).
4. **ΔSR eval** — task별 **30 test ep** (fit과 disjoint seed), baseline vs steered SR 측정,
   ΔSR을 Table 1/Table 4 target과 비교. 같은 seed pair로 condition 간 noise 제거.

산출 config: `scripts/safe/pi05/libero/steer/coast_pi05_libero_hyperparams.json`.
