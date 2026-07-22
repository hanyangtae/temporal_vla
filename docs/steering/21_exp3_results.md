# exp3(구 pq3) 최종 결과 — COAST 축 정렬 검증 (2026-07-16 완주)

계획: `~/.claude/plans/dynamic-riding-aurora.md` (v9) · 핸드오프 `docs/steering/20`.
판정: `outputs/eval/robocasa/groot_n15/steer_eval_exp3/aggregate_f/` (decision_sha
ab0e3abd59ddf52c, 동결 `exp3_decision.py`). 총 900판 (5 cell × 6 arm × 30판, 사전등록 4 arm
+ 탐색 within 2 arm), expected-count 전판 충족·host-block 위반 0·incomplete 0.

## 사전등록 판정 (6-Holm, 단측 exact paired McNemar) — 전부 기각 실패

| 가설 | n | W/L | Δ판 | p | p_adj | CI95 상한(Δ판율) |
|---|---|---|---|---|---|---|
| H1 drawer (perm>base) | 60 | 4/3 | +1 | .50 | 1.0 | +0.083 |
| H2 drawer (gated>base) | 60 | 11/7 | +4 | .24 | 1.0 | +0.183 |
| H3 drawer (gated>perm) | 60 | 10/7 | +3 | .31 | 1.0 | +0.167 |
| H1 ppcc | 90 | 10/6 | +4 | .23 | 1.0 | +0.122 |
| H2 ppcc | 90 | 9/10 | −1 | .68 | 1.0 | +0.067 |
| H3 ppcc | 90 | 5/10 | −5 | .94 | 1.0 | +0.011 |

- null 관문: drawer null−base=+3, ppcc +1 (margin 4 이내) → H1 해석 유효.
- **final_status = all_null_nonreplication**: H1 CI 상한(drawer +0.083, ppcc +0.122) < COAST
  +0.16 → **"COAST +0.16은 축 전부 정렬 + scene-diverse 조건에서 비재현"** 선언 성립.
- 위약(null perm, 라벨 permutation seed 1): drawer +3/W4 L1, ppcc +1/W7 L6 — 실제 perm arm과
  구분 불가 수준 (exp2(구 pq2) 위약 동률 재확인).

## 탐색 arm — per_instruction(within) vs cross_instruction fit

| | drawer W/L (n=60) | ppcc W/L (n=90) |
|---|---|---|
| perm cross | 4/3 | 10/6 |
| gated cross | 11/7 | 9/10 |
| **perm within** | **7/2** | 8/8 |
| **gated within** | **7/1** | 8/6 |
| null | 4/1 | 7/6 |

- drawer 에서는 within 이 cross 를 상회(gated within 7/1, 단측 이항 ~.035 수준·미보정)하고
  손실이 거의 없으며 unseen 에도 이득이 실렸다 — 그러나 **PPCC 에서 비재현**(perm 8/8,
  gated 8/6 ≈ 위약 7/6). 2 cell·탐색·미보정이므로 **단정 금지**, 후속 라운드의 가설 후보로만.
- cross gated 의 drawer +4 는 seen(fit scene) 집중(+5)·unseen 음(−1), right/unseen 1/4 역전 —
  in-sample rescue 서명.

## 부수 관찰

- pizza_cutter: base SR 0.93 천장 — 개선 검출력 없음(무해성만 확인: gated within 29/30).
- Stage1 기하 layer: drawer L0(pool·epeq 일치), PPCC perm pool L4 vs epeq L15 불일치 →
  sweep 시험으로 L15 채택(in-sample perm +8). epeq(길이-강건) 기준이 sweep 과 일치.
- gated 성립 게이트(enforce, floor 10/0.8): 두 task 모두 pass, LOO 0.87–0.99.
- S9 3-host canary: A100 두 대는 10/10 ep 동일, 로컬만 1 ep 분기 — cell-블록 배정 유지
  (예외 1건: pizza null 을 48→50 이동, A100↔A100 일치 근거 각주).
- beer base 로컬 완주분(22/30 과 별개 실행)은 host-효과 참고용으로
  `e1_partial_localmoved/` 보관.

## Confound audit (스킬 체크리스트)

| # | 게이트 | 판정 | 근거 |
|---|---|---|---|
| 1 | Length | N/A | 판정은 SR(이진), latent 시간-pool 주장 없음 |
| 2 | Task identity | pass | 판정 paired within-cell, task 별 가설 분리 |
| 3 | Instruction balance | pass | cell=instruction 단위 paired — 변형 간 불균형이 오염 불가 |
| 4 | In-sample rescue | pass | fit∩eval-unseen=∅ 5 cell 동결산물 재검증(교집합 0); seen 15 는 설계 명시·분해 병기 |
| 5 | Rollout pooling | pass | per-record·per-step conceptor, episode-mean 없음 |
| 6 | Phase/dwell | N/A | SR 판정 (gated fit 은 phase-bin) |
| 7 | Observation≠causation | pass | 본 결과가 곧 개입(ΔSR) 측정 |
| 8 | Scene-local≠general | pass | 판정 pooled·scene-diverse; 국소 신호(drawer_left seen)는 아티팩트 후보로 명시 |

**Claim-strength: intervention effect** (전 arm ΔSR 인과 재측정, 사전등록 판정).

## 결론

raw 대조 conceptor steering 은 **COAST 축을 전부 정렬하고 scene-diverse 로 확장해도
위약 이상 효과가 없다** (exp2 종결 확증을 축-정렬 조건에서 재확증, COAST +0.16 비재현).
남은 미청산 관찰은 drawer 한정 within-instruction fit 의 약신호(7/1) 하나 — 재현 라운드
(confirmatory fresh seeds) 없이는 채택 불가.
