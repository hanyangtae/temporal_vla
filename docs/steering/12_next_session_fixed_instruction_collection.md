# 다음 세션 핸드오프 — fixed-instruction 재수집 (option A)

작성: 2026-06-08.

## 목적

instruction confound(`docs/steering/11_instruction_confound.md`) 때문에 mixed-instruction run의
succ/fail 분리가 instruction identity 아티팩트로 오염됨. 이를 제거하려면 **instruction을 고정**해
(객체/방향/목적지 강제) within-instruction succ/fail 신호를 충분한 표본으로 확보해야 한다.

→ **instruction별 50 episode**, seed 고정, VL+DiT pathway 캡처로 fresh 수집.

## 수집 대상 — 10 instruction × 50 ep (option A: 전부 fresh)

기존 pathway_pertoken run(100ep mixed)에서 이미 ≥50 인 5개는 **재사용**(아래 "재사용" 표),
<50 인 10개만 새로 fresh 수집.

| # | task | instruction (정확 lang) | 기존 |
|---|---|---|---|
| 1 | PickPlaceCounterToStove | `Pick the onion from the plate and place it in the pan.` | 9 |
| 2 | PickPlaceCounterToStove | `Pick the apple from the plate and place it in the pan.` | 7 |
| 3 | PickPlaceDrawerToCounter | `Pick the tongs from the drawer and place it on the counter.` | 18 |
| 4 | PickPlaceDrawerToCounter | `Pick the wooden spoon from the drawer and place it on the counter.` | 14 |
| 5 | PickPlaceCounterToCabinet | `Pick the potato from the counter and place it in the cabinet.` | 4 |
| 6 | PickPlaceCounterToCabinet | `Pick the bread from the counter and place it in the cabinet.` | 4 |
| 7 | OpenDrawer | `Open the left drawer.` | 38 |
| 8 | NavigateKitchen | `Navigate to the fridge.` | 18 |
| 9 | NavigateKitchen | `Navigate to the coffee machine.` | 14 |
| 10 | CoffeeSetupMug | `Pick the mug from the counter and place it under the coffee machine dispenser.` | 0 (run 미포함) |

**재사용 (≥50, 재수집 불필요):**

| task | instruction | 기존 |
|---|---|---|
| OpenCabinet | `Open the cabinet door.` | 50 |
| OpenDrawer | `Open the right drawer.` | 62 |
| CloseToasterOvenDoor | (단일) | 100 |
| TurnOnMicrowave | (단일) | 100 |
| TurnOnSinkFaucet | (단일) | 100 |

## 수집 시 주의

- **instruction 강제 고정**: PnP는 객체(onion/apple/tongs/wooden spoon/potato/bread/mug),
  OpenDrawer는 behavior(left), Navigate는 목적지(fridge/coffee machine)를 env 단에서 고정.
  랜덤 샘플이 아니라 매 episode 같은 instruction이 뜨도록.
- **seed 고정**: 표준 EVAL/collection seed (CLAUDE.md 평가 표준, seed start 100000). 기존
  mixed run과 직접 병합 금지(랜덤샘플 seed/scene 불일치) → fresh 10개는 독립 run으로.
- **VL+DiT pathway 캡처**: `--capture-vl` + DiT multilayer(per-token, layer {0,2,4,8,16,24,31}).
  `collect_pathway_parallel.sh` 경로 사용.
- **CoffeeSetupMug**는 moderate-10/seen18(robocasa365) 등록 task. COAST의 "Coffee Mug"와 동일 →
  COAST 비교 일관성 좋음. 단일 instruction(mug 고정).
- env source robocasa365.

## 산출 후

instruction-pure 데이터로 within-instruction succ/fail LDA·conceptor를 다시 fit →
instruction confound 없는 깨끗한 pathway 분리력/steering 검증.
