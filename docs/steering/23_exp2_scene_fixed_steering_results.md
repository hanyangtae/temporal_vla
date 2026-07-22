# 23. exp2(구 pq2) — scene 고정·모델 random 조건의 activation steering 유의성 (최종 결과)

> 2026-07-14 작성. 단일 출처: 이 문서 + `outputs/eval/robocasa/groot_n15/steer_eval_exp2/aggregate_v2/`
> (arms.tsv·matrix.md·summary.json). 원장: `docs/collab/2026-07-10-steering-redesign-gate1.md`.
> 재설계 배경·결정: `docs/steering/17_steering_experiment_redesign.md`.

## 1. 실험 질문

**scene을 고정하고 diffusion noise(모델)만 random하게 둘 때, raw 대조 conceptor
activation steering이 Success Rate를 유의하게 올리는가?**

- 대상: GR00T N1.5 × RoboCasa, 두 task = PickPlaceCounterToCabinet(bread) ·
  PickPlaceCounterToStove(apple).
- steering: C_steer = C_success ∧ ¬C_failure, h' = h·Mᵀ, M = (1−β)I + βC_steer (COAST 계열,
  백본 재학습 없음).
- 직전 라운드가 3중 오염(α 오배선·apple 채점 오류·fit 표본 부족)으로 무효화된 뒤, 이를 전부
  교정한 공정 재실험.

## 2. 설계 (scene 고정 · 모델 random)

- **scene 고정**: cell = (task, EVAL_SEED=100000 고정) → 한 scene(layout/style/object/instruction
  동일). cell 내 변동은 **diffusion noise(inference seed = ep×1000)뿐**.
- 8 cell = bread 4 (base SR 0.35~0.83) + apple 4 (base SR 0.70~0.97).
- arm/cell: base · **위약(null)** · positive-only · {perm, gated} × {per_scene, cross_scene, grand}
  × fit{15,30} + denoise 진단 = 총 115 arm.
- **fit/선택/평가 완전 분리**: fit = ep0-59의 fit-half, P2 선택 = select-half, **본평가 = ep60-119
  (held-out, n=60/arm)**. fit↔eval episode 교집합 0 (아티팩트 실물 검증).
- 채점: apple = corrected 0.10 단일(0.07 병기), bread = 원판정. 사전 확정.
- **위약(null)**: scene×instruction 층 내 episode 라벨 permutation fit — "방향성 없는 개입"의
  대조군. steering 효과가 진짜면 위약을 이겨야 함.

## 3. 결과 — primary contrast (gated_per_scene_fit15 vs base, n=60)

성공/60 (괄호 = base 대비 Δ판, vs_null = 위약 대비).

| cell (task) | base SR | gated Δ | 위약(null) Δ | gated vs 위약 |
|---|---|---|---|---|
| bread84 (Cabinet) | 47/60 (0.78) | −2 | +2 | −4 |
| bread_s300028 | 50/60 (0.83) | +2 | +0 | +2 |
| bread_s300033 | 25/60 (0.42) | **+8** | **+8** | **+0** |
| bread_s400020 | 21/60 (0.35) | −1 | −1 | +0 |
| apple74 (Stove) | 42/60 (0.70) | −2 | +1 | −3 |
| apple_s100353 | 52/60 (0.87) | +1 | −3 | +4 |
| apple_s100395 | 48/60 (0.80) | +2 | +1 | +1 |
| apple_s100422 | 58/60 (0.97) | −2 | −3 | +1 |

- **net Δ>0 인 cell = 4/8** (s300028·s300033·s100353·s100395) → "어떤 scene에선 SR 상승"은 사실.
- **그러나 위약을 유의하게(>SE) 이긴 cell = 0/8.** 가장 큰 상승 s300033(+8)은 위약도 +8로 **완전
  동률**. SE(~6~9%p ≈ 4~5판) 안에서 vs_null 최대는 s100353 +4 하나뿐.
- pooled Δ = +6판 / 8 cell (cell당 평균 +1.3%p). 권고 기준 δ=+7.5%p(cell당 +4.5판) **명확 미달**.

## 3.1 apple 채점 기준(0.07 vs corrected 0.10) 불변성

apple은 pan 중심거리 임계를 원판정 0.07 → corrected 0.10으로 바꾼 이력이 있다(근거·도구:
`docs/steering/18_apple_success_rejudge.md`). D1 결정에 따라 eval은 corrected 0.10 단일 채점,
0.07은 병기(진단 전용)한다. **기준 변경이 exp2 steering 판정을 바꾸는지** 신규 apple 3 cell에서
직접 대조(같은 ep 매칭, 성공/60):

| cell | base@0.10 | base@0.07 | gated Δ (10 / 07) | perm Δ (10 / 07) |
|---|---|---|---|---|
| s100353 | 52 | 52 | +1 / +1 | 0 / 0 |
| s100395 | 48 | **39** | +2 / +3 | −1 / −3 |
| s100422 | 58 | 58 | −2 / −2 | −1 / −3 |

- 기준 변경은 **절대 base SR만 이동**시킨다. s100395가 0.07에선 39/60(−9판) — pan 중심 0.07~0.10
  구간 안착이 많은 discordant scene이라 그렇다(s100353·s100422는 discordant 0, 두 기준 동일).
- 그러나 **steering Δ는 두 기준에서 방향 동일·크기 ≤2판으로 robust**. "위약 미초과·scene 일관성
  없음" 판정은 어느 기준으로 채점해도 불변.
- 구 라운드도 동일 결론: 채점 오류는 해악의 **크기만 과장**했을 뿐 **부호를 바꾸지 못함**
  (문서 18 §6 — 최대 사례 s100104 절대 SR +36판 교정돼도 base 대비 −11 유지).
- 요컨대 apple 기준 변경의 실질 효과는 **절대 SR 재조정에 국한**되고 steering 인과 판정은 불변.

## 4. 핵심 판정 — "일관성 없음"을 넘어 "위약과 구별 안 됨"

사용자 요약("어떤 scene에서는 SR 상승하지만 일관성은 없다")은 맞다. 데이터는 여기서 한 단계 더
강한 진술을 지지한다:

1. **일관성 없음**: 부호가 cell마다 뒤섞임(+8 ~ −2), scene-일관 개선 방향이 없음.
2. **위약 미초과**: 상승한 cell조차 방향성 없는 위약(라벨 셔플)과 크기가 같거나 그 이내. 즉
   양성은 "성공 부분공간으로의 steering" 신호가 아니라 저SR cell의 내재 변동.
3. **cross_scene(핵심 축)**: s300028(+2~+5, SE 이내) 외 전 cell 0~음수 — scene 간 공유 방향 없음.
4. **positive-only 최악**(−16/−13): 대조 없는 성공-부분공간 투영은 오히려 유해.

## 5. flip 분석 — net 뒤에 숨은 대량 churn (scene 고정이라 측정 가능)

scene 고정 + ep-paired라서 base↔gated를 episode 단위로 매칭해 flip을 셀 수 있다 (net = F→S − S→F,
전 cell에서 aggregate Δ와 일치 검증).

| cell | 실패→성공 | 성공→실패 | net | 총 flip(/60) |
|---|---|---|---|---|
| bread84 | 6 | 8 | −2 | 14 |
| s300028 | 5 | 3 | +2 | 8 |
| **s300033** | **20** | **12** | **+8** | **32 (53%)** |
| s400020 | 9 | 10 | −1 | 19 |
| apple74 | 4 | 6 | −2 | 10 |
| s100353 | 6 | 5 | +1 | 11 |
| s100395 | 7 | 5 | +2 | 12 |
| s100422 | 0 | 2 | −2 | 2 |

- **작은 net 뒤에 양방향 flip이 대량**. s300033는 +8을 위해 20판 살리고 **12판을 죽임**(60판 중 53%
  뒤집힘). 위약도 같은 +8 → 이 churn은 "실패 구제"가 아니라 **결과 재추첨(re-roll)**.
- 순수 실패-구제였다면 성공→실패가 0에 가까워야 하나, **전 cell에서 2~12판씩 성공을 파괴**.
- s100422(천장 0.97)만 flip 2판 — 뒤집을 여지 자체가 없음.

## 6. COAST 대조 — 우리 조건과의 정합/불일치

| 항목 | COAST | 우리 |
|---|---|---|
| 평가 무대 | scene 매 episode 랜덤 재샘플 | **scene 1개 고정** |
| base↔steered pairing | 조건별 fresh rollout (paired 아님) | **ep-paired** |
| denoising 처리 | step별 벡터 개별 stack | K개 mean-pool |
| 토큰 풀링 | 49토큰 전체 | action 16토큰 |
| 보고 단위 | marginal SR (일부 표 best-of-sweep) | held-out 고정 config + flip |
| GR00T×RoboCasa 결과 | mean ΔGlob +0.16 | pooled +1.3%p (≈null) |

- **COAST는 성공→실패 flip을 논문 어디에도 보고하지 않음** — 그들의 랜덤-scene·비-paired 설계로는
  flip이 원천적으로 계산 불가(marginal rate만 가능). 우리 flip(§5)은 그들 방식으로는 안 보이는 정보.
- 우리가 고른 두 task는 COAST 자체에서도 Global 반응 최약체: **PP Cabinet(bread) ΔGlob +0.07(7 task
  중 공동 최하위), PP Stove(apple) +0.14(평균 +0.16 미만)**.
- base가 COAST regime(0.73)에 가깝고 헤드룸 있는 cell(apple74 0.70·s100395 0.80·bread84 0.78)에서조차
  COAST 예측(+8/+8/+4판)이 안 나옴 → 헤드룸만의 문제가 아니라 **우리 조건에서 효과 자체가 미검출**.
- 단 COAST(scene-랜덤 평균)와 우리(scene-고정 단일)는 측정 대상이 달라 "COAST가 틀렸다"까지는
  결론 못 냄. 확정 가능한 것 = **우리 조건에서 미재현**.

## 7. Confound 감사표

| # | 게이트 | 판정 | 근거 |
|---|---|---|---|
| 1 | 길이 | N/A | 개입 endpoint=SR(길이 아님). 단 fit은 COAST식 record-pool이라 null은 "COAST식 fit 한정" 스코프 각주 |
| 2 | task 정체성 | 통과 | cell 내 task/scene 고정, base와 동일 무대 |
| 3 | instruction 균형 | 통과 | cell당 instruction 1개 고정, base·arm 동일 |
| 4 | in-sample rescue | 통과(실물) | fit 29건 ep≤59 ∧ eval 117건 ep60-119, 교집합 0 |
| 5 | rollout pooling | 통과 | per-record fit, episode-mean 없음 |
| 6 | phase/dwell | N/A | SR endpoint. gated phase 라벨은 per-record |
| 7 | 관측≠인과 | 통과 | 본 실험 자체가 개입(base+위약 대조) |
| 8 | scene-국소≠일반 | 통과 | 국소 양성은 scene-국소로 기술, 일반화 주장 없음. 위약 대조로 국소 양성도 기각 |
| + | 위약 대조 | 통과 | 최대 양성(s300033 +8)이 위약 +8과 동률 → 방향성 신호 아님 |

**주장 강도: intervention effect** (n=60/arm, held-out, EVAL_SEED 고정, 위약 대조).

## 8. 결론

- **scene 고정·모델 random 조건에서 raw 대조 conceptor steering은 scene-일관 SR 개선을 내지 못한다.**
  일부 scene에서 net +상승이 관측되지만(4/8 cell), 방향성 없는 위약과 크기가 같거나 그 이내라
  steering 고유 효과로 볼 수 없다. flip 분석은 그 상승이 실패-구제가 아니라 양방향 대량 재추첨임을
  보인다(s300033: +8 = 20 구제 − 12 파괴).
- 직전 라운드의 "raw 대조 conceptor 종결" 판정이 공정 조건 + 위약 대조에서 **확증**됨.
- **다음 방향**: ① SAE로 scene(암기) feature 분리 후 conceptor(문서 14 메인 스트림), ② COAST 재현
  잔여 축(49토큰 풀링·step-stack fit·scene-랜덤 무대), ③ COAST 고반응 task(Open Drawer·Close
  Fridge 등)로 grid 이동 시 검출력 재확인.
