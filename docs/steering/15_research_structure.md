# VLA Latent Steering: Pathway·실패유형 기반 조종 (연구 구조)

> 작성 2026-06-19. 선행 토대(NOTALL/COAST/SAFE)와 현재까지의 측정·재현 결과를 연구
> 질문/가설/근거/검증 설계로 구조화한 문서. 메인 method·단일 출처는
> [`14_pathway_phase_online_steering.md`](14_pathway_phase_online_steering.md).

## 0. 한 줄 요약

성공/실패가 latent에서 분리된다면(SAFE) 성공 쪽으로 조종할 수 있고(COAST), **실패가
VL(goal)에서 OOD인지 DiT(motor)에서만 OOD인지에 따라 종류가 다르며, 그 종류에 따라 조종
대상도 달라야 한다**는 것이 핵심 주장.

---

## 1. 연구 질문

**메인 RQ**: VLA 백본 재학습 없이, 추론 중 latent를 성공 쪽으로 조종해 SR을 올릴 수 있는가
— 그리고 **실패 유형(goal vs motor)에 맞춰 조종을 라우팅**하면 단일·전역 조종보다 나은가?

**하위 RQ**
- RQ1 (분리): 길이·instruction confound를 통제했을 때도 succ/fail이 latent에서 분리되는가?
- RQ2 (조종): 분리 방향으로 write-in하면 인과적으로 SR이 오르는가?
- RQ3 (유형): 실패는 VL-OOD / DiT-only-OOD로 *종류*가 갈리는가, 아니면 단일 심각도 축인가?
- RQ4 (라우팅): 유형별로 조종 대상(VL vs DiT)을 맞춰야 효과가 나는가?

---

## 2. 배경 — 선행 토대 3편

| 논문 | 보인 것 | 우리가 빌리는 것 | 한계(우리가 메우는 곳) |
|---|---|---|---|
| **SAFE** | succ/fail이 feature-space에서 분리·검출 가능(per-step LSTM) | 분리 가능성 = 조종 가능성의 전제 | pathway 구분 없음; 길이 confound는 직접 통제하나 유형 구분 안 함 |
| **COAST** | contrastive conceptor `C_steer=C_succ∧¬C_fail`로 조종 → SR↑ | 조종 연산자(multi-dim write-in) | 전 timestep pool → 길이·phase confound; pathway 미분리 |
| **NOTALL** | VL(goal "what")·DiT(motor "how") 기능 분리 | pathway 분해 근거 | online 아님; 실패 *유형*·phase-matched 조종 안 함 |

---

## 3. 가설 체계

### C1 — 분리 (관찰) · *부분 확립*
**진술**: succ/fail은 latent에서 분리된다.
**근거(우리 측정)**:
- SAFE식 분리 재현됨. 길이 무관(시간방향과 직교)한 *실패 전 신호* 실재 — 고정-t within-task
  AUROC 0.6~0.7 (약·다차원).
- 실패는 task-무관하게 공유 zone으로 수렴, 성공은 task별 분산.
- 실패 onset 두 regime: **초기조건형**(frame0부터 유의, ~절반 task) vs **실행표류형**(f10+에야
  유의) → 후자가 개입 여지 큼.

**confound(필수 통제)**: 실패=항상 timeout이라 time-pooled 분리(AUROC 0.998)는 **길이
아티팩트**. 분리 주장은 길이/phase 고정에서만.

### C2 — 조종 (인과) · ★최근 확립 (2026-06-19)
**진술**: 분리 방향으로 write-in하면 SR이 오른다.
**근거**: 충실 스택(native ZMQ + 공식 env + pretrain + replan5)에서 **mean ΔSR +0.114**
(COAST 보고 +0.16과 같은 방향). 초기의 −0.11/−0.079는 method 실패가 아니라 **eval
operating-point 4축 아티팩트**였음. n=30으로 확정 중.
**의의**: C1→C2 다리(read-out ≠ write-in)가 경험적으로 통과. 이로써 RQ3/RQ4가
무의미해지지 않음(gate 통과).

### C3 — 실패 유형 (관찰) · *측정 중, 비대칭이 아티팩트 의심*
**진술**: 실패는 VL-OOD냐 DiT-only-OOD냐로 종류가 갈린다.
**근거(우리 측정, `pathway_step_attribution.py` 10task)**: VL-only-OOD ~30% 흔함,
**DiT-only ~2%로 거의 0** (OpenDrawer-right만 존재).
**경고(비단정)**: 이 비대칭은 **DiT를 이른 창(t≤8)에서 과소측정**한 결과일 수 있음(DiT
신호는 늦음, block31 t≥11). 기존 score는 t≤8 풀링. → 창 보정 전엔 "DiT-only 없음"으로
결론 금지.
**구조적 confound**: 직렬 Eagle→VL-SA→DiT. VL-OOD는 거의 항상 DiT도 OOD로 만듦 → 진짜
질문은 "VL로 설명되는 것 *이상*의 DiT-OOD가 있나".

### C4 — 라우팅 (인과) · *미검증, 핵심 기여*
**진술**: 유형별로 조종 대상을 맞춰야(VL실패→VL steer, DiT실패→DiT steer) 효과가 난다.
**근거(예비)**: causal online 검출 cross-task 일반화 — DiT block31 t_d=11 **AUROC 0.92**,
VL 이른 t_d=5 약신호. length-fair.
**caveat**: both==dit(스케일) 미분리, unseen holdout 2개 쉬움, LOO 필요.

---

## 4. 검증 설계

**RQ3 (오프라인, 지금 가능 — remote-compute)**
- pathway별 신호 창에 맞춰 측정: VL=t≤8, DiT=block31 t≥11 (같은 t에서 둘 다 재지 말 것).
- 2×2 contingency {VL-OOD?}×{DiT-OOD?} — VL-only와 **DiT-only 둘 다에 mass**가 있어야
  유형론 성립. 전부 `both`로 쏠리면 종류가 아니라 심각도.
- downstream 통제: 성공 분포에서 DiT-OOD를 VL-OOD로 회귀 → **잔차**가 큰 게 genuine
  DiT-only. (강한 버전은 VL을 success로 rescue 후 DiT OOD 잔존 여부 — GPU.)
- instruction-skew 가드: within-instruction 또는 OpenDrawer(균형) 위주, SlideDishwasher(쏠림)
  분리.

**RQ4 (GPU, 로컬, RQ3 통과 후에만) — crossover**

```
              steer VL    steer DiT
VL-OOD 실패      ΔSR ↑↑       ΔSR ~0
DiT-OOD 실패     ΔSR ~0       ΔSR ↑↑
```

증거 = 평균 ΔSR 아니라 **대각선(type×intervention 상호작용)**. 특이성 control: 미스매치
steer는 무효, 성공 에피소드 steer는 무해.

**검증 순서(사다리 — 약신호 noise-fit 방지)**: C2 gate(통과) → C3 유형 실재 → C3 통과 시에만
C4. 각 단계 ΔSR 비교(global → pathway-split → +phase-bin).

---

## 5. 기여 / 미점유 niche

**내부 latent × online × 실패 TYPE(goal/motor) × phase-matched steer** — NOTALL 저자(CWRU)도
미점유. 경쟁자 Path-Deviation-Heads(arXiv 2603.13782)와의 델타는 별도 1문단으로 정리 필요.

---

## 6. 리스크 · Falsification 조건

- **RQ3 falsify**: 창 보정·downstream 잔차 후에도 DiT-only 칸이 ~0 → "VL/DiT 종류" 프레임
  폐기, "VL-OOD 심각도" 단일 축으로 후퇴.
- **RQ4 falsify**: VL/DiT steer가 양 유형을 동등하게 구제 → 유형론이 real이어도 라우팅 무용.
- **상시 confound**: 길이(time-pooled 분리 금지), instruction-skew(VL AUROC 부풀림), 직렬
  downstream 전파, detector both==dit 스케일.
