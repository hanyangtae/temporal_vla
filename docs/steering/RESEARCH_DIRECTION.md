# 연구 방향 — VLA Latent Steering (단일 출처)

**방향·연구질문·가설·open problem·검증 설계의 단일 출처.** 라운드별 실측 결과는
[`RESULTS.md`](RESULTS.md), 개별 분석은 번호 문서. 충돌 시 이 문서가 방향을, RESULTS가 사실을 이긴다.

> 구 `14_pathway_phase_online_steering.md`(방향) + `15_research_structure.md`(RQ/가설 구조) 통합.
> 최종 갱신 2026-08-04 (RL2-VLA 선행연구 반영; 이전 2026-07-30 exp2~exp5 결과 반영).

## 0. 한 줄

VLA 백본 재학습 없이, **추론 중(online) 어느 pathway(goal=VL / motor=DiT)·어느 phase에서
실패하는지 식별**하고 그 시점에 맞는 성공 활성화 분포로 **steer**해서 SR을 올린다.

## 1. 연구 질문

**메인 RQ**: 추론 중 latent를 성공 쪽으로 조종해 SR을 올릴 수 있는가 — 그리고 **실패 유형에
맞춰 조종을 라우팅**하면 단일·전역 조종보다 나은가?

| RQ | 질문 | 현재 상태 |
|---|---|---|
| RQ1 (분리) | 길이·instruction·scene confound를 통제해도 succ/fail이 latent에서 분리되는가 | **부분 확립** — scene·길이·dwell·seed 통제 후 drawer 0.84·beer 0.62 |
| RQ2 (조종) | 분리 방향으로 write-in하면 인과적으로 SR이 오르는가 | **부정 우세** — §3 C2 |
| RQ3 (유형) | 실패는 VL-OOD / DiT-only-OOD로 *종류*가 갈리는가, 단일 심각도 축인가 | **측정 중** |
| RQ4 (라우팅) | 유형별로 조종 대상을 맞춰야 효과가 나는가 | **미검증, 핵심 기여** |

## 2. 선행 토대

| 논문 | 보인 것 | 빌리는 것 | 한계(우리가 메우는 곳) |
|---|---|---|---|
| **SAFE** | succ/fail이 feature-space에서 분리·검출 가능(per-step LSTM) | 분리 가능성 = 조종 가능성의 전제 | pathway 구분 없음, 유형 구분 안 함 |
| **COAST** | contrastive conceptor `C_steer=C_succ∧¬C_fail`로 조종 → SR↑ | 조종 연산자(multi-dim write-in) | 전 timestep pool → 길이·phase confound, pathway 미분리 |
| **NOTALL** | VL(goal "what")·DiT(motor "how") 기능 분리 | pathway 분해 근거 | online 아님, 실패 *유형*·phase-matched 조종 안 함 |
| **RL2-VLA** (arXiv 2607.26991) | **실패 감지 시에만 개입**(SAFE+CP 게이트)이 상시 개입을 인과적으로 이김(+8.9pp); 성공/실패 상태 분리 scaling law | "언제" 축의 외부 인과 검증 + 시변 CP threshold 부품 | 게이트가 binary — 유형(goal/motor)·phase 해상도 없음; 개입은 activation이 아니라 action-space(velocity 합성+verifier); per-VLA RL 학습·per-task rollout 수집 필요. 상세: [`../references/reading_notes/rl2_vla_adaptive_steering.md`](../references/reading_notes/rl2_vla_adaptive_steering.md) |

### 왜 빈 자리인가 — 세 'step' 축

"inference step마다 점이 하나"인데 step에는 3축이 있다:
(a) action-token 위치(chunk 내 phase) (b) denoising step K (c) **rollout env-step t**.

| 방법 | rollout-time(t) 처리 | 함의 |
|---|---|---|
| **COAST** | 전 t를 클래스별 `R=E[hhᵀ]`에 **pool** (길이통제 없음, "per-step"은 denoising K뿐) | 길이 confound 그대로, **rollout-phase 축이 비어 있음 ← 우리 자리** |
| **SAFE** | per-step **순차**(LSTM), 시간가변 threshold | 시간 1급, 탐지 전용 |
| **NOTALL** | per-episode 인과개입, action-token per-token 유지 | 분석만, 개입 처방 아님 |

미점유 niche = **내부 latent × online × 실패 TYPE(goal/motor) × phase-matched steer**.
경쟁자: Path-Deviation-Heads(arXiv 2603.13782). **RL2-VLA**(arXiv 2607.26991)가 이 중
"online 검출로 게이팅" 축을 선점(2026-07) — 단 binary 게이트·action-space 개입이라
TYPE·phase·activation write-in 세 칸은 여전히 비어 있음. 우리 기여 주장의 무게는
유형/phase 해상도 + 무학습·무verifier 단일-forward 개입 쪽에 둘 것.

## 3. 가설 체계

### C1 — 분리 (관찰) · *부분 확립*

succ/fail은 latent에서 분리된다.

- SAFE식 분리 재현. 길이와 직교하는 *실패 전 신호* 실재 — 고정-t within-task AUROC 0.6~0.7(약·다차원).
- 실패 onset 두 regime: **초기조건형**(frame0부터 유의, ~절반 task) vs **실행표류형**(f10+에야 유의)
  → 후자가 개입 여지 큼.
- scene·길이·dwell·seed 전부 통제한 scene-matched 조건에서도 분리 실재(drawer 0.84, z 4.4;
  **t=0에 이미 0.71** = 초기조건형).

**상시 confound**: 실패=항상 timeout이라 time-pooled 분리(AUROC 0.998)는 **길이 아티팩트**.
겉보기 분리의 상당수는 task 정체성·scene 암기로도 설명된다. 분리 주장은 길이·phase·scene 고정에서만.

### C2 — 조종 (인과) · **부정 우세 (2026-07-30 갱신)**

> ⚠ 이 항목은 2026-06-19에 "★최근 확립, mean ΔSR +0.114"로 적혀 있었다. **그 판정은 뒤집혔다.**
> 당시 근거였던 충실 스택 재현은 이후 재현되지 않았고(COAST 재현 실패, 원인 미상),
> exp2~exp5 네 라운드에서 위약을 넘는 개입 효과가 나오지 않았다. 상세: [`RESULTS.md`](RESULTS.md).

현재 판정: **read ≠ write.** 읽히는 방향으로 밀어도 SR이 안 오르거나 무너진다.

- raw 대조 conceptor: exp2(위약 동률) → exp3 fit15(6-Holm 전부 null) → exp3 fit30(위약 요동 ±5~6)
  → **3연속 확증**. COAST +0.16은 축을 전부 정렬해도 비재현.
- setpoint mean-diff(setM): exp5-3 within-scene에서 β=1.0 permanent −51판(구제 1 / 해악 52).
  용량을 낮추면 회복되지만 이득도 사라지고 위약을 못 넘는다.
- 오라클 상한: donor activation 통째 이식도 2.6%(p=0.50). 단 action-replay 대조는 15.6% —
  **개입으로 뒤집을 창 자체는 존재**한다.
- **유일한 위약-분리 양성**: exp5-2 섭동-유도 실패 회복(ppcc P1, setM DiT L10 β0.3,
  .50 vs 위약 .25, 6:0 p≈.03). n=24 탐색 지위.

C1→C2 다리가 아직 통과되지 않았다는 것이 현재 최대 장애다. RQ3/RQ4는 이 다리를 전제로 하므로,
**C2를 여는 것이 선결 과제**다(§5 열린 질문).

### C3 — 실패 유형 (관찰) · *측정 중*

실패는 VL-OOD냐 DiT-only-OOD냐로 종류가 갈린다.

- 근거(`pathway_step_attribution.py` 10 task): VL-only-OOD ~30% 흔함, **DiT-only ~2%로 거의 0**
  (OpenDrawer-right만).
- **판정(2026-07-29)**: 이 비대칭은 **DiT를 이른 창에서 과소측정**한 아티팩트다. 기존 score는
  t≤8 풀링이라 두 pathway를 같은 창에서 재지 않았다. → "DiT-only 없음"으로 결론 금지.
- **구조적 confound**: Eagle→VL-SA→DiT는 직렬. VL-OOD는 거의 항상 DiT도 OOD로 만든다 →
  진짜 질문은 "VL로 설명되는 것 *이상*의 DiT-OOD가 있나".

### C4 — 라우팅 (인과) · *미검증, 핵심 기여*

유형별로 조종 대상을 맞춰야(VL실패→VL steer, DiT실패→DiT steer) 효과가 난다.

- 근거(예비): causal online 검출 cross-task 일반화 — DiT block31 t_d=11 **AUROC 0.92**, length-fair.
- **⚠ VL 쪽은 근거 아님**: "VL 이른 t_d=5 약신호"는 per-instruction 재평가에서 **VL 상시 발화
  아티팩트**로 판정(FPR 1.0 ≈ 무작위). 실검출기는 DiT뿐이다.
- caveat: both==dit(스케일) 미분리, unseen holdout 2개 쉬움, LOO 필요.

## 4. Pathway를 나눠 보는 근거 (타이밍 아님)

> ⚠ 예전 근거 "VL 이른 t≤8 / DiT 늦은 t≥12"는 **반증됐다** — 차이 +0.013에 측정 창 불일치.
> pathway 간 감지 시점 차이는 근거 없음. 상세: [`08_pathway_separation_analysis.md`](08_pathway_separation_analysis.md) 상단.

나눠 보는 실제 근거는 **실패 원인·case에 따라 써야 할 pathway가 다르다**는 것이다.

- **goal-vs-motor task 분열**: goal-type(방향/목표 오인) task는 VL 우위, motor-type(정밀 조작)은
  DiT 우위, 일부 task(navigate/일부 PnP/sink)는 두 pathway 모두 선형 분리 안 됨.
- **VL을 써야 하는 대표 case**:
  - **카메라 섭동** — exp5-2 C1에서 DiT setM은 해악, VL 평균이동(`setpoint_vl`)이 정합(ppcc 3:0).
  - **wrong-grasp** — 재탐색 구간에서 VL activation이 확연히 갈린다([`22_wrong_grasp_vl_separation.md`](22_wrong_grasp_vl_separation.md)).
- **주의**: Eagle→VL-SA→DiT는 직렬이라 "따로"가 진짜 독립이 아니다. "VL만 / DiT만 / 둘 다"를
  ablation으로 가른다.

### pathway 매핑 (GR00T)

| pathway | tap | 차원 | 비고 |
|---|---|---|---|
| VL (goal) | `action_head.vlln` (post-LN) | D=2048 | seq-mean-pool 이득. NOTALL의 VL-SA bridge |
| DiT (motor) | `action_head.model.transformer_blocks[i]` | D=1536 (최종 pre-velocity 1024) | per-token 필요 |
| Eagle-LM (goal, 12층) | 미tap | — | goal-type의 또 다른 후보 |

## 5. ★ 중심 미해결 문제

### 5.1 online에 pathway·phase를 읽을 수 있는가

이게 안 되면 아무리 좋은 연산자도 "어디로 / 언제 밀지"를 모른다 → 라우팅 불가.
따라서 첫 질문은 "steer가 듣나"가 아니라 그 위의 **"phase / type을 online에 읽을 수 있나"**다.

phase를 online에 아는 방법 (열린 설계):

| 방법 | 장점 | 한계 |
|---|---|---|
| 절대 t-bin | 싸다 | 거칠다 — 길이가 달라 t의 의미가 다름 |
| progress-normalized(0~1) | 의미가 맞다 | **online 계산 불가**(총길이 모름) |
| subtask phase(접근/파지/이송/배치) | 최선 | phase 검출기 필요 |

→ 접었던 **VITA식 progress predictor가 보조 부품으로 부활 가능**(메인 아님, phase/progress 신호 공급원).

**"언제"를 푼 선행 사례 — RL2-VLA의 처리 방식** (상세: [reading note](../references/reading_notes/rl2_vla_adaptive_steering.md)):

- **검출**: SAFE-LSTM(action-expert latent 조건, per-timestep causal) — 단 task별 online
  rollout 300판 수집 필요, sim→real 일반화 실패로 실기 재수집.
- **threshold**: 시변 conformal prediction band를 성공 rollout으로 보정("성공이 확률 1−α로
  band 아래"), α는 balanced-accuracy 휴리스틱으로 task별 선택. **개입 빈도 자체가 튜닝
  대상**이라는 것을 보여줌.
- **인과 근거**: 성공/실패 상태를 분리한 test-time scaling law — 개입(다양성 주입)은 실패
  상태에서만 이득, 성공 상태에선 해악. adaptive vs always +8.9pp.
- **우리에게 함의**: (a) "감지 시에만 개입" 게이팅 축은 외부 인과 검증을 얻었다 — 우리
  online 검출기(DiT block31 AUROC 0.92)에 시변 CP threshold를 얹는 조합은 즉시 이식 가능한
  설계. (b) 단 RL2의 게이트는 binary "실패냐 아니냐"뿐 — **어느 pathway가·어느 phase에서**는
  묻지 않으므로 5.1의 문제는 그대로 남아 있다. (c) RL2의 개입은 best-of-N 후보 선별이라
  verifier가 오답을 걸러주지만, 우리 단일-forward write-in은 그 안전망이 없다 → 게이팅의
  정밀도(false positive 시 해악)가 우리 쪽에서 더 load-bearing.

### 5.2 C2를 여는 것 — 어떤 연산자인가

exp2~exp5가 닫은 것은 "raw 대조 conceptor + setM"이고, 열린 축은 아직 있다.

- **연산자**: WA-LQR 계열(diff-of-means + LQR) 재현 검토, 평균 연산자, 평균+분산 연산자(whitened
  mean-diff — [`24d_exp4-3_variance_aware_direction_input.md`](24d_exp4-3_variance_aware_direction_input.md)).
- **scene 성분 분리 후 steering**: SAE 경로([`31_`](31_sae_g1_results.md)·[`32_`](32_g2_scene_residual_results.md)).
  단 unseen scene 전이는 구조적 제약이 있다(32 §3).
- **phase 앵커 재정의**: 절대 t가 아니라 이벤트 기준.
- **개입 시점**: 오라클 t0로 상한을 먼저 재고, 검출기는 그 다음.
- **모델·task별 차이**: 분리되는 layer와 분리 양상이 모델/task마다 다른가. 다르면 어떻게 steer할 것인가.

## 6. 검증 설계

### 사다리식 ablation

복잡도를 한 번에 올리면 약한 신호(0.6~0.7)에서 noise를 fit한다. **이전 단계가 신호를 보일 때만** 다음으로:

1. ~~COAST positive control~~ — **중단**. faithful COAST N1.5 global steering 재현 실패
   (ΔSR≈0 vs 논문 +0.16, 원인 미상). 추가 시도 안 함.
2. **Pathway-split**: VL만 / DiT만 / 둘 다.
3. **+Phase-bin**: 절대 t-bin부터. 효과 보이면 progress / subtask로 격상.

### RQ3 (오프라인, CPU — remote-compute)

- **같은 창에서 둘 다 측정한다.** 예전 지침(VL=t≤8 / DiT=t≥11로 창을 나눠 재기)이 비대칭을
  만들어냈다 — C3 판정·`08_` 반증의 원인. 창을 나눠야 할 근거가 있으면 그 근거를 먼저 제시할 것.
- 2×2 contingency {VL-OOD?}×{DiT-OOD?} — VL-only와 **DiT-only 둘 다에 mass**가 있어야 유형론 성립.
  전부 `both`로 쏠리면 종류가 아니라 심각도.
- downstream 통제: 성공 분포에서 DiT-OOD를 VL-OOD로 회귀 → **잔차**가 큰 게 genuine DiT-only.
- instruction-skew 가드: within-instruction 또는 OpenDrawer(균형) 위주.

### RQ4 (GPU, RQ3 통과 후에만) — crossover

```
              steer VL    steer DiT
VL-OOD 실패      ΔSR ↑↑       ΔSR ~0
DiT-OOD 실패     ΔSR ~0       ΔSR ↑↑
```

증거는 평균 ΔSR이 아니라 **대각선(type×intervention 상호작용)**. 특이성 control: 미스매치
steer는 무효, 성공 에피소드 steer는 무해.

### 평가 표준

EVAL_SEED=100000 · N_ENVS=2 · N_EP=20 · per-episode TSV. **fit-seed와 eval-seed 분리 필수**
(held-out 아니면 in-sample rescue 아티팩트). 위약(라벨 순열) 대조 · paired McNemar.
검정력: 조건당 20판이면 MDE +0.43 — **null이 "효과 없음"이 아니라 "검출 불가"일 수 있다.**

## 7. 리스크 · Falsification

- **C2 falsify(진행 중)**: 연산자·시점·scene 분리를 다 바꿔도 위약을 못 넘으면 → latent write-in
  계열 종결, 방향 재설정.
- **RQ3 falsify**: 창 보정·downstream 잔차 후에도 DiT-only 칸이 ~0 → "VL/DiT 종류" 프레임 폐기,
  "VL-OOD 심각도" 단일 축으로 후퇴.
- **RQ4 falsify**: VL/DiT steer가 양 유형을 동등하게 구제 → 유형론이 real이어도 라우팅 무용.
- **상시 confound**: 길이(time-pooled 분리 금지) · instruction-skew(VL AUROC 부풀림) ·
  scene 암기 · 직렬 downstream 전파 · detector both==dit 스케일 · in-sample rescue.
