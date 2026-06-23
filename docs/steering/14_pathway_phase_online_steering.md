# 14 — Pathway-resolved + Phase-matched Online Steering (메인스트림)

> 이 문서가 **현재 연구 방향의 단일 출처**다. 세부 실험 기록(`06`/`09`/`10`/`11`..)은 시점
> 스냅샷이고, 방향·가설·open problem·검증 설계는 여기로 수렴한다. (확정 2026-06)

## 한 줄

실패하는 VLA를, **추론 중(online)에 어느 pathway(goal=VL / motor=DiT)·어느 phase에서
실패하는지 식별**하고, 그 시점에 맞는 **성공 활성화 분포로 steer**해서 — 백본 재학습 없이 — SR을 올린다.

## 가설

1. **Pathway 분리 steering**: VL(goal "what")과 DiT(motor "how")를 **각각 따로** steer.
   - 근거: NOTALL의 기능 분리(인과개입), 우리 Phase 3(VL 이른 신호 t≤8 / DiT 늦은 신호 t≥12, `09_`).
   - 주의: Eagle→VL-SA→DiT 는 **직렬** → "따로"가 진짜 독립이 아님. "VL만 / DiT만 / 둘 다"를 ablation으로 가른다.
2. **Phase-matched DiT steering**: DiT는 **rollout phase(시간)에 조건부로** steer.
   - 근거: 길이 confound(`01_`) — COAST는 전 timestep을 한 공간에 pool해 phase를 섞음. 두 실패 regime(초기조건형 / 실행표류형). NOTALL temporal ablation: DiT는 approach phase 지나면 expendable.

## ★ 중심 미해결 문제 (타당성 판정)

**추론 중에 어느 pathway가 실패했고(goal vs motor) 어느 task-phase인지 식별할 수 있는가?**

- 이게 안 되면 아무리 좋은 conceptor도 "어디로 / 언제 밀지"를 모른다 → 라우팅 불가.
- 따라서 다음 실험의 첫 질문은 "steer가 듣나"가 아니라 그 위의 **"phase / type을 online에 읽을 수 있나"**.
- 미점유 niche: internal-latent × online × failure-TYPE 구분 steer ([[notall-online-failuretype-niche]]). 경쟁자: Path-Deviation-Heads(2603.13782).

## 왜 이게 빈 자리인가 — 세 'step' 축과 선행연구 집계 방식

"inference step마다 점이 하나"인데, 사실 step에는 3축이 있다:
(a) action-token 위치(chunk 내 phase) (b) denoising step K (c) rollout env-step t.

| 방법 | rollout-time(t) 처리 | 함의 |
|---|---|---|
| **COAST** | 전 t를 클래스별 R=E[hhᵀ]에 **pool** (길이통제 없음; "per-step"은 denoising K뿐, rollout-phase 아님; action-token mean-pool) | 길이 confound 그대로; **rollout-phase 축이 비어 있음 ← 우리 자리** |
| **SAFE** | per-step **순차**(LSTM), 시간가변 threshold (max는 평가용 max-so-far) | 시간 1급; 탐지 전용 |
| **NOTALL** | per-episode 인과개입(pooled 분류 아님); action-token **per-token 유지** (단 GR00T VL-SA는 mean-pool 이득 EV 83~89%→99%) | 분석만; 개입 처방 아님 |

→ 우리 가설은 정확히 COAST가 비워둔 **pathway × rollout-phase** 두 축을 conceptor에 넣는 것.

## 검증 설계 — 사다리식 ablation

복잡도를 한 번에 올리면 약한 신호(0.6~0.7)에서 noise를 fit한다. 단계별로 ΔSR을 보고, **이전
단계가 신호를 보일 때만** 다음으로:

1. **(중단) COAST positive control**: faithful COAST N1.5 global steering 재현을 시도했으나 **실패**(ΔSR≈0, 논문 +0.16 재현 안 됨, **원인 미상**). 추가 시도 안 함. → 사다리는 우리 per-phase conceptor 의 pathway-split(2)부터 ΔSR 로 직접 검증한다.
2. **Pathway-split**: VL만 / DiT만 / 둘 다.
3. **+Phase-bin**: 절대 t-bin부터. 효과 보이면 progress / subtask로 격상.

- 표준: EVAL_SEED=100000, N_ENVS=2, N_EP=20, per-episode TSV. ([[eval-seed-standard]])
- confound: fixed-instruction 데이터로 VL 신호 진위 확보 (`11_instruction_confound`, `12_`).

## phase를 online에 어떻게 아나 (열린 설계)

- **절대 t-bin**: 싸지만 거칠다(길이가 달라 t의 의미가 다름).
- **progress-normalized(0~1)**: 의미는 맞지만 **online 계산 불가**(총길이 모름).
- **subtask phase(접근/파지/이송/배치)**: 최선이나 phase 검출기 필요.

→ 여기서 **접었던 VITA식 progress predictor가 보조 부품으로 부활** 가능 (메인 아님, online phase/progress 신호 공급원).

## pathway 매핑 (GR00T)

- **VL(goal)**: `action_head.vlln` (post-LN, D=2048, seq-mean-pool 이득). = NOTALL의 VL-SA bridge 쪽.
- **DiT(motor)**: `action_head.model.transformer_blocks[i]` D=1536 / 최종 pre-velocity D=1024 (per-token 필요).
- **Eagle-LM(goal "what", 12층)**: 아직 tap 안 함 — goal-type의 또 다른 후보. ([[pathway-resolved-steering]])

## 근거 문서

길이 confound·진짜 신호 `01_` / VL·DiT timing `09_` / instruction confound `11_instruction_confound` / 발표 정리 `13_`. (COAST 재현은 시도했으나 실패 — 원인 미상, 추가 안 함.)
