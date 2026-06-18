# 랩미팅 발표 아웃라인 — VLA Activation Steering (2026-06)

> **목적**: 지난 "실패 loop = 메모리 부재" 프레이밍을 → "실패를 미리 알고, 그 시점의 성공 분포로
> activation 을 steer 한다(학습 없이)"로 리프레이밍.
> **청중/길이**: 랩미팅, 약 20~30분, 18장 본문 + 부록 3장.
> **톤 규칙**: 분석 성과 중심. 단 **분석이 진행 중이므로 모든 정량 수치는 "현재 스냅샷, 갱신될 수
> 있음"** — 단정 표현 금지. confound·한계는 약점이 아니라 *엄밀성의 성과*로 제시.
> **수식**: 본문 최소, 부록 A 에 모음. (달러 LaTeX 미사용 — plain/unicode)
> **그림**: 경로는 repo 루트(`/home/dongkyu/pkt_ws/temporal_vla/`) 기준, 전부 실존 확인됨. 새로 그리지 않음.

핵심 메시지 한 줄:
> **"실패를 미리 안다면, 그 순간(timestep / subtask phase)에 맞는 성공 action 분포 쪽으로 activation 을 steer 하면 — 백본 재학습 없이 — SR 이 오르지 않을까?"**

---

## A. 도입 & 리프레이밍

### S1 — 타이틀
- **제목**: 실패를 미리 알고, 그 순간 성공 쪽으로 — VLA Activation Steering
- **부제**: 백본 학습 없이 Success Rate 올리기
- **발표 멘트**: "오늘은 지난번에 말씀드린 '실패 loop' 이야기가 어떻게 바뀌었는지, 그리고 왜 activation steering 으로 방향을 잡았는지 보여드리겠습니다."

### S2 — 지난 랩미팅 (then)
- **핵심 불릿**:
  - 관찰: VLA 가 실패하면 같은 행동을 반복하는 loop 에 빠진다.
  - 가설: 과거를 기억 못 해서다 — **메모리 부재** 때문일 것이다.
  - 함의: 메모리/RNN 을 붙여 loop 를 탈출시키자.
- **발표 멘트**: "지난번엔 실패의 원인을 메모리 부재로 보고, 거기서 해법을 찾으려 했습니다."

### S3 — 직접 까보니 (now): 프레이밍 전환
- **핵심 불릿**:
  - 실패 rollout 을 하나하나 분류해 보니 loop 는 여러 표면 유형 중 하나일 뿐: `stuck_loop / action_repetition / state_stagnation / gripper_oscillation / frozen`.
  - loop 그 자체로는 의미 있는 신호가 아니었고, **메모리 부재와 직접 연결할 근거도 약했다.**
  - 결정적 단서(다음 섹션에서 정량화): 실패는 거의 **100% 가 그냥 끝까지(timeout) 간 것** → "반복/loop" 은 "끝까지 갔다"의 부산물.
  - **질문을 바꾼다**: 실패를 어떻게 *막을까*? 실패할 걸 *미리 안다면* 조작으로 줄일 수 있나? — 그것도 **학습/finetuning 없이.**
  - 그 답으로 도달한 것이 **activation steering**.
- **발표자 노트(출처)**: 분류기 `scripts/analysis/analyze_loop_patterns.py`. 정량 근거는 S9.
- **발표 멘트**: "loop 는 원인이 아니라 증상이었습니다. 그래서 '왜 반복하나'가 아니라 '실패를 미리 알고 막을 수 있나'로 질문을 바꿨습니다."

---

## B. Activation / Steering 이란 (직관)

### S4 — Activation 이란
- **핵심 불릿**:
  - VLA 가 추론하는 도중 내부에 흐르는 **hidden state** (입력도 출력도 아닌 중간 표현).
  - GR00T 구조: **Eagle-VLM (무엇을) → VL-SA bridge → DiT action expert (어떻게)** — NOTALL 의 pathway 분해.
  - 우리가 들여다보는 두 지점(tap): **VL**(`action_head.vlln`, 2048차원, goal) / **DiT**(action-token residual, motor).
  - 캡처/주입 경로: `scripts/serve/safe_hooks.py`(capture) + `scripts/serve/steering_hooks.py`(steer), GR00T serve 는 ZMQ feature_server.
- **발표 멘트**: "Activation 은 모델이 행동을 만들기 직전의 내부 생각이라고 보시면 됩니다. 우리는 그걸 두 군데서 꺼내 봅니다 — '무엇을' 하려는지(VL)와 '어떻게' 움직일지(DiT)."

### S5 — Steering 이란
- **핵심 불릿**:
  - 추론 시 그 activation 을 **성공 쪽으로 미는 것** — 백본 가중치는 0 학습.
  - 직관: "성공엔 있고 실패엔 없는" 부분공간으로 **soft-project** 한다.
  - 적용은 한 줄: `h' = h · Mᵀ` (M = steering 행렬). 수식 디테일은 **부록 A**.
  - 단일 벡터를 빼는 게 아니라 **다차원 부분공간 연산자**(conceptor) — 왜 그런지는 S10 에서 데이터로.
- **그림**: `outputs/weekly_report/figures/2026-06-w2/coast_fig1_overview.png` (COAST 파이프라인 개념도)
- **발표 멘트**: "Steering 은 그 내부 생각을 성공한 사례들이 모여 있는 방향으로 살짝 미는 겁니다. 모델을 다시 학습시키지 않고, 추론할 때만 개입합니다."

---

## C. Related Works

### S6 — 관련 연구 (1장 표)
- **표** (각 1줄 + 우리와의 관계; PDF 전부 `docs/references/`):

  | 논문 | 한 줄 요약 | 우리와의 관계 |
  |---|---|---|
  | **COAST** (arXiv 2605.17144, 2026-05) | succ/fail action-expert latent 에 conceptor steering → 추론 시 SR↑ | **메인 메서드**. 동일 문제설정. π0.5 RoboCasa +0.16 / LIBERO-10 +0.37 / MetaWorld +0.25, GR00T N1.5 RoboCasa +0.16, Diffusion Policy +0.14, 실로봇 +0.40 |
  | **SAFE** (arXiv 2506.09937) | VLA 실패 탐지 (마지막 action-expert layer feature) | 탐지 파트 — 우리 SAFE-LSTM 재현·길이통제 |
  | **NOTALL** (Not All Features Are Created Equal, ICLR 2026) | VLM/expert pathway 를 인과개입으로 분해 (goal vs motor) | VL/DiT pathway-resolved steering 의 동기 |
  | **conceptors** (Jaeger 2014) | soft-projector Boolean algebra (NOT/AND/OR) | 수학적 기반 |
  | **CAA** (단일벡터 activation addition, LLM) | 한 방향을 더하/빼는 baseline | 우리 multi-dim 의 대조군 |
  | **Path-Deviation-Heads** (2603.13782) | 경로 이탈 탐지 head | 경쟁 method |
  | **I-FailSense** (2509.16072) / **VITA** (progress, 보류) | 실패 감지 / 진척도 예측 | 인접 연구 |

- **발표 멘트**: "메인은 COAST 입니다. 우리와 문제설정이 정확히 같고, 단일 벡터가 아니라 multi-dim 연산자라는 점도 같습니다. SAFE 는 탐지, NOTALL 은 pathway 분해를 줍니다."

### S7 — 우리의 niche
- **핵심 불릿**:
  - 미점유 영역: **internal-latent × online × failure-TYPE(goal vs motor) 구분 steering.**
  - COAST = steering 이지만 type 무관·time-pooled / SAFE = 탐지만 / NOTALL = 분석만(개입 *처방* 아님).
  - 우리 = **탐지 + type별 + online steer** 를 결합.
- **발표 멘트**: "탐지와 분석은 각각 있지만, '실패 *유형*에 맞춰 *그 시점*에 개입한다'는 조합은 아직 비어 있습니다."

---

## D. 현재 보고 있는 것 — 분석 성과 (핵심)

> 데이터 출처(공통): `outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/...`
> 및 `outputs/weekly_report/2026-06-w2.md`(N1.6 moderate-10).
> **주의 문구(슬라이드 하단 공통)**: "현재 스냅샷 — 분석 진행 중, 수치 갱신될 수 있음."

### S8 — 데이터셋
- **핵심 불릿**:
  - seen18: 18 task × 100 ep = **1800 rollouts**, GR00T N1.6 robocasa365 ckpt120000, 전체 SR 0.537.
  - 추가로 GR00T **N1.5 데이터는 내가 별도 수집** → faithful 재현용(다음 단계).
- **발표 멘트**: "분석은 1800개 rollout 을 직접 까본 데서 출발합니다."

### S9 — ★ 길이 confound (리프레이밍의 정량 근거)
- **핵심 불릿**:
  - 실패는 **100% 가 45 step timeout**, 성공은 평균 **17.7 ± 8.4 step** 에 조기 종료.
  - → **길이만으로 succ/fail 을 AUROC 0.998 로 구분**할 수 있다.
  - 시간-pooled(episode 평균) feature 분리도 AUROC 0.978 — 대부분 **길이 누수**.
  - 결론: "loop/반복" 은 그냥 끝까지 갔다는 표면. **순진한 succ/fail 분리는 아티팩트** → 길이 통제가 필수.
- **그림**: `…/visualizations/length_confound/length_confound_hmean_dmean.png`
- **발표자 노트(출처)**: `…/length_confound/length_confound_hmean_dmean.json` — `auroc_length_only=0.9984`, `succ_len_mean=17.68`, `fail_len_mean=45.0`, `frac_fail_at_maxlen=1.0`.
- **발표 멘트**: "여기가 핵심입니다. 실패가 특별해서가 아니라, 그냥 끝까지 간 것뿐입니다. 길이만 봐도 99.8%로 갈리니, 길이를 통제하지 않은 모든 분리는 가짜일 수 있습니다."

### S10 — ★ 길이 통제 후 '진짜' 실패 신호
- **핵심 불릿**:
  - 같은 timestep 에서 비교하면 분리력은 **AUROC ~0.60**, t=12 에서 **0.70** 까지.
  - 실패 방향은 **시간 방향과 거의 직교** (cos ≈ 0.04) → 단순 시간 인코딩 아님.
  - 1~2차원으론 약하고 **~20차원**은 돼야 신호가 잡힘 (20D AUROC 0.636).
  - → 실패 전 신호는 **실재하지만 약하고 다차원** ⇒ 단일 벡터 additive 가 아니라 **부분공간(conceptor) 연산자**가 맞다 (S5 의 정당화).
- **그림**: `…/visualizations/failure_direction/failure_direction.png`
- **발표자 노트(출처)**: `…/failure_direction/failure_direction.json` — `auroc_full_t0=0.624`, `cos_ufail_utime=0.0425`, `dim_auroc["20"]=0.636`, t=12 `auroc_single=0.702`.
- **발표 멘트**: "길이를 통제해도 실패 전 신호가 *남습니다*. 다만 약하고 여러 차원에 퍼져 있어서, 벡터 하나 빼는 방식이 아니라 부분공간을 다루는 conceptor 가 필요합니다."

### S11 — 공유 실패 zone (SAFE 재현)
- **핵심 불릿**:
  - 실패는 task 와 무관하게 **공통 영역으로 수렴**, 성공은 task 별로 흩어짐.
  - task-whitened centroid spread 비(fail/succ): 진척 0%→100% 에서 **0.98 → 0.75** (실패가 더 뭉침).
  - SAFE 논문의 "task-agnostic 실패 attractor" 주장과 일치.
- **그림**: `…/visualizations/evolution/failure_zone_centroid_spread.png` (+ `evolution/progress_mean/` 시계열)
- **발표 멘트**: "실패들은 서로 닮아가고, 성공들은 과제마다 다릅니다. 실패에는 공통의 '빠지는 골짜기'가 있다는 거죠."

### S12 — 실패는 예측 가능한가? (SAFE-LSTM)
- **핵심 불릿**:
  - 공정 metric(각 task 최소 길이 T 로 cap, 길이 누수 차단)으로 학습한 SAFE-LSTM:
    - **seen task 0.683 / unseen task 0.434 (≈ 무작위)**.
  - → seen 에선 신호가 있으나 **unseen 일반화는 아직** (정직한 한계).
  - 단서: 길이 통제를 안 하면 0.99 까지 나오지만 그건 길이를 맞춘 것일 뿐.
- **발표자 노트(출처)**: `docs/seen18_safe_detector_verification.md`, `outputs/eval/robocasa/groot_n16/safe_seen18_4unseen_100ep/final_detector/`.
- **발표 멘트**: "탐지가 본 적 있는 과제에선 어느 정도 되지만, 처음 보는 과제로의 일반화는 아직입니다. 여기는 솔직히 더 해야 합니다."

### S13 — 두 가지 실패 regime
- **핵심 불릿**:
  - **초기조건형**: frame 0 부터 유의 (결과가 시작부터 거의 결정) — OpenDrawer/OpenCabinet/SlideDishwasherRack 등 ~절반.
  - **실행표류형**: frame 10+ 에야 유의 (실행 중 어긋남) — TurnOffStove/TurnOnSinkFaucet/PnP-CounterToStove 등.
  - → **steering 으로 개입할 여지는 실행표류형이 더 크다** (개입할 시간 창이 있음).
- **그림**: `…/visualizations/auroc_tables/auroc_table_abs.png`
- **발표자 노트(출처)**: `docs/steering/01_seen18_latent_analysis.md` §3 (permutation test, n=200).
- **발표 멘트**: "실패가 시작부터 정해진 과제도 있고, 가다가 어긋나는 과제도 있습니다. 우리가 손댈 수 있는 건 후자입니다 — 그래서 '언제' 개입하느냐가 중요해집니다."

### S14 — 타이밍 & pathway (VL vs DiT)
- **핵심 불릿**:
  - 고정 timestep 분리력: VL(goal)은 **이른 시점(t≤8) 우위**, DiT(motor)는 **늦게(t≥12) 따라잡음**.

    | t (step) | VL (goal) | DiT (motor) |
    |---|---|---|
    | 4 | 0.677 | 0.648 |
    | 8 | 0.713 | 0.701 |
    | 12 | 0.745 | 0.752 |
    | 16 | 0.724 | 0.754 |

  - → online early steering 엔 **VL 의 개입 여지**가 크다(모터 commitment 이전).
  - **반드시 함께 말할 confound 경고**: 헤드라인이던 SlideDishwasherRack VL 0.93 은 instruction(in 93% / out 13%) 쏠림 **아티팩트 의심**. 신뢰 가능한 건 OpenDrawer(45/45 균형) DiT 0.888, CloseToasterOvenDoor(단일 instruction) VL 0.800 둘뿐. → **fixed-instruction 재수집으로 재검증 진행 중**.
- **그림**: `outputs/eval/robocasa/groot_n16/target_atomic_moderate10_pathway_pertoken_100ep/analysis/vl_dit_stepwise/SlideDishwasherRack/stepwise_trajectory.png`
- **발표자 노트(출처)**: `outputs/weekly_report/2026-06-w2.md` (결과1·2·instruction confound), `docs/steering/11_instruction_confound.md`.
- **발표 멘트**: "'무엇을'(VL)이 먼저 흔들리고 '어떻게'(DiT)가 나중에 흔들립니다. 단, 이 수치엔 instruction 쏠림이 섞여 있어서, confound 없는 데이터로 다시 보고 있는 중입니다 — 단정은 하지 않겠습니다."

---

## E. 가설 & 다음

### S15 — 현재 steering 시도 상태 (분석 성과 + 진행중 단서)
- **핵심 불릿**:
  - COAST 메서드·인프라 구축 완료: 32 layer 전수 sweep + SR eval.
  - **우리 N1.6 DiT-only 변형은 ΔSR ≤ 0** (개선 없음): conceptor quota 0.07~0.12 (COAST 0.3~0.5 의 ~1/3), 중간층 peak 부재.
  - 부정된 가설: token-pooling(49 vs 16) / layer 선택 / task scope / baseline noise — 전부 통제 후에도 음수.
  - **남은 가설**: N1.6(32L AlternateVLDiT, cross-attn) vs **N1.5(16L plain DiT)** architecture 차이 → 내가 수집한 N1.5 데이터로 **faithful 재현 진행 중** (COAST published N1.5 는 +0.16 기대).
  - ※ **아직 진행 중 — 새 결과가 오면 이 슬라이드는 바뀝니다.**
- **그림**: `outputs/weekly_report/figures/2026-06-w2/fig2_quota_sweep.png`
- **발표자 노트(출처)**: `docs/steering/06_coast_groot_n16_summary.md`, `04_coast_reproduction_map.md`.
- **발표 멘트**: "솔직히 우리 변형으론 아직 효과가 안 났습니다. 다만 차이 나는 지점들을 하나씩 지웠고, 남은 건 architecture 차이라 N1.5 로 충실 재현을 돌리는 중입니다."

### S16 — ★ 핵심 가설 (punchline)
- **핵심 불릿**:
  - **"실패를 미리 안다면, 그 시점(timestep / subtask phase)에 맞는 *성공 action 분포*로 steer 하면 SR 이 오를 것."**
  - 지금까지의 steering 은 **global / time-pooled** — 모든 순간에 같은 방향.
  - 그런데 우리 분석은 **타이밍이 중요**(S13 두 regime, S14 VL 이른/DiT 늦은)하다고 말한다 → **phase-matched steering** 의 근거.
  - 인프라도 준비됨: truncated window W = 성공 길이 [mean, mean+1σ], per-T pathway capture, fixed-instruction 수집.
- **발표 멘트**: "그래서 제 가설은 이겁니다 — 실패를 미리 알고, 그 *순간에 맞는* 성공 분포로 밀면, 지금의 뭉뚱그린 steering 보다 잘 오를 것이다. 분석이 가리키는 방향이 정확히 '언제'입니다."

### S17 — Next steps
- **핵심 불릿**:
  1. faithful **N1.5 재현**으로 baseline 효과(positive control) 확인.
  2. **fixed-instruction** 으로 confound 제거 후 VL/DiT 분리력 재검증.
  3. **phase-matched steering** 설계 → ΔSR **인과 재측정** (EVAL_SEED=100000 표준).
  4. 더 단순한 환경(**LIBERO / robosuite**)에서 교차검증.
- **발표 멘트**: "다음은 충실 재현으로 효과를 확인하고, confound 를 걷어낸 뒤, 시점에 맞춘 steering 으로 SR 변화를 인과적으로 재보는 겁니다."

---

## F. 한계 + 부록

### S18 — 한계 / 주의
- **핵심 불릿**:
  - 실패 전 신호는 **약하다**(0.6~0.7).
  - confound 가 도처에: **길이**, **instruction**.
  - unseen 일반화 미확인, steering positive **아직 미확인**.
  - → **단정 금지. 분석 진행 중이라 수치는 갱신될 수 있음.**
- **발표 멘트**: "정리하면 — 방향은 잡혔지만 신호는 약하고, 아직 증명된 단계는 아닙니다. 진행 중인 결과라는 점을 강조드립니다."

### 부록 A — conceptor 수식 (백업)
- 상관행렬: `R = E[h hᵀ]`
- conceptor: `C = R (R + α⁻²I)⁻¹` (고유값 λ = σ/(σ + α⁻²) ∈ [0,1) — 데이터 많은 방향은 통과, 아니면 죽임)
- Boolean algebra (Jaeger 2014): `¬C = I − C`, `C₁ ∧ C₂ = (C₁⁻¹ + C₂⁻¹ − I)⁻¹`
- contrastive: `C_steer = C_success ∧ ¬C_failure` ("성공엔 있고 실패엔 없는" 부분공간)
- 적용: `M = (1 − β) I + β · C_steer`, `h' = h · Mᵀ` (β=0 무개입, β=1 완전 사영)
- quota: `q(C) = tr(C) / d` (부분공간 크기 비율)
- 구현: `src/conceptor/core.py`, `src/conceptor/steering.py`
- **그림**: `outputs/weekly_report/figures/2026-06-w2/coast_fig3_lowrank.png` (contrastive subspace 가 low-rank)

### 부록 B — 실험 표준
- EVAL_SEED = 100000 고정, N_ENVS = 2, N_EP = 20 / condition.
- per-episode TSV 로깅 (success + instruction).
- GR00T serve = ZMQ feature_server (`action_head.model` hook), capture/steer = `safe_hooks.py` / `steering_hooks.py`.

### 부록 C — 수치 출처 매핑
| 슬라이드 | 수치 | 출처 파일 |
|---|---|---|
| S9 | length AUROC 0.998, len 17.7/45 | `…/length_confound/length_confound_hmean_dmean.json` |
| S10 | t0 AUROC 0.624, cos 0.0425, 20D 0.636 | `…/failure_direction/failure_direction.json` |
| S12 | SAFE-LSTM 0.683 / 0.434 | `docs/seen18_safe_detector_verification.md` |
| S13 | 두 regime onset | `docs/steering/01_seen18_latent_analysis.md` §3 |
| S14 | VL/DiT AUROC, instruction confound | `outputs/weekly_report/2026-06-w2.md`, `docs/steering/11_instruction_confound.md` |
| S15 | quota 0.07~0.12, ΔSR ≤ 0 | `outputs/weekly_report/2026-06-w2.md`, `docs/steering/06_coast_groot_n16_summary.md` |

---

## 슬라이드 빌드 메모
- 그림 경로 13개 전부 실존 확인됨(2026-06-18). 빌드 시 `ls` 재확인 후 임베드, 없으면 표로 대체(추정 경로 금지).
- 본문 산문 재사용 1순위: `outputs/weekly_report/2026-06-w2.md` (related-works·N1.6 재현·confound 가 figure 경로·판정까지 정리됨).
- 서사 보강: `docs/steering/01_seen18_latent_analysis.md`, `docs/steering/README.md`.
