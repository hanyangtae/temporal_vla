# seen18 SAFE detector 검증 요약

작성: 2026-05-29 · 대상 run: `outputs/eval/robocasa/groot_n16/safe_seen18_4unseen_100ep`
입력 핸드오프: [`docs/seen18_steering_detector_handoff.md`](seen18_steering_detector_handoff.md) §4
참조: SAFE [arXiv 2506.09937](https://arxiv.org/abs/2506.09937), COAST [arXiv 2605.17144](https://arxiv.org/abs/2605.17144)

## TL;DR

seen18 GR00T(RoboCasa)에서 SAFE-LSTM을 풀 파이프라인(48 + 27런 sweep → finalize → CP)으로 재현. **SAFE 본래 공정 metric(task별 min rollout length T, ROC-AUC@s_T)** 기준 결과는

- **val_seen 0.683 / val_unseen 0.434** (≈ 우리 `at earliest stop`).

이는 SAFE Table 1의 SAFE-LSTM 평균 **0.823 / 0.770** 대비 **각각 14·34 점 낮습니다**. 즉 SAFE 자체 기준으로도 seen18 GR00T detector는 약하고, unseen은 chance 미만입니다.

대비로, 길이 비통제 변형(`by final end` max-so-far) AUROC는 val_seen 1.000 / val_unseen 0.992인데 length-only baseline(0.996)과 거의 같습니다 — **이 값은 SAFE 헤드라인이 아니라 §6.2 CP 분석용 변형**입니다 (앞서 "헤드라인 = 0.99 = 길이 아티팩트"로 통칭했던 표현은 부정확했습니다 — 본 문서에서 정정).

## 1. Protocol parity with SAFE

| 항목 | SAFE (LIBERO-10) | 본 재현 (seen18) |
|---|---|---|
| 모델 | OpenVLA / π0 / π0-FAST | GR00T N1.6 (robocasa365 ckpt120000) |
| 벤치 | LIBERO-10 (10 task) | RoboCasa target_atomic seen18 (18 task, 100 ep/task) |
| 분할 단위 | **task-level** seen/unseen | 동일 |
| unseen 비율 | 3/10 | 4/18 (CloseFridge·OpenCabinet·PickPlaceSinkToCounter·TurnOnMicrowave) |
| seen train/eval | 60/40 | 75/25 |
| 학습/하이퍼 선택 | D_train / D_eval-seen ROC-AUC | val_seen bal-acc (SAFE summarize 그대로) |
| 테스트 | D_eval-unseen (zero-shot) | val_unseen |
| **길이 통제** | task별 min rollout length T, s_T | 동일 (`task_min_step`, `at earliest stop`) |
| Feature | "last layer, before token logits / velocity field" | DiT(`action_head.model`) 출력 pre-velocity, `[K=4 denoise, H=16 horizon, D=1024]` |
| Aggregation | agg_token (AR) / agg_hori × agg_diff (π0) | `diff_idx_rel`/`horizon_idx_rel` ablation (48런) |
| Hparam | grid (lr × λ × seed) | grid 동일 (27런) |
| Threshold | split conformal + functional CP | 동일 (`split_conformal_binary`, 3 eval_time, α-sweep) |

차이는 seen 비율(75/25 vs 60/40)뿐. 그 외 절차는 SAFE를 그대로 따랐습니다.

## 2. 결과 (산출물 경로)

선정된 best: aggregation `horizon=mean, diff=1.0`, lr=3e-4, λ=1.0, seed2.

| 지표 | val_seen | val_unseen |
|---|---:|---:|
| `at earliest stop` (= SAFE Table 1 metric) | 0.683 | **0.434** |
| `by earliest stop` (max-so-far up to T) | 0.683 | 0.434 |
| `by final end` (max-so-far over full T) | 1.000 | 0.992 |
| length-only baseline (step-count) | 1.000 | 0.996 |
| Permutation null 95% band | [0.44, 0.56] | [0.44, 0.56] |

per-task val_unseen `by earliest stop`: task11 0.69·task16 0.64 (약·유의), task1 0.40·task5 0.32 (≤chance). 각 task의 length-AUROC=1.000 → 길이 confound가 완전.

산출물:
- `outputs/.../final_detector/{README.md, manifest.json, *_cp_eval.csv, per_rollout_scores.csv, length_control_verification.md, length_control_verification.png}`
- `experiments/{aggregation_ablation,hparam_sweep_hmean_d1_v2}/reports/seen18_*_summary.{json,md}`
- `outputs/.../visualizations/conformal_figure/by_final_end/`

## 3. SAFE Table 1과의 비교 (ROC-AUC × 100, min-length T)

| 모델 + 벤치 | SAFE-LSTM | SAFE-MLP |
|---|---:|---:|
| OpenVLA + LIBERO-10 | 70.2 / 72.5 | 72.7 / 73.5 |
| π0 + LIBERO-10 | 93.0 / 84.5 | 90.1 / 80.4 |
| π0-FAST + LIBERO-10 | 77.0 / 71.1 | 73.5 / 73.3 |
| π0 + SimplerEnv | 88.9 / 80.1 | 89.5 / 84.8 |
| **평균(전 벤치)** | **82.3 / 77.0** | **81.4 / 78.0** |
| **GR00T + RoboCasa seen18 (본 실험)** | **68.3 / 43.4** | — |

(Seen / Unseen)

seen 68은 SAFE의 가장 어려운 셀(OpenVLA+LIBERO 70)과 비슷한 수준. unseen 43은 SAFE의 어떤 셀(베이스라인 포함)보다도 낮음. 베이스라인(Mahalanobis/RND/LogpZO/STAC)조차 SAFE에서 평균 65–80을 찍습니다.

## 4. 길이 confound — SAFE는 직접 통제, COAST는 안 함 (정정 포인트)

SAFE §6.1 (verbatim):

> "…failed ones always have the maximum length, but successful ones are shorter… **if a failure detector simply learns to count the time elapsed, i.e., s_t = t, it will achieve perfect failure detection** since failed rollouts have a fixed and longer duration). To ensure a fair comparison, **for evaluation in Table 1, we compute the minimum rollout length for each task and use that as T for that task. The failure detection performance (in ROC-AUC) is then determined based on s_T**, where T is the same for all successful and failed rollouts within each task."

→ **SAFE의 헤드라인 Table 1은 길이 통제된 값**입니다. 우리 `at earliest stop`이 그것의 직접 대응. `by final end` 0.99는 SAFE가 §6.2 CP/detection-time 분석에서만 쓰는 비공정 변형.

**COAST**는 동일한 길이 confound를 fit에서 다루지 않습니다 — single layer ℓ의 action expert residual stream에서 action token mean-pool한 per-step 벡터를 쌓아 conceptor에 그대로 들어갑니다. 논문이 쓰는 "normalized trajectory time"은 v1(C_steer) projection 그림용 시각화이고, "matched-cost"는 추론 연산비용 매칭이지 길이 매칭이 아닙니다. (이 프로젝트의 향후 COAST 분석에선 동일한 길이 통제를 명시적으로 걸어야 합니다 — [[truncation-length-standard]] 참고.)

## 5. SAFE 논증과 우리 결과의 긴장

SAFE의 "왜 되는가" (§4.1, §B.1):

> "We **hypothesize** that these features also capture the high-level and abstract knowledge about task execution success/failure, by separating features from successful/failed rollouts into different regions."

> Fig 1 캡션: "When the VLA is failing, even though from different tasks, **the features fall in the same 'failure zone'**."

이 핵심 가설 — **task-agnostic shared failure zone** — 은 우리도 비순환 방식으로 재현했습니다 (memory `seen18-shared-failure-zone`: task-whiten centroid-spread fail/succ 비 p=1.0에 0.75). **공유 zone 가설은 seen18 GR00T에서도 부분 성립**합니다. 그런데 detector가 안 됩니다. 가설은 살아 있는데 detection으로는 못 가는 상황.

가능한 이유:
1. **Feature 출처의 표상공간 차이** (§6 참고 — 가장 유력한 후보).
2. **벤치 난이도**: LIBERO-10 < RoboCasa seen18.
3. **Onset 분포**: seen18 task의 절반은 frame0부터 유의·절반은 f10+에야 유의 ([[seen18-failure-onset-regimes]]). 초기조건형/표류형 혼재 + 4-task pooling이 unseen pooled를 깎음.

## 6. Architectural 가설: GR00T DiT vs π0/OpenVLA의 표상공간 차이

세 모델 모두 action 생성 시 VLM에 attend하지만 **구조가 다름** — code 확인 결과 (`src/policies/Isaac-GR00T/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:148-241`, `modules/dit.py:140-200`):

| 모델 | Action 생성기 | VLM 결합 방식 |
|---|---|---|
| OpenVLA | LLM 자체 | Action 토큰 = LLM 토큰. 전체 sequence joint self-attention. |
| π0 / π0-FAST | Action expert (PaliGemma MoE-sibling) | 공유 self-attention + MoE routing. VLM과 같은 attention pass. |
| GR00T n1.6 | 별도 DiT | **단방향 cross-attention** (Q=action/state, K/V=VLM Eagle features). VLM은 frozen 출력만. |

SAFE/우리가 뽑는 *pre-velocity* feature의 표상공간:
- π0: VLM과 **공유된** attention 공간 — SAFE 가설의 "VLM 추상 success/failure 지식"이 직접 살아 있음.
- GR00T: **별도 DiT의 action-국소 공간** — VLM 추상은 cross-attention으로 "한 번 조회"된 뒤 action 예측에 맞게 재투영됨. SAFE 가설이 약하게 전이될 가능성.

이게 같은 "마지막 층, velocity 직전"이라도 SAFE-LIBERO 0.93 vs 본 실험 0.43의 차이를 만드는 architectural 후보 1번입니다.

## 7. 다음 단계

- (a) **Eagle backbone(VLM) hidden state**에서 직접 feature 추출 hook → 같은 SAFE-LSTM 파이프라인. (OpenVLA/π0와 가까운 setup. 신호가 살아나면 가설1 지지.)
- (b) **DiT 다층 sweep**: `feature_server.py:42-46`의 multi-layer 캡처(`groot_n16_dit_block_residual_pooled_multilayer`) 활용. SAFE의 limitation("future work")이자 본 실험의 한 가지 변수.
- (c) **선택 metric 교정**: 현재 hparam/agg 선택이 길이 비통제 val_seen bal-acc에 좌우됨. 선택을 `at earliest stop` AUROC로 바꾸면 winner가 달라질 가능성. (방법론적 보강.)
- (d) **베이스라인 비교 부재**: SAFE 표의 Mahalanobis/RND/LogpZO/STAC를 seen18 feature에 돌려 LSTM과 같이 약한지 확인. 약하면 feature 자체 한계, 강하면 detector 아키텍처 한계.

본 검증의 결론은 핸드오프 §4의 "두 번째 detector 세션"에 대한 응답이며, 메인 방향 [[project-direction-latent-steering]]을 바꾸진 않습니다 (steering 쪽이 우선).
