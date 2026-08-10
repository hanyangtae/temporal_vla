# D. VLA/정책 "실패 직전 순간" 검출·개입시점 문헌 조사 (2026-08-10)

**질문**: "정책이 실패하려는 순간을 언제 알아채고 언제 개입하는가" 축의 최신(2024~2026, 특히 2026) 연구.
5개 하위축(불확실성·OOD / 개입시점·검출지연 / 시간축 구조 / 실패유형 구분 / 검출 후 대응)으로 조사.

**검증 기준**: `[abs]` = WebFetch로 arXiv abstract 페이지 실물 확인. `[full]` = WebFetch로 arXiv HTML 본문
실물 확인(더 깊은 수치 포함). `[검색만]` = WebSearch 스니펫에서만 등장, 실물 미확인 — §10에 별도 격리,
본문 서술에서 사실로 인용하지 않음. 모든 arXiv ID·저자·venue는 fetch 시점(2026-08-10) 기준.

## 0. 총평 (먼저 읽을 것)

1. **"내부표현/불확실성으로 VLA 실패를 감지할 수 있는가" 축은 사실상 포화**. 2025-06 SAFE(NeurIPS'25)를
   기점으로 2026-04~08 사이에만 15편 이상의 후속·경쟁 논문이 쏟아짐(SAFECAST는 2026-08-04 제출 —
   조사 시점 6일 전). 신호원만 다른 변주가 대부분(마지막 layer feature, LSTM, Mahalanobis, conformal
   quantile regression, normalizing flow, action-chunk entropy, velocity-field ensemble disagreement,
   activation perturbation disagreement …) — **이 하위질문 자체는 "이미 풀렸다" 쪽에 가깝다.**
2. **"phase(조작 단계)별로 검출기를 달리 두거나 phase별 성능차를 정량 보고"한 논문은 사실상 0에 가깝다.**
   SAFE 논문 본문에 직접 확인한 바 "No explicit phase-based performance breakdown" — 저자들 스스로
   생략. ActProbe가 유일하게 rollout 진행률 2점(25%/50%)을 비교하지만 이는 "조기검출 성능"이지
   reach/grasp/transport/place 같은 **조작 subtask phase 조건부 검출기**가 아니다. 가장 근접한 두 건도
   각각 좁은 도메인(직물 조작 단일 task) 또는 coarse 2분류(planning/execution)에 그친다 (§8-Q2).
   → **이 프로젝트의 phase-matched 검출/개입 축은 외부 문헌에서 아직 점유되지 않았다.**
3. **"goal(목표오인) vs motor(실행오류)" 그대로의 이분법은 없으나 근사물은 있다** — Guardian의
   planning/execution 5+4+2 세부 taxonomy, Safe Embodied AI의 semantic-misgrounding/execution-drift
   4분류. 단 **전부 사후(post-hoc) VLM 판정**(완료된 rollout 영상을 다시 보고 채점)이며, **온라인
   내부활성 기반으로 실시간에 goal-type인지 motor-type인지 분기하는 논문은 확인 못함** — 이 프로젝트가
   이미 파악한 "NOTALL online failure-type niche" 미점유 상태가 유지된다.
4. **"늦게 감지하면 개입 못 한다"는 문제의식은 명시적으로 다수가 다룬다** — AEGIS는 제목부터
   "calling a stronger policy before long-horizon failures **compound**", Doomed from the Start(인접
   도메인)는 "compute가 아직 남아있는 라운드에 내부상태가 이미 표면행동보다 앞서 안다"를 정면으로
   정량화. 그러나 **정량 delay(스텝/초 단위)를 abstract 수준에서 명시하는 논문은 소수** —
   REPAIR-Bench(초 단위), Early Warning Signals(15-step horizon, layer별 AUROC), AEGIS(첫 30%),
   ActProbe(F1-timeliness Pareto) 정도. FIPER·FAIL-Detect처럼 "earlier/faster than baseline"이라고만
   말하고 구체 수치를 본문 표로 미루는 경우가 더 흔하다.
5. **검출 후 대응은 재시도(FAR)·정책전환(AEGIS)·인간호출(Human-in-Loop)·VLM reasoning 기반 복구
   행동생성(FailSafe)·behavior-tree 스킬 보정(Unified Framework)으로 갈라지며, activation-level steering
   개입으로 직결하는 논문은 없다** — 이 프로젝트의 conceptor steering 축과 겹치지 않고 보완적이다.
6. 이 프로젝트가 이미 깊이 다룬 두 문헌은 교차참조만 하고 재조사하지 않음: **SAFE**(2506.09937,
   `SAFE-LSTM`으로 이미 재구현·재학습되어 fair-metric 하 unseen ~chance 확인됨 — memory
   `seen18-safe-detector-verified`)와 **RL2-VLA**(2607.26991, SAFE+conformal 게이트로 "언제 개입"을
   해결한 직접 선행연구 — `docs/references/reading_notes/rl2_vla_adaptive_steering.md`). 상세는 §9.

---

## 1. 전체 목록 (실물확인 32편, 확인일 2026-08-10)

| 약칭 | arXiv ID (제출일) | Venue | 1차 소속(확인분만) | 주축 |
|---|---|---|---|---|
| SAFE | 2506.09937 (25-06-11) `[full]` | NeurIPS 2025 | UofT/Vector Institute/Toyota Research Institute | 1,2,3 |
| FIPER | 2510.09459 (25-10-10) `[abs]` | NeurIPS 2025 | TUM (Schoellig lab) | 1,2,3 |
| FAIL-Detect | 2503.08558 (25-03-11) `[abs]` | RSS 2025 | (미확인) | 1,3 |
| UNISafe | 2505.00779 (25-05-01) `[abs]` | CoRL 2025 | CMU (Bajcsy lab) | 1,3 |
| Early Warning Signals for OpenVLA | 2606.29699 (26-06-29) `[abs]` | arXiv preprint | (미확인) | 1,2 |
| VLAConf | 2605.29605 (26-05-28) `[abs]` | arXiv preprint | (미확인, 8인) | 1 |
| VLA-FAIL | 2606.21386 (26-06-19) `[full]` | arXiv preprint (cs.LG) | (미확인) | 1,2 |
| Perturbation-Based Uncertainty | 2606.20754 (26-06-18) `[abs]` | arXiv preprint | (미확인, 2인) | 1 |
| Uncertainty Quantification for Flow-Based VLA (SAVE) | 2606.18043 (26-06-16) `[abs]` | arXiv preprint | TUM (Schoellig lab) | 1 |
| ReconVLA | 2604.16677 (26-04-17) `[abs]` | arXiv preprint | (미확인, 3인) | 1 |
| RC-NF | 2603.11106 (26-03-11) `[abs]` | CVPR 2026 | (미확인) | 1,4 |
| PATCH | 2606.16690 (26-06-15) `[abs]` | arXiv preprint | (미확인) | 1 |
| How VLAs Fail Differently | 2605.28726 (26-05-27) `[full]` | ICRA'26 워크숍 | 단독저자·소속불명 ⚠ | 3,4 |
| Diagnostic Runtime Monitoring w/ Martingales | 2407.21748 (24-07-31) `[abs]` | arXiv (cs.RO) | (Pavone 관여 추정) | 3,4 |
| Hide-and-Seek in Trajectories | 2605.30834 (26-05-29) `[abs]` | arXiv preprint | (Sharon Li 관여 추정) | 1,3 |
| Foresight | 2606.23085 (26-06-22) `[abs]` | arXiv preprint | (Jenkins 관여 추정) | 1,3 |
| SAFECAST | 2608.04246 (26-08-04) `[abs]` | arXiv preprint | (Thomason 관여 추정) | 1,3 |
| AEGIS | 2606.06660 (26-06-04) `[abs]` | arXiv preprint | 단독저자·소속불명 ⚠ | 2,5 |
| ActProbe | 2606.08508 (26-06-07) `[full]` | arXiv preprint | Tsinghua AIR / UESTC / Nanjing | 1,2,4 |
| Phase-Conditioned Imitation Learning(직물) | 2605.29407 (26-05-28) `[abs]` | IEEE/ASME T-Mechatronics(accepted) | Tohoku Univ | 2,3(좁음) |
| Safe Embodied AI Cross-layer | 2606.05660 (26-06-04) `[abs]` | arXiv preprint | Seoul Natl Univ(Sungroh Yoon) 추정 | 4 |
| How Visible Are Silent Manipulation Failures | 2606.03134 (26-06-02) `[abs]` | arXiv preprint | UC Berkeley | 4 |
| Guardian(Scaling Cross-Env Failure Reasoning) | 2512.01946 (25-12-01, v3 26-03-30) `[full]` | arXiv preprint | INRIA(WILLOW 추정) | 4 |
| AHA | 2410.00371 (24-10-01) `[abs]` | arXiv (cs.RO) | NVIDIA/UW | 4(배경) |
| FAR | 2607.01111 (26-07-01) `[abs]` | arXiv preprint | (CMU 추정) | 5 |
| Human-in-the-Loop Confidence-Aware Recovery | 2602.10289 (26-02-?) `[abs]` | HRI 2026 | Cornell/Princeton | 2,5 |
| FailSafe | 2510.01642 (25-10-02) `[abs]` | IROS 2026 | (UW/NTU 혼합 추정) | 5 |
| REPAIR-Bench | 2606.29937 (26-06-29) `[abs]` | arXiv preprint | (Angelique Taylor 관여 추정) | 2,5 |
| Unified Framework (VLM+BT) | 2503.15202 (25-03-19) `[abs]` | arXiv preprint | (미확인) | 5 |
| SC-VLA (Fast/Slow) | 2405.17418 (24-05-27, v2 25-03-19) `[abs]` | arXiv (cs.CV) | (미확인, 11인) | 2,5(배경) |
| Early Failure Detection — Surgical Soft-Tissue | 2501.10561 (25-01-17) `[abs]` | RSS'25 OOD 워크숍 | Univ. of Utah | 2(인접도메인) |
| Doomed from the Start | 2607.06503 (26-07-07) `[abs]` | arXiv preprint | (미확인) | 2,3(★인접도메인, LLM agent) |

⚠ = 단독저자·소속 불명, 워크숍/preprint뿐 — 증거 등급 낮음(WA-LQR 판례와 동일 취급, 인용 시 명시할 것).

---

## 2. 축1 — 불확실성 추정·OOD 검출 (내부표현 우선)

| 논문 | 신호원 | 핵심 수치 |
|---|---|---|
| SAFE | VLA 최종 layer feature(OpenVLA/π0-FAST: 마지막 transformer block, π0: velocity field 투영 직전) → LSTM(1층, hidden 256) → scalar | unseen 최고 ROC-AUC ≈ 84.5%(π0-FAST+LIBERO, SAFE-LSTM) |
| FIPER | ① RND(policy embedding space에서 random network distillation) OOD, ② action-chunk entropy(생성 액션 불확실성). 시간윈도우 집계 + conformal 보정(성공 rollout으로) | "기존보다 정확·조기" (본문 수치 미확인) |
| FAIL-Detect | policy 입출력→scalar로 distill, sequential OOD를 conformal p-value로 재정의 | "SOTA보다 정확·빠름" (구체 수치 abstract 미기재) |
| UNISafe | latent world model의 epistemic uncertainty를 HJ reachability의 증강 상태공간에 결합, conformal로 threshold 보정 | Franka 시뮬레이션+실기 |
| Early Warning Signals for OpenVLA | OpenVLA activation 위 **layer별** logistic probe (L8/L10/L16 비교) | **layer 16 AUROC 0.972, AUPRC 0.352**, 15-step horizon, occlusion으로 SR 57%→17% 유도 |
| VLAConf | frozen VLA 내부표현 + step-conditioned modeling → calibrated confidence | LIBERO, 앙상블/토큰확률 baseline 대비 우위(수치 미확인) |
| VLA-FAIL | LLMD(last-layer Mahalanobis, token-wise) + ACC(action-chunk 겹침 불일치, receding-horizon 특성 활용) | AUCPDT(정밀도·재현율·검출시간 결합 지표) 도입 |
| Perturbation-Based Uncertainty | transformer hidden activation에 Gaussian perturbation 주입 → 여러 action 예측 disagreement | LIBERO/LIBERO-PRO, sampling 기반 UQ보다 distribution shift에서 우위 |
| SAVE (Uncertainty Quantification for Flow-Based VLA) | flow-matching(π0류) velocity-field disagreement(작은 앙상블) | uncertainty-guided data acquisition ≥22% 샘플 절약 |
| ReconVLA | state-level: Mahalanobis distance / action-level: Conformal Quantile Regression 이중 모니터 | 벤치마크 미확인 |
| RC-NF | robot-conditioned normalizing flow의 확률밀도를 anomaly score로 | **<100ms** 반응 레이턴시, LIBERO-Anomaly-10(이상 3범주) |
| PATCH | action-chunk 조건부 latent patch 예측 대비 잔차(self-motion으로 설명 안 되는 residual) 누적 | PATCH-Router 대비 우위(구체 수치 미확인) |

**패턴**: 신호원은 크게 (a) 내부 feature 거리/분포(Mahalanobis, kNN, RND, normalizing flow, CQR),
(b) 앙상블/perturbation 불일치(velocity-field disagreement, activation perturbation), (c) action 자체
통계(entropy, chunk consistency)로 3분된다. **내부표현 기반이 압도적 다수** — 이 프로젝트의
"내부 latent" 우선순위와 방향은 일치하나, 전부 **succ/fail 이진 스칼라**이고 phase나 실패 TYPE
조건부는 아니다(§0-2,3).

---

## 3. 축2 — 개입 시점 결정 / 검출지연(delay)을 지표로 쓰는 연구

| 논문 | "언제" 프레이밍 | 정량 수치 |
|---|---|---|
| AEGIS | frozen weak-policy activation probe, **trajectory 첫 30%** 윈도우에서 high-risk step만 escalate | AUROC 0.764 [CI 0.70,0.84], **38%** 스텝만 강한 정책으로 전환, 궤적회복 **10.1%**(blind escalation 4.6%, random 5.1%) |
| ActProbe | action space만으로 TCE+ACM(task-conditioned LSTM-MLP) | **F1-timeliness Pareto frontier +12.7%** hypervolume, early-detection ROC-AUC(미학습 task +9.0%), q=0.25/0.50 두 시점 비교 |
| VLA-FAIL | AUCPDT = precision·recall·detection-time 결합 threshold-무관 지표 | 구체 수치는 본문 표(미확인), 지표 정의만 확인 |
| SAFE | "accuracy vs average detection time" trade-off, Fig.4 (여러 conformal 유의수준 α) | π0-FAST+LIBERO unseen ROC-AUC ≈84.5%가 최고, "조기 단계에도 고정확도로 검출 가능" 정성 서술 |
| REPAIR-Bench | 실패 lifecycle(감지→위치화→복구예측) 벤치마크, 위치화 오차를 **초 단위**로 직접 보고 | **평균 부호오차 −0.51초, 중앙값 절대오차 2.97초** — 조사한 논문 중 유일하게 delay를 "초" 단위로 벤치마크화 |
| Human-in-Loop Confidence-Aware | "어느 module이 원인인가"(module selector) + "지금 인간에게 물을까 자율진행할까"(querying algorithm) **두 결정 분리** | 모듈별 calibrated uncertainty × 개입비용 모델 결합(HRI 2026) |
| Doomed from the Start(★인접도메인, LLM agent) | hidden-state probe cascade, "1라운드째부터 최종실패 예측" — **행동기반 탐지가 늦어지는 정확한 라운드에 내부상태는 이미 앞섬**을 정량 비교 | 1라운드 AUC가 표면-스코어러의 **2~3라운드 뒤 피크치와 동급/우위**, 토큰생성 60.2%(TextCraft)·54.9%(WebShop) 절감 |

**메타 관찰**: FIPER·FAIL-Detect·UNISafe처럼 "더 빠르다/더 이르다"를 **정성적으로만** abstract에 쓰고
구체 delay 수치를 본문 표로 미루는 논문이 많음 — 즉 "조기검출"을 표방하지만 delay 자체를 1급 지표로
승격한 논문은 위 6~7건 정도로 좁다. **Doomed from the Start(비-embodied)가 가장 명시적으로 "delay가
개입 가능성을 결정한다"는 인과를 수치화**한 사례라는 점이 역설적 — embodied VLA 쪽에는 이 정도로
"delay 자체가 종속변수인" 실험 설계가 드물다.

---

## 4. 축3 — 시간축 구조: 시퀀스모델·누적통계·conformal·CUSUM·martingale

| 논문 | 방식 | 세부 |
|---|---|---|
| How VLAs Fail Differently ⚠ | **conformal p-value → CUSUM**: `S_t = max(0, S_{t-1} + 1[p_t<α] − α)`, alarm when `S_t > h` | "formal false-alarm bound"는 고전 CUSUM 이론(Page 1954) 인용뿐, 논문 자체 유도 없음. SafeContract 툴킷, n=450 에피소드(PushT+ALOHA 14-DOF) |
| Diagnostic Runtime Monitoring with Martingales | **복수의 conformal martingale을 동시 배포**, 각각 **다른 종류의 distribution shift를 진단** — "근본원인 진단이 배포 생명주기 전반의 적절한 중재를 가능케 한다" | 시뮬레이션+실하드웨어, 안전필수 로봇 설정. 근본원인별 martingale 분리 자체가 "실패 TYPE 구분"의 통계적 아날로그(§5 연결) |
| SAFE-LSTM | per-timestep causal LSTM, BCE loss, 전 타임스텝에 rollout label 방송 | (§2 참조) |
| Hide-and-Seek in Trajectories | 기존의 "모든 timestep에 균일 trajectory-label" 비판 → **inter-/intra-trajectory contrastive objective**로 시간적 구조 도입, conformal prediction 하 accuracy-timeliness trade-off | LIBERO/VLABench/실로봇, OpenVLA/π0/π0.5 |
| Foresight | action-conditioned world-model latent, **functional conformal prediction(FCP)**으로 장기간(long-horizon) threshold 보정 | LIBERO-Long/ManiSkill-Long/BEHAVIOR-1K |
| SAFECAST | hidden-state risk probe + **functional conformal prediction** + contrast-set 섭동으로 calibration 견고화 | DROID/LIBERO |

**정리**: "시간축 구조"는 대부분 **conformal prediction 계열의 sequential/functional 확장**으로 수렴하고
있고(FIPER/FAIL-Detect/UNISafe/Foresight/SAFECAST/Hide-and-Seek 전부 conformal 어휘 공유), **CUSUM은
1편(How VLAs Fail Differently, 워크숍·단독저자·저신뢰), martingale은 1편**(2024, 로봇 일반 안전 —
VLA/조작 특정 아님)뿐이라 이 프로젝트가 관심 가진 "누적 통계" 계열은 conformal 바깥에서는 여전히
드물다. LSTM 같은 고전 시퀀스모델은 SAFE 원조 이후 거의 기본값으로 재사용(ActProbe도 LSTM-MLP).

---

## 5. 축4 — 실패 유형(failure type) 구분

| 논문 | 분류체계 | 판정 방식 |
|---|---|---|
| **Guardian**(Scaling Cross-Env Failure Reasoning) | **Planning(5)**: wrong object manipulated / wrong object state·placement / wrong order / missing subtask / contradictory subtasks. **Execution — 시뮬(4)**: no gripper close / wrong object state·placement / wrong object manipulated / imprecise grasping·pushing. **Execution — 실로봇(2)**: task-execution semantic mismatch / revert action | 다중시점 VLM이 지시문+제안된 계획/subtask+관측을 입력받아 **사후(post-hoc) VQA 형태로 판정**. Table II는 planning-accuracy·execution-accuracy를 분리 보고하나 phase 세부 분석은 없음. 벤치마크: RoboFail/RoboVQA/UR5-Fail(신규) |
| Safe Embodied AI Cross-layer | **4대 위험유형**: semantic misgrounding / subtask-level error propagation / execution drift / contact-rich physical risk | planning-time/policy-time/execution-time 3-layer 개입지점 프레이밍(포지션 논문에 가까움). "조작 특화 안전 벤치마크 부재"를 스스로 지적 |
| How VLAs Fail Differently ⚠ | **아키텍처 계열별 실패 시그니처**: discrete(autoregressive/VQ-VAE) 계열 = jerk·reversal-rate 최강 예측자(VQ-BeT jerk 2.7×), continuous(diffusion/flow/chunking) 계열 = stall 특성 | black-box action 통계만으로 사후 분류. "실패 원인(goal/motor)"이 아니라 "**어떤 모델 계열이 어떤 통계적 흔적을 남기는가**"이므로 이 프로젝트가 찾는 유형축과는 다른 축 |
| VLA-FAIL | (저자 프레이밍 아님, 재해석) LLMD≈분포이탈/상태측, ACC≈행동일관성/모터측 — 두 신호를 결합만 하지 goal/motor로 명명하지 않음 | — |
| AHA(2024, 배경) | VLM 기반 실패 탐지+원인 추론의 원조격. 구체 taxonomy는 본 조사에서 본문 미확인(abstract만) | — |
| How Visible Are Silent Manipulation Failures | **"silent failure/false-success"**: 로봇 자체 성공판정이 실제 실패를 놓치는 경우 — grasp-fail·placement-fail과 직교하는 축("검출 자체의 무결성") | proprioception vs vision 신호 비교, ALOHA cube-transfer/peg-insertion |

**핵심 공백 확인**: 5건 모두 (a) **사후 판정**(완료되거나 최소 관측 구간을 다 본 뒤 VLM/통계로 라벨링)
이거나 (b) **모델 아키텍처 시그니처**(How VLAs Fail Differently)처럼 "원인 유형"이 아닌 다른 축이다.
**"온라인, 내부 activation만으로, goal-오인 vs motor-오류를 실시간 분기"하는 논문은 이번 조사에서
1건도 확인하지 못했다** — memory `notall-online-failuretype-niche`의 미점유 판정과 정합.

---

## 6. 축5 — 검출 후 무엇을 하는가

| 논문 | 대응 | 세부 |
|---|---|---|
| AEGIS | **정책 전환**(hand-off) | 약한 정책의 조기경보 → "필요한 스텝에만"(38%) 강한 별도 정책으로 제어권 이전. 감지-대응이 같은 프레임워크에 결합된 사실상 유일한 사례 |
| FAR | **재시도**(retry) | Failure-Contrastive Preference Adaptation(실패에서 멀어지는 선호학습) + 경량 action perturbation으로 지역탐색. 감지 메커니즘 자체는 논문 초점 아님(주어진 것으로 가정). SR +17.6%(sim)/+11.7%(real), baseline=표준 diffusion policy |
| FailSafe | **VLM reasoning 기반 복구행동 생성** | 실패사례-복구행동 쌍을 자동생성해 학습. ManiSkill, π0-FAST/OpenVLA/OpenVLA-OFT 대상 최대 +22.6% |
| Human-in-the-Loop Confidence-Aware | **인간 호출 여부 결정** | module selector(원인 모듈 특정) + querying algorithm(자율 진행 vs 인간에게 물을지, 개입비용까지 모델링) — "누구에게/언제 물을지"를 명시적 최적화 대상으로 삼은 드문 사례 |
| REPAIR-Bench | **사용자-선호 복구전략 예측** | 감지+위치화 이후 "이 사람이 선호할 복구방식"을 추론(hierarchical recurrent modeling, QLoRA-Mistral-7B: Hit@5=0.76, F1@5=0.32) |
| Unified Framework(VLM+BT) | **Behavior-tree 자기수선** | 기존 BT 조건 검증 → 누락 전제조건 추가 → 필요시 신규 스킬 생성의 3단계. ABB YuMi 실로봇 + AI2-THOR |
| SC-VLA(2024, 배경) | **fast/slow 이중시스템 self-correction** | fast=직접 행동예측, slow=실패 성찰(원인 파악→전문가피드백 요청→반성→교정) — 전환 트리거는 abstract에 불명확 |

**정리**: 대응 방식은 재시도(action-space) / 정책전환(policy-space) / 인간호출(외부) / 기호적
재계획(BT·subtask) 4갈래로 수렴하고, **activation을 직접 조작(steering)하는 대응은 이 축의 논문
중에는 없다** — RL2-VLA(§9, action-space 합성+verifier)가 그나마 가장 가까운 예이며, 그마저도
"activation write-in"이 아니라 "action-space 합성"이다(memory `rl2-vla-competitor-analysis` 재확인).

---

## 7. 인접 도메인 사례 (embodied 조작 아님 — 방법론적 참고만)

- **Doomed from the Start**(2607.06503) — **LLM 디지털 에이전트(TextCraft/WebShop) 도메인, 로봇 아님**.
  hidden-state probe cascade로 "1라운드째부터" episode 실패를 예측. "internal states predict failure
  before behavior does"의 가장 정량적인 근거(§3). 로봇 도메인 논문이 이 정도로 명시적인 "delay가
  개입 가능성을 결정" 실험을 설계한 사례가 없다는 점에서 **방법론 수입 대상**으로 가치 있음(probe
  cascade 구조 자체는 phase-conditioned 검출기 설계에 참고 가능).
- **Early Failure Detection — Surgical Soft-Tissue**(2501.10561, RSS'25 OOD 워크숍, Utah) — deep
  ensembles가 MC dropout보다 강한 신호, dVRK 수술로봇 실기 검증, zero-shot sim2real +47.5%. "조기
  식별 시스템"이라고만 서술, 구체 시간/스텝 수치는 abstract에 없음 — 표방하는 문제의식(수술 로봇은
  실패 감지가 늦으면 물리적으로 되돌릴 수 없음)은 이 프로젝트의 "늦으면 개입불가" 프레이밍과 정확히
  같으나 조작(kitchen-style) 도메인이 아님.
- **Phase-Conditioned Imitation Learning**(2605.29407, Tohoku, T-Mechatronics) — **직물(옷걸기) 조작
  단일 task**. 시각+힘+자세 융합 멀티모달 phase 예측기가 실시간으로 phase를 추정하고, 접촉실패(시각
  단독으로는 안 보임)를 감지해 자율 복구궤적을 트리거. T-shirt 걸기 SR 56%→87%. **"phase detector"가
  명시적으로 존재하는 유일한 확인 사례**이지만 (a) VLA/일반 조작이 아닌 특정 로봇팔+직물 세팅,
  (b) phase는 제어 스케줄링용이지 "phase별로 다른 실패 확률 분류기"가 아님 — §0-2 판정에 영향 없음.

---

## 8. 특히 관심 질문에 대한 답

### Q1. "실패를 늦게 감지하면 개입할 수 없다"는 문제의식을 다룬 연구가 있는가

**있다, 명시적으로 다수.** AEGIS는 제목 자체("...before long-horizon failures compound")가 이 문제의식.
Doomed from the Start(인접도메인)는 "행동기반 탐지가 정확도상 따라잡는 라운드엔 이미 계산예산 대부분이
소진된 상태"라는 표현으로 delay=개입불가를 직접 정량화. Early Failure Detection(수술)은 "물리적으로
되돌릴 수 없는 조직손상 전에 잡아야 한다"는 동기를 명시. REPAIR-Bench는 위치화 delay를 초 단위 벤치마크
지표로 승격시킨 유일한 사례. **다만 embodied VLA 조작 도메인 안에서는 이 문제의식을 "정량 delay가
개입성공률에 미치는 인과"로까지 실험 설계한 논문은 확인하지 못했다** — 대부분 "더 이르게 잡을수록
좋다"는 동기 서술에 그치고, delay 자체를 독립변수로 스윕해 개입성공률 변화를 측정한 논문은 없음. 이
지점이 이 프로젝트의 "온라인 phase/type 식별이 안 되면 steering을 라우팅할 수 없다"는 핵심 미해결
문제와 정확히 같은 결의 공백이다.

### Q2. 검출기를 조작 단계(phase)별로 다르게 두거나 단계별 성능차를 보고한 연구가 있는가

**거의 없다.** 확인된 것 중:
- SAFE 원논문이 **직접 "phase 기반 성능 분해 없음"을 확인**(본문 fetch로 재확인) — 정성적으로
  "초기 단계에도 높은 정확도로 검출 가능"이라고만 서술.
- ActProbe는 rollout 진행률 q=0.25/0.50 **2점 비교**만 제공 — 조작 subtask phase(reach/grasp/
  transport/place)가 아니라 시간축 %.
- Guardian은 planning-accuracy vs execution-accuracy를 **분리 보고**(coarse 2분류)하지만 execution
  내부의 세부 phase 차이는 없음.
- Phase-Conditioned Imitation Learning(직물)은 phase detector가 있지만 실패-검출기 성능을 phase별로
  비교하는 논문이 아니라 phase를 제어 스케줄링에 쓰는 논문.
- Diagnostic Runtime Monitoring with Martingales는 "여러 distribution-shift 유형을 각각 다른
  martingale로" 감지한다는 점에서 구조적으로 가장 근접하지만, 이는 **원인(shift-type)별 분리**이지
  **시간(phase)별 분리**가 아니다.

**결론**: "phase 조건부 검출기" 또는 "phase별 검출 성능 ablation"을 명시적으로 실험한 VLA/조작
논문은 이번 조사(32편 실물확인)에서 **0건**. 이 프로젝트가 이미 내부적으로 갖고 있는 findings —
`seen18-failure-onset-regimes`(초기조건형 vs 실행표류형), `n16-online-detection-feasible`(DiT
t_d=11 vs VL t_d=5, pathway별 검출시점 차이), `per-instruction-detector-eval`(task별 검출기 성능
편차) — 는 외부 문헌에 아직 대응물이 없는 것으로 보인다.

---

## 9. 이 프로젝트 문헌과의 교차참조 (재조사 안 함)

- **SAFE**(2506.09937) — `SAFE-LSTM`으로 이미 이 프로젝트가 재구현·재학습. memory
  `seen18-safe-detector-verified`: "SAFE 공정 metric(min-length T)로 unseen ~chance(0.43), 0.99는
  비공정 변형". 즉 논문이 보고한 84.5% AUROC는 이 프로젝트 자체 태스크·조건에서는 재현되지 않음(다른
  실패 confound 통제 방식 차이로 추정) — 인용 시 "논문 원 보고치"와 "이 프로젝트 재현치"를 반드시
  구분해서 쓸 것.
- **RL2-VLA**(2607.26991) — SAFE-LSTM + 시변 conformal threshold로 "언제 action-space 다양성을
  주입할지" 게이팅, adaptive vs always 최대 +8.9pp. "성공 상태 개입=해악, 실패 상태만 이득"이라는
  독립 인과 증거는 이 프로젝트의 apple/drawer β=1.0 관측과 정합. 상세는
  `docs/references/reading_notes/rl2_vla_adaptive_steering.md`. 이번 조사로 새로 확인된 SAFE 계열
  후속 논문들(FIPER/FAIL-Detect/VLA-FAIL/ActProbe 등)은 RL2-VLA의 검출기를 교체할 후보군으로 볼 수
  있으나, **전부 phase/type 조건부가 아니므로 RL2-VLA의 "구분 없음" 한계 자체는 그대로 남는다**.
- **VLA TTS Adoption Survey**(`vla_tts_adoption_survey_2026-08.md`) — "상시→조건부 개입" 게이팅
  패러다임 전환을 이미 정리(Gated GeoBoN/τ0-VLA confidence routing/RL2-VLA). 이번 조사는 그 게이팅을
  구동하는 **검출기 자체**의 최신 문헌을 보강하는 위치.

---

## 10. 미확인 (검색 스니펫만, 실물 미검증 — 인용 금지·후속 확인 필요)

- Robot failure mode prediction with deep learning sequence models (Neural Computing and
  Applications, non-arXiv, 2024) — "GIGP"/"ATFA" 언급, 페이월 추정으로 미확인.
- RePO-VLA: Recovery-Driven Policy Optimization — arXiv:2605.09410
- Reliable Robotic Task Execution in the Face of Anomalies — arXiv:2510.23121
- Multi-Rank Subspace Change-Point Detection for Monitoring Robotic Swarms — arXiv:2506.18562 (스웜
  도메인, CUSUM 확장이라 §4 후보였으나 미확인)
- World Model Failure Classification and Anomaly Detection for Autonomous Inspection —
  arXiv:2602.16182 (검사 도메인, 조작 아님)
- Failing Forward: Adaptive Failure-Informed Learning for VLA — arXiv:2605.08434
- Failure Detection for Surgical Robot Imitation Policies via Flow-Matching World Modeling —
  arXiv:2607.27511
- Identifying Precursors to Failures in Robotic Lift-and-Place Tasks (OpenReview, ID 미확보)
- Pixels to Proofs: Probabilistically-Safe Latent World Model Control via Parallel Conformal
  Robust MPC — arXiv:2606.15594
- Fail2Progress: Learning from Real-World Robot Failures with Stein Variational Inference —
  arXiv:2509.01746

---

## 11. 결론 — 무엇이 풀렸고 무엇이 비어 있는가

**이미 풀린 것(경쟁 밀도 높음, 재발명 불필요)**:
- VLA 내부표현/불확실성으로 succ/fail 이진 검출 (SAFE 계열 15편+) — AUROC 0.7~0.97대, 아키텍처
  무관하게 다수 검증됨.
- Conformal prediction을 시퀀셜/functional 형태로 확장해 threshold를 통계적으로 보정하는 것 —
  사실상 이 분야의 공용어가 됨.
- "실패 예상 시에만 개입"이라는 게이팅 원칙 자체(RL2-VLA로 인과까지 확인) — VLA TTS survey에서 이미
  "2026년 실질 전선"으로 판정.

**여전히 비어 있는 것(이 프로젝트가 채울 수 있는 자리)**:
1. **Phase(조작 subtask 단계) 조건부 검출기** — 검출기를 phase별로 달리 두거나 phase별 성능차를
   ablation한 VLA/조작 논문 0건 확인. (§8-Q2)
2. **온라인·내부활성 기반 실패 TYPE(goal/motor) 실시간 분기** — 근사물(Guardian, Safe Embodied AI)은
   전부 사후 VLM 판정이고 온라인 activation 기반이 아님. (§5)
3. **Delay 자체를 독립변수로 한 "개입성공률 vs 검출지연" 인과 실험** — 문제의식은 흔하지만(§8-Q1)
   embodied 도메인에서 이를 정량 스윕한 논문은 없음(REPAIR-Bench가 delay를 재긴 하나 인과 실험은
   아님).
4. **검출 결과를 activation-level steering으로 직결하는 개입** — 검출 후 대응은 재시도/정책전환/
   인간호출/기호적 재계획으로 수렴하고, activation write-in으로 이어지는 사례는 이 조사 범위에서
   없음(RL2-VLA도 action-space).

→ 이 프로젝트의 "pathway-resolved + phase-matched steering, 온라인 phase/type 식별이 중심 미해결
문제"라는 현재 방향(`docs/steering/RESEARCH_DIRECTION.md`)은 이번 문헌 조사로도 반증되지 않았고,
1·2·4의 교집합(내부 latent × phase-matched × type-분기 × write-in)은 여전히 미점유로 확인된다.
