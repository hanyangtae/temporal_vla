# 인턴 온보딩 커리큘럼 — VLA Latent Steering 연구

이 문서는 이 연구실의 **VLA latent steering** 연구에 합류한 신규 인턴(학부 4학년)이
한 학기(≈12주) 동안 스스로 따라갈 수 있는 학습 로드맵이다. 무엇을(개념) → 어떤 순서로
→ 무엇을 읽고 → 무엇을 직접 재현하는지를 주차별로 묶었다.

## 0. 이 연구가 무엇인가 (첫날 10분)

목표: GR00T VLA 모델에서 성공/실패의 latent(내부 활성화) 표현을 **pathway별
(VL=goal "무엇을" / DiT=motor "어떻게")** 로 구분하고, **phase-matched contrastive
conceptor steering** 으로 추론 시 활성화를 성공 부분공간으로 밀어 Success Rate(SR)를
올린다. **백본은 재학습하지 않는다** — 추론 시점의 개입만으로 행동을 바꾼다.

- 메인 method = pathway-resolved + phase-matched activation steering.
  - contrastive conceptor: `C_steer = C_success ∧ ¬C_failure` (Boolean AND-NOT).
  - conceptor `C = R(R+α⁻²I)⁻¹`, `R = E[hhᵀ]` (활성화 covariance), aperture α.
  - steering gate `M = (1-β)I + β·C_steer`, `h' = h·Mᵀ` (곱셈형, COAST 계열).
- 두 축: (1) **pathway 분리** — VL(이른 신호 t≤8) / DiT(늦은 신호 t≥12, 후기 block 23~31)
  를 각각 steer. (2) **phase-matched** — DiT는 rollout phase(시간 t)에 조건부로 steer.
- ★ 중심 미해결 문제: 추론 중(**online**) 어느 pathway가 어느 phase에서 실패했는지
  식별할 수 있는가. 안 되면 steering을 라우팅할 수 없다.

단일 출처(SSOT): [`docs/steering/14_pathway_phase_online_steering.md`](../steering/14_pathway_phase_online_steering.md),
RQ/가설은 [`docs/steering/15_research_structure.md`](../steering/15_research_structure.md).

## 이 문서 쓰는 법

- 각 주차는 4블록: **(A) 학습 개념 / (B) 읽기 / (C) 실습(재현 중심) / (D) 산출물·확인질문**.
- 주차 끝 **산출물(D)** 은 멘토에게 보여주고 sign-off 받는다. 특히 **W4·W6은 게이트** —
  통과 못 하면 다음으로 넘어가지 않는다(이 둘을 건너뛰면 이후 분석이 전부 오염된다).
- `[SKIP if known]` 구간은 사전지식이 있으면 건너뛰고, 그 시간을 분석(Phase II) 주차에 재투자.
- 가정 baseline: 파이썬 OK + 학부 수준 ML/선형대수. transformer/PyTorch 실전, 논문 읽기,
  로보틱스/manipulation은 보장하지 않으므로 커리큘럼에서 build-up 한다.
- 무게중심: ① 실험 운영·재현(주 목표, W1~10 전반) → ② 분석·표현 연구 기여(W4~10) →
  ③ method 개발(W11 stretch, 보너스).

---

## 1. 개념 의존 그래프 (티어 구조)

`→` = "먼저 알아야 함". Tier 0(선행지식)부터 Tier 4(미해결 문제)로 쌓인다.

```
TIER 0  선행지식 (build-up, [SKIP if known])
  T0a Python/git/Docker/CLI            T0d 논문 읽는 법 (claim→method→ablation)
  T0b PyTorch·Transformer (attention, residual stream, hidden state)
  T0c 선형대수 (covariance, eigen/PSD, projection, pseudo-inverse)
  T0e 로보틱스 기초 (arm/action space, rollout, success rate, RoboCasa)
            │
            ▼
TIER 1  VLA & 인프라  (무게중심 ①)
  T1a GR00T 아키텍처: Eagle VLM → VL-SA → DiT flow-matching head (denoise K / horizon H / dim D)  ◀ T0b
  T1b 통일 HTTP API (/act /act_with_features /reset /health) + VLAClient
  T1c Processor(추론) vs Dataset+Adapter(학습) 구분
  T1d Docker 7서비스 / cache 경로(path_setup·cache_env) / remote_compute / eval 로컬-전용
  T1e SAFE feature = DiT pre-velocity activation; 파이프라인 collect→split→steer→analyze→vis
            │
            ▼
TIER 2  통계·표현분석  (무게중심 ①→②, 커리큘럼 무게중심)
  T2a succ/fail 분리 + AUROC                                  ◀ T0c,T0d
  T2b ★길이 confound: 실패=45-step timeout / 성공=조기종료 → AUROC 0.998 아티팩트; fixed-t 통제
  T2c within-task 약신호 (fixed-t AUROC 0.6~0.7) 실재 + permutation test + 시간축 직교
  T2d ★instruction confound (slide in/out 쏠림 → VL AUROC 부풀림); balanced task만 신뢰
  T2e LDA + ★in-sample 과적합 함정 (train/test split, CV)
  T2f pathway 분리 (VL early t≤8 / DiT late t≥12, block 23~31)  ◀ T1a
  T2g 두 실패 regime (초기조건형 vs 실행표류형) + cross-task 공유 zone
            │
            ▼
TIER 3  Method: conceptor steering  (무게중심 ②, ③ 진입)
  T3a R=E[hhᵀ], C=R(R+α⁻²I)⁻¹, aperture α, eigen λ=σ/(σ+α⁻²)   ◀ T0c
  T3b Boolean algebra (¬/∧/∨) + contrastive conceptor C_steer = C_s ∧ ¬C_f
  T3c steering gate M=(1-β)I+β·C_steer, h'=h·Mᵀ (COAST/multiplicative)
  T3d 수치 주의 (float64 계산/float32 저장, N≪d → pinv, 대칭화), α·β 선택
  T3e pathway-resolved + phase-matched; 사다리 ablation (global→pathway-split→+phase-bin), ΔSR
            │
            ▼
TIER 4  ★중심 미해결 문제  (무게중심 ③ = stretch)
  T4a online phase/failure-type 식별 (안 되면 steering 라우팅 불가)
  T4b serial pathway confound (Eagle→VL-SA→DiT) crossover 검증
  T4c SAFE-LSTM detector + conformal prediction; VITA progress(보조 phase신호)
  T4d 라우팅 crossover(C4) 설계, falsification, 경쟁자 Path-Deviation-Heads
```

핵심 간선:
- **T0c → T3a**: 선형대수(covariance/eigen/pseudo-inverse)가 conceptor 수학의 전제.
- **T0b → T1a → T2f**: hidden state 개념 → GR00T 구조 → pathway 분리.
- **T2b는 Tier 2 전체의 전제**: 길이 통제 없이는 모든 표현 분석이 오염된다.
- **T2f + T3e → T4**: pathway 분리 + 사다리 steering이 online 라우팅의 토대.

---

## 2. 주차별 12주 커리큘럼

모든 (B)/(C) 경로는 repo에 실재함을 확인했다. 경로는 repo 루트 기준.

### Phase I — 인프라 가동 + 선행지식 (W1~3, 무게중심 ①)

#### W1 — 레포 지형 + 환경 세팅 + 첫 서버 ping
- **(A)** 레포의 두 덩어리(인프라 vs 연구), 경로 SSOT 규칙(하드코딩 금지), Docker/cache 모델,
  VLAClient. `[SKIP if known]` git/CLI 기초.
- **(B)** `docs/README.md` → `CLAUDE.md` → `docs/00_repo_orientation.md`(코드 지도·SSOT 맵)
  → `docs/01_serving_interface.md` → `docs/cache_paths.md`, `docs/02_docker_guide.md`.
- **(C)**
  1. `scripts/utils/cache_env.sh` source / `scripts/path_setup.py` import 해서
     `CHECKPOINTS_ROOT`, `DATA_ROOT` 출력 (경로 하드코딩 절대 금지를 체득).
  2. `docker-compose.yml`의 서비스 구조를 읽고, GR00T N1.6 서버 1개를 띄워 `/health` 200 확인.
  3. `scripts/utils/vla_client.py`의 `VLAClient`로 `/health`를 호출하는 코드 한 줄.
- **(D)** 서버 up + `/health` 200 캡처 + 레포 두 덩어리를 3문장 요약.
  Q: "체크포인트/데이터 경로는 어디서 결정되나? 왜 하드코딩 금지인가?"

#### W2 — 통일 API 추론 + eval 1회 완주  · **[Milestone M1]**
- **(A)** `/act` vs `/act_with_features` 요청/응답 스키마(`observation.images.*` base64 PNG /
  `observation.state.*` / `action.*` `[n_steps,dim]` / `features.hidden_states` blob),
  eval 표준(EVAL_SEED=100000, N_ENVS=2, N_EP=20), eval 로컬-전용 규칙. `[SKIP if known]` HTTP/base64.
- **(B)** `docs/01_serving_interface.md`(스키마 정독), `CLAUDE.md` "평가 표준" 섹션,
  `docs/groot/n16_02_eval.md`.
- **(C)** VLAClient로 더미 obs → `/act` action shape 확인 → `/act_with_features`로
  `hidden_states` blob shape `[K,16,1024]` 확인. `scripts/eval/robocasa_eval.py`로
  **작은 N_EP** RoboCasa eval 1회 완주 → `outputs/eval/...`에서 SR 산출물 확인.
- **(D)** SR 산출물 경로 + "eval은 왜 로컬, 대용량 rollout은 왜 remote".
  **M1 = eval 1회 성공.**

#### W3 — GR00T 아키텍처 + PyTorch/Transformer build-up
- **(A)** GR00T 흐름(Eagle VLM → VL-SA → DiT flow-matching head; denoise step K / horizon H /
  dim D), SAFE feature = DiT pre-velocity activation 위치, transformer hidden state/residual
  stream, manipulation 기초(rollout, success rate, RoboCasa task). `[SKIP if known]` transformer
  기초 → 그 시간을 GR00T flow map 정독 + W4 예습에 재투자.
- **(B)** `docs/groot/00_groot_flow_map.md`(call chain), `docs/groot/00_groot_steering_explorer.html`
  (브라우저로 열어 code map/수식 한 화면), `docs/groot/n16_03_safe_overview.md`,
  `docs/groot/n16_06_safe_inference_semantics.md`. 외부(필요시): The Illustrated/Annotated
  Transformer로 attention·residual stream 감 잡기. flow matching은 "denoising으로 action 생성"
  수준 개념만.
- **(C)** `src/policies/groot/`에서 Eagle→VL-SA→DiT 호출 경로를 grep으로 추적, SAFE feature
  hook 지점 1곳 찾기. seen18 task 목록을 `scripts/safe/groot_n16/robocasa/split/prepare_seen18_manifest.py`
  / `configs/robocasa/`에서 확인.
- **(D)** GR00T 데이터 흐름 손그림 1장(VL hidden / DiT hidden 위치, K/H/D 표기).
  Q: "VL과 DiT pathway는 직렬인가 병렬인가? 그게 왜 나중에 crossover 문제가 되나?"

### Phase II — 표현 분석 재현 + confound 통제 (W4~7, 무게중심 ①→②, **핵심**)

#### W4 — ★길이 confound + seen18 재현  · **[게이트 / Milestone M2]**
- **(A)** succ/fail 분리 + AUROC, **길이 confound** — 실패=항상 45-step timeout / 성공=조기종료
  → 길이만으로 AUROC 0.998 (time-pooled feature 분리는 아티팩트). fixed-t truncation으로 통제.
- **(B)** `docs/steering/01_seen18_latent_analysis.md`(정독 — 길이 confound가 모든 해석의 전제),
  `docs/steering/README.md`, `docs/steering/15_research_structure.md`의 C1 섹션,
  `docs/seen18_safe_detector_verification.md`. 외부: **SAFE**(`docs/references/SAFE.pdf`,
  요약 `docs/references/SAFE.txt`).
- **(C)** `scripts/safe/groot_n16/robocasa/vis/seen18.py` 재현(길이 confound·AUROC·실패방향 figure).
  time-pooled feature로 AUROC≈0.998을 직접 재현한 뒤, fixed-t로 잘랐을 때 떨어지는 것을 본인 손으로 확인.
- **(D)** seen18 figure + "왜 0.998이 가짜인가" 5분 설명. **M2.**
  게이트: 여기서 길이 confound를 못 짚으면 이후 모든 분석이 무의미 → 통과 필수.

#### W5 — within-task 약신호 + permutation test (시간축 직교)
- **(A)** 통제 후에도 within-task fixed-t AUROC ~0.6~0.7 약신호가 실재. permutation test로
  우연 아님을 검증, 신호가 시간축과 직교함을 확인.
- **(B)** `docs/steering/01_*`의 within-task/permutation 부분 재독,
  `docs/steering/15_*` C1 "실패 전 신호 실재" 근거. 외부: SAFE 평가 방법(AUROC/conformal) 재독.
- **(C)** `scripts/safe/groot_n16/robocasa/vis/run_feature_visualization.py`를 경유해
  `vis/analyses/{auroc,length,temporal,windows}.py` 드라이버 실행 → fixed-t AUROC + permutation
  null 분포 재현. (대용량이면 분석은 remote `~/anaconda3/bin/python`, SR/eval만 로컬 —
  `scripts/utils/remote_compute.sh`.)
- **(D)** fixed-t AUROC 점추정 + permutation p-value 표 1개.
  Q: "이 약신호가 그냥 '에피소드 길이/진행도'를 다시 재는 게 아니라는 걸 어떻게 보였나?"

#### W6 — ★instruction confound + LDA 과적합 함정  · **[게이트]**
- **(A)** **instruction confound**(특정 instruction, 예 slide in/out이 성공률과 상관 →
  VL AUROC 부풀림; instruction-balanced task만 신뢰). **LDA in-sample 과적합 함정**
  (train/test split·CV 없으면 분리가 가짜).
- **(B)** `docs/steering/11_instruction_confound.md`(정독 — VL/DiT LDA 사분면 방법),
  `docs/steering/11_phase4_n15_instruction_fixed_plan.md`(instruction-fixed 재수집 동기).
  외부: **NOTALL**(`docs/references/NOT ALL FEATURES ARE CREATED EQUAL_ICLR2026.pdf`,
  요약 `docs/references/NOTALL.txt`).
- **(C)** `scripts/safe/groot_n16/robocasa/analyze/vl_dit_lda_analysis.py` 재현하되
  **반드시 train/test split** 으로. in-sample LDA(전체 fit→전체 평가)와 held-out LDA의 AUROC
  차이를 직접 수치화해 과적합 폭을 본다. instruction-skew task(예 SlideDishwasherRack)와
  balanced task(예 OpenDrawer)를 분리해 VL AUROC 차이 확인.
- **(D)** "in-sample vs held-out LDA AUROC" 비교표 + "instruction-balanced로 거른 뒤 VL 신호" 표.
  Q1: "왜 in-sample LDA는 거의 항상 잘 나오나?" Q2: "VL AUROC가 instruction 아티팩트인지
  어떻게 가르나?" 게이트.

#### W7 — pathway 분리 + 두 실패 regime  · **[Milestone M3]**
- **(A)** pathway 분리(VL 이른 t≤8 / DiT 늦은 t≥12, 후기 block 23~31; "같은 t에서 둘 다 재지
  말 것"), 두 실패 regime(초기조건형 frame0부터 vs 실행표류형 mid-rollout → 후자가 개입 여지 큼),
  cross-task 실패 공유 zone, 직렬 confound(Eagle→VL-SA→DiT) 첫 노출.
- **(B)** `docs/steering/08_pathway_separation_analysis.md`(정독),
  `docs/steering/15_*` regime/유형 섹션. 외부: NOTALL 재독(pathway 기능분리 근거).
- **(C)** `scripts/safe/groot_n16/robocasa/analyze/pathway_separation.py` 재현(VL early / DiT late
  분리력 곡선). `analyze/within_task_failure_modes.py` 또는 `analyze/failure_trajectory_modes.py`로
  두 regime 분포 재현.
- **(D)** VL/DiT 분리력 곡선 + 실패 regime 비율 표. **M3.**
  Q: "DiT-only-OOD가 거의 0으로 보이는 게 왜 '창(window) 과소측정' 아티팩트일 수 있나?"

### Phase III — conceptor method 이해 + steering eval 재현 (W8~10, 무게중심 ②, ③ 진입)

#### W8 — conceptor 수학 + Boolean algebra
- **(A)** `R`/conceptor/aperture/eigen, Boolean(¬/∧/∨) + contrastive conceptor
  `C_steer = C_s ∧ ¬C_f`. covariance/eigen 기초만 `[SKIP if known]`; conceptor 자체는 신규이므로 skip 불가.
- **(B)** `src/conceptor/README.md`(수식↔COAST 섹션 매핑 정독), `src/conceptor/core.py`
  (`compute_conceptor` / not·and·or / `contrastive_conceptor`), `src/conceptor/analysis.py`.
  외부: **COAST**(`docs/references/COAST.pdf`, 요약 `docs/references/COAST.txt`) Sec 3.1/Eq.4 +
  Jaeger 2014 conceptor Boolean algebra 개념.
- **(C)** `src/conceptor/README.md`의 toy 예제(성공/실패 랜덤)로 `contrastive_conceptor` 실행 +
  `conceptor_quota` / `conceptor_overlap` / `eigenvalue_spectrum` 출력 해석. aperture α를
  로그스케일로 sweep 해 overlap이 최소가 되는 지점 찾기.
- **(D)** α sweep vs overlap 그래프 + "C_steer가 성공방향은 보존하고 실패방향만 죽인다"를
  eigenvalue로 설명. Q: "C=R(R+α⁻²I)⁻¹에서 α를 키우면 무슨 일이 일어나나?"

#### W9 — steering gate + 수치 주의 + 실제 feature로 fit
- **(A)** `M=(1-β)I+β·C_steer`, `h'=h·Mᵀ` (곱셈형 gate). 수치 주의(float64 계산/float32 저장,
  N≪d라 pinv, 대칭화), β 선택.
- **(B)** `src/conceptor/steering.py`(`build_steering_matrix` / `apply_steering`),
  `src/conceptor/README.md` §5 수치 정밀도 + §6 hyperparameter,
  `docs/steering/07_steering_methods_survey.md`(conceptor가 현 방식인 이유 + 후보들).
- **(C)** `scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py` 흐름을 따라 실제 수집된
  succ/fail pkl에서 z_mean 풀링 → `C_steer` fit → steering matrix 저장. (수집 캐시가 있으면 그걸
  쓰고, 없으면 `collect/collect_rollout.py` + `split/prepare_seen4_unseen2_split.py` 산출물을
  멘토에게 받는다. 대용량 수집/분석은 remote.)
- **(D)** 실제 GR00T feature로 만든 `steering_matrix.npy` 1개 + quota/overlap 진단값.
  위험: N≪d(예 N=15, d=1024) singular → pinv를 써야 하는 이유, float32로 fit하면 깨지는 이유.

#### W10 — steering 사다리 ΔSR + SAFE-LSTM detector  · **[Milestone M4]**
- **(A)** 사다리식 ablation(global → pathway-split → +phase-bin), 각 단계 ΔSR.
  eval operating-point 아티팩트 경고(초기 음수 ΔSR이 method 실패가 아니라 eval 설정 아티팩트였던
  사례).
- **(B)** `docs/steering/14_pathway_phase_online_steering.md`(메인 thesis 정독 — 사다리 ablation·
  open problem), `docs/steering/15_*` C2(조종, mean ΔSR +0.114) 섹션,
  `docs/groot/n16_07_safe_detector_report.md`.
- **(C)**
  1. `scripts/safe/groot_n16/robocasa/steer/eval_steer_compare.sh`(또는 `eval_steer_vl.sh`)로
     baseline vs global-steer ΔSR 1단계 재현(표준 EVAL_SEED=100000, 로컬).
  2. `analyze/pathway_lstm_detector.py` + `analyze/verify_detector_length_control.py`로
     SAFE-LSTM detector AUROC를 **length-control 하에서** 재현.
- **(D)** 사다리 1~2단계 ΔSR 표 + length-fair detector AUROC. **M4.**
  Q: "ΔSR이 음수로 나왔을 때 method가 틀린 건지 eval 설정이 틀린 건지 어떻게 구분하나?"

### Phase IV — 미해결 문제 탐색 + 마무리 (W11~12, 무게중심 ③ = stretch)

#### W11 — ★online 식별 + 라우팅 (stretch, 보너스)
- **(A)** **online phase/failure-type 식별**(중심 미해결 문제 — 안 되면 라우팅 불가),
  serial pathway crossover 검증, conformal prediction / VITA progress(보조 phase 신호),
  라우팅 crossover(C4) 설계 + falsification 조건.
- **(B)** `docs/steering/14_*` open problem + ablation 사다리 재독,
  `docs/steering/15_*` §4 검증설계(RQ3/RQ4 crossover)·falsification,
  `docs/insight/02_progress_prediction.md`. 외부: **VITA**(`docs/references/VITA.pdf`,
  progress predictor), **Path-Deviation-Heads**(`docs/references/PathDeviationHeads_2603.13782.pdf`,
  경쟁자 델타), 여유되면 `docs/references/I-FailSense_2509.16072.pdf`,
  `docs/references/Steerable VLAs.pdf`, `docs/references/Mechanistic Interpretability for Steering.pdf`.
- **(C)** (택1, 작은 범위) `analyze/pathway_online_detection.py` 또는
  `vis/compute_and_plot_online_detection.py`로 online detector cross-task 일반화(예 DiT block31
  late window) 재현 시도. LOO/unseen holdout 난이도 caveat 확인.
- **(D)** online detection 재현 결과 1개 + "이 비대칭/AUROC가 진짜인가, both==dit 스케일/holdout
  너무 쉬움 같은 caveat은 무엇인가" 메모. (재현만 성공해도 통과. 새 method 제안은 보너스.)

#### W12 — 종합 정리 + 발표 + 핸드오프  · **[Milestone M5]**
- **(A)** 전체 서사 재구성(분리 C1 → 조종 C2 → 유형 C3 → 라우팅 C4), 재현 가능성·confound
  통제가 이 연구의 핵심 자산임을 내면화.
- **(B)** `docs/steering/13_lab_meeting_ppt_notes.md`(발표 덱 구성·그림↔수치 출처 매핑),
  `docs/steering/README.md` 결과 위치 표.
- **(C)** 12주간 본인이 재현한 figure/표를 한 덱으로 묶기(길이 confound → fixed-t AUROC/permutation
  → pathway 분리 → LDA held-out → conceptor fit → 사다리 ΔSR → online detection). 본인 산출물
  경로를 README 스타일로 정리.
- **(D)** 랩미팅 발표 1회 + 재현 노트(어떤 스크립트→어떤 산출물→어떤 결론, confound 통제 포함).
  **M5.** Q: "이 연구에서 '길이/instruction confound 통제'가 왜 단순 디테일이 아니라 핵심 기여인가?"

---

## 3. Milestone / 통과 기준

| M | 시점 | 객관적 통과 기준 |
|---|---|---|
| **M1 인프라** | W2 | 서버 `/health` 200 + `/act_with_features` hidden_states shape 확인 + RoboCasa eval 1회 완주(SR 산출, EVAL_SEED=100000) |
| **M2 길이 게이트** | W4 | `seen18.py` figure 재현 + time-pooled AUROC≈0.998 직접 재현 후 fixed-t 하락 확인 + "왜 가짜인가" 구두 설명 |
| **M3 표현분석** | W7 | fixed-t within-task AUROC + permutation p + pathway 분리 곡선 + **held-out** LDA(in-sample 대비 수치화) + instruction-balanced 필터 |
| **M4 conceptor+steer** | W10 | 실제 feature `C_steer` fit(quota/overlap) + steering matrix 저장 + 사다리 1단계 ΔSR + length-control detector AUROC |
| **M5 종합/stretch** | W12 | online detection 재현 시도 1개 + 전 구간 재현 노트 + 랩미팅 발표 |

게이트: **M2(W4)** 와 **W6(instruction/LDA 과적합)** 는 멘토 sign-off 후 진행. 이 둘을 건너뛰면
이후 분석이 전부 오염된다.

---

## 4. 읽기 리스트 (우선순위)

### 외부 논문
- **P0 (W1~8, 필수)**: SAFE → NOTALL → COAST. `docs/references/`의 `.txt` 요약 먼저 → PDF 정독.
  세 편이 우리 연구의 SAFE(분리)·NOTALL(pathway)·COAST(conceptor 조종) 토대.
- **P1 (W9~11)**: COAST 수학 deep(Sec 3.1/Eq.4), VITA(progress predictor = 보조 phase 신호).
- **P2 (W11~12, 선택)**: Path-Deviation-Heads(경쟁자 델타), I-FailSense, Steerable VLAs,
  Mechanistic Interpretability for Steering.
- 배경(선택): robocasa365, Scaling World Model, CoT-VLA.

### Repo 핵심 문서 (읽기 순서)
1. `docs/README.md` → `CLAUDE.md` → `docs/00_repo_orientation.md` → `docs/01_serving_interface.md` (인프라)
2. `docs/groot/00_groot_flow_map.md` + `docs/groot/00_groot_steering_explorer.html` (GR00T 흐름)
3. `docs/steering/14_pathway_phase_online_steering.md`(메인 thesis, SSOT) → `docs/steering/15_research_structure.md`
4. `docs/steering/01_seen18_latent_analysis.md`(길이 confound 전제) → `08_pathway_separation_analysis.md`
   → `07_steering_methods_survey.md` → `11_instruction_confound.md`
5. `src/conceptor/README.md` (conceptor 수학 ↔ COAST 매핑)
6. `docs/groot/n16_03_safe_overview.md`, `n16_06_safe_inference_semantics.md`,
   `n16_07_safe_detector_report.md`, `docs/steering/13_lab_meeting_ppt_notes.md`

---

## 5. 함정 지도 (어디서 가르치나)

| 함정 | 증상 | 주차 | 방어 |
|---|---|---|---|
| **길이 confound 무시** | time-pooled AUROC 0.998을 "분리됐다"로 착각 | **W4 게이트** | fixed-t truncation, 0.998 직접 재현 후 통제 |
| 약신호를 시간/진행도로 재측정 | within-task 신호가 사실 길이 proxy | W5 | permutation test, 시간축 직교 확인 |
| **instruction confound** | VL AUROC 부풀림(slide in/out 쏠림) | **W6 게이트** | instruction-balanced task만 신뢰, 사분면 분석 |
| **in-sample LDA 과적합** | 전체 fit→전체 평가로 "분리 잘됨" 착각 | **W6 게이트** | train/test split·CV, in-sample vs held-out 차이 수치화 |
| pathway 직렬 crossover | "DiT-only-OOD 없음"을 성급히 결론 | W7, W11 | 같은 t에서 둘 다 재지 않기, VL early/DiT late 창 보정 |
| conceptor 수치 폭발 | N≪d singular, float32 fit | W9 | float64 계산·pinv·대칭화 |
| eval operating-point 아티팩트 | ΔSR 음수를 method 실패로 오판 | W10 | 표준 seed/N_ENVS/N_EP 고정 |
| detector caveat | both==dit 스케일 미분리, holdout 너무 쉬움 | W11 | LOO/unseen holdout, length-fair 검증 |
| 경로 하드코딩 / eval을 remote에서 | 재현 깨짐 | W1~2 | path_setup/cache_env, eval 로컬·분석 remote 규칙 |

---

## 부록 A. 배경 개념 사전 (Tier 0 build-up 시 참조)

- **ML/통계**: confound·survivorship bias, AUROC(threshold-sensitivity·saturation),
  permutation test(null·p-value), cross-validation(in-sample vs held-out), PCA/LDA,
  covariance·eigen·PSD·pseudo-inverse(Moore-Penrose), whitening(circularity 주의).
- **모델**: transformer/attention/residual stream, vision-language model, diffusion/flow matching
  (denoising step K), VLA pipeline(Eagle VLM → VL-SA → DiT action expert).
- **로보틱스**: EEF position/rotation/gripper, relative vs absolute action, rollout/success rate,
  RoboCasa kitchen task.
- **소프트웨어/DevOps**: REST API·endpoint·payload schema, Docker/compose·volume·network,
  SSH/rsync·git 동기화, `sys.path`/`PYTHONPATH`.

각 개념은 해당 주차의 (B) 읽기에서 연구 맥락과 함께 다시 등장한다. 사전지식이 있으면
`[SKIP if known]` 표시 구간을 건너뛰고 분석(Phase II) 주차에 시간을 더 쓴다.
