# G. "Activation steering으로 로봇 SR을 올렸다"는 논문 전수 조사 (2026-08-10)

**질문**: 정책의 내부 활성(activation/hidden state/residual stream)에 **추론 시점**에 개입해 로봇
조작/제어 task의 **success rate(SR)**를 올렸다고 주장하는 논문이 (COAST·WA-LQR·Häon 외에) 더 있는가?

**포함 기준(3개 모두 만족)**: ① 개입 대상 = 모델 내부 표현(activation/hidden/residual/latent).
② 개입 시점 = 추론 중(inference-time). 백본 재학습·파인튜닝 제외. ③ 로봇 조작/제어 task의 SR
향상을 주장.

**검증 기준**: `[WebFetch]` = WebFetch로 abstract/HTML/PDF 원문 확인. `[로컬PDF]` = 이 repo의
`docs/Activation_steering_basic/`·`docs/references/`에 이미 있는, 과거 세션이 PDF 전문을 정독해
작성한 노트에서 재확인(표·수식 번호까지 인용돼 있어 실물 확인으로 취급). `[검색스니펫]` = WebSearch
결과 요약에서만 확인, WebFetch 직접 접근 실패 — **사실로 인용하지 않고 별도 표시**.

---

## 0. 결론 요약 (TL;DR)

1. **포함 기준을 모두 만족하는 논문은 4편 확정 + 1편 미확인(정황상 유력)** — COAST, WA-LQR, LAE,
   CTRL-STEER(신규 발견) + GuideVLA(미확인, OpenReview 접근 차단으로 원문 미검증).
2. **사용자가 이미 아는 "3편" 중 1편(Häon et al. CoRL 2025)은 실제로는 기준을 충족하지 않는다** —
   이 논문은 본문에 "success rate"라는 표현이 **단 한 번도 나오지 않는다**(우리가 과거 세션에서
   전문 검색으로 확인). displacement/height 같은 연속 행동량만 보고하고 SR을 주장하지 않는다.
   §3.1 참조. Steering 방법론 논문으로는 유효하지만 "SR을 올렸다"는 이번 조사 질문에는 해당 안 됨.
3. **위약(placebo/random-direction) 대조 + 통계적 유의성 검정을 모두 갖춘 것은 COAST 1편뿐**이다
   (random-direction ablation이 baseline과 동급임을 보이고 p-value 보고). LAE는 유의성 검정은
   있으나(paired t-test) placebo가 아니라 대안 방법 비교 위주. WA-LQR·CTRL-STEER는 둘 다 없음.
4. **"activation steering"이라는 용어를 쓰는 로봇 논문은 훨씬 많지만(20편 이상 스캔), 대다수는
   개입 대상이 내부 activation이 아니다** — diffusion denoising의 noise/velocity 예측(VLS,
   DynaGuide, PPGuide, DSRL), action-space 재조합/재선택(RL2-VLA, TACO, DREAMSTEER, Action Token
   Intervention), 명령/프롬프트 추상화(SteerableVLAs/InSight, ReSteer, Häon 후속 다수),
   파인튜닝·RL 재학습(ZPRL, Object-Centric Residual RL, ReSteer) 등 "제외 기준"에 해당한다.
   상세는 §4.

---

## 1. 포함 기준을 만족하는 논문 (확정 4편)

### 1.1 COAST — Contrastive Conceptor Activation Steering

| 항목 | 내용 |
|---|---|
| 서지 | Miao, Kim(공동1저자), Yang, Ungar (Univ. of Pennsylvania) · **arXiv:2605.17144v1** [cs.RO] (2026-05-16) · preprint |
| 검증 | `[로컬PDF]` `docs/references/COAST.pdf` 전문 정독 완료(과거 세션, 표1/10 좌표 재추출로 교차검증) |
| 개입 지점 | VLA **action expert 내부 단일 layer**의 residual stream (grid search로 선택: π0.5 대부분 ℓ=11/18, GR00T DiT ℓ=10/16). action-token 축으로 mean-pool한 스텝당 벡터 h |
| 개입 연산 | **곱셈형 soft projection**. M=(1−β)I+β·C_steer, h'=h·Mᵀ. β는 grid cell 90%+ 에서 0.1~0.3 |
| 방향 도출 | **자연 성공/실패 rollout 대조**. C_success, C_failure를 각각 conceptor로 fit(닫힌해 C=R(R+α⁻²I)⁻¹) 후 Boolean 대수로 C_steer=C_success∧¬C_failure. 단 R=E[hhᵀ]가 rollout 내 시점(phase) 구분 없이 전체 env-step×denoising-step을 통째로 pool(우리 프로젝트가 지적하는 결함) |
| SR 수치 | MetaWorld ML45(π0.5) 0.69→0.94(**+0.25**); LIBERO-10(π0.5) 0.43→0.80(**+0.37**); π0-FAST 0.62→0.84(+0.23); RoboCasa(π0.5) 0.40→0.55/0.56(+0.15/+0.16); **GR00T N1.5 RoboCasa 0.59→0.75(+0.16)**; Diffusion Policy RoboCasa 0.32→0.46(+0.14); 실물 DROID 3-task 평균 **+40pp** |
| 실험 규모 | 15개 fitting rollout(fit) + 30개 held-out test rollout(eval, fit-set과 분리). 4개 벤치마크 × 3~4개 아키텍처(π0.5 flow-matching, π0-FAST AR, GR00T N1.5, Diffusion Policy) |
| 대조군·유의성 | **있음(4편 중 가장 엄격)**. Table 10: Random-direction ablation(C_steer와 **같은 고유값 스펙트럼**이지만 고유벡터는 무작위 직교) → GR00T RoboCasa 0.58(사실상 baseline과 동급, −0.02), π0.5 MetaWorld 0.55(baseline과 동급). Linear(mean-diff additive) ablation도 별도 비교(Global 0.75 vs Linear 0.62). 본문 수치는 p<.001/p<.01/pz=.006 등 p-value 명시 |
| 모델·벤치마크 | π0.5(flow-matching), π0-FAST(autoregressive), GR00T N1.5(diffusion action head), Diffusion Policy — MetaWorld ML45, LIBERO-10, RoboCasa, 실물 DROID |
| 비고 | **우리 프로젝트가 이 연산자를 직접 재현 시도**(exp2/exp3, 2700+ rollout, 위약 포함 사전등록 6-Holm)했으나 **전면 null**로 나왔다(project memory: `conceptor-steering-final-verdict`). 논문 자체의 방법론적 엄격성(위약+유의성+held-out)은 이번 조사 기준에서 가장 높지만, 독립 재현에서 재현 실패했다는 점은 이 노트의 범위 밖 사실로 별도 기록해 둔다. |

### 1.2 WA-LQR — Steering Robustness into World Action Models

| 항목 | 내용 |
|---|---|
| 서지 | Hong*, Skifstad*, Dai, Chan, Chou (Georgia Tech, Trustworthy Robotics Lab) · **arXiv:2607.14943** (2026-07-16) · **RSS 2026 Robot World Models Workshop 채택(워크샵, 메인 트랙 아님)** |
| 검증 | `[로컬PDF+코드]` 저자 공개 코드(`github.com/trustworthyrobotics/steering_robustness_WAMs`) + 논문 전문 분석 완료(과거 세션) |
| 개입 지점 | World Action Model(비디오+action 통합 DiT) **28개 DiT block 전부**의 residual stream 출력, 3-partition(0-9/10-19/20-27)으로 관리 |
| 개입 연산 | (a) **ActAdd**: 층별 diff-of-means 벡터를 그대로 더함(open-loop, 매 denoising step). (b) **WA-LQR**: k=64 부분공간 SVD 위에서 층간 Jacobian 기반 LQR 피드백 제어(closed-loop, self-gating — setpoint 도달 시 자동 감쇠), chunk 진행에 따라 R이 exp 증가(개입 강도 지수 감쇠) |
| 방향 도출 | **clean-vs-perturbed 대조**(자연 성공/실패 아님). 카메라·노이즈 교란은 state-matched(같은 MuJoCo state, 교란 유무만 다름). **gripper 교란만 예외적으로 succ/fail rollout을 outcome으로 버킷팅**해 사후 재짝짓기 — 사실상 outcome 대조가 섞여 있음 |
| SR 수치 (Cosmos-Policy 2B, 교란 하) | 카메라 46.0→59.3(**+13.3pp**, WA-LQR); gripper 61.3→72.7(**+11.4pp**, WA-LQR); 노이즈 26.7→67.3(**+40.6pp**, ActAdd가 WA-LQR(58.7)보다 우세); DiT4DiT gripper 65.7→71.7; **LingBot-VA는 거의 무효과**(사전 선형분리도와 상관 r≈−0.7) |
| 실험 규모 | LIBERO-10, 3개 World Action Model(Cosmos-Policy 2B, DiT4DiT, LingBot-VA), 교란 3종 × 30 trial/task, task 간 전이(collect task 1개→적용 task 다른 것) |
| 대조군·유의성 | **없음** (코드 전체 grep으로 확인 — "random"은 교란 seed 변경뿐, random-direction 없음). mean±std over 30 trial만, p-value 없음 |
| 모델·벤치마크 | Cosmos-Policy 2B / DiT4DiT / LingBot-VA(World Action Model, video+action DiT) — LIBERO-10 |
| 비고 | 이득은 전부 **교란된 입력에서** 측정(nominal 입력 SR 개선 증거 아님). PI(Glen Chou)는 2024-11 부임 신진 조교수, 워크샵 채택 수준 — 증거 등급 낮게 취급 필요. |

### 1.3 LAE — Latent Activation Editing (multirobot navigation)

| 항목 | 내용 |
|---|---|
| 서지 | Das, Chiu, Huang, Lindemann, Sukhatme (USC / ETH Zürich) · **arXiv:2509.20623v2** (2026-06-01 갱신) |
| 검증 | `[로컬PDF]` `docs/references/LAE_LatentActivationEditing_2509.20623.pdf` 전문 정독(과거 세션) |
| 도메인 주의 | **VLA/조작이 아니라 decentralized multi-quadrotor 충돌회피 RL 정책** — task는 "제어"(navigation)이지 manipulation이 아님. 조사 지시문의 "로봇 조작/제어 과제"에서 "제어" 쪽에 해당해 포함 |
| 개입 지점 | 정책 내부 fused latent **Z1**(obs-encoder 출력, d=30) — self/neighbor/obstacle attention-fusion 직후, downstream MLP 이전 |
| 개입 연산 | **예측적 치환(predictive replace)**, 벡터 덧셈이 아님: online 이진 분류기 B_w가 Z_t를 unsafe로 flag하면, 학습된 GRU world model(LCWM)이 예측한 "n-step 후 latent"로 Z_t를 통째로 교체. safe면 무개입 passthrough |
| 방향 도출 | 데이터 기반이지만 succ/fail 대조가 아니라 **time-to-collision 라벨**(사후 hindsight)로 지도학습한 이진 분류기 + 별도 forecasting model(LCWM, MSE loss) |
| SR 수치 | 2600개 config(base policy가 최소 1회 충돌하는 config만) 기준: base RL policy SR **0.58** → LCWM-edited SR **0.64**(+10.3% relative); 충돌 5,623→583(−89.6%) |
| 실험 규모 | 2600 env config, 비결정론 시뮬레이터 10회 반복 재확인, 실물 Crazyflie 2.1(4-드론 crossing) 별도 검증 |
| 대조군·유의성 | **paired t-test 평균감소 1.94/run, 95% CI [1.86,2.01], p<1e-300, Cohen's d=1.0**. 대안 5종(KD-tree retrieval, SAE steering, UMAP/Barlow-Twins/AE 압축 편집, Transformer-LCWM) 전부와 비교해 GRU-LCWM이 최상 — random-direction/무개입 placebo는 아니지만 대안 방법론 비교는 충실 |
| 모델·벤치마크 | 저차원(d=30) MLP+attention RL 정책 — QuadSwarm sim + 실물 Crazyflie |
| 비고 | 편집 대상을 잘못 잡으면(self-dynamics 성분까지 편집) 67,951충돌로 **폭발**하는 ablation도 보고 — "일부 latent만 선택적으로 편집해야 한다"는 경고. VLA로의 일반화는 저자도 future work로만 언급, 미검증. |

### 1.4 CTRL-STEER — Closed-Loop Neural Activation Control in VLA (신규 발견)

| 항목 | 내용 |
|---|---|
| 서지 | Babu, Kaur, Bastian, Kotevska, Jha, Wu, Jha, Roy · **arXiv:2606.00269** (2026-05-29 제출) · **CVPR 2026 Workshop on Visual Concepts (VisCon) 채택(워크샵)** |
| 검증 | `[WebFetch]` arXiv HTML 전문 확인(2026-08-10) |
| 개입 지점 | OpenVLA-7B(Prismatic VLM, LLaMA-2 백본)의 **모든 FFN layer**에서, motion 개념(vocab logit-lens 투영으로 선정)과 정렬된 **10개 뉴런**의 활성값 |
| 개입 연산 | Häon et al. 2025의 정적 클램프(활성값을 고정 α로 override)를 **온라인 폐루프로 확장**. 오차 e^t=목표값−현재값에 대해 (a) **PID**: α^t = K_P·e^t + K_I·Σe^τ + K_D·(e^t−e^{t−1}), α∈[0,20]; (b) **RL(PPO)**: task별 학습, 상태=[a_t, Δa_t, α^{t−1}, t/T], 보상 r_t = r_steer(t) + λ·r_task(PID로 warm-start) |
| 방향 도출 | **succ/fail 데이터 대조가 아니라 concept-word 기반**(Häon과 동일 계보 — FFN value vector를 vocab에 투영해 "fast"/"height"류 개념 뉴런 선정). 목표는 특정 행동 성질(height/speed) setpoint 도달이지 "성공 방향"이 아님 |
| SR 수치 (LIBERO 4-suite 평균, OpenVLA) | **Height**: baseline 71.37% → Static(C=20) 27.37%(붕괴) → PID 71.00%(거의 중립) → **RL 73.88%(+2.51pp)**. **Speed**: baseline 71.37% → Static 1.88%(거의 전붕괴) → PID 72.50%(+1.13pp) → **RL 76.12%(+4.75pp)**. X-VLA 전이(LIBERO-Goal): baseline 59%→PID/RL 60% |
| 실험 규모 | LIBERO-Goal/Object/Spatial/Long 4-suite(각 10 task, Long은 libero-10 부분집합) + BridgeData V2(SimplerEnv) 보조. trajectory 길이 T=920(20 warm-up+900 실행). 정확한 trial/episode 수는 본문에 명시 없음(LIBERO 관행상 task당 50~100 추정) |
| 대조군·유의성 | **없음** — random-direction/placebo 없음, p-value·CI 없음, "0.832±0.249" 식 표준편차만. Static(C=20, Häon 방식) 자체를 강한 negative 대조로 삼아 "고정 강도는 파괴적, 폐루프는 아니다"를 보이는 것이 핵심 논증 |
| 모델·벤치마크 | OpenVLA-7B(주), X-VLA(전이 검증) — LIBERO 4-suite, BridgeData V2 |
| ⚠ 초록↔구현 불일치 | 초록은 "individual neurons 가정을 버리고 **motion-aligned residual directions** 로 steer한다"고 쓰지만, **"residual direction" 은 논문 전체에서 초록에 1회만 등장**하고 본문 §3.2.3 구현은 "all FFN layers … select **10 neurons**" — 즉 Häon 식 뉴런 클램프 그대로다(‘neuron’ 39회). 초록 문구만 보고 방법을 판단하면 오독한다 (2026-08-11 PDF 실물 확인). |
| 비고 | RL 컨트롤러의 보상에 **r_task(과제 성공)가 직접 포함**돼 있어 "성공 쪽으로 일부 직접 최적화된" 컨트롤러다 — 순수 관찰적 steering이라기보다 task-reward를 일부 흡수한 폐루프 게인 스케줄러에 가깝다. 또한 RL 컨트롤러는 **task별 학습이 필요**(zero-shot 아님), PID만 학습 불필요. Häon et al.(§3.1)의 직접 후속작으로 읽힌다 — "정적 클램프는 파괴적이니 온라인 피드백으로 강도를 조절하자"는 정확히 WA-LQR과 같은 문제의식을 OpenVLA/LIBERO에 적용한 것. |

---

## 2. 미확인 (정황상 유력하나 원문 미검증)

### 2.1 GuideVLA

- **제목**: "GuideVLA: Steering Vision-Language-Action Models with Soft Interventions for Out-of-Distribution Generalization"
- **출처(미확정)**: OpenReview `id=KnE4CfljqN`, 2026-06-07 무렵. **WebFetch 2회 시도 모두 OpenReview 봇 차단 페이지만 반환** — 저자/소속/venue/정확한 layer·수식을 원문에서 확인하지 못함. arXiv 단독 검색으로도 별도 ID를 찾지 못함(워크샵 전용 제출일 가능성).
- **검색엔진 요약 기반(★사실로 인용 금지, 참고만)**: "성공 rollout의 activation 패턴 평균을 내 방향으로 삼아 추론 시점에 내부 표현을 steer" — **단일 클래스(성공만) 평균, COAST 같은 성공/실패 대조가 아님**. 보고 수치(검색 스니펫): 미학습 물체 파지 성공 +5.7%, task completion +0.8%, 명령 재구성(rephrasing) 강건성 +3.5%/+1.4%.
- **판정**: 포함 기준(내부 표현·추론시·SR 향상 주장) 자체는 만족하는 것으로 보이나, **원문 미확인이므로 이번 조사의 확정 목록에는 넣지 않고 별도 표시**. 상세 모델·벤치마크·대조군 여부 불명.

---

## 3. "이미 아는 3편" 재검증 — 1편은 기준 미충족 (중요 정정)

### 3.1 Häon et al., CoRL 2025 — **SR을 주장하지 않음** (사용자 목록 정정 필요)

| 항목 | 내용 |
|---|---|
| 서지 | Häon*, Stocking*(공동1저자), Chuang, Tomlin (UC Berkeley EECS) · **arXiv:2509.00328v1** (2025-08-30) · **9th CoRL 2025 (Seoul)** |
| 검증 | `[로컬PDF]` `docs/references/Mechanistic Interpretability for Steering.pdf` 전문 정독 + **저자 공개 코드로 우리가 직접 재현(Phase A) 완료**(`docs/steering/16_mechinterp_reproduction.md`) |
| 개입 지점 | OpenVLA(Llama2-7B 백본) FFN down_proj / π0-FAST(PaliGemma-3B) FFN — **VLA의 base VLM 내부**, 별도 action expert 없음 |
| 개입 연산 | 선택된 FFN 뉴런의 활성값을 **고정 스칼라 α로 클램프(override)**(더하기 아님) |
| 방향 도출 | vocab logit-lens 투영으로 "fast"/"slow"/"low"/"high" 같은 **의미 개념 클러스터** 선정(succ/fail 데이터 대조 아님) |
| **왜 기준 미충족** | **논문 본문에 "success rate"라는 표현이 0회 등장**(우리가 전문 검색으로 직접 확인). 보고 지표는 전부 **end-effector displacement(mm)/height(cm)** 같은 연속 행동량. LIBERO-Long: fast/slow 클러스터 간 displacement +27.73%(평균), UR5 실물: low/slow 개입이 최저높이·속도를 낮춤(방향성은 확인). **하지만 "이 개입이 task 성공률을 올리는지/낮추는지는 이 논문 범위 밖"** — 저자 스스로도 성공률을 측정하지 않았다. |
| 대조군 | random-vector 개입 ≈ no-intervention(의미기반 방향 선택이 유효함을 통제) — 있음. 단 SR 지표가 없으니 "SR에 대한 위약 대조"는 애초에 성립하지 않음 |
| 우리 재현에서 추가 확인한 것 | 길이(episode length) 미통제 시 효과가 부풀려짐(+26.3%→길이통제 시 +15.7%, ~반감) — 논문이 보고하는 효과크기 27.73%의 상당부분이 길이 confound |
| 결론 | **활성값 개입 방법론 논문으로는 유효**하고 이후 COAST·CTRL-STEER가 명시적으로 인용하는 직접 선행연구이지만, **"SR 향상"을 주장하는 이번 조사의 포함 기준 ③을 충족하지 않는다**. 사용자가 "이미 아는 3편"에 넣은 건 방법론 계보상으론 정확하나, "SR 주장"이라는 이번 질문 기준으로는 정정이 필요. |

### 3.2 COAST, WA-LQR — 기준 충족 확인 (§1.1, §1.2 참조, 중복 서술 생략)

---

## 4. "개입은 있으나 기준 미충족" — 분류별 정리

### 4.1 활성 개입은 하지만 SR "증가"를 주장하지 않음 (같은 저자군 SAE 3편)

이 3편은 우리 프로젝트가 이미 전문 정독 완료한 논문들(`docs/references/reading_notes/dr_vla_sae.md`,
`event_grounded_sae.md`, `observing_controlling.md`, `docs/Activation_steering_basic/notes/NOTALL.md`)이며,
이번 조사에서 "SR 증가 주장" 여부만 재확인했다.

| 논문 | 개입 | SR 관련 실제 결과 | 왜 기준 미충족 |
|---|---|---|---|
| **NOTALL** (2603.19233, ICLR 2026) | 6개 아키텍처에 activation injection/ablation/boosting | boosting은 **항상 파괴적**(7배에서 −14%, 15배에서 −50.7%); "steering→SR 상승 positive 실험이 없음"이 논문 자체 한계로 우리가 이미 기록 | SR을 올리는 개입 사례 자체가 논문에 없음. fragility 진단이 목적 |
| **SAE_VLA_pi05 / Dr.VLA** (2603.19183, Swann et al.) | π0.5/OpenVLA SAE feature additive/ablative steering | ablation은 SR **붕괴**로 causal 검증(DROID 97.5%→0%), additive는 정성적 행동변화(object 선택 편향, gripper 개폐)만 — "steering→SR 상승을 직접 측정한 실험이 없다"고 **저자가 명시적으로 인정**(우리 노트 한계 섹션) | 논문 스스로 "이 실험은 아직 안 했다"고 인정 |
| **Event-Grounded SAE** (2605.17204, Jin et al.) | OpenVLA/π0.5 SAE feature zero-out/soft 개입 | event-aligned 랭킹 zero-out: SR **70.0%→48.8%(−21.2pp, 유의)** — causal 증거를 위한 **감소** 실험 | 목적이 "얼마나 깨지나"이지 "얼마나 올리나"가 아님. 증가 실험 없음 |
| **Observing & Controlling** (2603.05487, Buurmeijer et al.) | π0.5/OpenVLA 선형 observer+제약최적화 controller (gripper/height/speed) | gripper: SR **90%대 유지**(constraint 만족+SR 보존); height: 제약이 과제를 어렵게 만들어 **SR 소폭 하락**; speed: SR 거의 유지 | 목표가 "행동 성질을 목표구간에 두면서 SR을 보존"이지 "SR을 baseline보다 올리는 것"이 아님 — 정의상 다른 목표 |

### 4.2 "activation steering"을 표방하지만 실제 개입 대상이 내부 표현이 아님 (인접/제외)

WebSearch로 스캔한 20여 편 중 개입 지점을 확인한 것만 표로 정리. 전부 §0-④에서 말한 제외 사유
중 하나(action-space 합성/재선택, denoising noise/velocity guidance, 명령/프롬프트 레벨, RL
재학습·파인튜닝, human-in-the-loop)에 해당한다.

| 논문 | arXiv | 실제 개입 지점 | 제외 사유 | 보고 SR(참고용) |
|---|---|---|---|---|
| VLS | 2602.03973 `[로컬PDF]` | diffusion/flow denoising의 noise/velocity 예측(VLM reward gradient) | sampling-time, hidden state 아님 | LIBERO-PRO +13pt, CALVIN 87~94% |
| DynaGuide | 2506.13922 `[WebFetch]` | 별도 학습된 dynamics 모델의 denoising guidance | sampling-time | CALVIN 70%, goal-conditioning 대비 5.4배 |
| PPGuide | 2603.10980 `[로컬노트]` | obs-action MLP 임베딩 공간의 classifier guidance(action 차원에만 gradient) | policy 내부 latent 아닌 별도 임베딩 공간 + denoising 개입 | Stack D1 +2%, Transport +8% 등 |
| DSRL | 2506.15799 `[WebFetch]` | diffusion latent-noise 공간, **RL로 학습** | RL 학습 기반(추론시 고정개입 아님) | abstract 수치 없음 |
| TACO(anti-exploration) | 2512.02834 `[WebFetch]` | 샘플링된 action chunk 중 pseudo-count 최대인 것 선택 | action-space 선택(verifier형) | RoboTwin1.0 +9.1%, LIBERO(π0.5) +1.8% |
| DREAMSTEER | 2607.02865 `[WebFetch]` | action chunk 후보를 world-model+value-model로 순위매김·선택 | action-space 재선택 | 23.75%→66.25% |
| Action Token Intervention | 2606.15021 `[WebFetch]` | **디코딩된 action token을 사용자 조이스틱 입력으로 결정론적 치환**(Dirac delta) | 내부표현 아닌 출력 토큰 직접 치환 + human-in-the-loop(자율 아님) | drawer 10.0%→72.5%(Wilcoxon p=.003), sponge 16.7%→93.8%(p<.001) — **조사한 논문 중 유일하게 제대로 된 유의성 검정 보유, 그러나 기준 ①③ 모두 미충족** |
| Flow Reversal Steering | 2606.13675 `[WebFetch]` | latent noise 역산(reversal) + 일부 BC 재학습 | noise-space + 일부 재학습 혼재 | "최대 95% SR 향상" (근거 불명확, 미확정) |
| Do What You Say | 2510.16281 `[WebFetch]` | 여러 후보 rollout 샘플링 + VLM verifier 선택 | action-space/verifier | 최대 +15% |
| ReSteer | 2603.17300 `[WebFetch]` | 정책 재학습(steerable data로 self-improvement) | 파인튜닝 기반, 추론시 고정개입 아님 | steerability +11%(SR 아닌 별도 지표) |
| ZPRL(Beyond Action Residuals) | 2605.19919 `[WebFetch]` | 별도 VIB latent에 **RL로 학습된 residual** | RL 학습 + action 레벨에 조건화 | 실물 4-task +33.7% |
| Object-Centric Residual RL | 2606.18953 `[WebFetch]` | action-space residual(RL 학습) | RL 학습 + action-space | FR3 42%→76% |
| CTRL-STEER Static(C=20) | 2606.00269 (§1.4 참고) | (§1.4와 동일 지점이나 baseline 대조용) | — | 71.37%→27.37%(**파괴**, Häon류 정적 클램프의 한계 실증) |
| Learning What to Say to Your VLA | 2606.12299 `[WebFetch]` | **언어 프롬프트**(test-time language feedback) | 프롬프트 레벨, 내부표현 아님 | sim +24.7%, 실물 +65.0% |
| SteerVLA(주행) | 2602.08440 `[WebFetch]` | VLM reasoning→fine-grained 언어 지시 | 프롬프트 레벨 | 주행점수 +4.77, long-tail +8.04 |
| SteerVLM | 2510.26769 `[WebFetch]` | VLM(텍스트/이미지 전용) activation, **로봇 아님** | 도메인 자체가 로봇 제어 아님(VLM hallucination) | 로봇 SR 해당 없음 |
| InSight(SteerableVLAs) | 2606.24884 `[로컬PDF]` | 언어 프롬프트 세분화 + **LoRA 재학습** | 프롬프트+파인튜닝 | LIBERO 75%, 실물 xArm twist 92%/pour 96% |
| Steerable VLA Policies(Chen) | 2602.13193 `[WebFetch]` | 명령 추상화 레벨(subtask/motion/pixel) | 프롬프트/명령 레벨 | 수치 미확인 |
| VISTA(ScalingWorldModel) | 2602.10983 `[로컬PDF]` | world model이 생성한 discrete goal image로 하위 정책 조건화 | conditioning input, activation 아님 | OOD unseen object 14%→69% |
| Khan et al.(sparse latent dir) | OpenReview wtf3ww1EOL `[WebFetch 실패, 검색스니펫]` | Magma 모델 residual stream SAE steering vector | 내부표현이긴 하나 **단일 task 정성 데모, SR 벤치마크 수치 없음** | 없음("action 선택이 바뀜"만 정성 보고) |

### 4.3 로봇 도메인 아님 (기준 밖)

- **Policy Gradient Steering** (2607.27574) — abstract에 로봇 관련 내용 없음, 체스·경쟁 풋볼 사례만.

---

## 5. 조사 방법 메모

- 로컬 corpus: `docs/Activation_steering_basic/`(정독 52편, VLA §6 섹션 12편) + `docs/references/reading_notes/`(SAE 3편 통합·WA-LQR·RL2-VLA) + `docs/references/related_works_map.md`(2026-07-23 갱신 지도) — 이미 정독된 COAST/Häon/NOTALL/SAE 3편/WA-LQR의 표·수식 인용은 이 corpus에서 재확인.
- 신규 WebSearch: "activation steering robot manipulation success rate", "hidden state steering VLA inference-time", "residual stream intervention robot policy", "conceptor steering robot", "representation engineering robot control", "steering vector diffusion policy", "activation patching robot manipulation causal", "VLA activation steering real robot -COAST -Häon", "phase-conditioned activation steering VLA" 등 9개 쿼리 + 개별 논문 WebFetch 15회.
- **RL2-VLA**(2607.26991)는 이미 우리 프로�지트가 전문 분석 완료(`docs/references/reading_notes/rl2_vla_adaptive_steering.md`) — action-space velocity 합성+verifier 선별이라 이번 기준 ①(내부표현) 미충족, 이번 조사에서 재확인만 하고 표에서 생략(이미 알려진 것이므로 "신규 발견" 목록에서도 제외).
- **미탐사**: GuideVLA 원문(OpenReview 봇 차단), semantic scholar 인용 그래프(rate limit 429로 실패) — 시간이 더 있다면 GuideVLA의 정확한 모델/벤치마크/대조군 확인이 최우선 후속 작업.

---

## 6. 최종 집계

| 구분 | 편수 | 목록 |
|---|---|---|
| 포함 기준 3개 모두 확정 충족 | **4** | COAST, WA-LQR, LAE, CTRL-STEER |
| 미확인(유력) | 1 | GuideVLA |
| 위약 대조 + 유의성 검정 **모두** 있음 | **1 / 4** | COAST만(random-direction ablation + p-value) |
| 유의성 검정만 있음(위약은 대안-방법 비교로 대체) | 1 / 4 | LAE(paired t-test, 대안 5종 비교) |
| 위약·유의성 **둘 다 없음** | 2 / 4 | WA-LQR, CTRL-STEER |
| "이미 아는 3편" 중 기준 미충족 판명 | 1 | Häon et al.(SR 미보고) |
| 신규 발견(포함 기준 충족) | 1 | **CTRL-STEER** (2606.00269) |
| 신규 발견(미확인) | 1 | GuideVLA |
| 인접(제외, 표 4.2) | 19편 스캔 | VLS, DynaGuide, PPGuide, DSRL, TACO, DREAMSTEER, Action Token Intervention, Flow Reversal Steering, Do What You Say, ReSteer, ZPRL, Object-Centric Residual RL, Learning What to Say, SteerVLA(주행), SteerVLM, InSight, Steerable VLA Policies(Chen), VISTA, Khan et al. |
| 개입은 하나 SR 증가 미주장 | 4 | NOTALL, SAE_VLA_pi05, Event-Grounded SAE, Observing&Controlling |
