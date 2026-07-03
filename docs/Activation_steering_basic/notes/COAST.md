# Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States (Miao, Kim, Yang & Ungar 2026)

- 출처: Miranda Muqing Miao, Subin Kim(공동1저자), Brandon Yang, Lyle Ungar (University of Pennsylvania) · arXiv:2605.17144v1 [cs.RO] (2026-05-16) · PDF: docs/references/COAST.pdf · 섹션=§3 Activation Steering with Contrastive Conceptor(방법, 전체 정독) + §4.2-4.4(실험·기전·transfer) + §5(한계/결론) — **배정된 "§6"은 실물 논문에 없음**(실제 구성: §1 Intro–§2 Related Work–§3 Method–§4 Experiments–§5 Limitations/Broad Impact/Conclusion + References + Appendix A/B; 아래는 §3 정독에 Appendix A.3(ablation)·A.7-A.9(추출·수집·연산 디테일)를 보강해 작성) · tier=must★★★ · 한줄역할: 우리 headline 연산자(C_steer=C_success∧¬C_failure, h'=h·Mᵀ, M=(1−β)I+β·C_steer)를 VLA action expert에 처음 이식한 **직접 선행연구** — conceptor 수학은 그대로 빌리고, COAST가 시간축을 denoising-step으로만 쪼갤 뿐 rollout-phase는 통째로 pool하는 지점이 우리가 메우는 gap.

## 문제·동기
VLA는 웹스케일 VLM pretrain의 지각 prior를 물려받지만 실제 로봇 태스크에서 자주 실패한다. 저자들은 "decoding bottleneck" 가설을 세운다: frozen VLM backbone이 만든 조건화 표상은 이미 태스크 이해를 담고 있는데, 별도 학습된 action expert(flow-matching/AR decoder)가 최종 action으로 디코딩할 때 이 정보를 충분히 활용하지 못한다는 것. 기존 해법(fine-tuning, contrastive representation regularization)은 재학습 비용이 크고 "이미 배운 능력의 활용 극대화"라는 목표와 개념적으로 어긋난다. LLM 쪽 activation steering(ActAdd/CAA, 이를 subspace로 확장한 Zou 2025·Postmus & Abreu)은 재학습 없이 유사 문제를 풀었지만, VLA 적용 선행연구(Häon 2025의 FFN 뉴런, SAE 계열)는 backbone feature 단위 개입에 그쳐 종합적인 task 성공률 개선을 보고하지 못했다.

## 핵심 아이디어
Jaeger(2014)·Postmus & Abreu(2024)의 conceptor 대수를 action expert의 residual stream에 최초 이식. 성공/실패 rollout에서 각각 conceptor C_success, C_failure를 fit하고 Boolean AND-NOT으로 C_steer=C_success∧¬C_failure를 합성 — "성공에는 있고 실패에는 없는" 다차원 판별 subspace만 남긴다. 추론 시 이를 곱셈 게이트로 residual stream에 적용, identity와 β로 블렌딩해 개입 강도를 조절. gradient·외부 VLM·multi-sample search 없이 15개 rollout만으로 closed-form fit.

## 방법(정밀)
- **Conceptor(식1)**: min_C (1/N)‖X̃−X̃C‖²_F + α⁻²‖C‖²_F → 닫힌해 C=R(R+α⁻²I)⁻¹, R=X̃ᵀX̃/N (mean-centered 공분산, X̃는 평균제거 활성화행렬). 고유값 관계(식2) μ_i=λ_i/(λ_i+α⁻²): 고분산(λ_i≫α⁻²) 방향은 μ_i≈1로 거의 통과, 저분산(λ_i≪α⁻²) 방향은 0으로 억제 — soft variance-adaptive filter.
- **Boolean 대수**: ¬C=I−C(aperture 무관 정확식), A∧B=(A⁻¹+B⁻¹−I)⁻¹(식3, near-singular 대응 위해 Moore-Penrose pseudoinverse 사용). 대조 conceptor(식4) C_steer=C_success∧¬C_failure — 두 outcome이 공유하는(=outcome-irrelevant) 방향은 상쇄되고 판별에 필요한 얇은 잔차 subspace만 남는다.
- **활성화 추출**: action expert 내부 단일 layer ℓ에서 residual stream을 뽑아 action-token 축으로 mean-pool → 스텝당 벡터 h∈R^d 1개. Rollout Collection Protocol(App. A.8)은 "activation data ... saved at every inference step"이라 명시 — 즉 episode 내 **매 env-step(재계획 호출)마다** 저장되고, 그 안에서 다시 매 **denoising step**(π0.5: Euler 10-step, 식6 x_{k+1}=x_k+Δt·v_θ(x_k,t_k))마다 h가 하나씩 나온다. 15개 fitting rollout(성공+실패 혼합)의 모든 env-step×denoising-step h를 통째로 쌓아 X=[h₁;...;h_N]∈R^{N×d}, 여기서 conceptor(식1)를 fit — **이 집계 과정에 rollout 내 "어느 시점(phase)"인지 라벨이 전혀 없다.**
- **곱셈 게이트(식5)**: M=(1−β)I+β·C_steer, h'=h·Mᵀ. β=0→무개입, β=1→conceptor 전체 적용. 실측 최적 β는 grid cell의 90% 이상에서 0.1~0.3.
- **3가지 steering 전략(전부 denoising-step 축에서만 구분, rollout-phase 축과는 무관)**: (1) Global — 전체 denoising step(및 전체 rollout env-step)을 하나의 R로 pool, 매 step 동일 게이트. (2) Per-step — denoising step 인덱스(0~9)별로 별도 C_steer를 fit(단, 각 denoising-step-conceptor 안에서도 rollout 전체 env-step은 여전히 뒤섞여 pool됨). (3) Positive-only — C_steer=C_success(대조항 제거), 대조 연산의 필요성 검증용.
- **하이퍼파라미터**: layer ℓ(π0.5 오라클 대부분 ℓ=11/18, LIBERO 일부 ℓ=5; GR00T DiT는 ℓ=10/16), aperture α, β — 15 fitting rollout으로 grid search(fitting-set SR 기준 선택) 후 30 held-out test rollout으로 평가. "conceptor quota"(layer 선택)·success-failure overlap(aperture 범위 축소)으로 오라클 성능의 93%를 grid의 3-8%만 평가해 재현하는 geometric heuristic도 제시(App. A.10.2).
- **연산비용**(App. A.9.3-A.9.5): fit은 d×d 역행렬 CPU 1코어 <1-3초/(task,layer,α)(π0.5 d=1024, GR00T d=1536), 추론 오버헤드 <0.1ms/denoising-step(A100, <1% wall-clock) — 완전 closed-form, gradient 없음.

## 실험·결과(좌표 기반 재추출로 Table 1/10 교차검증한 수치만)
- **MetaWorld ML45**(π0.5, flow-matching, 10 task 서브셋): Base 0.69 → Global/Per-step 0.94(+0.25, p<.001); pick-place-wall 0.20→1.00, stick-push 0.20→0.73.
- **LIBERO-10**(π0.5): Base 0.43 → Per-step 0.80(+0.37, p<.001, 전 벤치마크 중 최대 절대 gain); π0-FAST: 0.62→0.84(+0.23, p<.001; Cheese+Butter는 π0-FAST baseline 실패 0건이라 conceptor fit 불가, COAST 컬럼은 9-task 평균).
- **RoboCasa**(가장 어려운 세팅, 매 episode 레이아웃/오브젝트/텍스처/조명 랜덤): π0.5 Base 0.40 → Global/Per-step 0.55(+0.15, p<.01)/Pos-only 0.56(+0.16). GR00T N1.5 Base 0.59 → Global 0.75(+0.16, pz=.006). Diffusion Policy Base 0.32 → Per-step 0.46(+0.14).
- **실물 로봇**(DROID, Open Drawer/Close Microwave/Put Duck in Cabinet, 태스크당 15회, π0.5-DROID는 데모 파인튜닝 안 된 체크포인트): π0.5+COAST 평균 +40%p, Open Drawer 최대 개선, Close Microwave는 **성공 rollout 단 1개**만으로도 SR 46%까지 상승.
- **기전 분석(§4.3)**: C_success/C_failure 고유값 스펙트럼 급격히 감쇠 → C_steer 유효 차원 ≈ 전체 hidden dim(π0.5 d=1024)의 약 1%(low-rank지만 rank-1은 아님 — CAA 같은 단일벡터가 부분적 이득만 회수하는 이유). overlap sim(C+,C−)=tr(C+C−)/√(tr(C+²)tr(C−²))와 ΔSR의 Spearman ρ=0.59(p=0.002) — succ/fail subspace가 많이 겹칠수록 개입 여지 큼.
- **Cross-task transfer(§4.4)**: 실패 subspace는 태스크 간 공유되는 반면 성공 subspace는 태스크 고유(Fig.5 joint PCA). failure-containment tr(C_src^f C_tgt^f)/tr((C_src^f)²)와 transfer ΔSR 상관 r=0.30(LIBERO, p=4.5e-3)/r=0.49(RoboCasa, p=1.1e-3), success-containment는 무상관(|r|<0.13, p>0.4).
- **Ablation(Table 10)**: GR00T N1.5/RoboCasa — Base 0.59, **Global(contrastive conceptor) 0.75(+0.16)**, Linear(mean-diff additive, h'=h+α·v) 0.62(+0.03), **Random-direction(같은 고유값 스펙트럼·무작위 직교 고유벡터) 0.58(-0.02, 사실상 베이스라인과 동일)**. π0.5/MetaWorld도 동일 패턴(Base 0.58, Global 0.80, Linear 0.68, Random 0.55). → 이득의 원천이 "아무 subspace 필터링"이 아니라 succ/fail 판별 방향 그 자체임을 통제.

## activation-steering 흐름 위치(conceptor를 VLA에 이식)
ActAdd/CAA(단일벡터 additive) → Postmus & Abreu conceptor(soft projection, NLP) → COAST가 이 연산자를 VLA action expert(flow-matching/AR decoder) residual stream에 최초 이식. VLA steering 선행연구(Häon 2025 FFN 뉴런, SAE 계열)는 backbone feature 단위 개입에 그쳤던 반면, COAST는 action expert 내부에 직접 개입해 종합 task-SR 개선을 처음 보고. 우리 프로젝트의 표기 h'=h·Mᵀ, M=(1−β)I+β·C_steer는 이 논문 식5를 그대로 채택한 것.

## 우리 프로젝트 연결(빌리는 것=연산자 / 메우는 곳=phase pool·pathway 미분리)
- **빌리는 것 = 연산자.** C_steer=C_success∧¬C_failure, M=(1−β)I+β·C_steer, h'=h·Mᵀ는 이 논문(및 그 직계 선행 Postmus & Abreu, `docs/Activation_steering_basic/notes/Conceptors.md`)의 수식을 그대로 채택. layer 선택·aperture 튜닝 절차(quota/overlap heuristic)도 참고 가능.
- **메우는 곳 ① phase pool.** App. A.8("saved at every inference step")과 §3.2 전략 정의를 대조하면, COAST의 R=E[hhᵀ]는 **rollout 내 env-step(=task-phase) 축을 전혀 구분하지 않고 통째로 pool**한다. "Global"은 denoising-step까지도 pool하고, "Per-step"조차 **denoising-step(0~9, 한 action-chunk 예측 내부의 노이즈 제거 반복)만 나눌 뿐 rollout의 물리적 진행 시점(task-phase)은 여전히 섞는다** — 즉 COAST의 두 시간축(denoising-step vs rollout-phase)은 서로 다른 축인데 phase 축은 아예 다루지 않는다. 우리 phase-matched 확장 = phase-bin별로 R_phase를 따로 fit(같은 conceptor 수학, 집계 축만 phase로 세분).
- **메우는 곳 ② pathway 미분리.** COAST는 단일 layer ℓ 하나만(quota로 선택) 골라 그 layer의 residual stream 전체를 게이팅 — VL(goal) pathway와 DiT(motor) pathway를 구분하지 않는다. 우리 (1)pathway 분리 steering이 겨냥하는 지점.
- Table 10 random-direction ablation(같은 고유값 스펙트럼·무작위 고유벡터가 baseline과 동급)은 우리 conceptor 방법론의 타당성 근거로 재사용 가능하다 — "성공/실패를 실제로 판별하는 방향"이 이득의 원천임을 이미 통제된 형태로 보여준다.

## 면접 포인트(Q→A; COAST가 phase를 왜 놓치나)
1. Q: "COAST가 phase(task-execution 시점)를 왜 놓치는가?" A: "COAST의 conceptor는 R=E[hhᵀ]를 fit할 때, rollout collection protocol(App. A.8)이 명시하듯 매 env-step마다 저장한 활성화를 **어떤 시점 라벨도 없이** 15개 rollout 전체에 걸쳐 통째로 pool한다. 3가지 전략(Global/Per-step/Positive-only) 중 유일하게 시점을 나누는 'Per-step'조차 그 축은 **denoising-step(한 action-chunk 예측 내부의 Euler 노이즈 제거 0~9단계)**이지 **rollout의 물리적 진행 시점(task-phase)**이 아니다. 이름이 헷갈리기 쉬운 두 시간축을 혼동하지 말아야 하는데, COAST는 후자(phase)를 다루는 축 자체가 없다."
2. Q: "COAST가 rank-1 단일벡터 대신 conceptor(subspace)를 쓰는 근거는?" A: "고유값 스펙트럼이 급격히 감쇠하되 rank-1은 아니라는 실증(Fig.3A)에 더해, ablation(Table10)에서 Linear(mean-diff additive)가 Global(contrastive conceptor)보다 명확히 낮다(GR00T RoboCasa +0.03 vs +0.16). 무작위 방향(같은 고유값 스펙트럼, 무작위 고유벡터) 조건은 baseline과 거의 동일(-0.02)이라, 이득이 '아무 subspace 필터링'이 아니라 succ/fail 판별 방향 자체에서 온다는 것을 통제 실험으로 보여준다."
3. Q(우리 프로젝트): "COAST 연산자를 그대로 쓰면서 우리가 추가하는 것은 정확히 무엇인가?" A: "수식(C_steer, M, h'=h·Mᵀ)은 100% 동일하게 재사용한다. 차이는 R=E[hhᵀ]를 fit할 때 넣는 **X(활성화 집합)를 어떻게 자르는가**다 — COAST는 rollout 전체 env-step을 통째로(Global) 혹은 denoising-step으로만(Per-step) 잘라 pool하고, 우리는 여기에 rollout-phase bin 축을 추가해 phase별 R_phase를 따로 fit한다. 동시에 COAST는 단일 layer(action expert 내부)만 게이팅하는데, 우리는 VL pathway와 DiT pathway를 별도로 게이팅한다."

## 한계·비판
- Contrastive conceptor는 성공+실패 rollout이 모두 필요 — 실패가 0인(near-ceiling) 태스크는 positive-only로 폴백(저자도 App. A.4에서 인정). 우리 프로젝트처럼 실패가 항상 timeout인 경우 phase별로 더 잘게 쪼개면 phase당 표본이 COAST보다 더 희소해지는 문제가 심해진다.
- overlap sim(C+,C−)이 작은(이미 잘 분리된) 태스크는 개입 여지가 작다는 자기보고 한계(Fig.3B) — 스티어링 효과가 클수록 baseline이 나쁘다는 뜻이라 "이미 잘하는 태스크엔 별 도움 안 됨"은 우리 phase-matched 확장에도 동일하게 적용될 위험.
- geometric hyperparameter selection(quota/overlap heuristic)이 테스트된 아키텍처(π0.5, GR00T DiT, Diffusion Policy) 밖으로 일반화되는지는 저자 스스로 미검증(§5 Limitations).
- Table 1의 π0-FAST/LIBERO Base·∆ 수치는 Cheese+Butter 태스크(π0-FAST baseline 실패 0건이라 conceptor fit 불가)를 COAST 컬럼에서만 제외한 9-태스크 평균이라, 표에 보이는 "Base" 컬럼(10-태스크 평균)과 ∆ 컬럼(9-태스크 기준)이 서로 다른 모집단이다(각주로만 명시, 표 자체에는 구분 없음) — 수치 인용 시 주의.
- Broader Impact 섹션 스스로 인정: steering은 action trace만으로는 탐지 불가능해 안전장치 없이는 동일 연산자가 실패 방향으로도 이식 가능(dual-use). 저자는 배포 시 적용된 conceptor를 정책 스펙에 로그로 남기고 output-level 모니터링 병행을 권고.
