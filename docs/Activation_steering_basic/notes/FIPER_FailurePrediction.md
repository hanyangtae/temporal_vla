# Failure Prediction at Runtime for Generative Robot Policies (Römer et al. 2025, FIPER)

- 출처: NeurIPS 2025 · arXiv:2510.09459v2 [cs.RO] (TU Munich, Learning Systems and Robotics Lab) · PDF: docs/references/FIPER_FailurePrediction_2510.09459.pdf · 섹션=§7 Conclusions and Limitations(VLA 확장은 실험 없이 미래연구로만 언급) + §4 Methodology/§6 Results 정독 · tier=must · 한 줄 역할: 정책 observation-embedding의 RND 기반 OOD 점수 + action-chunk entropy를 실패데이터 없이 conformal 보정해 결합 — latent 기반 online 실패 조기예측의 대표 baseline/대조군.

## 문제·동기
Diffusion·flow-matching 기반 IL 정책은 대규모 데이터로 학습해도 분포 밖(OOD) 관측이나 action 오류 누적에서 예측불가 실패를 일으킨다. 실패 예측은 일반적 분류 문제로 풀 수 없다 — (1) 안전상 실패 예시를 의도적으로 만들기 어렵고, (2) 실패 모드의 범위가 사실상 무한해 라벨링이 불가능하다. 기존 접근은 두 갈래로 갈라져 각각 결함이 있다: (a) 순수 OOD 검출기는 정책이 실제로 일반화 가능한 새로운 상황에도 오탐한다(Success-OOD와 Fail-ID를 구분 못함), (b) VLM 기반 외부 모니터링은 실패가 이미 드러난 후에야 경보해 조기개입 여지가 없다. 가장 가까운 선행연구(FAIL-Detect, Xu et al. 2025)도 관측만 보므로, 미래 행동을 결정하는 action 분포 쪽의 조기 경고 신호를 놓친다.

## 핵심 아이디어
실제 task 실패는 (i) 관측이 성공 rollout 패턴에서 연속적으로 이탈(OOD)하는 것과 (ii) 생성된 action chunk 분포의 불확실성(entropy)이 지속적으로 높아지는 것이 동시에 나타난다는 관찰이 출발점이다. 두 지표 모두 실패 데이터 없이 소수(M=50, 실환경 M=10)의 "성공" rollout만으로 학습·conformal 보정하고, 둘 다 문턱을 넘을 때만(AND) 실패로 판정한다 — 이 결합이 "정책이 처리 가능한 OOD"(Success OOD)와 "진짜 실패"(Fail ID)를 구분하는 핵심 장치다.

## 방법 (RND 기반 embedding OOD score + action-chunk entropy, conformal threshold)
- **RND-OE(관측 OOD)**: 랜덤 초기화 후 고정한 target network g(·)와 학습되는 predictor f_theta(·)가 정책의 (freeze한) observation encoder h(·) 임베딩을 공유 입력으로 받는다. 성공 데이터에서 L(theta)=E[‖f_theta(O_t)−g(O_t)‖²]로 f_theta만 학습 — 본 적 있는 관측일수록 두 출력이 비슷해지고 novel 관측일수록 벌어진다(s_RND = ‖f_theta−g‖₂). 정책 자체의 사전학습 인코더를 재사용해 (a) 정책 임베딩 공간에서 직접 이상탐지하고 (b) 소량 데이터로도 RND 학습이 가능하다.
- **ACE(action-chunk entropy)**: 매 policy timestep마다 action chunk를 B개 배치 샘플링. 전체 chunk(H개 스텝)를 하나의 결합 히스토그램으로 잡으면 차원이 지수적으로 커지므로, 스텝별로 분리해 차원별 binning histogram entropy를 계산하고 H개 스텝을 합산(dimension-wise binning이 KDE 등 다른 entropy 추정보다 계산효율·안정성·튜닝 용이성에서 우수하다고 보고). eef Cartesian space에서 계산.
- 두 점수 모두 sliding window(w_O, w_A)로 최근 k step만 합산(전체 누적이 아님) → eta_O, eta_A. 문턱 γ_O,t·γ_A,t는 conformal prediction(functional-data band, Diquigiovanni et al.)으로 **성공 calibration set만** 사용해 시간가변으로 산출 — 1-δ 커버리지(성공 rollout을 실패로 오탐할 확률 상한)를 이론적으로 보장(Proposition 1). 최종 F(τ)=1(eta_O>γ_O,t ∧ eta_A>γ_A,t) — AND가 OR보다 TWA·정확도가 높음(OR은 TNR 0.05까지 붕괴, AND는 TNR 0.65면서도 실패의 91%를 여전히 검출).

## 실험·결과
- 5개 환경(SORTING·STACKING·PUSHT는 sim, PRETZEL·PUSHCHAIR는 실환경), diffusion(U-Net)·flow-matching(ACT transformer) 백본, ResNet-18 인코더.
- baseline 4종: PCA-kmeans(임베딩 클러스터 거리), logpZO/FAIL-Detect(flow-matching 우도), STAC(Agia et al. 2024, 연속 action-chunk 분포 divergence), RND-A(RediffUser 유사, action에 RND 적용).
- FIPER 평균 TWA(제안 지표, 조기 검출에 가중치) 0.65 / Acc 0.78 / DT 0.30 — 전 조합 중 최고. RND-OE·ACE 단독은 DT는 더 짧지만(더 빠름) 정확도가 낮음. RND-OE·ACE 각각 baseline(PCA-kmeans/logpZO, STAC/RND-A) 대비 Success-OOD와 Fail-ID 사이 점수 gap이 명확히 크다(Fig.4). sliding window 집계가 누적합(STAC식, rollout 길이 confound로 늦게 검출)이나 단일 timestep(logpZO식, TNR 붕괴)보다 견고함을 ablation으로 확인.

## activation-steering 흐름 위치(latent 기반 검출)
이 논문은 steering 자체가 아니라 "언제/누구에게 개입해야 하는가"를 정책의 latent(observation embedding)에서 online으로 읽어내는 축이다 — steering 파이프라인의 "검출기(트리거)" 부분에 해당하며, 우리가 풀려는 "phase/failure-type 온라인 식별" 문제와 방법론적으로 가장 가까운 선행연구. RND-OE는 정책 자체의 frozen encoder 임베딩에서 self-supervised novelty score를 뽑는다는 점에서, steering을 걸 시점(threshold-crossing)을 latent만으로 판단하는 실전 레시피를 제공한다.

## 우리 프로젝트 연결 (latent OOD 검출 방식 대조·우리 pathway별 OOD 귀인과 비교)
- **검출 방식의 근본적 차이**: FIPER의 OOD는 "관측 임베딩의 novelty"를 재는 방향 없는 스칼라 거리(RND)이고, 우리는 "성공/실패 조건부 분포 사이의 subspace"(conceptor C_success ∧ ¬C_failure)를 재는 방향 있는 통계량이다. FIPER는 실패 데이터가 전혀 없어도 되지만(장점) 대가로 실패의 "종류"를 전혀 구분하지 못한다(단일 스칼라, pathway 귀인 없음). 우리는 이미 실패 rollout을 보유하므로 pathway별(VL/DiT) 귀인이 가능하다는 점에서 상보적.
- **2-신호 AND 결합의 구조적 유사성**: FIPER의 "RND-OE(관측 쪽) AND ACE(action 쪽)" 게이트는, 우리의 "VL(goal) pathway 실패조건 AND DiT(motor) pathway 실패조건" 분리 개념과 구조적으로 대응시킬 수 있다 — RND-OE ~ VL(입력측 관측 인코더) OOD, ACE ~ DiT(출력측 action decoder) 불확실성. 다만 FIPER는 정책 전체의 단일 observation encoder 임베딩만 쓰고, 우리처럼 VL-SA/DiT block별 layer-wise 귀인(N16 pathway_step_attribution 계열)은 하지 않는다 — 이 세분화가 우리가 이 논문 대비 갖는 차별점.
- **재사용 가능한 실용 레시피**: conformal band 문턱 + sliding-window 집계(누적도 단일 timestep도 아닌 최근 k-step)는 우리 online phase/type 검출기의 threshold calibration에 바로 참고할 수 있다 — "문턱 보정은 성공 데이터만으로"라는 원칙은 우리 conceptor fit(실패 데이터도 쓰지만 threshold는 성공 기준으로 두는 것이 더 안전)과 결합 여지가 있다.
- **niche 공백**: FIPER는 단일-task vision-based IL 정책(diffusion U-Net·ACT flow)만 검증했고 VLA(OpenVLA/GR00T/pi0)에는 미검증 — §7에서 "대규모 VLA로 확장 가능할 것으로 기대한다"고만 서술하고 실험이 없다. GR00T에서 이 아이디어(RND류 novelty score)를 pathway별로 나눠 검증하면, 이 논문이 다루지 않은 "VLA + pathway-resolved online OOD 귀인"이라는 공백(우리 메모 notall-online-failuretype-niche와 직결)을 메울 수 있다.

## 면접 포인트 (Q→A)
1. Q: "FIPER가 실패 데이터 없이 어떻게 실패를 예측하나?" A: "두 신호(RND 기반 관측 novelty, action-chunk entropy)를 성공 rollout만으로 학습·conformal 보정하고, 두 신호가 동시에 문턱을 넘을 때만 실패로 판정한다. 핵심 난제는 '정책이 처리 가능한 OOD'와 '진짜 실패'를 구분하는 것인데, 관측만 보는 OOD 검출은 이를 못하고, 행동 쪽 불확실성을 같이 요구하면 구분력이 생긴다."
2. Q: "왜 AND(논리곱)를 쓰고 OR을 안 쓰나?" A: "OR은 TPR은 높지만 TNR이 0.05까지 붕괴해 대부분의 성공 rollout도 실패로 오탐한다. AND는 두 신호가 동시에 필요해 견고성이 크게 오르면서도 실패의 91%는 여전히 검출한다 — 실패가 관측 이탈과 행동 불확실성 둘 다를 동반한다는 저자 가설의 실증이다."
3. Q(우리 프로젝트 관점): "FIPER를 우리 VLA steering 파이프라인에 어떻게 접목하나?" A: "FIPER의 RND-OE는 VL 임베딩 OOD, ACE는 DiT action 불확실성에 각각 대응시켜, 우리 pathway 분리 검출기의 '두 신호 AND' 설계 근거로 참고할 수 있다. 다만 FIPER는 실패 종류(goal 대 motor)를 구분하지 못하는 이진 검출기이므로, 우리는 여기에 pathway별 conceptor 귀인을 더해 '어느 pathway가 실패에 기여했는가'까지 답하는 확장을 목표로 한다."

## 한계·비판
- 정책과 별도로 RND-OE 모델을 학습해야 하는 추가 파이프라인이 필요(저자도 명시적 한계로 인정).
- 최고 TWA도 0.65, Acc 0.78 수준 — 조립라인처럼 오탐 비용이 큰 응용에는 여전히 부족하다고 저자 스스로 인정(Appendix D).
- vision-based 단일-task IL 정책(RGB 1~2장+proprioception)만 검증, VLA·language/touch/audio 입력은 §7·Appendix D에서 "작동할 것으로 기대"라는 미검증 가설로만 언급 — 실제 VLA 실험 전무.
- time-varying threshold는 성공 rollout들의 시간적 패턴이 유사함을 암묵 가정 — 같은 task를 다른 타이밍으로 완수하는 경우(예: 두 번째 시도에 grasp 성공) 시간축이 어긋나면 오탐 가능성 있음(저자도 CP-constant 대안 필요성으로 언급).
- aleatoric uncertainty(시연 자체의 다양성)와 epistemic uncertainty(진짜 실패 신호)를 분리하지 못함 — 시연 다양성이 크면 ACE 문턱이 커져 TPR 저하 가능성(저자도 미해결 과제로 명시).
- 실패의 원인·종류에 대한 진단 정보를 전혀 제공하지 않는 이진 검출기 — steering 라우팅에 바로 쓸 수 있는 pathway/구간 정보가 없다(우리 프로젝트가 채워야 할 gap).
- 5개 seed, task별 성능 편차가 큼(SORTING TWA 0.54 vs PUSHCHAIR 0.83) — 일반화 강건성은 확실치 않음.
