# SAFE: Multitask Failure Detection for Vision-Language-Action Models (Gu et al. 2025)

- 출처: NeurIPS 2025 · arXiv:2506.09937v2 [cs.RO] (University of Toronto/Robotics Institute/Vector Institute + Toyota Research Institute) · PDF: docs/references/SAFE.pdf · 섹션=§2/§6(지정) — 실제로 §3(문제정의)·§4(방법)·§5-6(실험·결과)·Appendix B.5(min-length T fairness)·F.3(steering future work) 정독 · tier=must · 한 줄 역할: succ/fail이 VLA latent feature space에서 task-agnostic하게 분리·검출 가능함을 실증(per-step LSTM/MLP failure probe + functional conformal threshold) — 우리 pathway/phase steering의 "분리 가능성" 전제를 제공하는 선행토대이자, 우리가 그대로 재현해 길이통제 기준선으로 쓰는 논문.

## 문제·동기
generalist VLA는 seen task에서 SR 80~90%지만 unseen task/환경에서 30~60%로 급락한다. 기존 실패검출기는 대부분 task마다 개별 학습·보정(specialist)하는데, VLA는 배포 시 매번 새 instruction/환경을 만나므로 매 task마다 rollout을 모아 재학습하는 게 비현실적이다. 최근 task-generic 검출기들도 다중 action 샘플링(STAC, 10~256개)이나 대형 VLM 질의(AHA 등)를 요구해 실시간 로봇 제어에 부담이 크다. 저자들은 "multitask failure detection" — unseen task에서 추가 rollout 수집·finetune 없이 zero-shot 검출 — 을 문제로 정식화한 것이 이 논문 이전에는 없었다고 주장한다.

## 핵심 아이디어
VLA 내부 feature를 π0-FAST+LIBERO-10에서 t-SNE로 시각화(Fig.1)하면, 실패 시 서로 다른 task의 feature라도 같은 "failure zone"에 모이고, 성공 rollout은 이 zone 밖에 머문다. 즉 task-agnostic한 고수준 success/failure 지식이 VLA latent에 이미 인코딩돼 있다는 관찰이 출발점이다. 이 위에 가벼운(1~2 layer) MLP/LSTM probe를 여러 task로 학습시키면 unseen task에도 전이(zero-shot generalize)된다는 것이 핵심 주장.

## 방법 (per-step LSTM failure detector, feature-space, min-length 통제)
- **Feature 추출**: 마지막 transformer layer, action logit/velocity로 디코딩되기 직전 hidden state 텐서 E∈R^{n×d}(n=생성토큰 수/denoising step/horizon 등 모델마다 다름)를 First/Last/Mean/First&Last 중 Deval-seen 최고 성능 방식으로 aggregate해 단일 벡터 e_t로 만든다.
- **SAFE-MLP**: f_MLP(e_0:t) = Σ_τ σ(g(e_τ)) — 시점별 MLP 스칼라를 누적합. L1 loss(성공은 s_t↓0, 실패는 s_t↑t로 push): L = Σ_i [y_i·Σ_t(t−s_t) + (1−y_i)·Σ_t s_t].
- **SAFE-LSTM**: f_LSTM(e_0:t) = σ(LSTM(e_0:t)) — 순차 처리, BCE loss. 둘 다 의도적으로 1~2 layer로 작게 유지(과적합 방지·unseen 일반화 목적). 학습은 rollout-level 라벨(성공/실패)만 필요, 정확한 실패 시작 시점은 불요.
- **Threshold: functional conformal prediction**(Diquigiovanni et al.) — 성공 calibration rollout만으로 시간가변 상한 band C_α={[−∞, μ_t+h_t]}를 구성, exchangeability 가정 하에 성공 rollout이 이 band를 벗어날 FPR ≤ α를 보장. s_t가 band를 넘으면 failure flag.
- **길이 confound 통제(§6.1, Appendix B.5 — 우리 프로젝트가 가장 주목하는 부분)**: LIBERO류 시뮬레이터는 성공하면 조기종료, 실패하면 항상 max length까지 진행 → "그냥 경과시간을 세는(s_t=t)" detector도 완벽한 분류를 낼 수 있는 아티팩트가 생긴다. 저자는 이를 §6.1에서 명시적으로 경고("if a failure detector simply learns to count the time elapsed... it will achieve perfect failure detection")하고, **task별 최소 rollout 길이 T_min을 구해 성공·실패 모두 같은 T_min에서 평가한 s_T**를 Table 1 헤드라인 ROC-AUC 메트릭으로 채택한다 — 시간 누수를 원천 차단하는 직접 통제.

## 실험·결과(검출 성능)
- 벤치: OpenVLA/π0/π0-FAST × LIBERO-10(10 task), π0*(open-pi-zero 재구현) × SimplerEnv(Google Robot 4 task + WidowX 4 task), 실물 Franka(13 task, π0-FAST-DROID), 실물 WidowX(8 task, OpenVLA).
- 길이통제 s_T 기준(Table 1) ROC-AUC 평균: SAFE-LSTM seen 82.3/unseen 77.0, SAFE-MLP seen 81.4/unseen 78.0 — Embedding-distance(Mahalanobis/k-NN)·RND·LogpZO·STAC 등 baseline(평균 unseen 58~74) 대비 unseen에서 최고이거나 근접 최고. Token uncertainty(LLM UQ 이식) 계열은 전반적으로 저조.
- STAC·Sample-consistency 계열은 다중 action 샘플링(10~256개)이 필요해 π0/π0-FAST 추론이 각 152%/221% 느려짐 — SAFE는 <1ms(<1% 오버헤드, LSTM 2.3M 파라미터)로 실시간 배포에 실용적.
- functional CP α-sweep으로 얻은 balanced-accuracy vs detection-time trade-off(Fig.4)에서 SAFE 곡선이 대부분 top-left(정확+조기)를 지배. 사람이 라벨링한 GT 실패 시점 대비 실제 실패보다 먼저 경보하는 사례가 다수(π0-FAST에서 40%가 첫 timestep 근방 예측).
- 실물 로봇(Franka/WidowX)에서는 SAFE-MLP가 최고(unseen ROC-AUC 64.2/88.4), task 다양성이 커 t-SNE상 failure zone이 뚜렷하지 않은데도 detector 성능은 유지.

## activation-steering 흐름 위치(분리=조종의 전제)
SAFE 자체는 steering이 아니라 "검출"만 한다 — activation을 읽어 이진 실패 확률 s_t를 산출할 뿐 latent를 조작해 행동을 바꾸지는 않는다. 그러나 저자는 Appendix F.3(Future Works)에서 명시적으로 "성공/실패 embedding이 latent space에서 분리됨을 보였으므로, 이 latent activation을 조작해 VLA 행동을 바꾸는 steering vector를 학습할 수 있을 것"이라 제안하며 ITI(Li et al., [50])와 Adaptive Activation Steering([76])를 직접 인용한다. 즉 **"분리(separability) 실증이 곧 steering 가능성의 근거"**라는 논리를 저자 스스로 제시했고, 우리 프로젝트가 전제로 삼는 명제("검출 가능 = 조종 가능의 필요조건")는 이 논문의 미래연구 제안을 실제로 수행하는 작업에 해당한다.

## 우리 프로젝트 연결(빌리는 것·메우는 곳; 길이 confound 통제 방식)
- **빌리는 것**: (1) succ/fail이 VLA feature space에서 task-agnostic하게 분리·검출 가능하다는 핵심 가설과 실증 프로토콜(multitask per-step probe + functional CP, seen/unseen task-level split). (2) 길이confound는 반드시 task별 min-length T로 truncate한 s_T로 평가해야 한다는 방법론 — 우리 truncation-length-standard(성공 길이 [mean, mean+1σ] window)와 seen18 재현(`docs/seen18_safe_detector_verification.md`)의 직접 기준선이자 대조군. (3) Appendix F.3의 "분리→steering vector" 논리적 연결고리 자체.
- **메우는 곳(SAFE 한계)**: pathway 구분 없음 — 마지막 layer의 단일 pooled 임베딩만 쓰므로 VL(goal)/DiT(motor) 어느 pathway가 실패에 기여했는지 답할 수 없다(저자도 Appendix F.1에서 "multi-layer fusion은 future work"라 인정). 실패 '유형' 구분 없음 — 단일 스칼라 failure score만 내므로 goal 오인식 vs motor 실행붕괴를 구분 못한다. 온라인 "왜"에 답하지 못하고 "언제"만 답한다(탐지기이지 진단기가 아님).
- **길이 confound 통제 방식 대비(핵심 정정 포인트)**: SAFE는 §6.1에서 "s_t=t만으로 완벽한 검출이 나올 수 있다"를 스스로 경고하고 task별 min-length T로 truncate한 s_T를 헤드라인으로 채택 — **직접·명시적 통제**. 반면 COAST는 전체 rollout 길이에 걸쳐 action-token을 mean-pool해 class-wise covariance(conceptor) R=E[hh^T]에 그대로 넣고, "normalized trajectory time"은 v1(C_steer) projection 시각화용이지 fit 시 길이 매칭이 아니며, "matched-cost"는 추론 연산비용 매칭이지 rollout 길이 매칭이 아니다 — **통제 부재**. 우리는 SAFE 쪽 방식을 표준으로 채택했다.
- **우리 재현과의 긴장**: seen18 GR00T(N1.6, RoboCasa)에서 SAFE 프로토콜을 protocol-parity로 재현(task-level unseen split 4/18, 동일 min-length T truncation, split conformal + functional CP)한 결과 val_seen 0.683 / val_unseen 0.434 — SAFE 논문 평균(seen 82.3/unseen 77.0) 대비 각각 14·34점 낮고, unseen은 chance(permutation null 95% band [0.44, 0.56]) 근방이다. 길이 비통제 변형(`by final end`, max-so-far 전체 T)은 val_seen 1.000/unseen 0.992지만 길이-only baseline(step count, 0.996)과 사실상 동일해 순수 아티팩트임을 확인했다. 공유 "failure zone" 가설 자체는 부분 재현됐으나(task-whiten 후 centroid-spread 비율 p=1.0→0.75) probe 성능은 재현되지 않는다 — GR00T DiT가 별도 모듈 + 단방향 cross-attention(Q=action/state, K/V=frozen VLM Eagle 출력)이라, VLM과 attention을 공유하는 π0/OpenVLA보다 마지막-layer feature에 VLM의 추상적 success/failure 지식이 덜 보존됐을 가능성을 architectural 가설로 제기했으나 미검증이다.

## 면접 포인트 (Q→A; 분리가 왜 조종의 전제인가)
1. Q: "SAFE가 정확히 뭘 보여주고, 그게 왜 steering의 전제가 되나?" A: "SAFE는 VLA 마지막 layer feature에서 성공/실패 rollout이 task와 무관하게 구분되는 영역(failure zone)으로 분리된다는 것을, 그 분리를 이용하는 검출기(LSTM/MLP probe)를 학습·전이시켜 실증한다. Steering은 latent를 특정 방향(성공 쪽 부분공간)으로 밀어 행동을 바꾸는 개입인데, 애초에 성공/실패가 latent 공간에서 분리 가능한 방향이 없다면 '어느 방향으로 밀 것인가'를 정의할 수 없다. 즉 분리 가능성은 조종 벡터/부분공간이 존재하기 위한 필요조건이고, SAFE는 이를 검출기라는 형태로 실증했으며 Appendix F.3에서 저자 스스로 이를 steering의 근거로 명시한다."
2. Q: "SAFE의 길이 confound 처리와 COAST의 차이는?" A: "시뮬레이터는 성공하면 조기종료, 실패하면 항상 최대 길이까지 진행되므로, 경과 시간만 세도 완벽한 분류가 나오는 아티팩트가 생긴다. SAFE는 §6.1에서 이를 명시적으로 경고하고 task별 최소 길이 T로 truncate한 s_T를 헤드라인 지표로 삼는다 — 직접 통제. COAST는 전체 길이를 mean-pool해 conceptor에 넣어 이 통제가 없다 — 우리는 SAFE 쪽 통제를 우리 실험 표준(truncation-length-standard)으로 채택했다."
3. Q: "우리 GR00T 재현에서 SAFE가 왜 안 통했나?" A: "동일 프로토콜(task-level unseen split, min-length T truncation, split/functional conformal)로 GR00T N1.6+RoboCasa seen18에 재현했더니 val_unseen 0.434로 chance 근방이었다(SAFE 평균 77.0 대비 -34점). 공유 failure zone 가설은 약하게 재현됐지만(centroid-spread 비율 이동은 관찰) probe 성능이 안 나온다. 가장 유력한 가설은 GR00T의 action 생성기가 별도 DiT + 단방향 cross-attention(VLM은 K/V로 한 번 조회만 됨)이라, π0/OpenVLA처럼 VLM과 attention을 공유하는 구조보다 마지막-layer feature가 VLM의 추상적 success/failure 지식을 덜 보존할 수 있다는 것 — 다만 미검증 가설이다."
4. Q: "SAFE는 pathway나 실패 유형을 구분하나?" A: "아니다. 마지막 layer의 pooled 임베딩에서 단일 스칼라 failure score만 낸다. 어느 pathway(VL/DiT)가, 어느 phase에서, 어떤 원인(goal 오인식 vs motor 실행붕괴)으로 실패했는지는 전혀 답하지 않는다 — 이게 우리가 pathway-resolved + phase-matched steering으로 메우려는 공백이며, SAFE도 Appendix F.1에서 multi-layer fusion을 미해결 future work로 남긴다."

## 한계·비판
- 마지막 layer 단일 임베딩만 사용 — layer간 fusion, 중간 layer 탐색은 저자 스스로 future work로 남김(Appendix F.1). LLM hallucination 문헌(Azaria&Mitchell, Kossen et al.)처럼 최적 layer가 last가 아닐 수 있음을 알고도 미탐구.
- 완전 zero-shot이 아니라 "multitask 학습 후 unseen task 전이" — 신규 배포 전 반드시 policy를 굴려 수백 개의 성공/실패 rollout(여러 task)을 모아야 학습이 가능하다(저자도 명시).
- 실물 실험(Franka/WidowX)에서는 task 다양성이 커 t-SNE 상 성공/실패 embedding이 뚜렷이 분리되지 않고(Appendix C.1), 그 결과 모든 detector의 ROC-AUC가 최대 64에 그침 — "task-agnostic failure zone" 가설이 task 다양성이 커지면 약해질 수 있음을 저자 스스로 인정.
- functional CP의 exchangeability 가정이 multitask(특히 unseen task) 세팅에서 엄밀히 성립하지 않아, TNR이 이론적 하한 1-α에서 벗어나는 벤치마크가 다수(Appendix C.2) — 저자도 형식적 보장이 아니라고 인정.
- OpenVLA 임베딩은 t-SNE상 통합된 failure zone을 형성하지 않음(작은 blob들로 분산)에도 detector는 여전히 잘 작동 — 저자는 "t-SNE로 안 보이는 상관관계를 추출했을 것"이라 추정만 할 뿐 설명하지 못함.
- 우리 GR00T 재현에서 unseen 성능이 chance 근방으로 나온 것은 논문이 보고하지 않은 실패 사례 — "generalist VLA 전반에 통한다"는 주장이 최소 하나의 아키텍처(cross-attention 기반 별도 action-DiT)에서는 깨질 수 있음을 시사한다(단, 원인 규명은 우리 측에서도 미완).
