# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models (Zhao et al. 2025)

- 출처: arXiv:2503.22020v1 [cs.CV] (2025-03-27) · NVIDIA / Stanford / MIT(Qingqing Zhao 1저자, Tsung-Yi Lin 등 교신) · PDF: `docs/references/CoT-VLA.pdf` · 정독 섹션: 논문 전체(§3 방법·§4 실험·§5 한계, 총 8페이지+부록) · 서베이 배치: §6 VLA · tier: must · 한줄 역할: hidden-state steering이 아니라 **subgoal 이미지를 명시적으로 autoregressive 생성해 action을 조건화**하는 visual-CoT 아키텍처 — activation steering과는 개입 지점이 다르지만, "phase/subgoal 구조"를 명시적 생성물로 다룬다는 점에서 우리 phase-matched steering의 문제의식과 인접한 참고 사례.

## 문제·동기
기존 VLA(OpenVLA류)는 관측→행동 direct mapping만 학습해 중간 reasoning step이 없고, 그 결과 temporal planning/reasoning 능력이 결여된다. LLM의 CoT prompting이 단계별 사고로 성능을 끌어올린 것처럼, 로봇 조작에도 명시적 중간 표현(언어 설명·keypoint·bounding box 등, 선행연구)이 시도됐지만 대개 추가 전처리 파이프라인이 필요했다. 저자들은 로봇 데모 데이터에 이미 자연스럽게 존재하는 "미래 프레임"을 별도 라벨링 없이 subgoal 이미지로 바로 쓸 수 있다는 데 착안한다.

## 핵심 아이디어
행동 예측을 두 단계로 분리한다: (1) 현재 관측 s_t와 언어 instruction l로부터 n프레임 뒤의 subgoal 이미지 ŝ_{t+n}을 autoregressive하게 생성(visual reasoning), (2) s_t, l, ŝ_{t+n} 모두를 조건으로 action chunk {â_t...â_{t+m}}을 생성. Vanilla VLA(â_t ~ P(a_t|s_t,l))와 달리 "먼저 시각적으로 생각한 뒤 행동"하는 구조다. 핵심 이점은 (1) subgoal 생성 단계는 action label이 필요 없어 EPIC-KITCHENS-100·Something-Something V2 같은 action-less 영상까지 사전학습에 흡수 가능하고, (2) action generation 단계만 로봇 데모 D_r로 학습해도 된다는 점 — 두 손실을 분리 학습함으로써 unlabeled video의 규모 이점을 VLA로 끌어온다.

## 방법(visual CoT/subgoal, action 생성)
- 백본: VILA-U(통합 이해+생성 멀티모달 모델). RQ-VAE 기반 residual quantization으로 이미지를 16×16×4 이산 토큰(residual depth 4)으로 인코딩, depth transformer가 잔차 토큰을 autoregressive 예측.
- **Hybrid attention**(Fig.3): 이미지/텍스트(subgoal) 생성은 causal attention + next-token prediction. 반면 action 토큰 예측은 **full attention**으로 전환해 action 시퀀스 내 모든 토큰이 서로 참조하며 병렬 디코딩([x],[θ],[g] 특수토큰 사용) — 기존 VLA(OpenVLA 등)의 순차적 autoregressive action decoding과 다른 설계.
- **Action chunking**: action 하나를 7개 토큰(각 차원 256-bin 이산화, 텍스트 vocab의 최저빈도 256토큰 재사용)으로 표현, chunk size는 데이터셋별로 10 내외.
- 손실: L = L_visual(subgoal 잔차토큰 CE) + L_action(action 토큰 CE). L_visual은 D_r+D_v(action-less video) 모두에서, L_action은 D_r에서만 학습.
- subgoal 예측 horizon n은 데이터셋별로 [n_l, n_u] 구간에서 균일 샘플링해 수작업 설정(Bridge n∈[5,10], TOTO n∈[20,24] 등, 부록 Table 4) — 학습된/온라인 적응형 horizon이 아니라 고정 휴리스틱.
- 2단계 학습: (1) OpenX 서브셋+action-less video로 사전학습(LLM backbone+projector+depth transformer 학습, vision tower 고정), (2) 다운스트림 로봇셋업(LIBERO/Bridge-V2/Franka-Tabletop)에서 태스크별 파인튜닝.
- 테스트타임 closed-loop(Algorithm 1): 매 스텝 subgoal 이미지 재생성 → action chunk 생성·실행 m+1 스텝 → 새 관측으로 반복.

## 실험·결과
- LIBERO 4-suite 평균: CoT-VLA 81.13% vs OpenVLA 76.5%, Octo 75.1%, Diffusion Policy 72.4%. 특히 LIBERO-Long(장기 태스크)에서 69.0% vs OpenVLA 53.7%로 격차가 가장 큼 — subgoal reasoning이 장기 planning에 특히 기여한다는 근거.
- Bridge-V2(4개 일반화 축, 10 trial/축): Visual/Language 축은 OpenVLA보다 소폭 낮음(action chunking으로 인한 grasping 실패, 시각추론 오류 아님), Motion/Semantic 축은 OpenVLA보다 우세(45%→60%, 40%→50%) — 균일 우세는 아니고 혼재된 결과.
- Franka-Tabletop(6태스크, 소량 데모 10~150개): 전 baseline 대비 평균 최고, 특히 multi-instruction 태스크에서 OpenX 사전학습 모델(Octo/OpenVLA/CoT-VLA)이 Diffusion Policy보다 우세.
- Ablation(Fig.6): VLA(base) → +action chunking → +hybrid attention → +CoT 순으로 LIBERO-Spatial/Goal에서 단계적 성능 향상, CoT 추가가 최종 최고 성능.
- Pretraining ablation: OpenX+비디오 사전학습이 Franka-Tabletop에서 53.7%→78.8%(46.7% 상대개선).
- OOD subgoal 품질 실험(Table 3, 우리 프로젝트와 특히 관련): 미학습 장기 태스크에서 "모델이 직접 생성한 goal 이미지" vs "실제 GT goal 이미지"로 조건화한 action 비교 — GT 사용 시 SR이 절대 +40%p(sub-task1: 20%→60%, sub-task2: 0%→40%) 상승. **subgoal(=goal 표상)의 품질이 곧 action 성능을 직접 좌우함**을 인과적으로 보여준다.

## activation-steering 흐름 위치(steering 아님, phase/subgoal 구조 참고)
이 논문은 activation steering 계보에 속하지 않는다. 개입 지점이 hidden state가 아니라 **출력 모달리티 자체**(subgoal 이미지를 픽셀 공간에 명시적으로 생성)이고, 이 생성물을 다음 단계 action 생성의 조건 입력으로 재사용하는 end-to-end 학습된 아키텍처 설계다. ActAdd/CAA/RepE/conceptor류가 forward pass 도중 h←h+α·v로 잠재표현을 사후 개입하는 것과 달리, CoT-VLA는 학습 시점부터 "subgoal→action" 두 단계 생성을 목적함수에 명시적으로 포함시킨다 — steering이 아니라 **아키텍처 차원의 phase 분해**다. 서베이 맥락에서는 VLS_SteerViaVLM(개입 지점이 action-space/denoising sampling인 사례)과 함께 "activation이 아닌 다른 층위에서 phase/subgoal 구조를 다루는" 대조군으로 위치한다.

## 우리 프로젝트 연결(subgoal=phase 구조 유비; 직접연결 약하면 명시)
직접연결은 약하다. 명시적으로 밝힌다: CoT-VLA는 succ/fail latent 분리, 온라인 phase/failure-type 식별, hidden-state steering 중 어느 것도 다루지 않는다.
- **구조적 유비만 유효**: subgoal 이미지는 "무엇을 해야 하는가(goal, VL pathway가 encode하는 것과 개념적으로 유사)"를 명시적으로 externalize한 표상이고, 그 뒤 action chunk 생성은 "어떻게 실행하는가(motor, DiT pathway)"에 해당한다 — 우리 project의 VL(goal)/DiT(motor) pathway 분리와 역할상 유비는 가능하지만, CoT-VLA는 이걸 별도 latent subspace 조작이 아니라 두 개의 순차 생성 단계(이미지 토큰 생성 → 액션 토큰 생성)로 구현한다는 점이 다르다.
- **horizon n을 데이터셋별 수작업 [n_l,n_u]로 고정**하는 설계는, 우리가 풀려는 "rollout phase를 online으로 식별"하는 문제와 정반대 접근이다(CoT-VLA는 phase/horizon을 학습·추론 시점에 판별하지 않고 고정 분포에서 샘플링) — 온라인 phase 식별 문제의 미해결성을 보여주는 방증으로만 인용 가능.
- **Table 3(GT vs generated goal 비교)**은 "goal/phase 표상의 품질이 downstream action 성능을 인과적으로 좌우한다"는 우리 project의 근본 전제(steering으로 성공 방향 latent를 명확히 하면 SR이 오른다)와 정성적으로 같은 방향의 방증이다 — 다만 이건 픽셀공간 subgoal 품질 실험이고, 우리가 다루는 latent-space steering 인과성과는 실험 설계가 다르므로 방증 수준 이상으로 확대해석하지 않는다.
- closed-loop마다 subgoal을 재생성하는 구조는 암묵적으로 매 스텝 "지금 phase에서 다음 목표가 뭔가"를 다시 묻는 것과 같지만, 이 재생성 신호를 phase-conditioned steering에 활용하는 실험은 없다 — 우리 핵심 미해결 문제(online phase 식별)에 대한 답을 주지 않는다.

## 면접 포인트(Q→A)
1. Q: "CoT-VLA도 activation steering 계열인가?" A: "아니다. hidden state를 사후 개입하는 게 아니라, subgoal 이미지를 픽셀 공간에 autoregressive 생성해 action 생성의 조건 입력으로 쓰는 아키텍처·학습목적함수 설계다. 개입 지점이 잠재표현이 아니라 출력 모달리티라는 점에서 steering과 구분된다."
2. Q: "hybrid attention이 뭐고 왜 필요한가?" A: "이미지/텍스트(subgoal) 생성은 causal attention+next-token prediction, action 토큰 생성은 full attention으로 전환해 action chunk 내 모든 토큰이 서로 참조하며 병렬 디코딩한다. 순차 생성이 필요한 subgoal과 병렬 생성이 유리한 action의 성격이 다르기 때문에 attention 패턴을 나눴다."
3. Q: "이 논문이 우리 project의 phase-matched steering과 어떻게 다른가?" A: "CoT-VLA의 subgoal horizon n은 데이터셋별로 수작업 고정한 [n_l,n_u] 범위에서 샘플링되고, phase를 온라인으로 판별하지 않는다. 우리가 풀려는 문제(추론 중 rollout phase를 실시간 식별해 steering을 조건화)는 이 논문에서 다뤄지지 않는다 — 오히려 그 문제가 여전히 미해결임을 보여주는 방증에 가깝다."
4. Q: "성능 향상의 핵심 근거는?" A: "LIBERO 평균 81.13%(OpenVLA 76.5%)이고 특히 장기 태스크(LIBERO-Long)에서 69.0% vs 53.7%로 격차가 가장 크다. Ablation에서 action chunking→hybrid attention→CoT 순으로 단계적 개선을 보였고, GT subgoal 이미지를 쓰면 OOD 태스크 SR이 +40%p 오른다(Table 3) — goal 표상 품질이 action 성능을 직접 좌우한다는 인과적 근거다."
5. Q: "실전 배포 관점에서 이 방법의 가장 큰 걸림돌은?" A: "추론 시 action 전에 256개 이미지 토큰을 생성해야 해서 평균 7배 느려진다(action chunk=10 기준). 실시간 closed-loop 제어에서 이 latency는 심각한 제약이고, 저자도 이를 첫 번째 한계로 명시한다."

## 한계·비판
- **추론 지연**: subgoal 이미지 256토큰 생성이 action 생성 전에 선행돼 평균 7배 slowdown(action chunk=10 기준) — 고빈도 실시간 제어와 상충. 우리가 지향하는 "online 신호로 즉시 steering 조건화"와는 반대로, 이 아키텍처는 매 스텝 무거운 생성이 필요해 온라인 개입 설계의 참고 사례로는 부적합.
- autoregressive 이미지 생성의 시각 품질이 diffusion 기반(SUSIE 등)보다 낮음 — subgoal 자체의 신뢰도에 한계.
- action chunk 경계에서 불연속 행동 발생 가능(고빈도 피드백 부재), 저자도 future work로 명시.
- subgoal horizon [n_l,n_u]가 데이터셋별 수작업 하이퍼파라미터 — 학습되거나 online 적응되는 값이 아니라 일반화·재현성에 취약할 수 있는 설계.
- Bridge-V2에서 OpenVLA 대비 균일 우세가 아님(Visual/Language 축은 오히려 낮음, action chunking에 의한 grasping 실패로 저자가 귀인) — "CoT가 항상 이긴다"는 단순한 서사는 아니다.
- OOD subgoal 생성 자체는 여전히 취약(Table 3에서 generated goal 사용 시 GT 대비 SR 격차 40%p) — 저자도 "advances in visual reasoning"에 대한 의존성을 미해결 한계로 인정.
- succ/fail latent 분리나 내부 표현 수준의 인과 개입 실험이 전무 — activation-steering 서베이의 핵심 질문(무엇을 어떻게 개입해 SR을 올리는가)에 직접 답하지 않는, 구조적 유비 참고용 논문임을 재확인.
