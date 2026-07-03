# VLS: Steering Pretrained Robot Policies via Vision-Language Models (Liu et al. 2026)

- 출처: arXiv:2602.03973v1(2026-02-03) · University of Washington / University of Oxford / NUS / Allen Institute for AI(Shuo Liu, Jiafei Duan, Ranjay Krishna 공동교신) · PDF: `docs/references/VLS_SteerViaVLM_2602.03973.pdf` · 정독 섹션: 전체(§III 문제정식화 · §IV 방법 · §V 실험) · 서베이 배치: §6 VLA · tier: must · 한줄 역할: activation이 아니라 diffusion/flow policy의 denoising 샘플링 과정을 VLM이 즉석 생성한 differentiable reward로 guide하는 training-free 프레임워크 — 우리 프로젝트의 hidden-state steering과 "개입 지점"이 다른 대조축.

## 문제·동기

Pretrained diffusion/flow-matching 로봇 정책은 학습분포 내에선 강하지만 obstacle 근접, support surface 변화, 약한 clutter, unseen instruction 등 OOD 입력에서 급격히 실패한다. 저자들은 이 실패가 "모터 스킬 부재"가 아니라 "action generation이 학습시 특정 spatial config에 지나치게 결합되어, 이미 존재하는 스킬을 선택적으로 못 꺼내쓰는" 문제라고 규정한다. Retrain/finetune은 비용이 크고 개념적으로도 부적절(이미 존재하는 행동을 다시 배우는 셈) — 그래서 문제를 "inference-time control"로 재정의하고 base policy는 frozen 유지한다.

## 핵심 아이디어

Classifier guidance(Dhariwal&Nichol)를 로봇 정책에 확장: 우도 log p((o,l)_OOD | a) 를 직접 못 구하므로, VLM으로 OOD 조건을 3D keypoint scaffold P로 grounding하고, 다시 VLM에게 stage별 differentiable reward function R_s를 PyTorch 코드로 "즉석 합성"시켜(fVLM), 그 gradient g=∇_a R_s를 diffusion noise 예측/flow velocity에 주입한다. VLM 자체는 미분 불가능한 off-graph 컴포넌트이고, VLM이 만들어낸 reward 함수만 action에 대해 미분 가능 — 이 비대칭이 핵심 트릭이다.

## 방법(diffusion/flow policy denoising을 VLM reward로 guide, 학습불필요)

- 3단계: (1) SAM+DINOv2로 OOD 관측을 keypoint set P로 grounding, (2) VLM이 task를 S단계로 분해하고 각 stage에 programmatic reward R_s(a,P) 생성, (3) denoising loop에서 gradient 주입 + RBF 반발항(초기 다양성) + Feynman-Kac 리샘플링(gradient-free, 후보군을 reward 가중치로 resample) 결합.
- 갱신식: 확산모델은 노이즈예측 보정 ϵ_hat = ϵ − λ√(1−ᾱ_k)·g, flow-matching은 속도장 보정 v_hat = v + λ·g (g=∇_a R_s). 매 denoising step에 MCMC식 다중 inner update로 gradient noise 완화.
- Closed-loop stage 전환: reward 값 R^t_s의 상대 비율로 guidance strength λ_t를 sigmoid 스케줄링(Eq.10, 초반엔 강하게 강제교정 후 점점 base policy에 위임), Schmitt-trigger(hysteresis)로 R_high/R_low 임계를 둬 stage advance/maintain/reinforce를 판정 — VLM이 전환 시점마다 다음 stage reward를 선택.
- base policy·VLM 파라미터 모두 미고정(frozen), 매 episode/stage마다 reward 함수만 새로 합성 — 이게 "training-free"의 의미.

## 실험·결과

- LIBERO-PRO: 고정 π0.5(LeRobot) + VLS overall SR 36.81% vs baseline 23.69%(+13pt), OpenVLA/π0/π0.5는 대부분 task perturbation에서 SR≈0%(극단적 OOD 취약성).
- CALVIN: movable objects 94%(base DP 대비 7.4배), articulated parts 87%(9.6배), 기존 steering(DynaGuide, ITPS) 대비 +15~25pt.
- ablation: gradient guidance 제거시 88%→17.3%로 붕괴 — dense reward gradient가 핵심 driver, FK resampling·RBF diversity는 SR보다는 효율/안정성(episode length, inference time)에 기여.
- 실물 Franka: in-distribution 69%(baseline 대비 +19pt), OOD(외형/위치/물체 치환) 중 최악 케이스(unseen mug 치환)에서 baseline 0% vs VLS 40%.
- 한계로 저자 스스로 명시: batch sampling+MCMC+FK 중첩으로 inference latency가 큼(약 1000~1200ms, Fig.4).

## activation-steering 흐름 위치(개입 지점: 활성화 vs 샘플링)

개입 지점이 우리 서베이의 다른 논문들과 근본적으로 다르다. ActAdd/CAA/RepE/conceptor 계열은 모델 forward pass 도중 hidden state h를 직접 조작(h←h+α·v)하는 "activation-time steering"이다. VLS는 hidden state를 전혀 건드리지 않고, denoising sampling 과정의 출력(action trajectory)에 대한 noise/velocity 예측만 classifier-guidance 방식으로 보정하는 "action-space / sampling-time steering"이다. 저자 스스로 Intro에서 이 계보를 PPLM·classifier guidance·SDEdit 등 LLM/이미지생성의 inference-time steering으로 명시(우리 서베이의 activation steering 계보와는 별개 갈래). VLM은 reward 코드를 생성하는 역할만 하고 gradient path 밖에 있어, VLM 내부 activation도 개입 대상이 아니다.

## 우리 프로젝트 연결(개입 지점 대조; hidden-state steering과 차이)

- 대조축: 우리는 GR00T DiT의 internal activation h를 conceptor로 succ 방향에 project(h'=h·Mᵀ) — 개입이 policy "내부" representation에서 일어난다. VLS는 policy를 블랙박스로 두고 denoising 궤적(policy "외부" 출력공간)을 reward gradient로 밀어낸다. 면접에서 activation steering의 정의 경계를 설명할 때 이 논문을 반례로 세우면 "무엇을 조작하는가"가 명확해진다.
- phase 문제의식은 닮음: VLS의 stage-wise reward + Schmitt-trigger 전환은 "단일 global steering이 아니라 rollout 진행에 따라 개입을 바꿔야 한다"는 우리 phase-matched 문제의식과 같은 결이다. 차이는 VLS가 explicit reward 값(R^t_s)과 VLM 판단으로 stage를 명시적으로 감지하는 반면, 우리 핵심 난제는 explicit reward 없이 hidden activation만으로 phase/failure-type을 온라인 추정하는 것 — VLS는 이 난제를 "VLM+정의된 reward"로 우회한다.
- Eq.10의 adaptive guidance strength(R^t_s/R^base_s 비율로 λ_t 조절)는 우리가 conceptor 강도(alpha)를 online phase 신호에 조건화하려는 아이디어의 직접적 선례다. 다만 VLS는 명시적 스칼라 reward가 있어 비율 계산이 가능하지만, 우리는 succ/fail latent 분포만 있고 explicit reward가 없어 이 스킴을 그대로 이식할 수 없다.
- VLS의 OOD grounding은 "무엇이 OOD인지"가 사전에 알려진 세팅(object/position/task perturbation 축이 명시)이라 우리 문제(자연발생 실패, 사전 라벨 없는 온라인 failure-type 식별)보다 근본적으로 쉬운 문제다.

## 면접 포인트(Q→A; sampling-time vs activation-time steering)

**Q1. Activation steering과 VLS 같은 sampling-time steering의 차이는?**
A. Activation steering은 모델 forward pass 도중 hidden state를 직접 이동시켜(h+α·v) 이후 계산 전체에 영향을 준다. VLS/classifier guidance 계열은 모델 파라미터와 내부 activation은 그대로 두고, 출력(denoising 궤적)에 대한 외부 gradient로 샘플링 분포만 재구성한다 — 개입 대상이 "내부 표현"이냐 "생성 과정의 출력 궤적"이냐로 나뉜다.

**Q2. VLS가 "training-free"라는 게 정확히 무슨 뜻인가?**
A. base policy와 VLM 파라미터 둘 다 학습하지 않는다는 뜻이다. 대신 매 episode/stage마다 VLM이 그때그때 differentiable reward function(PyTorch 코드)을 프롬프트로 즉석 합성해 denoising loop에 꽂아 넣는다 — "guidance function 자체를 추론 시점에 생성한다"는 게 novelty의 핵심이다.

**Q3(우리 프로젝트 관점). VLS의 stage-switching을 우리 phase-matched steering에 이식할 수 있나?**
A. "진행도에 따라 개입을 바꾼다"는 아이디어는 이식 가능하지만 메커니즘은 그대로 못 가져온다. VLS는 explicit reward 값과 Schmitt-trigger hysteresis로 stage를 감지하는데, 우리는 explicit reward가 없고 hidden activation만으로 phase/failure-type을 온라인 추정해야 하는 게 아직 풀리지 않은 부분이다.

## 한계·비판

- batch sampling(B) + MCMC + FK resampling이 중첩되어 latency가 큼(≈1~1.2초, Fig.4) — 실시간 제어 부담, 저자도 future work로 "compute efficiency 최적화"를 명시.
- known OOD axis(object/position/task perturbation)를 가정한 세팅 — perturbation 종류를 사전에 모를 때도 VLM이 알아서 대응할지는 미검증. 우리 문제(사전 라벨 없는 자연발생 실패)보다 쉬운 세팅이다.
- reward 함수 품질이 VLM의 spatial reasoning 정확도에 전적으로 의존 — keypoint grounding이 틀리면 guidance 자체가 오염되는데 이 실패모드에 대한 분석은 없다.
- ablation에서 FK/RBF 제거 효과가 gradient guidance 제거 효과에 비해 작다(85.3%/86.0% vs full 88% vs w/o grad 17.3%) — 두 컴포넌트의 필요성이 상대적으로 약하게만 입증됨.
- OpenVLA/π0/π0.5 baseline이 LIBERO-PRO에서 거의 0%라는 결과는 LIBERO-PRO 자체의 난이도(순수 memorization 방지 설계)에 상당 부분 기인할 수 있어, "VLS가 필요하다"는 주장이 벤치마크 난이도 선택에 다소 의존적일 수 있다.
- 2026-02 프리프린트로 공개 코드/외부 재현 검증이 아직 없다.
