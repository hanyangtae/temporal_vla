# Observing and Controlling Features in Vision-Language-Action Models (Buurmeijer 2026)

- 출처: arXiv 2603.05487v1 [cs.RO], 5 Mar 2026 (Hugo Buurmeijer*, Carmen Amo Alonso*, Aiden Swann†, Marco Pavone*‡ — Stanford Aero/Astro·Mech Eng, NVIDIA Research) · PDF: `docs/references/Observing and Controlling Features in Vision-Language-Action Models.pdf` · 섹션=§6 VLA(서베이 배정 섹션) — 논문 본문 전체(9쪽, §I 서론~§VI 결론) 정독 · tier=must · 한줄역할: **VLA(π0.5, OpenVLA) transformer 표현에 선형 observer(상태·액션 프로브)와 선형 controller(제약최적화 폐형해 개입)를 결합해, 고전 제어이론의 observability/controllability 정의를 VLA feature에 형식적으로 이식하고 closed-loop success-rate 대 constraint-satisfaction trade-off까지 측정한 논문** — SAE_VLA_pi05(Swann 2026, 저자 겹침)가 "미래연구"로 인용하는 선행 논문이자 우리 프로젝트 관측/제어 개념의 이론적 기반.

## 문제·동기
VLA는 구조적으로 LLM(transformer 백본)을 공유하지만, 멀티모달 입출력·연속 action·closed-loop(환경과 상호작용하며 다음 입력이 이전 출력에 영향받음) 특성 때문에 LLM mechanistic interpretability/steering 기법이 그대로 이식되지 않는다. 기존 VLA steering 연구(Häon et al. 2025, FFN neuron 개입)는 constraint 만족이나 steerability 중 하나만 따로 보고하며, 자연스러움(naturalness)·closed-loop 성능 보존이라는 LLM steering의 핵심 관행이 VLA에서는 검증되지 않은 채 남아있었다. 저자들은 "관측 가능한 특징을 최소 개입으로 정확히 원하는 영역에 두면서, 정책의 나머지 행동은 보존한다"는 목표를 제어이론(Kalman 1960) 언어로 형식화해 이 공백을 메운다.

## 핵심 아이디어
Transformer 층의 활성화 x_ℓ에 대해 두 개념을 정의한다: **feature-observability**(정의1) — 관측자 f_ℓ(x_ℓ)=ζ가 존재해 특징 ζ를 복원 가능; **feature-controllability**(정의2) — 제어자 g_ℓ(x_ℓ)=x̃_ℓ가 존재해 그 수정된 표현을 이후 층에 전파했을 때 ζ가 목표집합 D에 들어가게 만들 수 있음. 두 성질은 고전 제어이론처럼 독립적(관측 가능해도 제어 불가능할 수 있고 그 역도 가능)이지만, 이 논문이 제안하는 controller는 observer의 출력을 그대로 이용하므로 실용적으로는 controllable layer 집합이 observable layer 집합의 부분집합이 되도록 설계한다(L_C ⊆ L_O). 두 map 모두 선형으로 제한함으로써(linear representation hypothesis) 학습·개입 모두 계산량이 거의 없는 폐형해를 얻는다.

## 방법 (VLA feature 관찰·steering 메커니즘)
- **대상 아키텍처 2종**: (a) transformer 전용 자기회귀 VLA(OpenVLA, action을 토큰화해 최종층 x_T에서만 디코딩) (b) transformer+flow-matching 하이브리드(π0.5, VLM transformer가 이미지·언어를 처리하고 별도 "action expert"가 flow matching으로 연속 action을 생성하며 중간층 표현 x_1..x_T에 조건화됨). 논문은 **두 아키텍처 모두에서 transformer 구성요소만** 관측·개입 대상으로 삼고, π0.5의 flow-matching 층 자체는 건드리지 않는다(future work로 명시).
- **관측 대상 feature**: 로봇 상태 s=(x,y,z,φ,θ,ψ,g)(카테시안 위치+roll/pitch/yaw+정규화 gripper 개도)와 행동 a=Δs(상대 변위)로 국한. 위치/자세는 회귀 프로브, gripper는 이진 프로브.
- **선형 observer** f_ℓ(x)=W_ℓx+b_ℓ (식3): 입력-특징 쌍 {s^(i), ζ^(i)}으로 각 층까지 순전파해 활성화-특징 쌍을 모으고, 층별로 cross-entropy/회귀 손실(식4)을 최소화해 W_ℓ,b_ℓ 학습(Algorithm 1). 모든 층에서 독립적으로 학습, 최적 층을 사후 선택.
- **선형 controller** g_ℓ(x)=x+u_ℓ (식5): u_ℓ는 ‖u‖²_2를 최소화하면서 f_ℓ(x_ℓ+u)가 목표 구간 D=[ζmin,ζmax](1차원 박스 제약)에 들어가도록 하는 제약최적화(식6)의 해. Cheng & Amo Alonso(2024, arXiv 2405.15454 — 이 논문 공저자 Amo Alonso 본인의 LLM steering 논문) Theorem 4.1의 폐형해를 그대로 재사용: ζ_ℓ이 상한을 초과하면 u_ℓ=(ζmax−ζℓ)·W_ℓ/‖W_ℓ‖², 하한 미달이면 대칭식, 구간 안이면 u_ℓ=0(식7). 즉 **고정 크기 벡터를 항상 더하는 activation-addition류(ActAdd/CAA)와 달리, 개입 크기가 현재 관측값과 목표 구간의 거리에 따라 예제마다 달라지는 최소노름 사영이며, 이미 목표 안에 있으면 개입하지 않는다.**
- **추론 시 결합**(Algorithm 2): 순전파 중 ℓ∈L_O이면 관측, ℓ∈L_C이면 관측값 기반으로 u_ℓ 계산해 즉시 표현에 더함 — 표준 forward-pass 대비 오버헤드가 무시할 수준.
- Remark 2로 closed-loop(VLA는 자신의 출력이 환경을 바꾸고 그 환경이 다음 입력이 됨) vs open-loop(LLM 생성)의 근본 차이를 명시하되, "closed-loop 상호작용이 probe 학습 데이터 분포를 벗어나지 않는 한" 관측/제어 개념이 그대로 성립한다고 조건부로 주장.

## 실험·결과
- **Observability**(§V-A, Fig.3): π0.5×Libero(spatial suite), OpenVLA×BridgeData V2에서 전 층에 프로브 학습, 최고 성능 층 기준으로 평균예측/다수클래스 baseline 대비 MAE·accuracy 모두 우세 — 상태·행동이 두 아키텍처 모두에서 선형으로 관측 가능.
- **Robustness**(Fig.4): 단일 층에 x_ℓ+α 섭동을 가하고 관측된 action 변화를 측정. π0.5는 α 증가에 따라 매끈하게 단조 증가(관측이 견고). OpenVLA는 delta yaw가 특히 견고하지 않고 delta gripper만 약한 순서를 보임. 표현의 L2-norm이 층 깊이에 따라 커져 고정 크기 섭동의 효과가 깊은 층일수록 줄어듦(→ 초반 층 개입이 더 효과적).
- **Controllability 정합성**(Fig.5, layer 9): 제안 controller는 개입된 표현의 관측 이미지가 항상 [ζmin,ζmax] 안에 들어가도록 강제(hard constraint) — 무개입/고정벡터 섭동은 산포된 채로 남음.
- **Closed-loop LIBERO 평가**(spatial suite 10 task × 10 rollout/method, 단일 NVIDIA 5090): gripper state(open/close), end-effector height(low/high), 파생량인 end-effector speed(slow/fast, v=‖Δx,Δy,Δz‖/dt) 세 특징을 no-intervention / prompting(유리한 초기조건) / control(제안기법)과 비교.
  - gripper: 거의 완벽한 constraint 만족 + SR 90%↑ 유지(Fig.6).
  - height: 거의 완벽한 constraint 만족, 제약이 태스크를 본질적으로 더 어렵게 만들어 SR 소폭 하락(Fig.7-8).
  - speed: 느리게(slow) 유도는 신뢰도 높으나 빠르게(fast) 유도는 정확도 낮음(고속 구간 학습데이터 부족 추정, Häon et al. 2025와 동일 현상 재확인) — SR은 거의 완벽 유지(Fig.9-10).
  - 핵심 주장: 선행연구([5])는 constraint 만족 또는 steerability 중 하나만 따로 보고했지만, 이 논문은 **constraint satisfaction과 closed-loop SR을 항상 함께(trade-off curve로) 보고** — LLM steering의 "naturalness 보존" 원칙을 VLA의 "task 성공률 보존"으로 옮긴 방법론적 기여.

## activation-steering 흐름에서의 위치 (VLA 해석+제어)
LLM 진영의 선형표현가설(Park et al. 2024)·activation addition(Turner 2023)·ITI(Li 2023)·특히 저자 본인의 이전 작업인 "linearly controlled generation with performative guarantees"(Cheng & Amo Alonso 2024)를 VLA로 최초 형식화 이식한 논문이다. VLA mech-interp 계보에서: Lu et al.(선형 프로브로 상태 디코딩, 개입 없음) → **Häon et al. 2025(FFN neuron steering vector, 우리가 이미 재현한 CoRL2025 논문, constraint/steerability 분리 평가)** → **본 논문(observer+controller 형식화, 제어이론 정의, SR-constraint 공동평가, SAE는 future work로 명시)** → **Swann et al. 2026 SAE_VLA_pi05(같은 저자군 Buurmeijer/Swann 참여, 본 논문의 future-work 제안을 받아 SAE 기반 dictionary-learning으로 발전)** 순으로 이어진다. 즉 이 논문은 "raw FFN 개입"과 "SAE 기반 sparse feature 개입" 사이에서 **선형 프로브+제약최적화 폐형해**라는 중간 지점을 형식적으로 정립한 노드다.

## 우리 프로젝트 연결 (feature 관찰·제어 접근 비교)
- **VL→DiT 직렬 결합의 실증 사례**: π0.5는 VLM transformer backbone(우리 프로젝트의 VL pathway에 대응)과 별도 flow-matching action expert(DiT류)로 구성되는데, 이 논문은 **VLM backbone만 개입하고 flow-matching 층은 손대지 않는다**. 그런데도 gripper/height/speed 같은 DiT급(motor) 출력이 안정적으로 바뀐다 — 이는 우리가 이미 우려하는 "Eagle→VL-SA→DiT 직렬이라 pathway를 따로 스티어링해도 진짜 독립이 아니다"라는 캐비어트를 실증적으로 뒷받침하는 사례로 읽을 수 있다. 우리도 VL pathway steering의 DiT 출력 leak을 정량화할 필요가 있다.
- **개입 스케일 차이**: 우리 method(C_steer=C_success∧¬C_failure, h'=hMᵀ)는 succ/fail 활성화 분포에서 학습한 다차원 contrastive conceptor(subspace 사영)인 반면, 이 논문의 controller는 1차원 구간 제약(D=[ζmin,ζmax])에 대한 폐형해로 명시적 저자가 다차원 D는 "일반적으로 비볼록"이라 트랙터블하지 않다고 인정한다(§IV-B). 즉 우리 conceptor 접근은 이 논문이 스스로 한계로 남긴 다차원 목표집합 문제를 다루는 셈 — 서베이에서 "1D 폐형해 대 다차원 conceptor"로 대비시킬 수 있다.
- **phase-matching 프로토 사례**: closed-loop 평가에서 첫 15~25 step만 개입하는 constraint window는 온라인으로 phase를 식별한 것이 아니라 사람이 고정한 값이지만, 결과적으로 우리 phase-matched steering의 가장 단순한 baseline(고정 구간 steering)에 해당한다. 그 조차 constraint 강도에 따라 SR이 사다리형으로 변하는 trade-off 곡선(Fig.8/10)을 보이는데, 이는 우리 사다리식 ablation(global→pathway-split→phase-bin) 검증에서 결과를 어떤 곡선 포맷으로 보고할지 참고할 수 있다.
- **관측값 정의의 차이**: 이 논문의 ζ는 사람이 사전 정의한 해석 가능한 물리량(위치/자세/gripper/속도)이라 관측자 학습이 지도(supervised) 회귀/분류로 바로 된다. 우리 succ/fail 방향은 사람이 사전 정의할 수 없는 데이터 기반 대비(contrastive) 방향이라 같은 선형 관측자 프레임을 쓰더라도 라벨(D 목표집합)의 성격이 근본적으로 다르다 — "관측 가능성"의 정의(정의1)는 그대로 빌려올 수 있지만 "무엇을 D로 둘 것인가"는 우리 쪽이 더 어렵다.

## 면접 포인트 (Q→A)
1. Q: "feature-observability와 feature-controllability를 왜 별도 정의로 나눴나?" A: "고전 제어이론(Kalman 1960)처럼 두 성질이 독립적이기 때문이다. 관측자 f_ℓ(x_ℓ)=ζ가 존재해도(정의1), 그 표현을 원하는 목표집합 D로 밀어넣는 제어자 g_ℓ가 반드시 존재하는 것은 아니다(정의2). 다만 이 논문이 제안하는 controller는 observer 출력을 그대로 사용하므로 실제로는 controllable layer 집합이 observable layer 집합의 부분집합이 되도록 제한한다."
2. Q: "controller의 폐형해가 일반적인 activation addition(ActAdd/CAA)과 다른 점은?" A: "ActAdd류는 고정 크기 벡터를 항상 더하지만, 이 논문의 u_ℓ은 ‖u‖²을 최소화하며 관측값이 목표구간 [ζmin,ζmax]에 들어가게 하는 제약최적화의 폐형해다(Cheng & Amo Alonso 2024 Theorem 4.1 재사용). 현재 활성화가 이미 목표 구간 안에 있으면 개입량이 0이 되는 최소-노름·조건부 개입이라는 점이 다르다."
3. Q(우리 프로젝트): "이 논문이 π0.5의 flow-matching 층을 건드리지 않고 VLM backbone만 개입해도 gripper/height/speed 같은 motor-level 출력이 바뀌는 게 우리 pathway 분리 가정에 무엇을 시사하나?" A: "우리가 이미 우려한 VL→DiT 직렬 결합(pathway를 따로 스티어링해도 진짜 독립이 아님)을 실증적으로 뒷받침한다. VL 표현만 개입해도 DiT급 출력이 바뀐다는 것은 두 pathway가 downstream에서 강하게 결합돼 있다는 방증이며, 우리도 VL steering의 DiT leak을 정량화해야 한다는 근거가 된다."
4. Q: "이 논문의 controller가 다차원 목표집합으로 확장 가능한가?" A: "저자들 스스로 '일반적으로 식(6b)의 제약이 비볼록일 수 있다'고 인정하며, 폐형해는 D가 1차원 구간(box constraint)일 때만 성립한다고 명시한다. 우리 project의 다차원 contrastive conceptor(C_success∧¬C_failure)는 이 논문이 미해결로 남긴 다차원 목표집합 문제에 해당하는 접근이다."
5. Q: "closed-loop 평가에서 speed를 빠르게 유도하는 것이 느리게 유도하는 것보다 왜 어려운가?" A: "저자들은 학습 데이터에 고속 구간 예시가 적기 때문일 것으로 추정하며, 이는 Häon et al.(2025)의 동일 현상 재확인과 일치한다. 이는 activation steering 일반의 한계 — 개입이 훈련 분포 밖 영역으로 표현을 밀어낼 때는 효과가 약해진다는 점 — 를 다시 보여주며, 우리 succ/fail steering 벡터도 fit 데이터 분포를 벗어나는 방향에서는 약해질 수 있음을 시사한다."

## 한계·비판
- **다차원 목표집합에 대한 일반해 없음**: 폐형해는 D가 1차원 구간일 때만 성립, 다차원/비선형 관측자로 일반화되면 트랙터블하지 않다고 저자 스스로 §IV-B에서 인정.
- **flow-matching/diffusion head 자체는 다루지 않음**: π0.5의 action expert 내부는 개입 대상에서 제외 — VLM backbone 개입의 downstream 효과만 관측, end-to-end 개입은 future work로 명시(우리 GR00T DiT block hook 같은 실험은 이 논문 범위 밖).
- **저수준 특징에 국한**: 상태·행동(위치, 자세, gripper, 속도)만 다루고 task-goal·object affordance·관계형 특징 같은 상위 의미론은 명시적으로 future work로 유보 — 우리의 VL="goal" 쪽 스티어링 근거로는 아직 못 쓴다.
- **아키텍처 간 견고성 불균형**: OpenVLA는 π0.5보다 관측 견고성이 떨어짐(delta yaw가 섭동에 불안정) — 두 아키텍처 모두에 일반화된다는 주장이 다소 약화.
- **speed 조절의 방향 비대칭**(느리게는 잘 됨, 빠르게는 잘 안 됨)은 훈련 데이터 분포 편향으로 추정만 할 뿐 원인 규명 실험은 없음.
- **관측자 robustness는 경험적 확인뿐**: ε-perturbation에 대한 안정성(Remark 1)을 사후 검증했다고만 서술, 원칙적 보장(bound)은 없음 — 저자도 안전성 보장 확립을 future work로 명시.
- **평가 규모가 작음**: closed-loop 실험은 10 task × 10 rollout/method 수준(Libero spatial suite)이라 SR-constraint trade-off 곡선의 통계적 신뢰구간이 넓을 수 있음(Fig.8/10의 타원은 표준편차만 표시).
- constraint window(첫 15~25 step)는 온라인 phase 식별이 아니라 사람이 고정한 값 — 우리가 문제삼는 "online phase/failure-type 식별" 자체는 이 논문에서 다루지 않는다.
