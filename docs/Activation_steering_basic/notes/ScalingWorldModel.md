# Scaling World Model for Hierarchical Manipulation Policies (Qian et al. 2026, VISTA)

- 출처: Long Qian, Yueze Wang, Jiaxi Song 외(공동1저자 3인) · BAAI/XJTU/Tsinghua/NUS/IA-CAS · arXiv:2602.10983v2 [cs.RO] (2026-02-12) · PDF: docs/references/Scaling World Model.pdf · 섹션=§6 world model(배정) — **실물 논문의 §VI은 Conclusion**이라 배정과 어긋남; world model 방법론 자체는 §II-B(관련연구)·§III(문제정의)·§IV-A(World Model Planner)에 있고, 아래는 이 세 절 + §IV-B(GoalVLA)·§V(실험)·§VII(한계) 정독 기반 · tier=must · 한줄역할: activation steering과 무관한 **별개 계열**(생성적 world model이 만드는 discrete visual subgoal로 하위 VLA를 top-down 조건화) — §6 world-model-latent controllability 맥락의 대조군/참고 사례.

## 문제·동기
VLA는 OOD(미학습 물체·장면)에서 취약하고, 특히 실로봇 데이터가 적을 때(본 논문 셋업: 물체 5종, 2시간 수집) 심하다. 기존 계층적 분해는 트레이드오프가 있다: 언어 서브골은 의미적으로 일반화되지만 공간·물리 제약이 없고, dense video prediction은 디테일은 풍부하나 장기 지평에서 temporal drift·물리적 비일관성이 생긴다.

## 핵심 아이디어
world model W가 전역 지시문 L을 텍스트 서브태스크 l_i("무엇을")와 시각적 goal image g_i("어떻게")의 교차 시퀀스로 분해하고, 하위 정책 GoalVLA(π0 기반)가 현재 관측+l_i+g_i를 조건으로 action chunk를 생성한다. 즉 "제어"는 discrete/pixel 공간에서 명시적으로 생성된 goal image를 통해 이루어지며, dense 미래 프레임 대신 sparse key milestone만 예측해 hallucination과 drift를 줄인다.

## 방법
- **World model 표상**: 이미지(IBQ tokenizer)와 텍스트(Qwen3 tokenizer)를 공유 vocabulary로 통합해 S=(φ(I0),φ(L),φ(l0),φ(g0),...)를 하나의 이산 토큰 시퀀스로 만들고, 표준 autoregressive next-token 학습(teacher forcing, causal mask)으로 결합분포를 학습. 추론은 beam search로 서브태스크·goal 토큰을 반복 생성 후 역토크나이저로 픽셀 복원. EMU3.5 체크포인트에서 continued training(2000 step).
- **GoalVLA**: π0 아키텍처(flow matching) 그대로, 입력에 현재관측 토큰+goal image 토큰을 concat. subtask-aware action padding(경계 넘어가는 chunk는 제로패딩)과 random goal image offset augmentation(경계 근방에서 g_i/g_{i+1} 랜덤 사용)으로 stage 경계 불안정성 완화.
- **데이터**: OXE+AgiBot World Beta+Mobile Aloha 1.2M trajectory를 RDP 기반 물리적 상태변화 검출 + Qwen2.5-VL 72B 캡션으로 자동 milestone 라벨링해 15.2B 토큰(14 embodiment) 구축. Any-to-Image 코퍼스(15.0B 토큰)로 co-train.

## 실험·결과
- 실물로봇 pick-place 5물체, 2시간 수집 → π0(baseline), π0-subtask(언어 서브태스크만 대체), Ours(VISTA) 비교, 234 rollout.
- 헤드라인: OOD 21 unseen object 시나리오에서 π0 14% → VISTA 69%.
- Table I: basic(in-domain) Suc는 π0 0.96 > π0-subtask 0.91 > Ours 0.93(오히려 ours가 baseline보다 낮음). unseen distractor 0.73(π0)→0.82(ours). unseen target(가장 어려운 조건) App/Suc: π0 0.40/0.04, π0-subtask 0.73/0.31, Ours 1.0/0.67 — goal image의 이득은 OOD·특히 unseen target에서 집중적으로 발생.
- 정성분석(Fig.9): π0/π0-subtask는 unseen bottle을 학습 시 본 can 모양으로 오인해 파지 실패, goal image가 정확한 접근 pose를 명시적으로 지정해 성공.

## activation-steering 흐름 위치(world model latent 맥락)
이 논문의 "world model latent"는 residual stream의 연속 hidden vector가 아니라, IBQ/Qwen3 공유 vocabulary 위의 **이산 토큰 시퀀스**이며 픽셀로 디코딩되는 goal image다. 개입(제어)은 activation 공간의 벡터 연산(더하기/투영/게이팅)이 전혀 없고, 생성된 symbolic/visual 조건을 하위 정책에 입력으로 주입하는 **conditioning 기반 top-down 제어**다. activation-steering 계보(ActAdd→CAA→conceptor 계열, 우리 h'=h·Mᵀ)와는 메커니즘이 다른 별개 축이며, §6에는 "world model 출력으로 정책 행동을 macro하게 조종할 수 있다"는 상위 개념(behavior steering의 또 다른 실현형)의 대조 사례로 배치되는 것으로 이해했다.

## 우리 프로젝트 연결(직접연결 약함 — 명시)
- **직접연결은 약하다.** activation/latent steering 수식(conceptor, 방향벡터, 게이팅)과 겹치는 부분이 없다. 우리가 빌릴 연산자·수식은 없음.
- **개념적 참고점 하나**: 우리 핵심 난제("online phase/failure-type을 어떻게 식별하는가")를 VISTA는 정면돌파하지 않고 **우회**한다 — world model이 미리 검증 가능한 visual milestone(goal image)을 만들어 두고, "현재 관측이 goal image와 정렬되는가"를 비교하는 subtask switcher로 phase 전환을 외부에서 판별한다. 즉 phase 신호를 policy 내부 activation이 아니라 별도 생성모델의 출력(픽셀 비교)에서 얻는다 — 우리가 겨냥하는 "internal activation만으로 온라인 phase 식별"과는 정반대 해법 방향이라, 우리 문제의 난이도(외부 world model 없이 내부 신호만으로 풀어야 함)를 상대화해서 보여주는 참고 사례 정도.
- steering routing 관점에서 "표현을 어떻게 나눠 개입하는가"(우리는 pathway/phase-bin, VISTA는 subtask-stage) 구조적 유사성은 있지만, VISTA는 그 나눔을 별도 world model의 명시적 출력으로 하고 우리는 동일 policy의 내부 activation에서 읽어야 하므로 방법론 이식 가능성은 낮다.

## 면접 포인트(Q→A)
1. Q: "이 논문이 activation steering 서베이에 왜 들어가는가?" A: "직접적인 activation steering 기법은 아니다. world model이 생성한 discrete visual subgoal로 하위 VLA를 조건화해 행동을 통제한다는 점에서, '중간 표상을 조작해 정책 행동을 원하는 방향으로 유도한다'는 상위 개념(world model latent controllability)을 공유하는 대조 사례로 §6에 배치했다. 개입 지점이 residual stream의 continuous activation이 아니라 discrete/pixel goal이라는 점이 핵심 차이."
2. Q: "VISTA의 이득은 어디서 오는가?" A: "표 1을 보면 in-domain(basic)에서는 π0 대비 오히려 소폭 낮고(0.96→0.93), unseen target처럼 가장 어려운 OOD 조건에서 이득이 집중된다(0.04→0.67). 즉 goal image 조건화는 언어만으로 grounding하기 어려운 미학습 물체의 공간적 파지 pose를 명시적으로 제공하는 데서 이득이 온다."
3. Q(우리 프로젝트 관점): "VISTA의 subtask switcher와 우리 phase-matched steering의 phase 식별 문제는 같은 문제인가?" A: "문제의 겉모습은 비슷하지만(둘 다 '지금 어느 stage인가'를 알아야 함) 풀이 방식이 다르다. VISTA는 world model이 미리 만든 검증가능한 goal image와의 정렬을 외부에서 비교해 해결하고, 우리는 policy 내부 activation만으로 온라인에 풀어야 한다 — 별도 world model 없이 internal signal만 쓰는 우리 쪽이 근본적으로 더 어려운 문제."

## 한계·비판
- 저자 스스로 명시(§VII): 실물 평가가 pick-and-place에 국한, 액체 붓기·변형물체 등은 "잠재력은 있으나" 미검증. goal image가 실제 subtask 종료지점에서 어긋나면(padding/offset augmentation으로 완화하되 완전 해결 아님) GoalVLA 실행 실패로 직결.
- in-domain(basic) 세팅에서는 ours(0.93)가 π0(0.96)보다 오히려 낮다는 수치가 있는데 본문은 이 트레이드오프를 논하지 않음 — world model 조건화가 easy task에는 약간의 부담일 가능성.
- 매 서브태스크 전환마다 world model의 autoregressive beam search + 이미지 토크나이저 디코딩이 필요 — activation steering(1회 forward + closed-form 개입, ms 단위 오버헤드)과 달리 실시간성 비용이 훨씬 클 것으로 추정되나 논문은 latency/연산비용을 전혀 보고하지 않음.
- cross-embodiment transfer 결과(Fig.8)는 world model의 goal-image 생성 쪽 정성 평가이고, GoalVLA(저수준 실행)의 embodiment transfer 자체는 정량 검증되지 않음.
