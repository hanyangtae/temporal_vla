# Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation (Das et al. 2025)

- 출처: Das, Chiu, Huang, Lindemann, Sukhatme (USC / ETH Zürich) · arXiv:2509.20623v2 [cs.RO] (2026-06-01 갱신, project page lae-robotics.github.io) · PDF: docs/references/LAE_LatentActivationEditing_2509.20623.pdf · 섹션=전체 정독(논문 총 10p; **배정된 "§6 VLA"는 실물에 존재하지 않음 — VI는 CONCLUSION이고 논문 전체가 VLA를 다루지 않는 순수 RL multi-quadrotor 사례. 아래는 실제 내용 기준으로 작성, "한계·비판"에 재기록**) · tier=must · 한줄역할: online classifier로 위험 latent 검출 후 학습된 latent world model로 activation을 치환·개입하는, "검출→라우팅→편집" 게이팅 구조를 로봇 제어 정책에 최초로 실증한 사례.

## 문제·동기
데센트럴라이즈드 multi-quadrotor 충돌회피 RL 정책은 평균 성능은 강하지만 obstacle-rich 환경의 edge case에서 드물지만 치명적인 충돌이 잔존한다. 재학습/파인튜닝은 비용이 크고 catastrophic forgetting 위험이 있으며, 이미 강한 정책은 재최적화해도 한계효용이 작아(asymptotic plateau) "95%→99.9%" 같은 마지막 gap을 메우기 어렵다. LLM activation steering·비전 latent editing에서 영감을 받아, 가중치·구조를 바꾸지 않고 추론 중 latent만 편집해 특정 행동(안전)만 선택적으로 고치는 것이 목표.

## 핵심 아이디어
LAE(Latent Activation Editing) = 두 단계 게이팅 파이프라인: (i) 중간 latent activation Z_t를 감시하는 online 이진 분류기 B_w(safe/unsafe), (ii) unsafe로 flag되면 편집모듈 E_θ가 Z_t를 대체 Z'_t로 치환 후 downstream(가중치 고정) 네트워크에 그대로 전달, safe면 무개입 통과. 핵심 가설: "정책의 내부 위험 인지를 인위적으로 증폭하면 더 이르고 조심스러운 회피가 유도된다" — 이를 위해 충돌 직전 latent 궤적을 예측하는 action-free Latent Collision World Model(LCWM, GRU)을 편집기로 써서, 현재 latent를 그 모델이 예측한 "미래(더 위험한) latent"로 치환한다.

## 방법(중간 activation online classifier 검출→activation edit 개입, 드론 충돌회피)
- 정책 구조: obs(self/neighbor/obstacle)를 3개 MLP로 인코딩 → neighbor/obstacle은 multi-head attention으로 fuse → concat한 Z1(fused latent, d=30) → downstream MLP → 액션헤드 직전 Z2 → 4-rotor thrust. Z1, Z2가 편집 후보.
- 데이터 수집: QuadSwarm sim rollout에서 (Z_t, τ, t)만 기록. 충돌시각 집합 C(τ)로부터 time-to-collision heuristic 라벨링: 충돌로부터 H step 이내면 unsafe, 아니면 safe(식1) → D 구축(이 D로 분류기와 LCWM 시퀀스 데이터 모두 생성).
- 분류기 B_w: latent→{safe,unsafe}, BN+ReLU+dropout MLP, 지도학습.
- LCWM 학습: collision-bearing trajectory만 사용. 충돌 전 윈도우 [t_c−H, t_c] 내 각 t에서 n-step 히스토리 버퍼 Z_h=[Z_{t−n},...,Z_t] → 타깃 Z_{t*}, t*=min(t+m, t_c)(m step 미래, 충돌시점 넘지 않게 clamp). GRU: h_i=GRU(h_{i−1}, Z_i), Z'_t=W h_t+b. loss=MSE(식4).
- Inference loop(Algorithm 1): 매 step B_w(Z_t) 예측 → safe면 그대로 forward하고 버퍼 리셋; unsafe면 버퍼에 append, 버퍼가 n에 차면 LCWM으로 Z'_t 생성해 그것으로 forward.
- 결정론적 시뮬레이터(모터·센서 noise, action sampling 고정)로 재현성 확보, base policy가 최소 1회 충돌하는 2600개 env config로만 평가(이미 안전한 케이스 제외해 결과 부풀림 방지).

## 실험·결과
- Base RL policy: 2600 config 누적 5,623 충돌, zero-collision traj 0개, 평균 success rate 0.58.
- LCWM(GRU): 583 충돌(−89.6%), zero-collision 2,175개(83.7%), success rate 0.64(+10.3% relative). paired t-test 평균 감소 1.94/run, 95% CI [1.86,2.01], p<1e-300, Cohen's d=1.0. 비결정론 시뮬레이터(10회 반복)에서도 동일 경향 재확인.
- 대안 편집전략 비교(Table I): KD-tree retrieval(2,678충돌, 메모리기반이라 일반화 약함), SAE steering(2,896충돌, 로봇 latent에서 unsafe 표상이 sparse unit으로 안 깔끔히 분리돼 hparam에 매우 민감), UMAP/Barlow-Twins/AE 압축 편집(3,766/3,355/37,749충돌, AE는 완전 붕괴), Transformer-LCWM(612충돌, GRU와 비슷하나 연산비용 커 실시간 부적합) — GRU-LCWM 최상.
- Ablation(편집 대상 latent): Z1_Partial(self-dynamics 성분은 보존, neighbor/obstacle 성분만 편집)이 최선(583충돌); Z1_All(자기 동역학까지 편집)은 67,951충돌로 폭발(동역학적으로 불가능한 궤적); Z2_All(action head 직전, 이미 비선형 통과해 얽힘)은 더 나쁨.
- Horizon H(unsafe 라벨 윈도우): H=50 너무 늦음(1,124충돌) → H=250 최적(583충돌, success 0.64) → H=300 소폭 악화.
- 예측 horizon m: m=10 최적, m≥25부터 발산(m=30에서 60,547충돌) — 너무 멀리 예측하면 정책의 실제 latent dynamics를 벗어나 불안정.
- 실사 Crazyflie 2.1: LAE 모듈(2-layer MLP classifier 64-hidden ~2k param + GRU editor hidden 32 ~7k param, 총 10k 미만 param, fp32 40KB 미만) C로 재구현, STM32에 탑재, <1ms/step로 1kHz 안정화 루프와 호환. 4-드론 bilateral/four-way crossing에서 baseline은 충돌, LAE는 전원 무충돌 도달.

## activation-steering 흐름 위치(로봇 정책 inference-time editing)
LLM activation steering(ActAdd, RepE)·SAE 계열의 "학습된 방향/유닛으로 hidden state를 편집"하는 계보를 비-LLM 로봇 제어 정책(MLP+attention fusion RL policy)에 최초로 옮긴 사례. 다만 편집 메커니즘은 고정 벡터를 더하는 additive steering이 아니라, 학습된 forecasting 모델(LCWM)이 예측한 미래 latent로 현재 latent를 통째로 치환(replace)한다는 점에서, "concept vector 덧셈"보다 world-model 기반 predictive latent 편집에 가깝다. online classifier(위험 검출) → 조건부 activation edit(unsafe일 때만 개입, safe는 passthrough)이라는 게이팅 구조는 정확히 검출→라우팅→개입 파이프라인이며, 이 위상을 로봇 정책에서 실증한 초기 사례다.

## 우리 프로젝트 연결(검출→라우팅→개입 구조 대조; 도메인/신호 차이)
- 구조적 동형: (검출기 B_w: latent→safe/unsafe) ↔ (우리: online pathway/phase/failure-type classifier); (편집 E_θ: unsafe latent 대체) ↔ (우리: contrastive conceptor C_steer=C_success∧¬C_failure로 h'=h·Mᵀ steer). 둘 다 always-on이 아니라 검출될 때만 개입하는 게이팅 구조.
- 도메인/신호 차이(핵심): (1) LAE는 RL 정책 — 저차원(d=30) 단일 timestep 벡터, 우리는 VLA(VLM+diffusion DiT) — per-token/per-block 고차원 latent에 chunk 예측. (2) LAE의 unsafe 라벨은 시뮬레이터에서 값싸게 얻는 "실제 충돌까지의 time-to-collision" ground-truth인 반면, 우리 라벨(성공/실패, phase, pathway 귀인)은 훨씬 약하고 노이즈가 큼(실패=timeout confound 등 — 우리 프로젝트의 seen18 길이confound 문제와 유사 구조의 hindsight 라벨링 위험). (3) LAE는 예측된 미래(더 위험한) latent로 현재 latent를 통째로 대체하는 world-model 치환인 반면, 우리는 성공 subspace로의 conceptor soft projection이라 편집 목표 표상 자체가 다르다.
- 가장 유용한 교훈: LAE의 핵심 ablation(자기 동역학 성분을 건드리면 정책이 파국적으로 붕괴, Z1_Partial만 안전)은 "일부 latent 성분만 선택적으로 편집해야 한다"는 근거이며, 우리의 pathway 분리 steering(VL/DiT를 따로, DiT는 phase-matched)과 구조적으로 공명한다 — 편집 범위를 잘못 잡으면(전체 latent를 편집) 개입 자체가 제어 붕괴를 유발할 수 있다는 경고.
- existence proof: LAE는 온라인 검출→라우팅→개입이 실사(마이크로컨트롤러, 1kHz, <1ms)에서도 작동함을 보여, "online 식별이 실제 latency budget 안에서 가능한가"에 대해 최소한 저차원 RL 정책에서는 긍정적 선례를 제공한다. 다만 VLA의 고차원·다중 서브모듈(Eagle→VL-SA→DiT) 직렬 구조로 그대로 일반화될지는 별개 문제.

## 면접 포인트(Q→A)
1. Q: LAE의 "editing"은 통상 activation steering(고정 벡터 additive)과 뭐가 다른가?
   A: 고정 방향 벡터를 더하는 게 아니라, 학습된 action-free world model(LCWM, GRU)이 과거 n-step 히스토리로부터 "이 latent가 충돌로 이어질 경우 m step 후 어떤 모습일지"를 예측해 현재 latent를 그 예측치로 통째 치환한다. 미래의 (의도적으로 더 위험한) 상태로 현재를 대체함으로써 정책이 위험을 더 일찍 인지하게 만드는 predictive/forecasting 기반 편집이다.
2. Q: 왜 Z1_All(전체 latent 편집)이 오히려 성능을 파국적으로 악화시켰는가(67,951충돌)?
   A: Z1은 self-dynamics 임베딩과 neighbor/obstacle 임베딩이 concat만 되어 아직 분리 가능한데, self-dynamics 성분까지 편집하면 정책이 로봇 자신의 물리적 상태를 잘못 인지해 동역학적으로 불가능한 명령을 낸다. 안전 표상이 자기-동역학 표상과 얽혀 있으면 "위험만 증폭"하려는 개입이 제어루프를 붕괴시킨다.
3. Q: SAE 기반 steering이 왜 LCWM보다 못했는가?
   A: LLM/비전과 달리 이 로봇 정책 latent에서는 unsafe 관련 정보가 소수 sparse unit에 깔끔히 분리되지 않고 여러 차원에 얽혀 있었다. 그 결과 SAE steering은 dictionary size·sparsity penalty에 성능이 매우 민감했고, 선택된 유닛이 자기 동역학 정보까지 포함해 편집 시 불안정한 거동을 유발했다.
4. Q: unsafe 라벨(time-to-collision heuristic)의 강점과 한계는?
   A: 시뮬레이터에서 충돌 시점이 정확히 관측 가능해 라벨을 값싸게 대량 확보할 수 있다는 게 강점이다. 다만 이는 결과가 이미 알려진 사후적(hindsight) 라벨이라, 실제 causally 위험한 latent와 우연히 충돌 근처에 있었을 뿐인 latent를 구분 못할 수 있다(상관 vs 인과) — 우리 프로젝트의 실패 라벨(timeout confound)과 유사한 구조적 약점.
5. Q(우리 프로젝트): 이 논문이 우리 "online phase/type 식별→라우팅" 문제에 주는 시사점은?
   A: 검출기가 unsafe를 flag할 때만 개입하는 게이팅 구조가 실사 마이크로컨트롤러(<1ms)에서도 작동한다는 existence proof를 준다. 다만 도메인이 저차원 단일벡터 RL latent라 VLA의 고차원·다중 서브모듈 구조로 그대로 일반화되진 않고, 편집 범위(우리는 pathway 분리)를 잘못 잡으면 제어가 붕괴한다는 ablation 경고가 가장 직접적인 교훈이다.

## 한계·비판
- 배정된 "§6 VLA" 섹션은 실물 논문에 없음: 논문은 총 10p이고 VI는 CONCLUSION, 전체가 순수 RL 기반 multi-quadrotor 내비게이션 사례로 VLA(Vision-Language-Action) 언급이 전혀 없다. 서베이 템플릿에서 다른 논문과 섹션 태그가 혼동된 것으로 보이며, 이 노트는 실제 논문 내용을 로봇 정책 일반 사례로 취급해 작성했다.
- 평가가 "base policy가 최소 1회 충돌하는 2600개 config"로 제한되어(이미 안전한 케이스 제외), false-positive 개입에 의한 부작용(불필요한 편집이 성능을 깎는 경우)은 별도로 정량화되지 않는다.
- LCWM은 collision-bearing trajectory에만 학습되어, 학습분포 밖의 새로운 실패 모드에 일반화될지 불명.
- latent 차원 d=30의 저차원·단일 RL 정책 특성상, VLA처럼 고차원·다중 서브모듈(VL encoder, DiT 등) latent로 이 프레임워크가 그대로 확장될지는 미검증 — 저자도 future work로 다른 behavior axis/플랫폼 확장만 언급, VLA는 다루지 않는다.
- 편집이 "미래 위험 예측치로 치환"하는 방식이라 예측 길이 m이 너무 길면(m≥25) 오히려 발산(60,547충돌)하는 등 안정성이 hyperparameter(H, m, n)에 민감하다 — 우리 conceptor 방법도 phase/window 선택에 유사하게 민감할 위험을 시사한다.
