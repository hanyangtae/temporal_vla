# Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies (Jin et al. 2026)

- 출처: arXiv 2605.17204v1 [cs.RO], 17 May 2026 (Xinchen Jin, Aditya Chatterjee, Pranav Kumar, Rohan Paleja — Purdue CS) · PDF: `docs/references/Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies.pdf` · 섹션=§6 VLA(서베이 배정) — 논문 전체(§1~5 + Appendix A~L) 정독 · tier=must · 한줄역할: SAE feature 후보 선정을 rollout에서 뽑은 **행동 이벤트(kinematic keyframe→task-local event cluster)**에 anchoring해 SAE 활성 통계에 기대지 않고 전체 alive feature를 스코어링하고, closed-loop zero-out/soft 개입으로 causal 검증까지 완결한 파이프라인 — 우리의 phase/event 조건부 개입에서 "event 신호원" 후보.

## 문제·동기
LLM/VLM mech-interp 도구(logit lens, SAE 텍스트 라벨링)가 VLA에 그대로 안 옮겨진다: 출력이 텍스트가 아니라 action이라 vocab projection이 신뢰할 의미를 못 주고, closed-loop 검증은 비싸다(1-step 개입은 semantic readout이 없어 읽기 어렵고, multi-step 개입은 매 스텝이 정확히 맞아야 함). 동시연구 Swann et al.(SAE_VLA_pi05)은 SAE 활성 통계에서 먼저 눈에 띄는 feature를 뽑고 rollout video와 수동 대조로 라벨을 붙인다 — behavior가 해석 단계에만 들어가고, 수작업이라 전체 SAE basis를 커버 못 하며 시각적 co-occurrence는 causal 근거가 아니다. 이 논문은 SAE 활성과 완전 독립적인 kinematic 신호로 이벤트를 먼저 정의하고, 모든 alive feature를 그 이벤트에 대해 자동 스코어링해 이 공백을 메운다.

## 핵심 아이디어
4단계 파이프라인(Figure 1): (1) BatchTopK SAE를 (policy stream, LIBERO suite, layer)별로 개별 학습 (2) AWE(Automatic Waypoint Extraction)로 SAE와 무관한 kinematic keyframe 추출 (3) keyframe을 visual+state+temporal-progress 임베딩으로 task-local 클러스터링(+VLM phrase/phase 라벨, OpenVLA만) (4) 4개 랭킹 전략(event-aligned/window-mean/task-mean/random-alive)으로 feature 스코어링 후 top-K를 closed-loop residual-preserving latent edit으로 causal 검증. 매 단계가 분석 대상 SAE 활성과 독립인 신호(궤적 kinematics)로 앵커링된다는 것이 골자.

## 방법 (SAE feature를 action event에 grounding)
- **SAE**: BatchTopK(k=64), post-block residual stream, per-token(mean-pool 안 함 — Grant et al./NOTALL과 일치). fidelity 판정은 FVE/alive%/L0뿐 아니라 **Hooked SR**(reconstruction-only hook 하에서 closed-loop SR)로 최종 확정 — OpenVLA는 layer31만 SR 생존(34.8%, baseline~70%), π0.5는 전 층 ≥95% 유지(discrete action-token vs continuous flow-matching의 강건성 차이).
- **이벤트**: AWE가 end-effector 궤적을 오차예산 η=0.05 내 waypoint로 압축(SAE·의미 라벨과 완전 독립) → task-local agglomerative clustering(visual/state/temporal-progress 가중 코사인거리, threshold 0.18, λv>λs>λp) → 에피소드 50% 이상 재현되는 클러스터만 채택 → OpenVLA만 VLM(gemini-3.1-pro-preview)으로 phrase+phase(pre_grasp/immobilization/contact/detach/post_grasp/transition) 라벨(시각화 전용, 랭킹·개입에는 미사용).
- **랭킹 4종**: event-aligned(±w=5 window 내 pulse/step-up/step-down 3템플릿 매칭, window-mean 차감으로 baseline level 배제), window-mean(같은 window의 평균활성, 시간모양 무시), task-mean(전체 rollout 평균활성, 이벤트 무관), random-alive(하한 통제, 다른 세 랭킹과 배타).
- **개입**: hard zero-out(α=0)과 soft(α∈[0,1] 연속 스윕) 두 모드. **residual-preserving latent edit** x'=x+Dec(z')−Dec(z)=Dec(z')+err(x) — SAE로 설명되는 성분(z)만 편집하고 재구성오차 err(x)는 그대로 보존. Swann et al.의 고정 decoder-vector 덧셈(x'=x+α·d_i, 샘플 무관)과 달리 현재 샘플의 코드값에 조건화된 개입이며, Appendix K에서 decoder-vector 방식(α=150)이 ρ=‖Δx‖/‖x‖=1.54로 native residual을 압도(overwrite)함을 직접 실증해 대비.

## 실험·결과
- **OpenVLA layer31**: event-aligned 랭킹이 가장 큰 SR 하락(baseline 70.0%→48.8%, −21.2pp), window-mean(−6.2)·task-mean(−6.5)·random(−1.3)과 뚜렷이 분리, top-3 feature가 하락분 대부분 담당.
- **π0.5 PG(VLM backbone)**: 전 층·전 랭킹에서 단일 feature 편집이 baseline 근접(최대 −2.7pp, layer11) — cross-attention KV-cache가 여러 prefix 토큰을 섞어 단일 feature 편집을 희석.
- **π0.5 AE(action expert)**: 거의 모든 (layer, 랭킹) 조합에서 SR 0% 근처로 붕괴(random-alive조차 파괴적: layer0 −0.7 vs layer5 −23.4) — event-specific selectivity 아니라 broad causal sensitivity, Swann et al.의 PG/AE 비대칭과 일치.
- **soft α 스윕**: π0.5 AE는 좁은 α 구간에서 급전환(스위치形, 다이얼 아님) — 실로봇 튜닝 시 근소한 과교정도 위험(arm-instability 유발).
- **target/off-target 특이성 probe**(OpenVLA): event-selected feature zero-out 시 target task SR(0.641)≈off-target SR(0.631) — task-private 회로가 아니라 approach/grasp/transport/release 등 suite 내 공유 조작 회로.
- **실로봇**(Mobile ALOHA, LoRA π0.5, red/yellow chip approach): AE layer17 SAE suppression이 가장 강한 dose-response. Häon et al. FFN value-vector steering baseline은 prompt-only보다 훨씬 noisy(색상 방향이 vocab space엔 있어도 action에 깨끗이 전달 안 됨) — 언어-행동 subspace 부분 얽힘의 증거이자 event-grounded SAE 채택 동기.

## activation-steering 흐름 위치(VLA SAE 계열)
Cunningham/Gao(TopK+AuxK) LLM SAE 계보를 VLA에 이식하는 흐름에서: Lu et al.(프로브만, 개입 없음) → Häon et al. 2025(FFN neuron steering, 우리가 재현한 CoRL2025) → Buurmeijer et al. 2026(선형 observer/controller, SAE는 future work로 제안) → Swann et al. 2026(SAE 학습+SAE-내부 통계 기반 general/memorized 분류, 개입은 고정 decoder-vector 덧셈) → **본 논문(동시연구)**: SAE 활성과 독립적인 행동 이벤트로 후보를 선정해 전체 alive feature coverage 확보 + sample-dependent residual-preserving edit. 같은 notes 폴더의 SAE_VLA_pi05·ObservingControlling과 저자 계보가 겹치는(상호 인용) VLA-SAE 소그룹의 최신 노드.

## 우리 프로젝트 연결(event=phase 신호 후보, NOTALL/SAE_VLA와 비교)
- **event=phase 신호 후보**: AWE keyframe→클러스터의 VLM phase 라벨(pre_grasp/immobilization/contact/detach/post_grasp/transition)은 우리가 찾는 online phase 식별의 오프라인 ground-truth 후보로 재사용 가능하다. 단 이 논문은 phase 라벨을 랭킹·개입에 안 쓰고 시각화에만 쓴다(§Limitations ii) — 실제로 phase-conditional steering에 쓰려면 이 라벨을 신호로 승격시키는 작업이 우리 몫으로 남는다.
- **pathway 비대칭이 세 번째로 재현**: π0.5 PG(VL)는 단일 feature 편집에 둔감, AE(DiT류)는 과민 — 우리 VL="goal"/DiT="motor" 분리 및 DiT fragility(NOTALL: GR00T DiT −68pp@9x) 가정과 독립적으로 재확인. 다만 이 논문 AE 결과는 event-aligned도 random-alive도 거의 다 붕괴라 "phase-matched selectivity"의 증거로는 약하고, 오히려 DiT는 무엇을 건드려도 깨진다는 SAE_VLA_pi05식 fragility 경고를 강화한다.
- **개입 연산자 설계 참고**: residual-preserving latent edit(x'=Dec(z')+err(x))은 우리 conceptor steering(h'=h·Mᵀ, C_success∧¬C_failure)과 마찬가지로 현재 샘플 활성값에 조건화된 개입이라는 점에서 Swann et al.의 고정 decoder-vector보다 철학적으로 가깝다 — 다만 우리는 SAE dictionary가 아니라 raw activation subspace에서 직접 conceptor를 fit한다는 차이가 있다.
- **event 경계 추출법의 재사용 가능성**: 우리는 현재 phase를 rollout 길이 비율/fixed timestep으로 근사하는데, 이 논문의 AWE류 kinematic waypoint 압축은 순수 궤적 신호로 이벤트 경계를 뽑는 대안 방법론 하나로 참고할 수 있다.
- **공유 실패 zone 경고와 정합**: target/off-target 특이성 결과(0.641 vs 0.631)는 이 논문의 event feature가 task-private이 아니라 공유 조작 회로임을 보여준다 — 우리 succ/fail conceptor 방향도 task 특이적 실패가 아니라 여러 task가 공유하는 실패 zone(메모리: seen18 shared failure zone)에 올라탈 위험을 함께 점검해야 한다.

## 면접 포인트 (Q→A)
1. Q: "이 논문의 이벤트 anchoring이 Swann et al.(SAE_VLA_pi05)과 근본적으로 다른 점은?" A: "Swann et al.은 SAE 활성 통계에서 먼저 눈에 띄는 feature를 뽑고 활성 궤적을 rollout video와 수동 대조해 라벨을 붙인다 — behavior가 해석 단계에만 들어가 전체 SAE basis를 커버 못하고 시각적 co-occurrence는 causal 근거가 아니다. 이 논문은 SAE 활성과 완전 독립적인 kinematic keyframe(AWE)으로 이벤트를 먼저 정의하고 alive feature 전부를 자동 스코어링한 뒤 closed-loop zero-out으로 causal 검증까지 한다 — coverage와 causal 근거를 이벤트 신호 하나로 함께 얻는다."
2. Q: "OpenVLA와 π0.5에서 개입 결과가 왜 이렇게 다른가?" A: "OpenVLA는 action을 7개 토큰×256bin으로 이산 양자화해 하나의 잘못된 bin이 self-reinforcing stall로 이어지는 반면, π0.5의 연속 flow-matching action chunk는 작은 재구성 오차에 강건하다. 그래서 OpenVLA는 layer31만 Hooked SR이 살아남고(34.8%), π0.5는 전 층 ≥95% 유지된다 — 이후 causal intervention에서도 OpenVLA는 랭킹 간 뚜렷이 분리되고 π0.5 AE는 랭킹 무관하게 거의 다 붕괴하는 차이로 이어진다."
3. Q(우리 프로젝트): "이 논문의 VLM phase 라벨을 우리 phase-matched steering에 바로 쓸 수 있나?" A: "라벨 체계(pre_grasp/immobilization/contact/detach/post_grasp/transition) 자체는 재사용 가능한 phase taxonomy 후보지만, 저자들도 명시하듯 이 라벨은 시각화 전용이고 랭킹·개입 스코어링에는 전혀 안 들어간다 — 우리가 online phase 신호로 쓰려면 이 라벨을 오프라인 ground truth로 두고 우리 activation 기반 online phase classifier를 그에 대해 검증하는 방식으로 가져와야 한다."
4. Q: "residual-preserving latent edit이 decoder-vector steering보다 나은 이유는 수식으로 어떻게 설명되나?" A: "decoder-vector steering은 x'=x+α·d_i로 샘플의 현재 코드값과 무관하게 고정 벡터를 더한다. 이 논문은 x'=Dec(z')+err(x) — SAE로 설명되는 성분(z)만 목표대로 스케일링하고 SAE가 설명 못하는 재구성오차는 보존한다. Appendix K에서 decoder-vector steering(α=150)이 ρ=‖Δx‖/‖x‖=1.54로 native residual을 압도함을 보여, 자기 방식이 이런 overwrite를 피한다고 논증한다."
5. Q: "이 논문이 보고하는 '실패'가 우리 프로젝트에 주는 시사점은?" A: "target/off-target 특이성 probe에서 event-selected feature를 zero-out해도 target task SR(0.641)과 off-target task SR(0.631)이 거의 같다 — 이 feature들은 task-private 회로가 아니라 approach/grasp/transport/release 같은 suite 전역 공유 조작 회로다. 우리 succ/fail conceptor 방향도 특정 task 고유 실패가 아니라 여러 task가 공유하는 실패 zone에 올라탈 위험이 있다는 경고로 읽을 수 있다."

## 한계·비판
- **binary SR만 결과 지표**: 개입이 행동을 얼마나 다르게 바꿨는지 세분화 못하고 성공/실패 이분만 본다 — 저자도 명시적 한계로 인정, finer-grained closed-loop metric을 future work로 남김.
- **단일-feature 개입에 국한**: multi-feature edit은 효과가 얽혀 귀속이 안 돼 본문에서 배제(joint dropout은 Appendix H로 격하) — 여러 feature 조합 steering의 causal 분해는 미해결.
- **VLM phase 라벨 미활용**: phase taxonomy를 만들어놓고도 실제 랭킹·개입 스코어링에는 전혀 안 씀 — 의미 구조를 갖췄지만 활용은 시각화에 그침.
- **OpenVLA에서도 task-private이 아님**: target/off-target SR 차이가 거의 없어(0.641 vs 0.631), 이벤트 grounding이 "task별 고유 실패 원인"까지는 못 잡고 suite 전역 공유 조작 회로만 잡는다는 자기 한계.
- **실로봇 실험은 안전 제약으로 conservative suppression만**: 강한 개입은 시뮬레이션 붕괴·실로봇 큰 arm displacement 위험이라 안전상 약한 개입만 테스트 — steering 상한 검증 자체가 안전 문제로 막혀 있음.
- **π0.5 AE에서 랭킹 전략 간 구별 실패**: event-aligned/window-mean/task-mean 모두 붕괴 floor(~−96pp)에 몰려 어떤 랭킹이 "진짜" causal feature를 골랐는지 이 실험만으로는 판별 불가.
- 4-suite LIBERO + 단일 real-robot task(2지선다 색상)로 스케일이 작음 — cross-architecture 일반화(π0.5/OpenVLA 2종)도 NOTALL(6개 아키텍처)에 비해 좁음.
