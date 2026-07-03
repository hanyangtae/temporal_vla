# VITA: Zero-Shot Value Functions via Test-Time Adaptation of Vision-Language Models (Ziakas & Russo 2026)

- 출처: ICLR 2026 conference paper (Imperial College London, Dept. of Computing) · arXiv:2506.10085v5 [cs.CV] (2026-02-27) · PDF: docs/references/VITA.pdf · 섹션=§7 VLA방향(phase 신호) 배정이나 **실물 논문에 §7은 존재하지 않음**(실제 구성: §1 Intro-§2 Preliminaries-§3 Method(VITA)-§4 Experiments-§5 Related Work-§6 Discussion/Limitations/Future Work + Appendix A-G, References로 끝) — "VLA"라는 단어 자체가 본문에 등장하지 않고(제목도 VLM), 가장 가까운 내용은 §6.2 Limitations and Future Work의 "closed-loop control 적용은 향후 과제" 한 문장뿐. 아래는 §2.2(TTT 배경)+§3(방법 전체)+§4(실험)+§6.2(한계) 정독으로 작성 · tier=must · 한줄역할: activation-steering 논문이 아니라 **CLIP 기반 온라인 progress/value 예측기**(test-time training) — 우리 phase-matched steering이 요구하는 "online phase 신호"의 후보 공급원(구 TTA/loop-escape 방향의 VITA 원전).

## 문제·동기
Goal-conditioned value function(관측+언어 목표 → task 진행률 스칼라 [0,1])을 zero-shot으로 추정하려는 시도들은 두 갈래로 갈린다. (1) contrastive VLM(CLIP)의 frame-goal cosine 유사도를 그대로 쓰는 방법(VLM-CL/VLM-RM)은 프레임을 독립적으로 봐서, 시각적으로 비슷하지만 task 진행 단계가 다른 상태(예: 셔츠를 개는 중 vs 펴는 중)를 구분 못한다 — temporal reasoning 부재. (2) autoregressive VLM(GVL, Gemini 기반)은 전체 궤적을 prompt에 넣어 시간 맥락을 쓰지만, pretrain 데이터가 시간순으로 정렬돼 있어 "진행률은 항상 단조증가한다"는 편향을 물려받는다(GVL은 이를 프레임 셔플로 우회하지만, 셔플은 시간 순서 정보 자체를 버리는 대가를 치른다). 둘 다 frozen pretrained representation에 의존해 generalization과 temporal reasoning이 동시에 제한된다.

## 핵심 아이디어
frozen CLIP encoder는 그대로 두고, 그 위에 얹는 **가벼운 adaptation module**을 test-time training(TTT, Sun et al. 2020/2024) 방식으로 매 timestep마다 1-step gradient 업데이트한다. 이 self-supervised loss 자체는 미리 정해진 게 아니라 **meta-learning으로 학습**된다(내부 update가 downstream progress-prediction loss를 실제로 개선하도록 최적화, MAML 계열). 순차적으로(reset 없이) 업데이트를 누적하면 adaptation module의 파라미터 θ_t 자체가 과거 관측 이력을 암묵적으로 저장하는 **implicit memory**가 된다 — RNN의 hidden state나 Transformer의 KV cache처럼 activation에 이력을 두는 대신, **파라미터에** 이력을 둔다는 것이 핵심 차별점. 여기에 인접 프레임 redundancy로 인한 shortcut learning(후반부 시각 패턴에 과적합)을 막기 위해 pairwise-dissimilarity 기반 sub-trajectory 샘플링을 학습 시 적용한다.

## 방법 (progress/value 예측 + test-time adaptation, TTT 방식)
- **입력**: 궤적 τ=(o1,...,oT)와 언어 목표 g를 frozen OpenCLIP ViT-B/32로 인코딩해 z_t=[φ_v(o_t); φ_g(g)] ∈ R^2d(joint concat, element-wise product보다 검증성능 우위로 채택).
- **self-supervised loss(식1)**: 학습 가능한 선형사영 P_K(perturbed input view 생성), P_V(target)를 두고 ℓ_self(z_t; θ_{t-1}, P_K, P_V) = ‖f_adapt(P_K z_t; θ_{t-1}) − P_V z_t‖². reconstruction 형태지만 이 loss 자체가 downstream progress 예측을 돕도록 meta-learn된다(직접 최소화 대상 아님, Sun 2024 방식 계승).
- **test-time update(식2)**: θ_t = θ_{t-1} − η∇_θ ℓ_self(z_t; θ_{t-1}) — 매 timestep 1 gradient step(t_ep=1, η=0.1), reset 없이 순차 누적.
- **value 출력(식3)**: 메타학습된 P_Q로 z_t를 adaptation space로 사영 후 f_adapt(·;θ_t) 통과, frozen 2-layer MLP regression head h로 V(z_t;g)=h(f_adapt(P_Q z_t; θ_t)) 산출. 학습 시엔 정규화된 timestep 라벨 y_t=t/T(§2.1, 전문가 궤적은 단조증가 가정)로 MSE(ℓ_pred) supervised.
- **meta-training**: 총 손실 = ℓ_pred + λ·ℓ_self(λ=0.5). test-time adaptation 업데이트 경로를 통해 ℓ_pred까지 역전파(gradient-based meta-learning, Finn 2017) — θ_0 초기값, P_K/P_V/P_Q, head h를 함께 최적화. **f_adapt는 2-layer residual MLP+GELU, d'=64** — 파라미터 수·연산량 모두 매우 가벼움.
- **dissimilarity-based sampling(식4-5)**: sliding window(w_tr=8, stride s=1)로 후보 sub-trajectory 집합 W를 만들고, 전수 조합 탐색(NP-hard) 대신 각 window에 "다른 모든 window와의 거리 합" 점수를 매겨 상위 k=8개를 선택하는 greedy heuristic(다항시간, GPU cdist로 340 MFLOPs 수준 — 무시할 오버헤드).
- 이 논문은 activation을 편집(steering)하지 않는다 — **파라미터를 test-time에 gradient로 업데이트**하는 방법론이며, 우리가 다루는 h'=h·Mᵀ 식의 activation-space 개입과는 범주가 다르다(TTT 계열 vs conceptor/activation-steering 계열).

## 실험·결과
- **학습**: BridgeData V2 curated subset(2,986 demo, WidowX 250, ToyKitchen 4 configs, pick-and-place만 — fold/sweep/stack 미포함), OpenCLIP ViT-B/32 frozen.
- **분포이동 일반화(Table 1, VOC=predicted progress와 시간 인덱스의 Spearman 상관)**: ID(tk_pnp) 0.782, 환경이동(lm_pnp/fold류/sweep) 0.49~0.73, 임베디먼트 이동(DeepThought 로봇) 0.70~0.82 — CLIP-GRU(명시적 RNN hidden state로 이력 인코딩)와 근소하게 경쟁하며 10개 중 6개에서 앞서고, GVL-0S/1S(Gemini 1.5 Pro in-context)는 전반적으로 크게 하회(특히 stacking/pick-place에서 fold류보다 약함 — autoregressive VLM의 fold 편향 시사).
- **expert vs 비-expert 판별(BinVOC, Table 2a)**: VITA=1.00, GVL도 1.00, CLIP-GRU=0.80 — VITA가 CLIP-GRU보다 우수해 "파라미터 implicit memory가 RNN hidden state보다 temporal shortcut에 덜 취약"하다고 해석.
- **offline RL reward shaping(Meta-World MT10, IQL, Table 2b)**: VITA 기반 dense reward로 학습한 정책이 IQM 0.815로, 시뮬레이터의 fuzzy-logic dense reward(0.779)와 다른 모든 CLIP 계열 baseline을 능가 — **real-world에서 학습한 value 추정기가 시뮬레이션 reward shaping에 zero-shot 전이**됨을 보임.
- **ablation(§4.6.2, Table 6, 우리 맥락에서 가장 중요)**: VITA(순차 implicit memory, reset 없음) vs TTT-TR(전체 궤적 batched 1-step), TTT-RS(매step reset, memoryless), TTT-EX(매step reset, local window batched) — VITA만 VOC 0.6~0.82대, 나머지 세 변형은 모두 0.0~0.21대로 사실상 무의미(붕괴) — **"매 timestep 순차 업데이트 + reset 없음"이 이 방법의 성능 대부분을 만든다**는 강한 증거.

## activation-steering 흐름 위치(phase/progress 신호원)
이 논문은 steering 계열(ActAdd→CAA→conceptor→COAST)과 별개 축인 **test-time training(TTT, Sun et al. 2020/2024) 계열**에 속한다. VITA는 activation을 편집하지 않고 "관측+언어 → 진행률 스칼라"를 예측하는 **외부 monitor 모듈**을 학습·적용하는 것이 전부다. 우리 서베이 맥락에서 이 논문의 위치는 steering 연산자 자체가 아니라, **phase-matched steering이 필요로 하는 online phase 신호를 어디서 얻을 것인가**라는 질문에 대한 한 가지 답 — FIPER(latent OOD 검출)와 마찬가지로 "검출/신호 공급" 축이지만, FIPER는 실패 여부(이진)를, VITA는 진행률(연속값, [0,1])을 준다는 점에서 상보적이다.

## 우리 프로젝트 연결 (phase 신호 공급 보조부품 가능성)
- **직접 연결점**: 우리 메모(project-direction-latent-steering, phase-selective-steering-plan)에 명시된 대로, phase-matched DiT steering의 중심 미해결 문제는 "온라인에 어느 task-phase인지 식별 가능한가"다. VITA의 V(o_t;g)∈[0,1]은 바로 이 phase index 후보 — env-step 카운트(episode마다 길이가 달라 confound, cf. 메모 seen18-rollout-length-confound)보다 **의미 기반(semantic) 진행률**이라 길이 정규화 문제를 우회할 잠재력이 있다.
- **repo 내 기존 접점**: 메모(ttt-code-removed/TTT code status)에 따르면 `src/ttt`에 이미 ProgressHead가 존재하고, "phase-selective steering 실패 시 online 검출 fallback"으로 지정돼 있다 — VITA는 이 ProgressHead 설계의 직접적 선행연구/원형으로 볼 수 있다(TTT 기반 progress predictor라는 점에서 개념적으로 동일 계열). VITA의 두 핵심 설계 원칙 (a) 순차 implicit-memory 업데이트(reset 없음, Table 6에서 압도적 우위) (b) dissimilarity-based sampling(shortcut/redundant-frame 과적합 방지)은 우리가 ProgressHead를 재도입/재학습할 때 그대로 재사용 가능한 레시피다.
- **한계가 곧 우리 gap**: VITA는 VLA의 **내부 latent(DiT/VL-SA residual stream)를 전혀 보지 않는다** — 별도의 frozen CLIP encoder로 RGB+language만 다시 인코딩하는 외부 monitor다. 우리가 최종적으로 원하는 것은 "정책 자신의 activation에서 온라인으로 phase/failure-type을 읽어내는 것"(니치 메모 notall-online-failuretype-niche와 직결)인데, VITA류 접근은 이를 우회해 **외부 보조 모듈을 하나 더 얹는 방식**이라 (i) 추가 인퍼런스 비용(CLIP forward 1회/step) (ii) steer 대상 latent와 다른 표현공간이라는 두 가지 대가를 치른다. 그래도 "일단 phase 신호가 있어야 사다리식 ablation(phase-bin 추가) 실험을 시작할 수 있다"는 점에서, 내부 latent 기반 phase 검출이 막힐 경우의 **fallback 보조 부품**으로는 여전히 유효.
- **failure 판별과의 거리**: VITA의 BinVOC 실험은 expert vs "randomized scripted controller" 궤적을 구분하는 것이지, 실제 policy가 만든 실패 rollout(우리 관심사)을 다루지 않는다 — "실패 시 progress 신호가 어떻게 무너지는가"는 이 논문이 직접 답하지 않으며, 우리가 검증해야 할 부분으로 남는다.

## 면접 포인트 (Q→A)
1. Q: "VITA는 activation steering 논문인가?" A: "아니다. VITA는 CLIP 기반 goal-conditioned value function을 test-time training(TTT)으로 온라인 적응시키는 방법으로, activation을 직접 편집하지 않고 별도의 경량 adaptation module 파라미터를 매 timestep gradient로 업데이트한다. 우리 서베이에서는 steering 연산자가 아니라, phase-matched steering이 요구하는 'online phase 신호'의 후보 공급원으로 배치했다."
2. Q: "VITA가 temporal history를 인코딩하는 방식이 RNN/Transformer와 어떻게 다른가?" A: "RNN은 hidden state에, Transformer는 KV cache에 이력을 activation 형태로 담는다. VITA는 대신 매 timestep 1-step gradient update를 reset 없이 순차 누적해 adaptation module의 **파라미터 θ_t 자체**를 implicit memory로 쓴다. ablation(Table 6)에서 이 순차·무리셋 방식만 유의미한 VOC를 내고, 궤적 전체를 batched로 한 번에 업데이트하거나(TTT-TR) 매 step reset하는 변형(TTT-RS/EX)은 거의 chance 수준으로 붕괴한다 — 순서를 보존한 순차 업데이트 자체가 핵심임을 통제 실험으로 보였다."
3. Q(우리 프로젝트 관점): "VITA를 우리 phase-matched steering 파이프라인에 어떻게 끼워 넣을 수 있나?" A: "V(o_t;g)를 온라인 phase index로 써서, 어느 phase-bin의 C_steer_phase를 적용할지 라우팅하는 역할을 맡길 수 있다. 다만 VITA는 정책 자신의 DiT/VL-SA latent가 아니라 별도 frozen CLIP encoder로 RGB+language를 다시 읽는 외부 monitor라서, steer하는 latent와 신호를 얻는 latent가 분리된다 — 우리가 최종적으로 원하는 '내부 latent에서 직접 phase를 읽는' 목표에는 못 미치지만, 그게 막혔을 때의 fallback 보조 부품으로는 쓸 수 있다. repo에 이미 있는 ProgressHead(src/ttt)가 이 개념의 구현체에 해당한다."

## 한계·비판
- 저자 스스로 인정(§6.2): 매 timestep마다 파라미터를 gradient-update하는 것은 실시간 배포에서 **잠재적으로 안전하지 않을 수 있음** — 연산 오버헤드는 무시할 만하다고 주장하지만, 폐루프 제어(control loop)에 얹었을 때의 안정성·latency 영향은 미검증(§6.2 "future work"로만 언급).
- 평가지표 VOC/BinVOC는 모두 **순서(rank) 정합성**만 잰다 — 예측값의 절대 스케일이 실제 진행률과 얼마나 잘 보정(calibrated)되는지는 안 보여준다. phase-bin routing에 쓰려면 절대 threshold 보정이 추가로 필요.
- 학습·평가 모두 **expert(성공) demo 기반**이며, BinVOC의 "non-expert"도 실제 policy 실패가 아니라 randomized scripted controller 궤적이다 — 우리가 필요로 하는 "실제 실패 rollout에서 progress 신호가 어떻게 붕괴하는가"는 이 논문의 범위 밖.
- "VLA" 실험이 전무하다 — 제목·본문 모두 VLM(비전-언어) 값함수를 다루지, VLA(action) 정책과의 결합·closed-loop 실증은 없다(§6.2에서 "향후 과제"로만 언급). 배정된 "§7 VLA방향" 섹션은 실물 논문에 존재하지 않는다.
- CLIP ViT-B/32라는 비교적 작은/구식 encoder에 의존 — GR00T/pi0 등 최신 VLA의 VLM backbone(Eagle 등)과 표현공간이 달라 그대로 재사용은 어렵고, 우리 latent에 적용하려면 별도 재학습이 필요.
- "재학습 없는 steering"을 표방하는 우리 메인 방법과 달리, VITA는 adaptation module 자체를 **2차 미분을 쓰는 meta-learning으로 사전 학습**해야 한다 — closed-form fit(COAST의 conceptor)보다 훈련 비용·복잡도가 크다.
