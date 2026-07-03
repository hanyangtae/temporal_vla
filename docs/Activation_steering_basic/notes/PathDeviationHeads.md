# Your VLA Already Has Attention Heads For Path Deviation Detection (Jeong et al. 2026, Navigation Heads / Hnav)

- 출처: arXiv:2603.13782v1 [cs.RO] 14 Mar 2026 (Korea University + UCLA + NVIDIA, Jeong/Zhu/Lin/Jaimes/Vu/Joo/Kim/Jawed) · PDF: docs/references/PathDeviationHeads_2603.13782.pdf · 섹션=§7 VLA방향(전체 §3 방법론 + §4 실험 + Supp.A/B/C/D 포함 정독) · tier=must · 한줄역할: NaVILA(VILA-8B 기반 navigation VLA)의 attention head 중 vision-instruction spatiotemporal grounding을 담당하는 소수(Navigation Heads, Hnav)를 찾아 그 entropy로 path deviation을 training-free·거의 zero-overhead로 온라인 검출하고, 검출 즉시 VLA를 우회해 별도 RL 컨트롤러로 물리적 rollback을 실행하는 시스템 — "내부 신호로 온라인 실패 검출"이라는 문제 정의는 우리와 원형적으로 같지만, 검출 신호가 hidden activation이 아니라 attention weight이고 개입이 latent steering이 아니라 액션공간 우회라는 점이 결정적으로 다르다.

## 문제·동기
VLN(vision-language navigation)용 VLA는 LLM 백본의 고질적 hallucination을 물려받아, 긴 시각 히스토리·다단계 언어지시를 따라가다 trajectory drift·잘못된 subgoal 선택·궤적 이탈을 일으킨다. 기존 대응은 세 갈래로 다 부족하다: (1) motion-heuristic(정지/좌표불변 감지, ETPNav 등)은 post-facto라 recall이 낮고, (2) learning-based auxiliary adapter(SMNA progress monitor, IEDL 등)는 별도 네트워크 파인튜닝 비용이 들고, (3) 외부 LLM/VLM을 zero-shot으로 불러 판단(SayNav 등)은 training-free지만 autoregressive API 호출 지연이 실시간성을 해친다. 저자들의 질문: "frozen VLA 내부에 이미 navigation이 제대로 grounding되고 있는지 알려주는 메커니즘이 있는가" — 있다고 답한다.

## 핵심 아이디어
attention head는 학습 중 자연히 기능 특화된다(BERT syntactic head, pruning 연구 등 선행연구 계승). 그중 일부(Hnav)는 명시적으로 "시각 프레임 시퀀스 ↔ 지시문의 해당 세그먼트"를 시간에 따라 대각선으로 정렬시키는 spatiotemporal grounding을 수행한다 — 정상 주행 시 이 정렬이 매끄럽게 진행되고, 이탈 시 구조가 붕괴(attention이 흩어짐/entropy 급증)한다. 이 붕괴를 실시간으로 재는 것만으로 추가 파라미터·재학습 없이 이상탐지가 된다는 것이 핵심 주장.

## 방법(path deviation head로 실패/이탈 검출)
1. **Trajectory phase labeling(§3.2)**: reference waypoint 경로 W와 각 스텝의 target-tracking(non-decreasing index) + progress delta Δd_t로 GT를 만든다. 한 방향 상태기계(patience p)로 Normal(N)→Anomaly(A) 비가역 전이를 정의(연속 p스텝 Δd_t>0 이면 A로 전이, 첫 이탈 시점부터 소급 라벨링; 연속 p스텝 정상화되면 노이즈 보정으로 에피소드 truncate). Val-Unseen에서 "only N" 비율(56.0%)이 baseline SR과 거의 일치 = GT 신뢰성 검증(Table 1).
2. **Navigation head 선정(§3.3, Fig.4)**: 두 기준의 AND. (a) alignment score I_diag(h) — SPL=1.0인 "이상적" episode에서 head별 attention matrix(frame×instruction-token)를 4개 통계로 채점: Suniform(프레임 에너지 균일성), Speak(단일 instruction 세그먼트 집중도), Sdiag(대각선 근접도), Sshift(시간에 따른 매끄러운 후방 이동). (b) cognitive anomaly sensitivity — N/A 두 구간의 Speak 분포 차이를 Cohen's d로 정량화, d(h) 랭킹 top-K를 Hnav로 채택. 최종 top-3 = L21H12, L16H1, L14H1(1000+ 헤드 중 3개).
3. **온라인 검출(§3.4)**: 매 스텝 Hnav의 정규화 Shannon entropy 평균 E_t를 구하고, 롤링윈도우(W) 평균 대비 상대엔트로피 R_t = E_t / (rolling mean + eps)를 계산. R_t가 threshold τ를 P 연속 스텝 초과하면 Anomaly로 판정하고 직전 안전 checkpoint C_safe={pose, 시각 히스토리, entropy buffer, attention state}를 기록. (K,W,τ,P)는 train split에서 FER<10% 제약 하 EDR 최대화로 그리드서치(9000 조합).
4. **개입(§3.5-3.6)**: 검출 즉시 무거운 VLA(0.3Hz)를 완전히 우회하고, 별도로 학습된 경량 PPO 액터-크리틱 RL 정책(10Hz, CNN costmap + MLP goal encoder, 4.89M param)이 LiDAR 기반 2D costmap을 보고 마지막 안전 checkpoint까지 충돌회피 shortest-path rollback을 실행. Fast-LIVO2(CPU 오도메트리) + AprilTag EKF로 글로벌 좌표계를 유지해 rollback 목표점을 특정한다.

## 실험·결과
- **검출**: R2R Val-Seen/Unseen에서 K=3 heads로 episode-level EDR 44.6%/41.9%, FER 11.7%/9.6%, step-level F1 78.3%/76.4% — Stagnation·ETPNav Action-Failure heuristic보다 recall이 압도적으로 우세(heuristic은 step anomaly 60%+ 놓침, ETPNav 변형은 Gap이 음수). RxR(더 복잡한 언어)에도 head/hyperparameter를 재사용해 Val-Unseen EDR 32.7%, FER 9.7%로 일반화 확인.
- **overhead**: NaVILA 단독 대비 검출 로직 추가는 +19.8ms/step, VRAM 증가 0MB(Jetson AGX Orin) — attention weight는 forward pass의 부산물이라 O(1) lookup만 필요.
- **RL rollback 정책**: IsaacLab 60m×60m 장애물 환경에서 SR 87.3%(APF/DWA/MPPI/TEB 대비 최고, 2위 APF 57.4%보다 +30%p), CR 10.6%, TR 2.0%. 실물(AgileX Scout 2.0 + Jetson AGX Orin)에서도 재현.
- 전체 통합 시스템(NaVILA+검출+Fast-LIVO2+RL)의 총 추가 지연은 NaVILA 단독 대비 +0.55s(Table 13) — "거의 공짜" 프레이밍을 뒷받침.

## activation-steering 흐름 위치
이 논문은 steering이 아니다 — hidden state(residual stream)에는 전혀 개입하지 않는다. 신호원은 **attention weight**(Instruction-to-Frame alignment matrix)이지 우리가 쓰는 **activation(hidden state) 공간의 conceptor**가 아니다. 게다가 검출 후 개입은 VLA 내부를 조금도 건드리지 않고 **VLA를 통째로 우회**해 별도 RL 정책이 액션공간에서 물리적 rollback을 수행한다. 즉 파이프라인 상 위치는 "activation을 latent-space에서 steer하는 개입 레이어"가 아니라, 그보다 훨씬 앞단인 **"언제 개입할지 트리거하는 online detector"** 자리이며, 그 트리거가 작동시키는 것도 latent steering이 아니라 **외부 컨트롤러 전환**이다. Sentinel(runtime monitor, 별도 노트)과 같은 층위지만 신호원이 출력 action 분포/픽셀이 아니라 attention weight라는 점에서 조금 더 모델 내부에 가깝다 — 그래도 hidden activation은 아니다.

## 우리 프로젝트 연결(경쟁자 델타: 우리 pathway×phase×TYPE steer와 무엇이 다른가)
- **개입 지점이 근본적으로 다르다**: 이 논문 = "검출 → 외부 RL로 우회(bypass)". 우리 = "검출/식별 → VLA 내부 hidden state를 h' = h·Mᵀ로 직접 steer". 저자들 스스로 Limitations(§5)에서 "물리 궤적만 되돌리고 VLA 내부 context를 동기화하지 않으면 반복적 인지오류(repetitive cognitive error)가 재발할 수 있다"고 인정하며 "active replanning/re-prompting으로 내부 상태를 맞춰야 한다"를 future work로 남긴다 — 이것이 정확히 우리가 이미 하고 있는 지점(내부 latent를 직접 성공 subspace로 이동)이다. 즉 이 논문의 한계가 우리 method의 존재 이유를 뒷받침한다.
- **failure TYPE 미구분**: 이 논문은 "path deviation"이라는 단일 이진 이상(정상/이탈)만 검출한다. hallucination의 원인이 시각-언어 grounding 실패(우리 식으로는 goal/VL)인지 모터 실행 붕괴(motor/DiT)인지 구분하지 않는다 — Hnav 자체가 vision×instruction 결합 신호라 원인을 pathway로 분해하지 않는다. 우리 핵심 니치(goal vs motor TYPE 식별)는 이 논문에 대응물이 없다.
- **phase 개념이 다르다**: 이 논문의 "phase"는 GT reference-path 이탈여부로 정의된 이진 상태기계(N/A)일 뿐, task-internal subphase(예: reach/grasp/lift)에 조건부로 steering 표적을 바꾸는 우리의 phase-matched 개념과 다르다. 그들의 phase labeling은 오프라인 GT 라벨(궤적 거리)로 offline head selection에만 쓰이지, online steering 라우팅에 쓰이지 않는다.
- **도메인 차이**: VLN-CE(연속 waypoint 따라가기, 접촉 없음) vs 우리(RoboCasa 조작, 접촉多 phase). 이 논문의 "이탈"은 본질적으로 위치오차(연속 공간의 좌표편차)로 측정 가능하지만, manipulation 실패는 접촉·grasp 성공/실패 같은 이산적 이벤트라 같은 검출 프레임을 그대로 이식하기 어렵다.
- **인과검증 부재**: 이 논문은 검출 정확도(EDR/FER/F1)만 보고하고, "검출 후 개입이 실제 task 성공률을 올리는가"의 인과효과(ΔSR)를 별도로 측정하지 않는다(rollback 성공 여부는 RL obstacle-avoidance SR로만 간접 검증). 우리는 스티어링의 인과효과를 ΔSR로 직접 재측정하는 것이 표준이다.
- **참고할 만한 부분**: Cohen's d 기반 head/방향 민감도 랭킹, "이상적 episode(SPL=1.0)만으로 오프라인 채점" 방법론, attention entropy의 rolling-window 상대비(R_t) 트릭은 우리 online phase/failure-type 검출기에도 저비용 보조 신호로 이식 가능 — 단, hidden activation 대신 attention weight를 쓴다는 점은 유지한 채 참고해야 한다.

## 면접 포인트(Q→A; 우리 novelty를 이 논문 대비 어떻게 방어하나)
1. Q: "둘 다 내부 신호로 온라인 실패를 잡는데, 결국 같은 아이디어 아닌가?" A: "문제 정의(내부신호×온라인×실패검출)는 원형적으로 같지만 신호원과 개입지점이 다르다. 이 논문은 attention weight의 entropy를 재서 검출하고, 검출되면 VLA를 완전히 우회해 외부 RL 컨트롤러로 액션공간에서 rollback한다 — VLA의 내부 상태는 전혀 손대지 않는다. 우리는 hidden activation(residual stream)에서 성공/실패 subspace를 conceptor로 fit하고, 검출된 조건(pathway×phase)에 맞춰 그 activation 자체를 h' = h·Mᵀ로 밀어 VLA의 다음 action generation을 직접 바꾼다. '검출 후 무엇을 하는가'가 다르다 — 우회 vs 개입."
2. Q: "이 논문이 이미 3개 head만으로 44%+ 검출을 달성했는데, 우리가 pathway/phase까지 나누는 게 과잉설계 아닌가?" A: "이 논문은 실패 TYPE을 구분하지 않는 이진 검출기다 — '이탈했다/아니다'만 안다. Manipulation에서 실패 원인이 목표 오인식(VL)인지 모터 실행 붕괴(DiT)인지에 따라 올바른 개입 방향이 다르므로(pathway마다 steer 대상 latent가 다름), TYPE을 모르면 steering을 어디에 걸지 라우팅할 수 없다. 우리의 pathway 분리는 '검출 정확도'를 올리려는 게 아니라 '개입을 라우팅'하기 위한 필수 전제다."
3. Q: "이 논문의 Limitations이 뭐라고 하나, 그게 왜 우리한테 유리한가?" A: "저자들은 물리 rollback만으로는 VLA의 내부 인지상태(hallucination을 유발한 context)가 그대로 남아 반복 오류가 재발할 수 있다고 명시하고, 이를 풀려면 replanning/re-prompting으로 내부 상태를 동기화해야 한다고 future work로 남긴다. 이건 정확히 activation steering이 겨냥하는 지점 — 우리는 물리적 위치가 아니라 latent 자체를 성공 방향으로 이동시켜 이 재발 문제를 구조적으로 다룬다."
4. Q: "이 논문의 head-selection 방법(Idiag + Cohen's d)을 우리 conceptor fit에 그대로 쓸 수 있나?" A: "선택 대상이 다르다 — 그들은 '어떤 attention head를 볼지'를 고르는 head-importance/pruning 계열 기법(NOTALL의 head-ranking과 유사)이고, 검출 전용이라 steering 표적이 아니다. 우리 conceptor는 '어떤 hidden activation 방향으로 밀지'를 성공/실패 분포의 공분산 구조에서 직접 fit한다. 다만 Cohen's d로 N/A 분포 분리도를 정량화하는 아이디어 자체(어떤 신호가 실패에 민감한지 오프라인 랭킹)는 우리 online detector 후보 feature 선별에 재사용 가능하다."
5. Q: "도메인이 navigation인데 manipulation에 이식 가능한가?" A: "이탈(waypoint 거리)이라는 연속적 GT가 있는 navigation과 달리 manipulation은 접촉·grasp 등 이산 이벤트라 phase labeling 방식(§3.2의 state machine)을 그대로 쓰기 어렵다. 또한 그들의 검출 신호가 vision-instruction 결합 attention이라 pathway로 분해되지 않는데, GR00T 같은 VL-SA/DiT 분리 아키텍처에서는 애초에 신호원 자체를 pathway별로 나눠 뽑을 수 있어 우리 쪽이 구조적으로 TYPE 식별에 더 유리하다."

## 한계·비판
- EDR이 40%대에 그친다(FER<10% 제약 하) — 즉 이탈 episode의 절반 이상을 episode-level에서는 놓친다(step-level recall은 61~69%로 더 나음). "44%면 낮은 게 아닌가"라는 반론에 저자는 zero-overhead·zero-training 트레이드오프로 방어하지만, 안전이 critical한 배포에는 recall 부족이 그대로 리스크.
- Head selection이 SPL=1.0인 "이상적" episode(오프라인 GT 필요)로 이루어짐 — 완전한 self-supervised가 아니라 라벨링된 성공 궤적 데이터가 전제조건. "training-free"라는 프레이밍이 정확히는 "gradient-free"에 가깝고, 오프라인 통계 산출(grid search 9000 조합)에는 상당한 offline compute가 든다.
- 단일 백본(NaVILA/VILA-8B)에서만 검증 — Hnav의 layer/head index([21,12],[16,1],[14,1])가 다른 VLA 아키텍처·다른 체크포인트에도 재현되는지는 미검증(모델별 재탐색 필요할 가능성 높음).
- 저자 스스로 인정한 한계: 물리 rollback만으로는 VLA 내부 인지오류가 동기화되지 않아 같은 실패가 재발할 수 있음(위 "우리 연결" 참조) — 이 논문의 시스템이 완결적 해법이 아니라는 뜻.
- "training-free 실시간" 주장과 별개로 실제 지연은 여전히 NaVILA forward pass(≈3.5s, Table 13) 자체에 지배되어 0.3Hz로만 동작 — 검출 자체는 빠르지만 상위 VLA 추론 주기가 느려 "실시간"의 체감 해상도는 낮다(그 사이는 RL 정책이 대신 채움).
- FER 정의(episode 단위 false positive)와 step-level precision(90%+)이 함께 제시되지만, 실사용에서 중요한 것은 "불필요한 rollback 트리거로 인한 임무 지연"인데 이에 대한 정량 비용(우회 소요 시간, 임무 실패로 이어지는 빈도)은 보고되지 않음.
