# B. 정책 내부 표현에서 phase/subtask 읽기 — 문헌 조사

조사일 2026-08-10. 조사 질문: VLA·diffusion policy·모방학습 정책의 **internal
activation(hidden state)** 에서 조작 단계(phase/subtask/skill segment)를 읽어낼 수
있는가, 그 정보를 downstream(실패검출/개입시점/라우팅)에 쓴 사례가 있는가, 내부표현
vs 관측(비디오) 기반 phase 판독의 장단점을 논한 연구가 있는가.

**규율**: 표 안 "확인" 열은 WebFetch로 실물(초록 이상)을 직접 확인한 것만 표시.
초록만 확인되고 method/실험 세부를 못 얻은 건 "부분확인"으로 별도 표시. 검색 스니펫에만
등장하고 WebFetch 재확인이 안 된 수치는 본문에 출처를 "WebSearch 요약, 미재확인"으로
명기.

관련 기존 노트: [SAE_synthesis_and_design.md](../SAE_synthesis_and_design.md)(Dr.VLA/
Event-Grounded SAE/Observing&Controlling 통합), [rl2_vla_adaptive_steering.md](../rl2_vla_adaptive_steering.md)
(SAFE-LSTM 기반 online 실패검출→개입 게이팅), [steering_robustness_wam_lqr.md](../steering_robustness_wam_lqr.md).
이 프로젝트에서 이미 깊이 다룬 **SAFE**(Multitask Failure Detection, NeurIPS 2025)는
success/failure 이진 예측이지 phase 분류가 아니라서 이번 표에는 재포함하지 않음(배경으로만 언급).

---

## 1. 요약 표 (전체)

| # | 논문 | arXiv/venue | phase 출처 | 정확도/평가 | downstream 사용 | 우리 세팅 근접도 | 확인 |
|---|---|---|---|---|---|---|---|
| 1 | PAMAE: Phase-Aware-MoE Action Experts | 2606.27144, 2026-06 | 내부(backbone context hᵗ) + 저차원 실행기술자(gripper/속도/progress) 결합, 라벨=휴리스틱(gripper·속도 임계값) | PCP(Phase-Conditioned Dominance Purity) 89.0%만 보고, 분류 accuracy 없음 | 전문가 라우팅 → π0 73.8%→83.0%, π0.5 85.8%→91.4% SR | **매우 높음** — flow-matching VLA action expert가 바로 이 논문 대상 | 확인 |
| 2 | What Frozen VLAs Already Know About Success | 2605.28527, 2026-05 | 내부(OpenVLA LLM층/π0.5 VLM·action-expert층) frozen activation linear probe | offline R² 0.55(π0.5), matched-pair 정렬정확도 92~94% (label-shuffle 대조군 50%) | **있음** — test-time action 후보 선택(value-guided)에 실사용, push-plate 26.7%→44.3% | 높음 — flow-matching VLA action-expert층까지 probing | 확인 |
| 3 | Move-Then-Operate | 2604.23620, ICML 2026 | 학습: video+MLLM 라벨(외부). **추론**: VLM backbone 내부 semantic feature 위 MLP 라우터(z=argmax pφ(z\|f_t)) | 분류 accuracy 미보고(라우팅→태스크 SR로만 검증) | dual-expert 라우팅 → π0 대비 +24.1%p SR(RoboTwin2) | 높음 — VLA에 phase 라우터를 얹은 구조, 다만 π0/모놀리식 VLA용 | 확인 |
| 4 | Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models (**=Dr.VLA**, 기존 리뷰됨) | 2603.19183, 2026-03 | 내부(π0.5 action-expert/PaliGemma층, OpenVLA Llama2층) SAE dictionary | motion-primitive-like feature 발견(F1129 grasp/place, F1902 transport, F128 pre-grasp, F158 DROID sub-phase 전이) — generality 분류기 100% LOO-CV(30개 라벨 feature), **phase 분류 정확도 자체는 없음** | steering 인과는 검증(정성) — 저자 스스로 "not applied toward model improvement"라 명시, 즉 **분석/검증만** | 매우 높음 — action-expert가 DiT-계열 구조와 직접 대응 | 확인 (프로젝트 기존 리뷰, 이번엔 phase 각도 재추출) |
| 5 | Hide-and-Seek in Trajectories | 2605.30834, 2026-05 | 내부(VLA action token/action head의 per-timestep embedding) | bACC 85.2%(seen)/83.4%(unseen), wACC 85.3%/82.8%, TWA 66.0%/66.3% — **phase 자체가 아니라 궤적 내 실패시점 시간적 국소화** | 실시간 runtime monitoring, 실기(xArm6, π0.5)에서 검증, 0.001s/step | 중상 — 시간축 국소 신호 추출 방법론은 근접, 목적은 phase 아님 | 확인 |
| 6 | ProbeAct | 2606.09740, 2026-06 | 내부(VLA **layer 8** 중간 activation, 공간토큰 4×4 pooling→PCA) | 3D좌표 R²=0.968(layer 8) — **phase 아닌 object position만 probing** | 실패복구: kinematic state machine + CBF 필터로 온라인 개입 | 중 — internal probing→온라인 개입이라는 파이프라인 패턴은 근접, 대상이 phase가 아님 | 확인 |
| 7 | SAFECAST | 2608.04246, 2026-08 | 내부(VLA **pre-final-layer** hidden state) risk probe | 실기(DROID) π0 ROC-AUC 0.38(SAFE 0.26 대비), 시뮬(LIBERO) 0.45(SAFE 0.33 대비) | 실시간 실패검출(성공/실패 이진, **phase 미분류**) — SAFE(NeurIPS 2025) 직접 후속 | 중 — 우리와 같은 "internal activation probe" 계열이나 대상이 phase 아닌 이진 성공/실패 | 확인 |
| 8 | ActProbe | 2606.08508, 2026-06 | **action space만**(TCE=연속 청크 간 겹침 MSE, ACM=청크 L2 norm) — 정책 내부 접근 명시적으로 회피 | F1-timeliness hypervolume +12.7%, early-detection ROC-AUC +9.0%(미본 태스크), 3ms/call | 실시간 실패검출 게이팅, RL fine-tuning 시 상호작용 2.9배 절감 | 낮음(방법론) / **높음(Q3 논거)** — "internal 접근 불필요"를 명시적 설계원칙으로 내세운 대조 사례 | 확인 |
| 9 | SARM2 | 2606.10305, 2026-06 | **외부관측**(3-view 카메라 프레임+proprio) 기반 causal Transformer stage estimator — VLA 내부 activation 아님 | Demo MSE 0.006(S1)/0.031(S2), rollout 분류점수 ρ 0.833(T1)/0.667(T2). *(WebSearch 요약에 "micro-accuracy 85.22%, task별 70.85~98.27%" 언급 있었으나 직접 재확인 안 됨 — 신뢰도 낮음, 인용 보류)* | stage→MMoE gate 선택→dense reward. reward model 학습용 | 낮음(관측기반) / downstream 패턴(phase→라우팅 게이트)은 참고할 만 | 확인 |
| 10 | PACE | 2606.00537, 2026-06(v2 07-29) | **예측된 action chunk 자체**(kinematic valley, 속도 프로파일)에서 후처리 검출 — policy-agnostic, 내부 접근 없음 | 분류기 없음(결정론적 valley 검출); RoboTwin2 SR 57.8%→64.2%, 실기 50.7%→70.4% | subtask 경계=re-planning(청크 재생성) 시점 결정 | 낮음(방법론) / Q3 대조 사례로 유용 | 확인 |
| 11 | LOTUS | 2311.02058, ICRA 2024 | **외부** open-vocab 비전 인코더(DINOv2) feature 위 계층적 클러스터링 — policy 내부 아님 | SR +11%(FWT), NBT +2%(WebSearch 요약, 직접 재확인은 abstract까지) | continual hierarchical skill library 구성(skill policy 자체를 만드는 데 사용) | 낮음(내부표현 아님) — "관측기반 phase" 대조사례로 유용 | 확인(초록), 방법론 세부는 WebSearch 보강 |
| 12 | PALM | 2601.07060, CVPR 2026 | affordance-latent 기반 progress head(비전 조건부, VLA 내부 activation probe는 아님) | LIBERO-LONG SR 91.8%, CALVIN ABC→D 평균길이 +12.5% | continuous within-subtask progress → subtask 전환 부드럽게(정책의 action 생성에 조건부) | 중 — progress를 policy 구조에 내장해 조건부로 쓰는 패턴은 참고할 만 | 확인(초록수준) |
| 13 | LAR-MoE | 2603.08476, 2026-03 | 제목상 "latent-aligned routing"(정책 latent 정렬 시사) — 방법론 세부 미확보 | 미확보(PDF 텍스트층 파싱 실패) | phase 주석 없이 감독 MoE baseline과 동등(초록 주장) | 불명 — 확인 필요 | **부분확인**(초록만) |
| 14 | SAGE | 2509.19853, 2025-09 | HMDP 아키텍처에 내장된 latent state("state transition network"가 hidden state 추론) — 기존 정책을 probing하는 게 아니라 **자체 구조에 내장** | 미확보(수동라벨 13%만 필요, "100% task success"는 출처·태스크 난이도 불명) | state-aware action policy가 관측+hidden state에 조건부 | 불명 — 확인 필요, 아키텍처가 우리와 다름(HMM 명시적 모델링) | **부분확인**(초록만) |
| 15 | HiMaCon | 2510.11321, 2025-10 | 자기지도 cross-modal correlation network(자체 인코더 학습, 기존 정책 activation probing 아님) | 미확보(정량 없음, "significantly improves"만) | concept-augmented policy 성능 개선(수치 미확보) | 불명 | **부분확인**(초록만) |
| 16 | Offline Discovery of Interpretable Skills | 2602.01018, 2026-02 | **raw trajectory**(state-action) 기반 2단계 약지도 세그멘테이션 — 기존 policy의 activation을 읽는 게 아니라 skill을 처음부터 구성 | termination 경계 검출 93.7% accuracy(±4 timestep) | hierarchical policy 구성(skill switching) — 정책 실행에 직접 사용 | 낮음(내부표현 아님) | 확인(html) |
| 17 | Beyond Task Success: WAM/VLA 진단 | 2606.01095, 2026-06 | SAE 기반, activation을 memorized/reactive/predictive로 분류 — phase 특정은 불명확 | 미확보(PDF 파싱 실패) | 진단 프레임워크(분석용으로 추정) | 불명 | **부분확인**(초록 요약 수준) |
| 18 | VLA-Trace | 2605.30117, 2026-05 | CKA 기반 representation dynamics 추적(cross-modal, checkpoint-drift) | 미확보 | **분석만**으로 보임("future directions"만 제시) | 불명 | **부분확인**(초록만) |
| 19 | DAISS | 2603.07663, 2026-03 | phase-aware architecture(양팔 초음파 삽입) — 세부 미확보 | 미확보 | 미확보 | 낮음(의료 도메인, 세부 불명) | **부분확인**(초록 일부만) |
| 20 | Hwang & Tani (고전) | 1706.02423, IEEE TCDS 2017 | RNN 계층 internal representation, t-SNE로 정성 시각화(WebSearch 스니펫: "각 계층에서 task phase별로 표현이 창발") | 정량 없음(정성적 시각화만, 검증 안 됨) | 없음 — 순수 분석/시각화 | 낮음(휴머노이드 제스처 모방, VLA 아님) — **역사적 선례**로만 의미 | 확인(초록만, t-SNE 세부는 미재확인) |

---

## 2. 핵심 논문 상세 (Tier S)

### 2.1 PAMAE — Phase-Aware-MoE Action Experts (2606.27144)

가장 우리 세팅에 근접: **flow-matching VLA(π0/π0.5)의 action expert**를 sparse MoE로
교체하고, phase-aware router가 전문가 간 행동 생성을 배분한다.

- **라우터 입력**: `r_t^τ = [h_t, φ_t, τ]` — `h_t`=사전학습 VLA backbone의 멀티모달
  context(내부 activation), `φ_t`=저차원 "실행 기술자"(gripper 상태·gripper 변화량·
  action magnitude·정규화 progress cue), `τ`=flow time.
- **phase 라벨**: 사람이 주석단 게 아니라 **gripper 닫힘 검출 + 속도 임계값** 휴리스틱으로
  pre-contact/contact/post-contact 3단계를 자동 생성 (pseudo-label).
  → 즉 "내부 activation을 직접 probing해 phase accuracy를 측정"하는 논문이 아니라,
  **저차원 운동학 신호 + backbone context를 결합한 router를 phase 정합 손실(KL)로
  학습**하는 구조. 순수 activation-probe 정확도 수치는 없음(PCP 89.0%만 보고).
- **추론 시 phase 신호 없음**: "PAMAE learns phase-consistent expert allocation
  without explicit phase labels at inference time" — 명시적 phase 신호 없이 router가
  학습된 매핑으로 암묵적으로 phase에 맞는 전문가를 고른다. 이 지점이 우리 질문("online
  phase 식별 가능한가")과 정확히 맞닿음: 이 논문은 "된다"는 쪽에 실증적 증거(SR 개선)를
  제공하지만, **phase 판별 자체의 정확도는 검증하지 않고 다운스트림 SR 개선으로만
  간접 검증**한다.
- **다운스트림**: 5개 조작 태스크(Table-Cleaning/Drawer-Cycle/Lid-Open/Shelf-Insert/
  Cup-Upright)에서 π0 73.8%→83.0%, π0.5 85.8%→91.4% SR. ablation에서 routing alignment
  loss 제거가 가장 큰 하락(76.2%)폭.
- **한계**: DiT/diffusion transformer 여부, phase 주입 레이어 등 아키텍처 세부는
  abstract/html 발췌로는 확인 안 됨(본문 표·그림 필요).

### 2.2 What Frozen VLAs Already Know About Success (2605.28527)

Q1·Q2 모두에 가장 강하게 답하는 논문. **phase 자체가 아니라 연속적 "progress/value"
신호**를 다루지만, 정량적 엄밀성이 이 리스트에서 제일 높다.

- **probing 대상**: OpenVLA LLM layer 7/20, π0.5 vision encoder/VLM 최종출력/VLM
  layer 8, 대조군으로 DINOv2/CLIP도 probing. **frozen** 표현 위에 linear probe.
  target = Monte-Carlo discounted outcome (`v_t = 0.99^(T-1-t)`, 성공/실패 이진에서
  파생) — "progress"는 이 discount 구조를 통해 시간축에 자연히 녹아든다(하지만
  discrete phase label을 직접 예측하진 않음).
- **정확도**: offline R² 0.55(π0.5 vision encoder, LIBERO-Goal) vs random projection
  0.39. 핵심은 **matched-pair control**(같은 task·같은 timestep 내에서만 비교) —
  이 통제 하에서도 92~94% pairwise ordering accuracy 유지(label-shuffle 대조군은
  50.05%=chance). 즉 **task/시간 shortcut이 아니라 genuine value-like 신호**라는 주장을
  가장 엄격하게 검증한 논문.
- **downstream 실사용**: test-time action candidate ranking(16개 후보 시뮬레이터
  평가) — value-guided가 greedy 대비 push-plate 26.7%→44.3%, wine-rack 35.7%→44.0%,
  drawer는 이득 없음(38.7%→39.33%). **분석에 그치지 않고 실제 행동을 바꾸는 실험**이
  있다는 점에서 이 표에서 downstream 증거가 가장 강함.
- **한계**: phase/subtask *분류*가 아니라 스칼라 progress/value 회귀이고, 여기서 π0.5의
  action-expert(=DiT-계열)는 probing 대상에 포함되지만 강조되진 않음(주로 VLM/vision
  encoder 쪽 수치가 보고됨).

### 2.3 Move-Then-Operate (2604.23620, ICML 2026)

- **phase 정의/라벨**: MLLM이 비디오+언어 instruction을 보고 move(coarse relocation)
  vs operate(contact-critical) 2단계로 자동 라벨링(30fps 샘플링, 반복 정제). **학습
  시점엔 외부 관측 기반**.
  단 **추론 시점**엔 별도 관측 파이프라인 없이 "VLM backbone의 global semantic
  feature 위에 얹은 경량 MLP 라우터"가 `z_t = argmax p_φ(z|f_t)`로 능동/수동 전문가를
  고른다 — **이게 내부 activation에서 읽는 부분**. 학습 라벨은 외부, 추론 판독은 내부라는
  점에서 PAMAE와 유사한 하이브리드 패턴.
- **다운스트림**: dual-expert(이동/조작) 라우팅 → RoboTwin2 8태스크 평균 SR 68.9%,
  π0 baseline 대비 +24.1%p, 10배 적은 데이터로 동등 성능, 학습 스텝 40% 절감.
- 우리 프레임과의 차이: phase가 2단계로 매우 coarse(우리는 phase-matched steering을
  위해 더 세밀한 rollout-phase 조건화를 목표). router 자체의 분류 정확도(f1/accuracy)는
  보고되지 않고 다운스트림 SR로만 간접 검증.

### 2.4 Dr.VLA(=Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA
Models, 2603.19183) — phase 각도 재조명

이 논문은 프로젝트에서 이미 `dr_vla_sae.md`로 리뷰됨(generality-vs-memorization
metric 중심). 이번 조사에서 **phase/motion-primitive 각도**로 재추출한 내용:

- SAE dictionary에서 **동작 국면에 대응하는 feature**가 다수 발견됨: F1129(grasp/place,
  잡기 순간에 발화하며 sub-goal 수에 따라 onset count 스케일), F1902(transport, 잡은
  뒤부터 활성화하며 목표에 가까워질수록 활성 크기가 선형적으로 증가 — **진행도(progress)와
  거의 동일한 신호**), F128(pre-grasp alignment, end-effector가 목표물 위에 있을 때
  발화), F158(DROID 데이터셋에서 조작 sub-phase 전환 시점에 발화).
- **레이어**: π0.5는 PaliGemma층(0/5/11/17)과 **action expert층(0/5/11/17, d=1024)**
  둘 다에서 SAE 학습 — action expert는 GR00T의 DiT와 구조적으로 가장 가까운 대응물.
  OpenVLA는 Llama2 층 0/8/16/24/31.
- **검증**은 "phase 분류 정확도"가 아니라 **general vs memorized 판별**(4개 활성화
  통계: episode coverage/mean onset count/activation magnitude/relative run length →
  logistic regression 100% LOO-CV, 30개 사람이 라벨링한 feature 대상). phase다움 자체를
  정량 채점하진 않음(정성적 사례 제시).
- **downstream**: closed-loop steering으로 인과관계는 확인(F128 steer→그리퍼가 목표물
  위에서 멈춤, F1902 steer→목표로 직행하는 정성적 변화) 하지만 저자 스스로 **"이 결과들이
  모델 개선에 적용되지는 않았다"**고 명시 — 순수 분석/검증 단계이지 실사용 downstream(실패
  검출/라우팅 등 실제 파이프라인 배치)은 없음.

### 2.5 Hide-and-Seek in Trajectories (2605.30834)

- "VLA의 action token 또는 action head에서 timestep별 action embedding을 추출"
  — 이것도 내부 activation. inter-/intra-trajectory contrastive objective로 **궤적
  내에서 실패가 시작되는 시점**을 국소화(uniform trajectory-level label만 갖고도 시간
  구조를 유도). 목적함수 자체가 "정상 실행 구간 vs 실패 구간"의 시간적 분리를 만드는
  것이라, **이진(phase 아님)이지만 "온라인으로 궤적 내 시점을 internal activation에서
  구분해낸다"**는 방법론이 phase-matched steering의 "어느 시점"부분과 방법론적으로
  가장 가깝다.
- bACC 85.2%(seen)/83.4%(unseen), 실기(xArm6, π0.5)로 CUBE/KITCHEN 두 세트에서 검증,
  추론 0.001s/step (VLM baseline 2.343s/step 대비 ~2000배 빠름 — 이 속도차가 §4의
  Q3 논거로 재사용됨).

---

## 3. Q3 — "내부표현 vs 관측(비디오) 기반 phase 판독"의 장단점을 정면으로 다룬 연구는?

**정면으로 이 대조를 연구질문으로 삼은 논문은 못 찾음(gap).** 다만 여러 논문이 설계
선택의 근거로 이 축을 암묵적으로 다룸 — 아래는 실제로 WebFetch 확인된 내용만 정리:

- **ActProbe(2606.08508)**가 가장 명시적으로 반대 입장을 취함: "기존 온라인 실패
  검출기는 policy internals에 대한 white-box 접근을 요구하거나 resampling·관측
  기반 신호로 런타임 오버헤드를 추가한다"고 internal-probing 계열의 **한계(모델별
  재구현 필요, 접근성)**를 지적하고, 대신 순수 action-space 신호(TCE/ACM)로 대체.
  즉 "내부표현이 필요없다"는 것 자체가 이 논문의 셀링포인트 — internal 접근이
  **이식성(portability)** 비용을 문다는 방증.
- **Hide-and-Seek(2605.30834)**는 반대로 internal 신호의 **장점**을 속도로 정량화:
  action embedding 추출은 forward pass의 부산물이라 0.001s/step, 외부 VLM 기반
  baseline은 2.343s/step — **~2000배** 차이. (참고: 프로젝트에 이미 리뷰된
  RL2-VLA도 동일 논지 — SAFE latent 추출 1~2ms vs 전체 VLA forward 232ms+로
  "forward pass 부산물이라 거의 공짜"라는 동일 패턴을 보고함. 두 논문이 독립적으로
  수렴하는 논거.)
- **PAMAE·Move-Then-Operate**는 둘 다 "추론 시점에는 명시적 phase 신호(즉 외부
  perception 파이프라인)를 요구하지 않는 것"을 설계 목표로 삼음 — 이는 명시적 비교
  실험이라기보다 **내부표현 기반 판독을 선호하는 아키텍처 선택**으로 나타남(배포 시
  별도 perception 모듈이 필요없다는 게 암묵적 장점으로 취급됨).
- **SARM2**는 반대로 **외부 관측(카메라+proprio) 기반** stage estimator를 택하면서
  "task별 주석이 필요해 정확하지만 태스크 특이적" vs "coarse하지만 범용적인 VLM
  reward model" 사이의 트레이드오프를 논함 — internal-vs-external 축은 아니지만
  인접한 "specificity vs generality" 트레이드오프.

**종합**: 명시적 정면 비교(같은 태스크에서 "internal probe" 조건과 "video 기반" 조건을
둘 다 구현해 정확도·지연시간을 나란히 측정)는 확인된 문헌에 없음. 간접 증거들은
일관되게 internal 쪽의 장점을 "거의 공짜인 latency"(forward-pass 부산물)로,
단점을 "모델별 white-box 접근·재구현 필요(이식성 저하)"로 수렴시킴.

---

## 4. 대조 사례 — phase를 정책 내부가 아니라 다른 데서 읽는 연구 (Q1 경계 확정용)

우리 질문은 "정책 **내부** 표현"에서 읽는 것이므로, 아래는 "phase를 읽되 소스가
내부가 아닌" 명확한 반례들 — 경계를 분명히 하는 데 유용:

- **PACE**: 정책이 생성한 **action chunk(출력)**의 운동학(속도 valley)에서 subtask
  경계를 검출. Policy-agnostic·post-hoc이 장점(어떤 청크 생성 정책에도 적용).
- **SARM2**: 카메라 3-view + proprio 관측 기반 별도 causal Transformer. VLA
  activation과 무관하게 독립 모듈로 학습.
- **LOTUS**: DINOv2(범용 사전학습 비전 인코더) feature로 계층적 클러스터링 —
  "정책의" 내부표현이 아니라 **일반 비전 인코더**의 표현.
  (ICRA 2024, UT Austin RPL, Wan·Zhu·Shah·Zhu.)
- **Move-Then-Operate**: phase *라벨*은 비디오+MLLM(외부)에서, 하지만 *추론 시 판독*은
  VLM backbone 내부 feature 위 라우터 — 부분적으로만 "외부" (§2.3 참조).
- **Offline Discovery of Interpretable Skills**: raw state-action trajectory에서
  skill을 처음부터 구성 — 기존에 학습된 정책의 activation을 읽는 게 아니라 skill
  구조 자체를 새로 만듦.

---

## 5. 우리 세팅(GR00T류 flow-matching VLA DiT residual stream)과의 근접도 총평

1. **가장 가까운 것 = PAMAE**: flow-matching action expert에 phase-aware routing을
   얹은 유일한 논문. 다만 (a) phase가 3단계로 매우 coarse(pre/contact/post-contact,
   우리가 원하는 rollout-phase 조건화보다 거칠음), (b) phase 판독 자체의 정량 정확도가
   없고 다운스트림 SR로만 간접 검증, (c) 라벨이 gripper/속도 휴리스틱이라 우리
   프로젝트가 이미 겪은 "길이/phase confound" 이슈([[seen18-rollout-length-confound]])를
   어떻게 통제했는지 불명.
2. **가장 엄밀한 정량 증거 = What Frozen VLAs Already Know**: matched-pair control로
   task/시간 shortcut을 배제한 게 이 리스트에서 유일하게 방법론적으로 우리 confound
   audit 기준([[confound-audit]] skill)에 근접. 다만 대상이 discrete phase가 아니라
   continuous value/progress.
3. **DiT/action-expert 레이어의 SAE feature가 phase에 대응(Dr.VLA)** — 이미 알고 있는
   결과지만 이번에 "phase" 프레임으로 다시 보면, 우리가 하려는 "온라인 phase 식별"의
   **표현적 전제**(그런 정보가 activation에 인코딩되어 있다는 것) 자체는 이 논문의
   motion-primitive feature들이 방증. 단, 그 feature를 "온라인에 phase를 판독하는
   분류기"로 형식화한 실험은 없음 — 우리가 채워야 할 자리.
4. **온라인 성능/지연시간 논거(Hide-and-Seek, ActProbe, RL2-VLA)**: "activation은
   forward pass 부산물이라 거의 공짜"라는 반복되는 논거는 우리의 온라인 phase 식별
   기획(추가 perception 없이 DiT residual만으로 판독)의 타당성을 간접 지지.
5. **정확히 우리가 찾는 논문("대형 사전학습 VLA의 residual stream에 linear/MLP
   probe를 얹어 discrete phase label 분류 정확도를 직접 보고하고, 그걸 실패검출·개입
   시점·라우팅에 실사용하는 논문")은 확인된 범위에서 없음** — 가장 가까운 조각들
   (PAMAE의 라우팅, What-Frozen-VLAs의 progress probing, Dr.VLA의 phase-aligned SAE
   feature, Hide-and-Seek의 시간국소화)이 각각 한 부분씩만 충족.
