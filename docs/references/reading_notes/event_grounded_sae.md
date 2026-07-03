# Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies — 정독 노트

- 원문: `docs/references/Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies.pdf` (23p, arXiv:2605.17204v1, 2026-05-17 preprint)
- 저자: Xinchen Jin, Aditya Chatterjee, Pranav Kumar, Rohan Paleja (Purdue CS). Code: https://github.com/xc-j/Event-SAE
- 대상 모델/벤치: OpenVLA (discrete action token) + π0.5 (continuous action chunk = PaliGemma backbone "PG" + action expert "AE") × LIBERO 4 suites, + real-robot LoRA-finetuned π0.5 (Mobile ALOHA chip-approach).

---

## 0. 한 줄 요약 (우리 팀 관점 먼저)

이 논문의 "event-grounded"는 **SAE 학습 loss나 dictionary에 event를 넣은 것이 아니다.** SAE는 완전 표준 unsupervised BatchTopK SAE(reconstruction + AuxK)이고, "event-grounding"은 **학습된 feature 중 어느 것을 causal test할지 고르는 feature-ranking(선택) 단계**에만 들어간다. 즉 event = "closed-loop rollout에서 추출한 kinematic keyframe cluster"이고, feature를 이 event window에서의 활성 **시간형태(temporal shape)** 로 점수매겨 랭킹한다. 따라서 우리가 기대하던 "event-conditioned dictionary / auxiliary event loss / outcome disentanglement" 는 이 논문에 **없다.** 우리에게 유용한 것은 (a) event를 feature를 **stratify/조건화**하는 축으로 쓴다는 발상, (b) residual-preserving latent edit이라는 안전한 SAE-space steering primitive, (c) 아키텍처별 SAE causal 특성(특히 π0.5 AE ≈ 우리 GR00T DiT)에 대한 경고. **outcome(success/failure)을 scene/length에서 분리하는 문제는 이 논문이 풀지 않으며, 오히려 그들의 Table 10에서 우리와 똑같은 confound(task-id로도 success가 부분예측)를 드러낸다.**

---

## 1. 문제·주장

**해결하려는 문제.** LLM/VLM의 mechanistic interpretability 도구(logit lens, SAE feature를 텍스트 context로 이름붙이기)가 VLA로 깨끗이 전이되지 않는다. 이유 두 가지:
- (i) **feature naming이 어렵다**: VLA 출력은 action space라 internal direction을 vocabulary로 projection해도 신뢰할 만한 의미가 안 나온다 (logit lens 실패).
- (ii) **behavioral validation이 어렵다**: language model은 출력 token이 곧 semantic readout이지만, VLA는 여러 downstream step이 다 맞아야 semantic outcome이 나온다. 1-step intervention은 읽기 어렵고, multi-step intervention은 매 step이 정확히 맞아야 해서 신뢰 불가. intervention 검증이 비싼 closed-loop rollout으로만 가능.

**주장.** 그래서 feature 분석을 language vocabulary projection이 아니라 **rollout에서 재발하는 behavioral event에 grounding**하자. 특히 concurrent work Swann et al.[14](= "Steerable VLAs" 논문)은 SAE feature를 activation 통계로 surface한 뒤 rollout 비디오와 수동 정렬해 label — behavior가 해석 단계에서만 들어오고, 수동 라벨링은 scale 안 되며, 시각적 co-occurrence로는 causal role을 못 세운다. 이 논문은 대신 **살아있는(alive) 모든 SAE feature를 SAE-independent behavioral event에 대해 점수화** → SAE basis 전체 coverage. Extraction·scoring 자동 → per-feature 수동 라벨링을 넘어 scale.

**기여/발견 (저자 요약).**
1. 자동 event-grounded SAE pipeline: rollout에서 kinematic keyframe 추출 → task-local event로 cluster → 모든 alive feature를 event에 점수화(수동 라벨 없이) → closed-loop intervention으로 검증. Event anchor는 SAE 활성과 독립.
2. 아키텍처·site 의존적 intervention: 같은 pipeline인데 OpenVLA는 event-aligned feature가 SR을 측정 가능하게 떨어뜨리고, π0.5 PG backbone은 single-feature edit에 거의 무반응, AE는 거의 아무 ranking에나 붕괴. Intervention site는 아키텍처마다 다르고 직접 전이 안 됨.
3. binary closed-loop 평가의 한계: single-feature intervention은 causal variable을 깔끔히 분리 못 하고 binary SR은 coarse effect만 잡음. 완전한 mechanistic account를 주장하지 않음.

---

## 2. "event-grounded"의 정확한 메커니즘 (핵심)

**결정적 사실: event는 SAE 학습에 들어가지 않는다.** Figure 1 / Sec 3.1이 명확히 보여주듯 SAE loss는

  L_SAE = (1/B) Σ_{i=1}^B ‖h^(i) − ĥ^(i)‖₂² + λ_aux · L_aux

로 **reconstruction + AuxK(dead feature 소생) 뿐**이다. event-conditioned dictionary도, event supervision term도, event label로 gating하는 구조도 **없다**. "event-grounding"은 전적으로 **feature ranking(어떤 feature를 causal test할지 선택)** 단계에 있다.

### 2.1 "event"의 정의 (4단계 파이프라인 중 2·3단계)

**Stage 2 — Kinematic keyframe 추출 (Sec 3.2).** AWE(Automatic Waypoint Extraction[22])를 end-effector position trajectory에 적용. AWE는 error budget η(=0.05) 하에서, 연속 waypoint 사이 linear interpolation이 모든 timestep에서 원 trajectory의 η 이내에 머무는 **가장 작은 waypoint 집합**을 고른다. 이 waypoint들이 kinematic keyframe = event anchor 후보. **AWE는 end-effector motion에만 의존 → SAE 활성/semantic label/feature score와 완전 독립** (representation-independent proposal). 각 keyframe ti에 대해 local observation bundle 구성: ±(4,2,0,2,4) 5-frame image strip, 짧은 video clip, metadata(task desc, episode idx, success label, waypoint rank, timestep).

**Stage 3 — Event clustering (Sec 3.3).** keyframe을 task-local event cluster로 묶음. end-effector position만으로는 부족(가까운 위치에서 의미 다른 event 발생: grasp vs release). 그래서 cluster descriptor를 다음 concat으로 구성:
- v: normalized visual embedding (image strip)
- s: normalized robot state (end-effector pose + gripper status)
- p: normalized temporal-progress scalar
각 성분 독립적으로 ℓ2-normalize 후 weight (λv=1.0 > λs=0.5 > λp=0.4) 적용, concat 후 다시 ℓ2-normalize. Agglomerative clustering, cosine threshold 0.18, **episode의 ≥50%에서 재발하는 cluster만 유지** → 이 retained cluster가 "event unit". (weight 서열의 직관: visual scene이 가장 discriminative, eef state가 tie-break, temporal progress는 약한 ordering prior.)

**(선택) VLM annotation.** OpenVLA+LIBERO만, 각 cluster를 VLM(gemini-3.1-pro-preview)에 exemplar bundle 주고 phrase + 6-way phase label {pre_grasp, immobilization, contact, detach, post_grasp, transition} 붙임. phase taxonomy는 Chen et al.[34](Robo2VLM) 각색. **중요: 이 VLM label은 ranking/intervention에 전혀 안 쓰이고 heatmap·bundle 시각화 aid로만 사용** (저자 스스로 limitation (ii)로 인정 — "cluster-level semantic structure largely unexploited").

### 2.2 feature를 event에 grounding하는 수식 (Stage 4 직전, Sec 3.4 / Appendix D)

Notation: z_{ρ,t} ∈ R^m = rollout ρ, env timestep t의 SAE 활성; z_{ρ,t}[f] = feature f 성분. window radius w=5 (offset δ∈{−w,…,w}). event i는 (ρi, ti). windowed 활성 z^(i)_{f,δ} := z_{ρi, ti+δ}[f], window mean z̄^(i)_f := (1/(2w+1)) Σ_δ z^(i)_{f,δ}.

**Event-aligned ranking (핵심 grounding 수식).** 각 event i에서 feature f의 활성 trace가 3개 template 중 하나(pulse / step-up / step-down)에 얼마나 맞는지. window mean을 빼서 **baseline 활성 크기가 아니라 시간형태(temporal shape)만** 보게 함:

  s^q_i(f) = max( Σ_{δ=−w}^{w} (z^(i)_{f,δ} − z̄^(i)_f) · q(δ) ,  0 )          … (Eq.2)

  q_pulse(δ) = 1 − |δ|/(w+1),   q_up(δ) = (−1 if δ<0 else +1),   q_down = −q_up   … (Eq.3)

각 template q는 mean-center + ℓ2-normalize 후 projection (cross-template max를 공통 scale에서 비교). cluster-level 점수(episode 균등 가중, best-matching template):

  A_{r,f} = (1/|E_r|) Σ_{e∈E_r} max_{q∈Q} (1/|C_{r,e}|) Σ_{i∈C_{r,e}} s^q_i(f)      … (Eq.4)

suite-level 점수(cluster 균등 평균):

  R_event(f) = (1/|R|) Σ_{r∈R} A_{r,f}                                            … (Eq.5)

**비교용 3개 baseline ranking** (모두 같은 SAE, 다른 선택 기준):
- Window-mean (Eq.6,7): 같은 event window지만 temporal shape 무시, plain window-mean 활성. suite-level은 event count nr 가중.
- Task-mean (Eq.8): event window 자체 무시, 전 (rollout,timestep) 평균 활성. R_task(f) = mean_{(ρ,t)} z_{ρ,t}[f].
- Random-alive: alive feature(≥1 rollout에서 활성)에서 uniform 샘플, 단 다른 3 ranking이 이미 고른 feature 제외 → lower bound control.

즉 **"event-grounded" = feature를 external behavioral event window의 활성 시간형태로 랭킹해 causal test 후보를 고르는 것.** 학습 inductive bias가 아니라 **selection/scoring inductive bias.**

### 2.3 Intervention (steering) primitive — residual-preserving latent edit (Sec 3.5)

hidden state x∈R^d, code z=Enc(x)∈R^m. target subset S를 α로 scale:

  z'_i = α·z_i  (i∈S),   z'_i = z_i  (i∉S)                                        … (Eq.1)

α=0 zero-out, α∈(0,1) soft suppress, α=1 no-edit, α>1 amplify. **full reconstruction으로 덮어쓰지 않고 edit로 인한 변화분만 적용:**

  x' = x + Dec(z') − Dec(z) = Dec(z') + err(x),   where err(x)=x−Dec(Enc(x))

즉 hidden state의 **SAE로 설명되는 성분만 수정하고 reconstruction error는 보존**. 이것이 prior work(Swann et al.[14])의 fixed decoder-vector addition x' = x + α Σ_{i∈S} d_i (현재 sample의 latent 값과 무관)와 다른 점. Appendix K에서 decoder-vector add는 α=150일 때 relative update magnitude ρ=‖αd_i‖/‖x_t‖ ≈ 1.54 > 1 이라 native residual stream을 압도(=덮어씀); residual-preserving edit이 이 overwrite를 피한다고 주장.

---

## 3. SAE 아키텍처·학습 상세

- **SAE 종류**: BatchTopK SAE[33]. encoder z=BatchTopK(W_enc h + b_enc), decoder ĥ=W_dec z + b_dec. BatchTopK = minibatch(B) 전체에서 top (k·B) pre-activation만 남기고 zero → L1 penalty를 architectural top-k로 대체. **active budget k = 64** (per-example 평균 활성 수).
- **Tap site**: **post-block residual stream (resid_post)**, per-token 활성 (**mean-pool 안 함** — token-across pooling 금지). Appendix L sanity check: resid_post가 mlp_out보다 reconstruction fidelity는 낮지만(0.67 vs 0.83) causal effect가 확실히 커서 main hook으로 채택.
- **stream × layer × SAE size** (Table 1, Table 5a):
  - OpenVLA residual (action-token decoding 중 LM block): layers {0,16,24,31}, 4096→**32768** (16× expansion). lr 5e-5, 4k steps.
  - π0.5 PG backbone (PaliGemma prefix-prefill block): layers {0,5,11,16}, 2048→**2048** (1×). PG layer 17 제외(prefix hidden이 KV-cache 구성 후 폐기라 post-block edit이 AE에 못 감). lr 1e-4, 10k steps.
  - π0.5 AE action expert (denoising block): layers {0,5,11,17}, 1024→**1024** (1×). AE 활성은 (query timestep, denoising step, action-token offset j)로 index, offset j는 env timestep t+j에 정렬. lr 1e-4, 10k steps.
- **공통 hyperparam**: batch 40,000, Adam, warmup 1000, decay start 80%, activation normalization on, float32.
- **데이터**: suite당 task마다 50 closed-loop rollout의 per-token residual 활성.
- **Compute**: A100, SAE 하나당 few GPU-hours, LIBERO rollout당 1–2분.
- **SAE 품질 지표**: offline (FVE, alive fraction, avg L0) + **behavioral fidelity = Hooked SR** (x를 Dec(Enc(x))로 대체하는 reconstruction-only hook 하 closed-loop SR). Hooked SR로 downstream layer 선정. 발견: FVE는 다 높지만(≥0.91) alive fraction·FVE 어느 것도 Hooked SR을 예측 못 함 → Hooked SR을 primary fidelity metric으로.

---

## 4. feature ↔ event 대응 평가 + event-grounding의 이득

### 4.1 대응 시각화 (Figure 8, Appendix F)
event-aligned score heatmap (행=event cluster, 열=top layer-31 SAE feature, 색=Eq.5 score). 관찰: **sparse·structured**. 소수 feature가 여러 VLM-labeled event 행에 걸쳐 high-scoring vertical band 형성 → 일부 feature는 단일 event가 아니라 **의미적으로 관련된 manipulation phase(approach/contact/transport/release)에 걸쳐 재발**. 다른 high 항목은 개별 행에 localize(event-specific 후보). **단 이 recurrence는 OpenVLA에만.** heatmap은 discovery aid일 뿐, causal relevance는 zero-out으로 검증.

### 4.2 causal 검증 = event-grounding이 unsupervised 대비 낫다는 증거 (Table 3)
top-K feature(OpenVLA K=5, π0.5 K=3)를 각각 따로 zero-out(α=0), ΔSR = SR_{α=0} − SR_baseline. ranking이 random-alive control보다 더 negative하면 informative.

- **OpenVLA layer 31** (baseline 70.0%): Event-aligned **−21.2** ≫ Window-mean −6.2 ≈ Task-mean −6.5 ≫ Random-alive −1.3. → **temporal event-alignment이 활성 크기(magnitude)만으로는 못 얻는 causal 정보를 더한다.** top-3 feature가 SR drop의 거의 전부를 담당(Appendix I). ranking overlap(Table 4): event-aligned는 OpenVLA에서 window/task와 **5%만 겹침**(거의 disjoint), window-mean≈task-mean(95–97%). → event-aligned은 "temporal change에 큰 baseline magnitude가 필요 없음"을 보임.
- **π0.5 PG backbone** (baseline 96.8%): 모든 layer·ranking에서 single-feature edit이 SR 거의 안 건드림(−0.2 ~ −2.7). cross-attention KV-cache가 각 prefix token을 섞어 single-feature edit을 action 전에 희석. joint로 16 PG feature 떨어뜨려야 moderate 변화(Appendix H; PG layer 0만 예외로 −89.8, instruction grounding 담당 시사).
- **π0.5 AE** (baseline 96.4%): event/window/task ranking 모두 top feature가 SR을 **~0으로 붕괴**(−96 근처), random-alive도 상당(−7 ~ −23). → **event-specific selectivity가 아니라 broad causal sensitivity**. within-ranking ordering 구분 불가. AE feature는 action chunk를 직접 만드는 residual stream을 modulate, bottleneck 없음.

### 4.3 soft dose-response (Figure 4,5)
αf를 0→1 sweep하면 π0.5 AE SR이 collapse↔baseline 사이 monotonic 전이 → 붕괴 정도를 dial로 선택 가능. 단 전이가 **가파른 switch**(좁은 αf band에서 급상승) → real-robot tuning이 mis-calibration에 민감.

### 4.4 static outcome 예측 sanity (Table 10) — 우리 confound와 직결
mean-pooled SAE code z̄_ρ = (1/T)Σ_t z_{ρ,t}로 binary success를 L2-LR probe(5-fold stratified CV)로 예측한 balanced accuracy:

| Suite | SAE code | Raw hidden | **Task-id only** | Shuffled |
|---|---|---|---|---|
| LIBERO-Spatial | 0.794 | 0.976 | 0.550 | 0.498 |
| LIBERO-Object | 0.926 | 0.942 | 0.636 | 0.498 |
| LIBERO-Goal | 0.837 | 0.896 | 0.627 | 0.576 |
| LIBERO-10 | 0.791 | 0.893 | 0.538 | 0.484 |

→ SAE code가 outcome 신호를 담지만 lossy(raw hidden보다 낮음). **주의: task-id만으로도 0.54–0.64가 나온다** = success 예측에 task/scene confound가 이미 섞여 있음. 저자는 length/scene control 안 함, mean-pool은 static probe에만 쓰고 main result는 per-token. **이 표가 곧 우리 팀이 겪는 "outcome이 scene/task와 얽힘" 문제를 이 논문도 통제 안 하고 지나갔다는 증거.**

---

## 5. control / steering

- **한다.** residual-preserving latent edit(Eq.1, x'=x+Dec(z')−Dec(z))이 main steering primitive. hard zero-out(causality on/off readout) + soft α sweep(dose-response).
- **real-robot** (Sec 4.4, Fig 5): LoRA π0.5로 "red/yellow chips 접근". AE feature suppression이 color-conditioned approach를 측정 가능하게 바꿈, 최강 효과 **AE layer 17**(강한 suppression → prompted cluster에서 더 큰 이탈). early layer는 noisy. 결론: SAE coordinate가 real-robot prompt-following에 영향, effect는 layer/feature/strength 의존. **안전상 conservative suppression만** 권고(aggressive edit은 arm 불안정, 사람 위험).
- **decoder-vector baseline** (Appendix K, Table 9): Swann식 x'=x+αd_i (α=150)는 SR 0% vs matched random 52% → d_i가 behaviorally relevant. 그러나 ρ≈1.54>1 이라 residual을 덮어씀(clean feature edit 아님). residual-preserving edit이 대안.
- **FFN value-vector baseline** (Appendix E, Häon et al.[15] 재현): lexical(red/yellow) 정렬 vector steering은 prompt-only baseline보다 훨씬 noisy, random-vector control과 overlap → "VLA에서 language·action subspace가 부분적으로 얽혀 lexical alignment는 약하고 간접적인 behavioral control proxy". 이게 event-grounding 동기.

---

## 6. 평가·headline·한계·confound

**Headline 숫자.**
- OpenVLA는 **layer 31만** closed-loop fidelity 유지(Hooked SR 34.8% vs raw ~70%); early layer는 discrete-token quantization(7 token × 256 bin) error가 step 간 누적 → near-zero action stall. π0.5는 flow-matching continuous chunk라 robust, 모든 probed layer에서 Hooked SR ≥95%.
- event-aligned ranking causal 우위는 **OpenVLA layer 31에서만** 깨끗(−21.2 vs random −1.3).

**저자 인정 한계 (Sec 5).**
1. **Shared, not task-specific features**: target/off-target zero-out probe(Table 8) — target task SR 0.641 vs off-target 0.631 (거의 동일). 즉 event-selected feature가 task-private circuit이 아니라 reach/grasp/transport/release처럼 **task 간 공유 manipulation role**을 함 → single-feature edit으로 하나의 behavioral factor를 isolate 불가.
2. **VLM semantics 미활용**: phrase/phase label이 시각화 aid뿐, ranking·intervention에 안 들어감.
3. **coarse, safety-limited 평가**: binary SR은 edit이 얼마나 selective하게 바꾸는지 못 잼; 강한 edit은 real robot을 불안정하게 해서 mild intervention만 안전.
- 추가: SAE feature는 polysemantic, 큰 edit은 policy를 off-distribution으로. hard zero-out은 "ranked coordinate의 causal sensitivity"를 probe할 뿐 clean factor-level isolation이 아님. 완전한 mechanistic account 주장 안 함.

**Confound 관점(우리 시각).** 이 논문은 outcome을 분석 target으로 삼지 않았다. event는 **outcome-agnostic**(approach/contact는 성공·실패 모두에서 발생). success 예측은 Table 10의 곁다리 probe이고 length/scene을 통제하지 않는다. 즉 **"event-grounding이 outcome을 scene/length에서 분리한다"는 근거는 논문에 없다.**

---

## 7. 우리 적용성 (GR00T N1.5 latent steering) — 정직한 판정

### 7.1 그들의 event ↔ 우리 event/phase 매핑
- 그들의 event = task-local kinematic keyframe cluster(approach/contact/transport/release/withdrawal), AWE+clustering+VLM로 **추정**한 것. 우리는 sim event **grasp/place/release**와 phase **reach/transport/insert-settle**를 **정답(ground-truth)** 으로 이미 갖고 있다. → 그들의 Stage 2·3(AWE·agglomerative·VLM) 휴리스틱을 **통째로 스킵**하고 정확한 event timestamp로 대체 가능. 이건 event 품질에서 우리가 그들보다 엄격하게 우위.
- 우리 phase 3개(reach/transport/insert-settle)는 그들의 6-way phase taxonomy의 실용 축소판. 그들은 이 label을 안 썼지만 우리는 이걸 **stratification 변수**로 정면 활용할 수 있다(그들이 남긴 gap = limitation (ii)를 우리가 메움).

### 7.2 그런데 — event-grounding이 우리 핵심 문제(outcome vs scene/length 분리)를 풀어주나? → **아니다, 그대로는 안 된다.**
- 그들의 "grounding"은 **feature SELECTION heuristic**이지 **outcome disentanglement**가 아니다. event-aligned ranking은 "event 순간에 pulse/step치는 feature"를 고를 뿐, success feature와 failure feature를 구분하지 않는다. 그들은 outcome subspace를 만들지 않는다.
- 우리에게 진짜 유용한 재해석: **event/phase를 matching/조건화 변수로 써서, "같은 event·phase 안에서" succ/fail contrast를 계산**한다. 우리 memory(seen18 length confound: 실패=항상 timeout이라 시간-pooled 분리가 아티팩트)를 정확히 겨냥한다. 즉 event-grounding의 가치는 **loss가 아니라 confound를 통제하는 stratification 축**에 있다. contrastive conceptor C_steer = C_success ∧ ¬C_failure를 **phase bin별로 따로 fit**하면 length/phase confound가 구조적으로 제거된다(matching on event/phase). 이게 우리가 실제로 가져갈 것.

### 7.3 구체적 build sketch (GR00T N1.5)
1. **Tap**: 그들의 3-stream 교훈을 우리 2-stream에 대응.
   - **DiT block residual [L=7, K=4, D=1536]** ← 그들의 **π0.5 AE**에 해당(flow-matching action expert, denoising step index = 우리 K=4, action-token offset = env timestep 정렬). BatchTopK SAE, per-token, resid_post, k≈64.
   - **VL [2048]** ← 그들의 **π0.5 PG backbone**에 해당(2048-d, cross-attention으로 DiT에 간접 전달). SAE size 1× (2048→2048~4× 사이).
2. **Event window**: sim grasp/place/release timestamp ±w로 window, phase(reach/transport/insert-settle) bin. 우리 memory(chunk predict16/execute5 → feature 5-step 해상도)에 맞춰 w를 phase 길이에 비례로 잡기.
3. **Outcome contrast (그들에게 없는 우리 확장)**: event-aligned "important feature 랭킹" 대신 **phase-conditioned succ/fail 방향**을 뽑는다. 두 옵션:
   - (a) 각 phase bin에서 SAE code z를 succ/fail로 나눠 conceptor C_succ, C_fail fit → C_steer = C_succ ∧ ¬C_fail. length는 phase bin으로 통제.
   - (b) (진짜 "event-grounded SAE"로 가려면) SAE 학습에 **auxiliary event/phase head** 추가: L = L_recon + λ_aux L_auxK + λ_ph · CE(head(z), phase) 또는 outcome head. 이건 **논문에 없는 우리 자체 확장**(그들의 grounding은 selection-only). phase-supervised sparse dictionary(phase별 서브딕셔너리 partition)도 가능. 단 이건 backbone 재학습은 아니지만 SAE 학습 비용·라벨 필요.
4. **Steering primitive**: 그들의 residual-preserving edit x'=x+Dec(z')−Dec(z)을 SAE-space steering에 채택(raw decoder-vector add보다 안전, ρ<1 유지). 우리 conceptor는 multi-dim이라 그들의 single-feature scaling보다 shared/polysemantic feature 문제에 더 강함.

### 7.4 데이터·라벨 요구량 vs 우리 3–4 failures/scene — 리스크 정면 평가
- **event-grounding은 low-power(소수 실패) 문제를 완화하지 못한다.** 그들의 causal scoring은 suite-level에서 episode·cluster 평균(50 rollout/task)에 의존하고, success probe(Table 10)도 balanced CV를 썼다. scene당 실패 3–4개면 SAE-code outcome probe/conceptor fit이 우리 기존 AUROC≈1 confound처럼 **overfit**한다. phase-conditioning은 length confound를 줄이지만 **표본 수(통계적 power)를 늘려주지 않는다.** → 여전히 (i) 더 많은 실패 수집, 또는 (ii) scene을 통제 covariate로 두고 여러 scene에 pool하는 설계가 필요.
- **아키텍처 리스크(핵심 경고)**: 우리 GR00T DiT는 그들의 π0.5 AE와 가장 가깝다. 그들의 AE 결과 = **거의 모든 ranking(random 포함)에서 broad collapse, event-selectivity 없음, switch-like dose-response**. 즉 GR00T DiT에서 SAE는 **넓게 causal하지만 selective하지 않은, 붕괴하기 쉬운 intervention basis**일 가능성이 높다. clean event-specific feature를 기대하면 안 되고, conceptor(부분공간) 접근이 single-feature보다 적합. VL[2048]은 그들의 PG처럼 **single-feature edit에 약할** 수 있음(KV-cache/cross-attn 희석) → VL steering 기대 낮추기.
- **polysemantic/shared feature**: target/off-target 동률(Table 8)은 우리 "shared failure zone" memory와 일치. single feature로 outcome isolation은 비현실적 → multi-dim contrastive conceptor로 가는 우리 방향이 옳음을 뒷받침.

### 7.5 결론적 verdict
- 채택할 것: (1) **event/phase를 confound 통제용 stratification 축으로** 쓰는 발상(→ phase-conditioned conceptor fit), (2) **residual-preserving latent edit** steering primitive, (3) **per-token, resid_post, BatchTopK(k≈64)** SAE 설계 관례, (4) 아키텍처별 causal 특성 경고(DiT≈AE broad/collapse, VL≈PG weak).
- 채택하지 않을(또는 확장해야 할) 것: 그들의 "event-grounding"은 selection-only라 **outcome을 scene/length에서 분리해주지 않는다.** 진짜 event-grounded outcome SAE를 원하면 auxiliary phase/outcome loss 또는 phase-partitioned dictionary를 **우리가 추가**해야 하고, 이는 라벨·학습 비용을 부르되 소수-실패 통계 power 문제는 안 풀어준다.
- 한 문장: **이 논문은 "event로 feature를 고르는 법"을 주지만 "outcome을 confound에서 떼는 법"은 안 준다. 우리는 event를 selection이 아니라 conditioning/matching 축으로 재해석해 phase-conditioned succ/fail contrastive conceptor를 만드는 데 써야 하고, 그래도 3–4 failures/scene의 low-power 문제는 별도로(더 많은 실패 or scene-pooling) 해결해야 한다.**

---

## 8. 애매/불명확 지점 (추측 아님, 명시)
- 그들이 GR00T를 다루지 않으므로 DiT[7,4,1536]·VL[2048] 매핑은 π0.5 AE/PG와의 **구조 유사성에 근거한 우리 유추**이지 논문 주장이 아님.
- 6-way VLM phase label을 실제로 어떤 예시에 매핑했는지의 세부 예시는 Figure 9(keyframe bundle)에 있으나 텍스트 추출로는 확인 불가(이미지). 본 노트는 taxonomy 이름과 prompt(Appendix C)까지만 확정.
- "recurrence across phases는 OpenVLA에만"이라고 명시되었으나 π0.5에서 왜 heatmap band가 안 생기는지의 정량 근거는 논문에 없음(관찰만).
