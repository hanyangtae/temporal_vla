# Locating and Editing Factual Associations in GPT (Meng, Bau, Andonian, Belinkov 2022)

- 출처: NeurIPS 2022. arXiv:2202.05262 (v5, 2023-01-13)
- PDF: `docs/Activation_steering_basic/ROME_2202.05262.pdf`
- 정독 섹션: §2 Interventions on Activations for Tracing Information Flow (causal tracing 분석 중심), §3 개관
- tier: must
- 한줄역할: activation patching / causal mediation analysis를 LLM 해석에 도입한 원조 논문. "어느 위치의 활성화가 인과적으로 출력에 기여하는가"를 정량화하는 방법론(causal tracing)을 정립하고, 그 결과를 이용해 rank-one weight edit(ROME)까지 이어감.

## 문제·동기

GPT류 autoregressive transformer가 사실 지식("Space Needle is in Seattle")을 어디에 저장하는가는 그때까지 불명확했음. masked LM(BERT)에서는 Geva et al.(2021), Dai et al.(2022) 등 선행연구가 있었지만, 단방향 attention·생성 구조를 가진 GPT류에는 적용 안 됨. probing classifier 기반 분석은 "표현에 정보가 있다"는 상관관계만 보여줄 뿐 "그 정보가 실제로 예측에 인과적으로 쓰이는가"는 답하지 못한다는 것이 핵심 문제의식(Belinkov 2021 비판 인용).

## 핵심 아이디어

두 단계 접근.
1. **Causal Tracing**: hidden state 활성화 각각의 indirect effect(간접 인과효과)를 측정해 "사실 회상에 결정적인 위치"를 특정.
2. **ROME**: 그 위치(중간층 MLP, subject 마지막 토큰)의 weight를 rank-one 업데이트로 직접 수정해 causal tracing 가설을 검증(locate 결과가 실제로 edit 가능한 저장소인지 재확인).

두 단계는 상호검증 구조: causal tracing이 "어디"를 가리키고, ROME 편집이 그 위치에서 실제로 유효하면 가설이 강화됨(§3.4에서 layer-token grid로 상관 확인, Figure 5).

## 방법

### Causal Tracing (§2.1, 핵심 배정 섹션)

세 번의 forward run을 비교하는 causal mediation analysis(Pearl 2001; Vig et al. 2020b 확장):

- **Clean run**: 정상 프롬프트 x를 넣고 모든 hidden state {h_i^(l)} 수집. 정답 o(예: Seattle) 확률 P[o].
- **Corrupted run**: subject 토큰 임베딩에 가우시안 노이즈 ε~N(0, ν)를 더해 subject 정보를 훼손. ν = 3×(토큰 임베딩 표준편차). 확률 P*[o]는 크게 하락(27.0%→8.47%, GPT-2 XL 평균).
- **Corrupted-with-restoration run**: corrupted 상태로 계산하다가 특정 (token i, layer l) 하나만 clean run의 값으로 강제로 되돌리는(hook/patch) 개입. 이후 계산은 개입 없이 진행.

정의:
- Total Effect (TE) = P[o] − P*[o]
- Indirect Effect (IE) of h_i^(l) = P*,clean h_i^(l)[o] − P*[o] (해당 state 하나를 복원했을 때 얼마나 정답 확률이 회복되는지)
- 1000개 factual statement에 대해 평균 → Average Indirect Effect (AIE)

결과(§2.2, GPT-2 XL 1.5B): ATE=18.6%. 두 개의 강한 causal site 발견 — (a) "late site": 마지막 토큰, 늦은 layer(attention 지배, 예상된 결과), (b) "early site": **subject의 마지막 토큰, 중간 layer** — 이것이 새로운 발견. MLP 기여가 이 early site를 지배(AIE 피크 6.6%, attention은 1.6%에 불과).

**modified causal graph로 MLP 역할 격리(§2.2, Figure 3)**: MLP 경로를 corrupted 상태로 severed(고정)시킨 채 hidden state만 복원하면, early site(낮은 layer)의 causal effect가 사라짐 → 그 인과효과가 MLP 계산 자체를 반드시 거쳐야 함을 확인. late site에서는 이런 severing 효과가 없음(attention이 주도). 이는 단순 correlation이 아니라 "경로별(path-specific) 인과 효과"를 분리해서 보여주는 실험.

강건성 체크(Appendix B): 노이즈 분포(spherical/multivariate Gaussian, uniform) 바꿔도 결과 유지, integrated gradients(gradient saliency)는 이런 구조화된 패턴을 못 보여줌(=activation patching이 gradient 기반 attribution보다 우월하다는 근거로 제시).

### ROME: MLP as Key-Value Memory + Rank-One Edit (§3)

- MLP 2층(W_fc, W_proj)을 linear associative memory로 봄: W_proj·K ≈ V (Kohonen/Anderson 고전 결과 재사용).
- **Step 1 (key k*)**: subject를 나타내는 여러 prefix 문장에서 subject 마지막 토큰의 MLP 내부(비선형 통과 후) 활성화를 평균 → k*.
- **Step 2 (value v*)**: 목적함수 최소화로 v* 최적화 — (a) MLP 출력을 v*로 치환했을 때 목표 object o* 확률 최대화 + (b) KL divergence 페널티로 "{subject} is a" 류 프롬프트에서 essence(주체 정체성) drift 억제.
- **Step 3 (삽입)**: 제약 최소자승 rank-one update Ŵ = W + Λ(C⁻¹k*)ᵀ. C=KKᵀ는 Wikipedia 텍스트에서 사전 추정한 key covariance(기존 기억 보존용 정규화 역할), Λ은 residual 오차 비례 벡터. 닫힌 해(Appendix A, Lagrangian 유도)라 학습(gradient descent) 없이 즉시 편집 가능(~2초/GPT-2 XL, A6000).
- 편집 layer는 causal tracing이 지목한 early site 중간층(GPT-2 XL: layer 18) 사용 — locate 결과를 edit에 직접 반영.

## 실험·결과

- **zsRE**(Table 1, GPT-2 XL): ROME efficacy 99.8%, paraphrase 88.1%, specificity 24.2% — FT/FT+L/KE/MEND 대비 specificity 최고, efficacy 최고. 단 zsRE의 specificity는 무작위 unrelated fact라 둔감한 지표라고 저자들이 지적.
- **COUNTERFACT**(저자들이 직접 구축, 21,919 records; 어려운 반사실 편집 + neighborhood bleedover + fluency/consistency까지 측정): Score(S, harmonic mean of ES/PS/NS) 기준 ROME이 GPT-2 XL 89.2, GPT-J(6B) 91.5로 최고. 다른 방법들은 F1(overfit, 일반화 실패) 또는 F2(underfit, neighbor bleedover) 중 하나에 빠짐 — ROME만 둘 다 회피.
- **layer/token sweep**(Figure 5): rewrite 성능이 causal tracing이 지목한 subject 마지막 토큰·중간층에서 정확히 피크 — locate와 edit의 정합성 재확인.
- **human eval**: ROME이 FT+L보다 1.8배 더 "counterfactual과 일관"됨으로 평가됨. 단 fluency는 1.3배 더 낮게 평가(자동 entropy 지표는 이 fluency 저하를 못 잡음).
- **attention 대신 edit(AttnEdit, Appendix I)**: late-site attention을 수정하면 원본 prompt는 재현하지만 paraphrase 일반화 실패 — MLP=회상, attention=late 단어예측 이라는 역할 분리 가설을 추가 검증.

## activation-steering 흐름 위치 (patching 계보 원조)

Vig et al.(2020b) causal mediation analysis를 GPT류 autoregressive 구조에 처음 본격 적용해 **"corrupt → 단일 활성화 patch(복원) → 출력 회복량 측정 = 그 활성화의 causal 기여도"** 라는 activation patching의 표준 실험 설계를 확립. 이후 인과적 국소화(localization) → 표적 개입(intervention)이라는 2단계 패턴(mechanistic interpretability 전반의 기본 워크플로)의 초기 대표 사례. 후속 MEMIT(Meng et al. 2022b, 다중 fact 동시 편집)으로 스케일 확장. steering 계보에서는 "어디를 건드려야 하는가"를 데이터 기반으로 찾는 causal localization 단계의 원형이며, activation steering(강도 조절 개입) 자체보다는 weight-edit(영구 개입)이라는 점에서 steering과 개입 형태는 다르지만, "causal attribution → 표적 개입"이라는 방법론적 조상.

## 우리 프로젝트 연결 (인과 귀인·개입)

- 우리의 **pathway 귀인 문제**(VL(goal) vs DiT(motor) 중 어디가 OOD/실패에 기여하는가)는 ROME의 "어느 hidden state가 예측에 인과적으로 기여하는가"와 동형 문제. ROME의 corrupt-and-restore 패러다임은 우리 online detection 실험에서 causal(=length-fair, 미래 정보 누출 없는) 검증 기준으로 이미 참고되고 있는 사고방식과 일치(예: N16 online detection 실험의 "causal" 프레이밍).
- ROME이 "correlation(probing)으로는 부족하고 causal intervention이 필요하다"고 강하게 주장하는 지점은 우리가 activation steering의 인과성(ΔSR 재측정)을 요구하는 것과 같은 근거 구조 — 표현 분리(succ/fail latent 분리, AUROC 등)만으로는 부족하고 steering 개입 후 SR 변화로 causal 검증해야 한다는 우리 프로젝트 원칙과 정확히 대응.
- MLP를 key-value associative memory로 보는 관점과 rank-one 삽입 수식(Ŵ = W + Λ(C⁻¹k*)ᵀ)은 우리의 conceptor 수학(C_steer = C_success ∧ ¬C_failure, multi-dim 연산자)과 표면적으로 유사한 "선형대수적 subspace 개입"이지만, ROME은 weight를 영구히 바꾸는 model editing이고 우리는 추론 시 activation에 곱하는 steering(h' = h·Mᵀ)이라는 점에서 개입 대상(weight vs activation)이 다름 — 이 차이를 면접에서 구분해 설명할 필요.
- "early site(MLP, subject 처리 중) vs late site(attention, 출력 직전)"라는 역할 분리 발견은, 우리의 Eagle→VL-SA→DiT 직렬 구조에서 "어느 단계가 어떤 종류의 정보를 인과적으로 결정하는가"를 찾는 작업과 방법론적으로 유사 — 다만 우리는 다운스트림 결합(직렬 confound)이 있어 ROME처럼 깔끔하게 두 site로 분리되지 않을 수 있음에 유의.

## 면접 포인트 (Q→A)

**Q1. Correlation vs causation을 이 논문은 어떻게 구분하는가?**
A. Probing classifier는 표현에 정보가 선형분리 가능하다는 상관관계만 보여줄 뿐, 모델이 실제로 그 정보를 예측에 쓰는지는 말해주지 않는다(Belinkov 2021 비판). ROME은 corrupted run에서 개별 hidden state 하나만 clean 값으로 되돌리는 patch 개입을 하고, 출력 확률이 얼마나 회복되는지(Indirect Effect)를 측정한다. 이건 관찰이 아니라 개입(do-operator, Pearl의 causal mediation analysis)이므로 인과적 주장이 성립한다.

**Q2. Activation patching이 정확히 뭔가, 이 논문 기준으로 설명하라.**
A. 세 번의 forward pass(clean, corrupted, corrupted-with-restoration)를 돌려서, corrupted input(주어 임베딩에 노이즈 주입)으로 예측이 망가진 상태에서 특정 (layer, token) 하나의 활성화 값만 clean run의 값으로 바꿔치기(patch)하고 나머지는 그대로 corrupted 상태로 진행시킨다. 그 하나의 patch만으로 정답 확률이 얼마나 되돌아오는지가 그 활성화의 causal 중요도(indirect effect)다.

**Q3. 왜 노이즈 크기(3σ)를 그렇게 크게 잡았나, 너무 작으면?**
A. Appendix B.4: 노이즈가 너무 작으면(σ=σt) total effect 자체가 작아서 clean과 corrupted 차이가 잘 안 보여 indirect effect 측정이 어려워진다(신호 대 잡음). 3σt는 total effect를 27.0%→8.47%로 충분히 떨어뜨려 개입 효과를 구별 가능하게 만드는 실용적 선택이며, 여러 노이즈 분포(uniform, multivariate Gaussian)로 강건성도 확인했다.

**Q4. MLP를 key-value memory로 본다는 게 무슨 뜻이고, rank-one edit은 왜 가능한가?**
A. 2층 MLP의 W_proj를 선형 연산자로 보면 WK≈V를 푸는 associative memory로 해석 가능(최소자승 해 W=VK⁺). 새 사실 하나 = 새 (key, value) 쌍 하나 추가이므로, 기존 최소자승 해에 등식제약(Ŵk*=v*) 하나만 추가한 constrained least-squares 문제가 되고, 이건 정확히 rank-one(Λ(C⁻¹k*)ᵀ) 업데이트로 닫힌 해가 나온다(Lagrangian 유도, Appendix A). 학습 없이 즉시 계산 가능한 게 장점.

**Q5. FT(fine-tuning)나 hypernetwork(KE, MEND) 대비 ROME이 이기는 이유는?**
A. FT는 특정 wording에 overfit해서 neighborhood(관련 없는 이웃 subject)까지 틀리게 만들거나(F2, bleedover), 반대로 L∞ 제약을 걸면(FT+L) generalization을 잃는다(F1). Hypernetwork(KE, MEND)는 두 문제를 동시에 보이며 종종 단순 regurgitation(단어 반복)에 그친다. ROME은 causal tracing으로 정확히 국소화된 위치에 최소 개입(rank-one, KL essence-drift 제약 포함)을 가하기 때문에 specificity와 generalization을 동시에 유지한다.

**Q6. 이 논문의 한계는 뭔가, 우리 연구에 어떤 시사점이 있나?**
A. 단일 fact 편집만 가능(다중 동시 편집은 후속 MEMIT), 편집이 방향성(directional)이라 역방향 관계는 별도 편집 필요, 논리·공간·수치 지식은 검증 안 됨, 편집 후 모델이 그럴듯하지만 근거 없는 새 사실을 "추측"하는 문제도 있음. 우리 맥락에서는 — causal localization이 곧 "좋은 개입 지점"이라는 가정 자체가 후속 비판(Hase et al. 2023, "Does Localization Inform Editing?")을 받는다는 점을 알아둘 필요: 우리 pathway/phase 귀인도 "여기가 인과적으로 중요하다"가 곧 "여기를 steer하면 성공적으로 고쳐진다"를 보장하지 않을 수 있다.

## 한계·비판

- **국소화가 곧 최적 개입 지점이라는 가정의 취약성**: 후속 연구(Hase, Bansal et al. 2023, "Does Localization Inform Editing?")는 causal tracing이 지목한 layer가 아닌 다른 layer에서 edit해도 비슷하게 잘 작동함을 보여 causal tracing과 "어디를 edit해야 하는가"의 연결이 저자들 주장만큼 타이트하지 않을 수 있음을 지적.
- **corruption rule의 임의성**: 가우시안 노이즈로 subject를 "지운다"는 조작 자체가 자연스러운 실패 모드를 대표하는지는 불확실(저자들도 노이즈 종류를 바꿔가며 강건성만 확인, 근본적 정당화는 약함).
- **fluency 저하**: 자동 지표(entropy 기반 GE)로는 안 잡히는 fluency 손실을 human eval이 드러냄 — 정량 지표의 맹점.
- **스케일 한계**: 단일 fact 전용, 실제 활용(수천~수만 fact 동시 수정)에는 부적합(저자도 인정, MEMIT으로 후속 해결).
- **"지식 편집=신념 변경"이라는 프레이밍 자체에 대한 논쟁**: 모델이 정말 "사실을 새로 저장"한 것인지, 국소적 표면 패턴만 바뀐 것인지는 여전히 논쟁적(편집 후 guessing behavior가 그 증거).
