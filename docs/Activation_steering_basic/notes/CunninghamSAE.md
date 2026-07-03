# Sparse Autoencoders Find Highly Interpretable Features in Language Models (Cunningham et al. 2023)

- 출처: EleutherAI / MATS / Bristol AI Safety Centre / Apollo Research. ICLR 2024 (원 게재 arXiv).
- arXiv: 2309.08600 (v3, 2023-10-04)
- PDF: `docs/Activation_steering_basic/CunninghamSAE_2309.08600.pdf`
- 정독 섹션: §2(방법, 집중) + §1·§3·§4·§5·§6·부록 A/B/C/E/G 훑음
- tier: must
- 한줄역할: SAE를 LLM residual stream에 적용해 monosemantic feature dictionary를 학습한 원조격 논문(Anthropic Towards Monosemanticity와 동시기). Activation-steering 파이프라인에서 "무엇을 조작할 방향인가"를 unsupervised로 찾는 feature-discovery 단계의 토대.

## 문제·동기(superposition·polysemanticity)

- 뉴런 단위 해석의 한계: 개별 뉴런이 여러 무관한 맥락에서 활성화되는 polysemanticity 관찰(Olah et al. 2020). residual stream처럼 뉴런 basis에 정렬될 이유가 없는 activation은 더 심함.
- 원인 가설 = superposition(Elhage et al. 2022b, Toy Models of Superposition): 모델이 레이어 차원 수보다 많은 feature를 표현하려고, overcomplete하고 서로 비직교인 방향들에 feature를 나눠 담음. 벡터공간은 차원 수만큼만 직교 벡터를 가질 수 있으므로 이는 필연적으로 non-orthogonal overcomplete basis가 됨.
- superposition이 성립하려면 feature가 충분히 sparse하게 활성화해야 함(sparse하지 않으면 비직교 feature 간 간섭이 성능을 깎아 superposition의 이득이 없어짐) → 따라서 "activation을 sparse dictionary(=Olshausen & Field 1997의 sparse coding)로 분해하면 원래 feature를 복원할 수 있다"는 것이 핵심 통찰.

## 핵심 아이디어

- activation x_i가 알려지지 않은 ground-truth feature 집합 {g_j}의 sparse linear combination이라고 가정: x_i = Σ_j a_ij g_j (a_i sparse).
- dictionary {f_k}를 학습해 각 g_j ≈ 어떤 f_k에 대응하도록 만든다. 이는 완전 unsupervised, task-agnostic(특정 downstream task를 겨냥해 학습하지 않음).
- 1-hidden-layer autoencoder + L1 sparsity penalty를 hidden code에 부여 = sparse dictionary learning을 신경망으로 구현.

## 방법(sparse dictionary learning, L1, feature 해석)

- 구조: encoder c = ReLU(Mx+b), decoder x̂ = M^T c = Σ_i c_i f_i. M ∈ R^(d_hid×d_in), 각 행(=dictionary feature f_i)은 row-normalize. d_hid = R·d_in (R=overcompleteness ratio, hyperparameter).
- **tied weight**(encoder=decoder^T) 사용 이유: (a) feature를 검출하는 방향과 정의하는 방향이 같아야 한다는 기대, (b) 파라미터/메모리 절반, (c) encoder/decoder 방향 중 어느 게 "진짜" feature 방향인지의 모호성 제거. residual stream에선 성능 저하 없음, MLP data에선 저하 관찰 → MLP는 별도로 encoder/decoder 분리(Eq 5,6, M_e ≠ M_d).
- Loss: L(x) = ||x - x̂||²₂ (reconstruction) + α||c||₁ (sparsity). α가 sparsity-accuracy tradeoff를 조절. Appendix B: α sweep에서 smooth tradeoff만 관찰(knee/bump 없음) → "유일하게 옳은 분해"라는 원칙적 기준 부재를 시사.
- 대상: Pythia-70M(d=512), Pythia-410M(d=1024)의 residual stream (주로), MLP sublayer(부분적).
- 학습: Pile 데이터로 activation 캐싱 → Adam lr=1e-3, 5-50M activation vector, 1-3 epoch, A40 1장으로 1시간 이내.
- 해석(§3): OpenAI autointerpretability 프로토콜(Bills et al. 2023) 재사용 — feature가 활성화되는 top 20 텍스트 조각을 GPT-4에 보여줘 설명 생성 → GPT-3.5가 그 설명만 보고 다른 텍스트에서의 활성화를 시뮬레이션 → 실제 활성화와의 상관계수 = interpretability score(top-random scoring).

## 실험·결과(interpretability·인과 개입)

- §3 해석성: SAE feature가 baseline(default basis, random directions, PCA, ICA) 대비 평균적으로 훨씬 높은 top-random interpretability score(Fig.2). 다만 초기 레이어에서 격차가 크고, 레이어가 깊어질수록 격차가 줄어 마지막 레이어에서는 ICA와 비슷해짐(원인 불명 — SAE 자체 한계 vs 자동해석 방법의 한계가 혼재).
- §4 인과 개입(activation patching, IOI task): target(counterfactual) 문장의 dictionary feature 활성값을 base 문장에 patch. ACDC 알고리즘(Conmy et al. 2023)으로 feature 순서를 매겨 KL divergence를 낮추는 순으로 patch. 결과: 같은 KL divergence(target과의 근접도)에 도달하는 데 **SAE feature가 PCA component보다 더 적은 patch 수, 더 작은 edit magnitude**로 충분(Pareto frontier 개선, Fig.3). non-sparse dictionary(α=0)는 이 이점이 사라짐 → sparsity 자체가 핵심 원인.
- §5 case study: input(활성 토큰 — 아포스트로피 전용 feature 등, 세부 맥락별로 별도 feature로 분리됨), output(feature ablation 시 다음 토큰 logit이 직관적으로 변화 — 아포스트로피 feature 제거 시 다음 "s" logit 감소), intermediate(이전 레이어 feature를 ablation해 다음 레이어 target feature 활성 감소량으로 causal tree/circuit 재구성 — 닫는 괄호 feature 사례).
- 부록: reconstruction이 완전하지 않음 — Pythia-70M layer 2 residual을 재구성값으로 치환 시 Pile perplexity가 25→40으로 상승(정보 손실 존재). MLP sublayer는 dead feature(전혀 활성 안 함)가 많아 성공 제한적, tied weight 풀어도 middle/late MLP는 진짜 overcomplete basis 학습 실패.

## activation-steering 흐름 위치

- 이 논문 자체는 steering을 직접 수행하지 않고 "feature 발견(discovery)+해석+causal 검증(ablation/patching)"에 집중. Anthropic의 Towards Monosemanticity(Bricken et al. 2023)와 거의 동시기에 SAE-on-LLM-residual-stream을 성공적으로 보인 두 편 중 하나.
- steering 파이프라인 관점: CAA/ActAdd류(그 뒤 tier)는 positive/negative pair의 activation 차이로 "하나의" steering 방향을 supervised하게 찾는 반면, 이 논문의 SAE는 라벨 없이 대량(overcomplete) feature dictionary를 통째로 학습 — 그 중 원하는 concept에 대응하는 monosemantic feature를 뽑아 steering vector(add/ablate)로 재사용할 수 있는 후보 pool을 만든다.
- §5.2의 ablation(feature 제거 → 다음 토큰 logit이 예측 가능하게 변화)이 이미 최소 단위의 "steering" 증거 — feature 방향을 빼거나 스케일하면 원하는 방향으로 출력이 바뀐다는 것을 보여, 이후 "SAE feature를 steering vector로 add/remove"하는 연구(GemmaScope, feature-steering 논문류, AxBench 등)의 이론적 근거가 됨.

## 우리 프로젝트 연결

- SAE는 우리 프로젝트에서 concept 해석 + 자동 실패데이터 수집 후보로 이미 표시됨(NOTALL이 GR00T DiT에 per-token SAE 수십 개를 적용). 이 논문은 그 NOTALL이 쓰는 SAE 학습 레시피(tied-weight 1-hidden-layer autoencoder, L1 sparsity, per-token activation 캐싱)의 원형에 해당.
- 적용 가능성: VL-SA / DiT residual activation에 그대로 이 파이프라인을 얹어, "실패 직전에만 활성화되는" monosemantic feature를 unsupervised로 자동 탐색할 수 있음. §4의 activation-patching 실험 설계(counterfactual pair 간 feature 값 patch → KL/행동 변화 측정)는 우리 succ/fail activation pair에 그대로 대응(성공 rollout activation을 target으로, 실패 rollout에 patch).
- 우리 confound(길이·instruction skew)와 연결: 이 논문의 feature-search 방식(어떤 토큰/문맥에서 활성화되는지 histogram으로 확인, §5.1)을 그대로 적용하면 "feature가 특정 phase/timestep에서만 뜨는지" 혹은 "rollout 길이와 상관돼있을 뿐인지"를 사후 점검할 수 있음 — phase-matched steering 후보 feature를 이 SAE 파이프라인으로 스크리닝 가능.
- 다만 이 논문은 순수 offline(대량 corpus, unsupervised)에서 dictionary를 학습하는 것으로, 우리의 핵심 난제인 "online phase/failure-type 식별" 자체에 답은 아님 — 활용한다면 2단계 구조(오프라인에서 dictionary 미리 학습 → 추론 시 해당 feature의 activation을 monitor/steer)가 되고, dictionary가 우리 rollout 데이터량(LLM 대비 훨씬 적음)으로도 충분히 sparse-overcomplete하게 학습되는지 별도 검증이 필요.

## 면접 포인트(Q→A)

Q1. SAE가 단순 linear probing보다 나은 점은?
A. probing은 supervised로 특정 concept 하나에 대해 방향 하나를 학습하는데, 그 방향이 superposition 때문에 다른 feature와 얽혀 있을 수 있음(=concept이 여전히 다차원적으로 섞인 activation 위에서 찾아짐). SAE는 라벨 없이 대량 overcomplete dictionary를 동시에 학습하고 L1 sparsity로 개별 feature를 monosemantic하게 분리하므로, 사전에 라벨링하지 않은 concept도 발견할 수 있고 서로 얽힌 표현을 원자적 방향들로 분해한다.

Q2. tied weight를 쓰는 이유와 위험은?
A. 인코더(검출) 방향과 디코더(정의) 방향이 같다는 기대, 파라미터 절반, encoder/decoder 방향 모호성 제거가 장점. 하지만 residual stream에서만 무손실이고 MLP data에서는 성능이 떨어져 논문도 MLP에는 encoder/decoder를 분리(M_e≠M_d)해 사용했다 — "검출 방향=정의 방향" 가정이 항상 성립하진 않는다는 방증.

Q3. reconstruction이 완전하지 않다(perplexity 25→40)는 게 왜 중요한가?
A. dictionary가 activation의 정보를 다 못 담는다는 뜻 = 일부 feature를 놓쳤거나 basis가 불완전. steering 관점에서 이는 "feature 몇 개만으로 activation 전체를 대체하면 모델 성능이 실제로 떨어진다"는 뜻이라, causal feature가 아직 dictionary 밖에도 있다는 신호. 이후 연구(TopK SAE, JumpReLU, GemmaScope)가 이 reconstruction-sparsity trade-off 개선에 집중한 이유.

Q4. IOI activation patching 결과(PCA 대비)가 의미하는 바는?
A. 같은 KL divergence 도달에 SAE feature는 PCA보다 더 적은 개수·더 작은 edit magnitude로 충분 = SAE가 실제 causal mechanism을 PCA보다 더 국소적(monosemantic)으로 포착했다는 증거. 이는 steering 개입 시 "적은 feature만 건드려도 원하는 인과효과"를 낼 수 있다는 뜻이라, steering vector 후보로서 SAE feature의 우수성을 뒷받침한다. 단, α=0(non-sparse) dictionary에서는 이 이점이 사라지므로 sparsity 자체가 이 효과의 근원임을 확인했다.

Q5. 우리 VLA latent steering 프로젝트에 적용한다면 어떤 한계에 부딪히나?
A. (1) 이 논문은 대량 unlabeled text corpus로 offline 학습 — VLA rollout 데이터는 훨씬 적고 activation 분포가 task/phase마다 강하게 non-stationary해서 dictionary가 충분히 학습될지 불확실. (2) online 실시간 monitoring을 다루지 않음 — 우리 핵심 난제(online phase/failure-type 식별)에 직접 답이 아니라 "사전에 dictionary를 학습해두고 추론 시 activation을 그 위에 투영해 monitor"하는 2단계 우회가 필요. (3) MLP-유사 nonlinearity 구간에서 dead feature 문제가 있었는데 DiT block residual도 유사 위험. (4) feature가 특정 phase/token에서 활성된다는 관찰이 causal한지 vs 그냥 길이 confound와 얽힌 건지는 별도 activation-patching 검증이 필요(우리 프로젝트의 상시 confound와 동일한 함정).

## 한계·비판

- reconstruction 불완전 → 정보 손실 존재(perplexity 상승으로 정량 확인), dictionary가 activation의 전부를 설명 못함.
- 레이어가 깊어질수록 baseline(ICA) 대비 해석성 개선폭이 줄어듦 — SAE 자체의 한계인지 자동해석(GPT 기반) 방법의 한계인지 논문도 명확히 구분 못함.
- MLP sublayer 적용 시 dead feature 다량 발생(특히 middle/late layer) → 진짜 overcomplete basis 학습 실패, tied weight를 풀어도 완전히 해결 안 됨.
- automatic interpretability(Bills et al.) 자체의 한계를 상속: next/previous-token 패턴처럼 현재 토큰 중심이 아닌 패턴은 GPT가 잘 못 찾고, 이 프로토콜은 출력 변화를 보고 가설을 검증하지 않음(논문은 별도로 ablation 실험을 했지만 automatic interp 루프에는 통합 안 함).
- sparsity hyperparameter α의 sweep이 smooth trade-off만 보여줌(knee 없음) → "몇 개의 feature가 정답인가"에 대한 원칙적 기준이 없고, 사실상 hyperparameter 선택이 자의적.
- outlier dimension(Adam optimizer/residual basis 특이 방향) 문제가 feature search에서 자주 최상위로 잡힘 — 진짜 semantic feature 탐색을 방해.
- circuit detection(§5.3)은 소규모 case study 수준의 ablation heuristic이며, scale된 정량 검증(다수 feature/다수 태스크)은 없음.
- 이 논문은 steering을 직접 실행하지 않음(ablation/patching까지만) — 실사용 steering 적용은 후속 연구(Golden Gate Claude, feature-steering, AxBench 등)의 몫으로 남김.
