# Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 (Lieberum 2024)

- 출처: Google DeepMind. arXiv preprint(report 형식, 비게재 저널/컨퍼런스).
- arXiv: 2408.05147 (v2, 2024-08-19)
- PDF: `docs/Activation_steering_basic/GemmaScope_2408.05147.pdf`
- 정독 섹션: §5(산업/생태계 관점, 집중) + §1·§2·§3·§4 훑음
- tier: must
- 한줄역할: JumpReLU SAE를 Gemma 2 2B/9B(+27B 일부)의 **모든 층·모든 sublayer**에 학습해 400개+ SAE·2,000개+ weight를 CC-BY-4.0으로 공개한 오픈소스 SAE 인프라 이벤트 — "SAE 학습은 산업 랩만 감당 가능"하던 진입장벽을 커뮤니티에 개방.

## 문제·동기

- SAE가 causally relevant/interpretable directions를 찾는 유망한 방법으로 부상(Bricken/Cunningham/Gao/Templeton 등)했지만, 학습 비용이 매우 커서 industry 밖 커뮤니티의 야심찬 interpretability 연구를 제약. steering vector나 probing과 달리 SAE는 학습 자체가 expensive·difficult.
- 기존 modern-model SAE 연구는 대개 단일 층 residual stream 하나에만 국한(Engels/Gao/Templeton), 상당수가 비공개(proprietary) 모델(OpenAI GPT-4, Anthropic Claude)에서 학습돼 커뮤니티가 재사용·재현 불가능.
- 목표: comprehensive(전 층·전 sublayer) + open(가중치 공개, 널리 쓰이는 오픈 모델) SAE suite를 제공해 circuit analysis 같은 야심찬 응용(Marks et al. 2024를 더 큰 모델로 scale-up)을 가능케 함.

## 핵심 아이디어

- "Everywhere all at once": 하나의 층·하나의 site가 아니라 Gemma 2 2B/9B **전 층**(2B=26층, 9B=42층) × **3개 site**(attention head output pre-linear, MLP output, post-MLP residual stream) 전부에 SAE를 학습해 공개.
- JumpReLU 활성함수 채택(Rajamanoharan et al. 2024b, 동일 그룹의 원 아키텍처 논문) — L0 sparsity를 straight-through estimator로 직접 최적화해 TopK/Gated 대비 소폭 Pareto 우위 + 토큰별 활성 latent 수가 가변적(TopK는 고정 k).
- 폭(width)·sparsity를 스윕한 다중 SAE 세트를 함께 공개(feature-splitting suite 포함)해 "어떤 SAE를 쓸지"를 커뮤니티가 직접 비교할 수 있게 함 — 단일 체크포인트가 아니라 연구 인프라 자체를 공개하는 것이 핵심.

## 방법(Gemma 2 전 층 JumpReLU SAE 스위트 공개, 학습 규모)

- JumpReLU: f(x) = JumpReLU_θ(Wenc·x + benc) = z ⊙ H(z − θ) (θ: latent별 학습 임계값), decoder x̂ = Wdec·f + bdec(선형). Loss = ‖x − x̂‖² + λ‖f(x)‖₀ — L1 근사가 아니라 L0을 straight-through estimator(kernel density bandwidth ε=0.001)로 직접 규제.
- 학습 데이터: Gemma 1 사전학습 분포와 동일 텍스트, BOS/EOS/padding 토큰 제외, activation을 shard로 셔플 후 학습. 활성값은 unit mean-squared-norm으로 정규화(레이어 간 하이퍼파라미터 이전성 확보).
- 스케일: 폭 2^14(16.4K)~2^20(~1M) latent, 토큰 수는 폭에 따라 4B(16.4K)/16B(1M)/그 외 8B. 메인 릴리스 SAE 400개+, 학습된 latent 총합 3천만+(중복 가능), site·layer·sparsity 조합까지 합치면 공개 가중치 2,000개+. GPT-3 학습 compute의 20%+ 사용, 활성값 디스크 저장량 ~20 PiB, SAE 파라미터 총합 수백억.
- 부가 릴리스: 동일 site에 폭만 다르게 학습한 "feature-splitting" suite(2^14~2^19, 레이어 고정), MLP sublayer transcoder 1 suite(Appendix B), Gemma 2 9B **instruction-tuned** 모델 활성값으로 학습한 SAE(§4.5) — PT 모델에서 학습한 SAE가 IT rollout도 잘 재구성됨을 확인.
- 인프라: TPUv3 4x2(대부분)/TPUv5p, Megatron sharding, 공유 disk-read 서버(다중 SAE가 같은 site/layer 활성값을 amortize), fp32 저장 후 bf16 추론은 성능 저하 거의 없음(§4.7, 다중 SAE를 동시에 splice하는 circuit analysis에 유용).

## 실험·결과(feature 수·평가)

- 평가 지표: delta LM loss(SAE reconstruction을 forward pass에 splice했을 때 cross-entropy 증가, 주지표) + FVU(fraction of variance unexplained, 보조).
- residual stream SAE의 delta loss가 MLP/attention SAE보다 항상 높음(residual stream이 병목이라 작은 재구성 오차도 이후 전체 층에 전파되기 때문) — FVU는 site 간 비슷.
- 시퀀스 위치 효과: 토큰 위치가 뒤로 갈수록 재구성 loss 증가(attention/residual SAE는 단조 증가 후 plateau, MLP SAE는 10번째 토큰 근방에서 peak 후 소폭 감소).
- width 효과: 넓은 SAE일수록 같은 sparsity에서 재구성 fidelity 우수하지만 "feature splitting"(좁은 SAE의 상위 개념 latent가 넓은 SAE에서 여러 세분화 latent로 쪼개짐) 발생 — 유익한지는 불명확.
- interpretability(§4.4, 동반 논문 Rajamanoharan 2024b 재인용): human rater 평가 + LM 기반 설명 시뮬레이션 상관 모두에서 JumpReLU/TopK/Gated 세 SAE 아키텍처 간 뚜렷한 차이 없음.
- IT 전이(§4.5): PT 모델에서 학습한 SAE가 IT 모델 rollout activation을 스플라이스해도 IT 전용 SAE와 거의 동등한 loss 증가만 유발 — "베이스 모델에서 학습한 SAE를 파인튜닝 모델에 재사용 가능"이라는 실용적 결과.
- Pile 서브셋 일반화(§4.6): DeepMind mathematics(포뮬러틱)에서 최선, Europarl(다국어)에서 최악 — 학습 코퍼스(영어 중심)와의 분포 불일치가 원인으로 추정.

## activation-steering 흐름 위치

- 이 논문 자체는 steering을 수행하지 않음(SAE 학습·평가·공개까지). 계보상 Cunningham/Bricken(발견) → Gao TopK/Rajamanoharan JumpReLU(스케일·아키텍처 표준화) 다음 단계로, "이 표준 레시피를 **오픈 모델·전 층으로 대량 공개**"하는 **인프라/생태계 확산** 이벤트.
- §5 "Using SAEs to improve performance on real-world tasks" 항목에서 SAE-feature steering을 classic steering vector(ActAdd/ITI 계열)와 명시적으로 나란히 놓음: "steering vectors(Turner et al.)를 SAE feature steering(Conmy & Nanda, Activation steering with SAEs)이나 clamping(Templeton et al., Golden Gate Claude)과 비교" / "SAE로 steering vector에서 irrelevant feature를 제거해 개선할 수 있는가" — 이 논문이 SAE 기반 steering을 supervised steering-vector 계열의 대안·보완으로 공식 포지셔닝한 지점.
- HuggingFace(CC-BY-4.0) 가중치 공개 + Neuronpedia 인터랙티브 데모(feature dashboard, autointerp 열람)를 함께 제공 — AxBench류 후속 steering-vector-vs-SAE 비교 연구가 SAE를 직접 재학습하지 않고 Gemma Scope 가중치를 그대로 가져다 쓸 수 있게 한 실질적 인프라(개별 연구실의 GPU-day 절감).

## 우리 프로젝트 연결

- 우리가 찾는 "concept 해석 + 자동 실패데이터" 인프라의 오픈소스 선례: Gemma Scope는 "모델 전 층에 SAE를 미리 학습해 공개 → 누구나 그 위에서 feature 탐색·steering·red-teaming"이라는 2단계 워크플로우를 업계 표준으로 만든 사례. NOTALL이 GR00T DiT에 붙인 per-token SAE도 이 계보(TopK/JumpReLU) 위에 있을 가능성이 높음.
- 직접연결은 약함: (1) Gemma Scope는 텍스트 LLM residual stream 전용이라 VLA(Eagle-VLM+DiT)의 이질적 activation 구조에 그대로 못 옮김. (2) 우리는 백본 하나(GR00T-N1.5/1.6)에 대해 자체 activation을 수집·conceptor를 fit하는 중이라 "공개된 SAE 가중치를 그대로 재사용"하는 옵션 자체가 없음(공개 VLA SAE 부재) — 우리가 만들면 이 생태계의 VLA판 선례가 되는 셈.
- 방법론적으로 재사용 가능한 부분: (a) "전 층·전 site 스윕 후 sparsity-fidelity trade-off로 SAE/layer 선택" 절차는 우리가 VL-SA/DiT block 중 어디에 conceptor를 fit할지 고를 때(현재는 근거 기반 layer 선택, NOTALL t≤8/t≥12) 참고할 수 있는 체계적 대안. (b) §4.5 PT→IT 전이 결과는 "베이스 GR00T에서 학습한 conceptor/SAE가 파인튜닝된 체크포인트에도 전이되는가"라는 유사 질문을 던짐(우리는 아직 검증 안 함).
- §5 open-problem 리스트(steering-vector vs SAE-feature-steering 비교, irrelevant feature 제거로 steering 개선)는 우리 conceptor 방식(C_steer = C_success ∧ ¬C_failure)과 같은 문제의식 — "supervised pair-diff steering vector"(우리 conceptor의 저차원 버전에 해당) 대비 "unsupervised SAE로 찾은 feature 조합"이 더 정밀한 steering을 줄 수 있는가라는 질문은, pathway-분리(VL/DiT) 축과 별개로 향후 conceptor 대신·보완으로 SAE-latent를 쓰는 옵션을 시사.

## 면접 포인트(Q→A)

Q1. Gemma Scope가 "논문"이라기보다 "인프라 이벤트"로 불리는 이유는? 왜 오픈 SAE가 중요한가?
A. 새 알고리즘을 제안하지 않고(JumpReLU 자체는 Rajamanoharan et al. 2024b가 별도 제안), 기존 표준 SAE 레시피를 Gemma 2 전 층·전 sublayer에 학습해 400개+ SAE·2,000개+ 가중치를 CC-BY-4.0으로 공개한 것이 핵심 기여. 이전에는 GPT-4/Claude급 SAE가 OpenAI/Anthropic 내부에만 있어 커뮤니티가 circuit analysis 등 야심찬 응용을 시도할 토대 자체가 없었는데, 이 공개로 오픈 모델(Gemma 2) 기준 SAE 연구 진입장벽이 사실상 사라졌다 — 개별 연구실이 GPT-3 compute의 20%+·활성값 20PiB 규모를 매번 재현할 필요 없이 즉시 feature 탐색·steering 실험을 시작할 수 있다.

Q2. JumpReLU를 TopK/Gated 대신 채택한 이유는?
A. TopK는 모든 토큰에 정확히 k개 latent를 강제해 토큰별로 필요한 sparsity가 다를 수 있는 현실을 못 담는 반면, JumpReLU는 latent별 학습 가능한 threshold θ로 L0을 straight-through estimator로 직접 규제해 토큰마다 활성 latent 수가 가변적이다. 저자들(Rajamanoharan et al. 2024b)이 이를 TopK/Gated 대비 소폭 Pareto 우위로 보였고, Gemma Scope는 이를 메인 아키텍처로 채택하되 §4.4에서 세 아키텍처 간 실제 해석성 차이는 크지 않음도 함께 보고한다.

Q3. residual stream SAE의 delta loss가 MLP/attention SAE보다 항상 높은 이유는?
A. residual stream은 모든 이전 층이 이후 층과 통신하는 유일한 병목(bottleneck)이라 작은 재구성 오차도 이후 전체 forward pass에 누적 전파되는 반면, 개별 MLP/attention sublayer의 출력은 residual 전체에서 작은 기여분이라 같은 절대 오차라도 residual 관점에서는 상대적으로 작다.

Q4. 이 논문이 activation-steering 계보에서 갖는 위치는?
A. 스스로 steering을 수행하지 않지만 §5에서 "steering vector vs SAE feature steering/clamping" 비교, "SAE로 steering vector의 irrelevant feature 제거"를 공식 오픈 문제로 지목해, classic supervised steering-vector 연구(ActAdd/CAA/ITI)와 SAE 기반 unsupervised feature steering(Golden Gate Claude, AxBench)을 같은 문제의식의 두 갈래로 명시적으로 연결한다. 계보: Cunningham/Bricken(발견) → Gao/Rajamanoharan(스케일·아키텍처) → Gemma Scope(오픈 인프라 확산) → AxBench 등(스티어링 벤치마크 비교) — 이 논문은 "누구나 재현 없이 바로 스티어링 실험을 시작할 수 있게" 만든 확산 지점이다.

Q5. 우리 VLA latent steering 프로젝트에 이 사례를 적용한다면 어떤 한계가 있나?
A. (1) 텍스트 LLM residual stream 전용 레시피라 VLA의 이질적 activation(Eagle-VLM/VL-SA/DiT)에 그대로 옮기기 어렵고 공개된 VLA SAE 자체가 없어 "재사용"이 불가능 — 직접 만들어야 한다. (2) comprehensive-suite 학습 규모(GPT-3 compute 20%+, 활성값 20PiB)는 우리 rollout 데이터·컴퓨트 예산과 자릿수가 다르므로 축소 버전(특정 layer만, phase-conditioned)으로 설계해야 한다. (3) 어떤 SAE도 online phase/failure-type 식별이라는 우리 핵심 난제에 답하지 않는다 — offline dictionary 학습 후 추론 시 투영/모니터링하는 2단계 구조가 필요한 건 이전 SAE 논문들과 동일한 한계.

## 한계·비판

- SAE 품질을 재는 "합의된 지표"가 아직 없음을 저자도 인정(§4 서두) — delta loss/FVU 모두 근사적 대리지표.
- feature splitting 현상(넓은 SAE가 latent를 더 세분화)이 정말 유익한 "더 많은 진짜 feature 발견"인지, sparsity penalty가 유도한 인위적 조합(Anders et al. 2024, toy model 관찰)인지 미해결 — §5에서도 open problem으로 남김.
- ultra-high-frequency latent cluster가 폭을 넓혀도 사라지지 않음(§4.3) — 진짜 atomic feature가 아니라 outlier·구조적 아티팩트일 가능성.
- 평가 대부분이 Gemma 1 사전학습 분포(주로 영어) 내부에서만 수행 — Pile 서브셋 비교(§4.6)에서도 다국어(Europarl) 성능 저하가 이미 관찰됨, 분포 밖 일반화는 별도 검증 없음.
- interpretability 평가(§4.4)는 이 논문 자체가 아니라 동반 논문(Rajamanoharan 2024b)의 human-rater/LM-simulated 결과를 재인용한 것 — Gemma Scope 릴리스 전체(400+ SAE)에 대한 자체 interpretability 검증은 아님.
- steering 응용은 §5 "future work list"로만 제시 — 이 논문 자체는 causal steering 실험·결과를 전혀 포함하지 않는다(공개=인프라 제공, steering 검증은 후속 연구 몫).
- 대형 SAE(1M-width)는 소수 layer에만 학습돼(Table 1) 전 층 커버리지가 폭 전체에 균일하지 않음 — "everywhere all at once"라는 제목과 달리 큰 폭에서는 여전히 선택적.
