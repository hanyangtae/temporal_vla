# AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders (Wu et al. 2025, ICLR)

- 출처: arXiv:2501.17148 (v3, 2025-03-03) · Zhengxuan Wu, Aryaman Arora(공동1저자), Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, Christopher Potts (Stanford NLP)
- PDF: docs/Activation_steering_basic/AxBench_2501.17148.pdf
- 정독 섹션: §3 AxBench 방법(비판적 벤치마크 구성) 중심, §1·2·4~7 개관 확인
- tier: must
- 한줄 역할: steering/SAE 방법 10여 개를 prompting·finetuning과 통일 프로토콜로 직접 대결시킨 대규모 벤치마크 — "diff-in-means 같은 단순 baseline이 SAE steering을 이긴다"는 비판적 결과로, 복잡한 steering method가 반드시 넘어야 할 최소 기준선을 제시.

## 문제·동기

SAE, LAT, supervised steering vector, linear probe, ReFT 등 representation-level steering 방법이 난립했지만 통일된 대규모 벤치마크가 없었다(기존 벤치마크는 toy-scale·소수 방법 비교에 그침, Pres et al. 2024/Braun et al. 2024 지적). SAE steering이 open-vocabulary·long-form generation 같은 현실적 세팅에서 prompting/finetuning 대비 실제로 유효한지 검증되지 않은 상태. AxBench는 자연어 concept 목록을 입력받아 LLM으로 synthetic train/eval 데이터를 생성하고, concept detection(C)과 model steering(S) 두 축에서 방법들을 동일 모델·동일 evaluator로 비교한다.

## 핵심 아이디어

concept마다 LLM(gpt-4o-mini)으로 positive(개념 포함 응답)/negative(개념 없는 응답) 쌍을 합성 생성(Dtrain n=144, concept당) + 다의어 기반 hard negative가 포함된 별도 평가셋(Dconcept). 두 축 평가: (1) concept detection — 토큰 표현에서 개념 존재를 분류(AUROC/F1, max-pool로 시퀀스 스코어화). (2) model steering — Alpaca-Eval instruction에 개입 적용 후 LLM judge가 concept/instruct/fluency 3점(0~2) 척도로 채점, harmonic mean으로 종합. Gemma-2-2B/9B instruction-tuned, GemmaScope SAE가 존재하는 4개 site(2B: L10/L20, 9B: L20/L31) × 500 concepts(CONCEPT500). 신규 방법 ReFT-r1도 제안: probe형 supervised detection과 supervised steering objective를 결합한 rank-1 representation finetuning.

## 방법(AxBench 구성)

- SDL(supervised dictionary learning) 계열 — 전부 라벨된 소량 데이터로 rank-1 방향을 학습: DiffMean(양/음 평균차, Marks & Tegmark 2024), PCA(양성 집합 1주성분), LAT(pairwise diff PCA, Zou et al. 2023), Probe(BCE logistic), SSV(supervised steering vector, LM loss로 벡터 직접 최적화), ReFT-r1(신규 — detection+steering 결합, TopK 게이팅 + L1 정규화).
- SAE 계열: 사전학습 GemmaScope SAE(완전 비지도) + SAE-A(같은 학습셋으로 AUROC 최고 latent를 사후 선택해 fair comparison 시도).
- 비-representation baseline: BoW+logistic(detection 전용), 프롬프트(detection은 LLM judge rating, steering은 LLM이 생성한 프롬프트를 prepend), gradient 기반(I×G, IG — detection 전용), finetuning(full SFT, LoRA, LoReFT — steering 전용).
- steering 개입은 대부분 activation addition h_i + α·w(α=steering factor, concept당 hold-out split으로 최적값 선택) — SSV만 학습된 벡터를 그대로 더함, ReFT-r1만 TopK 게이팅된 additive.
- detection 스코어는 방법 대부분 dot product(SAE는 encoder+sigmoid, ReFT-r1/SSV는 ReLU)를 시퀀스 내 max-pool.

## 실험·결과

- Concept detection(AUROC, Table1, 평균): DiffMean 0.942, Probe 0.940, ReFT-r1 0.938 — 통계적으로 유의차 없는 최상위권. Prompt 0.929, SAE-A 0.917, BoW 0.914, SSV 0.912가 근소하게 뒤짐. **SAE(vanilla) 0.695, PCA 0.652, IG/IxG 0.4대**로 5개 지도 방법에 유의하게 뒤짐. class-imbalance(F1, positive ~1%) 세팅에서도 순위는 유지되나 SAE·LAT·PCA는 더 크게 저하.
- Model steering(overall score, Table2, 평균): **Prompt 0.894로 1위**, LoReFT 0.741, SFT 0.676, LoRA 0.615, ReFT-r1 0.543(2B에선 prompt급, 9B에선 크게 뒤짐)이 뒤를 잇는다 — representation steering 중 유일하게 finetuning급 성능. 그 아래로 DiffMean 0.239, **SAE 0.165, SAE-A 0.157**(AUROC 선택이 오히려 vanilla SAE보다 소폭 낮음 — "detection이 좋다고 steering이 좋아지지 않음"), LAT 0.127, PCA 0.105, Probe 0.098, SSV 0.026(거의 무효).
- Winrate vs SAE(Table3): ReFT-r1 88.0%, DiffMean 61.6%(특히 이른 레이어에서 高)만 50% 초과 — 다른 representation 방법은 SAE 이하.
- steering factor를 키우면 전 방법에서 instruct score(능력)가 단조 감소하지만, concept score는 이른 layer에서 증가 후 감소(역U), 늦은 layer에서는 대체로 단조 증가 — ReFT-r1만 全구간에서 Pareto-optimal 경로를 그림.

## activation-steering 흐름 위치

ActAdd/CAA/RepE/ITI(mean-diff 단일벡터 additive) → Conceptors(Boolean 대수·soft projection) → SAE 계열(Templeton, TopK/JumpReLU SAE 등, 비지도 dictionary learning)로 이어지는 지도(SDL) vs 비지도(SAE) 두 갈래를 이 논문이 처음으로 같은 잣대(동일 모델·동일 evaluator·동일 steering protocol) 위에 올려 직접 대결시켰다. 결과는 "SAE steering이 가장 단순한 지도 baseline(DiffMean)보다도 못하다"는 것 — SAE 계열 논문이 reconstruction/interpretability 지표로 자축한 것과 달리, 행동 개입(steering) 효용이라는 잣대에서는 SAE가 뒤처진다는 회의론을 정량적으로 못박았다. 이후 문헌에서 "AxBench 결과"는 SAE steering 비관론의 표준 인용점이 됐다. ReFT-r1(SDL+steering 공동학습)은 "representation steering이 아직 죽지 않았다"는 반론 축.

## 우리 프로젝트 연결

- 우리 conceptor(C_steer = C_success ∧ ¬C_failure)는 AxBench 분류상 SDL(지도, 라벨=succ/fail rollout)에 해당한다. AxBench 결과가 시사하는 바 — "지도 신호가 있으면 unsupervised(SAE)보다 단순 지도 방법조차 이긴다" — 는 succ/fail 라벨이 존재하는 우리 세팅에서 SAE류 대신 SDL류(conceptor)를 택한 선택을 방법론적으로 뒷받침한다.
- 더 뼈아픈 시사점은 "DiffMean(가장 단순한 rank-1 mean-diff, 우리 conceptor가 rank-1로 퇴화한 특수 케이스)이 이미 다른 정교한 SDL(Probe, LAT, PCA, SSV)을 대부분 이긴다"는 것 — 우리가 conceptor(다차원 subspace·Boolean AND/NOT)의 복잡도를 정당화하려면 최소한 이 diff-in-means baseline을 유의하게 이겨야 한다. 이것이 사다리식 ablation(전역 mean-diff → pathway 분리 → phase-bin 조건부, 이전 단계가 신호를 보일 때만 다음 복잡도를 추가)에서 **첫 단이 반드시 mean-diff여야 하는 근거**.
- detection 최고 방향(SAE-A)이 steering에서는 오히려 vanilla SAE보다 낮다는 결과는, 우리가 succ/fail 분리도(representation 분석)를 잘 보인다고 해서 그 방향으로 steer했을 때 SR이 오른다는 보장이 없음을 시사 — 분리도 검증과 별개로 causal steering(ΔSR) 실험이 항상 필요한 이유를 뒷받침.
- steering factor에 따른 instruct-vs-concept trade-off가 늦은 layer일수록 단조적(Fig4)이라는 관찰은, 늦은 layer/모터 성격을 가진 DiT pathway가 이른 layer/목표 성격의 VL pathway보다 강도 스윕에 더 예측 가능하게 반응할 수 있다는 방향성 가설과 연결되나 우리 도메인에서 별도 검증 필요.

## 면접 포인트

Q1. AxBench의 핵심 결론은 무엇이고 SAE가 왜 실망스러운가.
A. concept detection·model steering 두 축 모두에서 SAE는 최하위권이다. detection AUROC는 DiffMean 0.942 대 SAE 0.695, steering overall score는 Prompt 0.894·DiffMean 0.239 대 SAE 0.165다. SAE는 라벨 없는 비지도 사전학습으로 개념 사전을 만들지만 feature-to-natural-language 라벨링(auto-interp)이 얕고 토큰 수준 개념에 편중돼 있어(저자 자인) 고차원 개념 조작에 부적합하다. 반면 144개 라벨된 synthetic 데이터로 학습한 가장 단순한 지도 방법(평균차)조차 SAE를 크게 이긴다 — "self-supervised 스케일"이 "소량이라도 정확한 라벨"을 이기지 못한다는 결과.

Q2. detection 성능과 steering 성능이 항상 같이 가는가.
A. 아니다. SAE-A(학습셋으로 AUROC 최고 latent를 사후 선택)는 detection에서 vanilla SAE보다 낫지만(0.917 대 0.695) steering에서는 오히려 vanilla SAE보다 살짝 낮다(0.157 대 0.165). 분류를 잘하는 방향과 개입해서 원하는 행동을 유도하는 방향이 다를 수 있다는 뜻이다. 우리 프로젝트에서도 succ/fail을 잘 분리하는 conceptor 방향이 곧 steer해서 SR을 올리는 방향이라는 보장은 없고, 별도 causal steering 실험(ΔSR)으로 검증해야 한다.

Q3. representation steering이 왜 prompting·finetuning을 못 이기나.
A. 정확한 원인 규명까진 아니지만 두 단서를 준다. (1) steering factor를 키우면 모든 representation 방법에서 instruct score(능력)가 단조 감소한다 — 개념 주입과 능력 보존이 근본적 trade-off다. (2) 대부분 SDL이 쓰는 단일 rank-1 벡터로는 concept-conditioned 다양한 문맥에 걸친 개입을 표현하기 부족하다. 저자들이 제안한 ReFT-r1(detection+steering 공동학습, TopK 게이팅)이 격차를 일부 좁힌 것은 "표현력 부족"이 원인 중 하나임을 시사한다.

Q4. 이 논문이 우리 프로젝트에 주는 방법론적 교훈은.
A. 복잡한 steering method(conceptor, pathway 분리, phase-matched)를 제안하려면 반드시 diff-in-means류 단순 baseline과 같은 잣대(같은 데이터·같은 평가 프로토콜)로 비교해야 하고, 이기지 못하면 복잡도를 정당화할 수 없다. 이것이 우리 실험의 사다리식 ablation(전역 mean-diff → pathway 분리 → phase-bin 조건부, 이전 단계가 신호를 보일 때만 다음 복잡도 추가) 설계의 직접적 근거다.

## 한계·비판

- LLM judge 기반 steering 평가는 rater bias·prompt wording sensitivity에 취약 — 다른 steering 논문(CAA 등)도 공통 지적하는 한계이며, rater 간 신뢰도 검증은 보고되지 않음.
- concept list가 GemmaScope Neuronpedia auto-interp에서 나와 token-level/shallow 개념에 편중(저자 자인) — SAE 성능 저하가 SAE 방법 자체의 한계인지 concept 라벨 품질의 한계인지 완전히 분리되지 않는다.
- steering factor를 hold-out split에서 선택하는 절차는 방법마다 개별 hyperparameter search이며, SAE는 activation addition만 주 결과로 보고(clamping은 부록 ablation)돼 SAE에 최적화된 개입 방식이 아니었을 가능성이 있다.
- 두 모델(Gemma-2-2B/9B)·GemmaScope SAE 한 종류에 국한 — 다른 아키텍처(Llama, Mistral)나 다른 SAE 레시피(TopK, BatchTopK, JumpReLU 변형)로 일반화되는지 미검증.
- SFT는 리소스 제약으로 500개 중 첫 20개 concept만 학습·평가(9B는 SFT 결과 자체 없음) — finetuning 비교의 통계적 근거가 다른 방법보다 약하다.
- 텍스트/LM 단일 forward, next-token 생성 전용 벤치마크 — 연속 제어(action) 도메인, multi-step rollout, "성공/실패"가 이산 라벨이 아니라 궤적 전체의 결과인 우리 세팅으로의 전이는 검증되지 않음(직접 확인해야 할 도메인 격차).
