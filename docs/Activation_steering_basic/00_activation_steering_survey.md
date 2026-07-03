# Activation Steering 종합 서베이 — 하나의 흐름으로

> 목적: (1) 본인 연구(pathway-resolved + phase-matched conceptor steering)를 선행지형 안에
> 위치시키고, (2) **기술 면접 대비** — 각 개념을 정의→대표 방법→동기→tradeoff→우리 연결로
> 설명할 수 있게. 개별 논문 정독 노트는 [`notes/`](notes/) 에 52편. PDF는 기초=이 폴더,
> VLA=[`../references/`](../references/). Notion 미러: "Activation steer 전반 공부".
>
> 읽는 법: §1→§7 순서가 하나의 흐름(개념=방향 → 읽기 → 쓰기 → 무엇을 제어 → 산업 → VLA →
> 로봇 배포). 각 섹션은 **정의 → 대표 논문 흐름 → 축/tradeoff → 우리 프로젝트 연결 → 면접 포인트**.
> 수식은 달러기호 없이 plain/unicode.

# 한 장 요약 (핵심 메시지)

- **Activation steering = 모델 내부 표현(활성화)을 추론 시 읽고(analysis) 써넣어(steering),
  재학습 없이 행동을 바꾸는 기법.** 토대는 **linear representation hypothesis**(개념이 활성화
  공간의 "방향"으로 선형 인코딩된다).
- 방법은 두 단계다. **읽기(read-out)**: probing·lens·SAE·activation patching으로 개념 방향을
  찾고 인과를 검증. **쓰기(write-in)**: 그 방향을 **더하거나(additive)** **투영/억제(projective)**
  하거나 **학습된 개입(ReFT)** 으로 조종.
- 발전 축: **additive(ActAdd→CAA→ITI) → task/function vector → projective/subspace(conceptor,
  directional ablation) → 학습형(ReFT)**. 그리고 **신뢰성 비판(Tan, AxBench): steering은
  입력별로 요동치고, diff-in-means 같은 단순 baseline이 SAE보다 강하며, 종종 prompting에도 진다.**
- LLM/VLM에서는 **거부·정직·아첨·persona·환각**을 방향으로 제어하고(Arditi refusal, ITI,
  Persona Vectors, VTI). **산업 실서비스엔 activation *steering(쓰기)* 이 배포된 근거가 사실상
  없다** — 읽기(활성화 probe)만 Anthropic이 신중히 진입, 쓰기는 데모(Golden Gate)·연구·소수 B2B
  API(Goodfire)·오픈소스 도구(이미지 SEGA/SLD)에 국한. 못 쓰는 이유는 §3 brittleness(Tan/AxBench)·
  안전 역설(Rogue Scalpel).
- **VLA/로봇**으로 오면 개입 지점이 다변화한다(hidden state / denoising 샘플링 / subgoal /
  명령 추상화). 우리 프로젝트는 **hidden state에 다차원 contrastive conceptor(C_success ∧
  ¬C_failure)를 pathway(VL=goal/DiT=motor)별·phase별로 적용**하는 자리에 있고, ==미해결 난제는
  online에서 어느 pathway·어느 phase가 실패인지 식별==하는 것이다.
- 산업 배포로 잇는 갈래는 **online 실패검출(Sentinel, FIPER)·복구/introspection(KnowNo, VITA)·
  안전 gate(latent safety filter)**. 우리 phase-matched steering은 "요청/재계획/우회" 없이
  latent 안에서 즉시 복구하는 저비용 대안으로 정당화된다.
- **Fine-tuning과의 관계**: 대체재가 아니라 다른 축(§5.5) — ==소량 데이터·상태-조건부·가역 개입은
  steering, 대량 데이터의 정적 행동 이동은 FT==, 성숙 경로는 결합·증류(LoRRA/Circuit Breakers/ReFT).
  steering은 모델이 **이미 가진 표상의 mixture를 바꿀 뿐 없는 능력은 추가 못한다**는 게 본질 차이.

# 핵심 용어 (30초 정의)

- **활성화(activation) / hidden state / residual stream**: 트랜스포머 각 층이 내보내는 벡터. steering의 개입 대상.
- **linear representation hypothesis**: 개념이 활성화의 특정 "방향"(단위벡터)으로 선형 인코딩된다는 가설.
- **probing**: 활성화에 선형 분류기를 얹어 "그 개념이 표현돼 있나"를 측정(상관적).
- **activation patching / causal tracing**: 특정 위치 활성화를 바꿔치기해 출력에 대한 인과 기여를 측정.
- **SAE(sparse autoencoder)**: 활성화를 희소한 해석가능 feature 사전으로 분해(superposition 해소).
- **steering vector**: 더하면 특정 행동이 강해지는 방향. 보통 대조쌍 활성화 차(diff-of-means).
- **additive vs projective**: h' = h + αv(더하기) vs h' = h·C(방향으로 스케일/투영).
- **conceptor**: 활성화 집합의 공분산을 담은 soft projection 행렬 C = R(R+α⁻²I)⁻¹, Boolean(AND/OR/NOT) 조합 가능.
- **RepE(Representation Engineering)**: 내부 표상을 읽고(reading)·조작해(control) 행동을 제어하는
  기법의 **총칭**(Zou 2023 매니페스토의 용어, Wehner 서베이가 분야명으로 채택). 추론 시 활성화를
  수정하는 activation steering은 RepE의 부분집합이고, 표상 기준 손실로 가중치를 학습하는
  LoRRA·ReFT·Circuit Breakers도 RepE에 포함된다. "RepE vs FT" 비교에서 RepE 쪽 = steering 계열.

# §1. 무엇 & 왜 (Foundations & Motivation)

## 정의
Activation steering은 모델의 내부 표현을 **읽고(read-out)** 원하는 방향으로 **써넣어
(write-in)** 재학습 없이 행동을 바꾸는 것이다. 되돌릴 수 있고(끄면 원상복귀), 싸고(gradient 불필요),
가중치·데이터 접근 없이 행동 축을 조절한다는 점이 핵심 동기다. 더 큰 그림에서는 **해석가능성을
제어로 잇는 것**(안전/정렬: 거부·정직·해악 억제)이 목적이다.

## 대표 논문 흐름
"개념 = 방향"의 뿌리는 word2vec 아날로지(king−man+woman≈queen,
[notes/BolukbasiDebias.md](notes/BolukbasiDebias.md)의 배경)이고, **Bolukbasi 2016**이 편향을
방향으로 정량화해 **투영으로 제거**하면서 표현 편집·steering의 수학적 원형을 만들었다
([notes/BolukbasiDebias.md](notes/BolukbasiDebias.md)). **Toy Models of Superposition**(Elhage
2022, [notes/ToyModelsSuperposition.md](notes/ToyModelsSuperposition.md))은 신경망이 뉴런 수보다
많은 개념을 비직교 방향에 중첩(superposition)해 "선형표현이 왜 깨끗하지 않은지"를 설명하고 SAE의
이론적 동기를 준다. **Park 2023 (Linear Representation Hypothesis)** 이 이 모든 실증을
counterfactual로 형식화해 **"탐지 방향(probe) = 개입 방향(steering vector)"** 임을 증명했고
([notes/ParkLRH.md](notes/ParkLRH.md)), **Marks & Tegmark (Geometry of Truth)** 가 probe를 causal
개입으로 확장해 "읽기→쓰기" 다리를 놓았다([notes/GeometryOfTruth.md](notes/GeometryOfTruth.md)).
**Zou 2023 (Representation Engineering)** 이 이를 top-down "reading + control" 매니페스토로 통합해
이후 모든 섹션의 출발점이 된다([notes/RepE.md](notes/RepE.md)). 이 지형 전체의 **공식 지도**는
**RepE Survey**(Wehner, TMLR 2025, >130편 — identification→operationalization→control 3단계
taxonomy, DiM 최강·matrix>vector·multi-concept 간섭 등 실증 종합,
[notes/RepESurvey.md](notes/RepESurvey.md)) — 이 문서의 §2~§5 주장을 교차 검증하는 유일한
peer-reviewed 서베이.

## 축/tradeoff
① **reading vs writing** — 존재(probe)가 사용(causal)을 보장하지 않는다(§2 핵심).
② **선형가설의 한계** — 요일/월 같은 원형(다차원) feature도 있어(Engels 2024) 단일 방향으로 안
잡히는 개념이 있다. 이것이 우리가 단일벡터 대신 다차원 conceptor를 쓰는 이유와 통한다.

## 우리 프로젝트 연결
"탐지=개입 방향" 동치(Park)는 ==online 실패검출기의 방향을 그대로 steering
방향으로 재사용==할 수 있다는 이론적 근거다. superposition은 "성공/실패가 단일 방향에 깨끗이
대응하지 않음"을 뜻해 **다차원 contrastive conceptor**의 정당화가 된다.

<details><summary>🎤 면접 포인트</summary>

Q: "activation steering이 왜 통하나?" → A: "선형표현가설을 counterfactual로
형식화하면 프로브 방향과 steering 벡터가 causal inner product 아래 같은 방향임이 증명된다(Park).
즉 좋은 분류기를 찾으면 그게 곧 개입 방향이다. 뿌리는 word2vec 아날로지·Bolukbasi 편향 방향
제거까지 올라간다."

</details>

# §2. Activation 분석 — read-out (활성화에서 개념 읽기)

## 정의
내부 표현에서 개념·방향을 **식별하고 인과를 검증**하는 도구들. 이 섹션의 계보는
**상관(correlational) → 인과(causal)** 로 이동한다.

## 대표 논문 흐름
- **Probing**: 선형 분류기로 층별 표현의 선형 분리도를 측정(Alain & Bengio 2016). 비지도
  버전(CCS)도 있으나 "존재 ≠ 사용"이라는 **인과성 논쟁**이 핵심.
- **Lens**: 중간층 residual을 unembedding에 투사해 예측 궤적을 읽음(logit lens → **Tuned Lens**,
  affine probe로 보정).
- **방향 추출**: PCA / **mass-mean(diff-of-means)** 로 방향을 뽑고 인과 개입으로 검증 —
  **Geometry of Truth**([notes/GeometryOfTruth.md](notes/GeometryOfTruth.md))가 대표. 여기서
  diff-of-means가 분류 정확도는 같아도 **causal 효과는 크게 앞선다**(우리 conceptor의 diff-of-means
  다차원 확장 근거).
- **SAE(sparse autoencoder)**: superposition을 희소 사전으로 분해. **Cunningham 2023**(학계 독립
  검증, [notes/CunninghamSAE.md](notes/CunninghamSAE.md)) → **TopK SAE**(Gao/OpenAI, sparsity 직접
  통제·스케일링, [notes/TopKSAE.md](notes/TopKSAE.md)) → Gated/JumpReLU → **Gemma Scope**(오픈
  스위트, §5). Anthropic **Towards/Scaling Monosemanticity**(블로그)가 규모를 키웠다.
- **Activation patching / causal tracing**: 특정 위치 활성화를 바꿔 인과 기여를 측정. **ROME**(사실
  지식 국소화 + rank-one edit, [notes/ROME.md](notes/ROME.md)) → **IOI circuit**(head 단위 회로
  역공학, path patching, [notes/IOI.md](notes/IOI.md)) → attribution patching(스케일).

## 축/tradeoff
probing(상관, 싸다)·SAE(해석성, 무겁다)·patching(인과, 국소적). **SAE가 detection은
잘해도 steering은 실망스럽다**는 결과(§3 AxBench)를 미리 기억할 것.

## 우리 프로젝트 연결
SAFE([notes/SAFE.md](notes/SAFE.md))가 이 read-out 계열(succ/fail 분리·per-step
probe)이고, ROME/IOI의 인과 귀인은 우리 **VL/DiT pathway 귀인**의 방법적 조상이다. 주의: **길이
confound** — 실패가 항상 timeout이면 시간축만으로 분리돼(seen18 AUROC 0.998) time-pooled 분리는
아티팩트. probe 정확도 ≠ 인과 효과(Geometry of Truth)라는 교훈이 여기서도 성립.

<details><summary>🎤 면접 포인트</summary>

Q: "probing만으로 왜 부족한가?" → A: "선형 프로브가 개념을 분류해내도 그 방향이
출력에 인과적으로 쓰이는지는 별개다. Geometry of Truth는 분류 정확도가 같아도 causal 개입 효과가
크게 다름을 보였고, 그래서 activation patching(인과)과 steering(개입)으로 검증해야 한다." Q: "SAE가
뭐고 왜 쓰나?" → A: "superposition으로 뉴런이 다의적이라, 활성화를 희소 사전으로 분해해 monosemantic
feature를 얻는 것. 단 detection엔 좋아도 steering 성능은 단순 baseline에 밀릴 수 있다(AxBench)."

</details>

# §3. Steering 방법 — write-in (활성화 써넣기)

## 정의
개입 방향을 실제로 **써넣어** 행동을 조종. 이 섹션이 방법론의 핵심.

## 분류 축(면접 필수)
- **additive vs projective**: h' = h + αv (고정 이동) vs h' = h·C (방향별 soft scaling/투영).
- **single-vector vs subspace**: 방향 하나 vs 부분공간(공분산/여러 head).
- **boosting vs suppression**: 더해서 강화 vs 빼서/투영해 억제.
- **fixed vs learned**: 통계로 고정 vs gradient로 학습.

## 대표 논문 흐름
- **Additive**: **ActAdd**(대조 프롬프트쌍 활성화 차, 최적화 불필요, single-vector,
  [notes/ActAdd.md](notes/ActAdd.md)) → **CAA**(다수 대조쌍 평균차를 전 토큰에 주입, 표준 baseline,
  [notes/CAA.md](notes/CAA.md)) → **ITI**(probe로 head 선별 후 그 subspace에만 shift,
  [notes/ITI.md](notes/ITI.md)).
- **Task/Function vector**: **Function Vectors**(causal mediation으로 소수 head 식별→압축, bottom-up,
  [notes/FunctionVectors.md](notes/FunctionVectors.md)); ICL task vector 계열(skim).
- **Projective/subspace**: **★ Conceptors**(activation 집합을 타원체 soft projection C=R(R+α⁻²I)⁻¹로
  표현, Boolean AND/OR/NOT 조합, **우리 headline의 직접 근거**, [notes/Conceptors.md](notes/Conceptors.md));
  **Arditi refusal**(방향을 **directional ablation=투영 제거**로 억제, erasure≠addition,
  [notes/ArditiRefusal.md](notes/ArditiRefusal.md)).
- **Learned**: **ReFT/LoReFT**(hidden state에 저랭크 학습된 개입, ActAdd/RepE를 rank-1 비학습
  특수사례로 재정식화, [notes/ReFT.md](notes/ReFT.md)).
- **신뢰성 비판(★ 면접 차별화)**: **Tan**(per-sample steerability가 크게 요동, 절반이 anti-steerable,
  spurious "steerability bias", [notes/TanSteeringReliability.md](notes/TanSteeringReliability.md));
  **AxBench**(diff-in-means가 SAE를 압도, 심지어 prompting이 대부분 최강,
  [notes/AxBench.md](notes/AxBench.md)).

## 우리 프로젝트 연결
우리 **C_steer = C_success ∧ ¬C_failure** 는 Conceptor 논문의 Boolean 대수를
그대로 조합한 것: ¬C = I − C(aperture 무관 정확식), A∧B = (A⁻¹+B⁻¹−I)⁻¹. 적용식 h' = h·Mᵀ,
M=(1−β)I+β·C_steer. **additive boosting은 GR00T DiT에서 위험**(NOTALL fragility, §6)이라 projective·
suppressive가 안전. Tan/AxBench의 비판(단순 baseline이 강함, steering이 brittle)은 **우리가 global
steering 대신 phase/pathway 조건부를 택한 근거**이자, **복잡한 method는 diff-in-means baseline을
반드시 이겨야 한다는 사다리식 ablation** 규율을 준다.

<details><summary>🎤 면접 포인트</summary>

Q: "conceptor가 단일 steering vector보다 나은 이유?" → A: "단일벡터는 activation
cloud를 평균(점)으로만 요약해 상관·분산 구조를 버린다. conceptor는 공분산 R을 통째로 담은 soft
projection이라 패턴 안의 성분은 통과시키고 밖만 눌러준다. 실험적으로도 전 태스크에서 mean-centered
additive를 능가." Q: "steering의 대표적 실패는?" → A: "Tan: 입력별로 요동치고 절반이 반대로 작동,
위치/토큰 spurious bias. AxBench: SAE가 diff-in-means·prompting에 밀림. 그래서 조건부·검증된
개입이 필요."

</details>

# §4. LLM/VLM에서의 steering 연구 (무엇을 제어하나)

## 정의
steering이 실제로 **무엇을** 제어하는가의 대표 사례. §3 방법이 여기서 안전·행동으로 구체화된다.

## 대표 논문 흐름
- **안전/거부**: **Arditi**(13개 모델에서 거부가 **단일 방향**으로 매개, 제거하면 jailbreak, 추가하면
  무해 요청도 거부 — necessary+sufficient 이중 인과검증, [notes/ArditiRefusal.md](notes/ArditiRefusal.md)).
- **정직/아첨**: **ITI**(truthful 방향 주입, TruthfulQA 32.5→65.1%); **Sycophancy**(RLHF 선호데이터가
  아첨을 보상함을 규명 — steering이 겨냥하는 행동의 **원인 분석**, [notes/Sycophancy.md](notes/Sycophancy.md));
  RepE(다속성 통합).
- **대규모 SAE feature 제어**: Anthropic **Scaling Monosemanticity / Golden Gate Claude**(수백만
  monosemantic feature를 clamping해 행동 제어 — 화제성 큰 데모).
- **persona/성격**: **Persona Vectors**(evil·아첨·환각 trait를 대조 활성화로 자동 추출→모니터링·추론시
  완화·**학습데이터 영향 사전 예측**, [notes/PersonaVectors.md](notes/PersonaVectors.md)).
- **멀티모달(VLM)**: **VTI**(LVLM 환각을 vision encoder(non-causal·전토큰)와 text decoder(causal·
  last-token)에 **각각 다른 방향**으로 개입해 억제, [notes/VTI_VLMHallucination.md](notes/VTI_VLMHallucination.md));
  Textual Steering Vectors·AutoSteer·VISTA(skim).

## 축/tradeoff
개입은 강력하지만 **양날의 검**(거부 방향 제거 = jailbreak). 안전 연구가 곧 공격
연구다. 멀티모달은 **modality별로 개입 지점/방향이 달라야** 한다는 게 핵심(VTI).

## 우리 프로젝트 연결
**VTI의 vision/text 분리 개입 = 우리 VL/DiT pathway 분리의 멀티모달 선례.**
**Persona Vectors의 "생성 전 projection으로 향후 trait 예측" = 우리 online 실패유형 식별의 직접
선례.** Arditi의 ¬(방향)=directional ablation은 우리 ¬C_failure(실패 부분공간 억제)와 직접 유비.

<details><summary>🎤 면접 포인트</summary>

Q: "refusal이 단일 방향이라는 게 왜 중요한가?" → A: "안전장치조차 1차원으로
매개되면 그 방향을 지워 jailbreak가 가능하다는 뜻 — 표현 수준 안전의 취약성. 동시에 diff-of-means로
그 방향을 뽑아 necessary(제거→무력화)+sufficient(추가→과잉거부)를 인과 검증한 깔끔한 사례." Q:
"VLM steering이 LLM과 다른 점?" → A: "vision은 non-causal(전 토큰), text는 causal(last-token)이라
개입 지점·방향을 modality별로 달리해야 한다(VTI). 이게 VLA의 VL/DiT 분리로 이어진다."

</details>

# §5. 산업 적용 현황 — 정직한 현실 점검

## 핵심 한 줄
2026년 중반 기준, **실제 소비자 LLM/VLM/이미지 서비스의 서빙 경로에 inference-time
activation *steering(쓰기)* 가 배포된 근거는 사실상 없다.** 활성화를 *읽는* probe/monitor는 신중히
진입 중이고, *쓰는* steering은 데모·연구·소수 B2B API·오픈소스 도구 생태계 수준에 머문다.

## 5.0 왜 "산업 적용"이 헷갈리나 — 3축 구분(면접 필수)
- 읽기(probe/monitor) vs **쓰기(steering)** — 읽는 것과 써넣는 것은 완전히 다르다.
- 데모 vs 개발자 API/도구 vs **실제 소비자 서비스**.
- train-time(가중치에 새김) vs **inference-time(활성화 조작)**.
"산업에 적용됐다"가 이 셋을 뭉개면 과장된다.

## 5.1 배포 현실 (evidence)

### LLM/VLM
- **읽기(진입 중)**: Anthropic **차세대 Constitutional Classifiers** — Claude Sonnet 4.5 실트래픽에
  내부 활성화 **선형 probe**를 "production-grade"로 운영(shadow, flag rate 0.05%; 완전 상시화는
  문구상 불확정). → **읽기지 steer 아님.**
- **쓰기(유일한 명확한 실상업)**: **Goodfire Ember** — 오픈모델(Llama) feature를 실제 조작하는
  hosted API, 실고객(Rakuten·Apollo Research·Haize Labs), Series B 150M USD. 단 **대중 소비자 챗봇
  내장인지 백엔드/연구/레드팀인지 불명.**
- **데모**: Golden Gate Claude(24시간 한정). **연구**: Persona Vectors(감사용 읽기 + train-time
  steer), Gemini steering vectors(저자가 "conditional steering 있어야 production 가능"=현재 없음 명시),
  EleutherAI SOAR, Transluce Monitor(오픈 도구).
- **train-time(steer지만 추론 조작 아님)**: **Circuit Breakers**(Gray Swan, RepE를 LoRA로 가중치에
  새김; 회사는 레드팀 서비스 Cygnal 운영, 프론티어랩 실서빙 이식은 미확인,
  [notes/CircuitBreakers.md](notes/CircuitBreakers.md)).
- **steering 아님(대조)**: Anthropic/OpenAI sycophancy 개선 = RLHF·시스템프롬프트·분류기(1차 자료에
  "활성화 조작" 언급 0); Grok 페르소나 = 시스템 프롬프트; Character.AI류 = fine-tuning(Open Character
  Training [notes/OpenCharacterTraining.md](notes/OpenCharacterTraining.md) 이 정량 확인); Llama
  Guard = 별도 텍스트 분류기.

### 이미지 생성(diffusion)
- activation-space steering이 **오픈소스 도구 생태계엔 실재 탑재**: **SEGA**(Diffusers 공식
  SemanticStableDiffusionPipeline, [notes/SEGA.md](notes/SEGA.md)), **Safe Latent Diffusion**
  (Diffusers StableDiffusionPipelineSafe, opt-in, [notes/SafeLatentDiffusion.md](notes/SafeLatentDiffusion.md)),
  Prompt-to-Prompt(community pipeline), SEG(ComfyUI 노드), **Asyrp**(h-space, U-Net bottleneck 직접
  수정 = LLM residual steering 최유사, 연구단계 [notes/Asyrp.md](notes/Asyrp.md)).
- 그러나 **Midjourney·DALL·E 3·Firefly·Ideogram 등 대중 상용 서비스 탑재는 미확인**. 확인되는 상용
  기법(DALL·E 3 캡션 재작성)은 train-time 데이터 개조지 steering 아님.
- **Concept Sliders는 weight-space LoRA라 activation steering이 아님**(널리 쓰이지만 분류 주의).

### 정직한 판정
**읽기는 (일부) 실서비스 진입, 쓰기는 데모·연구·소수 B2B API·오픈소스 도구 수준.
대중 소비자 서비스의 제어 경로에 activation steering이 상시로 도는 사례는 확인되지 않는다.**

## 5.2 왜 중요한가 (찬성 논거)
- 재학습 없는 test-time 제어(RepE), on/off 토글·되돌리기.
- **조건부·프로그래머블 제어**: **CAST**(IBM, ICLR2025) — 무조건 steering이 실사용을 막는 병목을
  "if 조건 then steer"로 해결([notes/CAST.md](notes/CAST.md)).
- **개인화 UX**: **Google Steerable Chatbots** — 선호를 steering vector로, 강도를 슬라이더로 노출
  ([notes/GoogleSteerableChatbots.md](notes/GoogleSteerableChatbots.md)).
- **agentic 온라인 제어**: **ASA** — tool-calling 중 signed gate(rescue/suppress/abstain)로 실시간
  개입, backbone frozen·+10.9% latency로 실전급([notes/ASA_ToolCallingRepE.md](notes/ASA_ToolCallingRepE.md)).

## 5.3 왜 아직 안 쓰이나 (장벽 — evidence가 강함)
- **Brittleness**: Tan(입력별 요동·anti-steerable, [notes/TanSteeringReliability.md](notes/TanSteeringReliability.md)),
  AxBench(prompting·diff-in-means > SAE, [notes/AxBench.md](notes/AxBench.md)).
- **안전 역설**: **Rogue Scalpel** — 무작위 방향 steer만으로도 유해 순응 0→1~13%, SAE의 benign
  feature조차 위험 → "해석가능성=안전" 전제 도전([notes/RogueScalpel.md](notes/RogueScalpel.md)).
- **평가 미성숙**: **Reliable Evaluation** — 기존 성공 보고가 MCQ/정성 데모 의존, 4속성 재평가 시 CAA
  sycophancy 사실상 무효 등 효과 과장 확인([notes/ReliableEvalSteering.md](notes/ReliableEvalSteering.md)).
- **capability 손상 예측불가**: Minimizing Collateral Damage([notes/MinimizingCollateralDamage.md](notes/MinimizingCollateralDamage.md)).
- **범용 파이프라인 불가**: Open Character Training — steering 강도상수가 모델마다 0.7~525로 제각각,
  fine-tuning이 강건성·일관성 압도 → **업계가 fine-tuning을 택하는 정량 근거**.
- **조건부여야 함**: CAST/ASA의 no-gate ablation이 붕괴(always-on은 과잉거부·FPR 폭증).
- **운영 현실**: API-only 모델엔 못 씀, 대조데이터·모델별 튜닝 비용, 긴 대화서 감쇠(실무자 field
  guide). **툴링 위축**: Goodfire 공개 API → select-partner 축소.

## 5.4 전망 (언제/어떤 조건에서)
- **유력 경로**: agentic AI 온라인 제어(ASA), 개인화(Google), 조건부 안전 gate(CAST).
- **투자·로드맵**: Goodfire 150M USD Series B, "model design environment" 비전(evidence 투자 + opinion 비전).
- **규제 압력**: EU AI Act Art.13(투명성 요구, 특정 메커니즘 미명시 — 간접 압력).
- **채택 조건(핵심)**: (1) 신뢰성(brittleness 해결), (2) 표준 평가/QA, (3) **조건부·정밀 라우팅**
  (always-on 아님), (4) capability 손상 통제. → **정확히 우리 프로젝트가 겨냥하는 것들.**
- 견해는 갈림: Amodei "Urgency of Interpretability"(낙관) vs 실무자·커뮤니티 "이론·벤치·비용
  미성숙"(신중) — 대부분 opinion.

## 5.5 Steering vs Fine-tuning — 임시방편인가, 다른 축인가

**"steering = 소량 데이터로 OOD 상태를 ID로 재투영"이라는 직관은 절반만 맞다.**
projective/conceptor 계열엔 정확 — C_success 타원체(=ID manifold)로 soft projection하는 것이고,
additive 고배율이 실패하는 이유(NOTALL −68pp@9×, Rogue Scalpel)가 바로 활성화를 manifold
밖(OOD)으로 밀기 때문이다. 그러나 일반화하면 틀림: refusal 제거·persona·ITI는 OOD 복원이 아니라
**모델이 이미 표상하는 행동들 중 무엇을 출력으로 이을지 재가중**하는 것(TruthfulQA의 거짓말은
오히려 ID 행동). 통일 문장 = "steering은 이미 있는 표상의 mixture를 바꾸고, **없는 능력은 추가
못한다**" — 이것이 fine-tuning과의 본질 차이.

### Regime 지도 (데이터 양 × 실패 종류)
- **소량 데이터(<500 샘플)**: steering 우세 실증 — RepE Survey 메타에서 FT 전승 구간이고, COAST에선
  같은 15 rollout으로 SFT하면 오히려 음수(GR00T RoboCasa 0.59→**0.50**) vs conceptor +0.16.
  소량에선 FT가 과적합으로 역효과.
- **대량 데이터 + 정적 행동 이동**: FT 승(OCT: steering 강도상수 0.7~525로 모델별 제각각, FT가
  강건성·일관성 압도). "평균 행동의 prior 이동"이 목표면 데이터가 쌓인 뒤엔 FT가 정답.
- **상태-조건부 런타임 개입**: ==데이터가 무한해도 FT로 흡수 안 되는 축== — FT는 가중치에 정적
  prior를 새기는 것이라 "지금 이 rollout이 phase 3에서 드리프트 중"이라는 조건에 반응 못한다.
  FT로 풀려면 모든 드리프트 모드의 recovery demonstration이 필요(가장 비싼 데이터). steering은
  outcome 라벨(성공/실패)만으로 fit되는 폐루프 제어기. 비유: FT=플랜트 재설계, steering=피드백 제어
  — 잘 학습된 정책도 배포에선 계속 새 교란을 만나므로(런타임 모니터 Sentinel/FIPER가 존재하는 이유)
  후자는 전자로 대체되지 않는다.

### 결합·증류 경로 (대체가 아니라 합류 — 메타스터디에서 결합>단독 4/4)
- ① **나란히 쓰기**: FT된 모델 위에 런타임 steering. KL-then-steer(Stickland)는 아예 "steering에
  강건하도록 먼저 FT → 그 위에 steer".
- ② **representation loss로 FT ("가중치에 증류")**: steering 목표(표상을 특정 방향으로/밖으로)를
  손실함수로 만들어 LoRA를 학습 — **LoRRA**(RepE 원논문)·**Circuit Breakers**. 학습이 끝나면
  런타임 훅 없이 가중치 스스로 그 표상을 생성한다.
- ③ **learned intervention**: **ReFT** — 개입 자체가 학습 파라미터(steering과 FT의 중간형,
  LoRA급 PEFT로도 경쟁력).
- ④ **선형 개입의 정확 융합**: h'=h·M처럼 선형이면 다음 층 가중치에 W←W·M으로 **학습 없이 정확히
  흡수** 가능(Conceptors §3.2). 단 상시 적용일 때만 — 조건부 on/off·phase 스위칭엔 불가(우리
  경우가 정확히 융합 불가 지점).
- ⑤ **학습 중 예방 steering**: Persona Vectors — FT 도중 나쁜 trait 방향을 눌러 습득 자체를 방지.

### always-on 모순의 해소 (LLM 붕괴 vs COAST 생존)
- LLM의 always-on 붕괴는 **additive(h+αv, 증폭 가능·off-manifold 이탈)** 고강도·광역 적용의
  문제다. conceptor의 M=(1−β)I+β·C_steer는 고유값이 [1−β,1]인 **수축 연산자**라 증폭이 구조적으로
  불가능 — 손상 모드 자체가 다르다.
- 실증: **COAST cross-task 전이(§4.4, Table 2)** — 잘못 매칭된(다른 task의) conceptor를
  always-on으로 켜도 **최악 ~−0.2, 붕괴 없음**; 여러 pair에선 self-fit과 대등 이상. 전이 이득은
  **실패-subspace 공유**(containment r=0.30/0.49)가 결정하고 성공-subspace 겹침은
  무상관(|r|<0.13) — 매칭이 어긋난 성분은 해를 안 끼치되 이득도 못 만든다.
- 단, 우리 7-layer 동시 스택은 SR 0.000 붕괴 — 수축 연산자라도 **개입 예산(강도×layer 수×범위)
  초과 시 붕괴**한다. 면역이 아니라 예산 문제.
- 정리: ==손상의 하한은 연산자 기하가 보장하고, 이득의 상한은 매칭이 결정한다== —
  gating/phase-matching은 생존용이 아니라 **이득 회수용** 설계다.

**우리 프로젝트 함의**: 방어 논리는 "FT 전 임시"가 아니라 "==phase-조건부 복구는 weight에 새길 수
없는 폐루프 제어==". 동시에 정직한 리스크 — steering 이득이 filtered-BC(성공 데이터만 FT)로 동일하게
얻어지면 이 논리가 무너지므로, 사다리 ablation에 **SFT/filtered-BC baseline 필수**(COAST의 SFT
컬럼이 정확히 이 통제이고, 거기서 SFT가 음수였다는 것이 현재까지의 우리 편 증거).

## 우리 프로젝트 연결
우리는 **조건부(CAST)·online(ASA)·정밀(pathway/phase)·검증된(ΔSR ladder)**
steering을 지향 — §5.4의 채택 조건과 정확히 겹친다. 단 CAST는 조건 감지가 프롬프트 1회 정적,
ASA는 discrete parser 라벨이 있는 반면, 우리는 **라벨 없는 continuous phase/failure-type을 rollout
매 스텝 online 추론**해야 해 더 어렵다. Rogue Scalpel은 우리 ablation에 **random-direction negative
control**을, Minimizing Collateral Damage는 **anisotropic/β-blend로 부작용 통제**를 시사(주의: 이
논문도 자칭 "COAST" 2605.01167 — 우리 COAST 2605.17144와 동명이인).

<details><summary>🎤 면접 포인트</summary>

- Q: "activation steering이 실서비스에 쓰이나?" → A: "**쓰기(steering)는 아직 아니다.** 읽기(활성화
  probe)는 Anthropic이 Claude 실트래픽에 신중히 배포 중이지만, 쓰기 실상업은 Goodfire의 오픈모델
  B2B API가 거의 유일하다. 대중 챗봇/이미지 서비스 제어 경로엔 없다. 이미지 쪽은 SEGA·SLD가
  Diffusers·ComfyUI 오픈소스 도구엔 실제 탑재돼 있다."
- Q: "왜 안 쓰이나?" → A: "brittleness(Tan/AxBench), 안전 역설(Rogue Scalpel: 무작위 방향도 유해↑),
  범용 파이프라인 불가(모델마다 강도 0.7~525), 평가 미성숙. 그래서 fine-tuning·프롬프트가 더 안정적."
- Q: "그럼 전망은?" → A: "조건부(CAST)·agentic(ASA)·개인화(Google)에서 실전급 결과가 나오고 있고,
  채택 조건(신뢰성·평가·조건부·손상통제)이 풀리면 유력. 내 연구가 바로 그 조건을 겨냥한다."
- Q: "steering은 fine-tuning 전 임시방편 아닌가?" → A: "regime이 다르다. 500샘플 미만에선 steering이
  실증 우세(COAST에선 같은 15 rollout으로 SFT하면 오히려 음수), 대량 데이터의 정적 행동 이동은 FT 승
  (OCT). 그러나 상태-조건부 런타임 복구는 데이터가 많아도 FT로 흡수 안 되는 축이고 — FT는 '지금 이
  rollout이 드리프트 중'이라는 조건에 반응 못한다 — 성숙 경로는 대체가 아니라 결합·증류다(LoRRA/
  Circuit Breakers가 steering 목표를 손실로 만들어 LoRA에 새김, ReFT는 개입 자체를 학습)."

</details>

# §6. VLA / world model에서의 activation 분석·steering

## 정의
LLM/VLM steering을 로봇 정책(VLA)·world model로 확장. 핵심은 **개입 지점의 다변화**.

## 개입 지점 지도(면접용)
- **hidden state 직접**: 우리 프로젝트 · **NOTALL**(VL/DiT 기능분리) · **SAE-VLA**(Swann) ·
  **LAE**(online edit) · **Observing&Controlling**(제어이론) · **MechInterpSteering**(FFN 클램프).
- **denoising 샘플링**: **VLS**(VLM reward로 flow/diffusion 샘플링 guide, blackbox,
  [notes/VLS_SteerViaVLM.md](notes/VLS_SteerViaVLM.md)); DSRL(latent-noise RL)·DynaGuide(dynamics
  guide) skim.
- **명령/subgoal 층위**: **CoT-VLA**(subgoal 이미지 생성, [notes/CoT-VLA.md](notes/CoT-VLA.md));
  **SteerableVLAs=InSight**(primitive 라벨+LoRA data flywheel, [notes/SteerableVLAs.md](notes/SteerableVLAs.md));
  ScalingWorldModel=VISTA(world model이 subtask+goal image로 분해).

## 선행토대 트리오(우리 방법의 뼈대)
- **NOTALL**([notes/NOTALL.md](notes/NOTALL.md)): π0.5·SmolVLA·GR00T에서 **expert/DiT=motor,
  VLM=goal** 기능분리 재현. GR00T DiT는 fragile(ΔSR −68pp@9× amp), boosting은 양방향 파괴적,
  motor program이 **approach phase에 조기 커밋**(phase-matched 초기개입 근거). → 빌리는 것=pathway
  분해·fragility 원칙(projective·early·소강도). 메우는 곳=online·failure-type·phase-matched positive.
- **COAST**([notes/COAST.md](notes/COAST.md)): conceptor(C_success ∧ ¬C_failure)를 VLA rollout에
  이식. 단 **Per-step 전략의 축이 denoising-step이지 rollout task-phase가 아니고, 전 timestep을 한
  R로 pool** → 길이/phase confound·pathway 미분리. → 빌리는 것=조종 연산자. 메우는 곳=phase-bin별
  R fit(수학은 그대로, 집계 축만 phase로). ablation: GR00T N1.5 RoboCasa Base 0.59→Global 0.75.
- **SAFE**([notes/SAFE.md](notes/SAFE.md)): succ/fail이 feature-space에서 task-agnostic하게 분리되는
  failure zone + per-step probe + conformal threshold. **길이 아티팩트를 직접 경고하고 min-length로
  통제**(COAST엔 없음). 저자도 "검출=조종 근거"라며 ITI 인용. → 분리 가능성=조종 가능성의 전제.

## 그 밖의 VLA steering/해석
- **SAE-VLA**(Swann, [notes/SAE_VLA_pi05.md](notes/SAE_VLA_pi05.md); PDF는 `../references/dr_vla.pdf`):
  π0.5 hidden에 SAE→motion primitive feature+steering. **feature 94.9~99.6%가 memorized** →
  "clean feature ≠ reliable steerability", 우리 conceptor도 collection-episode 암기 방향에 올라탈
  위험 경고.
- **Event-Grounded SAE**([notes/EventGroundedSAE.md](notes/EventGroundedSAE.md)): per-token SAE
  feature를 **kinematic event(pre_grasp/contact/detach…)에 anchoring** → event=phase 신호 후보(단
  논문은 라벨을 시각화에만 씀; 우리가 신호로 승격시켜야).
- **Observing&Controlling**([notes/ObservingControlling.md](notes/ObservingControlling.md)):
  feature observability/controllability를 제어이론으로 형식화. π0.5는 VLM-only 개입도 motor 출력을
  바꿔 **VL→DiT 직렬 결합** 우려를 실증. 1D 구간제약만 폐형해 → 우리 다차원 conceptor가 그 한계를 메움.
- **MechInterpSteering**(Häon CoRL2025, [notes/MechInterpSteering.md](notes/MechInterpSteering.md);
  우리가 Phase A 재현): FFN 뉴런 클램프로 fast/slow 제어. **SR 미보고(displacement만)**, 우리 재현에서
  **길이 통제 시 +26%→+15.7%로 반감** — 길이 confound 교훈. OpenVLA/π0-FAST는 DiT 헤드가 없어 VL/DiT
  질문 자체가 미성립.
- **world model**: ScalingWorldModel=VISTA([notes/ScalingWorldModel.md](notes/ScalingWorldModel.md))는
  world model이 subtask switcher로 phase를 **외부 출력으로 우회** — 우리는 내부 latent만으로 풀어야 함.
  Video WM latents(action-relevance probe) skim.

## 우리 프로젝트 위치(미점유 niche)
==내부 latent × online × 실패 TYPE(goal/motor) × phase-matched
steer.== hidden state에 다차원 contrastive conceptor를 pathway별·phase별 적용. LAE(검출→라우팅→개입
구조 동형; **self-dynamics까지 편집하면 파국 → 편집 범위 선택이 성패**, [notes/LAE_LatentActivationEditing.md](notes/LAE_LatentActivationEditing.md))·
DynaGuide(dynamics 조건화)와 직접 비교 대상.

<details><summary>🎤 면접 포인트</summary>

Q: "VLA에서 개입 지점이 왜 다양한가?" → A: "hidden state(우리·NOTALL·SAE-VLA),
denoising 샘플링(VLS), subgoal 이미지(CoT-VLA), 명령 추상화(InSight) 등 층위가 다르다. 같은
'steering'이라도 activation-time(우리)과 sampling-time(VLS)은 메커니즘이 다르다." Q: "우리 method의
novelty는?" → A: "COAST가 conceptor를 VLA에 처음 이식했지만 전 timestep pool로 phase를 놓쳤고,
NOTALL이 VL/DiT 분리를 보였지만 online·failure-type·positive steering은 안 했다. 우리는 그 교집합 —
online에서 pathway·phase별로 조종을 라우팅 — 을 겨냥한다."

</details>

# §7. VLA 산업 적용 방향 (배포로 잇기)

## 정의
activation steering을 실제 로봇 배포로 잇는 실용 갈래: **검출 → 복구 → 안전 gate → 신뢰성**.

## 대표 논문 흐름
- **online 실패 검출**: **Sentinel**(실패를 **erratic(STAC=action-chunk 시간불일치, conformal FPR
  보장) vs progress(VLM QA)** 2분법으로 병렬 검출, output-only,
  [notes/Sentinel_RuntimeMonitor.md](notes/Sentinel_RuntimeMonitor.md)); **FIPER**(embedding OOD(RND)
  + action-chunk entropy로 무실패데이터 조기예측, [notes/FIPER_FailurePrediction.md](notes/FIPER_FailurePrediction.md));
  **PathDeviationHeads**(navigation head **attention entropy**로 이탈 검출→외부 RL rollback 우회,
  [notes/PathDeviationHeads.md](notes/PathDeviationHeads.md)); **I-FailSense**(별도 관찰자 VLM,
  [notes/I-FailSense.md](notes/I-FailSense.md)); SAFE.
- **복구/introspection**: **KnowNo**(conformal prediction으로 불확실성 정렬→singleton이면 실행/아니면
  "도움 요청", [notes/KnowNo_AskForHelp.md](notes/KnowNo_AskForHelp.md)); **VITA**(frozen CLIP+TTT로
  progress/value 예측 — **phase-matched steering의 online phase 신호 공급 후보**,
  [notes/VITA.md](notes/VITA.md)); FailSafe skim.
- **안전 gate**: latent safety filter(world model OOD를 conformal로 필터, steering과 메커니즘 최근접,
  skim); Circuit Breakers(§5).
- **신뢰성 벤치**: LIBERO-Plus(7축 섭동으로 VLA 취약성 정량, SR 붕괴, skim) — "언제 믿나".

## 우리 프로젝트 연결 / 산업화 로드맵
검출기들은 대부분 **output-only(Sentinel/FIPER)** 이거나
**우회(PathDeviationHeads)** 다. 우리 phase-matched steering은 **요청/재계획/우회 없이 latent 안에서
즉시 복구**하는 저비용 대안. **PathDeviationHeads 저자도 "물리 rollback은 VLA 내부 인지가
미동기화돼 실패가 재발 → replanning을 future work"** 라 인정 — 정확히 activation steering이 겨냥하는
지점. **산업 적용 필요조건**: (1) online 검출에 **conformal 보장**(KnowNo/Sentinel), (2) **phase 신호
공급**(VITA/Event-Grounded), (3) **pathway 라우팅**(NOTALL 기반), (4) **안전 gate**, (5) **길이/
instruction confound 통제**(SAFE/Tan), (6) **diff-in-means baseline 격파**(AxBench) — 이걸 다 갖춰야
"steer가 실제로 SR을 올린다"를 방어할 수 있다.

<details><summary>🎤 면접 포인트</summary>

Q: "실패를 어떻게 온라인에 아나?" → A: "output-only(Sentinel STAC·FIPER entropy),
attention(PathDeviationHeads), 별도 VLM(I-FailSense), progress predictor(VITA) 등이 있고, 통계적
보장은 conformal prediction(KnowNo)으로 준다. 우리는 정책 내부 latent에서 pathway·phase별로 읽어
개입까지 잇는 걸 목표로 한다." Q: "검출만 있으면 되지 왜 steering인가?" → A: "우회/rollback은 내부
인지를 못 고쳐 실패가 재발한다(PathDeviationHeads 저자 자인). latent를 성공 쪽으로 밀면 재학습·재계획
없이 내부 상태까지 교정할 수 있다는 게 가설."

</details>

# 부록 A. 면접 치트시트 (예상 질문 → 30초 답변)

1. **Activation steering이 뭐고 왜 하나?** — 추론 시 내부 활성화를 읽고 써넣어 재학습 없이 행동을
   바꾸는 것. 싸고 되돌릴 수 있고 가중치 접근이 불필요. 토대는 linear representation hypothesis.
2. **왜 통하나(이론)?** — Park: 선형표현가설을 counterfactual로 형식화하면 probe 방향=steering
   벡터. 좋은 분류기가 곧 개입 방향.
3. **additive vs projective 차이?** — h+αv(고정 이동, ActAdd/CAA) vs h·C(방향별 soft scaling,
   conceptor)/방향 제거(directional ablation, Arditi). projective가 magnitude 보존적이라 안전.
4. **conceptor가 단일벡터보다 나은 점?** — 평균(점)이 아니라 공분산(타원체)을 담아 상관·분산 구조
   보존. Boolean(AND/NOT)으로 목표 조합. C_success ∧ ¬C_failure.
5. **SAE가 뭐고 한계는?** — superposition 해소용 희소 사전 분해. detection엔 좋지만 steering은
   diff-in-means·prompting에 밀림(AxBench).
6. **activation patching?** — 특정 위치 활성화를 바꿔치기해 인과 기여 측정(ROME/IOI). 상관(probe)과
   달리 인과를 검증.
7. **steering의 대표적 실패/한계?** — Tan: 입력별 요동·anti-steerable·spurious position bias.
   AxBench: 단순 baseline이 SAE를 이김. → 조건부·검증된 개입 필요.
8. **refusal single direction의 함의?** — 안전장치가 1차원이면 지워서 jailbreak 가능. 표현 수준
   안전의 취약성 + 깔끔한 necessary/sufficient 인과검증.
9. **VLM steering이 LLM과 다른 점?** — vision(non-causal 전토큰)/text(causal last-token) 개입을
   달리해야 함(VTI). → VLA의 VL/DiT 분리로 연결.
10. **산업에 steering이 쓰이나?** — **쓰기는 아직 실서비스 배포 근거 없음.** 읽기(활성화 probe)만
    Anthropic이 Claude 실트래픽에 신중히 진입; 쓰기 실상업은 Goodfire B2B API가 거의 유일. 이미지는
    SEGA/SLD가 Diffusers·ComfyUI 오픈소스 도구엔 탑재. 못 쓰는 이유=brittleness·안전역설(Rogue
    Scalpel)·범용 파이프라인 불가(강도 0.7~525)·평가 미성숙. 전망=조건부(CAST)·agentic(ASA)·개인화(Google).
11. **VLA에서 개입 지점?** — hidden state / denoising 샘플링 / subgoal / 명령 추상화. activation-time
    vs sampling-time 구분.
12. **우리(내) 연구 novelty?** — 내부 latent × online × 실패 TYPE(goal/motor) × phase-matched
    conceptor steering. COAST(phase pool)·NOTALL(offline·no-type)·PathDeviationHeads(우회)의 빈자리.
13. **검출만으로 부족한 이유?** — 우회/rollback은 내부 인지 미교정→재발(PathDeviationHeads 자인).
    steering은 latent를 성공쪽으로 밀어 내부 상태까지 교정 시도.
14. **가장 큰 리스크/confound?** — 길이(실패=timeout), instruction-skew, GR00T DiT fragility,
    memorized feature(SAE-VLA 98%), detection≠steering.

# 부록 B. 논문 인덱스 (정독 52편, note 링크)

| # | 논문(key) | 섹션 | 폴더 | 한 줄 | note |
|---|---|---|---|---|---|
| 1 | Bolukbasi Debiasing | §1 | basic | 편향=방향, 투영 제거 = steering 원형 | [notes](notes/BolukbasiDebias.md) |
| 2 | Toy Models of Superposition | §1/2 | basic | superposition→다차원 필요·SAE 동기 | [notes](notes/ToyModelsSuperposition.md) |
| 3 | Park LRH | §1 | basic | probe 방향=steer 방향 형식화 | [notes](notes/ParkLRH.md) |
| 4 | RepE | §1/3 | basic | reading+control top-down 프레임 | [notes](notes/RepE.md) |
| 5 | Geometry of Truth | §2 | basic | diff-of-means, probe≠causal | [notes](notes/GeometryOfTruth.md) |
| 6 | Cunningham SAE | §2 | basic | 학계 SAE 독립검증 | [notes](notes/CunninghamSAE.md) |
| 7 | TopK SAE (Gao) | §2 | basic | sparsity 직접통제·스케일 | [notes](notes/TopKSAE.md) |
| 8 | ROME | §2 | basic | causal tracing+rank-one edit | [notes](notes/ROME.md) |
| 9 | IOI Circuit | §2 | basic | path patching 회로 역공학 | [notes](notes/IOI.md) |
| 10 | ActAdd | §3 | basic | additive 원형, 최적화 불필요 | [notes](notes/ActAdd.md) |
| 11 | CAA | §3 | basic | 대조쌍 평균차 표준 baseline | [notes](notes/CAA.md) |
| 12 | ITI | §3/4 | basic | head 선택 subspace, truthful | [notes](notes/ITI.md) |
| 13 | Function Vectors | §3 | basic | causal mediation 압축 | [notes](notes/FunctionVectors.md) |
| 14 | ★ Conceptors | §3 | basic | soft projection+Boolean = 우리 근거 | [notes](notes/Conceptors.md) |
| 15 | ReFT | §3 | basic | 저랭크 학습 개입 | [notes](notes/ReFT.md) |
| 16 | Tan 신뢰성 | §3 | basic | steering 요동·anti-steerable | [notes](notes/TanSteeringReliability.md) |
| 17 | AxBench | §3 | basic | diff-in-means > SAE 비판 | [notes](notes/AxBench.md) |
| 18 | Arditi Refusal | §4 | basic | 거부=단일방향, ablation=jailbreak | [notes](notes/ArditiRefusal.md) |
| 19 | Sycophancy | §4 | basic | RLHF가 아첨 보상(원인) | [notes](notes/Sycophancy.md) |
| 20 | Persona Vectors | §4 | basic | trait 추출·모니터·예측 | [notes](notes/PersonaVectors.md) |
| 21 | VTI (VLM 환각) | §4 | basic | vision/text 분리 개입 | [notes](notes/VTI_VLMHallucination.md) |
| 22 | Circuit Breakers | §5 | basic | RepE 안전제품화 | [notes](notes/CircuitBreakers.md) |
| 23 | Gemma Scope | §5 | basic | 오픈 SAE 인프라 | [notes](notes/GemmaScope.md) |
| 24 | SAE-VLA (Swann) | §6 | ref(dr_vla.pdf) | VLA SAE, memorized 98% 경고 | [notes](notes/SAE_VLA_pi05.md) |
| 25 | LAE | §6 | ref | online classifier+activation edit | [notes](notes/LAE_LatentActivationEditing.md) |
| 26 | VLS | §6 | ref | denoising sampling steering | [notes](notes/VLS_SteerViaVLM.md) |
| 27 | NOTALL | §6 | ref | VL/DiT 분리·fragility | [notes](notes/NOTALL.md) |
| 28 | COAST | §3/6 | ref | conceptor VLA 이식, phase pool 결함 | [notes](notes/COAST.md) |
| 29 | SAFE | §2/6 | ref | succ/fail 분리검출, 길이통제 | [notes](notes/SAFE.md) |
| 30 | Observing&Controlling | §6 | ref | 제어이론 형식화 | [notes](notes/ObservingControlling.md) |
| 31 | Event-Grounded SAE | §6 | ref | SAE feature=event(phase 후보) | [notes](notes/EventGroundedSAE.md) |
| 32 | MechInterpSteering(Häon) | §6 | ref | FFN 클램프, 우리 재현 | [notes](notes/MechInterpSteering.md) |
| 33 | CoT-VLA | §6 | ref | subgoal 이미지(비-activation) | [notes](notes/CoT-VLA.md) |
| 34 | SteerableVLAs(InSight) | §6 | ref | primitive 라벨+data flywheel | [notes](notes/SteerableVLAs.md) |
| 35 | ScalingWorldModel(VISTA) | §6 | ref | world model subtask 분해 | [notes](notes/ScalingWorldModel.md) |
| 36 | Sentinel | §7 | ref | erratic/progress 2분 검출 | [notes](notes/Sentinel_RuntimeMonitor.md) |
| 37 | FIPER | §7 | ref | embedding OOD 조기예측 | [notes](notes/FIPER_FailurePrediction.md) |
| 38 | KnowNo | §7 | ref | conformal ask-for-help | [notes](notes/KnowNo_AskForHelp.md) |
| 39 | PathDeviationHeads | §7 | ref | attention entropy→우회(경쟁자) | [notes](notes/PathDeviationHeads.md) |
| 40 | I-FailSense | §7 | ref | 관찰자 VLM 실패검출 | [notes](notes/I-FailSense.md) |
| 41 | VITA | §7 | ref | progress predictor(phase 신호) | [notes](notes/VITA.md) |
| 42 | CAST (Conditional Steering) | §5 | basic | 조건부 if-then steering(IBM) | [notes](notes/CAST.md) |
| 43 | Rogue Scalpel | §5 | basic | 무작위 방향도 유해↑(안전역설) | [notes](notes/RogueScalpel.md) |
| 44 | Google Steerable Chatbots | §5 | basic | 선호 기반 개인화 steering | [notes](notes/GoogleSteerableChatbots.md) |
| 45 | Reliable Evaluation of Steering | §5 | basic | steering 평가 과장 폭로 | [notes](notes/ReliableEvalSteering.md) |
| 46 | SEGA | §5 | basic | 이미지 semantic guidance(Diffusers) | [notes](notes/SEGA.md) |
| 47 | Open Character Training | §5 | basic | fine-tuning>steering(업계 선호) | [notes](notes/OpenCharacterTraining.md) |
| 48 | Asyrp (h-space) | §5 | basic | 이미지 hidden-state steering | [notes](notes/Asyrp.md) |
| 49 | Safe Latent Diffusion | §5 | basic | 이미지 안전 steering(Diffusers) | [notes](notes/SafeLatentDiffusion.md) |
| 50 | Minimizing Collateral Damage | §5 | basic | 부작용 완화(공분산 anisotropic, 자칭 COAST) | [notes](notes/MinimizingCollateralDamage.md) |
| 51 | ASA (Tool-Calling RepE) | §5 | basic | agentic 온라인 조건부 steering | [notes](notes/ASA_ToolCallingRepE.md) |
| 52 | RepE Survey (Wehner, TMLR) | §1 | basic | 분야 유일 peer-reviewed 지도(>130편 taxonomy) | [notes](notes/RepESurvey.md) |

## 스킴/web-only 확장
(정독 대상 아님; 전체 목록·출처는
[`_handoff/research_raw/master_index.md`](_handoff/research_raw/master_index.md)에 보존): Mikolov
word2vec, Othello-GPT, CCS, Tuned Lens, Gated/JumpReLU SAE, Attribution Patching, Subramani,
Hendel/ICV, MELBO, AutoSteer, VISTA(hallucination), DSRL, DynaGuide, Do-What-You-Say,
SteerVLM(VLM), SteerVLA(driving), Chain-of-World, Video-WM-latents, FailSafe, See-Plan-Rewind,
Code-as-Monitor, LIBERO-Plus, latent safety filters(2505.00779), RepE 서베이 대안본
Bartoszcze(2502.17601, 미게재 — 정독은 TMLR 게재본 #52 Wehner 채택) + Anthropic/Goodfire 블로그.
