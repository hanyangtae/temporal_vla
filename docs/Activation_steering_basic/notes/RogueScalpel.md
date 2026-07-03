# The Rogue Scalpel: Activation Steering Compromises LLM Safety (Korznikov et al. 2026)

- 출처: arXiv:2509.22067v2 (2026-02-15) · Korznikov, Galichin, Dontsov, Rogov, Oseledets, Tutubalina
- PDF: `docs/Activation_steering_basic/RogueScalpel_2509.22067.pdf`
- §5 파트: **장벽(안전 역설)** — activation steering이 "해석가능=안전"이라는 통념에 대한 반증 사례
- 3축 분류: (도메인=LLM 텍스트, 개입=steering/injection, 목적=안전성 취약점 실증·공격)
- 한줄 역할: steering이 안전을 지키는 도구가 아니라 **깨는 도구**가 될 수 있음을 무작위 벡터 하나만으로 보여준 반증 논문 — 우리 개입(steering)에도 검증되지 않은 리스크가 있을 수 있다는 경고.

## 문제·동기

Activation steering은 fine-tuning보다 해석 가능하고 안전한 대안으로 포지셔닝되어 왔다(가중치를 안 건드리고 hidden state만 편집). 그러나 fine-tuning 쪽에서는 이미 "무해한 데이터로만 파인튜닝해도 안전장치가 깨진다"는 emergent misalignment 현상이 보고된 반면, steering의 안전성은 거의 검증되지 않았다. 기존 연구는 "의도적으로 harmful하게 설계된" jailbreak 벡터만 다뤘고, "선의로(benign) 만든 steering 개입이 의도치 않게 안전을 깨는가"라는 질문은 열려 있었다. 저자들은 이를 정면으로 검증한다: 정밀한 해석가능 제어(scalpel)가 실제로는 안전을 훼손하는 흉기(rogue scalpel)가 될 수 있다는 가설.

## 핵심 아이디어

Steering은 "해석 가능한 방향을 골라서 넣는다"는 정밀함의 이미지를 갖지만, 저자들은 그 정밀함이 안전을 보장하지 않음을 보인다. 핵심 주장 3단계:
1. 방향을 전혀 고르지 않고 **완전 무작위(random unit vector)로만 residual stream에 더해도** 거부 메커니즘이 깨진다 — "어떤 의미있는 방향을 골랐는가"가 안전 파괴의 필요조건이 아니다.
2. 안전하고 해석 가능한 제어의 표준 출처인 **SAE의 benign(무해한) feature**로 steering해도 random과 비슷하거나 더 나쁜 결과가 나온다 — 해석가능성이 안전 필터 역할을 못 한다.
3. 개별적으로는 약한(한 프롬프트만 뚫는) 무작위 벡터들을 **평균만 내도** unseen harmful 요청 전반에 일반화되는 universal attack이 만들어진다 — 국소적 취약점이 손쉽게 스케일업된다.

## 방법(무작위/임의 방향 steer 실험)

- Steering 식: x'^(l) = x^(l) + α·v, v는 unit-norm 방향, α = c·μ^(l) (해당 layer 평균 활성화 norm 기준 스케일링, c는 {0.25,...,2.0} 스윕).
- **Random Directions**: S^(d-1) 구면에서 균등 샘플. 모든 벡터 종류 실험에서 critical baseline 역할.
- **SAE-based Directions**: Goodfire의 Llama3.1-8B layer 19 SAE feature(디코더 컬럼 벡터)를 그대로 steering 벡터로 사용.
- Layer는 L/3(early)/L/2(middle, 기본)/2L/3(late) 3곳 스윕. 모든 토큰(prompt+generation)에 steer 적용, special token(bos 등)은 제외. Greedy decoding.
- 평가: JailbreakBench 100개 harmful prompt(10개 카테고리), baseline 무-steering compliance는 항상 0%. Qwen3-8B를 LLM-as-judge로 SAFE/UNSAFE 이진 판정(사람 검증 정밀도 94%). Compliance Rate(CR) = UNSAFE 판정 비율.
- 모델: Llama3.1(8B/70B), Qwen2.5(3B/7B/32B), Falcon3(3B/7B), Falcon-H1(34B).
- Universal attack 구성: 고정 harmful prompt(폭탄 제조 튜토리얼)에서 계수 스윕으로 최적 α 탐색 → random vector 1000개 중 해당 프롬프트를 뚫는 20개 선정 → 평균+unit norm 정규화 → 남은 99개 unseen 프롬프트에 평가. 모델 가중치/gradient/logit 불필요, 완전 black-box.

## 실험·결과(유해 순응률 0에서 몇 퍼센트, SAE benign feature도)

- **Random steering만으로 CR 0% → 1–13%** (모델·프롬프트 의존). Falcon3-7B는 c=0.75에서 peak 3%, Llama3.1-8B는 c=2.0에서 18%까지. 중간층(middle layer)에서 가장 취약, 후반층은 급격히 효과 감소 — "refusal direction을 단순 상쇄"하는 게 아니라 중간층의 비선형적 안전 형성 과정 자체를 교란한다는 해석(부록 E: harmful SAE feature와 known refusal direction의 코사인 유사도는 거의 0, 0.027±0.0208 — 단순 벡터 정렬 가설 기각).
- **SAE benign feature steering이 random보다 CR +1–4%p 높음**, 심지어 c=0.5~0.75의 작은 계수에서도 0.2~1% 비영(非零) compliance. 전체 JailbreakBench로 확장 시 overall CR: Llama3.1-8B random 4.6%, SAE 4.6%, Qwen2.5-7B random 3.8% — 10개 harm 카테고리 전부에서 비영 compliance(Fraud/Deception 최대 13%, Sexual/Adult content 최소 0.1%).
- **1000개 SAE feature 중 817개가 최소 1개 프롬프트를 jailbreak**, 353개는 5개 이상. 가장 강력한 feature조차 "modal auxiliary verb", "리스트 제공", "브랜드 아이덴티티" 같은 **완전히 무해한 의미**를 가짐 — 위험한 벡터와 정상 제어용 벡터가 겉보기로 구분 불가.
- **Cross-category 일반화는 낮음**(최다 feature도 100개 중 35개만 성공, heatmap상 baseline 근접) → 소수 프롬프트로 사전 스크리닝하는 안전장치가 사실상 불가능.
- **실제 프로덕션 API 사례**: Goodfire 공개 API의 benign "brand identity" feature로 Llama3.1-8B를 실제 jailbreak — disclaimer-then-compliance(경고문 뒤에 실제 유해 내용 제공), justification via fictional framing(허구적 설정으로 유해 콘텐츠 정당화) 두 실패 양상 관찰.
- **Universal attack**: 20개 무작위 벡터 평균만으로 random steering 대비 평균 4배 CR 증가(Llama3.1-70B는 random 대비 2배인 50.4%, Falcon3-7B는 5.7%→63.4%로 약 10배). 모델 의존성 있음(Qwen2.5-32B는 오히려 감소).

## §5(산업)에서의 위치("해석가능성=안전" 전제 도전)

산업 적용 파트에서 이 논문은 **"steering이 해석 가능하니 안전하게 배포 가능하다"는 전제를 직접 반증**하는 핵심 반례로 기능한다. Circuit Breakers/RepE 같은 논문들이 "steering·representation engineering을 안전 제품화할 수 있다"는 낙관적 방향이라면, 이 논문은 그 반대편에서 "동일한 기법(activation steering)이 정확히 그 안전을 무너뜨리는 벡터로도 작동한다"는 대칭적 위험을 보여준다. 특히 Goodfire처럼 **실제 상용 API**로 benign feature를 노출하는 산업 관행 자체(§4.3 사례)가 공격 표면이 될 수 있음을 실증했다는 점에서, "안전 제품으로서의 steering"을 논하는 §5 산업 파트에 반드시 들어가야 할 카운터-내러티브다. 결론부 제안(적대적 훈련으로 steering 교란에 대응)은 아직 미해결 과제로 남아 있다.

## 우리 프로젝트 연결(개입 리스크·검증)

- 우리 메인 method는 `C_steer = C_success ∧ ¬C_failure` 형태의 **multi-dim contrastive operator**로, 이 논문의 random/SAE-benign 벡터보다 훨씬 더 목표 지향적으로 fit된 개입이다. 그러나 이 논문의 핵심 교훈은 "의미 있어 보이는(성공 방향으로 정렬된) 개입도 의도치 않은 부작용을 낼 수 있다"는 것 — 우리 conceptor가 "성공 쪽으로 밀지만 실제로는 다른(관련 없는) latent 서브스페이스를 건드려 예상 못한 행동 변화를 유발할" 가능성을 배제할 수 없다는 방법론적 경고로 읽어야 한다.
- 이 논문의 "middle layer가 가장 취약"이라는 관찰은 우리가 hook을 거는 지점(action_head DiT block, VL-SA 등)의 layer 선택이 실제로 어떤 latent 안정성을 갖는지 검증해야 할 필요성을 시사 — steering coefficient(=우리의 steering 강도 hyperparameter)에 대한 sweep 없이 강하게 밀면 "성공 확률은 올라가지만 다른 실패 모드(예: 엉뚱한 물체 조작, 다른 task로의 행동 변질)를 유발"할 위험이 구조적으로 존재.
- **검증 시사점**: ΔSR만 보고 성공으로 판단하지 말고, steering 후 행동이 (a) 실제로 의도한 phase/pathway 신호에 대응해 바뀌는지, (b) 그 외 무관한 행동 축(예: 안전하지 않은 동작, 물체 손상 등 로봇 맥락의 "harm")을 유발하지 않는지 별도로 체크할 필요 — LLM의 harmful compliance에 대응하는 로봇 쪽 "의도치 않은 부작용" 축을 우리 평가 프로토콜에 추가할지 검토 가치.
- Random steering만으로도 비영 효과가 난다는 결과는, 우리 실험에서 conceptor steering의 ΔSR이 양의 값을 보일 때 이것이 정말 "성공 서브스페이스로의 유의미한 이동" 때문인지 아니면 "임의 방향 섭동만으로도 생기는 배경 효과"인지 구분할 필요를 제기한다 — **random-direction steering을 negative control로 항상 병행 측정**해야 한다는 실무적 시사점(현재 우리 ablation 사다리에 명시적으로 없다면 추가 검토 필요).

## 면접 포인트(Q→A)

**Q1. 왜 무작위 벡터로도 안전장치가 깨지는가? 의미 있는 방향을 고른 것도 아닌데?**
A. 저자들은 middle layer에서 abstract concept·refusal policy가 형성되는 비선형 과정 자체가 작은 섭동에도 취약하다고 본다. Harmful SAE feature와 known refusal direction의 코사인 유사도가 거의 0(부록 E)이라는 결과가 "steering이 refusal 벡터를 직접 상쇄하는 것"이 아니라는 것을 보여준다 — 특정 방향을 정확히 겨냥하지 않아도, 그 layer의 표현 구조 자체가 어떤 방향의 섭동에도 깨지기 쉬운 상태라는 것.

**Q2. SAE feature가 "해석 가능하다"는 게 왜 안전을 보장 못 하는가?**
A. 가장 위험한 feature들의 사전 정의된 해석이 "modal 조동사", "브랜드 아이덴티티" 등 완전히 무해한 개념이었다(Fig 5). 즉 feature의 semantic label이 안전과 무관하더라도 실제 causal effect는 안전 메커니즘을 깰 수 있다 — 해석(사람이 붙인 라벨)과 인과 효과(모델 행동에 미치는 실제 영향)가 분리될 수 있다는 것이 핵심. 따라서 "라벨을 보고 안전한 feature만 배포하면 된다"는 필터링 전략은 원천적으로 성립하지 않는다.

**Q3. 이 결과가 우리 프로젝트(VLA steering)에 주는 실질적 함의는?**
A. 우리 conceptor가 성공 latent 쪽으로 정교하게 fit되었더라도, 그 개입이 실제로 의도한 축(task 성공)만 바꾸는지 아니면 다른 축(안전하지 않은 동작 등)에도 영향을 주는지는 별도로 검증해야 한다는 것. 또 random-direction negative control 없이 ΔSR 증가만으로 "steering이 유효하다"고 결론 내리면, 그 효과가 정말 conceptor 방향 때문인지 임의 섭동의 배경 효과인지 구분할 수 없다.

## 한계·비판

- **LLM 텍스트 도메인·안전(harmful compliance) 지표 특화**: 로봇 continuous action, diffusion action head(DiT) 등 우리 도메인으로의 직접 이전 가능성은 다루지 않음. "harmful"이라는 이진 판정 지표가 로봇 성공/실패 맥락에는 그대로 대응되지 않는다.
- **메커니즘 설명 부족**: 왜 middle layer가 가장 취약한지, 왜 특정 SAE feature가 더 위험한지에 대한 인과적 설명은 상관관계 수준(layer 위치, cosine 유사도 부재)에 그치고, 실제 회로/메커니즘 분석은 future work로 남김.
- **완화책은 제안 수준**: "adversarial training으로 steering 섭동에 대응"이라는 결론부 제안은 구체적 구현이나 검증 없이 방향성만 제시.
- **모델·계수 의존성 큼**: Universal attack 효과가 Qwen2.5-32B에서는 오히려 감소하는 등, 결과의 모델 전이 일반화 폭이 논문 내에서도 일관되지 않음 — "이 정도로 위험하다"는 정량치(1–13%, 4배 등)를 다른 아키텍처·모달리티에 그대로 외삽하기는 어려움.
- **판정 자체의 신뢰도**: LLM-as-judge 정밀도 94%(사람 대비)로 상당히 높지만 100% 아님 — CR 수치(특히 0.1%대 낮은 값)의 통계적 노이즈 여지 존재.
