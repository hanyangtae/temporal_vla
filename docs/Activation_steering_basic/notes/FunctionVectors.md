# Function Vectors in Large Language Models (Todd et al. 2023, ICLR 2024)

- 출처: arXiv 2310.15213v2 (ICLR 2024) · Northeastern University (Todd, Li, Sen Sharma, Mueller, Wallace, Bau)
- PDF: docs/Activation_steering_basic/FunctionVectors_2310.15213.pdf
- 정독 섹션: §2 Method (2.1 동기 관찰, 2.2 formulation, 2.3 causal mediation으로 FV 추출) — 배정 tag "§3 방법"에 대응하는 실제 논문 섹션
- tier: must
- 한줄 역할: ICL로 시연된 task 자체가 소수 attention head 출력의 합인 단일 벡터(function vector, FV)로 압축돼 있음을 causal mediation analysis로 입증 — population-level 대조통계 기반 steering vector(RepE/CAA)와 달리 **회로(특정 head) 매개 + "벡터 하나=기능 하나"** task-vector 계열의 대표 논문.

## 문제·동기
ICL(in-context learning)이 왜/어떻게 동작하는지는 "복사 행동(induction head)"이나 이론적 모델링 관점에서만 다뤄졌고, "시연된 task 자체가 모델 내부에 하나의 표현으로 존재하는가"는 불명확했다. 저자들은 상관이 아니라 **인과**를 원한다: 어떤 hidden state/구성요소가 task를 안다고 "말할 수 있는" 것을 넘어, 그것을 다른 context에 이식했을 때 실제로 해당 task 수행을 유발하는지를 검증하고자 한다.

## 핵심 아이디어
1. §2.1 동기 관찰: ICL prompt 마지막 토큰의 평균 활성화(레이어 평균, layer-average h̄)를 zero-shot context에 더하기만 해도 task가 어느 정도 유발된다(예: antonym 24.3%). 그러나 이는 레이어 전체의 뭉뚱그린 평균이라 약하다.
2. causal mediation analysis(activation patching)로 ICL 전체 attention head 중 소수(GPT-J 10개)가 task 정보 이동에 특권적으로 관여함을 식별 — 대부분 초-중간층에 몰려있고, 예시의 출력(label) 토큰에 강하게 attend.
3. 이 소수 head들의 task-conditioned 평균 출력을 합산한 것이 **function vector(FV)** — 레이어 평균보다 훨씬 강한 causal 효과(zero-shot 57.5% vs 레이어평균 9.5%, GPT-J).
4. FV는 context 형태에 강인(portable): 다른 ICL 템플릿·zero-shot·자연어 문장에 넣어도 task를 유발.
5. FV의 decoded vocabulary(logit lens)는 종종 출력 공간과 일치하지만 top-k 토큰만으로 FV를 재구성하면 원래 성능에 크게 못 미침 — "단어 목록"만으로는 FV를 설명 못함, 추가 정보를 담고 있음.
6. task 간 벡터 산술(합성)도 부분적으로 성립 — 단, word-embedding 산술과는 다른 "기능 공간"의 산술.

## 방법 (ICL task를 causal vector로 압축, attention head 매개, zero-shot 이식)
- **정식화**: task t마다 ICL prompt 집합 P_t(정답을 맞히는 prompt만 사용) + 라벨을 뒤섞은 uninformative prompt 집합 P̃_t(레이블 무관화)를 둔다.
- **causal mediation (Eq.2~4)**: 각 attention head a_ℓj의 task-conditioned 평균 출력 ā^t_ℓj를 P_t에서 계산 → 이를 P̃_t(정답 못 맞히는 손상된 prompt)에서 해당 head 활성화 자리에 patch(주입) → 정답 토큰 확률 증가량을 causal indirect effect(CIE)로 측정 → task 전체·prompt 전체 평균이 average indirect effect(AIE). AIE 상위 head 집합 A(GPT-J는 10개, 모델 크기에 비례해 확장: Llama2-7B 20개, 70B 100개)를 causal head로 채택.
- **FV 구성 (Eq.5)**: v_t = Σ_{a_ℓj∈A} ā^t_ℓj — task별로 고정된 소수 head들의 평균 출력을 그냥 더한 것. head 선택 A는 task 무관(전 task 평균 AIE로 한 번 정함), v_t만 task별로 다름.
- **개입(적용)**: 하나의 특정 레이어 ℓ(대략 |L|/3, GPT-J는 layer 9)에서 target prompt의 **마지막 토큰 위치**에 v_t를 더함(additive), 이후 생성되는 모든 후속 토큰 예측에도 이 개입을 유지. 별도 스케일 계수 없이(원 논문은 α 스윕을 명시하지 않고 raw v_t를 그대로 더함) 단일 레이어·단일 위치 삽입이 핵심.
- 대조: §2.1의 "레이어 평균 h̄^t_ℓ"은 전체 hidden state를 뭉뚱그린 population 평균(비-causal 선별), FV는 causal mediation으로 **먼저 정보를 옮기는 회로(소수 head)를 골라낸 뒤** 그 head들의 출력만 합산 — steering vector 문헌에서 흔한 "모든 activation의 평균 차이" 방식과 달리 **회로 식별이 선행**되는 것이 방법론적 차별점.

## 실험·결과
- **모델**: GPT-J(6B), GPT-NeoX(20B), Llama2(7B/13B/70B). **task**: 6개 대표 task(antonym, capitalize, country-capital, english-french, present-past, singular-plural) + 부록 34개 추가 task.
- **Table 2**: GPT-J baseline(uninformative shuffled-label) 39.1% → 레이어평균 79.5% → **FV 90.8%**; zero-shot에서는 baseline 5.5% → 레이어평균 9.5% → **FV 57.5%**(가장 극적인 gap). Llama2-70B는 zero-shot baseline 8.2% → FV 83.8%.
- **Fig.4**: 레이어별 스윕에서 초-중간층에 FV를 넣을 때 정확도가 급상승하고 후반층에서 급감 — FV가 단순 선형 가산이 아니라 **후속 비선형 연산을 트리거**한다는 근거.
- **Table 3**: 자연어 문장(ICL 템플릿과 무관한 문장 5종)에 antonym FV를 넣으면 0~2.7% → 46~68%로 상승 — ICL과 무관한 문맥에도 portable.
- **Table 5~6**: FV를 직접 vocabulary로 decoding하면 대부분 task의 출력공간과 일치(예: singular-plural FV → 복수형 단어들)하지만, top-100/전체 vocab 분포만 맞춰 재구성한 벡터(v̂_t)는 원 FV 성능에 크게 못 미침(예: antonym 48.2% vs top100 재구성 4.8%) — FV는 단순 "출력 토큰의 사전 지식"이 아니라 추가 절차적 정보를 담음.
- **Table 7**: v*_BD = v_AD + v_BC − v_AC 식 벡터 합성이 일부 task(Last-Country-Capital 0.60 vs FV 0.15, ICL 0.32)에서 원 FV·심지어 ICL보다 우수 — 기능 공간의 산술이 부분적으로 성립.

## activation-steering 흐름에서의 위치 (task vector 계열)
FV는 "steering vector"라는 명칭 자체를 쓰지 않지만, activation에 벡터를 더해 행동을 유발한다는 점에서 이 계열(ActAdd/CAA/RepE)과 형식적으로 동일한 additive intervention이다. 다만 계보상 위치가 다르다: RepE/CAA/ActAdd는 **대조 프롬프트 쌍의 population-level 평균 차이**로 방향을 뽑는 top-down 통계 접근인 반면, FV는 **causal mediation analysis(activation patching)로 먼저 회로(소수 attention head)를 식별**한 뒤 그 head 출력만 합산하는 bottom-up 회로-매개 접근이다. 또한 이름은 다르지만 개념적으로 "벡터 하나가 기능/task 하나를 인코딩한다"는 프레임을 공유하는 동시대 계열이 **task vector**(Ilharco et al. 2023, 파라미터 공간에서 fine-tuned − pretrained 가중치 차이)이며, 논문 §4에서 명시적으로 구분한다(task vector=weight space, FV=activation space). 후속 문헌에서 FV는 "in-context task → 재사용 가능한 단일 latent 벡터로 증류"라는 아이디어의 대표 사례로 인용되며, COAST류 multi-dim conceptor(우리 project가 쓰는 도구)의 "1차원 additive vector"에 대비되는 baseline 개념으로도 종종 언급된다.

## 우리 프로젝트 연결
- FV의 "벡터 하나 = 기능(task) 하나" 관점은 우리가 잠정적으로 쓰는 **VL pathway = goal("what") 방향** 개념과 표면적으로 유사해 보이지만 구조가 다르다: FV는 (a) 이산 vocabulary 위 discrete input-output 함수(antonym, translation 등)를 대상으로, (b) causal mediation으로 미리 선별한 소수 attention head의 출력만 더하는 **sparse 회로 개입**인 반면, 우리 VL 방향은 (a) continuous goal 표현(비-vocabulary)을 대상으로, (b) population-level 대조통계(succ/fail 분포 차이, conceptor)로 뽑은 **dense hidden-state subspace** 개입이다. "goal 방향 하나"라는 서술은 은유이지 FV처럼 회로적으로 검증된 것은 아님 — 이 차이를 면접에서 명확히 구분해야 한다.
- 방법론적으로 유용한 대비점은 **회로 식별의 유무**다: FV는 causal mediation(activation patching)으로 "이 정보가 어느 head를 거쳐 전달되는가"를 먼저 특정한다 — 우리의 온라인 phase/pathway 식별 난제(VL-SA vs DiT block 어디서 실패가 매개되는가)에 방법론적으로 참고할 수 있는 절차(head/block 단위 CIE·AIE 스윕)이나, 우리는 attention head가 아니라 pathway 전체(VL-SA 출력, DiT block residual) 단위로 유사한 patching 실험을 설계할 수 있다는 시사점.
- FV의 "레이어 |L|/3 부근에서 강함, 후반층에서 급감"이라는 실험 결과는 우리 DiT 실패신호가 특정 시간대(t≥12)·특정 block에서만 강하다는 관찰과 구조적으로 유사한 패턴(steering이 만능이 아니라 위치-민감적)이라는 점에서 참고할 만하다.
- FV 벡터 합성(Eq.7~8, "First-Copy+Last-Capital−First-Capital=Last-Copy")은 우리 conceptor의 C_success ∧ ¬C_failure 형태(합·차 연산으로 새 방향 구성)와 연산 형식이 닮았지만, FV는 단일벡터 산술이고 conceptor는 multi-dim 연산자(projector/subspace)라는 점에서 표현력의 차수가 다르다.

## 면접 포인트 (Q→A)
1. Q: "Function vector와 steering vector는 같은 건가?" A: "형식(activation에 벡터를 더해 행동을 바꾼다)은 같지만 도출 절차가 다르다. Steering vector(RepE/CAA/ActAdd)는 대조 프롬프트 쌍의 population-level 평균 차이로 뽑는 top-down 통계 방향인 반면, function vector는 causal mediation analysis(activation patching)로 먼저 '어느 attention head가 정보를 옮기는가'를 인과적으로 특정한 뒤 그 head들의 출력만 합산하는 bottom-up 회로-매개 벡터다. FV가 더 sparse하고 mechanistic하다."
2. Q: "Function vector와 task vector(Ilharco et al.)는 뭐가 다른가?" A: "이름은 비슷하지만 공간이 다르다. Task vector는 파라미터 공간에서 fine-tuned 모델과 base 모델의 **가중치 차이**로 정의되는 반면, function vector는 **activation(hidden state) 공간**에서 특정 attention head 출력을 합산한 것이다. 하나는 weight-space, 하나는 activation-space intervention."
3. Q(우리 프로젝트 관점): "VLA에서도 'goal을 인코딩하는 벡터 하나'가 있다고 볼 수 있나?" A: "FV 논문 수준의 근거(causal mediation으로 특정 head 집합을 식별하고, 그 head 출력만으로 zero-shot task 수행을 유발)는 우리에게 없다. 우리 VL 방향은 population-level 대조통계에서 뽑은 dense subspace이지, 회로적으로 검증된 sparse 벡터가 아니다. '벡터 하나=기능 하나'라는 서술은 은유 수준으로만 쓰고, FV식 회로 identification(head/block 단위 patching)을 온라인 pathway 식별 실험에 참고하는 것이 더 정직한 연결이다."

## 한계·비판
- **discrete, 짧은 단일 스텝 task 한정**: antonym, 번역, 대소문자 변환 등 한 토큰 입력→한 토큰(또는 짧은) 출력 함수에 최적화된 실험 설계. 여러 스텝에 걸친 절차적 task(로봇 조작 같은 continuous multi-step control)로의 일반화는 검증되지 않음 — 우리 VLA 맥락에 직접 이식하기엔 태스크 성격이 너무 다르다.
- **head 선택이 전 task 평균으로 고정**: causal head 집합 A는 task 전체 평균 AIE로 한 번 정하고 그 후 모든 task에 재사용 — task별로 다른 회로를 쓸 가능성을 배제한 단순화이며, 이질적인 task 집합(우리처럼 서로 다른 실패 유형)에서는 "공용 head 집합"이 최적이 아닐 수 있음.
- **후반층 급감의 원인 미해명**: Fig.4에서 후반층에 FV를 넣으면 효과가 급격히 사라지는 이유를 "비선형 트리거"라고만 서술하고 메커니즘을 규명하지 않음 — "어느 위치에 steer해야 하는가"라는 우리의 phase-matched 문제의식과 맞닿아 있지만 이 논문은 답을 주지 않는다.
- **decoded vocabulary의 불완전성**(Table 6): FV가 담은 정보 중 상당 부분이 top-k vocabulary로 설명되지 않음을 스스로 인정 — FV가 "무엇을" 인코딩하는지 완전히 해석되지 않은 채로 causal 효과만 확인된 상태(해석가능성 미완).
- **online/실시간 문제의식 부재**: 모든 실험이 정적 오프라인 ICL prompt에서 FV를 뽑고 별도 프롬프트에 삽입하는 구조 — 우리의 핵심 난제(rollout 진행 중 phase/실패유형을 온라인으로 식별해 steering을 라우팅)와 같은 시간축 문제는 다루지 않는다.
