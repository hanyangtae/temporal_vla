# Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (Wang, Variengien, Conmy, Shlegeris, Steinhardt 2022)

- 출처: ICLR 2023 (Redwood Research). arXiv:2211.00593 (v1, 2022-11-01)
- PDF: `docs/Activation_steering_basic/IOI_2211.00593.pdf`
- 정독 섹션: §2 Background(circuit·knockout 정의), §3 Discovering the Circuit(path patching 실전 적용, 핵심 배정 범위), Appendix B(path patching 알고리즘 정식화)
- tier: must
- 한줄역할: activation patching을 head 단위 "직접효과 vs 간접효과" 분리가 가능한 **path patching**으로 일반화하고, 이를 반복 적용해 GPT-2 small의 자연어 태스크(IOI) 전체를 26개 attention head·7개 클래스로 구성된 회로(circuit)로 역공학한 최초의 대규모 end-to-end 사례. steering의 전제인 "어느 head/방향을 건드려야 하는가"를 찾는 causal localization 방법론의 정점.

## 문제·동기

Mechanistic interpretability는 그때까지 (a) 작은 모델의 단순 행동(예: modular addition grokking) 또는 (b) 큰 모델의 복잡 행동을 "대략적으로"(broad strokes, 예: attention head 클러스터링) 설명하는 두 극단에 머물러 있었음. "자연어 상의 실제(in the wild) 행동을 큰 모델에서 세밀하게(head 단위까지) 끝까지 설명"한 사례가 없었다는 것이 이 논문의 공백. 저자들은 GPT-2 small(117M)에서 indirect object identification(IOI, "When Mary and John went to the store, John gave a drink to ___" → "Mary")이라는 언어학적으로 명확하고 해석 가능한 알고리즘("문장에 등장한 이름 중 주어로 중복되지 않은 이름을 출력")을 갖는 태스크를 골라, 이를 구현하는 회로를 처음부터 끝까지(logit에서 시작해 입력 쪽으로) 역공학한다.

## 핵심 아이디어

1. 모델을 계산 그래프 M(노드=neuron/attention head/embedding, 엣지=residual/attention/projection 상호작용)으로 보고, 특정 행동을 담당하는 부분그래프를 **circuit** C로 정의한다.
2. Circuit의 행동은 **knockout**(M\C의 노드를 특정 reference 분포 pABC 위에서 **mean-ablation**)으로 정의되는 함수 C(x)로 포착한다(zero-ablation은 암묵적 bias를 깨서 noisy).
3. Logit에서 출발해 "이 head가 직접 logit에 영향을 주는가?"를 반복적으로 역추적한다. 이때 간접효과(다른 head를 매개)와 직접효과를 분리하기 위해 **path patching**이라는 새 기법을 도입한다 — activation patching(Vig 2020, ROME 계열)이 "노드 하나 전체를 patch"하는 것과 달리, path patching은 "sender head h → receiver 집합 R로 가는 특정 경로(직접 residual/MLP 경로, 다른 attention head를 거치지 않는 경로)"만 선택적으로 patch한다.
4. 이 과정을 logits → Name Mover 후보 → 그 query에 영향 주는 head(S-Inhibition) → 그 value에 영향 주는 head(Duplicate Token / Induction) 순으로 역방향 반복해 7개 head 클래스(26개 head)로 구성된 회로를 구성한다.
5. 회로를 **faithfulness / completeness / minimality** 3개 정량 기준으로 검증하고, 그 과정에서 "Name Mover를 knock out해도 Backup Name Mover가 대신 역할을 수행"하는 redundancy(모델 구조가 개입에 따라 달라짐)를 발견한다.

## 방법(메커니즘)

### Circuit·knockout 정식화(§2.1)

- Circuit C ⊆ M은 행동을 담당하는 부분그래프. C(x)는 M\C의 모든 노드를 knockout하고 나온 logit으로 정의.
- Knockout = **mean ablation**: 노드를 reference 분포 위 평균 활성값으로 치환(0으로 치환하는 zero-ablation은 이후 노드가 암묵적으로 기대하는 bias를 깨뜨려 noisy).
- Reference 분포는 pIOI가 아니라 **pABC**(IO/S 두 이름 대신 무관한 이름 A/B/C 셋을 쓴 변형) — pIOI로 평균내면 "이름이 중복됐다"같은 IOI 태스크에 필요한 정보까지 상수로 남아 knockout이 불완전해짐. 동일 template 내에서만 평균내 문법 정보(주어/동사 등 위치별 역할)는 보존.

### Path patching(§3.1 도입, Appendix B 정식화) — 배정 핵심

목적: sender head h가 receiver 집합 R(다른 head의 key/query/value, 혹은 residual stream 최종 상태 = logit 직전)에 **직접**(다른 attention head를 거치지 않는 경로로) 미치는 인과효과만 분리 측정.

4단계 forward pass(원문 Figure 11, Algorithm 1):
1. xorig, xnew 각각에서 전체 활성화 수집(clean/corrupt 두 세트, ROME의 clean/corrupted run과 유사).
2. **forward pass C**: h를 제외한 모든 head 출력을 xorig 값으로 고정(freeze)하고, h만 xnew 값으로 patch한 채 forward. MLP·layer norm은 재계산. 이러면 h→p→r(p=다른 attention head 경유)로 가는 간접경로는 전부 차단되고, h→(residual/MLP만 거치는)→R 직접경로의 영향만 살아남는다.
3. 이 forward pass C에서 R의 활성값(사실상 재계산되지만 다른 곳에서는 덮어써지는 값)을 저장.
4. **forward pass D**: xorig 위에서 정상 forward를 돌리되, R의 값만 3단계에서 저장한 값으로 patch. 이때 R 이후 layer는 정상적으로 재계산됨.
5. D의 logit 결과(주로 logit difference)를 M(정상)의 결과와 비교 = h→R 경로의 causal 중요도.

핵심 차이(ROME/일반 activation patching 대비): 일반 activation patching은 "노드 하나를 통째로 patch"해 그 노드 이후의 **모든** 하류 경로(다른 head를 거치는 경로 포함)가 다 바뀐다. Path patching은 다른 head들을 xorig로 **freeze**함으로써 h→p→r 같은 간접경로를 인위적으로 차단하고, h→R로 가는 특정 경로 집합만 격리한다 — "직접효과 vs 간접효과"를 분리하는 causal mediation analysis(Pearl)의 head-단위 정밀화.

### 회로 발견 절차(§3, 역방향 반복)

logit → 입력 방향으로 4단계 반복:
1. **h → Logits 직접 path patching**: 9.9, 9.6, 10.0이 patch 시 logit diff 큰 폭 하락(양의 기여) → **Name Mover Heads**. 10.7, 11.10은 patch 시 logit diff 증가(음의 기여) → **Negative Name Mover Heads**. 검증: attention prob(IO에 평균 0.59) vs 이름 방향 logit 투영 상관 ρ>0.81; OV matrix로 "attend한 이름을 그대로 copy"하는지 **copy score**(top-5 logit에 원 토큰 포함 비율) 측정 — Name Mover 95%+ vs 평균 head 20% 미만.
2. **h → Name Mover Heads' query 직접 path patching**: 7.3, 7.9, 8.6, 8.10 발견 → **S-Inhibition Heads**. 이 4개를 patch하면 Name Mover의 attention이 IO에서 S1 쪽으로 이동 확인(§3.2 Figure 4c).
3. **h → S-Inhibition Heads' value 직접 path patching**: 두 그룹 발견 — S2→S1 attend하는 **Duplicate Token Heads**(0.1, 3.0), S2→S1+1 attend하는 **Induction Heads**(5.5, 5.9/6.9)로, 후자는 **Previous Token Heads**(2.2, 4.11)와 key composition으로 짝을 이룸(A B ... A 패턴 인식).
4. **token/position 신호 분리 실험**(Appendix A, 일반 activation patching 사용): S-Inhibition head 출력을 6종 counterfactual 데이터셋(이름 무작위 치환/IO↔S1 위치 뒤집기/IO↔S2 역할 치환 조합)에서 patch해, logit diff ≈ 2.31·S_pos + 0.99·S_tok로 분해 — **position 신호가 token 신호보다 크게 기여**함을 정량 확인.
5. **Backup Name Mover Heads 발견(§3.4)**: Name Mover 전체를 knockout했더니 logit diff가 5%만 하락(예상 밖 견고함). knockout 후 같은 path-patching 절차를 재실행하니 새로운 8개 head(9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9)가 direct effect를 대신 떠맡음 → **"knockout하면 다른 구조가 나타난다"** = 회로 탐색 자체가 개입 상태에 의존한다는 방법론적 함정을 스스로 노출.

### 정량 검증 3기준(§4)

- **Faithfulness**: |F(M)−F(C)| = 0.46 (F(M)=3.56의 13% 손실, C가 87% 성능 재현).
- **Completeness**: 임의 K⊆C에 대해 |F(C\K)−F(M\K)|(incompleteness score)가 작아야 함. 무작위/클래스 단위 K로는 회로가 complete해 보이지만, **greedy search**로 K를 찾으면 incompleteness score가 최대 3.09(87%)까지 커짐 — 즉 회로 밖에 아직 해석 못 한 보완 구조가 존재(Backup 현상의 연장선).
- **Minimality**: 각 노드 v에 대해 |F(C\(K∪{v}))−F(C\K)|가 큰 K가 존재해야 함(불필요한 노드 없음) — 전 노드가 최소 1% 이상 기여, 통과.
- 대조군(naive circuit: Backup·Negative 제외)도 faithfulness는 비슷하지만 completeness에서 훨씬 쉽게 무너짐 → completeness가 faithfulness보다 엄격한 기준임을 실증.

### Adversarial example(§4.4)

회로 이해(중복 토큰 검출에 의존)를 이용해 IO도 중복 등장하는 문장("John and Mary went... Mary had a good day. John gave a bottle of milk to ___")을 구성 → S가 IO보다 높은 logit을 받는 비율이 0.7%→23.4%로 급증. Circuit 이해가 실제로 예측 가능한 실패 모드를 찾아내는 데 쓰일 수 있음을 보인 downstream 활용 사례(단 저자도 "회로 없이도 발견 가능했을 정도로 단순하다"고 자평).

## 실험·결과

- 26개 head(전체 head·position 조합의 1.1%)로 태스크 대부분을 설명, 7개 기능 클래스(Duplicate Token / Previous Token / Induction / S-Inhibition / Name Mover / Negative Name Mover / Backup Name Mover)로 정리(Figure 2).
- GPT-2 medium에 대한 예비 분석에서도 sparse한 direct-effect head 집합은 존재하지만 IO/S에만 집중하지 않는 더 복잡한 패턴 관찰 — scale-up 시 반복 가능성은 불확실.
- MLP를 개별로 knockout하면(1층 제외) 성능 유지되지만, 1층 이후 MLP를 **모두** 동시 knockout하면 태스크 수행 불가(Appendix J) — MLP 상호 의존성은 이 논문 범위 밖(attention head만 분석).

## activation-steering 흐름 위치

ROME의 causal tracing이 "corrupt→단일 노드 복원→출력 회복량"으로 **위치**(어느 layer/token)를 찾는 데 그쳤다면, IOI는 이를 **head 단위·경로 단위**(sender→receiver 직접효과)로 정밀화하고, 이를 반복 적용해 로짓에서부터 입력까지 완전한 인과 그래프(회로)를 재구성했다는 점에서 activation patching 계보의 방법론적 정점 중 하나. "attend하는 대상을 그대로 copy하는 head"(Name Mover)를 찾아낸 것은, 이후 activation steering 문헌에서 "특정 head/방향의 activation에 벡터를 더하면 해당 기능이 증폭/억제된다"는 개입(steering)이 성립하려면 먼저 "그 head가 정확히 무엇을 copy/route하는지"를 회로 수준에서 알아야 한다는 전제를 제공. 또한 Negative Name Mover(의도적으로 반대 방향 기여)·Backup Name Mover(knockout 시에만 등장하는 보상 구조) 발견은, steering 개입이 "한 방향을 누르면 다른 head가 대신 그 역할을 떠맡는" 회로 수준 재편을 유발할 수 있다는 경고로 이후 steering 강건성 논의(예: ITI, CAA의 다층 개입)에 영향.

## 우리 프로젝트 연결

- **pathway 귀인 방법론의 직접 참고**: 우리의 "VL(goal) vs DiT(motor) 중 어디가 실패에 기여하는가"는 이 논문의 "어느 head가 logit에 직접 기여하는가"와 동형 문제. Path patching의 "sender를 patch하되 다른 경로는 freeze해서 간접효과를 차단"하는 아이디어는, 우리 Eagle→VL-SA→DiT **직렬 구조**에서 "VL이 DiT를 거쳐 미치는 간접효과"와 "VL이 (있다면) 직접 경로로 미치는 효과"를 분리하고 싶을 때 그대로 적용 가능한 틀 — 다만 우리 구조는 IOI의 다중 병렬 attention head grid보다 훨씬 얕고 직렬적이라 "direct vs indirect" 구분 자체의 실익이 더 제한적일 수 있음(CLAUDE.md에 명시된 "downstream 결합 고려" 주의사항과 정확히 대응).
- **Backup Name Mover = 우리의 pathway 재편 위험**: Name Mover를 knockout하니 Backup이 대신 역할을 맡는 현상은, 우리가 VL 또는 DiT pathway 하나를 steer(개입)했을 때 다른 pathway가 보상적으로 재편되어 "steer 전 관찰한 회로/신호"가 "steer 후에는 다른 경로로 흐를 수 있다"는 시사점을 준다 — phase-matched steering을 online에서 검증할 때, 개입 자체가 phase/pathway 식별 신호를 바꿀 수 있음을 염두에 둘 필요.
- **Completeness 기준 = 우리 conceptor의 "빠진 성분" 위험**: greedy search로 찾은 incompleteness set이 기존 7클래스 분류에 안 들어맞는 head 조합이었다는 결과는, 우리 succ/fail conceptor(C_success ∧ ¬C_failure)가 관찰된 두 클래스 분포만으로 fit되어 "관찰 안 된 실패 경로/보상 경로"를 놓칠 수 있다는 유사한 함정을 시사 — steering 후 ΔSR을 관찰(faithfulness급 검증)하는 것만으로는 부족하고, 우리도 일종의 completeness/minimality 감각(steering을 부분적으로만 걸었을 때 성능이 어떻게 깨지는지)을 갖고 검증할 필요.
- **Position vs token 신호 분리(Appendix A) = phase 신호와 content 신호 분리**: S-Inhibition head 출력이 "무엇이 중복됐는가(token)"와 "어디서 중복됐는가(position)"로 분해 가능하고 position이 더 강하게 기여한다는 결과는, 우리의 phase-matched steering에서 "이 시점에 무엇을 하고 있는가(phase/position)"와 "무엇이 실패했는가(failure type/content)" 신호를 유사하게 분리해서 다뤄야 한다는 아이디어와 대응 — 우리의 핵심 미해결 문제(online phase 식별)에 대한 방법론적 힌트.

## 면접 포인트 (Q→A)

**Q1. Path patching이 일반 activation patching과 정확히 어떻게 다른가?**
A. 일반 activation patching(ROME 등)은 노드 하나를 통째로 patch해 그 하류의 모든 경로(다른 attention head를 거치는 경로 포함)가 다 영향을 받는다. Path patching은 sender head h만 xnew로 patch하고 **h를 제외한 다른 모든 head는 xorig로 freeze**한 상태에서 forward를 돌려, h→(다른 head를 거치지 않는 직접 경로: residual/MLP만 통과)→receiver R로 가는 경로의 영향만 골라 R에 다시 patch한다. 즉 간접효과(다른 head 매개)를 인위적으로 차단해 direct effect만 분리 측정하는 기법이다.

**Q2. 이 논문이 발견한 7개 head 클래스를 기능적으로 요약하면?**
A. 3단계 알고리즘("이름 나열→중복 제거→나머지 출력")에 대응. Duplicate Token Heads/Induction Heads(+Previous Token Heads)가 "어느 이름이 중복됐는지" 위치 정보를 계산, S-Inhibition Heads가 그 정보를 Name Mover Heads의 query에 주입해 중복(S) 위치에 대한 attention을 억제, Name Mover Heads가 남은 이름(IO)에 attend해 copy. 여기에 반대로 기여하는 Negative Name Mover(hedging 가설), knockout 시에만 등장하는 Backup Name Mover가 추가된다.

**Q3. Faithfulness만으로 회로 설명이 충분하지 않은 이유는? (completeness 도입 동기)**
A. Faithfulness는 "회로가 전체 모델과 비슷한 성능을 낸다"만 확인한다. 만약 모델이 병렬·중복인 두 서브회로 C1, C2를 OR로 결합해 태스크를 수행한다면, C1만 찾아도 faithful하지만 실제로는 C2도 쓰이고 있으므로 불완전한 설명이다. 이 논문의 Backup Name Mover가 실제 사례 — Name Mover만으로 faithful했지만, knockout하니 Backup이 드러나 원래 설명이 불완전했음이 밝혀졌다. Completeness는 임의의 부분집합 K를 회로와 전체 모델에서 동시에 제거했을 때 성능 차이(|F(C\K)−F(M\K)|)가 항상 작아야 한다는, knockout에 대해 강건한 기준이다.

**Q4. Copy score는 무엇을 검증하는 실험인가?**
A. Name Mover Head가 "attend한 이름을 실제로 output에 복사한다"는 가설을 attention pattern 관찰만으로는 검증할 수 없다(attention이 높다고 그 정보를 쓴다는 보장은 없음, Jain & Wallace 2019 비판과 동일 문제의식). 그래서 이름 토큰의 residual state에 head의 OV matrix를 직접 곱하고(마치 완벽히 attend한 것처럼 시뮬레이션) unembed해 top-5 logit에 원 이름이 들어가는 비율을 측정한다. Name Mover는 95%+, 평균 head는 20% 미만 — attention weight와 별개로 "그 정보를 실제로 복사해 쓴다"는 기능적 증거.

## 한계·비판

- **일부 컴포넌트 미해명**: S-Inhibition Head의 attention pattern 자체(왜 S2에 attend하는가)와 MLP·layer norm의 역할은 이 논문 범위에서 설명되지 않음(저자 스스로 명시).
- **Completeness 실패**: greedy search로 찾은 incompleteness set이 최대 87% 크기의 손실을 유발 — 회로 밖에 여전히 해석 안 된 기여 구조가 존재함을 저자들이 인정.
- **스케일 한계**: GPT-2 small(117M) 1개 모델·1개 좁은 태스크에 국한. GPT-2 medium 예비 분석에서 벌써 "IO/S에 국한되지 않는 더 복잡한 패턴"이 관찰돼, 이 접근이 큰 모델·복잡 태스크로 스케일되는지는 미해결.
- **개입-의존적 구조(Backup Name Mover)**: 회로 탐색 결과가 knockout 여부에 따라 달라진다는 것은, "회로"라는 개념 자체가 정적(static)이 아니라 개입 상태에 의존하는 이동 표적일 수 있음을 시사 — 일반화된 해석 방법론으로서 재현성/안정성에 의문을 남김.
- **Adversarial example의 설명력 한계**: 저자 스스로 "회로 이해 없이도 발견 가능한 수준의 단순한 공격"이었다고 인정 — "회로를 알면 새로운 취약점을 예측할 수 있다"는 강한 주장까지는 아직 못 감.
