# Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (Bolukbasi et al. 2016)

- 출처: NeurIPS 2016 · arXiv:1607.06520 · PDF=`docs/Activation_steering_basic/BolukbasiDebias_1607.06520.pdf`
- 정독 섹션: §1 Introduction (무엇을 왜 하는가) 중심, §5·§6 방법부 발췌 확인
- tier: must
- 한줄역할: "개념=방향" 프레이밍과 "방향 제거=투영으로 개입"이라는 activation-steering 계열 전체의 원형(祖形)을 제공한 논문. 모델이 아니라 word2vec 임베딩 공간을 다루지만, 이후 모든 steering-벡터/투영-제거 계열 방법(ActAdd, CAA, ITI, RepE, conceptor 등)의 수학적 뼈대가 여기서 나옴.

## 문제·동기

word2vec(w2vNEWS, Google News 3M단어 학습, 300차원) 임베딩이 성별 고정관념을 노골적으로 담고 있음을 발견.
대표 사례: man - woman ≈ computer programmer - homemaker (유추 관계식이 그대로 성편향 answer를 냄).
이런 편향은 임베딩이 다운스트림(검색 랭킹, 이력서 파싱, 감성분석 등)에 광범위하게 쓰이기 때문에 사회의 편향을
그대로 반영하는 데 그치지 않고 증폭(amplify)시킬 위험이 있음. 목표는 임베딩의 "유용한 성질"(유의어 군집, 유추 풀이
능력)은 보존하면서 성별 고정관념만 제거하는 것.

## 핵심 아이디어

세 가지 경험적 발견이 이 논문 전체를 떠받침.

1. **편향이 방향(direction)으로 인코딩된다.** she-he, woman-man 같은 gender-pair 차이벡터들을 모으면 하나의
   지배적인 주성분(PC1)이 나오고(분산의 대부분을 설명, 그림6에서 랜덤벡터 대비 명확히 뚜렷함), 이 방향 g가
   "gender subspace"를 정의한다. → **개념=선형 방향**이라는, 이후 모든 activation-steering 문헌이 반복하는
   핵심 가정의 최초 정식화.
2. **gender-neutral 단어와 gender-definitional 단어는 이 subspace 밖에서 선형분리 가능하다.** (SVM으로 gender-specific
   words S를 분류, F-score ~0.63, class-balanced accuracy ~95%) → 편향을 "지워도 되는" 단어와 "정의상 성별을
   유지해야 하는" 단어를 나눌 수 있다는 전제.
3. **direct bias(단어-성별 직접연관)와 indirect bias(성별과 무관해 보이는 두 단어 사이에 성별을 매개로 생기는
   연관, 예: receptionist가 softball에 가까운 것)를 분리해 정량화**할 수 있다. β(w,v) 지표로 gender subspace를
   제거했을 때 유사도가 얼마나 변하는지를 측정.

## 방법 (bias direction 식별, hard/soft debiasing = 투영/중화)

**Step 1 — Identify bias subspace.** gender-pair defining set들(D_1..D_n, 예: {she,he}, {woman,man} 등 10쌍)의
각 pair 내부 평균-중심화 공분산행렬 C = Σ_i Σ_{w∈D_i} (w-μ_i)ᵀ(w-μ_i)/|D_i| 를 만들고 SVD 상위 k개 특이벡터를
bias subspace B로 정의 (k=1이면 단일 gender direction g). 논문 데이터에서는 PC1이 압도적으로 큰 고유값을 가져 사실상
1차원 방향으로 잡힘.

**Step 2a — Hard debiasing (Neutralize + Equalize).** 이것이 "투영 제거(projective steering)"의 원형.
- Neutralize: gender-neutral 단어 집합 N의 모든 단어 w에 대해 w := (w - w_B) / ||w - w_B|| — subspace B로의
  투영성분을 빼고 재정규화. 즉 hard projection-out(직교여공간으로 정사영 후 renormalize).
- Equalize: {grandmother, grandfather}처럼 정의상 성별이 다른 pair들을 subspace 밖에서는 평균으로 강제 일치시키고
  (equidistant 보장), subspace 안에서는 원점 중심으로 대칭 배치한 뒤 unit-length로 스케일. 목적: 임의의 중립단어가
  이 pair 양쪽에 완전히 등거리가 되게 하면서도, {male, female}처럼 정의적 성별 구분은 subspace 내에서 유지.
- 수학적으로 Observation 1에서 증명: 이 두 스텝 후 중립단어 w와 pair 내 임의의 두 단어 e1,e2 간 내적/거리가
  같아지고 PairBias=0이 됨을 formal proof로 보임.

**Step 2b — Soft debiasing.** 선형변환 T ∈ R^(d×d)를 학습해, 전체 임베딩의 pairwise inner product는 최대한
보존하면서(||(TW)ᵀ(TW) - WᵀW||²_F) 중립단어의 gender subspace 투영은 최소화(λ||(TN)ᵀ(TB)||²_F)하는 절충
최적화(SDP, X=TᵀT로 치환해 풀이). λ→∞이면 hard debiasing과 동일해짐. λ=0.2 사용. hard보다 편향 제거 효과는
약하지만(정성 결과: stereotypical analogy 19%→soft는 hard(6%)만큼은 안 줄어듦) 원 임베딩 기하를 더 보존.

두 알고리즘 모두 본질은 동일: **개념 방향을 찾고, 그 방향으로의 성분을 원하는 대상 집합에서 제거(hard=완전투영,
soft=정규화된 부분투영)**. 이것이 이후 활성화 스티어링에서 반복되는 "ablation/projection-out" 연산의 최초 형태.

gender-neutral word set N은 자동으로 결정(217개 사전기반 seed → SVM으로 전체 3M 단어로 일반화, balanced acc 95%).

## 실험·결과

- Direct bias: hard-debiasing 후 stereotype-판정 analogy 비율 19%→6% (crowd-worker 평가), appropriate analogy
  개수는 거의 유지(그림8). 예: "he:doctor::she:X" 가 원본은 X=nurse, debiased는 X=physician.
- Indirect bias: softball-football 축 위 극단단어에서 receptionist/waitress/homemaker 같은 성편향 단어가
  순위에서 밀려나고 infielder/major_leaguer 등 기능적으로 타당한 단어가 상위로 옴(그림3 정성결과, ground truth
  없어 정량 검증은 약함).
- Utility 보존: 표준 벤치마크(RG, WS word-similarity / MSR analogy)에서 debiasing 전후 점수 거의 동일
  (예: RG 62.3→62.4, analogy 57.0→57.0). "유용한 기하 구조는 안 깨졌다"는 근거.
- Occupation stereotype - crowd 판단 상관: she-he 축 투영과 crowd stereotypicality rating이 Spearman ρ=0.51,
  두 임베딩(w2vNEWS vs GloVe web-crawl) 간 occupation bias 상관 ρ=0.81 — 편향이 임베딩 특유 artifact가 아니라
  코퍼스/사회에 실재함을 시사.

## activation-steering 흐름 위치 (steering의 역사적 원형)

이 논문은 activation steering 계열의 계보 최상단에 위치한다고 볼 수 있음.

- **"개념=선형 방향" 가설**의 최초 명시적 정식화 및 정량 검증(주성분 분석으로 subspace 추출) — 이후 linear
  representation hypothesis(Park et al. 2023, geometry-of-truth 계열)의 직계 조상.
- **투영 연산으로 속성을 제거/중화**(Neutralize)한다는 아이디어는 이후 LLM activation steering의 "ablation"
  (예: refusal-direction ablation, Arditi et al. 2024)의 원형. 다만 이 논문은 "제거"만 하고 "추가"는 안 함
  (반대 방향으로 밀어넣는 additive steering은 아직 없음 — ActAdd/CAA류의 h + α·v 연산은 여기 없음).
- **soft debiasing의 λ-절충 최적화**는 "얼마나 세게 개입할지"를 조절하는 continuous steering-strength knob의
  선례 — 이후 steering coefficient α, conceptor의 aperture 파라미터 등으로 이어지는 개념.
- **차이는 대상과 시점**: 여기는 정적 임베딩 테이블(word2vec lookup table) 자체를 오프라인으로 영구 수정하는 것.
  트랜스포머의 forward-pass 중간 hidden state를 inference-time hook으로 실시간 개입하는 activation steering
  (2023~)과는 적용 대상(정적 벡터 vs 동적 residual stream)이 다르지만, "방향을 찾고 투영으로 조작한다"는
  수학적 커널은 동일하게 계승됨.

## 우리 프로젝트 연결 (방향 제거=projective steering의 조상)

우리 방법(contrastive conceptor C_steer = C_success ∧ ¬C_failure, h' = h·Mᵀ)의 계보를 거슬러 올라가면 이 논문의
"gender subspace 식별 → neutralize(투영-제거)" 스텝과 개념적으로 같은 자리에 있다.

- 우리 conceptor의 M(투영/변환 행렬)은 Bolukbasi의 T(soft debiasing 선형변환) 또는 (I - B Bᵀ)(hard debiasing의
  투영행렬)의 후예 — "다차원 subspace를 찾고 그 subspace를 기준으로 활성화를 변환한다"는 골격이 동일. 우리 문서
  에서도 이미 명시하듯 "단일벡터 additive가 아니라 multi-dim contrastive 연산자" 지향인데, Bolukbasi도 k>1
  subspace(SVD 상위 k개 성분)로 일반화 가능함을 §5.1에서 이미 언급(다만 실제 실험은 k=1).
- Neutralize(=hard projection-out)는 우리의 "¬C_failure"(실패 방향 제거) 쪽에 대응하고, Equalize(=pair를
  desired direction으로 재배치)는 "C_success로 끌어당김" 쪽에 유비될 수 있음 — 즉 우리 conceptor의 두 항
  (success 쪽으로 당기기 + failure 쪽 성분 죽이기)이 이미 2016년 이 논문에서 Neutralize/Equalize라는 두 개의
  별도 오퍼레이터로 분리돼 있었다는 점이 흥미로운 선례.
- 다만 결정적 차이: Bolukbasi는 **정적 lookup table을 오프라인 1회 수정**, 우리는 **inference-time online hook**.
  또한 이들은 "무엇을 지울지"(gender-neutral N)가 사전에 고정된 집합인 반면, 우리 문제의 핵심 난제(online
  phase/failure-type 식별)는 "언제 어느 pathway에 개입할지"를 매 스텝 실시간으로 판정해야 한다는 점 — Bolukbasi
  세팅에는 없는 "routing" 문제.

## 면접 포인트 (Q→A)

Q1. 이 논문이 이후 activation steering 연구에 남긴 핵심 유산은 무엇인가?
A. "개념(여기선 성별)이 임베딩 공간 안에서 하나의 선형 방향으로 표현된다"는 것을 정량적으로(PCA 고유값 분포,
랜덤벡터 대비 비교) 처음 보였고, 그 방향으로의 투영을 조작(neutralize=제거, equalize=재배치)해서 원치 않는
속성을 제어할 수 있음을 보였다. 이것이 "방향을 찾고 투영/가산으로 개입한다"는 이후 모든 activation-steering
방법론의 공통 커널이다.

Q2. hard debiasing과 soft debiasing의 차이와 트레이드오프는?
A. hard(Neutralize+Equalize)는 gender subspace 성분을 완전히 0으로 만드는 hard projection이라 편향 제거
효과가 강하지만(stereotype analogy 19%→6%) 원 임베딩의 미세한 기하(예: grandfather가 grandmother보다
"to grandfather a regulation"이라는 별도 의미를 갖는 것 같은 뉘앙스)까지 지워버릴 위험이 있다. soft는
선형변환 T를 SDP로 학습해 전체 pairwise 내적 보존과 편향 최소화 사이 λ로 절충하는데, 실험상 편향 제거
효과는 hard보다 약했다(같은 그림8 비교에서 stereotype 개수가 hard만큼 줄지 않음).

Q3. direct bias와 indirect bias를 왜 나눠서 정의했는가?
A. gender-pair 단어(brother/sister 등)만 지워도, gender-neutral 단어끼리(예: receptionist와 softball)
성별을 매개로 한 간접 연관은 여전히 남는다. β(w,v) = (w·v - w⊥·v⊥/(정규화)) / w·v 라는 지표로, 두 단어 유사도
중 gender subspace 성분이 차지하는 비율(%)을 측정해 이 간접 편향을 정량화했다. 이는 우리 프로젝트 맥락에서
"표면적으로 무관해 보이는 latent 차원들 사이에도 실패-관련 매개 성분이 숨어있을 수 있다"는 유비로 이어진다.

Q4. 이 방법의 근본적 한계는? 왜 최신 activation steering 논문들이 이걸로 안 끝났나?
A. (1) 정적 lookup table 전용이라 문맥의존적 표현(contextual embedding, 트랜스포머 hidden state)에는 그대로
적용 불가 — 온라인/inference-time 개입이라는 발상이 없다. (2) neutralize 대상 집합 N을 사전에 정적으로 정의
(dictionary+SVM classifier)해야 해서, "언제 개입할지"를 실시간으로 판정하는 routing 문제가 없다 — 우리
프로젝트의 핵심 난제(online phase/failure-type 식별)가 여기엔 아예 존재하지 않는 세팅이다. (3) 단일(또는 소수)
선형 방향 가정 자체가 이후 비판받음(아래 한계 참고).

## 한계·비판

- **편향이 정말 "하나의 선형 방향"으로 충분히 포착되는가**는 이후 강하게 논쟁됨. Gonen & Goldberg (2019,
  "Lipstick on a Pig")는 hard-debiasing 이후에도 gender-biased 단어들이 여전히 서로 군집(clustering)해
  classifier로 성별을 예측할 수 있음을 보여 "표면적 direct bias만 지워졌지 비선형적으로 인코딩된 편향은
  남아있다"고 비판. 이는 이후 activation steering 문헌에서도 반복되는 논쟁(단일 선형 방향 vs 다차원/비선형
  subspace)의 원조 격 사례.
- **N(gender-neutral) vs S(gender-specific) 분류 자체가 순환적/주관적.** 무엇이 "지워도 되는 편향"이고 무엇이
  "보존해야 할 정의"인지 정하는 기준 자체가 사람이 만든 사전과 SVM에 의존(F-score 0.63으로 그리 높지 않음).
- **평가가 주로 정성적 + crowd-worker 주관 평가.** indirect bias는 "ground truth가 없어 정량화가 어렵다"고
  저자 스스로 인정(§8). RG/WS/analogy 벤치마크로 "유용성 유지"는 확인했지만 "편향이 진짜 얼마나 줄었는지"의
  엄밀한 정량 지표는 약함.
- **다운스트림 효과 미검증.** 임베딩 자체의 기하는 바꿨지만 이 임베딩을 실제 분류기/검색엔진에 넣었을 때 최종
  태스크의 공정성이 개선되는지는 이 논문에서 직접 보이지 않음(이후 문헌에서 이 갭이 지적됨).
- **성별(F-M) 단일 축, 영어 한정.** 저자도 결론에서 인종 편향(minorities-whites 방향)도 유사하게 나타남을
  보였지만 정식 처리는 안 했고, 문법적 성(gender) 언어로의 일반화도 미해결 과제로 남김.
