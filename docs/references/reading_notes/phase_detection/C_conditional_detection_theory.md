# Phase 조건부 검출 구조의 이론적 근거 — Conditional/Regime-Conditional Detection 문헌 조사

- 조사일: 2026-08-10
- 조사 배경: "먼저 상태를 phase(국면)로 분류하고, 그 phase 전용 이상탐지기/분류기를 적용"하는 구조를 검토 중.
  우려 (a) phase 분류 오차의 뒤단 전파(cascade error propagation), 우려 (b) phase별 표본 감소로 인한 검정력 저하.
  이 구조 자체에 대한 일반 ML 이론/방법론 근거(로보틱스/VLA 국한 아님)를 조사.
- 조사 방법: 5개 병렬 조사(regime-conditional AD / cascade 오차전파 / MoE gating / conditional conformal
  prediction / changepoint+AD), 각 WebSearch로 후보 논문 탐색 후 WebFetch로 arXiv abstract·PMLR·ACL
  Anthology·PMC 등 실물 페이지를 직접 열어 제목·저자·연도·venue·핵심주장을 검증. 총 60편 이상 확인 시도,
  실물확인/서지확인/미확인을 각 항목에 명시. 조사관이 스스로 "확인 실패"로 폐기한 항목은 부록에 기록.
  이 문서 작성자(오케스트레이터)가 핵심 주장 2건(Barber et al. 2019, Ding et al. 2023)을 추가 spot-check함.
- 프로젝트 맥락: `docs/steering/RESEARCH_DIRECTION.md`의 "★ 중심 미해결 문제" — 추론 중(online) 어느 pathway가
  실패했고 어느 task-phase인지 식별 가능한가. 이 문서는 그 구조 선택(phase 먼저 분류 vs 무조건)에 대한
  일반 ML 근거를 다룬다.

---

## ★ 다이제스트

**"phase 먼저 분류 → phase별 검출" 구조가 이론·실증적으로 이득인 조건**은 5개 주제에 걸쳐 놀랍도록 수렴한다:

1. **phase 개수가 적고(대략 3~6개), 실제 국면 구조와 정확히 일치하며, 잘 정의되어 있을 때**만 조건화가
   무료(free)에 가깝다. Nguyen et al.(ICLR 2024, MoE 통계이론)는 fitted 국면 수가 진짜 국면 수와 정확히
   맞으면 hard/sparse 분할도 dense와 **동일한 parametric 수렴 속도**를 가짐을 증명 — 손해는 **국면 수를
   잘못 잡았을 때(특히 과대분할)** 집중적으로 발생한다. Han et al.(2024 서베이)도 동일 결론을 산업 fault
   diagnosis에서 독립적으로 제시: multi-model은 "조건이 적고 잘 정의되고 알려져 있을 때"만 유리하다.
2. **phase당 최소 표본수에 대한 정량적 하한선이 문헌에 이미 존재한다.** Ding et al.(NeurIPS 2023, conformal)
   은 목표 커버리지 1−α에서 그룹당 표본이 1/α−1 미만이면 검출 임계값이 아예 무의미(∞)해짐을 증명(α=0.1이면
   9개 미만은 붕괴). Vovk(2012) 저자 본인은 "카테고리당 100개 미만은 위험"이라는 경험칙을 제시. Li(2026,
   changepoint power analysis)는 단일 changepoint 탐지에 n≥30·효과크기≥2.0이 80% 검정력에 필요하다고
   정량화 — 이는 RoboCasa rollout(실패 시 ~45 step)을 3~4 phase로 쪼갤 때 **바로 위험대역에 들어간다.**
3. **오차 전파를 피하는 표준 처방은 "phase를 hard argmax로 확정하지 말고 soft/확률적으로 다음 단계에
   넘기는 것"** — 이 처방은 5개 주제 전부에서 독립적으로 재발견된다: Bourdev&Brandt soft cascade(2005),
   Bayesian Online Changepoint Detection의 run-length posterior(2007, 애초에 soft가 원설계), 산업
   multimode monitoring의 Bayesian soft mode-weighting(Yu&Qin 2008), Soft MoE(ICLR 2024, hard MoE 대비
   정량적으로 우월), Gibbs-Cherian-Candès(2023)의 joint 추정(calibration set을 물리적으로 쪼개지 않음).
4. **반증 사례도 있다 — hard가 무조건 나쁜 것은 아니다.** Hash Layers(NeurIPS 2021)는 완전 무작위/고정
   라우팅도 부하만 균형 잡히면 학습된 라우팅과 동등함을 보였고, Nguyen et al.(2)처럼 phase 수가 맞으면
   hard 분할도 무료다. 즉 핵심 레버는 "hard vs soft" 그 자체가 아니라 **"gate가 만드는 불균형·붕괴를
   막았는가"**와 **"phase 개수/경계가 실제 구조와 맞는가"** 이다.
5. **cascade 오차 전파는 실측 가능하고 실제로 발생한다.** Ren et al.(2026): 계층분류 cascade 오차의
   19.6%가 1단계에서 기원해 이후 단계에서 회복 불가능. Besbes et al.(Mozilla, 2026 실 프로덕션): changepoint
   기반 이상탐지 파이프라인에서 12.5% 오탐 + 6.8% 누락. Mangal et al.(ICLR 2023): 단계별 보장을 그대로
   합성 가능하다는 가정이 깨지면 인증 정확도가 97%→11%까지 붕괴할 수 있음(극단 사례, robustness
   certification 맥락).
6. **가장 구조적으로 가까운 이론 틀은 hierarchical classification의 "blocking problem"**(Silla & Freitas,
   2011 서베이) — top-down 조건부 분류에서 상위 오류가 하위에서 "절대 수정 불가능"해지는 현상. 완화책으로
   문헌은 (i) 애매한 분기 지점을 데이터 기반으로 평탄화(flatten, Naik&Rangwala 2016, +7%p), (ii) phase
   분류·검출을 joint 학습(Andor et al. 2016 label bias 이론, Melnyk et al. 2016 semi-Markov switching
   VAR로 regime-switching과 이상탐지를 단일 확률모델로 통합), (iii) 검출기 훈련 데이터에 phase 분류기의
   실제 오분류 패턴을 주입(Bengio et al. 2015 scheduled sampling의 train/test mismatch 논리)을 제시한다.
7. **★연구 공백(우리 프로젝트의 기회)**: "context/phase가 정확히 주어졌다는 가정 하의 이득"을 보인 논문은
   많지만(Bindini et al. 2026 TMLR, Ahmad et al. 2024, Islam&Carden 2026 등), "phase 분류기 자체가 틀렸을
   때 검출 성능이 정량적으로 얼마나 저하되는가"를 직접 측정한 논문은 이번 조사에서 **찾지 못했다** — 2026년
   최신 contextual anomaly detection 논문(Bindini et al., TMLR 2026)조차 실험에서 context는 항상 정확하다고
   가정한다. 이는 CLAUDE.md의 "★ 중심 미해결 문제(online phase 식별)"이 일반 ML 문헌에서도 여전히 열린
   질문임을 뒷받침한다.
8. **복잡도는 신호가 있을 때만 추가하라는 원칙이 독립적으로 재확인된다.** Mastriani et al.(2025)은 정교한
   changepoint 파생 구조가 단순 모델+양질의 세그멘테이션보다 못할 수 있음을 실증 — 프로젝트가 이미 채택한
   "사다리식 ablation(이전 단계가 신호를 보일 때만 복잡도 추가)" 원칙과 정확히 부합하는 독립적 근거.

**표준 처방 요약**: (1) phase 개수는 최소화하고 실제 국면 구조와의 일치를 먼저 검증, (2) phase 분류
결과는 hard argmax가 아니라 확률분포로 검출기에 전달(soft gating), (3) phase당 최소 수십~100 표본
확보를 게이트로 사용, (4) 검출기 훈련 시 phase 분류기의 실제 오분류를 주입, (5) 가능하면 phase 분류와
검출을 joint/end-to-end로 묶어 오차가 posterior 불확실성으로 흡수되게 하고, (6) 이 모든 것에 앞서
"phase 무관 단일 검출기" 대비 ΔAUROC/ΔSR을 반드시 실측 — 어느 문헌도 곱셈적 낙관 추정을 허용하지 않는다.

---

## 1. Regime-conditional / State-conditional Anomaly Detection (시계열)

### 1.1 원류 정의

**Song, Wu, Jermaine, Ranka — "Conditional Anomaly Detection"**
IEEE Transactions on Knowledge and Data Engineering, 19(5), 631-645, 2007 · DOI 10.1109/TKDE.2007.1009 ·
[서지 실물확인됨(dblp), 본문은 검색스니펫]

Context(=environmental attribute, 우리의 phase에 대응)를 먼저 조건화하고 그 안에서 target(=indicator
attribute)의 이상값을 찾는 "conditional anomaly detection(CAD)" 문제를 최초로 정식화한 원류 논문.
**이 논문은 context가 이미 정확히 주어졌다고 가정하며, context 추정 오차의 영향은 다루지 않는다** — 이
가정은 2026년 최신 후속 연구(§1.4)에서도 여전히 깨지지 않은 채 남아있다.

**Chandola, Banerjee, Kumar — "Anomaly Detection: A Survey"**
ACM Computing Surveys, 41(3), Article 15, 2009 · DOI 10.1145/1541880.1541882 ·
[서지 실물확인됨(dblp), 본문은 검색스니펫]

Contextual anomaly를 정의하며 이를 다루는 두 표준 기법을 명명: (i) **"reduction to point anomaly
detection"** — context attribute로 데이터를 분할한 뒤 각 context 내에서 기존 point anomaly 기법을
적용(=우리가 검토 중인 구조와 정확히 동일), (ii) 구조를 직접 모델링. 즉 우리가 검토하는 구조는 낯선
접근이 아니라 **contextual anomaly detection의 교과서적 baseline**이다 — 동시에 이 reduction 접근의
근본 취약점(context 분할 오류가 그대로 넘어감)은 2009년 시점에도 정량 분석되어 있지 않았다.

### 1.2 산업 공정 모니터링 — Multimode/Multiphase Process Monitoring

이 서브필드는 정확히 "국면을 먼저 식별하고 국면별로 이상탐지"를 30년 가까이 실무에서 다뤄온 성숙한
분야다.

- **Ge, Song, Gao — "Review of Recent Research on Data-Based Process Monitoring"**, Industrial &
  Engineering Chemistry Research, 52(10), 3543-3562, 2013 · DOI 10.1021/ie302069q ·
  [서지 실물확인됨(복수 독립 소스 일치), 본문은 403으로 미확인]

- **Yu & Qin — "Multimode process monitoring with Bayesian inference-based finite Gaussian mixture
  models"**, AIChE Journal, 54(7), 1811-1829, 2008 · DOI 10.1002/aic.11515 ·
  [서지 실물확인됨(402=실재 paywall), 본문은 검색스니펫]

  핵심: 공정 데이터를 mode별 Gaussian mixture component로 모델링하되, **hard mode assignment 대신
  Bayesian posterior probability(soft weighting)로 여러 mode의 통계량을 가중합산**. Tennessee Eastman
  등 3개 사례에서 기존 단일 PCA보다 우수. **왜 hard classification이 아니라 soft weighting을 썼는가가
  중요** — 이는 "mode 분류가 틀리면 오차가 그대로 전파된다"는 문제의식에 대한 업계의 표준 대응이 처음부터
  soft 결합이었다는 정황 증거.

- **Ge & Song — "Multimode process monitoring based on Bayesian method"**, Journal of Chemometrics,
  2009 · DOI 10.1002/cem.1262 · [서지만 확인, 원문 미확인] — 위와 같은 Bayesian soft mode-weighting 계열.

- **Sen, Raihan, Chidambaram — "Multiway continuous hidden Markov model-based approach for fault
  detection and diagnosis"**, AIChE Journal, 60(6), 2035-2047, 2014 ·
  [서지 실물확인됨(402), 본문 미확인] — Markov-switching/HMM 기반 이상탐지의 산업 사례.

- **Webb, Nnadili, Seghers, Briceno-Mena, Romagnoli — "Optimization of multi-mode classification for
  process monitoring"**, Frontiers in Chemical Engineering, 2022, article 900083 · [실물확인됨]

  PCA/UMAP + K-Means/DBSCAN/HDBSCAN + kNN 9개 조합을 유전 알고리즘으로 최적화해 mode label을 자동
  부여. 저자 스스로 "mode 오분류가 하류 fault detection 성능에 미치는 영향을 직접 실증적으로 측정하지
  않는다"고 인정 — cascade 검증 자체가 이 서브필드에서도 비어있는 지점. **정량치**: Tennessee Eastman
  20-fault 전체 분류 시 정확도 약 0.75(45개 클러스터), fault를 6개로 줄이면 정확도 상승; 3-mode
  pyrolysis 데이터에서는 모든 조합이 최대 정확도. → **phase 분류 자체의 정확도가 과제 복잡도에 크게
  좌우된다**는 실측.

- **Han, Liu, He, Ding, Zhou — "Multi-Condition Fault Diagnosis of Dynamic Systems: A Survey, Insights,
  and Prospects"**, arXiv:2412.19497, 2024(eess.SY) · [실물확인됨, 원문 인용 확보]

  ★이번 조사에서 찾은 가장 명시적인 "조건화 분기점" 진술. 원문 인용:
  > "single-model approaches are more suitable for tasks where the differences in data distribution
  > across operating conditions are relatively small... particularly advantageous in scenarios
  > involving diverse operating conditions or the presence of unknown conditions. In contrast,
  > multi-model approaches are more appropriate for scenarios with a limited number of well-defined
  > and known operating conditions."

  그리고:
  > "Although the aforementioned methods have demonstrated promising results in specific application
  > scenarios, they are highly dependent on precise condition identification."

  정리: **multi-model(=phase-conditional)이 유리한 조건 = 조건 개수가 적고, 잘 정의되어 있고, 알려져
  있을 때**. **single-model(=phase-agnostic)이 유리한 조건 = 조건 간 분포 차이가 작을 때, 혹은 조건이
  다양하거나 미지일 때**. 그리고 multi-model 전체가 "정확한 조건 식별에 크게 의존한다"는 것을 서베이
  저자들이 명시적으로 인정 — 우려 (a)를 문헌이 이미 알고 있다는 직접 증거.

### 1.3 Batch Process Phase Division (부분확인)

Phase 경계 오분류가 T²/SPE 통계량에 미치는 영향을 다루는 문헌 3편의 **존재는 검색으로 확인**되었으나
WebFetch가 전부 403으로 막혀 본문은 열람 실패([미확인]):
"Phase division and process monitoring for multiphase batch processes with transitions"
(Chemometrics and Intelligent Laboratory Systems); "Soft-Transition Sub-PCA Fault Monitoring of Batch
Processes"(I&EC Research, DOI 10.1021/ie3031983); "Stage-based soft-transition multiple PCA modeling
and on-line monitoring strategy for batch processes"(Journal of Process Control). 복수 논문 초록에서
반복 확인된(단 특정 논문에 귀속시키지 않는) 공통 서술: *"k-means 같은 hard-partition 클러스터링은 인접한
두 phase 경계 부근 샘플에서 오분류가 발생하며, 이를 무시하면 오탐이 증가한다"* — 대응은 "soft-transition"
(인접 phase 모델의 가중합)으로, §1.2와 동일한 처방 패턴("hard cascade 회피, soft 결합 채택").

### 1.4 Context 불확실성을 다루는 최신 연구 (2026) — 핵심 공백 발견

**Bindini, Perini, Nistri, Davis, Frasconi — "Dealing with Uncertainty in Contextual Anomaly Detection"**
TMLR(Transactions on Machine Learning Research), 2026-01 accept · arXiv:2507.04490 ·
[실물확인됨, 정량 표 확보]

Contextual AD에서 aleatoric/epistemic uncertainty를 명시적으로 모델링하는 "Normalcy Score(NS)"를
heteroscedastic GP 2개로 구성. **정량 근거**(Table 2, UCI): Abalone PR-AUC — NS 0.65±0.04 vs QCAD
0.28±0.04 vs ROCOD 0.40±0.05. SynMachine ROC-AUC — NS 1.00 vs QCAD 0.98 vs ROCOD 0.90. Toxicity
PR-AUC — NS 0.67 vs QCAD 0.45. 심장학 실사례(대동맥 직경 이상)에서도 유의 우수(DeLong p<0.001).

★**가장 중요한 발견**: 이 논문의 실험 프로토콜은 "contextual feature는 정확히 두고 behavioral value만
교란"하는 방식이다 — 즉 **"context 자체가 잘못 측정/추정된 상황"(우리의 phase 오분류 상황과 정확히
동일한 셋업)에 대한 체계적 degradation 분석은 2026년 최신 논문에도 없다.** 저자들은 epistemic
uncertainty가 "context가 희소한 영역"에서 발생한다고 언급하지만, "context 분류기 자체가 틀렸을 때"는
테스트하지 않는다.

**Calikus, Nowaczyk, Bouguelia, Dikmen — "Wisdom of the Contexts: Active Ensemble Learning for
Contextual Anomaly Detection"**, Data Mining and Knowledge Discovery, 2022 (arXiv:2101.11560, 2021 제출) ·
[실물확인됨]

원문 인용: *"identifying the right context can be very challenging in practice"*, *"there is no single
perfect context that successfully uncovers all kinds of contextual anomalies"* — 단일 고정 context(=
단일 phase 분류기)를 쓰는 접근 자체의 근본적 한계를 지적. 해법(WisCon): 여러 후보 context를 자동
생성하고 앙상블로 결합, context마다 다른 중요도 가중치 부여. 7개 데이터셋에서 baseline 대비 유의 우수
(정확한 수치 미확보). → **"phase 분류기 하나를 믿고 hard하게 넘기지 말고, 여러 후보 phase 분할을
동시에 유지·앙상블"**이라는 방법론적 대안의 실증 사례.

**Ahmad, Shadaydeh, Denzler — "Regime Identification for Improving Causal Analysis in Non-stationary
Timeseries"**, arXiv:2405.02315, 2024-04 · [실물확인됨(초록)]

비정상 시계열을 Riemannian 공간의 공분산 행렬로 국소 정상 구간(regime)으로 분해 후 regime별 인과분석이
전체 뭉뚱그림보다 낫다고 주장(synthetic + climate-ecosystem 데이터). **이 논문 역시 "regime 식별 오차가
하류 분석에 미치는 영향"은 다루지 않는다** — benefit-side만 있고 harm-side 분석이 없다는 점에서 Bindini
et al.과 동일한 패턴. **우연이 아니라 이 서브필드 전반의 공백으로 보인다.**

**Islam & Carden — "Product-Aware Deep Autoencoders for Robust Process Monitoring in Multi-Product
Cyber-Physical Systems"**, arXiv:2606.00052, 2026-05 · [실물확인됨]

Global(agnostic) 모델은 여러 product mode 분산을 모두 수용하려다 decision boundary가 지나치게 넓어져
미세 이상을 놓친다는 문제 제기, product별 학습 도메인을 좁힌 "product-aware" 방식 제안. **정량**:
stress-test에서 global 모델은 77.8% 탐지 실패, product-aware는 100% 탐지 — 이번 조사에서 확보한 가장
강한 "조건화=이득" 수치. 단, 표준 지표(F1/ROC-AUC)에서는 "comparable"이라고 저자 스스로 인정하며,
**mode label 오분류 효과나 mode별 표본 감소 비용은 다루지 않는다** — "mode label이 항상 정확하다"는
가정 하의 이론적 상한선으로만 참고할 것.

### 1.5 통계학 일반: 하위그룹 분석의 검정력 손실 (우려 b, 도메인 무관 원리)

**Cuijpers, Griffin, Furukawa — "The lack of statistical power of subgroup analyses in meta-analyses:
A cautionary note"**, Epidemiology and Psychiatric Sciences, Vol.30, 2021-12 · [실물확인됨, PMC 원문]

임상 메타분석 도메인이지만 통계 논리는 도메인 무관. 전체분석이 80% power에 6개 연구 필요할 때,
동일 조건에서 **subgroup 분석은 22개 연구 필요(3~4배)**. Subgroup 간 효과크기 차이가 0.3/0.2/0.1로
작아질수록 필요 연구 수는 56/120/498개로 지수적 증가, 극단적으로는 **주분석 대비 83배**. Subgroup 비율이
불균형(10% vs 90%)하면 350개 이상 필요. → "모집단을 하위그룹(phase)으로 쪼갤 때 필요 표본은 산술적이
아니라 훨씬 가파르게(3~80배) 늘어난다"는 통계 원리가 도메인 무관하게 성립한다는 근거.

### 1.6 이 절 요약

우려 (a)는 문헌이 이미 알고 있는 문제이지만 **정량화한 논문은 찾지 못했다** — 2026년 최신 contextual AD
연구조차 context가 항상 정확하다고 가정한다(공백, §연구공백 참조). 업계 표준 대응은 hard cascade가
아니라 **soft/확률적 결합**(Bayesian mode-weighting, soft-transition sub-PCA)이다. 유일하게 발견한
명시적 "분기 조건"(Han et al. 2024)은 조건이 "적고 잘 정의되고 알려져 있을 때"만 multi-model이
유리하다는 것. 우려 (b)는 도메인 무관 일반 통계 원리(Cuijpers et al.)로 뒷받침된다.

---

## 2. Cascade / Two-stage Classifier의 오차 전파와 완화 기법

### 2.1 Vision Cascade 고전 — 오차 전파의 수학적 정식화

**Viola & Jones — "Rapid Object Detection using a Boosted Cascade of Simple Features"**
IEEE CVPR 2001 · [실물확인됨(독립 출처 3개 이상 교차확인)]

얼굴 검출을 K단계(최대 32~38단계) cascade로 분해, 각 stage는 이전 stage를 통과한 양성 후보만 받음.
**전체 성능이 stage별 성능의 곱으로 결합된다는 공식을 명시**: 전체 false positive rate `F = ∏ f_i`,
전체 detection rate `D = ∏ d_i`. 예시: 32-stage cascade에서 전체 FPR 10⁻⁶에는 stage당 FPR≈65%면
충분하지만, 전체 detection rate 90%에는 **stage당 detection rate 99.7%**가 필요 — 앞단이 조금만
나빠져도 지수적으로 전체가 나빠지는 **우려 (a)의 원형적 수학 모델**. 완화책: 앞단을 의도적으로
high-recall/low-precision(false negative 최소화)으로 편향.

**Bourdev & Brandt — "Robust Object Detection via Soft Cascade"**, IEEE CVPR 2005 ·
[실물확인됨(Semantic Scholar+교차인용), 원문 PDF 미파싱]

Viola-Jones의 discrete/hard-stage cascade(이산적 reject/accept)를 "soft cascade"로 일반화 — 매 feature
평가마다 누적 점수에 대해 연속적 threshold를 둬, 한 지점의 오분류가 즉시 전체를 끊지 않게 함. 기존
최고 성능과 동등한 detection rate/speed를 유지하며 학습이 쉬워지고 필요 feature 수가 줆.
→ **phase 확률분포를 유지한 채 여러 phase 검출기를 가중 결합**하는 것이 20년 전 컴퓨터비전에서 이미
검증된 완화책.

**Weiss & Taskar — "Structured Prediction Cascades"**, AISTATS 2010(PMLR vol.9, 916-923) ·
[실물확인됨]

Structured prediction cascade에서 max-marginal 기반 "필터링된 출력 집합"을 표현, **filtering error와
filtering efficiency를 동시에 통제하는 convex loss**를 제안 — "다음 단계가 정답을 여전히 후보군에
포함하는가(oracle recall)"를 직접 학습 목표에 반영. Inference 복잡도를 최대 5자리수 감소시키면서도
정확도는 유지/향상(핸드라이팅 인식·POS tagging). → phase 분류기를 top-1이 아니라 **"정답 phase가 후보
집합에서 탈락하지 않는 것"을 목적함수로** 학습하는 것이 이 논문의 핵심 처방.

**Mangal, Wang, Zhang, Leino, Pasareanu, Fredrikson — "On the Perils of Cascading Robust Classifiers"**
ICLR 2023(arXiv:2206.00278) · [실물확인됨]

Adversarial robustness 인증 맥락이지만 구조적으로 관련 깊음 — cascading ensemble에서 "각 단계가
개별적으로 견고성 인증되었으니 전체도 인증된 것"이라는 **naive composition 가정 자체가 수학적으로
틀림**을 증명(cascade attack, CasA). **정량**: 전체가 robust하다고 인증된 샘플 중 **최대 88%에서 실제
adversarial example 존재**; 인증된 robust accuracy 97%가 실제 공격 하에서는 **11%까지 붕괴**. →
"각 단계 성능을 독립적으로 좋게 만들면 전체도 좋다"는 가정이 (인증 세팅에서는) 완전히 틀릴 수 있다는
강한 경고. **stage 간 오차가 독립이 아니라 상관되어 있을 가능성**을 실측해야 함을 시사.

### 2.2 NLP Pipeline 오차 전파 정량화

**Caselli et al. — "When it's all piling up: investigating error propagation in an NLP pipeline"**
SemEval-2015 Task4 관련 워크숍, CEUR-WS Vol-1386, 2015 · [실물확인됨(제목/저자9인/연도/venue)]

Cross-document event timeline 파이프라인(entity/event detection→coreference→SRL→temporal relation)의
오차 누적을 추적. 구성요소별 정밀 수치는 미확인이나, "표준 서브태스크를 각각 잘 풀어도 복잡한 최종
과제 해결에는 불충분"이라는 문제의식은 명확.

**Bohnet & Nivre — "A Transition-Based System for Joint Part-of-Speech Tagging and Labeled
Non-Projective Dependency Parsing"**, EMNLP 2012 · [실물확인됨(ACL Anthology)]

POS tagging 먼저 확정 후 그 결과로 parsing하는 전통적 pipeline의 오류 전파 문제를, tagging과 parsing을
하나의 transition-based 시스템에서 **동시에(jointly)** 풀어 완화 — tagging이 확정되기 전 parsing 신호가
tagging 결정에 영향을 줌. Pipeline 대비 두 태스크 모두 일관되게 향상(정확한 %는 미확인).

**Yan, Jia, Tu — "An Empirical Study of Pipeline vs. Joint approaches to Entity and Relation
Extraction"**, AACL-IJCNLP 2022(ACL Anthology 2022.aacl-short.55) · [실물확인됨]

"pipeline은 error propagation 때문에 나쁘다"는 통념을 통제 실험으로 재검증. 같은 span representation을
쓰면 **최고의 joint가 최고의 pipeline을 여전히 능가**하지만, ★**"제대로 설계되지 않은 joint 방법은
오히려 pipeline보다 못할 수 있다"**는 반대 방향 경고도 함께 제시. → 균형점: "잘 설계된 joint > 잘
설계된 pipeline" 정도의 조심스러운 결론이지, joint가 무조건 이기는 것이 아니다.

**Ren, Zhao, Sun — "Cascading versus Joint Modeling for Hierarchical Offensive Language Detection"**
arXiv:2607.16790, 2026-07 · [실물확인됨] — ★이 조사에서 가장 정밀한 오차 전파 정량치.

계층적 label 구조 3단 cascaded 분류 vs joint multi-task 모델을 정확도·파라미터·지연 3축으로 비교.
**정량**: (1) **cascade pipeline 전체 오류의 약 19.6%가 1단계 필터에서 기원하며 이후 단계에서 수정
불가능**(오차 전파 상한의 실측치). (2) 그럼에도 cascaded 구조가 세 서브태스크 모두 joint보다 정확도
높음, 가장 불균형한 서브태스크에서 **macro-F1 +7.1점**. (3) 대가: 파라미터 3배, 추론 지연 1.67배.
→ **양가적 답**: cascade가 실제로 ~20%의 회복불가 오차를 갖지만, 데이터가 불균형하면 여전히 joint보다
나을 수 있다(비용은 더 든다). 채택 여부는 "phase 분포가 얼마나 불균형한가"와 "여분 파라미터/latency
예산"에 달려 있다.

**Andor, Alberti, Weiss, Severyn, Presta, Ganchev, Petrov, Collins — "Globally Normalized
Transition-Based Neural Networks"**, ACL 2016(arXiv:1603.06042) · [실물확인됨]

**Label bias problem**: locally-normalized(단계별 독립 정규화) 모델은 각 단계에서 국소 최선을 선택하나
전역 최적이 아닐 수 있음을 이론적으로 증명; globally normalized 모델이 엄격히 더 표현력이 강함. **정량**:
greedy(사실상 pipeline) baseline 대비 **정확도 +1.8%**, UAS/LAS 약 +2%, 최종 UAS 94.41%(당시 SOTA). →
phase 분류기가 timestep마다 국소적으로 최적인 phase를 독립 결정하면 뒷단 검출기 입장에서 최적이 아닌
배정이 구조적으로 발생할 수 있음(label bias). +1.8~2%p는 "오차 전파가 있지만 파국적이지는 않다"는
규모감도 제공.

### 2.3 Exposure Bias / 자기회귀적 오차 누적

**Bengio, Vinyals, Jaitly, Shazeer — "Scheduled Sampling for Sequence Prediction with Recurrent Neural
Networks"**, NeurIPS 2015(arXiv:1506.03099) · [실물확인됨]

RNN 훈련 시 teacher forcing(항상 실제 이전 토큰) vs 추론 시 모델이 생성한(잠재적으로 틀린) 토큰을
쓰는 **train/test mismatch**가 오차를 빠르게 누적시킴. 완화책: 훈련 초반 실제 토큰, 점점 모델 생성
토큰을 섞는 curriculum(scheduled sampling). → 정확히 같은 현상이 우리 구조에도 있음: 검출기를 "정답
phase 라벨"로 훈련하고 서빙 시 "phase 분류기의 예측(오분류 포함)"을 받는다면 이것이 정확히 이 논문의
train/test mismatch. **완화책: 훈련 시 phase 분류기의 실제 예측(오분류 포함)을 검출기 학습 입력에
섞어야 함** — gold phase label로만 학습시키면 서빙 시점 성능이 과대평가됨.

### 2.4 Hierarchical Classification 이론 — 가장 근접한 이론 틀

**Silla Jr. & Freitas — "A survey of hierarchical classification across different application
domains"**, Data Mining and Knowledge Discovery, Vol.22, 2011(Springer, DOI 10.1007/s10618-010-0175-9) ·
[실물확인됨(3개 독립 출처 교차확인, abstract 전문 확보)]

★**이 조사 전체에서 우리 구조와 가장 직접적으로 동형인 이론 틀.** Hierarchical classification 문헌을
통합하는 서베이. **Top-down 접근법**(상위 레벨에서 먼저 분류 후 그 결과를 조건으로 하위 레벨 분류 —
"phase 먼저 분류→phase 조건부 검출기"와 정확히 동형)이 계산 효율은 좋지만 **"상위 레벨 예측 오류는
하위 레벨에서 절대 수정될 수 없다"**는 구조적 결함을 지적, 이를 **error propagation**이라 명명. 관련
개념 **"blocking problem"**: 중간 노드에서 "이 샘플은 이 하위트리에 속하지 않는다"고 (틀리게) 판정하면
그 아래 모든 세부 분류 기회가 원천 차단됨. 대안 구조군(big-bang/flat 접근 — 계층 무시하고 한 번에
분류, local-classifier-per-level 등)도 정리.

**Naik & Rangwala — "Inconsistent Node Flattening for Improving Top-down Hierarchical Classification"**
IEEE DSAA 2016(arXiv:1706.01214) · [실물확인됨]

Silla&Freitas의 top-down error propagation에 대한 구체적 완화책 — 계층 구조 안의 "inconsistent
node"(데이터상 근거가 약한 상위 분기)를 데이터 기반으로 탐지해 **평탄화(flatten)**, 즉 신뢰할 수 없는
상위 분류 결정 지점 자체를 제거해 오차 전파 경로를 원천 차단. **정량**: 최고 top-down baseline 대비
**Macro-F1 최대 +7%**. → "phase 경계가 애매한 timestep"(reach→grasp 전환 구간)이 이 논문의 "inconsistent
node"에 해당할 가능성 — 애매 구간에서는 phase를 확정하지 않고 넘기는(flatten/skip) 전략이 문헌상 근거
있는 완화책.

### 2.5 Soft Gating / Confidence-Gated 최신 완화 기법

**Mokssit, Karrakchou, Mousist, Ghogho — "Confidence-Gated Training for efficient early-exit neural
networks"**, arXiv:2509.17885, 2025-09 · [실물확인됨]

Early-exit 신경망(중간 layer 조기종료 cascade)에서 여러 exit 공동학습 시 깊은 exit gradient가 얕은
exit을 압도하는 gradient interference 발생. 제안(CGT): **"얕은 exit이 실패했을 때만 깊은 exit으로
gradient를 조건부 전파"** — 훈련 정책을 추론 정책과 일치시킴. → "phase 분류가 애매하면(confidence
낮으면) 여러 phase 검출기를 동시에 돌리거나 판단을 유보"하는 confidence-gated 설계가 최신 문헌에서
근거를 가짐.

### 2.6 정량적 "임계값" 분석 — 명시적 보편 정리는 없음

"앞단 정확도 X% 이하로 떨어지면 전체가 단일 모델보다 나빠진다"는 형태의 **깨끗한 보편 임계값 정리는
찾지 못했다**(정직하게 보고). 대신 3가지 근접 도구: (1) Viola-Jones 곱셈 공식(함수 형태), (2) Ren et
al.의 19.6%(경험적 실측), (3) Mangal et al.의 97%→11%(극단 붕괴 사례, robustness certification 맥락이라
직접 이식은 주의).

### 2.7 이 절 요약

Hard routing(phase argmax 확정 후 해당 검출기만 적용)은 Viola-Jones/Bourdev-Brandt/Silla&Freitas가
공통으로 경고하는 실패 모드다. 처방: soft gating, joint 학습(단 잘못 설계하면 역효과), 검출기 훈련에
실제 오분류 패턴 주입, 애매 구간 flatten/보류. 곱셈적 성능 저하를 기본 가정하되 stage 간 오차 상관구조를
반드시 실측.

---

## 3. Mixture-of-Experts / Conditional Computation — Hard vs Soft Gating

### 3.1 원조 이론

**Jacobs, Jordan, Nowlan, Hinton — "Adaptive Mixtures of Local Experts"**, Neural Computation 3(1):
79-87, 1991 · [실물확인됨(hinton 개인 페이지 직접 확인, DOI 10.1162/neco.1991.3.1.79)]

여러 local expert와 이를 조합하는 gating network로 구성된 지도학습 구조의 원조. (배경지식, abstract
원문 미재확인) softmax gating을 error function 내부(log-sum-exp)에 넣어야 expert가 경쟁하며 분업한다는
설계 통념이 이 논문에서 유래 — gating 설계에 따라 분업 유도 여부가 갈린다는 시사점.

**Bengio, Léonard, Courville — "Estimating or Propagating Gradients Through Stochastic Neurons for
Conditional Computation"**, arXiv:1308.3432, 2013 · [실물확인됨]

Stochastic/hard(non-smooth) 뉴런의 gradient 추정이 conditional computation의 근본 난제임을 정식화.
REINFORCE, 확률적/미분가능 분해, noise injection, straight-through estimator 4가지 비교. →
**hard gating(phase의 argmax 선택)은 그 자체로 미분 불가능/불연속이라는 근본 문제**를 안고 있음을
정식화한 원류 — 이후 모든 MoE 계열(noisy top-k, straight-through류)이 이를 우회하는 공학적 타협이라는
계보.

### 3.2 대규모 실증 MoE — Hard 붕괴와 완화

**Shazeer et al. — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"**
arXiv:1701.06538, ICLR 2017 · [실물확인됨]

Noisy top-k gating(softmax 전 학습가능 가우시안 노이즈 추가 후 top-k만 남김) + **importance loss**(expert별
gate 값 총합의 분산 벌점) + **load loss**(실제 처리량 분산 벌점)로 expert 불균형 통제. **이 논문 자체가
"naive hard gating은 붕괴한다"는 것을 전제로 설계됨** — 노이즈 없는 순수 top-k gate는 소수 expert만
계속 선택되는 self-reinforcing 불균형에 빠지기 쉬움. Gating 붕괴가 실제 문제였다는 최초의 시스템적
증거.

**Switch Transformers(Fedus, Zoph, Shazeer)**, arXiv:2101.03961, JMLR 2021 · [실물확인됨]

Top-k(k>1)를 top-1으로 극단 단순화해도(가장 극단적 hard routing) 품질 유지하며 연산 감소·성능 개선.
**정량**: 동일 계산자원 대비 최대 **7배** 사전학습 속도 향상(T5-Base/Large 대비), T5-XXL 대비 4배 속도.
→ 극단적 hard routing도 auxiliary load-balancing loss + 안정화 기법을 더하면 작동한다는 사례 — "hard
자체가 원죄가 아니라 안정화 장치 없이 쓰는 게 위험"이라는 쪽에 가까운 증거.

**Expert Choice Routing(Zhou et al.)**, arXiv:2202.09368, NeurIPS 2022 · [실물확인됨]

기존 token-choice(토큰이 expert를 고름) 방식은 "poor routing이 특정 expert의 under-training을
유발"한다고 지적, **expert-choice**(expert가 자신이 처리할 top-m 토큰을 고름)로 뒤집어 부하 균형을
구조적으로 보장. **정량**: 학습 수렴 2배 이상 개선, GLUE/SuperGLUE 11개 태스크 중 7개에서 동일 계산비용
대비 T5보다 우세. → routing "방향"을 뒤집는(phase가 detector를 고르는 대신 detector가 자신에게 맞는
sample을 고르는) 역방향 설계도 검토 가치.

**Hash Layers(Roller, Sukhbaatar, Szlam, Weston)**, arXiv:2106.04426, NeurIPS 2021 · [실물확인됨]

★**이번 조사에서 가장 반직관적인 반대 증거.** 학습되지 않은 고정 해시 함수로 토큰→expert를 결정론적
배정 — 라우팅 파라미터도 auxiliary loss도 불필요. **정량**: Switch/BASE 같은 learning-to-route 방법과
**동등하거나 우수**. → **gating이 "똑똑하게" 고르지 않고 완전히 무작위/고정으로 골라도, 부하만 균형
잡히면 학습된 라우팅과 성능이 비슷하다** — gating 오류 자체가 항상 치명적인 것은 아니며, gating이
야기하는 불균형/붕괴가 진짜 문제라는 해석을 뒷받침.

**BASE Layers(Lewis, Bhosale, Dettmers, Goyal, Zettlemoyer)**, arXiv:2103.16716, 2021 · [실물확인됨: 메타데이터]

토큰-전문가 할당을 선형 할당 문제로 공식화해 불균형을 최적화 문제로 원천 제거. "학습된 gating이 만드는
불균형"을 신뢰하지 않고 알고리즘적으로 균형을 강제하는 다른 축의 해법.

**CompeteSMoE(Pham et al.)**, arXiv:2402.02526(원판)/2505.13380(확장판), 2024 · [실물확인됨]

Representation collapse 해결을 위해 가장 높은 neural response를 보이는 expert에만 라우팅하는
competition 메커니즘, **이론적으로 optimal estimator와 동일한 수렴 속도**를 증명. 15개 벤치마크에서
zero-shot 성능 개선. 학습 시 모든 expert 활성화(soft/dense) → 빠른 router로 distill → 추론은 sparse.
→ **"학습은 soft/dense, 서빙은 hard"**라는 실용적 절충 패턴의 좋은 사례.

### 3.3 Soft Routing이 정량적으로 이겼다는 직접 증거

**Puigcerver, Riquelme, Mustafa, Houlsby — "From Sparse to Soft Mixtures of Experts"(Soft MoE)**
arXiv:2308.00951, ICLR 2024 · [실물확인됨]

기존 sparse MoE(hard routing)의 문제(training instability, token dropping, 확장 어려움, 비효율적
finetuning)를 명시. Soft MoE는 fully-differentiable하게, 모든 입력 토큰의 가중조합을 각 expert에
전달하는 implicit soft assignment로 대체. **정량**: Soft MoE Huge/14(128 experts)는 ViT-Huge/14 대비
파라미터 **40배 이상**인데 추론시간 증가는 **단 2%**; dense Transformer와 인기 hard MoE(Token/Expert
Choice) 모두를 **큰 차이로 능가**. → ★이번 조사 전체에서 유일하게 "동일 조건에서 soft가 hard를 정량적으로
이겼다"를 직접 명시. 저자가 지목한 hard 실패모드(instability, token dropping)는 우리의 "gate 오분류가
뒤로 전파"와 본질적으로 같은 현상. 단, vision token-level MoE라 시간적 phase gate로의 직접 전이는
유추이지 증명이 아님에 유의.

**Rastegar — "Soft-to-Hard Routing in Sparse Mixture-of-Experts Models"**, arXiv:2605.02124, 2026 ·
[실물확인됨: 존재/저자/연도만, 단독저자]

Softmax routing temperature→0에서 hard top-1 routing에 수렴함을 이론 분석. Zero-temperature 근사
정확도가 "routing 경계 근처 O(τ) neighborhood의 확률질량"에 의해 결정됨을 규명(수렴속도 O(τ^α)).
→ **soft/hard가 이분법이 아니라 temperature라는 연속 스펙트럼의 양극단**이라는 프레임 — 실무적으로
sharpness를 조절 가능한 hyperparameter로 두고 검증셋으로 최적점 탐색 가능. 매우 최근(2026-05) 단독저자
논문이라 재현/피인용 검증 안 됨에 유의.

### 3.4 통계 이론: Gating이 통계적 효율에 미치는 영향 (우려 b 직결)

**Nguyen, Akbarian, Yan, Ho — "Statistical Perspective of Top-K Sparse Softmax Gating Mixture of
Experts"**, arXiv:2309.13850, ICLR 2024 · [실물확인됨(ar5iv HTML로 원문 문장 직접 인용 확인)]

★이 조사에서 우려 (b)에 대해 가장 정밀한 답. 원문 인용 기반 핵심 결과:
- **모델이 정확히 특정된 경우**(true expert/regime 개수를 정확히 앎): "the convergence rates of density
  and parameter estimations are both parametric on the sample size" — **top-K sparse(hard) gating이
  dense/soft gating과 동일한 최적(parametric) 수렴 속도를 달성**. Sparse 자체가 통계적 손해를 주지 않음.
- **모델이 과대특정된 경우**(fitted expert 수 > 실제 개수): "while the density estimation rate remains
  parametric under this setting, the parameter estimation rates become substantially slow" — softmax
  gating과 expert function 간 내재적 상호작용(다항방정식계 해로 표현되는 identifiability 문제) 때문에
  파라미터 추정이 현저히 느려짐.

→ **"phase를 나누면 표본이 준다"는 단순 통념과 달리, 이론은 "몇 개로 나누는가(phase 개수)가 진짜
국면 수와 정확히 일치하면 hard 분할도 손해가 없고, 문제는 개수를 잘못 잡았을 때(특히 과대분할)
집중적으로 터진다"고 정밀화한다.** 실무 함의: "phase를 몇 개로 나눌지"의 정확성(under/over-segmentation
검증)에 투자하는 것이 "hard냐 soft냐"보다 더 근본적인 레버일 수 있음.

**Nguyen, Ho, Rinaldo — "Sigmoid Gating is More Sample Efficient than Softmax Gating in Mixture of
Experts"**, arXiv:2405.13997, NeurIPS 2024 · [실물확인됨]

Softmax gating(정규화되어 expert끼리 zero-sum 경쟁)은 불필요한 경쟁을 유발해 representation collapse를
일으킬 수 있음. Sigmoid gating(각 expert에 독립적 0~1 스코어, 정규화 없음)이 이론적으로 더 높은 sample
efficiency를 회귀 프레임워크에서 증명. **정량**: ReLU/GELU 등 활성화 함수 하에서 sigmoid gating이 더
빠른 수렴(동일 추정오차에 더 작은 표본 크기 필요, 정확한 지수는 미확인). → **"soft vs hard"보다 중요할
수 있는 축은 "competitive(합=1) vs non-competitive(독립 스코어)"**. Phase gate에서 "모든 phase 확률
합=1"을 강제하는 softmax보다, 각 phase에 독립 membership score(sigmoid)를 매기는 편이 표본 효율 면에서
이론적으로 유리할 수 있음.

### 3.5 Gating 오류의 Downstream 영향 직접 증거 (우려 a 직결)

**Yoon, Wang, Chen, Ok — "When Are Experts Misrouted? Counterfactual Routing Analysis in
Mixture-of-Experts Language Models"**, arXiv:2605.07260, 2026 · [실물확인됨: 존재/저자/연도, 정량 수치는 미확인]

고정된 학습 모델에서 실제 선택 route를 동일 연산량의 대안 route와 비교하는 counterfactual routing
분석. "Standard router는 confident token에서는 route utility와 잘 정렬되나, **hard reasoning을
주도하는 fragile token에서는 정보가 부족**"하며, 이런 token에서는 "더 낮은 loss를 내는 동일 연산량의
대안 route가 frozen model 안에 항상 존재하지만 선택되지 않는다"는 것을 실증. 마지막 층 router만
최소 업데이트해도 AIME 2024/2025, HMMT 2025 pass@K 개선(Qwen3-30B-A3B, GPT-OSS-20B). 정확한 수치는
논문 뒷부분(Table 3/9, Fig.7) 미접근으로 미확인. → **"fragile token"이 phase 전환 구간(reach→grasp
경계처럼 애매한 프레임)과 구조적으로 유사** — trained gate가 애매한 입력에서 특히 나쁜 선택을 하고
그게 실제 성능을 깎는다는, 우리 우려(a)와 구조적으로 가장 가까운 최신(2026) 사례. 단 LLM reasoning
MoE router 이야기라 유추이지 직접 증거는 아님.

**Ruggieri, Stranieri, Stella, Scutari — "Hard and Soft EM in Bayesian Network Learning from Incomplete
Data"**, Algorithms 13(12):329, 2020 · [실물확인됨: 논문 존재/저자/저널, "44/67 시나리오 hard EM 우세"
수치는 미확인(WebSearch 2차 요약에만 존재, abstract 원문에는 없음)]

확인된 정성적 결론만: "데이터 특성에 따라 hard/soft 중 한쪽을 추천할 수 있는 시나리오들이 있다" —
일방적으로 soft가 우월하다는 결론이 아니라 **데이터 의존적(no free lunch)**.

### 3.6 이 절 요약

우려 (a)는 광범위하게 확인됨(Shazeer 2017부터 GShard/Switch/BASE/Hash/Expert-Choice까지 거의 모든
대형 MoE 논문이 "naive learned hard routing은 붕괴한다"는 전제 위에 안정화 장치를 답), Soft MoE(ICLR
2024)는 동일 조건에서 hard 대비 정량적으로 우월함을 직접 보임. **그러나 "무조건 soft를 써라"로
단순화하면 안 됨** — Hash Layers는 무작위/고정 라우팅도 학습된 라우팅과 동등함을 보였고, CompeteSMoE는
잘 설계된 hard/competition routing이 optimal 수렴속도를 달성함을 증명. 핵심은 "hard냐 soft냐"가 아니라
**"gate가 만드는 불균형/붕괴를 어떻게 막는가"**. 우려 (b)의 가장 정밀한 답(Nguyen et al. ICLR 2024):
**phase 개수가 진짜 국면 수와 정확히 일치하면 hard 분할도 손해 없음, 손해는 개수를 잘못 잡았을 때
집중.** Soft/hard보다 근본적 축은 "competitive(softmax) vs non-competitive(sigmoid) gating"일 수
있음(NeurIPS 2024). 실용적 절충: "soft로 학습/fit, hard(또는 sharp)로 서빙."

---

## 4. Conditional / Group-conditional Conformal Prediction vs Marginal

### 4.1 원류 — Mondrian/조건부 ICP

**Vovk — "Conditional Validity of Inductive Conformal Predictors"**, JMLR: W&CP 25:475-490(ACML 2012),
arXiv:1209.2673 · [실물확인됨]

Split(inductive) conformal predictor는 원래 marginal coverage만 보장; 이 논문은 "conditional ICP" —
카테고리(taxonomy) K에 따라 calibration set을 층화해 카테고리별로 p-value를 따로 계산 — 를 정의하고
이것이 카테고리별 validity를 **exact**하게 달성함을 증명. 즉 **고정된 소수의 이산 카테고리라면 조건부
validity 자체는 이론적으로 달성 가능.** ★**용어 정정**: 이 논문은 실제로 "Mondrian"이라는 용어를 쓰지
않는다(본문 검색으로 확인) — "Mondrian confidence machine"의 원조는 Vovk, Lindsay, Nouretdinov,
Gammerman의 **2003년 기술보고서**이며, 이 2012 JMLR 논문은 그 개념을 "conditional ICP"로 재정식화한
후속 논문(더 많이 인용되는 정식화본)이다. **정량 근거**: 엄밀한 정리 형태의 "카테고리 크기 대 효율"
공식은 없으나, 저자 본인의 명시적 실무 휴리스틱이 있음(Section 8): *"we can approach example conditional
validity by using conditional ICPs but making sure that the size of a typical category does not become
too small (say, less than 100)."* → **카테고리(phase)당 대략 100 표본 미만은 위험**하다는 저자 본인의
경험적 가이드라인.

### 4.2 불가능성 정리 — 핵심 이론

**Barber, Candès, Ramdas, Tibshirani — "The Limits of Distribution-Free Conditional Predictive
Inference"**, arXiv:1903.04684(v1 2019-03), Information and Inference: A Journal of the IMA, 10(2):
455-482, 2021 · [실물확인됨 — abstract 및 정리 목록을 이 문서 작성자가 추가 spot-check(제목·저자 일치,
"exact conditional inference guarantees are known to be impossible without imposing assumptions on the
underlying distribution" 직접 확인)]

★우려 (b)의 이론적 뿌리. 핵심 정리(원 출처는 Vovk 2012, Lei&Wasserman 2014 결과의 재정리):
> "Suppose that C^n satisfies (1−α)-CC[conditional coverage]. Then for all distributions P, it holds
> that 𝔼[leb(C^n(x))] = ∞" (거의 모든 non-atomic x에서)

즉 **연속형 x에 대해 정확한 conditional coverage를 분포-무관하게 달성하려면 예측구간의 기대 길이가
무한대가 되어야 한다** — 사실상 불가능. 완화판(근사적 "(1−α,δ)-CC"로 목표를 낮춰도) 기대 길이 하한:

    𝔼[leb(C^n(X_{n+1}))] ≥ inf_{c∈[0,1]} { (1−α)/(1−cα) · L_P(1−cαδ) }

δ(허용하는 조건화 단위의 최소 확률질량)가 작아질수록(phase를 잘게 쪼갤수록) marginal coverage를 훨씬
강하게 잡아야 하고 구간이 극도로 넓어짐. 국소(local) 완화(x 주변 ball에 대해서만 조건부 coverage 요구)도
VC 차원 조건 VC_{a.e.}(𝔛) ≥ 2n+2이면 여전히 trivial한 방법보다 나을 수 없고, VC(𝔛) ≤ c·δ·n/log²(n)로
충분히 제약되면 오라클에 근접 가능 — **조건화 단위를 잘게/유연하게 만들수록 필요조건이 기하급수적으로
빡빡해지는 구조**를 수식으로 못박음.

**중요한 스코프 제한**: 이 정리는 엄밀히는 **연속형 혹은 무한히 세분화 가능한 조건화**에 대한
불가능성이다. Phase처럼 **고정·유한·소수(K=3~6개)인 이산 카테고리**라면 이 정리가 직접 막지는 않는다
(§4.1 Vovk 방식으로 exact validity 원리상 가능). 다만 "조건화 단위를 잘게 쪼갤수록 대가가 비선형적으로
급격히 나빠진다"는 정성적 구조는, phase 수를 늘리거나 온라인에서 phase를 거의 연속적으로 식별하려는
시도가 이 impossibility의 사정권에 들어갈 위험을 시사한다 — **이것이 프로젝트의 "★ 중심 미해결 문제
(online phase 식별)" 난이도의 이론적 근거로 인용할 만하다.**

### 4.3 근사적 완화 — 회피 전략

**Romano, Sesia, Candès — "Classification with Valid and Adaptive Coverage"(APS)**, NeurIPS 2020,
arXiv:2006.02544 · [실물확인됨]

Group-splitting이 아니라 **conformity score 설계 자체를 데이터 분포에 적응적으로 만들어**("approximate
conditional coverage") marginal을 지키며 조건부에 근접. 본문에서 명시적으로 Barber et al.의 불가능성
정리를 인용하며 출발점으로 삼음. **그룹 세분화 대가 자체를 정량적으로 다루지 않음**(Mondrian은 참고문헌
[25]로만 등장). → 우려 (b)에 대한 **회피 전략**: phase별로 검출기를 물리적으로 쪼개는 대신, 단일
검출기의 스코어 함수에 phase 정보를 조건부 입력으로 녹여 넣어 전체 calibration 표본을 유지하며 근사적
phase-adaptivity를 얻는 방향. "phase-matched steering"을 반드시 "phase별 완전 분리 검출기"로 구현할
필요는 없다는 대안 설계를 뒷받침.

### 4.4 정량적 Trade-off — 핵심 (우려 b에 가장 직접적인 답)

**Gibbs, Cherian, Candès — "Conformal Prediction With Conditional Guarantees"**, arXiv:2305.12616
(v1 2023-05, v4 2024-09), JRSS-B(accepted) · [실물확인됨(ar5iv, Corollary 1 검증)]

★우려 (b)에 가장 직접적이고 정량적인 답. Marginal과 exact conditional coverage 사이 스펙트�럼을 임의의
covariate shift 함수족 ℱ에 대한 coverage로 재정식화; ℱ가 유한차원(예: K개 그룹 indicator function이
span하는 공간)이면 그룹들에 대해 **동시에 exact finite-sample coverage** 달성 가능(Theorem 2). 그룹
conditional coverage는 다음의 특수 사례로 제시(Corollary 1):

    ℙ(Y_{n+1}∈Ĉ(X_{n+1}) | X_{n+1}∈G) ≥ 1−α                                       (하한, 항상 성립)
    ℙ(Y_{n+1}∈Ĉ(X_{n+1}) | X_{n+1}∈G) ≤ 1−α + |𝒢| / [(n+1)·ℙ(X_{n+1}∈G)]           (상한, 연속성 가정 하)

n = calibration 표본 수, |𝒢| = 그룹(phase) 총 개수, ℙ(X_{n+1}∈G) = 해당 그룹 상대 빈도. **이 부등식이
정확히 우려 (b)를 수식화한 것**: (i) phase 개수 |𝒢|가 늘수록, (ii) 특정 phase의 상대 빈도/길이가
짧아질수록, coverage 편차 상한이 **선형으로** 나빠진다. 이전 접근(Barber et al.=계산 불가능·과보수적,
Vovk Mondrian=그룹 겹침 불허)의 한계를 개선한다고 명시. → 실무: (1) calibration set을 물리적으로
쪼개는 대신 Gibbs-Cherian 방식(단일 joint quantile regression, 유한차원 그룹-지시함수 basis)으로
구현하면 exactness 유지하며 naive splitting보다 표본 효율적. (2) 원하는 최대 편차 ε을 정하면
|𝒢|/((n+1)·min_G ℙ(G)) ≤ ε로 **phase 수·최소 phase 빈도의 하한을 역산 가능**한 정량적 설계 공식 제공.

### 4.5 Many-class 붕괴 조건 — 가장 날카로운 정량 수치

**Ding, Angelopoulos, Bates, Jordan, Tibshirani — "Class-Conditional Conformal Prediction with Many
Classes"**, arXiv:2306.09335, NeurIPS 2023 · [실물확인됨 — 이 문서 작성자가 추가 spot-check(제목·저자
일치 확인; 구체 수치는 하위 조사관의 ar5iv Section 1.2 확인에 의존, 수학적으로 정합적이라 신뢰)]

Class(=phase) 수가 많고 class당 calibration 표본이 적을 때 순수 classwise conformal이 극단적으로
불안정해짐을 지적, 점수 분포가 비슷한 class를 클러스터링해 클러스터 단위로 calibration하는 "clustered
conformal prediction" 제안. **정량 근거**:
- 경험적 붕괴(ImageNet): "water jug"는 목표 coverage 대비 실제 class-conditional coverage 50.8%까지
  undercoverage, "flamingo"는 99.2%까지 overcoverage.
- ★**엄밀한 붕괴 조건**: "For any class y for which |I_y| < (1/α) − 1, we will have q̂_y = ∞, hence any
  prediction set generated by classwise will include y, no matter the values of the conformal scores."
  → **class당 calibration 표본이 1/α−1 미만이면(α=0.1이면 9개 미만) 검출 임계값이 아예 무의미(∞)해짐**
  — conformal quantile 정의상 수학적으로 정합적인 수치(n≥(1−α)/α 필요 조건과 동치).
- 표본이 임계값을 넘겨도 불안정: "class y가 calibration 10개, 90% coverage 목표 시 coverage는
  Beta(10,1) 분포 — 실제 coverage가 80% 미만일 확률 ≈0.107."

→ 실무: |I_y| < 1/α−1 공식은 **phase당 최소 표본수의 직접적 하한**(단, "완전 붕괴 회피"의 극단적
하한이지 "안정적" 기준이 아님 — Beta(10,1) 예시가 보여주듯 실제로는 훨씬 많아야 안정적). 저자들의
**clustered conformal**은 직접 이식 가능한 해법: 표본이 부족한 phase(매우 짧은 sub-phase)를 독립
검출기화하지 말고 스코어 분포가 유사한 인접 phase와 clustering(pooling).

**(참고, 부분확인) Bairaktari, Wu, Wu — "Kandinsky Conformal Prediction"**, arXiv:2502.17264, 2025-02
· [실물확인됨: 존재/제목/저자/abstract만. **구체적 수치("500 샘플/그룹" 등)는 원문 재확인 실패 —
미확인으로 폐기, 인용하지 않음**]

Mondrian conformal의 경직된 격자 구조(비중첩 그룹)를 비판, covariate/label에 걸쳐 겹치거나 부분적
소속을 갖는 그룹까지 유연하게 확장하는 프레임워크. 개념적으로만: phase 경계가 fuzzy/overlapping이면
(rollout마다 경계가 다르면) 이런 겹침-허용 프레임이 표본을 덜 낭비할 가능성 — 정량 근거로는 쓰지 말 것.

### 4.6 이 절 요약

phase처럼 유한·고정된 소수 카테고리에 대한 exact 조건부 coverage는 원리상 불가능하지 않다(Vovk 2012).
하지만 Barber et al.의 impossibility 정리가 보여주듯 조건화를 세밀하게/유연하게 할수록 필요조건이
비선형적으로 급격히 악화되며, 연속 극한에서는 진짜로 불가능(무한 구간)해진다. 정량적으로 가장 직접적인
답 두 개: Gibbs-Cherian-Candès의 **|𝒢|/((n+1)·P(그룹))**(그룹 수·빈도에 선형 비례하는 손해)와 Ding et
al.의 **class당 표본 < 1/α−1이면 붕괴**(+ Beta(n,1) 분산으로 인한 실질적 불안정). 실무 가이드라인:
Vovk의 경험칙(카테고리당 <100 위험) + Ding et al.의 엄밀 하한(1/α−1, 극단치)을 병기해 게이트로 사용.
회피 전략도 문헌에 있음: (i) 표본 부족 phase는 유사 phase와 clustering/pooling, (ii) calibration set을
물리적으로 쪼개지 않고 finite-dim shift-class 공동추정, (iii) 스코어 함수 자체에 phase 정보를 녹여
"쪼개지 않고도" 근사 조건부 성능 확보.

---

## 5. Changepoint Detection + Anomaly Detection 결합 문헌

### 5.1 기반 이론

**Adams & MacKay — "Bayesian Online Changepoint Detection"**, arXiv:0710.3742, 2007 · [실물확인됨]

Changepoint 이전/이후 파라미터 독립 가정 하에 run length(마지막 changepoint 이후 경과시간)의 **사후분포
전체**를 message-passing으로 온라인 갱신하는 정확 베이지안 추론(BOCPD). Finance/biometrics/**robotics**
3개 실데이터 적용. Anomaly detection과의 직접 결합은 이 논문 자체에는 없음. → ★핵심 설계 자체가 이미
"hard 판정이 아니라 soft/확률적 changepoint": 매 시점마다 이진 판정이 아니라 **run-length 전체의
확률분포**를 유지 — 배경에서 요청한 "changepoint 확률을 soft하게 downstream에 반영"의 원형이자, 후속
결합 논문들(§5.2)이 그대로 상속하는 설계 철학. "phase 분류기를 hard cut으로 먼저 확정한 뒤 downstream에
완결된 사실처럼 넘기는" 방식 자체가 BOCPD 원조 철학과 이미 어긋난다는 점이 중요.

**Van den Burg & Williams — "An Evaluation of Change Point Detection Algorithms"**, arXiv:2003.06222,
2020 · [실물확인됨](TCPD benchmark)

37개 실세계 시계열(5인 각각 주석)로 14개 알고리즘(BOCPD 포함) 최초 대규모 실데이터 벤치마크. F1-score +
covering metric 제안. 정확한 수치표는 PDF 바이너리 문제로 미확인. → **"사람 주석자 5명 간에도 changepoint
위치에 완전한 합의가 없다"**는 벤치마크 설계 자체가, changepoint 위치가 하나로 고정된 정답이 아니라
**본질적으로 불확실성이 있는 라벨**임을 보여줌 — phase 경계도 "정답 경계"를 가정하기보다 경계 자체에
불확실성이 있다고 모델링해야 함을 시사(§5.1 soft-changepoint 논지와 연결).

**Truong, Oudre, Vayatis — "Selective Review of Offline Change Point Detection Methods"**,
arXiv:1801.00718, Signal Processing Vol.167 Article 107299, 2020 · [실물확인됨]

140편 이상 서베이, (1)cost function (2)search method (3)change 개수 제약의 3요소 프레임. `ruptures`
패키지 제공. → "change 개수 제약(penalty)을 어떻게 설정하는가"가 곧 오탐(over-segmentation)과
미탐(under-segmentation)의 트레이드오프를 조절하는 표준 손잡이 — phase 개수를 몇 개로 볼지가 이미
편향-분산 선택이라는 프레임을 방법론적으로 뒷받침.

**Gharghabi et al. — Matrix Profile VIII(FLUSS)**, IEEE ICDM 2017[서지만 확인] / 저널확장판
"Domain agnostic online semantic segmentation for multi-dimensional time series", Data Mining and
Knowledge Discovery, PMC6373324 · [실물확인됨, abstract 전문]

파라미터 1개짜리 domain-agnostic 온라인 다차원 세그멘테이션(arc curve 기반). **중요한 부정적 발견**:
이 논문은 **명시적으로 anomaly detection과의 관계를 다루지 않는다** — "개별 phrase/gesture/phoneme
분할에는 관심 없다"고 스코프를 좁히며 semantic segmentation과 anomaly/discord detection을 별개 문제로
취급. **정량**: 사람 대비 성능(FLUSS 평균 오차 0.013 vs 최고 인간 0.011 vs 평균 인간 0.120); 혈역학
사례 93.5% 정확도; 처리속도 실시간 대비 약 36배; 32개 데이터셋에서 경쟁방법(Autoplait) 대비 3승 25패
4무. → 이 발견 자체가 유의미: "국면 분할"과 "이상탐지"는 이 계열 문헌에서 전통적으로 분리된 두 문제로
다뤄져 왔고, 명시적 결합은 상대적으로 최근(2021년 이후, §5.2)에야 시도되기 시작한 선례가 두텁지 않은
영역.

### 5.2 직접 결합 프레임워크 — 핵심 발견군

**Chen & Wu — "Bayesian online collective anomaly and change point detection in fine-grained time
series"**, arXiv:2508.06385, 2025-08 · [실물확인됨]

"Collective anomaly"(국소적으로 뭉쳐 나타나는 이상 구간)와 changepoint가 공존할 때 이 둘의 **joint
online detection**이 거의 연구되지 않았다고 지적, 함께 재귀적으로 추론하는 온라인 베이지안 알고리즘
제시(anomaly 제거로 선형 복잡도 변형도 포함). → "changepoint와 anomaly를 분리된 2단계로 순차 처리하지
말고 하나의 베이지안 추론에서 함께 풀어야 한다"는 문제의식이 최근 통계 방법론 문헌에서 독립적으로 부상.

**Wendelberger, Gray, Reich, Wilson — "Monitoring Deforestation Using Multivariate Bayesian Online
Changepoint Detection with Outliers"**, arXiv:2112.12899, 2021-12 · [실물확인됨]

표준 BOCPD는 **outlier와 실제 changepoint를 혼동**하는 문제가 있음 — outlier 1개가 changepoint로
오판될 수 있음. Outlier 강건성 메커니즘을 BOCPD에 도입, Myanmar 삼림벌채 모니터링 적용. → 우려 (a)의
**거울상**: 우리는 "phase 분류(changepoint) 오차가 anomaly 검출로 전파"를 걱정하지만, 이 논문은 반대로
"anomaly(실패로 인한 비정상 궤적)가 changepoint(phase 경계) 검출 자체를 교란"할 수 있다는 대칭적
위험을 보여줌. 처방: 한쪽을 다른 쪽의 노이즈 모델로 명시적으로 포함(outlier를 별도 잠재변수로 모델링).

**Youssef — "Risk-Calibrated Bayesian Streaming Intrusion Detection with SRE-Aligned Decisions"**,
arXiv:2510.09619, 2025-10 · [실물확인됨]

BOCPD의 **run-length posterior(soft changepoint 확률)**를 SRE 운영비용(에러 버짓) 하에서 최적화된
threshold로 매핑. UNSW-NB15·CIC-IDS2017에서 ECOD/COPOD/LOF 등 baseline 대비 precision-recall·calibration
reliability 개선(정확한 수치 미확인). 99.9% SLO 예시에서 threshold≈0.91 도출 구체 제시. → ★배경에서
요청한 "changepoint 확률을 soft하게 downstream anomaly detection에 반영하는 결합 프레임워크"의 가장
직접적인 실례 — hard changepoint 판정 후 검출기를 스위치하는 게 아니라, run-length posterior 자체를
비용함수에 넣어 연속 확률 가중치로 alert 결정. Phase-matched steering에 옮기면 "phase가 t=k에서
바뀌었다고 확정 후 검출기 스위치" 대신 **phase 사후확률로 가중된 steering 강도**가 이론적 선례를 가짐.

**Youssef — "CALIBURN: Operationally Calibrated Streaming Intrusion Detection with Regime-Dependent
Conformal Risk Control"**, arXiv:2605.24696, 2026(추정 5월) · [실물확인됨] — 위 논문의 후속/확장.

BOCPD(truncated)→isotonic calibration→cost-sensitive thresholding→Conformal Risk Control→burn-rate
alerting의 5단계 파이프라인. 핵심 기여: **calibration과 conformal risk control의 거동이 attack
prevalence(=우리 맥락의 phase에 대응하는 regime)에 강하게 의존**함을 실증. **정량**: 희귀공격 regime
(유병률 5.2%)에서 AUC-PR 0.943(최고 스트리밍 baseline 대비 2.21배, 최고 배치 baseline 대비 4.12배);
isotonic calibration이 Brier score 30% 감소; **단 base-rate가 높은 regime에서는 성능이 유의하게
저하**되며 2가지 구체적 실패 메커니즘(이론적 CRC overshoot, 낮은 alert 예산에서 empirical-density
degeneracy)을 식별. → **regime 기반 전처리+정교한 downstream 보정이 regime에 따라 정말로 깨지는** 가장
직접적이고 정량적인 경고 사례. "phase-conditional 파이프라인은 사전에 실패 모드를 명시적으로 점검해야
한다"는 실무 교훈.

### 5.3 실무 사례의 정량 증거 — 프로덕션 시스템

**Besbes, Mierzwinski, Mujahid, Leitner, Serebrenik, Hunt, Costa — "Exploring Statistical Change Point
Detection Techniques for Performance Anomaly Detection at Mozilla"**, arXiv:2606.18377, 2026 ·
[실물확인됨] — ★이번 조사 전체에서 가장 신뢰도 높은 실측치.

Mozilla 실제 CI 성능 회귀 탐지 시스템(Perfherder, Student's T-test 기반)의 결함률을 정량화, 25개
changepoint detection 방법 + 15개 앙상블을 174개(11인 엔지니어 직접 주석) 실 시계열에 벤치마크.
**정량**: 현재 프로덕션 시스템 기준 **생성된 alert group의 12.5%가 false positive**, **약 6.8%는 자동
시스템이 놓친 실제 회귀(false negative)**. 앙상블 voting으로 교체 시 F1-score 11% 개선(recall-precision
트레이드오프 존재). → 실제 프로덕션 changepoint 기반 anomaly 탐지 시스템에서 changepoint 판정 오차가
**12.5% 오탐 + 6.8% 누락**이라는 무시할 수 없는 크기로 downstream에 직접 반영됨을 보여주는 실측치.
25개 알고리즘 중 단일 최선이 압도적이지 않았다는 것도, phase 경계 판정 하나에 downstream 전체를
의존시키는 hard 파이프라인의 취약성을 뒷받침.

**Mastriani, Costa, Incardona, Munari, Spinello — "Segmentation over Complexity: Evaluating Ensemble
and Hybrid Approaches for Anomaly Detection in Industrial Time Series"**, arXiv:2510.26159, 2025-10,
IEEE SAMI 2026 심사중 · [실물확인됨]

스팀 터빈 산업 시계열에서 changepoint 파생 통계 피처·클러스터링 기반 substructure·hybrid 학습 전략이,
단순 Random Forest+XGBoost 앙상블(세그멘테이션된 데이터 위 학습)보다 일관되게 성능 낮았다는 역설적
결과. **정량**: 최종 채택된 단순 앙상블 AUC-ROC 0.976, F1-score 0.41, 정의된 시간창 내 조기탐지율
100%. **뉘앙스**: 이 논문은 "세그멘테이션을 뺀 모델 vs 넣은 모델"이 아니라 세그멘테이션은 유지한 채
**그 위에 얹는 downstream 모델의 복잡도**를 비교한 것 — "phase-conditional 구조 자체가 이득"의 근거로
과잉해석 금지, "복잡한 changepoint-파생 피처 엔지니어링이 반드시 이득은 아니다"라는 **모델 복잡도 대
세그멘테이션 품질** 경고로 읽어야 함. F1=0.41은 AUC-ROC=0.976 대비 낮아 불균형 데이터에서 운영 임계값
근처 정밀도-재현율은 여전히 취약함도 병기. → "phase 분류기+검출기 구조를 정교하게 만들수록(phase별
세부 conceptor, phase별 hazard function 등) 오히려 단순 global 검출기+단순 세그멘테이션보다 못할 수
있다"는 모델 복잡도 경고 — **프로젝트의 기존 사다리식 ablation 원칙(이전 단계가 신호를 보일 때만 복잡도
추가)과 정확히 부합하는 독립적 실증 사례.**

### 5.4 통계적 검정력 — 우려 (b)의 changepoint 문헌 내 유사 사례

**Li — "How Short Is Too Short? Power Analysis for BIC-Based Changepoint Detection in Ecological
Monitoring"**, arXiv:2603.21154, 2026-03 · [실물확인됨]

생태 모니터링의 짧은 시계열(10~50 관측치)에서 BIC 기반 changepoint detection 검정력을 계열 길이·효과
크기·자기상관 조건별로 정량화. **정량**: 단일 changepoint 탐지에 80% 검정력에는 **n≥30, 효과크기≥2.0**
필요; changepoint 2~3개(다중)는 **n≥50, 효과크기≥5.0** 필요(단일보다 훨씬 큼); AR(1) 자기상관 φ=0.6이면
검정력 **40% 감소**; PELT는 중간 수준 자기상관에서도 85~91% 검정력 유지(Binary Segmentation보다 강건);
효과크기<1.5면 changepoint detection보다 early-warning-signal 검정이 나음. **구분 필요**: 이 논문이
정량화한 것은 "changepoint 탐지 자체의" 검정력이지 우리 우려 (b)가 지목하는 "phase로 쪼갠 뒤 그 안에서
학습하는 downstream 검출기"의 검정력은 아니지만, 같은 통계 메커니즘(표본 축소→검정력 급락)이 작동하는
정량적 유비로 유용. → **RoboCasa rollout(실패 시 약 45 step)을 3~4 phase로 쪼개면 phase당 프레임 수가
이 논문이 "위험 구간"으로 표시한 n<30~50 대역에 바로 들어간다.** Episode 수를 곱해 표본을 늘릴 수
있어도(phase 라벨은 per-record 유지 규약), phase 경계 자체의 불확실성(±수 프레임)이 각 phase 표본에
라벨 노이즈로 섞이는 문제는 episode 수 증가로 해결되지 않음.

### 5.5 대안: 2단계 파이프라인을 우회하는 Joint 모델링

**Melnyk, Banerjee, Matthews, Oza — "Semi-Markov Switching Vector Autoregressive Model-based Anomaly
Detection in Aviation Systems"**, arXiv:1602.06550, 2016 · [실물확인됨]

★항공 안전 도메인(비행 phase: 이륙/상승/순항/하강/착륙 — 로봇 조작 subtask phase와 구조적으로 매우
유사)에서, 각 비행을 **semi-Markov switching VAR(SMS-VAR) 모델 하나**로 표현하고 모델 예측-실제 관측
괴리로 anomaly 탐지. Regime(phase) 전환과 anomaly 탐지를 분리된 2단계가 아니라 **하나의 통합 확률모델
안에서 동시에** 다룸. 병렬화 가능해 온라인 탐지 적용 가능. 정량 근거는 미확인(정성적 실증만 abstract
수준 확보). → phase 구조를 가진 궤적의 이상탐지에서, "phase를 먼저 hard하게 확정한 뒤 phase별 검출기를
얹는" 구조 대신 **regime-switching과 이상탐지를 하나의 생성모델로 묶어 동시 추론**하는 대안 설계가 이미
10년 전부터 존재. 우려 (a)에 대한 구조적 해법 후보: phase 분류와 검출을 순차 파이프라인이 아니라
**공유 잠재변수를 갖는 단일 확률모델**로 설계하면, phase 오분류가 "돌이킬 수 없는 downstream 입력 오류"
가 아니라 **posterior 불확실성으로 자연히 흡수**됨.

### 5.6 확인 실패 / 귀속 오류로 배제한 항목 (규율 준수 기록)

- "PELT로 MetaWorld(β=20)/ManiSkill2(β=30) 로봇 조작 궤적을 phase로 분할한다"는 주장을 WebSearch
  스니펫이 "PRIMT"(arXiv:2509.15607)에 귀속시켰으나, 해당 논문을 직접 열어보니 changepoint/PELT/phase
  segmentation 언급이 **전혀 없었음** — 귀속 오류, 출처 불명으로 폐기.
- "Contact-rich 로봇 조작에서 BOCPD로 force/torque 기반 과분할을 억제한다"는 서술 — 여러 스니펫에
  반복 등장했으나 특정 단일 논문을 확인하지 못해 미확인으로 남김.
- TCPD benchmark의 알고리즘별 실제 F1/covering 수치표 — PDF 바이너리 문제로 미확인(정성적 결론만 확인).

### 5.7 이 절 요약

우려 (a)는 프로덕션 실측치(Mozilla, 12.5%+6.8%)와 regime 의존적 붕괴 메커니즘(CALIBURN)으로 정량
확인됨. 처방: hard 파이프라인을 soft/joint 구조로(BOCPD의 run-length posterior를 그대로 downstream에
흘리기, 또는 Melnyk et al.처럼 phase-switching과 이상탐지를 단일 확률모델로 통합), 복잡도는 신호가
있을 때만 추가(Mastriani et al.). 우려 (b)는 changepoint 문헌에서 n<30~50 위험대역으로 정량화되며
(Li 2026), RoboCasa rollout에 직접 유비 성립. 결합 문헌 전반(2021년 이후 부상)은 "나누기(hard
segmentation)와 검출을 분리하지 말라"는 방향으로 수렴하며, 이는 프로젝트의 "사다리식 ablation"·"no
rollout pooling(per-record 유지)" 원칙과 정합적.

---

## 6. 종합 — 우리 프로젝트에 대한 실무 처방

### 6.1 다섯 주제를 관통하는 수렴 패턴

| 패턴 | 근거 (대표 논문) |
|---|---|
| Hard 파이프라인은 오차를 절대 복구 불가능하게 만든다 | Silla&Freitas 2011(blocking problem), Viola-Jones 2001(곱셈 공식) |
| 표준 완화책은 "hard argmax를 soft/확률적 전달로" | Bourdev&Brandt 2005, BOCPD 2007(원설계), Yu&Qin 2008, Soft MoE 2024, Gibbs-Cherian 2023 |
| 단, hard 자체가 원죄는 아니다 — 붕괴 방지가 핵심 | Hash Layers 2021, Nguyen et al. ICLR2024(개수 맞으면 무료) |
| Joint/end-to-end가 pipeline을 이기지만 잘못 설계하면 역효과 | Andor et al. 2016, Yan et al. 2022, Melnyk et al. 2016, Chen&Wu 2025 |
| 조건 개수·조건당 표본에 정량적 하한이 있다 | Ding et al. 2023(1/α−1), Gibbs-Cherian 2023(|𝒢|/(n·P(G))), Vovk 2012(~100), Li 2026(n≥30) |
| 조건화는 "조건이 적고 잘 정의되고 알려졌을 때"만 유리 | Han et al. 2024 survey(명시적 진술) |
| 복잡도는 신호가 있을 때만 추가하라 | Mastriani et al. 2025 — 프로젝트 기존 원칙과 독립적으로 부합 |
| "phase 오분류의 harm-side"를 직접 측정한 논문은 못 찾음 | Bindini et al. 2026(TMLR, benefit-side만), Ahmad et al. 2024(동일 패턴) → 연구 공백 |

### 6.2 "언제 이득/손해인가"에 대한 정량적 답 (있는 것만)

- **국면 개수가 진짜 구조와 정확히 일치 + 국면 수가 적음(≤6 수준)**: 통계적으로 무료(Nguyen et al.
  ICLR 2024). 이것이 틀렸을 때(특히 과대분할)만 손해가 집중.
- **phase당 최소 표본**: 완전 붕괴 회피 하한 = 1/α−1(목표 커버리지 1−α 기준, Ding et al. 2023); 안정적
  운용을 위한 경험적 하한 ≈ 100(Vovk 2012); changepoint 검정력 기준 n≥30·효과크기≥2.0(Li 2026). 셋 다
  독립적으로 "수십~100" 대역에 수렴 — RoboCasa rollout 실패 시 ~45 step을 3~4 phase로 쪼개면 이 경계에
  바로 걸린다는 점을 실무 경고로 채택할 만하다.
- **조건화 편차 상한의 명시적 공식**: |𝒢| / ((n+1)·P(그룹)) (Gibbs-Cherian-Candès 2023) — phase 수와
  phase 빈도의 역수에 선형 비례. 목표 편차 ε을 정하면 역산해서 "phase를 몇 개까지 쪼개도 되는가"를
  계산할 수 있는 유일한 정량 설계 공식.
- **cascade 오차 전파의 실측 규모감**: 정상 조건에서 ~1.8~2%p(Andor et al., NLP), 문제 있는 계층
  분류에서 ~19.6%가 1단계 기원(Ren et al.), 프로덕션 changepoint 시스템에서 12.5%+6.8%(Mozilla)까지
  분포 — "약간의 손해"부터 "무시 못할 손해"까지 폭넓게 실재.

### 6.3 표준 처방 체크리스트 (프로젝트에 바로 적용 가능한 형태)

1. **phase 개수를 최소화**하고, 실제 rollout 구조(reach/grasp/place 등)와 일치하는지 먼저 검증 —
   개수를 틀리는 것(특히 과대분할)이 hard/soft 선택보다 더 큰 리스크(Nguyen et al. ICLR2024, Han et
   al. 2024).
2. **phase 분류 결과를 hard argmax로 확정해 검출기에 넘기지 말 것.** 확률분포(soft weight) 또는 최소
   top-k phase 후보를 유지해 검출기 쪽에서 가중 결합(Bourdev&Brandt, Yu&Qin, Soft MoE, Weiss&Taskar의
   "oracle recall을 목적함수에 포함" 등 다중 독립 근거).
3. **phase당 최소 표본을 게이트로 사용**: 수십~100 표본 미만인 phase는 독립 검출기화하지 말고 인접
   phase와 pooling/clustering(Ding et al. 2023의 clustered conformal이 직접 이식 가능한 레시피).
4. **검출기 훈련 데이터에 phase 분류기의 실제 오분류 패턴을 주입**(gold phase label로만 학습 금지) —
   Bengio et al. 2015 scheduled sampling의 train/test mismatch 논리.
5. **가능하면 phase 분류와 검출을 joint/공유잠재변수 모델로 묶기**(Melnyk et al. 2016 semi-Markov
   switching VAR가 항공기 phase에서 보여준 정확한 선례) — 단, "제대로 설계하지 않은 joint는 pipeline
   보다 못할 수 있다"(Yan et al. 2022)는 경고를 함께 유지, 반드시 pipeline과 통제 비교.
6. **phase 경계가 애매한 timestep은 명시적으로 다룰 것** — flatten/skip(Naik&Rangwala, +7%p) 또는
   confidence-gated 판단 유보(Mokssit et al. 2025). Rastegar(2026)의 온도-스펙트럼 관점처럼, 위험은
   경계 근처에 집중되고 경계가 명확한 구간에서는 hard/soft 차이가 작다.
2. **곱셈적 성능 저하를 기본 가정하되 stage 간 오차의 상관구조를 반드시 실측**: "phase 분류 정확도"와
   "검출기 성능"을 독립적으로 측정해 곱으로 낙관 추정하지 말 것(Mangal et al. 2023의 97%→11% 붕괴가
   극단적 반례). Phase 무관 단일 검출기 대비 end-to-end ΔAUROC/ΔSR을 반드시 직접 비교.
8. **복잡도는 이전 단계가 신호를 보일 때만 추가**(Mastriani et al. 2025로 독립 재확인됨) — 프로젝트가
   이미 채택 중인 사다리식 ablation(global → pathway-split → phase-bin) 원칙 그대로 적용.

### 6.4 연구 공백 — 프로젝트의 기회

2026년 최신 문헌(Bindini et al. TMLR, Ahmad et al.)조차 "phase/context가 정확히 주어졌다"는 가정
위에서 이득만 보고하며, **"phase 분류기 자체가 틀렸을 때 검출 성능이 정량적으로 얼마나·어떻게
저하되는가"를 체계적으로 측정한 논문은 이번 조사(60편 이상)에서 찾지 못했다.** Regime-conditional AD,
changepoint+AD 결합 문헌 양쪽에서 독립적으로 같은 공백이 확인됨(우연이 아니라 서브필드 전반의 공백으로
보임). 이는 CLAUDE.md의 "★ 중심 미해결 문제(추론 중 online phase/failure-type 식별 가능한가)"가 로보틱스/
VLA에 국한된 질문이 아니라 **일반 ML 문헌에서도 여전히 열린 질문**임을 뒷받침한다 — 프로젝트가 이 harm-side
(phase 오분류 → 검출 성능 저하의 정량 곡선)를 직접 측정한다면, 문헌 공백을 메우는 것이므로 그 자체로
방법론적 기여가 될 수 있다.

---

## 부록: 조사 중 확인 실패로 폐기한 항목 전체 목록

- Bairaktari, Wu, Wu(Kandinsky Conformal Prediction, arXiv:2502.17264)의 "500 샘플/그룹, 250 샘플/class"
  구체 수치 — 원문 재확인 실패, 폐기(§4.5).
- Ruggieri et al.(Algorithms 2020)의 "44/67 시나리오에서 hard EM 우세" 수치 — abstract에 없음, 폐기(§3.5).
- GShard(Lepikhin et al. 2020)의 top-2 random noise routing 세부 메커니즘 — 본문 미파싱, 배경지식으로만
  언급, 정량 근거로 미사용(§3.2).
- Jordan & Jacobs 1994(Hierarchical Mixtures of Experts and the EM Algorithm) — 서지 다중교차 확인됨이나
  본문 403, EM soft-assignment 관련 서술은 배경지식으로만 표시(§3.1).
- "PELT로 MetaWorld/ManiSkill2 로봇 조작 궤적 분할"(arXiv:2509.15607 PRIMT 귀속) — 원문에 해당 내용
  전무, 귀속 오류로 완전 폐기(§5.6).
- "Contact-rich 로봇 조작 BOCPD force/torque 과분할 억제" — 특정 논문 미특정, 미확인(§5.6).
- TCPD benchmark 알고리즘별 F1/covering 수치표 — PDF 미파싱, 미확인(§5.6).
- "MDPI Machines 2026, R² 0.64→0.86" state-aware health assessment — WebFetch 403 반복 실패, 저자명도
  미확인, 참고용으로만 표시하고 결론에 미사용(§1.4 각주 상당).
- Yoon et al.(2026, When Are Experts Misrouted?)의 misrouting이 최종 성능에 미치는 정확한 % — 논문
  뒷부분(Table 3/9, Fig.7) 미접근으로 정량치 미확인, 정성적 결론만 채택(§3.5).
