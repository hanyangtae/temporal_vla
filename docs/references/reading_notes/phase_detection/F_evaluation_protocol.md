# 실패 검출기 공정 평가 프로토콜 — phase별 검출기 vs 단일 SAFE

- 조사일: 2026-08-10. 조사자: 평가 방법론 조사(본 노트).
- 배경: SAFE 계열 검출기(정책 latent → 실패확률)를 재현·보유(`docs/seen18_safe_detector_verification.md`).
  다음 단계로 **phase별 특화 검출기**를 만들어 원래(단일) SAFE와 비교하려 함. 과거 **길이 confound**
  (실패=항상 timeout=길고, 성공=조기종료=짧음 → 시간-pooled 지표가 아티팩트, [[seen18-rollout-length-confound]])
  에 데인 경험이 있어, 이번엔 평가 프로토콜을 먼저 문헌으로 굳히고 싶음.
- **출처 신뢰도 표기 규칙(이 문서 전체 적용)**:
  - `[로컬원문]` = 이번 세션에 `docs/references/SAFE.txt`(PDF에서 추출된 전문) 직접 grep·인용.
  - `[프로젝트기존노트]` = `docs/Activation_steering_basic/notes/*.md`의 기존 tier=must 정독 노트 재인용(이전 세션에서 로컬 PDF 확인됨, 이번 세션엔 재확인 안 함).
  - `[웹확인-본문]` = 이번 세션 WebFetch로 arXiv HTML/PDF 본문에서 직접 인용문 확보.
  - `[웹확인-초록]` = 이번 세션 WebSearch/WebFetch로 초록·2차 요약 수준만 확인(본문 미확인).
  - `[고전-2차출처]` = 원문이 유료·pre-arXiv라 여러 2차 출처(공식 저널 페이지·구현체·해설)로만 교차확인.
  - `[미확인]` = 논문 근거 없이 쓰는 일반 관행 서술. 이 표기가 없는 문장은 위 5개 중 하나가 암묵 적용된 것.

## 확인한 문헌 목록

| 문헌 | 신뢰도 | 한 줄 |
|---|---|---|
| SAFE, arXiv:2506.09937 (NeurIPS 2025) | `[로컬원문]` | 길이 confound를 **명시 경고+직접 통제**(task별 min-length T truncation, s_T가 헤드라인). functional CP는 **pooled**(task 안 가림) calibration. 스스로를 "task별 분리 캘리브레이션" 구파와 대비시킴 — §4 참조. |
| FIPER, arXiv:2510.09459 (NeurIPS 2025) | `[프로젝트기존노트]` | RND-OE(관측 OOD)+ACE(action entropy) AND 결합, sliding-window 집계, 성공만으로 functional CP. TWA/Acc/DT 3지표. |
| Sentinel/STAC, arXiv:2410.04640 (CoRL 2024) | `[프로젝트기존노트]` | η_t 단조누적 + 종단값 conformal → split CP로 FPR≤δ 증명(Prop.1/2). 실패유형 2분법(erratic/progression) 병렬 OR 결합. |
| KnowNo, arXiv:2307.01928 (RSS 2023) | `[프로젝트기존노트]` | conformal 배경 지식. **non-causal 시퀀스 CP ⇔ causal per-step 재구성 동치**(Claim 1) — phase-bin 온라인 적용의 이론적 근거. |
| Hide-and-Seek in Trajectories, arXiv:2605.30834 (2026-05) | `[웹확인-본문]` | SAFE를 baseline으로 직접 비교(bACC +2.9~+11.7%p). **trajectory-level 라벨을 전 timestep에 균일 전파하는 문제**를 정면 지적 — phase-국소화 동기가 우리 문제의식과 근접. **단, 길이confound는 언급 없음**(SAFE의 안전장치를 계승하지 않음, 아래 §2 참조). |
| SAFECAST, arXiv:2608.04246 (2026-08) | `[웹확인-초록]` | SAFE형 hidden-state probe + functional CP를 **분포이동(deployment shift) 하에서 재보정**하는 문제 — calibration 데이터가 배포조건과 안 맞으면 보장이 깨진다는 경고, §5와 연결. |
| VLA-FAIL, arXiv:2606.21386 (2026-06) | `[웹확인-초록]` | threshold-independent 결합지표 **AUCPDT**(정확도·재현율·검출시간 결합) 제안. 상세 미확인. |
| FAIL-Detect, arXiv:2503.08558 (RSS 2025) | `[웹확인-초록]` | OOD+CP 프레임(FIPER가 "가장 가까운 선행연구"로 인용). 본문 추출 실패, 상세 미확인. |
| How VLAs Fail Differently, arXiv:2605.28726 (2026-05) | `[웹확인-초록]` | 아키텍처(이산 토큰 vs 연속)마다 최적 검출 신호가 다름(가속도 신호는 이산에만 유효, AUROC 0.88 vs 연속 0.52) → **"단일 검출기가 보편적으로 안 통한다"**는 우리 phase-조건부 가설과 구조적으로 유사한 사례(단, 축이 phase가 아니라 architecture). 450 episode 동일조건 비교. |
| Cawley & Talbot 2010, JMLR 11:2079-2107 | `[웹확인-초록]` | **nested CV** 부재 시 모델선택(threshold/hparam 선택 포함)이 held-out 성능을 낙관적으로 부풀림 — 그 크기가 알고리즘 간 실제 성능차와 맞먹을 수 있음. |
| Gelman & Loken 2013 (Columbia, unpublished but highly cited) | `[웹확인-초록]` | "forking paths" — 사후에 그룹 경계·분석법을 데이터-의존적으로 고르면, 의식적 p-hacking 없이도 사실상 다중비교 문제 발생. |
| Vovk, Gammerman & Shafer 2005 (책, *Algorithmic Learning in a Random World*) — Mondrian CP | `[웹확인-초록]` | **그룹별(class/phase-bin) 사전 지정 partition** 안에서 conformal 보정 → per-group coverage 보장. 사후 그룹 선택이 아니라 사전 partition이 핵심 조건. |
| Romano, Barber, Sabatti, Candès 2020, HDSR — "With Malice Toward None" | `[웹확인-초록]` | Mondrian을 적응형(adaptively identified) 서브그룹까지 확장한 equalized coverage — 그래도 그룹 정의 함수 자체는 미리 고정. |
| Barber, Candès, Ramdas, Tibshirani 2021, Information & Inference (arXiv:1903.04684) | `[웹확인-초록]` | **정확한 conditional(점별/무한소 그룹) coverage는 분포무관 방법으로 원천 불가능** — Mondrian 같은 "성긴 partition"이 실전 타협점이라는 이론적 상한을 제공. |
| Ding et al. 2023, arXiv:2306.09335 — Class-Conditional CP with Many Classes | `[웹확인-본문]` | 그룹(클래스) 수가 많고 그룹당 calibration 표본이 적으면 개별 그룹 임계값이 불안정 → **유사 그룹 clustering 후 cluster 단위 CP**. phase-bin이 많고 rollout이 적은 우리 상황과 구조적으로 동일. |
| DeLong et al. 1988 (Biometrics) | `[고전-2차출처]` | 같은 표본에 대한 두 상관된 AUROC의 유의차 검정(U-statistic 공분산). paired 검출기 비교의 표준 도구. |
| Davis & Goadrich 2006, ICML | `[웹확인-초록]` | 클래스 불균형(실패가 희귀) 하에서는 PR curve가 ROC보다 판별력 있는 그림을 줌; ROC dominance ⇔ PR dominance 동치 증명. |
| Tatbul et al. 2018, NeurIPS (arXiv:1803.03639) | `[웹확인-초록]` | range-based precision/recall — point 단위가 아니라 구간(anomaly interval) 단위로 정밀도/재현율 재정의, 시간축 tolerance 파라미터화. |
| Wu & Keogh 2021, IEEE TKDE (arXiv:2009.13807) | `[웹확인-초록]` | TSAD 벤치마크의 구조적 결함(특히 **point adjustment**가 노이즈 예측을 유리하게 함) — "평가 프로토콜이 관대하면 진보가 착시가 된다"는 일반 경고. |
| Veeravalli & Banerjee, *Quickest Change Detection* (survey, arXiv:1210.5552) | `[웹확인-초록]` | CUSUM/ARL(average run length) — "false-alarm 상한 하에서 detection delay 최소화"라는 detection-delay 지표들의 이론적 원형. |
| Zhang et al., OpenOOD v1.5, arXiv:2306.09301 | `[웹확인-초록]` | OOD 검출기 비교를 위한 **표준화된 벤치마크 프로토콜**(고정 backbone·고정 평가 코드) 사례 — "동일 조건 보장"을 인프라로 강제한 선례. |

---

## 1. 표준 지표 — episode AUROC / per-timestep AUROC / TPR@FPR / detection delay / precision-recall

로봇 실패검출 논문들이 실제로 보고하는 지표는 서로 다른 이름을 쓰지만 **정확도축 + 시간축(적시성)** 두 축의 조합으로 수렴한다.

- **episode-level ROC-AUC (길이통제 s_T)** — SAFE의 헤드라인(Table 1). `[로컬원문]` 매 task마다 성공/실패 rollout을 공통 길이 T(그 task의 최소 rollout 길이)까지 잘라, 그 시점의 점수 s_T로 ROC-AUC 계산. "episode-level"이지만 "그 시점까지 본" 정보만 쓰므로 causal.
- **TPR/FPR/balanced-accuracy @ 특정 α** — 임계값 δ_t를 conformal α로 확정한 뒤의 이진 판정 지표. `[로컬원문]` SAFE §6.1: "we consider the following metrics: true positive rate (TPR), false positive rate (FPR), balanced accuracy (bal-acc), and averaged detection time (T-det)". Bal-Acc = (TPR+TNR)/2.
- **detection time / time-to-detection (T-det, DT)** — 실패 rollout에서 경보가 뜬 시점(정규화 여부는 논문마다 다름). SAFE는 raw step, Hide-and-Seek은 `[웹확인-본문]` "detection time (normalized by trajectory length)"로 **rollout 길이로 나눠 정규화** — 이게 그 자체로 길이 confound의 부분적 완화(짧은 rollout과 긴 rollout의 DT를 같은 [0,1] 척도에 놓음). 단, 정규화가 정오탐 판정(ROC-AUC)의 길이 confound까지 없애주진 않음(별개 문제, §2).
- **정확도-적시성 결합 단일 스칼라**: FIPER의 **TWA**(Time-Weighted Accuracy, "정확·조기 검출에 가중치") `[프로젝트기존노트]`, Hide-and-Seek도 동일 명칭의 TWA 채택 `[웹확인-본문]`, VLA-FAIL의 **AUCPDT**(Area-Under-Curve of Precision-Detection-Time, threshold-independent) `[웹확인-초록]`. 세 지표 모두 "정확도 vs 지연"이라는 2차원 trade-off를 곡선(α-sweep) 대신 단일 숫자로 접자는 시도 — 원형은 고전 순차분석의 **ARL(average run length) vs detection delay** trade-off(Lorden 1971 최적성, CUSUM) `[웹확인-초록]`.
- **precision-recall (episode 또는 range 단위)**: 실패가 희귀 이벤트(class imbalance)일 때 ROC보다 PR이 방법 간 차이를 더 잘 드러냄(Davis & Goadrich 2006) `[웹확인-초록]`. 시계열 특유의 "구간(anomaly interval) 단위" precision/recall 재정의는 Tatbul et al. 2018(NeurIPS) `[웹확인-초록]` — 여러 timestep에 걸친 하나의 실패를 어떻게 "몇 번 검출로 셀지"(existence/overlap/cardinality reward를 파라미터화)를 정식화. 로봇 실패검출 논문에서 이 range-PR을 그대로 쓰는 사례는 확인 못 함(로봇쪽은 episode 단위 라벨이 표준이라 point-vs-range 이슈가 상대적으로 약함) — **적용 시 직접 검증 필요, 현재 근거는 시계열 일반론뿐**.
- **per-timestep AUROC**: SAFE의 s_t는 매 t마다 정의되지만, 헤드라인 지표는 (앞서 말한 대로) "고정 T에서의" 단면 AUROC 하나다. "모든 t에서 AUROC를 각각 구해 곡선으로 본다"는 방식은 SAFE Fig.4(α-sweep bal-acc/T-det trade-off)에 더 가깝고, 순수 "per-timestep AUROC 곡선"을 메인 지표로 쓰는 사례는 이번 조사에서 확인 못 함 `[미확인]`.

## 2. 길이(episode length) confound 통제 — 명시적으로 다루는 문헌

- **SAFE §6.1 / Appendix B.5 — 가장 명시적이고 직접적인 처리** `[로컬원문]`. 원문(그대로 인용, `docs/seen18_safe_detector_verification.md`에 이미 옮겨져 있음):
  > "…if a failure detector simply learns to count the time elapsed, i.e., s_t = t, it will achieve perfect failure detection since failed rollouts have a fixed and longer duration. To ensure a fair comparison, for evaluation in Table 1, we compute the minimum rollout length for each task and use that as T for that task. The failure detection performance (in ROC-AUC) is then determined based on s_T, where T is the same for all successful and failed rollouts within each task."

  즉 **task별 min-length T truncation**이 유일하게 검증한 "명시적으로 confound를 지적하고 직접 통제한" 방법론이다. 이게 우리 프로젝트의 `truncation-length-standard`(성공 길이 [mean, mean+1σ])의 직접 선행연구이자 대조군.
- **후속 VLA 실패검출 논문들이 이 안전장치를 계승하지 않는 사례 확인됨 — 경고**. Hide-and-Seek(2605.30834, SAFE를 baseline으로 직접 능가한다고 주장하는 2026-05 논문)의 §4.2/§5.1을 직접 조회했을 때 `[웹확인-본문]`, window aggregation(비중첩 sliding window 평균)과 "detection time normalized by trajectory length"는 있지만, **"실패 rollout이 항상 더 길다"는 문제 자체를 지적하거나 SAFE식 min-length T truncation을 쓰는 서술은 찾지 못했다**. 이는 (a) Hide-and-Seek이 이 confound를 이미 다른 방식으로 우회했거나(coarse-to-fine 국소화 라벨이 부수적으로 완화했을 가능성), (b) 실제로 안전장치가 빠졌거나 둘 중 하나인데, **이번 조사로는 구분 불가** — 이 논문 수치를 인용할 땐 별도 검증 없이 "confound-free"로 가정하지 말 것.
- **COAST — 대조군(비통제)**. `[프로젝트기존노트]` 이미 프로젝트에 문서화됨(`docs/seen18_safe_detector_verification.md` §4): 전체 rollout 길이를 action-token mean-pool해 conceptor(class-wise covariance)에 그대로 투입, "normalized trajectory time"은 시각화용이지 fit 시 길이 매칭이 아님.
- **일반 시계열 이상탐지 문헌의 인접 경고(로봇 특화 아님)**: Wu & Keogh 2021(IEEE TKDE) `[웹확인-초록]`은 로봇이 아니라 범용 TSAD 벤치마크를 감사한 논문이지만, "point adjustment(하나라도 맞히면 그 이상구간 전체를 맞힌 것으로 카운트)가 노이즈 예측을 유리하게 만들어 진보를 착시로 만든다"는 지적은 "평가 관대함이 방법론 차이를 가린다"는 점에서 우리 길이-confound 문제와 **구조적으로 동류**(단, 메커니즘은 다름 — point-adjustment는 채점 관용성 문제, 우리 길이confound는 라벨-누수형 shortcut 문제). Tatbul et al. 2018의 range-based PR도 "구간 하나를 어떻게 셀지"를 명시적으로 파라미터화해 이런 관대함을 통제하려는 시도.
- **matched-length 샘플링(성공/실패를 같은 길이대만 뽑아 비교)**: 명시적으로 이 정확한 기법을 쓰는 로봇 실패검출 논문은 이번 조사에서 못 찾음 — SAFE의 "task별 공통 T truncation"이 사실상 이것의 한 형태(truncation은 matched-length의 특수케이스: 짧은 쪽 길이에 맞춰 자름)지만, "성공 표본을 실패 길이 분포에 맞춰 subsample/재가중"하는 역방향 matched-length는 `[미확인]` — 일반 통계학의 case-control matching 개념을 로봇 실패검출에 이식하는 것은 우리 자체 아이디어로 취급해야 함.
- **시간 정규화(time normalization)**: Hide-and-Seek의 "DT normalized by trajectory length" `[웹확인-본문]`가 유일하게 확인된 사례 — 단, 이건 **검출시간 지표의 정규화**이지 **AUROC 계산 자체의 길이confound 통제**가 아님(둘을 혼동하지 말 것 — 위 SAFE truncation과는 다른 문제를 푼다).

## 3. 검출기끼리 비교할 때 동일 조건 보장

- **같은 calibration 절차, pooled 단일 threshold — SAFE 자신의 설계 선택** `[로컬원문]`. SAFE는 스스로를 명시적으로 "task별로 따로 학습·보정하는" 구파와 대비시킨다:
  > "Unlike existing methods that train and calibrate separate classifiers per task, SAFE uses a single unified failure detector and works effectively on generalist policies like VLAs."

  그리고 실제 calibration은 `D_eval-seen`(여러 task를 **pooled**한 성공 rollout 집합) 하나에서 functional CP band 하나를 만들고, 그 **동일한 δ_t**를 모든 unseen task 평가에 적용한다(§6.2, `[로컬원문]`: "we use D_eval-seen to calibrate the functional CP band C_α and evaluate on D_eval-unseen"). **이 대비 구도(단일 pooled 검출기 vs task/group별 분리 검출기)가 정확히 우리가 지금 하려는 비교(단일 SAFE vs phase별 검출기)의 거울상**이다 — SAFE는 "그룹별 분리"를 자신이 극복한 구파로 규정했으므로, 우리 phase 검출기가 SAFE를 이긴다는 주장을 하려면 그 역방향 논증(그룹별 분리가 실제로 더 나은 경우가 언제인지)을 우리가 직접 방어해야 한다는 뜻이기도 하다.
- **SAFE 스스로 인정하는 균열**: `[로컬원문]` "calibration and evaluation may not come from the same distribution... TNR may deviate from the gray dashed line (1−α)" — calibration set(seen, pooled)과 test set(unseen)의 분포가 다르면 marginal FPR 보장 자체가 깨질 수 있음을 인정(Appendix C.2). 이건 우리가 phase-conditional 검출기를 만들 때 그대로 재발할 위험: **phase-bin별 calibration set이 그 phase-bin의 test 분포와 exchangeable하지 않으면(예: calibration에 쓴 rollout이 특정 seed/scene에 쏠림) 그 그룹의 FPR 보장이 깨진다.**
- **paired 통계검정 — 같은 rollout 집합에 두 검출기를 모두 태워 비교**: DeLong et al. 1988 `[고전-2차출처]`는 같은 테스트 표본에 대한 두 상관된 AUROC 차이의 유의성을 U-statistic 공분산으로 검정하는 표준 방법 — "SAFE AUROC=0.68, phase-detector AUROC=0.73, 차이가 유의한가"를 답하려면 두 검출기를 정확히 같은 held-out rollout 집합에 태우고 DeLong(또는 bootstrap paired CI)을 써야 한다. 이는 project가 ΔSR 비교에 이미 쓰는 McNemar/Holm 관행(`docs/steering/PITFALLS.md` §7)과 같은 계열이고, **detector AUROC 비교에도 동일 원칙이 적용되어야 함**(다만 도구는 McNemar가 아니라 DeLong/bootstrap — AUROC는 연속 스코어라 이진 불일치쌍 검정보다 DeLong류가 더 적합).
- **인프라로 강제한 "동일 조건"**: OpenOOD v1.5(arXiv:2306.09301) `[웹확인-초록]`는 OOD 검출기들을 비교할 때 "고정 backbone + 표준화된 평가 코드"로 조건을 강제하는 벤치마크 인프라 사례 — 로봇 실패검출엔 이런 표준 벤치마크가 아직 없지만(각 논문이 자체 LIBERO/RoboCasa 서브셋을 씀), **동일 rollout pool·동일 feature 추출 파이프라인·동일 CP α**를 코드 레벨에서 강제하는 게 원칙적으로 같은 방향.
- **아키텍처 간 이질성이 만드는 "공정 비교" 함정의 사례**: How VLAs Fail Differently(2605.28726) `[웹확인-초록]`는 450 episode를 세 아키텍처(VQ-BeT/Diffusion/ACT)에 동일 조건으로 태워 "단일 신호(속도 위반)가 이산 토큰 아키텍처에서만 통한다(AUROC 0.88 vs 연속 0.52)"는 걸 보였다 — 이는 phase 대신 **architecture**가 조건부 변수인 사례지만, "그룹마다 최적 검출기가 다를 수 있다"는 우리 가설의 구조적 유사 선례로 참고 가치가 있다(단, 이 논문이 사후선택·보정 문제를 어떻게 다뤘는지는 초록 수준이라 미확인).

## 4. 조건부(그룹별) 검출기 vs 단일 검출기 — 사후선택 함정과 방지 프로토콜 (핵심)

이 질문에 정확히 대응하는 로봇 실패검출 논문은 확인 못 했다 `[미확인]` — 그래서 통계학·conformal prediction·ML 방법론 일반 문헌에서 원리를 가져와야 한다. 확인한 4개 축:

1. **문제의 정확한 형태(사후 선택 편향)**: phase bin마다 threshold(또는 아예 다른 모델)를 고르고 나서, "가장 잘 맞은 bin"이나 "phase-conditional 검출기의 전체 성능"을 보고하면, 이는 데이터를 보고 나서 자유도(어느 bin 경계를 쓸지, 몇 개로 나눌지, 어느 bin의 threshold를 얼마로 잡을지)를 조정한 것과 같다.
   - Cawley & Talbot 2010(JMLR) `[웹확인-초록]`: 모델선택(hyperparameter/threshold 포함)에 쓴 데이터로 그대로 성능을 보고하면 **일반화 성능이 낙관적으로 부풀려지고, 그 부풀림의 크기가 알고리즘 간 실제 차이와 맞먹을 수 있다**는 것을 실증. 해법은 **nested cross-validation** — threshold/hparam 선택은 안쪽 fold에서만, 바깥쪽 test fold는 선택 과정에 한 번도 노출되지 않아야 함.
   - Gelman & Loken 2013("garden of forking paths") `[웹확인-초록]`: **의도적 fishing 없이도** — "이 정도면 유의미한 bin 경계였다"는 사후 판단만으로도 사실상 다중비교를 한 것과 같은 효과가 생긴다. 우리 맥락 번역: phase bin 경계를 눈으로 보고("이 구간에서 잘 갈리네") 조정하면, 그 경계가 "사전에 고정"된 것처럼 보고해도 이미 오염된 것.
2. **원칙적 해법 — 그룹은 사전에 고정(pre-registered), calibration은 그룹별로 하되 선택은 안 함**: Mondrian conformal prediction(Vovk, Gammerman & Shafer 2005) `[웹확인-초록]`이 정확히 이 형태의 원칙적 절차다 — **partition(우리 경우 phase bin 정의)을 미리 고정**하고, 그 안에서 각 그룹별 conformal quantile을 따로 잡으면 **각 그룹의 marginal coverage가 개별적으로 보장**된다. 핵심은 "그룹 정의 함수가 데이터(특히 test 성능)를 보고 만들어지지 않았다"는 것 — 이게 지켜지면 그룹별 threshold를 여러 개 쓰는 것 자체는 사후선택이 아니다. Romano et al. 2020("With Malice Toward None") `[웹확인-초록]`은 이를 "적응적으로 식별된" 서브그룹까지 확장하지만, 거기서도 **그룹을 식별하는 함수 자체는 test 라벨과 독립적으로 고정**되어야 한다는 요건은 남는다.
3. **이론적 상한 — 완벽한 조건부 보장은 애초에 불가능**: Barber, Candès, Ramdas, Tibshirani 2021(arXiv:1903.04684) `[웹확인-초록]`의 impossibility 결과 — 분포무관(distribution-free) 방법으로는 **정확한 conditional coverage(무한히 세밀한 조건, 예컨대 "이 정확한 phase-time에서") 자체가 원천적으로 불가능**하다. 그래서 실전은 항상 "성긴 partition"(Mondrian) 타협을 쓴다. 우리에게 주는 함의: phase bin을 몇 개(예: reach/grasp/transport/place 4개)로 **거칠게** 나누는 것 자체가 이론이 허용하는 현실적 최선이고, "모든 timestep마다 정확한 조건부 보장"을 요구하는 건 애초에 잘못된 기준.
4. **표본 부족 문제 — 그룹이 많아지면 그룹당 calibration 표본이 준다**: Ding et al. 2023(arXiv:2306.09335, Class-Conditional CP with Many Classes) `[웹확인-본문]` — 원문 취지 인용:
   > "existing conformal prediction methods do not work well when there is a limited amount of labeled data per class, as is often the case... when the number of classes is large"

   해법은 **유사한 conformal score를 갖는 그룹을 클러스터링해 클러스터 단위로 CP를 수행** — 그룹(클래스) 수 자체를 실질적으로 줄여 그룹당 표본을 늘린다. **phase bin이 세분화될수록(예: 4 phase → 8 sub-phase) 이 문제가 그대로 재발**한다 — bin을 늘리기 전에 bin당 rollout 수가 충분한지(project 자체 기준으로는 `docs/steering/PITFALLS.md`의 MDE 로직과 동일한 계열 — 조건당 표본이 적으면 검출 자체가 안 되는 게 아니라 "검출 불가능"이 되는 것) 먼저 점검해야 한다.

**우리 상황에 대한 구체적 프로토콜 번역(체크리스트는 최종 응답에 정리, 여기는 근거만 정리)**:
- phase bin 경계는 **test 성능을 보기 전에** (kinematic 이벤트 라벨러 등 outcome-무관 기준으로) 고정 — Gelman&Loken/Mondrian 요건.
- SAFE(단일)와 phase 검출기(그룹별)를 **정확히 같은 rollout pool**에서, **정확히 같은 train/calibration/test 3-way split**으로 비교 — split은 한 번만 만들고 두 검출기 모두 그 위에서 평가(Cawley&Talbot의 "선택에 쓴 데이터로 보고하지 말 것"을 지키는 최소 요건).
- phase 검출기의 "전체 시스템 성능"(여러 그룹별 threshold를 실제로 조합해 하나의 evaluation set에 적용한 결과)을 SAFE의 단일 숫자와 **1:1로** 비교 — "제일 잘된 bin만" 보고하는 것은 다중비교 오염(Holm 보정 없이는 안 됨, project 기존 관행과 정합).
- 그룹별 표본수 n_g를 항상 병기 — 표본 부족 그룹은 인접 그룹과 pool(Ding et al. clustering 정신)하거나 "검출 불가" 판정으로 명시.

## 5. Conformal prediction 기반 검출기의 보정·평가 관행

- **공통 골격(SAFE/FIPER/Sentinel-STAC/Hide-and-Seek 4개 모두 동일)**: `[로컬원문]`+`[프로젝트기존노트]`+`[웹확인-본문]` **성공 rollout만으로** 시간가변 band(μ_t + h_t, functional CP, Diquigiovanni et al. 원 방법)를 만들고, exchangeability 가정 하에 "성공 rollout이 이 band를 벗어날 확률 ≤ α"를 보장한다. 실패 데이터는 검출기 학습(있다면)에만 쓰고 **threshold 보정에는 안 씀** — 이건 실패 데이터가 항상 희소하다는 로봇 현실과, "calibration에 쓴 데이터로 평가하지 말라"(§4-1 Cawley&Talbot)는 원칙을 동시에 만족시키는 설계.
- **시퀀스 구조를 유효한 iid 통계량으로 접는 두 가지 트릭**:
  - KnowNo(arXiv:2307.01928) `[프로젝트기존노트]` Claim 1: 시퀀스 전체를 하나의 calibration 데이터포인트로 승격(min-confidence lift)한 뒤, **causal(온라인, 과거정보만) 재구성이 non-causal 시퀀스 CP와 논리적으로 동치**임을 증명 — 온라인 배포에도 시퀀스-레벨 보장이 유지됨.
  - STAC/Sentinel(arXiv:2410.04640) `[프로젝트기존노트]` — η_t가 단조증가하므로 "trajectory 중 한 번이라도 경보"라는 사건이 "종단 시점 η_H 값"이라는 단일 iid 통계량으로 환원 → 표준 split conformal 그대로 적용, FPR≤δ 증명(Prop.1/2).
  - 이 두 트릭은 **phase-bin마다 독립적으로 causal하게 threshold를 적용해도 되는가**라는 우리 질문에 직접 답을 준다 — "예, 단 그룹(phase-bin) 정의와 그 그룹 안에서의 계산이 온라인/causal하게 재구성 가능함을 먼저 보여야 한다"는 조건부.
- **분포 불일치 경고**: SAFECAST(arXiv:2608.04246) `[웹확인-초록]`의 문제의식 자체가 "calibration 데이터가 배포조건(deployment shift)과 안 맞으면 SAFE형 CP 보장이 깨진다"는 것 — phase bin별 calibration이 그 bin의 실제 test 분포와 다르면(예: 특정 phase의 calibration rollout이 특정 scene에 쏠림) 이론적 α 보장이 실측 FPR과 괴리된다(SAFE 자신도 Appendix C.2에서 이 괴리를 실측 보고).
- **실전 체크리스트로 확인된 관행**: (a) calibration은 항상 성공만, (b) threshold는 test 전에 고정(사후 조정 금지), (c) exchangeability 가정을 논문에 명시, (d) 이론적 커버리지(1−α)와 실측 커버리지의 괴리를 반드시 별도로 보고(SAFE Appendix C.2가 이 관행의 모범 사례 — "우리도 이론과 실측이 어긋날 수 있다"를 감추지 않고 수치로 보여줌).

---

## 메모리 연결

- [[seen18-rollout-length-confound]] — 우리 프로젝트의 길이confound 원발견.
- [[truncation-length-standard]] — SAFE의 min-length T truncation을 우리 표준으로 채택한 결정.
- [[eval-power-mde]](`docs/steering/PITFALLS.md` §7) — Holm/McNemar/MDE, 여기서 다룬 "다중비교·표본부족" 문제의 project 내 기존 적용 사례(ΔSR 맥락, 이번 문서는 detector-AUROC 맥락으로 확장).
- [[notall-online-failuretype-niche]] — phase/failure-type 온라인 식별이라는 우리 메인 미해결 문제와의 연결.
