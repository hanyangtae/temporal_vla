# 32. G2 결과 — scene 잔차화 후 succ/fail read (drawer_right scene-matched, L12)

작성 2026-07-27, exp5 세션. 데이터 = exp5-3 수집 drawer_right scene-matched 160판
(scenario_seed 20 × inference_seed 8, 승준). 코드 = `scripts/scene_sae/g2_residual_read.py`
(exp5-3 within-scene LOSO 프로토콜 문자 그대로 이식, codex 리뷰 5건 반영).
판정 프레임 = 대시보드 제안(창평균 0.847을 임계값으로 쓰지 않고, (a) 재현 앵커 →
(b) 잔차화 전 → (c) 잔차화 후 곡선 변화로 판정).

**⚠ 31 문서와의 관계**: 셀이 다르다(31=drawer_left fit30, 32=drawer_right scene-matched).
직접 비교가 아니라 재현·확장 관계.

## 0-pre. 판정 기준 개정 (사용자 승인, exp5-1_g2_verdict_decision.txt)

사전 등록 기준 중 "unseen scene 의 식별 정보까지 지워질 것"은 원리적으로 달성 불가능한
요구였음이 밝혀져(§0-3) **조건 오설계로 개정**한다. 개정 기준(승인 문구 그대로):

> G2 통과 = (a) **부분공간 추정에 포함된 scene 기준**으로 scene 식별력이 유의하게 떨어지고,
> (b) 그 제거 후에도 within-scene succ/fail read 가 exp5-3 기준선(L12 AUROC ≥0.85, 창 38)
> 이상으로 살아남는다.

★통과의 의미는 좁다: **"fit 에 포함된 scene 에 한해, scene 성분을 계산으로 제거해도 실패축이
살아남는다."** 이후 문서에서 "scene 은 제거 가능하다" 또는 "steering 가능"으로 확대 인용 금지.
**이번 통과는 drawer_right 단독** (mixer 는 §4.6 예비 — 판정 불포함).

## 0. 세 줄 결론

1. **배관 앵커 정합**: 우리 재계산 raw|all 창평균 = **0.847**, t=0 = **0.729** — exp5-3
   수치와 일치. 이후의 모든 차이는 배관이 아니라 처리의 차이다.
2. **outcome 신호는 scene 성분이 아니다** (개정 기준으로 G2 통과, drawer 단독): train-scene
   라벨 부분공간(rank 12~19)을 제거하면 본 scene 식별은 궤멸하는데 — all-agg 0.812→**0.162**,
   future-agg 0.767→0.339 (둘 다 `g2_L12_scene_r19.json` 의 split-train 참고 probe, agg 별 값) —
   succ/fail read 는 **유지+상승**. **판정 범위별 병기(§3-1 요구)**:
   | | 전체 scene | unseen scene (n=6) |
   |---|---|---|
   | between r12 | 0.889 (z 5.30) | 0.801 (z 2.45) |
   | between r16 | **0.900 (z 4.91)** | 0.808 (z 2.60) |
   | between r19 | 0.898 (z 4.56) | 0.793 (z 2.80) |
   개정 기준 (a) 가 seen-scene 기준이므로 판정은 전체 scene 값으로 하되, **unseen 범위는
   z<3·0.85 미달로 약하다는 사실을 숨기지 않는다** (표본 6 ep 의 검정력 한계 병존).
3. **★발견 A — 모델은 scene 을 "외운다"**: 어떤 제거(선형 rank19·SAE selective 35)도
   **unseen scene 의 식별 정보를 못 지운다** (fold별 probe: 제거 전 0.989 → r16 0.911 →
   r19 0.994). 20개 scene 평균이 만드는 방향을 전부 지워도 새 scene 은 91~99% 식별 —
   **새 scene 의 방향은 기존 scene 들이 span 하는 공간 밖**이다. scene 성분은 조합적
   속성(레이아웃 요소·재질)이 아니라 **scene 별 암기 방향**이고, 이 사실이 이후 모든 설계를
   제약한다(→ §5 온라인 잔차화 포기).

## 1. 설정

- 입력: `build_sae_inputs.py --scan-dir --split-by scene` 산출 (행 897k, per-token, K평균,
  fingerprint 대조). SAE = R_L12_m6144_k64_aux0_split_scene (scene축 학습, dead 0.166).
- read 프로토콜(전 arm 공통, analyze_sm2 이식): 창 [0,38) record, 혼재 scene만,
  within-scene 방향 + held-out scene 중심화 LOSO, episode 단위 AUROC, scene 내 episode 순열
  null (200/100회).
- record 집계: future 세그먼트 평균(1차, G1 근거) / 49토큰 전체 평균(=exp5-3 동치, 앵커) /
  action 세그먼트 (민감도).
- arm: raw / sae_full(재구성만) / sae_topN(selective 제거 후 재구성; selective 35개 =
  layout-라벨 quick probe 산정) / linear_between·linear_logreg (scenario_seed 라벨 부분공간
  r∈{1,2,4,8,12,16,19}, LOSO fold 내 재추정 — rank 상한 19 = 20클래스 이론 한계).

## 2. 결과 (창평균 AUROC | null_z | scene probe: 본scene참고 / fold별 unseen)

| arm (all-agg) | AUROC | null_z | scn(본) | scn(unseen) |
|---|---|---|---|---|
| raw | 0.847 | 4.2~4.4 | 0.812 | 0.992 |
| sae_full | 0.845 | 4.4 | 0.646 | 0.965 |
| sae_top10~all | 0.844~0.845 | 4.2~4.7 | 0.666~0.757 | 0.957~0.963 |
| linear_logreg r1~19 | 0.846~0.853 | 4.1~5.1 | 0.683~0.728 | 0.990~0.994 |
| **linear_between r12** | **0.884** | **4.97** | **0.162** | 0.968 |
| **linear_between r16** | **0.885** | **5.04** | **0.162** | 0.939 |
| **linear_between r19** | **0.879** | **5.05** | **0.162** | 0.995 |

future-agg 도 동일 패턴 (raw 0.800 → between r16 0.900/null_z 4.9; 본scene 0.767→0.339).
t=0 값은 전 arm 0.66~0.73 (raw|all 0.729 = exp5-3 일치). 공식 verdict 는 전 arm
`undecidable_removal_failed` — fold별(unseen) 기준을 만족하는 arm 이 없기 때문. §3-3 참조.

## 3. 판독

1. **(b)→(c) 판정: outcome 잔존 + 개선.** between r12~19 는 본 scene 식별을 20클래스
   우연(0.05) 근처(0.16)까지 지우는데 read 는 0.88~0.90 으로 **올라간다**. scene 구조가
   succ/fail 방향 추정의 잡음이었다는 뜻. 잔여 방향 = G3(write) 후보.
2. **SAE arm 은 동기였던 작업에서 선형에 진다**: selective 35 제거는 본 scene 식별도
   못 지움(0.65→0.67~0.76). 단서: selective 산정이 layout 라벨(5클래스)·quick probe 기반 —
   scenario_seed 기반 selectivity 로 재산정할 여지는 있으나, ③ 의 전이 문제는 그대로 남는다.
3. **unseen-scene 제거는 구조적으로 실패**: fold Q 는 train scene 라벨로 추정되므로 새 scene 의
   고유 방향을 원리적으로 못 담는다. 실측 0.94~0.99 유지가 그 증거. 해석: **scene 암기가
   scene 간 공유 부분공간이 아니라 scene 별 고유 방향에 저장**된다. 이는 (i) "scene feature
   제거 후 일반화" 접근(선형이든 SAE 든)의 근본 제약이자, (ii) within-scene 중심화(제거가
   아니라 scene 별 offset 상쇄)가 올바른 통제였다는 사후 정당화다.
4. **verdict 라벨 재해석**: `undecidable_removal_failed` 는 "unseen scene 식별까지 지워야
   성공"이라는 사전 기준의 산물. 위 ③ 이 구조적 불가능임이 밝혀졌으므로, G2 의 과학적 질문
   ("scene 과 무관한 outcome 신호가 있는가")에는 **본scene 제거 + read 잔존 + t=0 증거**로
   답한다: **있다**. 단 이 재해석은 사후이므로 명시해 둔다.
5. t=0 (같은 scene 이면 관측 동일, denoise noise 만 다름) = 0.729, within-scene 계산임을
   코드로 확인(승준 analyze_sm2.py loso/perm_null — 방향·중심화·순열 전부 scene 내부).
   noise 추첨이 최종 성패를 예측한다는 뜻이며, 이 신호가 "실패 원인 방향"인지 "운 좋은
   노이즈 방향"인지는 G3 개입만이 가른다.

## 4. G1-재판정 (drawer_right, scenario_seed 20클래스) — 진행 중

episode축 SAE(L12/L10/L8) × {scenario_seed, layout_id} probe 순열 50회가 실행 중
(순열당 ~90s). 완료 시 이 절 갱신. 예비 신호: scene축 SAE 의 layout probe 가
CV(본 scene) 0.99 vs held-out scene 0.63 — scene 일반화 격차 큼(§3-③ 과 정합).

## 4.5 exp5-3 실패축 판정과의 교차 검증 (2026-07-27 추가, exp5-3_fail_axis_drawer_mixer.txt)

exp5-3 이 같은 데이터에서 독립적으로 확립한 사실과의 정합:
- **G2 정답지(그쪽 §5-1)**: "잔차화 후 실패축 생존" 기준 = drawer L12 AUROC ≥0.85·같은 창 38.
  우리 linear_between r12~19 잔차화 후 read 0.88~0.90 — **기준 충족**. LOSO 부호 12/13 대응
  지표는 후속 산출 예정.
- **실패축은 전 layer 실재**(drawer 0.83~0.88, LOSO+위약 p<0.0001), **peak 는 task 의존**:
  drawer L12 / **mixer L2**. → mixer 재현은 L12 만으론 부족, L0·L2 추가 빌드·실행(진행 중).
  layer 는 cell 별 재선정이 원칙 (31 문서의 "단일 layer 확정 금지"와 정합).
- **전역 방향 ≈ 난이도 축**(cos 0.92~0.93, 그쪽 §3): exp4-1 cross-scene fit 이 왜 실패했는지의
  표현 수준 설명. within-scene 이득의 실질은 전패 scene 배제.
- **읽힘 ≠ 밀림(그쪽 §5-5)**: exp5-3 의 setM write 는 대해악(SR 0.344→0.025). 본 문서의
  어떤 결론도 "steering 이 된다"로 확대 금지 — G3 는 별도 인과 시험.
- SAE 필요성 논거(그쪽 §5-3: scene 성분이 손잡이 기하로 설명 안 됨·난이도/실패 얽힘)와
  우리 §3-② (현 selective 집합은 열세)는 모순이 아니라 "고차원 분해가 필요하나 지금
  선택법이 부족"으로 종합된다.

## 4.6 정답지 통계량 재판정 — **공식 판정은 drawer 단독, mixer 는 예비** (개정)

**통계량 주의**: exp5-3 실패축 표(0.878/0.788)는 loso_layer_sweep 의 **scene 별 AUROC 평균**,
우리 §2 는 analyze_sm2 의 **풀링+scene 중심화 AUROC** — 서로 다른 자다. mixer 앵커 불일치
(0.70 vs 0.79) 소동의 원인이었고, 우리 데이터로 그쪽 통계량을 재현하면 **0.788·13/15 정확
일치**(drawer 0.878·12/13 도 일치) — 배관 이상 없음 확인.

정답지 통계량(scene 별 AUROC 평균) + between 부분공간 LOSO-fold 제거 + scene 내 라벨순열
위약 400회:

| cell·layer | raw | 제거 후(r8/r16) | 부호 | 위약 | p |
|---|---|---|---|---|---|
| **drawer L12** | 0.878 (12/13) | 0.914 / **0.921** | 12/13 → **13/13** | 0.51±0.10 | **<0.0025** |
| **mixer L12** | 0.737 (13/15) | **0.741** (r8) | 14/15 | 0.51±0.10 | **<0.0025** |
| mixer L2 | 0.788 (13/15) | 0.702 (r8) | 13/15 | 0.50±0.10 | 0.010 |

**공식 판정: G2 통과 = drawer_right 단독** (개정 기준 §0-pre, 상회: 0.921·13/13).
**mixer 는 이번 판정에 불포함** (사용자 결정문 §3-3): 풀링 통계량 기준선이 z 2.6~3.2 로
전제조건 미달(`no_baseline_separation`)이었고, 위 표의 mixer 행은 정답지 통계량 **예비**
결과다 — L12 잔존(p<0.0025)·L2 감쇠(0.79→0.70, p=0.01)는 고무적이나 공식 판정은
exp5-3 와 기준선 합의 후 별도 라운드로. mixer L2 감쇠는 얕은 층에서 outcome 상관 분산
일부가 scene between 부분공간과 겹친다는 신호로 기록만 해 둔다.

## 4.7 발견 B — SAE 는 단순 선형보다 못했다 (본문 승격, 결정문 §2-B)

같은 데이터·같은 layer(L12)·같은 read 프로토콜(token_agg=future):

| arm | read AUROC(전체) | scene probe(제거 검증, split-train ref) |
|---|---|---|
| raw (제거 전) | 0.800 | 0.767 |
| SAE top10/20/40/all | 0.804 | 0.697~0.725 ← 거의 안 지워짐 |
| linear_between r12 | 0.889 | 0.339 |
| **linear_between r16** | **0.900** | 0.339 |
| linear_between r19 | 0.898 | 0.339 |

exp5 라운드의 기획("SAE 로 scene 을 분리한다")에 대한 **직접 반증 증거**다. 단 read/removal
비대칭 구분: **G1 은 유효** — SAE feature 는 scene 을 *읽을* 수 있었다(held-out z 5.8~6.5).
실패한 것은 그 feature 로 *지우는* 것이다.

**SAE 존폐 결론 (결정문 §5-3, 미루지 않음): (b) 선형 부분공간 제거로 전환한다.**
- 근거 1 = 위 표 (removal 성능 열세). 근거 2 = §4.8-[4-2] 실패 신호가 **1차원**이라
  (c)안(feature 집합 연산자)의 재개 조건도 기각됨. SAE 산출물(코어·G1 결과)은 "N1.5 DiT 에서
  scene 은 sparse feature 로 읽힌다(단 scene-특이적)"라는 관찰 기록으로만 남긴다.
- 유일한 잔여 반론(scenario_seed 기반 selectivity 재산정)은 발견 A(scene 방향이 scene-특이적
  이라 selective feature 도 전이 불가)로 사전 기대가 낮다 — 시도하려면 1회로 제한.

## 4.8 추가 계산 4건 (결정문 §4, 전부 기존 데이터)

- **[4-1] 방향 일관성 — 개선 없음 (최상 우선순위 항목)**: scene 별 실패방향 cos 평균
  제거 전 0.398 → r8/16/19 제거 후 0.397/0.406/0.394. **잔차화는 축을 살렸지만 steering
  입력으로서의 방향을 개선하지 못했다 → G3 기대치 대폭 하향.**
- **[4-2] 실패 신호는 1차원**: r̂ 성분 제거 후 잔차 read = 0.480(우연). 다차원 연산자
  (conceptor·SAE feature 집합) 불요 — **단일 고정 벡터**로 충분.
- [4-3] LOSO 부호: §4.6 표 (drawer 제거 후 13/13).
- [4-4] read 상승(0.800→0.900)의 r-곡선: r1 0.836 → r16 0.900 단조, **r19 에서 꺾임**(0.898)
  — 과제거 시작점 존재. "scene 분산이 nuisance 였다"는 해석과 부합하나 단정하지 않는다.

## 5. 다음 단계 (G3 설계 — 결정문 §5 반영)

- **온라인 잔차화는 포기한다** (발견 A: 새 scene 성분은 추론 중 제거 불가). 채택 설계 =
  **잔차화는 offline 에서 방향 r̂ 를 뽑는 데만 쓰고, 추론 시에는 고정 벡터 하나만 적용**.
  새 scene 전이가 필요 없다. ("scene 제거가 안 되니 G3 불가"라는 오독 방지용 명시.)
- 연산자 미확정: exp5-3 β sweep(exp5-1_to_exp5-3_beta_sweep_request.txt) 결과 + [4-2]
  1차원 판정을 합쳐 사전 등록 규칙으로 선택. [4-1] cos 무개선이라 **기대치는 낮게**.
- mixer 공식 판정: L0/L2 기준선 합의 후 별도 라운드. beer 는 exp5-3 §6 설계 선행 필요.

## 재현

- 산출물: `outputs/eval/robocasa/groot_n15/scene_sae/scene_matched_drawer_right/`
  `g2_L12_scene.json`(r≤8·SAE arm, n_perm 200) / `g2_L12_scene_r19.json`(r 12~19, n_perm 100)
- 명령: 위 디렉토리 파일들의 `source` 필드에 전체 argv 기록. seed 0, GPU3.
