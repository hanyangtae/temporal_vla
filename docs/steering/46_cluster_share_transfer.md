# 46. Cluster 공유도 → detector 학습 단위 판단 검증 (음성)

2026-08-19, branch `exp/cluster-transfer-criterion`,
스크립트 `scripts/analysis/grid_phase/cluster_share_transfer.py`.
협업: 전이 정답지(detector directed pair)는 exp/safe-length-ablation 세션이
`failure_detector_sim.py --arm loto --shards` 로 산출 (같은 3 seed, phase-gt 절제).

## 질문

**activation 만으로 계산한 slug 쌍 공유도가, 실제 failure-detector 의 zero-shot 전이
성능을 예측하는가?** 예측이 성립하면 새 task 에서 detector 를 학습해 보기 전에
"묶어 학습 vs 분리 학습"을 정할 수 있다 (docs/43 §7 의 미실행 갭 = family-pooled vs
per-variant 의 사전 판별 기준).

- **S1 (최종 판정, 사용자 결정 08-19)**: A 의 GT-phase 조건부 fail−succ diff-of-means
  방향을 B 에 투영한 succ/fail AUROC — LOTO 전이 실패의 병목이 방향 부호 반전이었다는
  실측(43 §7)을 직접 측정.
- **S2 (참고)**: A 의 PCA-64w+KMeans(k8) 를 B 에 이식했을 때 margin(vs clock) 유지율.
- 통제: phase-gt dwell cap(ceil(μ+σ), fit 규약 동일) + scene-중심화 전/후 병기 +
  relpos<0.5 조기 한정 + scene 블록 내 순열 z + length_auroc 병기 + scene split 은
  detector 러너의 `split_scenes` 동일 구현(seed, crc32).
- 데이터: grid v1 930판 segA (layer 12 · denoise 3 · 49토큰 평균 · 1536d), 승준.

## 결과 1 — 공유도 지표 자체는 잘 작동한다

배관 게이트: 자기전이 S1 0.86~1.00 (41 라운드 대역 정합). 합성 스모크에서 방향 반전
쌍을 z −4~−5 로 검출, 구조 지표(S2)는 반전 쌍에서도 ≈1.0 = "지도는 같고 나침반만
반대"를 S1 만 잡는다(설계 의도).

실데이터 공유도 패턴 (scene-중심화, seed-median):

| 패턴 | S1 | 기존 실측과 |
|---|---|---|
| PPCC 형제끼리 | 0.82~0.95 (z 4~5.6, 부호일치 1.0) | LOIO family 내 전이 성립과 정합 |
| drawer left↔right | 0.55~0.64 (z 음수) | LOIO right←left 역방향(0.32)과 정합 |
| OpenDrawer_right↔PPCC | 0.12~0.32, 부호 반전 | LOTO drawer-right 역방향과 정합 |
| OvenRack↔거의 전부 | 0.08~0.26 역방향 | OvenRack 제외 판정(43 §3)과 정합 |
| CoffeeSetupMug↔PPCC | 0.74~0.92 정방향 | cross-family 인데 높음 (pick-place 계열) |

즉 **공유도는 기존의 정성적 판정들(형제 공유·drawer 좌우 비대칭·OvenRack 고립)을
전부 재현**한다. 여기까지는 가설 지지.

## 결과 2 — 그러나 detector 전이를 예측하지 못한다 (핵심, 음성)

directed pair 정답지(12 run → 유효 (src,dst,seed) 53셀, lstm·phase-gt)와 per-seed join:

| 상관 (Spearman) | td10 | td20 |
|---|---|---|
| 전체 n=53 | **+0.07** | **−0.32** |
| 1→1 만 n=41 | +0.24 | −0.27 |
| pair-median n=18 | +0.10 | **−0.46** |

조기(td10) 무상관, 늦은(td20) 쪽은 오히려 약한 **음**의 경향. 방향 부호 일치(sign=1.0
vs <1) 이분도 td10 median 0.53 vs 0.44 로 미미, td20 은 역전.

반례가 구조적이다:

| 쌍 | S1 (공유도) | td10/td20 (실제 전이) | 판정 |
|---|---|---|---|
| bread→candle | 0.87 (높음) | 0.48/0.46 | 공유도 높은데 전이 안 됨 |
| marshmallow→jug | 0.24~0.88 (불안정·낮음) | 0.60~0.73/0.74~0.85 | 공유도 낮은데 전이 잘 됨 |
| drawer2→marshmallow | 0.41~0.52 (낮음) | **0.71~0.78/0.74~0.87 (표 최고)** | 〃 |
| candle→drawer_left | 0.89~0.91 | 0.68~0.73/0.56~0.67 | 이건 일치 |

## 해석

1. **전이는 source-측 방향 공유가 아니라 target-측 성질이 지배한다.** →marshmallow 는
   어디서 와도 잘 되고(0.53~0.78), →candle/→bread 는 형제에게서도 안 된다. 상대
   세션의 독립 관찰("target 의존이 source 의존을 지배")과 일치. S1 은 본질적으로
   source→target 방향 정합을 재는 지표라 이 구조를 놓친다.
2. **1→1 전이 자체가 약하다** — LOIO 4→1(marsh td10 0.79)에 비해 1→1 은 크게 낮다
   (bread→candle 0.48). 형제 전이의 실체는 개별 방향 공유가 아니라 **풀링**(다양한
   source 혼합)이 만드는 것으로 보인다. "공유도로 사전 판단"의 대상 자체(1→1 전이)가
   부실한 신호였던 셈.
3. **LSTM detector 는 diff-of-means 방향과 다른 것을 학습한다.** drawer2→marshmallow 가
   그 증거 — 방향은 공유되지 않는데(S1 0.47) 전이는 최고. 시퀀스 모델이 잡는 신호
   (정체·반복 패턴 등 방향-무관 성분)가 별도로 존재할 가능성.

## 판정 (계획의 판정 3항)

**cluster/방향 공유도는 detector 그룹핑의 사전 기준으로 부적합 (탐색 라운드 기준).**
detector 그룹핑은 실측(LOIO/directed pair)으로만 정하는 것이 현재 근거 있는 절차다.
공유도 지표가 기존 정성 판정들을 재현하는 데도 불구하고 전이 예측에 실패한 것 자체가
정보: "표현이 공유된다"와 "detector 가 전이된다"는 다른 성질이다.

## 한계 (confound-audit)

- **정답지 자체가 noisy**: td10 의 seed 간 변동이 큼 (candle→marsh 0.79/0.42/0.66).
  상관의 상한이 낮게 눌린 상태 — "예측 불가" 판정은 이 정답지 해상도 안에서의 판정.
- 탐색 라운드: 다중비교 보정 없음, pair-median n=18 (rho −0.46 도 p≈0.05 경계 —
  과해석 금지).
- 그룹 source 의 S1 은 멤버 평균 근사 (그룹 fit 아님).
- length_auroc 전 셀 1.0 (실패=timeout 데이터) — S1 은 cap+equal-budget 로 record 수
  비의존 설계이나 각주 필수. 1→1 약함에는 데이터량(60판) 효과 혼입 가능성.
- apple→bread(실패 0)·일부 jug seed(성공 0) 결손.

## 산출물·재현

- 공유도: `outputs/analysis/grid_phase/share_transfer_full/share_transfer.tsv`
  (601행, 로컬 cluster-transfer worktree + 승준 `~/workspace/temporal_vla_clustershare`)
- 전이 정답지: safe-length-ablation worktree
  `outputs/analysis/grid_phase/pairs_s{0,1,2}/{phase-gt,none}/<run>/sim_summary.tsv`
  (fold 매핑: run `<A>_<B>` 의 arm=loto·task=<held-out> 행 = 반대쪽→held-out)
- join·상관 스크립트는 세션 tmp (일회성) — 수치는 본 문서에 고정. 재현은 위 두 TSV 를
  (src,dst,seed) 로 join 해 Spearman.

## 후속 후보 (미실행)

- target-측 "전이 수용성" 지표 (→marshmallow 가 왜 만만한가): target 의 실패 신호가
  저차원·시불변인 정도를 재는 지표가 그룹핑 기준의 다음 후보.
- 방향-무관 공유 지표: LSTM 이 잡는 정체·반복 패턴 계열 (drawer2→marsh 설명 후보).
- 풀링 효과의 체계 측정: source 수 1→2→4 를 같은 target 에 사다리로.

---

## 라운드 2 — co-training 정답지 (사용자 방향 정정, 2026-08-19)

사용자 정정: 원래 목적은 zero-shot 전이가 아니라 **"한 detector 에 어떤 slug 데이터를
같이 넣을지"** 의 선별 기준. 정답지를 co-training Δ 로 교체해 재검증한다:
Δ(A|+B) = mixed(A+B 합쳐 학습, A 의 held-out scene 평가) − pertask(A 단독).
방향 부호가 반전된 데이터를 섞으면 그라디언트 충돌로 **간섭**, 방향이 공유되면 표본
증강으로 **시너지**가 기전상 예상된다 — zero-shot 과 달리 여기서는 S1 이 직접 관여.

### 사전 등록 예측 (결과 수신 전 고정, S1 scene-centered seed-median 기준)

| 쌍 | S1 A→B / B→A | min | 예측 |
|---|---|---|---|
| bread+candle | 0.87 / 0.83 | 0.83 | 시너지 |
| candle+marshmallow | 0.95 / 0.84 | 0.84 | 시너지 |
| bread+marshmallow | 0.95 / 0.82 | 0.82 | 시너지 |
| candle+drawer_left | 0.89 / 0.85 | 0.85 | 시너지 (cross-family) |
| coffee+bread | 0.80 / 0.75 | 0.75 | 시너지 (cross-family) |
| jug+marshmallow | 0.93 / 0.46 | 0.46 | 중간 (비대칭) |
| drawer_left+drawer_right | 0.64 / 0.55 | 0.55 | 중간 |
| drawer_left+ovenrack | 0.47 / 0.36 | 0.36 | 간섭 |
| candle+ovenrack | 0.37 / 0.21 | 0.21 | 간섭 |
| bread+drawer_right | 0.32 / 0.12 | 0.12 | 간섭 |
| marshmallow+drawer_right | 0.29 / 0.12 | 0.12 | 간섭 |

판정 규칙(사전 고정): 쌍별 Δ(양쪽 slug, td10·td20, 3-seed) 부호가 예측 3분류와 단조
정렬하는가 (Spearman + 시너지군 vs 간섭군 Δ 차이). ⚠ pertask baseline td10 의 seed
변동이 큼(candle 0.82/0.56/0.09) — Δ 는 3-seed 평균 + 보조 지표(maxscore_auroc,
tpr/fpr) 병용, 소표본 셀은 n_td 병기.

실행: mixed arm 11쌍 × 3 seed (safe-length-ablation 세션, detector_trunc 와 동일 설정).

### 라운드 2 결과 — 사전등록 채점 (2026-08-19)

Δ(mixed − pertask, lstm, 3-seed 평균, 원자료 직접 재계산):

| 채점 | Δtd10 | Δtd20 |
|---|---|---|
| pair-mean vs S1min Spearman (n=11) | **+0.26** | +0.16 |
| 시너지군 Δ평균 vs 간섭군 (사전 3분류) | **+0.051 vs −0.011** | +0.001 vs −0.025 |
| slug 단위 S1(상대→나) vs Δ (n=21) | +0.20 | — |

- **방향은 사전 예측과 정합** (시너지군 > 간섭군, rho 양수) — zero-shot 라운드(rho ≈ 0)
  보다 낫다. 간섭 예측의 대표 적중: candle+oven 에서 **oven Δtd10 −0.235** (최저 S1
  쌍에서 최대 피해), drawer_right 쌍들도 소폭 음.
- **그러나 효과 크기가 정답지 noise 대역(±0.1) 이내** — 단독 그룹핑 기준으로는 불충분.
- **실질 발견 = slug 비대칭**: 같은 쌍에서 한쪽만 얻고 한쪽은 잃는다.
  candle+marsh: marshmallow **+0.28** / candle **−0.19**. bread+candle 도 bread +0.06 /
  candle −0.13. marshmallow 는 LOIO·directed pair·co-train 3연속 최대 수혜 —
  "수혜자 성질(잡히기 쉬움)"이 지배하고, 공여 slug 는 신호 희석 비용을 치른다
  (상대 세션 독립 해석과 일치). S1(상대→나)로 수혜자를 집으려는 시도는 rho 0.20 으로
  약함 — candle 은 marsh→candle 0.84 인데도 −0.19 손해 (반례).
- 각주: coffee 는 mixed 행 결측(test 성공 0), jug n(f/s)=19/1, marshmallow·oven 일부
  seed 결측(n_sd=2), drawer_right 는 pertask 부터 FPR 불량 이력.

### 최종 판정 (질문: "activation 을 보고 detector 학습 단위를 정할 수 있는가")

1. **약한 방향 신호는 있으나 단독 기준으로는 불충분.** S1 공유도는 co-training 의
   시너지/간섭 방향을 맞추지만(특히 부호 반전 slug 혼입의 해악 — oven), 효과 크기와
   해상도가 실측 Δ 를 대체할 수준이 아니다.
2. **그룹핑의 실제 지배 변수는 공유도가 아니라 (a) 수혜자 성질, (b) 풀링 데이터량,
   (c) slug 비대칭 회계**다. "어떤 단위로 묶을까"보다 "누가 얻고 누가 잃는가"가 옳은
   질문이며, 이는 mixed vs pertask 실측(run 당 2~4분)으로 직접 재는 것이 현재로선
   가장 싸고 정확하다.
3. activation 지표의 유효한 용도(축소된 권고): **혼입 금지 필터** — S1 부호 반전
   (sign_agree 0, S1<0.35) slug 를 같은 detector 에 넣지 말 것 (oven·drawer_right 류
   스크리닝). 묶음 확정은 실측으로.
