# 46. Cluster 공유도 → detector 학습 단위 판단 검증 (음성)

> 인용된 40·41은 2026-08 문서 정리(7dc0619)에서 archive 됨 — 요지는 RESULTS.md·
> RESEARCH_DIRECTION.md 에 흡수, 원문 복원 해시는 docs/review/LEDGER.tsv.

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

---

## 라운드 3 — negative filter confirmatory (grid v2, 사전등록)

사용자 지시(08-19): "혼입 금지 필터" 판정을 더 넓은 데이터에서 확인. 데이터 = grid v2
(plan 3134e339de4c 포함 2236판, scene 15 × noise 15, `index_rollouts_v2.tsv` 정본,
실패 14셀 제외). v1(930판)에서 도출한 가설을 v2에서 검증 — **아래 가설은 v2 결과를
보기 전에 고정한다.**

### 사전등록 가설 (v1 도출)

- **H1 (S1 부호 구조 재현)**: v2 공유도 행렬에서 (a) PPCC 형제 정방향 S1≥0.7,
  (b) OpenDrawer_right↔PPCC 부호 반전(S1<0.4·sign≈0), (c) OvenRack↔{PPCC,drawerL}
  반전, (d) drawer left↔right 무공유(0.5 근방), (e) Coffee↔PPCC 정방향.
- **H2 (negative filter 인과, 주 가설)**: v2 co-training에서 부호 반전 쌍의 피해
  예측 slug Δtd10<0, 정렬 대조 쌍은 Δ≥0 부근. 사전 지정:
  - 해악 예측 4쌍 (피해 slug): candle+oven(**oven**), drawerL+oven(**oven**),
    bread+drawerR(**drawerR**), marsh+drawerR(**drawerR**)
  - 무해 대조 4쌍: bread+marsh, candle+marsh, coffee+bread, bread+candle
  - 판정: 반전군 피해-slug Δtd10 평균 < 0 이고 정렬군과의 차이가 순열 p<0.1
    (단측, 탐색→confirmatory 전환이므로 완화 α).
- **H3 (수혜자 재현)**: marshmallow 가 v2 co-training 에서도 최대 수혜군.

### 설계 고정

- v2 segA 추출: `extract_grid_matrix.py --tier segA` (승준,
  `analysis/grid_phase_v2/segA`). 머신 중복 셀은 추출기 규칙(max_cells 머신 선택).
- 공유도: `cluster_share_transfer.py --pairs all --seeds 0,1,2 --k 8` (scene 15 →
  split test2/calib2/train11, 동일 결정적 split).
- detector: 상대 세션 러너로 v2 pertask baseline(10 slug) + 위 8쌍 mixed × 3 seed
  × phase-gt (프로토콜 v1과 동일).

### 라운드 3 결과 (1/2) — H1 채점: 공유도 구조의 v2 재현

v2 공유도 600셀 (scene 15, bread=머신 union 각주). scene-centered seed-median:

| 가설 | v1 | v2 | 판정 |
|---|---|---|---|
| H1a PPCC 형제 정방향 ≥0.7 | 0.860 | **0.850** | ✓ 재현 |
| H1b drawerR↔PPCC 반전 <0.4 | 0.284 | **0.332** (전 8쌍 0.14~0.40) | ✓ 재현 |
| H1c oven 반전 | 0.226 | **0.467 — 재현 실패** | ✗ (아래) |
| H1d drawer 좌우 무공유 | 0.597 | 0.698 | △ 경계 |
| H1e Coffee↔PPCC 정방향 | 0.800 | 0.788 | ✓ 재현 |
| 전 cross 쌍 v1↔v2 Spearman | — | **0.550** (n=66) | 중간 재현 |

**H1c 세부 — oven 의 방향은 scene-set 의존으로 불안정하다.** v2 에서 bread→oven 이
0.22→**0.86 (z+5.9)** 로 정방향 반전, candle→oven 은 0.18(z−5.2) 역방향 유지,
jug/marsh→oven 역방향 유지 — 쌍마다 제각각. 43 §3 의 "oven detector 는 held-out
scene 에 따라 방향이 뒤집힌다"와 같은 성질이 cross-slug 방향에도 나타난 것.
반면 **drawerR 의 반전은 v1·v2 에서 전 쌍 일관** — 안정적 반전.

→ 시사점: negative filter 의 올바른 형태는 "S1 낮음 = 혼입 금지"가 아니라
**"안정적 부호 반전(drawerR형) 또는 scene 간 방향 불안정(oven형) 이면 혼입 금지"**.
두 유형 모두 혼입에 해로울 것으로 예상되나 기전이 다르다 — H2 채점에서 분리 관찰.
(v2 자기전이 게이트: 8/10 slug 0.88~1.0 통과, coffee 0.54 는 succ 21판 소표본 각주.)

### 라운드 3 결과 (2/2) — H2·H3 채점: confirmatory 실패 (최종)

Δtd10 = v2 mixed − v2 pertask, 3-seed, lstm·mlp 병행. 원자료
`v2_{pertask,cotrain}_s{0,1,2}/` (스팟체크 6셀 재계산 일치 확인).

- **H2 기각**: 반전군 피해-slug Δ 평균 lstm −0.021/mlp −0.005 (부호만 충족, 미미),
  **정렬군이 오히려 더 음수** (lstm −0.046/mlp −0.015) — 군간 차이가 예측 반대,
  순열 p(단측) 0.675/0.615 로 완전 미달.
- v1 의 대표 적중(candle+oven 의 oven −0.23)이 **비재현** (+0.034) — H1c 에서 본
  oven 방향의 scene-set 불안정과 정합: v1 해악은 그 scene 조합 특이였다.
  drawerL+oven 만 −0.147 로 방향 유지 (단독으론 채택 불가).
- drawerR 쌍들: 표현 수준 반전은 안정 재현(H1b ✓)인데 co-train Δ 는 +0.008~+0.020 =
  **해악 없음**. 방향 부호 반전이 LSTM co-training 을 해치지 않는다 — R2 의
  "LSTM 은 diff-of-means 방향과 다른 것을 학습한다"와 정합.
- H3 판정 불가: v2 는 base SR 상승으로 marshmallow test 실패 2판 (n_sd=2) —
  검출 불가. coffee 여전히 결측, bread 4쌍 머신혼재 각주.

### ★ 시리즈 최종 판정 (3라운드)

**activation cluster/방향 공유도는 detector 그룹핑에 관해 확증된 예측력이 없다.**
- R1: zero-shot 전이 예측 실패 (rho ≈ 0)
- R2: co-training 방향 신호(탐색) → R3 confirmatory 에서 기각 — 마지막 후보였던
  "혼입 금지 필터"도 미확증. v1 의 해악 사례는 scene-set 특이로 판명.
- 표현이 공유·반전되는 것(S1 이 안정적으로 재는 것, H1a·b·e 재현)과 detector 학습이
  영향 받는 것은 **다른 층위**다. LSTM detector 는 방향 부호에 둔감하다.
- **그룹핑 실무 절차 = 배포 데이터에서 mixed vs pertask 직접 실측** (run 당 수 분)이
  유일하게 근거 있는 방법. v1 실측 기준 잠정 구성(PPCC family-pooled·drawer 좌우
  분리)은 유지하되, v2 급 데이터가 쌓이면 재실측.
- 검정력 각주: v2 confirmatory 자체가 marshmallow·coffee 결측으로 눌려 있어
  "예측 실패"와 "검출 불가"가 섞여 있다. 단 정렬군이 더 음수인 방향 역전은 표본
  문제로 설명되지 않는다.
- S1 도구의 잔존 가치: detector 그룹핑이 아니라 **표현 구조 자체의 지도**
  (방향 공유·안정 반전·scene-불안정 slug 식별 — oven 형 진단). steering 쪽
  연산자 공유·phase gating 설계의 입력으로 유효.

---

## 부록 — phase 판독기 fit 단위 비교 (instruction vs task vs global, v2)

사용자 질문(08-25): SAFE failure detector 는 task(env family) 단위인데, phase 판독기
(PCA-64w→KMeans)도 task 단위로 가도 되는가. v2 segA 2236판, 평가는 항상 instruction 별
margin(vs clock)·purity. `intrinsic_phase.py --scope per-family` 신설.

| 비교 (같은 k) | 결과 (10 instruction, |Δmargin|≤0.02 동) |
|---|---|
| k16: instruction vs task | task 2승 / 0패 / 8동 |
| k24: instruction vs task | task 0승 / 4패 / 6동 |
| k24: task vs global | task 10승 0패 (예: drawerL +0.64 vs +0.14) |

- **판정: instruction ≈ task (동률), global 만 명확히 손해.** k8 에서 보였던 "PPCC 는
  family 로 뭉치면 이득"(candle +0.07→+0.44)은 k 부족 상황에서 표본 증가가 돕던 것 —
  k 를 16~24 로 주면 단위 차이가 사라진다. **지배 변수는 단위가 아니라 k** (v2 에서
  k16~24 ≫ k8, 전 instruction).
- 운영 권고: 품질 동률이므로 SAFE detector 와 단위를 맞춰 **task 단위 + k16~24** 채택
  가능 (판독기 수 10→6, family 내 새 instruction 에 재사용 여지). global(모델 1개)은 불가.
- 한계: GT-phase 정렬 기준 판정. succ/fail 층화 기준(41 §5)의 단위 비교는 별도이며,
  41 의 "global k24 < per-instruction k8"도 k 교란을 포함했을 수 있음 (미재검).
- 부수 실측 (rack out 2종, gating 설계용): intrinsic k8 은 push-in(역행)/pull-out(진행)을
  약하게만 구분 (다수결 0.73/0.79, 기저 0.67/0.56). 그러나 raw 1536d 선형 probe 는
  **0.905/0.862** (ep-holdout) — 정보는 있고 비지도 k8 이 그 축으로 안 자를 뿐.
  gating 은 cluster + 방향 판독기(w 투영 1회) 하이브리드 권고. GT 라벨 분포:
  성공판은 전원 pull-out 도달(111/111·75/75), 실패판은 disengage+reach 맴돌기 지배.
