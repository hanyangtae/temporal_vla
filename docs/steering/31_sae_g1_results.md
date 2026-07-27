# 31. SAE G1 결과 — drawer_left scene probe (exp5 1차)

작성 2026-07-27, exp5 세션. 계획·기준 = `docs/steering/30_sae_g1_port_handout.md` §4-D3 (사전 등록).
질문: **top-k SAE feature 가 scene(layout/style)을 실제로 인코딩하는가** — 통과 시에만 G2(잔차화 read).

**한 줄 결론: G1 PASS.** L8·L10·L12 × k64 세 조합이 사전 등록 기준 3개를 전부 충족.
최고 = **L12 k64** (z-probe acc 0.766, 원본 회복률 0.92, 순열 null 대비 z=6.5, selective 0.8%).
scene 인코딩은 **중간~후반층(L8-L12)에 있고 L0 sparse feature 에는 없다** (L0 는 원본 probe 0.771인데
SAE feature probe 는 null 동급 — G1 의 부산 발견물).

## 1. 설정

- 데이터: N1.5 fit30 `pq3_drawer_left` 30 ep (succ 17 / fail 13), record 2,567 → per-token 행 125,783 × D=1536.
  K(denoise) 축은 평균(fit_mean_diff 관례). 토큰 pooling 없음(per-record·per-token 유지).
- layer: L0·L2·L8·L10·L12 (capture_layers=[0,2,4,8,10,12,15] ∩ 후보 대역, atlas peak=L10). layer 별 SAE 독립 학습.
- SAE: top-k (동료 task_classification@88543a2 코어 lift, `src/sae/`), **원본 1536-d 입력**,
  m=6144(=4×D overcomplete), k∈{16,32,64}, seed 0, val early-stop, 학습 ~1분/run(GPU). PCA·토큰평균 미사용.
- split: **episode 단위** layout×success 층화 — train 20 ep(80,654행) / val 4(17,885) / test 6(27,244).
- scene 라벨: **layout_id 5클래스**. ⚠ scenario_seed 는 ep 당 1개(클래스=표본)라 probe 불가,
  style_id 는 layout_id 와 완전 공선 — 실측 (1,1)(2,2)(4,4)(6,9)(7,10) 쌍만 존재. 즉 fit30 의
  scene 해상도 상한은 layout 수준 (핸드아웃 §6-1 함정이 실측으로 확정).
- probe: 선형 logistic(lbfgs), episode 당 record 균등 subsample(길이 confound 보정 — fail 이 행의 73%),
  held-out test 6 ep. 기준선 ① 원본 X probe(상한) ② episode-단위 라벨 순열 null(30~100회).
- G1 판정 기준(사전 등록): (a) null 대비 z>3, (b) 원본 probe 회복률 ≥80%, (c) scene-selective
  feature 비율 <30% — 3개 모두 충족 시 PASS.

## 2. 결과 (15 run = 5 layer × 3 k)

| L | k | z-probe acc | X-probe acc(상한) | null_z | 회복률 | selective % | 판정 |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 0.268 | 0.771 | 1.5 | −0.15 | 0.00 | FAIL |
| 0 | 32 | 0.292 | 0.771 | 1.8 | −0.09 | 0.00 | FAIL |
| 0 | 64 | 0.311 | 0.771 | 2.3 | −0.05 | 0.00 | FAIL |
| 2 | 16 | 0.472 | 0.775 | 5.3 | 0.31 | 0.07 | FAIL |
| 2 | 32 | 0.526 | 0.775 | 5.2 | 0.44 | 0.20 | FAIL |
| 2 | 64 | 0.525 | 0.775 | 5.1 | 0.43 | 0.29 | FAIL |
| 8 | 16 | 0.562 | 0.771 | 5.1 | 0.52 | 0.18 | FAIL |
| 8 | 32 | 0.589 | 0.771 | 5.4 | 0.58 | 0.29 | FAIL |
| **8** | **64** | **0.716** | 0.771 | **5.8** | **0.87** | 0.55 | **PASS** |
| 10 | 16 | 0.593 | 0.785 | 5.8 | 0.58 | 0.21 | FAIL |
| 10 | 32 | 0.606 | 0.785 | 5.5 | 0.60 | 0.26 | FAIL |
| **10** | **64** | **0.714** | 0.785 | **6.2** | **0.84** | 0.65 | **PASS** |
| 12 | 16 | 0.597 | 0.803 | 5.8 | 0.56 | 0.26 | FAIL |
| 12 | 32 | 0.625 | 0.803 | 5.5 | 0.62 | 0.28 | FAIL |
| **12** | **64** | **0.766** | **0.803** | **6.5** | **0.92** | 0.76 | **PASS** |

(chance = 5클래스 test ep 구성상 ~0.18-0.33, 순열 null 실측 mean ≈ 0.18.)

PASS 조합 세부 (k64):

| L | dead frac | live feature | selective feature | 층화 acc (succ만/fail만) |
|---|---|---|---|---|
| 8 | 0.174 | 4,287 | 34 | 0.835 / 0.598 |
| 10 | 0.163 | 4,418 | 40 | 0.860 / 0.568 |
| 12 | 0.122 | 4,610 | 47 | 0.892 / 0.641 |

## 3. 판독

1. **G1 통과 — SAE feature 는 scene 을 실제로 인코딩한다.** null_z 5~6.5 로 우연이 아니고,
   k64 면 원본 선형정보의 84~92%를 회복하며, 그걸 나르는 feature 는 live 의 1% 미만(34~47개)로
   비자명하게 국소화돼 있다. "scene feature 만 골라 제거"(G2 잔차화)의 전제가 성립.
2. **깊이 구조**: L0 은 원본 activation 엔 scene 정보가 있는데(0.771) top-k SAE feature 로는
   null 동급(0.27~0.31) — 초기층 scene 정보는 sparse 코드로 안 잡히는 분산 표현. L2 는 중간(z 유의,
   회복 43%), L8-L12 에서 회복이 완성. **G2 잔차화 대상 layer 는 L8-L12 대역**이 자연 선택.
3. **k 의존성**: 회복률이 k 에 단조 증가(k16 ~0.55 → k64 ~0.85+). scene 정보가 소수 방향에
   압축되지 않고 수십 feature 에 걸쳐 있다는 뜻 — G2 에서 제거할 성분도 "feature 몇 개"가 아니라
   selective 집합(34~47개) 단위로 다뤄야 한다.
4. **주의(층화)**: fail episode 에서 probe 정확도가 일관되게 낮다(0.57~0.64 vs succ 0.84~0.89).
   실패 rollout 의 표현이 scene 구분을 흐린다는 신호로, seen18 "공유 실패 zone"(실패는 task 무관
   수렴) 관측과 정합. G2 판정 시 succ/fail 층화 보고 유지 필요.
5. **한계**: scene 라벨 해상도가 layout 5클래스(=style 공선) — scenario_seed 수준 "scene 암기"
   제거를 검증하려면 라벨 해상도가 부족하다. fit30 로는 이게 상한. test ep 6개라 acc 분산도 큼.

## 4. 다음 단계 제안 (G2 go/no-go 는 사용자 결정)

- **G2 설계**: L8-L12 × k64 SAE 의 scene-selective feature(34~47개) 성분을 잔차화한 뒤
  succ/fail read 잔존 검증 — 길이통제(dwell cap)·episode 순열 null·held-out (핸드아웃 Phase E).
- 확장 카드: ① drawer_right 재현(같은 파이프라인 그대로), ② m=8×D·k 격자 확대(선택),
  ③ scene 라벨 해상도 한계 대응 — 실패 대량수집 시 scene 당 다중 rollout 확보(25a §4 병행 검토).

## 재현 정보

- 코드: `src/sae/`(코어), `scripts/scene_sae/`(build/train/probe/드라이버), 브랜치 feat/scene-sae.
- 입력: 승준 빌드 → `outputs/eval/robocasa/groot_n15/scene_sae/pq3_drawer_left/inputs/` (X_L{0,2,8,10,12} fp16
  + stats + meta + split.json, split seed 424101).
- 산출: 같은 루트의 `L{L}_m6144_k{K}_s0/`(ckpt·metrics) + `probe_L*.json` 15개. 학습 seed 0, GPU 3장(0/1/2) 병렬.
- 순열 횟수: L0/L8/L10 의 k16·k32 는 100회, 나머지는 30회(속도 — null 분산 안정 확인함).
