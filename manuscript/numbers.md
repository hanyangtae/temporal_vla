# 원고 수치 대조표 (verify-before-relay 고정본)

제7회 한국 인공지능 학술대회(KICS 인공지능소사이어티) 투고 원고의 모든 수치와 그 정본 출처.
원고에 수치를 넣거나 고칠 때 반드시 이 표를 경유한다. 정본과 어긋나면 정본이 이긴다.

표기: 40 = `docs/steering/40_action_phase_readout_review.md` (dev),
41 = `git show exp/grid-phase-sep:docs/steering/41_grid_phase_separation.md`,
ref = `outputs/analysis/grid_phase/paper_ref/*.json`, supp = `outputs/analysis/grid_phase/paper_supp/*.json` (보강 실험).

## 설정

| 항목 | 값 | 출처 |
|---|---|---|
| 모델·캡처 지점 | GR00T N1.5, DiT residual layer 12 · denoise 3 · 49토큰 평균 · 1536d | 40 §2 (노션 "layer 15"는 오기 — git 이력에 없음) |
| 파이프라인 (메인) | **raw-1536 → AE(latent 16) → KMeans** (PCA 없음; 동료 실험 B 종합판정 "raw-1536 + AE, PCA 불필요, PCA 안 할 시 whitening 금지"). AE 규격은 task_classification 코드 실측: hidden 256×2 GELU, 대각 가우시안 log-likelihood, AdamW 1e-3/wd 1e-4, grad clip 5.0, early stopping | Notion 「Action phase 정리」 실험 B·종합판정, `task_classification/phase/models/autoencoder.py`·`conf/model/ae.yaml`·`conf/train/default.yaml` |
| 데이터 (동료 pq3) | 23 에피소드, GT phase 경계 91개, val 단일 split | 40 §한계 |
| 데이터 (grid 930판) | N1.5 grid 930 에피소드 (9 instruction × scene10 × noise10 + apple30), 89,766 record | 41 상단, ref(k24) n |
| margin 정의 | I(상태열;GT) − I(clock;GT), clock은 상태 수를 발견 상태열에 맞춘 진행도 분위 | 40 §2 |

## ★ 주장 1(메인) — activation으로 현재 phase를 읽을 수 있다 (held-out)

프로토콜: scene 단위 train/test 분리(test 2 scene, split seed 0/1/2 평균), 군집·최빈 phase
매핑·probe 전부 train scene에서만 적합. 스크립트 `scripts/analysis/grid_phase/phase_readout.py`,
결과 `outputs/analysis/grid_phase/phase_readout/readout.{tsv,json}`.

| 방법 | 정확도(중앙값) | macro-F1 |
|---|---|---|
| 다수 클래스 | 0.561 | 0.231 |
| 시간 대조군(진행도 분위) | 0.605 | 0.293 |
| **activation 비지도 군집→최빈 phase** | **0.856** | **0.570** |
| **activation 지도 probe(로지스틱)** | **0.893** | **0.721** |
| 시간만 넣은 지도 probe | 0.624 | 0.297 |

- activation > 시간 대조군: **9/10 instruction** (최대 격차 OpenDrawer left 0.26→0.73)
- 예외 DishwasherRack out: 군집 0.541 < 시계 0.570 (단 probe 0.747)
- 라벨 사용처는 군집→phase 매핑뿐 (구조 자체는 비지도)

## 주장 2 (보조) — GT보다 세밀한 하위 분할

| 수치 | 값 | 출처 |
|---|---|---|
| 구간 길이 (pq3) | GT 16.06 step vs 발견 4.3 step | 40 §5 부수 관찰 |
| 전환율 (pq3) | GT 5.0% vs 발견 18.4% (self-transition 0.816) | 40 §5 부수 관찰 |
| 구간 길이 (930판, k24 global) | GT 18.12 vs 6.00 step | ref align_global.json |
| 전환 수 (930판, k24) | GT 4,025 vs 발견 14,038 (≈3.5배) | ref align_global.json |
| 주기성 없음 | 전환 위치 mod 2/3/4/5 분포 균등 (최대/최소 1.02~1.47) | 40 §5 부수 관찰 |
| GT 경계 정렬은 우연 수준 | 24상태 boundary z −1.0~+0.5 (F1 0.20~0.23); GT 단위 병합 시 z +1.3~+4.9 | 40 §5 실험 1 |
| "k 산술" 반박 (raw+AE, k8) | 구간 길이 GT 대비 짧음 **10/10** (jug 4.2 vs 53.5, candle 4.8 vs 23.4, coffee 7.3 vs 29.8); 에피소드당 전환 수 > k−1=7 인 instruction **10/10** (7.3~30.0회 = 상태 재방문); OvenRack은 GT와 동급 해상도(8.5 vs 8.5) | ref ae_pertask_k8.json (mean_seg_len_*, n_transitions_*) |
| 새 수집분 GT 대비 | z −3.0~+0.3 (같은 패턴 재현) | 40 §5 실험 2 절 |

## 주장 2 — phase-순수성 (cluster는 특정 phase에 국한)

| 수치 | 값 | 출처 |
|---|---|---|
| MI (pq3, KMeans24) | 2.17 bits (상한 H(GT)=2.8243, cell_union) | 40 §3 |
| margin (pq3, KMeans24) | +1.665 bits (clock MI 0.51) | 40 §3 |
| purity_phase (pq3) | 0.796 | 40 §3·§5 |
| 자율 k 방식 열세 | HDBSCAN 최선 0.945, dendrogram gap 1.186 < KMeans24 1.665 | 40 §3 |
| 930판 per-task k8 | MI 0.25~0.76 bits, purity 0.54~0.93, margin +0.15~+0.51 (apple −0.63 퇴화 제외 시) | ref align_pertask.json (41 §5 요약과 일치) |
| 930판 k24 global pooled | MI 1.096, purity 0.751, margin +0.991 | ref align_global.json |
| scene 오염 (통제 전) | mi_scene 0.2~1.0 bits (dishwasher 1.029, drawer-right 1.018, OvenRack 0.947) | ref align_pertask.json |
| ① 게이트 (재현 검증) | rung3 per-task k8 대비 10/10 task **bit-identical** (max diff 0) | supp gate_check.json |
| 파이프라인 3종 비교 (930ep, k8, margin 중앙값 / 양수 수 / purity 중앙값) | PCA-64w 0.311·9/10·0.778 · raw 직결 0.291·10/10·0.857 · **raw+AE 0.343·10/10·0.875** → raw+AE 채택 | ae_raw/resid_compare_ae.tsv vs paper_supp·paper_raw 동일 파일 |
| ① scene 잔차화 후 mi_scene (raw+AE) | 중앙값 0.44 → **0.27** bits | ae_raw/resid_compare_ae.tsv |
| ① scene 잔차화 후 margin (raw+AE) | **10/10 instruction 양수 유지**, 중앙값 +0.343 → **+0.310** | ae_raw/resid_compare_ae.tsv |
| ① scene 잔차화 후 purity | 유지 (candle 0.910→0.901, jug 0.934→0.932, OvenRack 0.544→0.616) | supp resid_compare.tsv |
| ② off-phase 출현율 (raw+AE, per-cluster median) | instruction 중앙값 **0.07**; 최소 jug 0.002 ~ 최대 drawer-left 0.446 | ae_raw/contingency_pertask_k8_ae.json |
| ② purity (raw+AE, 930ep) | 중앙값 0.875 (candle 0.888, jug 0.955 ~ OvenRack 0.562) | ae_raw/resid_compare_ae.tsv |
| ② phase당 cluster 수 (argmax 기준) | 1.33~2.67 (중앙값 2.67 = phase 하나가 평균 2~3개 cluster로 세분) | supp contingency_pertask_k8.json |
| ② global k24 (raw+AE) | purity 0.717, off-phase median 0.371 | ae_raw/contingency_global_k24_ae.json |

## 주장 3 — 재현성 (특정 모델·데이터의 산물이 아님)

| 수치 | 값 | 출처 |
|---|---|---|
| AE↔SAE 같은 스텝 전환 일치 (pq3) | F1 0.66~0.75, 무작위 0.51~0.53, z +8.8~+13.5 | 40 §5 실험 2 |
| AE seed 간 재현 (새 수집분) | F1 0.55~0.58, 무작위 0.44~0.48, z +4.3~+5.1 | 40 §5 실험 2 |
| confound 소거 | denoise 고정 슬롯·cell/scene/seed는 에피소드 내 상수(23/23) → 기각 | 40 §5 해석 4 |

## 보조 — 세밀 phase 단위의 유용성 (succ/fail 층화)

| 수치 | 값 | 출처 |
|---|---|---|
| intrinsic k8 vs GT | intrinsic 우세 4/9 task, 동급 1, 열세 1 (drawer-left 0.93/z4.5 vs GT 0.94/z2.3; bread 0.77/z3.4 vs 0.67/z2.3; candle 0.73/z2.6 vs 0.62/z1.8) | 41 §5 |
| k sweep (하류 층화, 전 셀 mean z) | k6 +0.69, k8 +1.05, k12 +0.75, k16 +0.61, k24 −0.30, k32 +0.28 | 41 §8.3 |
| k sweep (내재 margin, task 중앙값·apple 제외) | k6 0.259, k8 0.337, k12 0.384, k16 0.428, k24 0.542, k32 0.582 — 단조 증가 (내재 지표만으론 최적 k 못 고름 = Fig.3 메시지) | ref align_pertask*.json (make_fig3.py 산출) |
| global k24 < per-task k8 | phase 구조가 task 특이적 | 41 §5 |
| ⚠ 한계 | 41은 탐색 라운드 (다중비교 보정 없음, n_perm 100~500, 위약 없음) — 원고에 보조로만 | 41 상단 |

## 온라인 적용 가능성 (결론 1문장용)

| 근거 | 출처 |
|---|---|
| 라벨 전이 = kNN(k=15), train에서만 클러스터 생성 → 재클러스터링 없이 스텝 단위 온라인 판독 가능 | 40 §2 |
| serve 실배선 존재 (`scripts/serve/lerobot.py --phase-readout`) | 인벤토리 (exp/grid-phase-sep) |

## 한계 절에 쓸 것

- GT 라벨 두 벌은 같은 라벨러의 두 버전 — 독립 반증 아님 (40 §한계)
- 전환 경계는 GT와 정렬 안 됨 (z≈0) — "phase 판독"의 근거는 MI/purity이지 경계가 아님 (40 §해석 3)
- scene 오염은 MI 쪽에만 걸림 (40 §5; ① 잔차화 결과로 방어)
- 길이 1 깜빡임 31~37% — 하위 구조 의미는 미검증 (40 §5 부수 관찰)
