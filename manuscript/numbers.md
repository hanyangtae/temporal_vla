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


## ★ 누수 제거판 (최종 게재 수치, 08-24) — Codex 리뷰 ⑤ 대응

`ae_split_readout.py`: split(seed 0/1/2)마다 표준화·AE·KMeans·매핑을 **train scene만으로**
재적합. 원자료 `manuscript/ref/split_ae_readout.json` (구 누수판 백업: readout_leakyAE.tsv).

| 지표 | 구(전체 AE) | 신(무누수) |
|---|---|---|
| 정확도 중앙값 | 0.856 | **0.822** |
| macro-F1 중앙값 | 0.570 | **0.574** |
| vs 행동 이벤트 우위 | 10/10 | **9/10** (예외 DishwasherRack 0.489<0.516) |

- Δacc instruction별 −0.05~+0.08 양방향(OvenRack +0.084 상승) → 체계적 누수 이득 없음.
- 일부 변화는 AE seed 차이(구 seed0 고정, 신 1000+split) 잡음 포함.
- k sweep도 무누수 재실행 (split_ae_ksweep.json, 완료 후 3.2·그림 갱신).

## ★ 주장 1(메인) — activation으로 현재 phase를 읽을 수 있다 (held-out)

프로토콜: scene 단위 train/test 분리(test 2 scene, split seed 0/1/2 평균), 군집·최빈 phase
매핑·probe 전부 train scene에서만 적합. 스크립트 `scripts/analysis/grid_phase/phase_readout.py`,
결과 `outputs/analysis/grid_phase/phase_readout/readout.{tsv,json}`.

| 방법 | 입력 | 정확도(중앙값) | macro-F1 |
|---|---|---|---|
| 다수 클래스 | — | 0.561 | 0.231 |
| 시간(절대 스텝 t, 인과적) | 시간 | 0.675 | 0.307 |
| 시간(진행도 t/T, 길이 오라클) | 시간 | 0.605 | 0.293 |
| 시간 지도 probe(t/T) | 시간 | 0.624 | 0.297 |
| 정책 행동 비지도 군집 | 외부 관찰 | 0.557 | 0.256 |
| 정책 행동 지도 probe | 외부 관찰 | 0.614 | 0.339 |
| **행동+시간 지도 probe (최강 외부 대조군)** | 외부 관찰 | 0.773 | 0.559 |
| **activation 비지도 군집→최빈 phase** | 내부 | **0.856** | **0.570** |
| **activation 지도 probe** | 내부 | **0.893** | **0.721** |

- 행동 특징 28차원 = record별 action(7) 평균·표준편차·마지막 + 에피소드 시작부터 누적합(팔 위치 대리값). 원자료 `traj.csv` (수집 산출물, 2,236 에피소드 색인)
- **같은 지도 조건 비교(probe vs 행동+시간): activation 승 9/10** — 예외 OpenDrawer left −0.094
- 비지도 군집 vs 최강 외부 대조군은 6/10 (지도/비지도 조건이 달라 참고치)
- 라벨 사용처는 군집→phase 매핑뿐 (구조 자체는 비지도)
- ⚠ 정책행동·행동+시간·다수클래스 대조군은 08-20 사용자 지시로 **논문에서 제외**
  (외부 관찰 역할은 아래 행동 이벤트 대조군이 대신). 위 두 불릿은 기록용.

### 행동 이벤트(behavioral event) 대조군 — Event-SAE(arXiv:2605.17204) 저자 파이프 재현

같은 930판·같은 scene split(seed 0/1/2, test [6,7]/[4,5]/[2,7] 일치 검증).
파이프: AWE dp_waypoint(pos_only, η=0.05) → 5프레임([-4,-2,0,2,4]) SigLIP-base 평균 →
descriptor [1.0·L2(vision)‖0.5·L2(zscore(state))‖0.4·zscore(progress)]→행L2 →
agglomerative(cosine 0.18). 전 단계 저자 코드 무수정(`run_esae.sh`), 판독 연결만 우리
프로토콜(`esae_readout.py`): z-score·군집·최빈 매핑 train scene에서만 적합, test
waypoint는 train centroid 최근접, record 예측 = 시간상 최근접 waypoint의 phase.

| instruction | 행동 이벤트 acc | (참고) clock | activation 군집 |
|---|---|---|---|
| CoffeeSetupMug | 0.748 | 0.703 | 0.862 |
| DishwasherRack_out | 0.516 | 0.570 | 0.541 |
| OpenDrawer_left | 0.451 | 0.517 | 0.733 |
| OpenDrawer_right | 0.449 | 0.647 | 0.784 |
| PPCC_apple | 0.721 | 0.760 | 0.861 |
| PPCC_bread | 0.780 | 0.739 | 0.906 |
| PPCC_candle | 0.685 | 0.550 | 0.852 |
| OvenRack_out | 0.402 | 0.510 | 0.487 |
| PPCC_jug | 0.849 | 0.850 | 0.916 |
| PPCC_marshmallow | 0.612 | 0.720 | 0.875 |
| **중앙값 (10/10)** | **0.648** | 0.675 | 0.856 |

- activation 군집 > 행동 이벤트: **10/10 전부** (그림 3b)
- OvenRack·marshmallow는 재수집 클린 영상 기준 최종치(08-21). marshmallow는 오염
  11판 교체 전후 acc 0.612로 동일(오염분이 전부 train scene 쪽이었음).
- 원자료 `outputs/analysis/grid_phase/phase_readout/esae.json` (n_seed=3 검증 완료)

### k 강건성 (판독 정확도 중앙값, 비지도 군집)

| k (무누수, 08-24) | 4 | 6 | **8** | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| 정확도 | 0.628 | 0.782 | **0.822** | 0.848 | 0.864 | 0.871 | 0.878 |
| macro-F1 | 0.289 | 0.525 | **0.574** | 0.574 | 0.606 | 0.607 | 0.687 |

(구 누수판: acc 0.621/0.798/0.856/0.876/0.878/0.879/0.888 — split_ae_ksweep.json으로 대체)

사람 phase 수(3~6)를 넘는 k≥8에서 평평 → **특정 k 선택에 기대지 않음**.
원자료 `outputs/analysis/grid_phase/phase_readout/k_sweep.json`.
(구 k=24는 동료 파이프라인 기본값을 물려받은 것으로 원칙적 근거 없음 — 이 sweep으로 대체)

### phase 빈도별 F1 격차 (activation probe − 행동+시간)

| phase 비중 | 행동+시간 | activation | 차이 |
|---|---|---|---|
| 희소 (<15%, n=18) | 0.351 | 0.460 | +0.109 |
| 중간 (n=12) | 0.572 | 0.653 | +0.081 |
| 흔함 (>35%, n=10) | 0.835 | 0.907 | +0.072 |

방향은 희소 쪽이 크지만 편차가 큼(CoffeeSetupMug transport +0.804 vs OpenDrawer left
disengage −0.725) → 원고에는 "드문 phase일수록 우위"라고 쓰지 말 것, "macro-F1에서도
우위 유지"까지만.

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
