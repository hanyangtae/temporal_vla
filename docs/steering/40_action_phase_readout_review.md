# 40. Action phase readout — 동료 실험 검증 기록 (2026-08-11)

동료(상우)의 "activation → action phase" 라인을 우리가 재구성·검증한 기록.
교수님 보고용 정리본은 Notion 「Action phase 정리」이고, **이 문서는 그 근거와 재현 방법**이다.

관련 메모리: `colleague-phase-repo`, `activation-transition-structure`.

> **용어** — 이 문서는 동료 코드/리포트 용어인 **상태열**(코드 키 `path`)을 그대로 쓴다.
> 대조가 쉬워야 하기 때문. 사람이 읽는 산출물(Notion 「Action phase 정리」, `docs/weekly/`)에서는
> 같은 것을 **클러스터 시퀀스**로 부른다 — 각 스텝의 activation이 어느 클러스터인지 시간 순으로
> 나열한 것이고, 값이 바뀌는 지점이 '전환'이다.

---

## 1. 대상과 접근 경로

| | |
|---|---|
| 동료 레포 | `ssh sangwoo_desktop` (166.104.67.158, user `iw`) → `~/code/task_classification` |
| 실행 환경 | **호스트에 torch 없음.** 이미 떠 있는 컨테이너에서 `docker exec -i task_classification python -` |
| 데이터 | `~/datasets/phase_cls_pq3/` (컨테이너 안에서는 `datasets_local/`로 bind-mount) |
| worktree | `~/code/task_classification_worktrees/wo_pca_test` (PCA 제거 실험) |
| 새 rollout | `kimseungjun@166.104.146.37 -p 11112` → `datasets/temporal_vla_store/groot/n15/grid/` |

노션 그림 ↔ 실험 디렉토리는 **md5 바이트 대조**로 확정했다(이전 회차 산출물이 섞여 있어 필수).

| 노션 그림 | 실험 디렉토리 |
|---|---|
| d08/d09/d10 (HDBSCAN) | `report/experiment_hdbscan_density` |
| a08/a09/a10 (Agglomerative) | `report/experiment_agglo_grid` |
| 타임라인 K=24 / K=70 | `report/experiment_agglo_k24` / `_k70` |
| sil01 / mi01 (PCA) | `wo_pca_test/report/experiment_pca_ablation_**v6**_ae128_{sil,mi}` |

제외 대상: `experiment_agglo_grid_k_deprecated`, `experiment_posthoc_*`(7월), `experiment_hdbscan_grid`·`_k10`,
**`experiment_hdbscan_density_eom`**(노션 최종 편집 이후 생성, 어떤 그림도 불일치), PCA v3~v5.

---

## 2. 파이프라인 사실 (노션 문서와 어긋나는 것 포함)

- **캡처**: DiT residual **layer 12 · denoise 3**, 49토큰 **평균**, 1536차원.
  → 동료 노션의 "Layer=15"는 코드와 불일치. git 전체 이력에 `layer: 15` 없음. **확인 필요 항목.**
- **판정 기준은 raw MI가 아니라 margin**:
  `margin = I(상태열 ; GT phase) − I(clock ; GT phase)` [bits].
  `clock`은 에피소드 내 진행도(0~1)를 **상태 수를 발견 상태열에 맞춰** 분위로 자른 대조군
  (`phase/metrics/structure.py: clock_clusters`). k 증가에 따른 MI 자동상승이 상쇄된다.
  → 길이/시간 confound가 이미 통제돼 있음. 우리가 따로 지적할 필요 없음.
- **라벨 전이**: train에서만 클러스터를 만들고 val/test는 **k-NN(k=15)** 로 옮김
  (`hdbscan_sweep.py: transfer_labels`, `KNN_TRANSFER=15`). 재클러스터링 아님 → **online 적용 가능**.
- **README는 자동 생성**(`agglo_sweep.py` 등의 `write_readme`). 그 안의 "권고"는 사람 판단이 아니라
  선택 규칙(knee / 가드 내 margin 최대)의 출력.
- **실루엣 절대값은 행 간 비교 불가** — 잠재 차원이 16~3072로 달라 고차원일수록 0으로 눌림
  (v6 README에 명시). 비교 가능한 것은 silhouette margin.

---

## 3. 실험 A — 클러스터 수를 데이터가 정하게 할 수 있는가

AE · val · seed 0-2 median. `report/experiment_agglo_grid/agglo_sweep.json`에서 직접 추출.

| 방법 | k | Silhouette | MI | clock MI | margin |
|---|---|---|---|---|---|
| clock 대조군 | - | - | 0.51 | 0.51 | 0 (정의) |
| KMeans (k 고정 24) | 24 | 0.256 | 2.17 | 0.51 | **1.665** |
| KMeans 대조 (sweep k 중앙값) | 16 | 0.267 | 2.01 | 0.50 | 1.508 |
| HDBSCAN 최선 (mcs75/ms7) | 7 | 0.106 | 1.37 | 0.43 | 0.945 |
| Agglomerative ward/euc thr 0.20 | 25 | 0.217 | 2.17 | 0.51 | 1.663 |
| Agglomerative complete/cosine thr 0.50 | 45 | 0.144 | 2.24 | 0.53 | 1.711 |

**핵심 결론 — threshold sweep은 k sweep의 재표현이다.** `distance_threshold`는 k에 단조라
"높이를 고르는 일"이 곧 "k를 고르는 일"이다(원 README도 동일 지적). 사람이 아무것도 안 정하는
유일한 답은 **dendrogram gap**이고, 그 결과는:

| linkage/metric | gap k | 최대 클러스터 비중 | margin |
|---|---|---|---|
| average/cosine | 6 | 33.0% | +1.186 |
| complete/cosine | 6 | 24.0% | +1.083 |
| ward/euclidean | 3 | 46.1% | +0.921 |
| 나머지 7쌍 | 2~6 | **94~100%** | +0.094 ~ −0.174 |

→ **자율 방식(gap 1.186, HDBSCAN 0.945)은 k 고정 KMeans(1.665)를 못 넘는다.**
게다가 gap도 linkage/metric 쌍 선택은 사람 몫이고 쌍에 따라 −0.174 ~ +1.186으로 갈린다.

**주의**: 같은 k에서 KMeans와 Agglomerative는 동률(k=24 → 1.665 vs k=25 → 1.663).
Agglomerative의 이점은 성능이 아니라 k를 미리 안 정해도 되는 것뿐이다.

---

## 4. 실험 B — PCA 압축이 정보를 잃고 있었는가

**⚠ 실험 A와 split이 다르다** — B는 `default.json`(val 1,919 step · H 2.8409 · clock 0.586),
A는 `cell_union.json`(val 1,831 step · H 2.8243 · clock 0.505). AE run도 별개
(`ae-log_likelihood-s*` vs `ae-log_likelihood-union-s*`). **두 실험의 절대값을 직접 맞대면 안 된다.**

MI margin [bits] (KMeans 24 states, val, seed 3개 평균):

| 모델 | PCA-64 | raw-1536 | PCA-1536 whitened |
|---|---|---|---|
| AE (latent 16) | 1.45 | **1.46** | −0.32 |
| AE (latent 128) | 1.38 | 1.43 | −0.30 |
| SAE (1536, top-k 16) | 1.02 | 1.15 | −0.44 |
| SAE (1536, top-k 192) | 1.25 | 1.38 | −0.10 |
| SAE (3072, top-k 16) | 0.55 | **1.20** | −0.44 |
| SAE (3072, top-k 384) | 1.33 | 1.36 | −0.28 |

- raw-1536 우세(AE는 미미, SAE는 큼), AE > SAE
- **PCA-1536 whitened는 margin이 전 행 음수** = 시계 대조군보다 못함
- 원인은 차원 축소가 아니라 **whitening**. `n_comp=1536, whiten=True`는 분산 100% 보존이므로
  정보 손실 0인 가역 변환인데도 무너진다 → 모든 축을 분산 1로 맞추면 노이즈 축이 증폭되고
  reconstruction loss가 그것까지 복원한다. **절단 후 whitening은 되고, whitening만은 안 된다.**
  (PCA-64는 EVR 87.7%로 노이즈 바닥을 먼저 잘라낸다. `phase/data/pca.py` docstring 참조 —
  PC5는 분산 2.2%인데 η²=0.215로 PC1과 맞먹어, whitening 없이는 버려진다.)

---

## 5. 경계 검출 · GT 기준 검토 (이번 세션 신규 실험)

**판정 도구**: 전환 개수는 그대로 두고 위치만 에피소드 내에서 무작위로 흩은 상태열 300회를
기준선으로 삼고 `z = (실제 F1 − 무작위 평균) / 무작위 표준편차`. boundary F1은 ±tol 허용
그리디 1:1 매칭(`phase/metrics/boundary.py`, 기본 tol=1 = ±5 env-step).

### 실험 1 — 해상도

| 해상도 | F1 | 무작위 기준선 | z | 전환 수 |
|---|---|---|---|---|
| 24상태 그대로 | 0.20~0.23 | 0.215~0.221 | −1.0 ~ +0.5 | 453 |
| GT phase 단위 병합 (train 다수결, 24→8종) | 0.27~0.41 | 0.220~0.249 | **+1.3 ~ +4.9** | 143 |

(GT 경계는 91개. tol을 ±25 env-step까지 늘려도 24상태의 precision은 0.17에서 정체 →
tolerance 문제가 아니라 과분할.)

### 실험 2 — 재현성 (GT 미사용)

| 비교 | 무엇을 확인 | F1 | 무작위 기준선 | z |
|---|---|---|---|---|
| AE vs SAE — pq3, **같은 스텝** | 구조를 바꿔도 같은 곳에서 전환하는가 (**일치**) | 0.66~0.75 | 0.51~0.53 | +8.8 ~ +13.5 |
| AE seed 끼리 — **새 수집분** | 다른 데이터에서도 같은 현상이 나오는가 (**재현**) | 0.55~0.58 | 0.44~0.48 | +4.3 ~ +5.1 |

> ⚠ **두 행은 성격이 다르다.** 위는 같은 스텝을 두 모델이 맞대는 *일치* 검사이고, 아래는 다른
> 데이터에서 같은 성질이 나오는지 보는 *재현* 검사다. 데이터가 다르면 에피소드가 달라
> 전환 위치를 직접 맞댈 수 없다 — "데이터를 바꿔도 같은 위치"가 아니다.
>
> 배제하는 것도 다르다. 위는 "이 모델 구조 탓", 아래는 "pq3 데이터 특유 · AE 파이프라인 특유".
> 아래는 seed가 AE 학습 전체를 바꾸므로 변동이 크다(그래서 z가 위보다 낮다).
> 더 강한 버전은 새 데이터에서 AE↔SAE cross-architecture 일치를 재는 것 — 미실행.

새 수집분 = 승준 아카이브 2026-08 grid 재수집(다른 머신 · `OpenDrawer/left`·`PPCC/bread` · 44 ep).
동료와 **같은 규격으로 캐시를 만들고 동료 트레이너로 AE를 seed 3개 새로 학습**했다
(`layer 12 · denoise 3 · 49토큰 평균 → PCA-64 whitened → AE(16) → KMeans24`).
같은 데이터에서 GT 대비는 **z −3.0 ~ +0.3**으로 여전히 우연 수준(pq3와 같은 패턴).

**재현 절차** (전부 컨테이너 `/tmp`, 동료 레포에는 쓰지 않는다): 승준에서 `[N,1536]` 특징+meta 추출 →
`docker cp`로 컨테이너 `/tmp` → `phase.data.pca`로 PCA-64 적합 후
`derived/L12-D3-pca64w/{split}.x.npy` + `.steps.parquet` + `meta.json` 작성 →
`python -m phase.train.posthoc data.root=/tmp/new_phase seed=N wandb.mode=disabled hydra.run.dir=/tmp/aeruns/ae-sN`

### 부록 — 압축기가 필요한가 (우리 추가 실험, 이번 보고 미포함)

동료 설계에 **"압축기 없음" 기준선이 없다.** 그들의 KMeans 대조군조차 AE latent 위에서 돈다.
pq3 · cell_union · k=24 · seed 3개로 채워 본 결과:

| 경로 | 차원 | MI | margin | boundary F1 | purity | mi_cell |
|---|---|---|---|---|---|---|
| PCA-64 → **AE** → KMeans24 | 16 | 2.197 | **1.692** | 0.169 | 0.813 | 1.490 |
| PCA-64 → KMeans24 (**압축기 없음**) | 64 | 2.139 | **1.634** | 0.166 | 0.800 | 1.569 |
| PCA-64 → **SAE** → KMeans24 | 128 | 2.067 | **1.562** | 0.145 | 0.748 | 1.410 |

seed별 margin — AE `1.692 1.667 1.696` / 없음 `1.634 1.682 1.570` / SAE `1.531 1.562 1.571`

- **SAE는 순손실** — 3 seed 모두 압축기 없음보다 낮다. "AE > SAE"가 아니라 **"AE ≳ 없음 > SAE"**.
- **AE는 미미한 이득** — +0.058 bits, 범위가 겹친다(없음 최대 1.682 vs AE 최소 1.667). n=3이라 유의성 판단 불가.
- silhouette은 차원이 달라(64/16/128) 행 간 비교 불가.
- 한계: 단일 split · k=24 고정 · SAE는 이 union 설정 하나(실험 B에서 SAE margin은 0.55~1.38로 흔들림).
- **이번 보고에는 넣지 않았다** — "encoder 불필요"는 더 큰 주장이라 제대로 설계해 따로 다룰 것.

### 부수 관찰

- **주기성 없음** — 전환 위치의 mod 2/3/4/5 분포가 거의 균등(최대/최소 1.02~1.47) → 고정 리듬 아티팩트 아님
- **길이 1 깜빡임이 구간의 31~37%** — 과분할의 실체. 최소 구간 길이 제약·시간 평활이 유력한
  개선 축이나 **미검증**
- 구간 길이: GT 평균 16.06 step vs 발견 상태 4.3 step. self-transition 0.816(전환율 18.4%) vs GT 5.0%.
  purity_phase는 0.796 — 스텝별 판독은 잘 되는데 전환이 잦다

### 해석

1. **클러스터 정체성은 phase와 강하게 연관** — MI 2.17 bits(상한 2.82), margin 1.665, purity 0.796.
   단 1:1 대응은 아니다(여러 클러스터가 한 phase를 나눠 가짐).
2. **전환은 실재한다** — 같은 데이터에서 구조가 다른 두 모델(AE·SAE)이 같은 지점에서 전환하고,
   다른 데이터에서 AE를 새로 학습해도 같은 성질이 재현된다. 특정 모델의 산물이 아니다.
3. **다만 전환이 GT phase 경계를 나타내는 것은 아니다** — 그쪽은 z ≈ 0. 즉 "클러스터가 phase 정보를
   담는다"의 근거는 **MI 쪽**이고 전환 쪽이 아니다. 둘이 공존하는 이유는 MI가 잘게 쪼갠 정도에
   둔감하기 때문(아래 §MI와 경계 F1이 갈리는 이유).
4. **그럼 전환은 무엇인가 — 후보 소거**

   | 후보 | 판정 |
   |---|---|
   | 로봇 동작 · 접촉 · 시각 변화 | **phase 계열** — 대안이 아니라 GT phase를 만드는 재료 그 자체 |
   | denoising step 특성 | **기각** — `denoise=3` 고정 슬롯(`resolve_slice`가 단일 인덱스로 해석). 시간축에서 불변 |
   | cell · scene · seed | **기각** — 에피소드 내 상수(실측 23/23 에피소드에서 불변). 상수는 에피소드 내 전환을 만들 수 없다 |
   | 모델 내부의 다른 축(재계획·불확실성 등) | 남음, 미검증 |
   | 시뮬레이션·렌더 아티팩트 | 남음, 미검증 |

   → 배제되고 남는 가장 자연스러운 설명은 **"phase와 같은 계열의, GT보다 세밀한 하위 구조"**.
   "~로 보인다"까지가 현재 근거로 갈 수 있는 최대다.
5. **그래서 기준 문제가 열린다** — (A) GT 유지(병합 해상도에서 평가) / (B) 자체 재현성을 1급 지표로 /
   (C) 하류 유용성(개입 효과)으로 판정. **미결.**

> cell·scene 오염 우려는 **MI 쪽에만** 걸린다(mi_cell 1.49, H(cell)≈2.30). 전환 쪽은 cell이
> 에피소드 내 상수라 구조적으로 무관하다.

### 한계 (먼저 말할 것)

- **"그 구조가 phase다"는 증거가 아니다.** 재현되는 무언가가 있다는 것까지.
- **GT 비정렬이 두 번 재현됐지만 독립 반증이 아니다** — pq3 라벨과 새 데이터 라벨은
  **같은 라벨러의 두 버전**(task 수만 확장). "서로 다른 정의 모두와 어긋난다"고는 말할 수 없고,
  "같은 라벨 체계를 다른 수집분·다른 task에 적용해도 같은 비정렬이 재현된다"까지가 정확한 진술.
- **병합 실험은 지도 성분이 섞인다** — train 다수결 매핑 사용. 누수는 아니나(매핑은 train 적합,
  평가는 val) "비지도 분절"이 아니라 "비지도 상태 + train 적합 판독"이다.
- 표본: pq3는 GT 경계 91개 · 23 에피소드 · val 단일 split. split 비교 단계는 공통 에피소드 6개(442 step).

### MI와 경계 F1이 갈리는 이유

`MI = H(Y) − H(Y|C)`이고 `H(Y)`는 상수이므로, MI는 **각 클러스터가 phase에 대해 얼마나 순수한가**를
크기 가중해 비트로 잰 값이다(관련 지표 `purity_phase = 0.796`). 여기서:

- 여러 클러스터가 한 phase를 나눠 가져도 MI는 안 떨어진다 (1:1 대응 불필요)
- **같은 phase를 가리키는 클러스터를 합쳐도 `H(Y|C)`가 거의 그대로** → MI는 "얼마나 잘게 쪼갰나"에 둔감
- 반면 경계 F1은 전환 횟수에 직접 반응한다

→ MI가 높은데 경계가 낮았던 것, 그리고 병합이 MI는 안 건드리고 경계만 올린 것이 같은 원리다.

---

## 6. 재현 방법

스크립트는 세션 임시 디렉토리에 있었으므로 **필요하면 이 문서를 보고 재작성**한다. 핵심은 다음 3줄.

```bash
# 동료 레포 — 컨테이너 안에서 실행 (호스트에 torch 없음)
ssh sangwoo_desktop 'docker exec -i -e OMP_NUM_THREADS=8 task_classification python -' < script.py

# 새 rollout — 승준 노드, anaconda python (base python3는 torch 없음)
ssh -p 11112 kimseungjun@166.104.146.37 'OMP_NUM_THREADS=8 ~/anaconda3/bin/python -' < script.py
```

동료 파이프라인 재적합은 `phase.analysis.cluster_view.refit(model, seed, combo, "cpu", "knn")`
하나로 끝난다. combo 예시:

```python
{"clusterer": "agglo", "id": "ward_euclidean_k24", "linkage": "ward",
 "metric": "euclidean", "k_target": 24}
```

새 rollout에서 특징 만들기: `hidden_states[t]`가 `[7, 4, 49, 1536]`이므로
`capture_layers.index(12)=5`, denoise `3`, 토큰축 평균 → `[1536]`.
GT phase는 `env_step_phases[i * env_step_n_action_steps]`.

**검증 게이트**: 무엇을 재계산하든 먼저 **tol=1이 기존 보고 수치를 재현하는지** 확인할 것
(k24 → F1 0.149 / prec 0.090 / rec 0.429).

---

## 7. 산출물

- Notion 「Action phase 정리」 — 교수님 보고용 (개조식, 그림 15장)
- 그림 원본 `outputs/action_phase_report/` (gitignore) —
  `notion_figs/`(동료 게재본 21장), `margin_figs/`(미게재 margin 3장, 붙임 완료),
  `agglo_alt/`(margin 곡선·heatmap), `resolution_timeline.png`(신규 생성)

## 8. 동료 확인 필요

1. **layer 불일치** — 코드 전 run이 layer 12인데 이전 회차 보고는 "Layer=15". 출처 확인
2. **K 표기 혼용** — 클러스터 수 / AE latent 차원 / SAE top-k 3중 사용. 그림 축 기준 통일 필요
3. PCA·whitening 통계의 train-only 적합 여부 (`pca.py` docstring은 train-only라고 명시, 호출 경로 미확인)
