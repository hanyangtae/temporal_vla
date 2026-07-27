# 29. 동료(robots-oh) SAE 레포 전수 검토 + 이식 준비 보고서

작성 2026-07-26. 대상 = `task_classification/` (dev 핀 커밋 `88543a2`, PR #2 `phase_post_hoc_analysis` 머지).
이식 목표 = exp4-3 권고 SAE scene-feature 분리(G1/G2/G3, `docs/steering/25a_exp4-3_to_exp4-1_recommendation.md`).

근거는 모두 `task_classification/` 기준 상대경로 `파일:라인`. 추측 배제, 코드 실측만.

---

## 0. 한 줄 결론

동료 레포의 SAE는 **잘 만든 top-k SAE 코어**(주입식·문서화 양호)지만, 목적이 **phase(하위상태) 비지도 발견**이지
scene-feature 분리가 아니다. **모델·학습·클러스터·metric 코어는 그대로 재사용 가능**하나, **입력 계약이
정면충돌**한다: (a) 토큰 축을 평균으로 없앤다(우리는 per-token 보존 필수), (b) 1536→PCA64로 압축한 뒤
넣는다(overcomplete 사전 전제와 상충), (c) 평가 라벨이 phase지 scene이 아니다. **이식 1단계 = 우리 데이터
계약([L,K,T,D]·per-record) 전용 어댑터 + per-token SAE 입력 빌더 신규 작성**, 코어 클래스는 lift-and-reuse.

---

## 1. 레포 전체 구조 지도

top-level: `phase/`(핵심 패키지) · `conf/`(Hydra) · `script/data/`(GT 라벨 생성 도구) · `tests/` · `process_management/`(문서) · `docker/`.

```
phase/
  data/      source.py(원본 step 테이블 읽기) pca.py(train-only PCA-whiten)
             build.py(데이터셋→PCA캐시) make_phase_dataset.py(pkl→데이터셋, 토큰평균 ★)
             make_phase_dataset_legacy.py(이미 pooled용) splits.py source.py
  models/    autoencoder.py(AE/SAE 전부 ★) factory.py(build_model 레지스트리) classifier.py
  train/     posthoc.py(AE/SAE 엔트리) classify.py(지도 상한) _loop.py(공용 fit/epoch)
  clustering/ posthoc.py(fit_clusters/assign) gpu.py(GPUKMeans/GPUGMM)
  metrics/   base.py(Metric·EvalContext) uncertainty.py purity.py silhouette.py
             boundary.py self_transition.py registry.py runeval.py report_fmt.py
  analysis/  scoped_report.py report.py video.py rollouts.py classification.py
             colormap.py inference.py load.py cross_report.py paths.py
```

- **엔트리포인트**: `python -m phase.train.posthoc experiment={ae,sae}` (비지도 발견 메인),
  `python -m phase.train.classify experiment=cls` (지도 상한 기준선), `python -m phase.data.build`(PCA 캐시).
- **문서**: `process_management/pipeline_guide.md`(전체 흐름 인수인계), `phase_cls_dataset_structure.md`(데이터 포맷 상세),
  `training_handoff.md`, `analysis_tool_refactor_plan.md`, `git_worktree_guide.md`. README.md는 한 줄뿐.
- **테스트**: `tests/test_metrics.py`(140줄) + `tests/test_analysis.py`(92줄) — metrics 로직·colormap·run 파싱만.
  **모델(autoencoder.py) forward/loss·SAE 자체 테스트는 없음.**
- **의존성**(`pyproject.toml`): torch, numpy, pandas, pyarrow, scikit-learn, hydra-core>=1.3, wandb.

**목적 요약**: "DiT hidden state에서 라벨 없이 phase(task 하위상태)를 발견"(`pipeline_guide.md:1-5`).
이산 구조는 학습에 개입 안 하고, 학습 후 잠재를 KMeans/GMM으로 사후 이산화(`posthoc.py:1-8`).

---

## 2. SAE 구현 상세

### 2.1 아키텍처 = top-k SAE (L1 없음)

- **Encoder** = `EncoderTopK` (`phase/models/autoencoder.py:95-133`): `h = top-k(ReLU(W_e x)) ∈ R^m`.
  `hidden=None`이면 `nn.Linear`(표준 선형 사전), 값 주면 2-layer MLP 백본(AE와 capacity 매칭 변형).
  top-k 마스킹은 k번째 값 임계 + `torch.where`(`:122-126`) — gather/scatter보다 짧으나 동점 시 k 초과 가능(연속값이라 실질 무발생).
  **희소성이 top-k 구조로 강제되므로 손실에 L1 페널티 없음**(`:98-100`). `density()` 진단 = 평균 활성비 k/m(`:128-133`).
- **Decoder** = `DecoderLinearDict` (`:136-156`): 단위노름 선형 사전 `w/‖w‖`(`:155`), 비선형층 없음(해석성 전제).
  clamp_min(1e-8)로 죽은 열 0-나눗셈 방지. `(x̂, logvar)` 계약.
- **컨테이너** = `BaseAE` (`:159-233`): enc/dec **주입**받아 조립. loss는 재구성 NLL 하나(`:195-223`).
  `loss='mse'`면 decoder.logvar를 0에 동결(`:189-193`) → NLL이 절대오차(=MSE argmin). `latent(x)`(`:225-233`)가 사후 클러스터링용 코드 반환.
- **조립** = `factory.py:24-43`: `sae = {encoder: topk, decoder: linear_dict}`, `ae = {encoder: mlp, decoder: mlp}`.
  모델 종류가 클래스가 아니라 **설정**(`kind`)으로 표현. `VariationalAE`는 DEPRECATED(`autoencoder.py:236-264`).
- **gated/jump-relu 없음.** vanilla dense AE + top-k SAE 두 종류뿐.

### 2.2 하이퍼파라미터 (`conf/model/sae.yaml`)

| 항목 | 값 | 근거 |
|---|---|---|
| input_dim | **64** | PCA-whitened 차원 (원본 1536 아님 ★) |
| latent_dim(m, 사전 크기) | **128** | "input_dim보다 커야 과완비" → **64 대비만 overcomplete** |
| k (top-k) | **16** | 매 시점 활성 코드 수 (density 목표 16/128=12.5%) |
| hidden | (생략) | 선형 사전 (표준 SAE) |
| loss | **mse** | logvar 0 동결 = 절대오차 |

### 2.3 학습 루프 (`phase/train/_loop.py` + `posthoc.py`)

- optimizer = **Adam, lr 1e-3, weight_decay 0**(`conf/train/sae.yaml`). wd=0 이유 = 사전 수축이 단위노름 전제와 충돌.
  (AE 기본은 AdamW wd 1e-4, `conf/train/default.yaml`.)
- **full-batch**(batch_size=null, `conf/train/default.yaml`) — train ~8.5k step이 GPU 상주, DataLoader 없이 텐서 인덱싱(`_loop.py:23-49`).
- epochs 800, patience 60, min_epochs 60, grad_clip 5.0. **val 손실 최소화 early stopping**(`_loop.py:62-89`), best state CPU 보관.
- **aux loss 없음, dead-feature resampling 없음.** dead feature는 학습이 아니라 **클러스터 단계에서** 처리 —
  `drop_inactive=True`(`conf/cluster/sparse.yaml`)로 "train에서 한 번도 안 켜진 차원 제거 후 적합"(`clustering/posthoc.py:14-42`).
- 학습 후 처리: train 잠재로만 KMeans+full-cov GMM 적합(누수 방지, `posthoc.py:69-74`) → cell별 평가.

### 2.4 입력 데이터 포맷·전처리

- 원본 shards: `*.dit.npy [N, L=7, D=4(denoise), F=1536] fp16` (`phase_cls_dataset_structure.md:88-95`), step 단위 `index.parquet`.
- **layer 12 / denoise 3 슬라이스 → 1536-d**(`conf/data/pq3_l12_pca64.yaml`, `source.py:50-80`).
- **PCA(train-only) + whitening → 64-d**(`pca.py`, `build.py:70-75`). whiten 이유 = 저분산 phase 방향 보존(`pca.py:1-8`).
  → **SAE가 보는 x는 원본 activation이 아니라 PCA-64 whitened 벡터.**
- 정규화 = PCA-whitening이 사실상 표준화 역할. 별도 mean-subtract/unit-norm 없음.

### 2.5 체크포인트 포맷 (`posthoc.py:100-109`)

`model.pt` = `{state_dict, config(resolved), summary, kmeans_centers, gmm_means, active(bool mask)}`.
`paths.npz` = 상태열 + episode/t. `metrics.json` = U(phase|z)·silhouette 등.

---

## 3. 동료의 사용 목적 vs 우리 목적

- **동료 목적** = **phase 비지도 발견**. SAE는 표현 후보 중 하나(AE/SAE 2종). 학습은 재구성만, 이산 상태는 사후 클러스터링.
  평가 = 발견 상태열이 **GT phase**를 얼마나 설명하나 → 주 지표 **Uncertainty Coefficient U(phase|z)**(`metrics/registry.py:22-32`),
  silhouette, purity. clock(시간분위) 기준선 대비 `beats_clock`(`registry.py:35-49`).
- **cell = task×object 정체성**(예: ppcc_bread, drawer_left)을 **nuisance**로 진단: `u_cell`, `cell_dominant`(상태열이 phase보다
  cell을 더 잡으면 경보, `registry.py:44-49`). → **동료도 "scene/task 정체성 오염"을 이미 인지하고 diagnostic으로 감시**.
  다만 **cell을 빼내는(잔차화) 방향이 아니라, phase를 더 잘 잡는지의 기준선으로만 사용.**
- **분류(task classification)**는 별도 지도 트랙(`classify.py`, `classifier.py`) — SAE feature→분류가 아니라 raw feature→phase의 지도 상한.
- **PR #2(phase_post_hoc_analysis)** = 사후 이산화 파이프라인 리팩토링(Hydra+W&B) + metrics 모듈화 + analysis 레이어 + scope별 리포트/영상
  (git log `a1963df`~`88543a2`). SAE 자체 로직 변경보다 **평가·리포팅 인프라** 정비가 핵심.

**우리 목적과의 갭 = 방향이 반대.** 동료: "발견 상태가 phase를 잡나(cell 오염은 감점)". 우리(G1/G2): "SAE feature가 **scene을
명시적으로 인코딩**함을 실측(scene 라벨 예측력)한 뒤 **scene을 빼고** succ/fail read 잔존을 본다". 즉 동료가 nuisance로 감시하던
cell/scene을 **우리는 타깃으로 삼아 분리·제거**한다. 코어(SAE 학습·클러스터·U-coefficient)는 공유하나 **실험 프레임·라벨·잔차화·write는 신규**.

---

## 4. 데이터 인터페이스 갭 (핵심)

| 축 | 동료 입력 | 우리 데이터 계약 | 갭 / 필요한 변환 |
|---|---|---|---|
| 모델 | GR00T **N1.5** RoboCasa DiT | N1.5 / N1.6 / Cosmos | 동일 계열(N1.5). N1.6·Cosmos는 layer·D·차원 다름 |
| activation | DiT hidden `[step, L=7, D=4, F=1536]` | N1.5 `[L,K,T,D=1536]` / N1.6 `[32,51,1536]` / Cosmos `[28,2048]` | **동료엔 토큰(T) 축이 이미 없음** |
| **토큰 축** | **평균으로 제거**(`make_phase_dataset.py:102-109`, `a.mean(axis=2)`) | **per-token(T) 보존 필수** (rollout pooling 금지, exp4-3 §4) | ★ per-token 빌더 신규 — 토큰 평균 제거, T를 record 축으로 펼침 |
| 차원 압축 | 1536 → **PCA-64 whiten**(`pca.py`) | 원본 D(1536/2048) 유지 권장 | ★ SAE 사전 overcomplete는 PCA-64 대비만. scene-feature엔 원본 D 입력 + m≫D 재설계 |
| 라벨 | GT **phase**(oracle, `env_step_phases`) | **scene**(layout/style/scenario_seed) + succ/fail | ★ scene 라벨 컬럼 신규(현재 `cell`=task×object, `seed` 별도) |
| 길이/phase 통제 | 없음(phase가 타깃) | dwell cap·순열 null·held-out (G2) | ★ 신규 — 우리 confound-audit 규약 |
| write | 없음(발견 전용) | steering(G3, `scripts/serve/steering_hooks.py`) | ★ 전면 신규 |

**결정적 충돌 = 토큰 평균**(`make_phase_dataset.py:102-109`, `_pool_dit`/`_pool_vl`). 동료의 **기본** 빌더가 토큰 축을
mean으로 없앤다(meta에 `token_pooling.applied=True` 기록, `:346-356`). 우리는 per-token SAE가 목표(exp4-3에서 N1.6 T=51
토큰을 일부러 보존해 둠). **동료의 데이터 빌더(make_phase_dataset)는 재사용 불가**, per-token 전용 빌더를 새로 써야 한다.
(주의: 이는 rollout-t pooling과 다른 토큰-pooling이지만, 우리 메모리 `feedback-no-rollout-pooling`·exp4-3 §4가 per-token 보존을
명시하므로 평균 금지.)

---

## 5. 이식 계획 (단계별 체크리스트)

### 5.A 그대로 가져올 코어 (lift-and-reuse, hydra 비의존)

- [ ] `phase/models/autoencoder.py` → `EncoderTopK`, `DecoderLinearDict`, `BaseAE` (VAE·SeedConditioned·Variational 제외).
      순수 torch, 외부 의존 없음. **이식 대상 1순위.**
- [ ] `phase/models/factory.py` → `build_model` 레지스트리(DictConfig 대신 dict도 받음, `_plain`으로 omegaconf 비의존, `:48-51`).
- [ ] `phase/train/_loop.py` → `epoch`/`fit`/`make_optimizer`. wandb_init만 옵션(mode=disabled 경로 있음, `:52-59`).
- [ ] `phase/clustering/posthoc.py` → `fit_clusters`/`assign` (dead-feature `drop_inactive` 처리 포함). sklearn 경로만 쓰면 gpu.py 불필요.
- [ ] `phase/metrics/{base,uncertainty,purity,silhouette}.py` + `registry.py` → G2의 "scene/succ-fail 라벨 예측력" 측정에 재사용.
      `EvalContext`에 `clock`(시간분위) 기준선 내장(`base.py`) → **길이 통제 대조군을 공짜로 얻음**(우리 confound-audit 정합).
- [ ] `phase/data/pca.py` → train-only PCA-whiten(누수 안전). scene용 전처리로 선택 사용(잔차화 前 정규화).

### 5.B 새로 써야 하는 것 (우리 데이터 계약 전용)

- [ ] **per-token SAE 입력 어댑터**: N1.5 fit30 pkl `[L,K,T,D]` → per-record(=per-timestep×per-token 유지) x-스트림.
      동료 `make_phase_dataset.py`의 토큰 평균(`:102-117`)을 **제거**하고 T를 record 축으로 flatten. layer·denoise(K) 선택 인자화(`source.resolve_slice` 참고).
- [ ] **scene 라벨 부착**: layout/style/scenario_seed를 record 메타에 추가(동료 `cell`=task×object로는 부족). 승준 원격 pkl 메타에서 추출.
- [ ] **G1 scene-feature 식별 metric**: SAE feature → scene 라벨 예측력(선형 probe accuracy / U(scene|feature)).
      `metrics/uncertainty.py`의 U-coefficient를 `phase` 자리에 scene 넣어 재사용 가능.
- [ ] **G2 잔차화 + succ/fail read 잔존 검증**: scene-feature 제거 후 succ/fail 분리 z. dwell cap·episode-라벨 순열 null·held-out(fit-seed 분리).
      우리 `confound-audit` skill 규약 적용. (동료 코드엔 잔차화·순열 null 없음.)
- [ ] **G3 write(steering)**: 잔여 방향을 `scripts/serve/steering_hooks.py` 훅에 배선. oracle rescue 규약(위약·noise_resample arm 동시, EVAL_SEED=100000). 전면 신규.
- [ ] **overcomplete 재설계**: 원본 D(1536)에 대해 m≫D(예: 4×~8×) + top-k. 동료 m=128은 PCA-64 대비만 과완비라 그대로 쓰면 안 됨.

### 5.C 배치 위치 + `scripts/event_sae/`와의 관계 (★ 반드시 구분)

- 우리 레포 기존 `scripts/event_sae/`는 **동료 SAE와 무관한 별개 라인**이다. 확인 결과:
  - `scripts/event_sae/README.md` = "Event-SAE Stage 3 media adapter" — **영상(MP4) 프레임**을 디코드해 5-frame temporal window
    번들을 만드는 **비전 측 파이프라인**(논문 Event-SAE). VLM annotation·clustering은 로컬 event-sae env.
  - `scripts/event_sae/stage3_media.py` = MP4→JPEG 프레임 패키징(`imageio`/`PIL`). `export_pq3_trajectories.py` = pq3 rollout pkl→
    trajectory JSON 스키마(AWE 파이프라인용). **DiT latent SAE 아님, 학습 코드 아님.**
  - **공통점은 원본 rollout 소스뿐**: 둘 다 `phase_event_pq3` 롤아웃(pq3_drawer_left 등 동일 cell, `export_pq3_trajectories.py:26-32`)을
    쓴다. 즉 **같은 데이터, 다른 모달리티**(우리 event_sae=video / 동료=DiT hidden). **중복도 파생도 아님.**
- **권고 배치**: 동료 코어를 `scripts/event_sae/`에 섞지 말 것(모달리티·목적 상이). 신규 디렉토리 예: `scripts/scene_sae/` 또는
  `src/sae/`(모델 코어) + `scripts/scene_sae/`(빌더·fit·eval). exp4-3 데이터 자산과 정렬.

### 5.D 의존성·규모 판단

- **의존성 충돌**: 코어(models/train/clustering/metrics)는 torch·numpy·sklearn만 → 우리 레포와 충돌 없음.
  **hydra-core·wandb는 코어에 불필요**(dict 설정으로 `build_model` 호출 가능, wandb는 disabled 경로). hydra는 이식하지 말 것 —
  우리는 `scripts/path_setup.py`/`cache_env.sh` 단일 소스 규약(메모리 `feedback-path-single-source`). config는 우리 방식으로.
- **원격 CPU 제약**: 승준 원격 torch는 `~/anaconda3/bin/python`, **scipy 없음**(sklearn은 있음 확인 필요). full-batch가 GPU 상주 전제라
  큰 record 수(per-token으로 T배 증가)면 CPU full-batch는 메모리 압박 → **미니배치 경로 추가**(`_loop.py:32` batch_size 인자 이미 지원) 또는 로컬 GPU.
- **규모**: SAE 자체는 초경량(선형 사전 D×m). 학습 부담은 record 수. per-token으로 N이 T배(N1.6 T=51) 커지면 로컬 GPU 권장,
  N1.5/Cosmos(토큰 少)는 승준 CPU 가능. **CPU ≤40%·OMP/OPENBLAS_NUM_THREADS≤16 cap 준수**(메모리 `cpu-budget-cap`).

---

## 6. 품질 평가

**강점**:
- 모델 코어가 **주입식·단일책임**(BaseAE는 loss만 소유, enc/dec 교체로 AE↔SAE, `autoencoder.py:159-174`). 문서화(docstring) 매우 충실.
- **누수 방지 규율**: PCA·클러스터 모두 train-only 적합(`pca.py:1`, `posthoc.py:69-74`). split은 episode 단위(step 누수 차단).
- metrics 다형 인터페이스(`Metric.compute(ctx)`) + clock 기준선 내장 → 우리 confound-audit에 바로 유용.

**위험·함정**:
- ★ **토큰 평균이 기본 빌더에 하드코딩**(`make_phase_dataset.py:102-109`) — 무심코 재사용하면 per-token 계약 위반. **빌더 재사용 금지.**
- ★ **SAE forward/loss 단위 테스트 부재**(tests는 metrics·colormap만). 이식 시 top-k 마스킹·단위노름 사전 회귀 테스트 신규 작성 필요.
- **하드코딩 경로**: `datasets_local/...` 상대경로, `input_dim=64`·`layer=12` config 고정, run_name 규약 파싱(`analysis/load.py`).
- **make_phase_dataset는 lerobot_safe env 필요**(pkl unpickle에 torch, `:33-43`) — 우리 원격 python 제약과 정합 확인.
- **cell = task×object**지 scenario_seed 아님 — scene 라벨로 직접 못 씀(seed 컬럼 별도 존재, `make_phase_dataset.py:266`).
- density()·drop_inactive만으로 dead feature 관리 → top-k SAE 표준의 resampling 없음. 큰 사전(m≫D)에서 dead feature 급증 가능, 모니터링 필요.

---

## 7. 이식 1단계 제안 (최소 검증 가능 단위)

1. `EncoderTopK`+`DecoderLinearDict`+`BaseAE`+`build_model`+`_loop.fit`를 `src/sae/`로 lift (hydra·wandb 제거, dict 설정).
2. **per-token 입력 어댑터**로 N1.5 fit30 한 셀(drawer 또는 bread)의 `[L,K,T,D=1536]`을 layer=atlas peak(L8-12) 슬라이스,
   토큰 평균 없이 record 스트림 생성. (동료 빌더 미사용, 신규.)
3. 원본 1536-d에 overcomplete SAE(m=4×~8×D, top-k) 학습(작게: 로컬 GPU 1장). smoke = 재구성 손실 수렴 + density≈k/m 확인.
4. **G1 게이트**: SAE feature→scene(layout/style/seed) 선형 probe accuracy 실측(`metrics` U-coefficient 재사용). scene 인코딩 실재 확인 후에만 G2로.

---

## 부록: 핵심 파일:라인 인덱스

- SAE 인코더: `phase/models/autoencoder.py:95-133` / 디코더: `:136-156` / 컨테이너·loss: `:159-233`
- 팩토리: `phase/models/factory.py:24-43`, `build_model:70-95`
- 학습 루프: `phase/train/_loop.py:23-89` / 엔트리: `phase/train/posthoc.py:35-115`
- 클러스터(dead feature): `phase/clustering/posthoc.py:14-42`
- ★토큰 평균: `phase/data/make_phase_dataset.py:102-117`, meta 기록 `:346-356`
- PCA-whiten: `phase/data/pca.py:15-39` / 데이터셋→캐시: `phase/data/build.py:51-87`
- metrics 레지스트리·U-coefficient: `phase/metrics/registry.py:22-49`, `base.py`(EvalContext·clock)
- SAE config: `conf/model/sae.yaml`, `conf/train/sae.yaml`, `conf/cluster/sparse.yaml`, `conf/data/pq3_l12_pca64.yaml`
- 우리 별개 라인: `scripts/event_sae/README.md`(Event-SAE video adapter), `export_pq3_trajectories.py:26-32`(동일 rollout 소스)
</content>
</invoke>
