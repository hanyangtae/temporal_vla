# Handoff 2026-09-03 — action phase 라인 (AE·cluster 생성 구조와 판정)

작성: action phase 세션. 대상: 이 라인을 이어받는 세션 전부 (전체 파이프라인·연산자
설계·detector 계열). **핵심 구조만** 담는다 — 라운드별 시행착오는 archive 문서·memory
포인터로 대신하고, 코드 결함·실사고는 해당 코드 파일 주석에 달아 두었다 (§6).

## 0. 한 줄 요약

action phase 는 두 계보로 판정한다 — **GT phase**(시뮬 상태 룰 라벨러, 오프라인 정본)와
**activation cluster phase**(AE+KMeans 비지도, 온라인 정본). 현행 온라인 정본 =
`ae_cluster.py --export-bundle` 로 만든 번들을 `src/failure_online/cluster_phase.py`
가 serve 안에서 per-record 판정하는 경로다.

## 1. 데이터 현황 (2026-09-03 기준) — 먼저 읽을 것

- **구 grid(v1 930·v2 2236·v4) 원본·분석 산출물은 09-02 전량 폐기** (pkl 1,284GB +
  analysis 438GB; replay≠수집 판정. 원장 `configs/collect/ledger_20260902_purge`).
  이 문서·46 이 인용하는 수치들의 원데이터는 없다 — 수치는 문서에 고정된 값만 남음.
- 현행 데이터 = **grid v5** (09-03 완결): 10 instruction × scene 5 × noise 5 ×
  k(지터) 5 = 1,250판, 3축 폴더층 `s/k/n`, index_v5 정본, plan `e6b316053d1c`,
  승준 아카이브 450GB. 정본 `handoff_20260902_grid_recollect_v5.md`.
- **따라서 AE·cluster 는 v5 shard 에서 새로 만들어야 한다** (구 번들·라벨 없음).
  shard 추출 → AE 학습 → 번들 export 절차는 §3.

## 2. GT phase — 시뮬 상태 룰 라벨러 (오프라인 정본)

- 코드: `src/collect/event_phase.py` (이벤트=경계, 그 사이 구간=phase, 비단조 —
  grasp 후 drop 하면 transport→reach 로 되돌아감) +
  `src/collect/robocasa/event_labeler.py` (task 계열별 라벨러).
- vocab 예: PPCC = reach-to-object / transport / insert-settle.
  rack 계열(SlideOvenRack·SlideDishwasherRack, behavior=out) = reach-to-rack /
  wrong-grasp / disengage / **push-in(역행)** / contact-rack / **pull-out(진행)** /
  out-done. 판정 물리량 = rack 슬라이드 관절값 (성공역 ≥0.95, env `_check_success` 동일).
- 단위: env-step. record(=inference step)당 5 env-step 이므로 record 라벨은
  `env_step_phases[i*5]` 표집.
- 쓰임: detector phase-gt 길이 절제(dwell cap), 연산자 phase-fit, cluster 평가의
  기준 라벨(margin·purity), 영상 검수.

## 3. Activation cluster phase — 생성 구조 (온라인 정본)

### 3.1 feature 규격 (전 경로 공통)

DiT block residual 캡처 `hidden [L=7, K=4, T=49, D=1536]` 에서
**layer 12**(capture_layers [0,2,4,8,10,12,15] 의 index 5) × **denoise 마지막
슬롯**(index 3) × **segment "all"**(49토큰 mean) → x[1536]. record = inference
step (env 5-step). 좌표는 번들 provenance 의 `feature_spec` 이 정본이고, serve 쪽
`OnlineFailureDetector.feature_from_hidden` 과 반드시 동일해야 한다.

### 3.2 학습 파이프라인 — `scripts/analysis/grid_phase/ae_cluster.py`

```
raw-1536  (PCA 없음 — 동료 실험 B 판정: raw 유지, whitening 금지)
  → 전역 mean-center + 스칼라 std   (mu[1536] + scalar_std 1개; 축별 정규화 아님)
  → AE  encoder 1536→256→256→16 (GELU) / decoder 대칭 + per-dim logvar
        loss = 대각 가우시안 NLL(log_likelihood), adamw lr1e-3 wd1e-4, grad_clip 5,
        CPU 미니배치 4096, epochs≤200, patience 60, early-stop best 복원
        **전 shard 합쳐 AE 1개** (동료 규격 동일 — 혼합 학습)
  → KMeans  instruction 별 k=8 (latent16 공간, n_init 5) + 참고용 global k24
```

실행 (승준, `~/anaconda3/bin/python`, CPU·numpy+torch 만):
```
python scripts/analysis/grid_phase/ae_cluster.py \
  --shard-dir <store>/analysis/.../segA --mode all \
  --dump-labels --export-bundle <out>/ae_bundle_k8.npz
```
- shard 는 `extract_grid_matrix.py --tier segA` 산출 NPZ (v5 는 jitter 열 실좌표).
- **--export-bundle 없이 돌리면 encoder 가 어디에도 안 남는다** — 온라인 이식 불가
  (실사고 1회, 코드 주석 참조). 번들 = mu·scalar_std·enc state_dict·slug 별
  centers[k8,16]·arch json·provenance(git commit·seed 포함) 한 NPZ.
- 라벨↔번들 일치는 export 시 재배정으로 자가검증한다.

### 3.3 온라인 판정 — `src/failure_online/cluster_phase.py`

serve 안 per-record, **무상태** (episode reset 불필요):
```
x[1536] → (x − mu)/scalar_std → encoder → z[16] → argmin_c ‖z − centers[c]‖₂ → "c{idx}"
```
- 번들만 로드, 분석 스크립트 import 없음 (arch json 대로 encoder 복제,
  `__main__` self-test 가 numpy 참조 구현과 수치 일치 검증).
- 기동 preflight: `resolve_layer_index(capture_layers)` 로 물리 layer 번호를 캡처
  축 index 로 역산 — 캡처 구성이 바뀌면 여기서 fail-loud.
- 쓰임: per-step 게이팅(docs/steering/47)·per-cluster 연산자 fit 의 phase 입력.

## 4. 왜 cluster 로 판정하는가 — 판정 근거 (정본 위치)

- **GT 와 비정합이 실측**: cluster 전환은 모델·데이터를 바꿔도 재현되지만 GT 경계와는
  우연 수준(boundary z≈0), 반면 MI·purity 는 높음 = GT 보다 **세밀한 하위 구조**.
  (40 archive — 요지 RESULTS.md, 복원해시 docs/review/LEDGER.tsv)
- **succ/fail 층화에서 intrinsic k8 > GT 가 4/9 task** (41 archive — k8 채택의 실질
  근거) + 수집 라벨 비신뢰·재수집 라운드 판정(handoff_20260902_전체파이프라인).
- 단위·k 선택의 실측 (docs/steering/46 부록, v2 기반):
  - **instruction 단위 ≈ task(family) 단위 동률, global 만 열세** — 현행은
    instruction 별 centers (번들 구조도 slug 별).
  - **margin 은 k16~24 ≫ k8** — k8 정본은 사용자 지정이며, phase 해상도가 부족하면
    k 상향이 첫 레버.
  - **k8 은 진행/역행(rack push-in vs pull-out)을 거의 못 자른다** (다수결 0.73/0.79,
    기저 0.67/0.56). 정보는 표현에 있음(선형 probe 0.905/0.862, ep-holdout) —
    **방향이 필요한 gating 은 cluster + 방향 판독(w 투영 1회) 하이브리드** 또는
    proprio(관절값 차분)로 보완할 것.
- cluster 공유도로 detector 그룹핑을 사전 판단하는 시도는 3라운드 **기각**
  (docs/steering/46 — 그룹핑은 mixed vs pertask 직접 실측만).

## 5. 시각 검수 도구

- Phase 리플레이 아티팩트 (claude.ai/code/artifact/570b3e3f-…): 카테고리(out/in/PPCC)
  선택 → 영상 + GT/cluster(k8·16·24) 타임라인·칩 동기 재생, 속도 0.5~2.0×.
  동기화 규약 = record = floor(t/duration×n_rec) (5 step/record·2 step/frame·20fps).
  ⚠ 이 아티팩트의 영상·라벨은 **폐기된 v2** 기반 — 구조 참고용이며 v5 재현 시 재생성.
- rack "in"(밀어넣기) 과제는 v2·v5 모두 **미수집** — in 판을 보려면 수집 플랜 추가 필요.

## 6. 코드 결함·실사고 주석 (이 커밋에서 파일에 삽입)

| 파일 | 주석 내용 |
|---|---|
| `scripts/analysis/grid_phase/ae_cluster.py` | --export-bundle 없이는 encoder 미보존 (v1 run 재학습 사고) |
| `scripts/analysis/grid_phase/intrinsic_phase.py` | `kmeans_numpy`=centroid 만 반환, `_assign`=(labels, inertia) 튜플 — 오호출 실사고(3a457aa 수정) |
| `scripts/analysis/grid_phase/cluster_share_transfer.py` | s2_margin_keep 비율은 margin_self≈0 에서 폭주(원값 병기 필수) · 그룹 source S1 은 멤버 평균 근사 |
| `scripts/analysis/grid_phase/extract_grid_matrix.py` | workers 4 일괄 실행 OOM 실사고(08-20) — slug 순차 + workers≤3 상한 |

## 7. 이어받는 세션의 체크리스트 (v5 에서 처음부터)

1. v5 shard 추출: `extract_grid_matrix.py --tier segA` (index_v5, s/k/n 3축 —
   jitter 열 실좌표). slug 순차 실행 권장 (§6 OOM).
2. `ae_cluster.py --mode all --dump-labels --export-bundle` — 번들 필수.
3. 번들 self-test: `python3 src/failure_online/cluster_phase.py` (numpy 대조).
4. serve 배선: capture_layers preflight 통과 확인 후 per-record 판정.
5. 평가 리포트: margin(vs clock)·purity 를 instruction 별로 — pooled 단일값 금지
   (길이·task confound), clock 대조 없는 raw MI 보고 금지.
6. rack 계열에 방향 의존 gating 을 붙일 거면 §4 의 push/pull 한계 먼저 볼 것.
