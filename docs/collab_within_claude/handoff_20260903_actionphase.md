# Handoff 2026-09-03 — action phase 라인 (AE·cluster 생성 구조와 판정)

> **2026-09-05 갱신 — 먼저 읽을 것**: v5 도 폐기됐고 **현행 정본은 v6**(11키 · 1,650판 ·
> 번들 `analysis/grid_phase_v6/ae_k8/ae_bundle_k8.npz`). 데이터 좌표·v6 에서 달라진 계약·
> 판정·오염 cluster 목록·검증은 **§6b**, 그 라운드의 실사고는 **§6c**, 다음 라운드 절차는
> **§7**(v6 기준으로 갱신됨)에 있다. §1~§5 는 v5 시점 기술이라 좌표는 낡았지만 **구조
> 설명(§2 GT 라벨러 · §3 생성 파이프라인 · §4 판정 근거)은 그대로 유효**하다.

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

## 6b. v6 라운드 (2026-09-04~05) — 정식 번들 완성

§1~§5 는 v5 시점 기술이다. **v5 도 폐기됐고 현행 정본은 v6** 이니 아래를 먼저 읽는다.

### 6b.1 데이터·산출물 좌표 (승준 `~/datasets/temporal_vla_store/groot/n15/`)

- 아카이브 `grid/08f1c9df8207/` — 1,800판(12키 × scene 3 × **j 5** × noise 10). 인덱스
  정본 `configs/collect/n15_grid_v6_scene_jitter/index_rollouts_v6.tsv` (dev 0181cd8).
- shard `analysis/grid_phase_v6/segA_scene/<slug>__s<i>.npz` **33개**(11키 × 3 scene,
  각 50판) · 병합본 `analysis/grid_phase_v6/segA/<slug>.npz` **11개**(각 150판).
  **PPCC/apple 은 사용자 지시로 영구 제외**(그래서 12키가 아니라 11키).
- **번들 `analysis/grid_phase_v6/ae_k8/ae_bundle_k8.npz`** (centers ×11, k8, latent16,
  800 epoch·best ep780 val 425.9) + `labels_<slug>_k8.npz` ×11 + `ae_pertask_k8.json` ·
  `resid_compare_ae.tsv`. MI 오염 표 = `outputs/analysis/grid_phase_v6_frozen/contam_v6_all.json`.

### 6b.2 v6 에서 달라진 것 (v5 대비 — 코드 계약에 영향)

- **지터 좌표 열이 둘이다**: `jitter_idx`(폴더 `j<jid>` = 좌표) vs `jitter_reset_idx`
  (reset 회수 = 성분). **oven·washer 는 j 5개가 전부 reset_idx=0** 이라 reset 회수를 좌표로
  쓰면 (scene,noise)당 5판이 한 키로 뭉쳐 `좌표 중복` 으로 죽는다(60그룹 실측). 추출기는
  `jitter_idx` 우선.
- **j 의 물리적 의미가 계열마다 다르다**(plan `extra.jitter_kinds`): PPCC·coffee = 물체
  재추첨(base 0) / oven·washer = 로봇 base 오프셋(reset 0) / drawer = **둘 다(교락)**.
  → "j = 지터" 로 계열을 묶어 말할 수 없다. shard 에 성분 3열(`jitter_reset_idx`·
  `base_lat`·`base_back`)을 실어 두었으니 층화는 그걸로.
- **키 의미 개정**: oven·washer 의 left↔right 가 "로봇 스폰 쪽"→"target fixture 쪽" 으로
  바뀌어 아카이브·shard·번들 키가 **맞교환**됐다(rebase `77e745c37b0f`→`08f1c9df8207`).

### 6b.3 판정 (근거는 `ae_pertask_k8.json`·`resid_compare_ae.tsv`·`contam_v6_all.json`)

1. **margin(vs clock) 은 11키 전부 양수**(+0.175~+0.575). cluster phase 의 유효성은 v6 에서
   재확인.
2. **★ 그러나 rack 계열의 cluster 구조는 상당 부분 scene 이다.** scene 잔차화 후 margin:
   drawer-R .499→**.248** · oven-L .399→**.217** · oven-R .266→**.139** · dish-L .400→.277
   (절반 붕괴) vs drawer-L .420→**.436** · candle .575→**.560** · bread·jug·coffee 유지.
   **marshmallow 는 mi_scene 0.990 = cluster ≈ scene 라벨**(세 scene SR 0.88/0/1.0 로 딴 세계).
   → rack 계열 per-cluster 연산자는 phase 가 아니라 **scene 으로 조건화**될 수 있다.
3. **초기 창(0:10)은 11키 전부 조건화가 없다** — 그 구간 GT phase 가 상수(MI 정의 불가)이고
   cluster 도 사실상 한 덩어리다. 즉 **초반 발화 record 에서 gt arm ≡ ck8 arm**, 차이는
   중·후반에서만 생긴다. "ck8 이 gt 를 못 이겼다" 는 집계는 **발화 시점 분포부터** 봐야 한다.
4. **detector 조기 발화 = 실패 예측이 아니라 j(초기조건) 판독**(`early_record_probe.py`):
   j 5-class 정확도가 첫 record 만으로 .78~.98(chance .2)인데, 같은 창의 succ/fail AUROC 는
   **"j 별 성공률만으로 맞히기" 대조군을 못 넘는다**(drawer .472 vs .573, oven .366 vs .676).
   대조군을 넘는 건 후기 창뿐. → timer 대조만으로 조기 검출을 주장할 수 없고, **j-only
   예측기를 대조군으로** 둬야 한다.
5. **셀 내부(j 고정) 조기 신호는 PPCC 계열에만 있다** — j 잔차화 후 0:1 창 AUROC 가
   bread .698 vs drawer .331·oven .515·dish .545. j 가 고정이면 그 셀 판들은 초기 관측이
   같고 denoise seed 만 다르므로, **bread 의 .698 은 "어느 노이즈 표본이 성공할지" 를 첫
   chunk 에서 읽는다는 직접 추정치**다(rsN_llr 전제). 계열 의존이 크다는 점이 중요하다.
6. **purity 는 키 간 비교 금지** — phase 종수가 rack 5~6 / PPCC·coffee 3 이라, jug 의
   purity 0.970(MI 0.232·F1 0.036)처럼 한 phase 지배만으로 높아진다.

### 6b.4 연산자 대상에서 뺄 cluster (max-j 점유율 ≥0.40)

| 키 | 제외 cluster (n, 최빈 j 점유율) |
|---|---|
| OpenDrawer_right | c1 (305, j1 0.46) · c5 (1565, j3 0.49) |
| OvenRack_out-left | c4 (2140, j4 0.59) |
| OvenRack_out-right | c4 (249, j0 0.52) |
| PPCC_candle | c5 (2824, j3 0.41) |

나머지 7키는 제외 0. JSON `per_cluster.<c>.exclude` 로 읽는다. 창별 MI 경고는
**OpenDrawer_right 0:10** 하나뿐(MI(c;j) norm 0.183).

### 6b.5 검증 (재현 시 그대로 다시 돌릴 것)

- **온라인 왕복** — 번들로 재계산한 cluster == dump 라벨, **11키 162,874 record 불일치 0**.
  serve phase == 분석 phase 의 실데이터 보증. 합성 self-test(`cluster_phase.py`)로는
  배관만 보므로 이 검사가 별도로 필요하다.
- **파일명 == meta_json.instruction** — shard 33 + 병합본 11 전부.
- **rebase 지문** — 아카이브 1,800셀 `(instruction, scene, jitter, noise, pkl_sha256)`
  5-튜플이 교환 매핑 적용 후 1:1 일치(`snapshot_archive_fingerprints.py --compare`).

## 6c. v6 라운드 실사고 (도구 주석에도 삽입)

| 파일 | 실사고 |
|---|---|
| `rename_swap_keys.py` | 파일명만 바꾸고 **NPZ 내부 `meta_json.instruction` 미패치** → 파일명은 신 키·meta 는 구 키인 모순. 파일명으로 찾으면 맞고 meta 로 찾으면 반대쪽 물리 대상. fail detector 세션 가드가 발견(오염 산출물은 없었다) → `patch_shard_meta_keys.py` 신설 |
| `patch_shard_meta_keys.py` | ① 2GB 초과 zip 멤버는 `force_zip64=True` 없이 스트리밍 쓰기 실패(합성 소파일 테스트로는 안 잡힌다) ② 대량 I/O 경합 중 일시적 `Bad CRC-32` — 원본 무손상, 재시도로 통과하되 소진 시 실패 |
| `run_actionphase_remote.sh` | 셀 감사가 **기대치를 추출에 쓴 같은 인덱스에서** 만들어, 이관 미완(meta 50/pkl 46) 셀이 "46/46 일치" 로 통과했다 → 추출 전 `has_pkl` 대조 가드 추가 |
| (운영) | 원격 장시간 작업을 일반 bg 로 돌리면 harness 가 죽인다 — **setsid 분리 필수**(병합 1건 중단 실측, tmp→replace 라 산출물은 무손상) |

## 7. 이어받는 세션의 체크리스트

**v6 번들은 이미 있다(§6b.1)** — 아래는 다음 라운드(v7 등)에서 처음부터 만들 때의 절차다.

1. shard 추출: `run_actionphase_remote.sh` (TAG·INDEX 필수). **격자를 하드코딩하지 않고
   인덱스에서 instruction·기대 판수를 읽는다.** 완료 단위가 (instruction, scene)이면
   `INSTR_SCENES="키:scene,..."` — 산출은 `segA_scene/<slug>__s<i>.npz` 로 **분리**한다
   (instruction shard 폴더에 두면 ae_cluster 가 scene shard 를 별개 instruction 으로 잡아
   KMeans 단위가 조용히 바뀐다).
2. scene 이 다 모이면 `merge_scene_shards.py --require-scenes 3` 으로 `segA/<slug>.npz`
   승격 — **재추출 없이** 붙인다. phase 코드북은 shard 마다 독립이라 union+재매핑이 필수다
   (안 하면 에러 없이 phase 라벨이 섞인다).
3. `ae_cluster.py --mode all --dump-labels --export-bundle --epochs 800 --patience 60`
   — 번들 필수. **200 epoch 는 미수렴**(v6 실측: 200ep val 1064 → 800ep 426).
4. 번들 검증 **두 단계**: 합성 self-test(`python3 src/failure_online/cluster_phase.py`)로
   배관 확인 + **실번들 왕복**(shard feature → 번들 배정 == dump 라벨, 전 record).
   전자만으로는 실데이터 계약을 보증하지 못한다.
5. 리포트: margin(vs clock)·purity 를 instruction 별로 — pooled 단일값 금지, clock 대조
   없는 raw MI 금지. **purity 는 키 간 비교 금지**(phase 종수가 다르다, §6b.3-6).
   **scene 잔차화 전/후 margin 을 나란히** 낼 것(§6b.3-2 가 그것으로 드러났다).
6. 오염 진단 `cluster_contamination.py` — cluster×j vs cluster×GT phase MI 를 창별로,
   cluster 별 max-j 점유율 ≥0.40 이면 `exclude=true`. 연산자 세션이 이 플래그로 대상을 고른다.
7. 전패/전승 scene 이 나오면 `scene_activity_check.py` 로 "관측 死 vs 어려운 배치" 를 가른다
   (v6 marshmallow s1 사례: time_var·across_var·j_acc 전부 정상 → 배치 원인으로 확정).
8. rack 계열에 방향 의존 gating 을 붙일 거면 §4 의 push/pull 한계 먼저 볼 것.
