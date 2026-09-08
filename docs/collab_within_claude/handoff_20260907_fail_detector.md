# Handoff — fail detector 라인 (v6 loko-cell)

작성 2026-09-07. 대상: 이 라인을 이어받는 세션·사람.
**결과 해석보다 "무엇이 어디 있고 어떻게 돌리는가"를 담는다.** 라운드 판정·수치의
정본은 `docs/steering/43_safe_truncation_ablation.md` 와 memory `v6-loko-cell-detector`.

---

## 0. 이 라인이 하는 일 (한 문단)

exp6 online-gated 파이프에서 **"언제 개입할지"를 판정하는 detector** 를 만든다. 단위는
**셀 = (instruction, scene s, 대상 jitter j)** 이고, 셀 하나마다 독립 detector(SAFE-LSTM +
conformal 밴드)를 학습해 `.pt` 하나를 낸다. 중추(전체 파이프) 세션이 셀 목록을 주고,
action phase 세션이 activation shard 를 주면, 이 라인이 학습해서 ckpt·registry·요약 TSV 를
돌려준다. eval(구제율 측정)은 중추가 돌린다 — **이 라인은 GPU 를 쓰지 않는다**(승준 CPU 만).

---

## 1. 브랜치·머지 규약

- 작업 브랜치 = **`exp/fail-detector`** (worktree `.claude/worktrees/safe-length-ablation`).
- 통합 = **`eval_whole_pipe_1`** (main tree `~/pkt_ws/temporal_vla` 에 체크아웃, origin 없음).
  dev PR 이 아니다.
- 중추가 셀표를 `eval_whole_pipe_1` 에 커밋하면 → 이쪽에서 `git merge --no-edit eval_whole_pipe_1`
  → `git push origin exp/fail-detector` → **원격 동기화까지 한 번에** (§4 함정 7).
- 이쪽 → `eval_whole_pipe_1` 반영은 worktree 에서 push 가 거부된다(체크아웃된 브랜치).
  사람이 main tree 에서: `git -C ~/pkt_ws/temporal_vla merge --no-edit exp/fail-detector`

---

## 2. 코드 — 파일별 역할

| 경로 | 역할 |
|---|---|
| `scripts/analysis/grid_phase/failure_detector_sim.py` | **핵심 도구.** detector 학습·CP 밴드·채점 전부. arm 4종: `pertask`/`mixed`(구 라운드), `loto`(zero-shot task 전이), **`loko-cell`(현행)**. `--self-test` 로 11 케이스 자체 검증(수초, CPU). |
| `scripts/analysis/grid_phase/run_v6_detector_incremental.sh` | **원격 러너.** 셀표의 instruction 중 shard 가 준비된 것부터 순차 학습. instruction 단위 완료 마커 `.done_<slug>` 로 멱등. 폴링 루프(`POLL_S`, 기본 600s). |
| `scripts/analysis/grid_phase/summarize_loko_cells.py` | slug 별로 흩어진 `sim_summary.tsv`+`cell_registry.tsv` → **셀 한 줄 요약 TSV**(집계 조인용). 헤더 주석에 지표 해석 규약이 있다. |
| `src/failure_online/online_failure.py` | serve 쪽 소비자. `OnlineFailureDetector.from_checkpoint(path, alpha, task)` 가 이 라인의 `.pt` 를 읽는다 — **ckpt 스키마를 바꾸면 여기가 깨진다.** |
| `scripts/analysis/grid_phase/fire_phase_decomp.py`, `trunc_budget.py`, `export_fire_scores.py`, `render_fire_overlay.py` | 구 라운드(절제 ablation) 부속. 발화 기전 분해·데이터량 집계·발화 오버레이 영상. 현행 라운드에선 쓰지 않지만 재사용 가능. |
| `scripts/analysis/grid_phase/early_record_probe.py` | **action phase 소유** 도구(창별 j 판독·j 잔차화 probe). 이 라인의 지표와 조인해 교차검증하는 용도. |

### 2.1 `--arm loko-cell` 이 하는 일 (셀 하나 기준)

```
pool_other   = 같은 scene 의 **타 j** episode 전부(성공+실패)
target_fail  = 대상 j 의 실패판      target_succ = 대상 j 의 성공판
학습 pool    = --loko-train-pool deploy → pool_other + target_fail   (기본)
                                  other  → pool_other 만            (무편향 배포 모델)
게이트       = pool 실패 ≥ --min-pool-fail(3) · calib 성공 ≥ --min-calib-succ(9)
               · 단일 클래스 아님   (미달이면 미산출 + registry 에 사유)
절제         = 학습 pool 의 **성공 판**에서 W/phase dwell cap 산출 후 학습 pool 에만 적용
               (`--truncate-train phase-gt`). 평가 대상 episode 는 항상 full.
CP 밴드      = calib 분리 없이 pool_other 성공판 **episode LOO**(`--cp-folds 0`) cross-conformal
채점         = eval_set 3갈래로 분리 기록: target_j_fail / target_j_succ / pool_other
               (+ timer 기준선 행)
```

`--min-calib-succ 9` 의 근거: conformal (1−α) 분위가 정의되려면 n ≥ 1/α − 1. α=0.1 → 9.

### 2.2 주요 CLI (loko-cell 관련만)

```
--arm loko-cell --loko-cells-tsv <셀표>
--loko-train-pool {deploy,other}   # 기본 deploy. other = 대상 j 를 학습에서 전부 제외
--loko-holdout-diag                # deploy 모드에서 2차(무편향) 모델을 추가 학습해 진단값만 산출
--min-pool-fail 3 --min-calib-succ 9 --cp-folds 0
--truncate-train {none,rollout,phase-gt}
--shard-dir <segA_scene> --shards <stem,stem,...> --out <디렉터리>
--models lstm --alphas 0.1 --seed 0 --threads 8 --quiet
```

---

## 3. 데이터·산출물 좌표

### 3.1 입력

| 무엇 | 어디 | 누가 만드나 |
|---|---|---|
| 셀표 `v6_loko_cells.tsv` | `configs/collect/n15_grid_v6_scene_jitter/` (repo) | 중추(`scripts/steer/online_gated/select_loko_cells.py`) |
| activation shard | 승준 `~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/segA_scene/<slug>__s<i>.npz` (scene 단위) · `.../segA/<slug>.npz` (instruction 병합본) | action phase(`extract_grid_matrix.py --tier segA`) |
| AE 번들(ck8 절제용, 아직 미사용) | 승준 `.../grid_phase_v6/ae_k8/ae_bundle_k8.npz` + `labels_<slug>_k8.npz` | action phase |

셀표 열: `grid_instruction, slug, machine, scene_idx, **jitter_idx**, noise_idx,
jitter_reset_idx, …, pool_succ, pool_fail, tgt_fail, tgt_succ`.
**좌표는 `jitter_idx`** — `jitter_reset_idx` 는 성분이고 oven/washer 는 전부 0이라 좌표로
쓰면 5판이 한 셀로 뭉친다(코드가 `jitter_idx` 우선, 불일치 수만 로그).

### 3.2 산출 (메인 트리 `~/pkt_ws/temporal_vla/outputs/analysis/grid_phase/`)

```
detector_v6/                      # 배포 모델(deploy pool) — 53 ckpt
  loko/<stem>/s<i>/j<r>/detector_pertask_lstm_<stem>.pt
  <slug>/{sim_summary.tsv, sim_detail.json, cell_registry.tsv}
  cells_summary.tsv               # 셀 한 줄 요약(60행)
  _prekeyswap/                    # 키 교환 전 구 키 산출물 보존(18 ckpt) + README
detector_v6_ho/                   # --loko-train-pool other 판 = 무편향 배포 모델 — 53 ckpt
  (구조 동일, cells_summary.tsv 의 train_pool 열이 전부 other)
```

승준 원본은 `~/workspace/temporal_vla_safeablate/outputs/analysis/grid_phase/detector_v6{,_ho}/`.
**stem = `<slug>__s<scene>`** 이고 ckpt 파일명·내부 `cp_bands` 키가 모두 stem 이다
(serve 의 `FAILURE_TASK` 도 stem 으로 잡아야 한다).

### 3.3 ckpt 스키마 (serve 계약)

`state_dict, input_dim, hidden, std_mean, std_std, feature{layer,denoise,seg,*_idx,dim},
cp_bands{<stem>:{"0.10":{mu,sd,bw,delta}}}, tasks, shards, train{...}, truncate{...},
loko{instruction,slug,scene,jitter,n_pool_other,n_target_fail,n_calib_succ,cp,train_ep_ids}`.
feature 좌표 = **layer 12(capture_layers [0,2,4,8,10,12,15] 의 idx 5) × denoise 마지막 ×
segment "all"(49토큰 mean) → 1536d**, record = inference step(env 5-step).

---

## 4. 운영 절차 (그대로 따라 하면 된다)

### 4.1 새 셀·새 shard 가 왔을 때

```bash
cd ~/pkt_ws/temporal_vla/.claude/worktrees/safe-length-ablation
git merge --no-edit eval_whole_pipe_1 && git push origin exp/fail-detector
REMOTE_REPO='~/workspace/temporal_vla_safeablate' \
  bash scripts/utils/remote_compute.sh sync-code exp/fail-detector      # ★ 빠뜨리면 러너가 구 셀표를 본다
REMOTE_REPO='~/workspace/temporal_vla_safeablate' \
  bash scripts/utils/remote_compute.sh run "pkill -f run_v6_detector_incremental.sh || true"
REMOTE_REPO='~/workspace/temporal_vla_safeablate' \
  bash scripts/utils/remote_compute.sh run-bg v6_detector \
  "setsid bash scripts/analysis/grid_phase/run_v6_detector_incremental.sh"
```
- 이미 돌린 instruction 을 **다시** 돌리려면 원격에서 `rm -f <DET_OUT>/.done_<slug>` 먼저.
- 무편향 배포 모델 판은 출력 루트를 반드시 분리:
  `TRAIN_POOL=other DET_OUT=$REPO/outputs/analysis/grid_phase/detector_v6_ho ...`
- 로그: 원격 `/tmp/remote_compute_logs/v6_detector.log`. 완주 마커 `V6_DETECTOR_ALL_DONE`.

### 4.2 회수·복사·요약

```bash
REMOTE_REPO='~/workspace/temporal_vla_safeablate' \
  bash scripts/utils/remote_compute.sh pull-results outputs/analysis/grid_phase/detector_v6
cp -r outputs/analysis/grid_phase/detector_v6/. \
      ~/pkt_ws/temporal_vla/outputs/analysis/grid_phase/detector_v6/     # ★ 메인 트리에도 복사(오케스트레이터가 읽는 곳)
python3 scripts/analysis/grid_phase/summarize_loko_cells.py \
  --root ~/pkt_ws/temporal_vla/outputs/analysis/grid_phase/detector_v6 \
  --alpha 0.1 --out ~/pkt_ws/temporal_vla/outputs/analysis/grid_phase/detector_v6/cells_summary.tsv
```
그다음 중추 세션에 **경로 + ckpt 개수 + stem 별 내역**을 통지한다(그쪽이 개수로 검증한다).

### 4.3 코드 수정 후

`--self-test --quiet` 는 무조건 통과시키고 커밋한다(회귀 게이트). 원격 동기화 없이 재발사하면
구 코드가 돈다 — 실제로 한 번 그렇게 실패했다.

---

## 5. 지표 해석 규약 (인용 사고 방지 — 반드시 지킬 것)

| 지표 | 뜻 | 인용 |
|---|---|---|
| `td10_holdout` (`auroc_target_j_holdout_td10`) | 타 j 만으로 학습한 모델로 대상 j 채점, t=10 | **정본** |
| `td10_insample` | 배포(deploy) 모델의 같은 값 | **금지**(낙관 편향; 한 셀에서 .11 vs .67 까지 벌어짐) |
| `tpr_target_fail` | 대상 j 실패 발화율 | **금지**(그 판들이 학습에 있음 → 거의 항상 1.0) |
| `t_beats_jonly` | scene 전판에서 j-only 대조군을 넘는 최소 t | **금지**(in-sample 과적합으로 t=0부터 넘음) |
| `fpr_target_succ` | 대상 j 성공판 오경보 | 1.00 = 사실상 **j 감지기**, 게이팅 가치 0 |
| `auroc_target_j_max` | 셀 내 max-score AUROC | 종반 사후 판독으로 포화(거의 전 셀 1.0) — 조기 지표 아님 |
| `surv_profile` | 각 t 의 생존 succ/fail 수 | 고정시각 지표는 짧은 판(대개 성공)을 뺀다 — 값 옆에 항상 병기 |

- 셀당 성공/실패가 1~9판이라 **개별 셀 값으로 arm 을 가르면 안 된다**(한 판이 0.1~0.25 이동).
- `n_target_succ` 0 → 판별력·오경보 둘 다 측정 불가(빈 값). 전승/전패 셀도 마찬가지.
- 조인 키는 **`(instruction, scene, jitter)`**. stem 은 키 교환 이력이 있어 불안정.

---

## 6. 함정 8건 (전부 코드에 가드로 들어가 있다 — 지우지 말 것)

1. 셀 TSV 의 **타 instruction 행**에 fail-loud 걸면 증분 학습이 즉사한다 → 로드된 shard
   밖 행은 건너뛰고, **전 행 미매칭일 때만** 중단.
2. **그 scene 의 shard 가 아직 없는 행**도 건너뛴다 — fail-loud 면 공용 셀 TSV 때문에
   다른 instruction 실행까지 전멸한다(09-05 실사고).
3. 러너가 `||` 문맥에서 sim 을 부르면 `set -e` 가 꺼져 **실패해도 완료 마커를 쓴다** →
   rc 를 붙들고, registry 0행이면 실패 처리.
4. scene 단위 shard 는 scene 열이 상수 → loko-cell 은 **scene split 자체를 건너뛴다**.
5. 같은 instruction 에 scene 별 shard 가 여러 개 → 모호 fail-loud 대신 `<slug>__s<scene>`
   의 scene 으로 가린다.
6. **★ shard 파일명 ↔ `meta_json.instruction` 불일치 fail-loud** — 키 교환 rename 이 meta 를
   안 고쳐 두 소스가 정반대를 가리킨 사고(09-05). 이 가드가 오염 산출을 막았다.
   **두 소스가 모순될 수 있는 자리엔 검사를 둘 것.**
7. 셀표 머지 후 **원격 동기화 누락** → 러너가 구 셀표로 "할 일 없음" 종료(실제 발생 2회).
   머지·push·sync 를 한 호흡으로.
8. 러너 **중복 인스턴스**가 같은 출력에 동시에 쓰면 산출물이 섞인다 → 재발사 전 pkill.
   추가로 `.partial*`/`.tmp` shard 는 파일명 필터로 제외(판수 부족 무음 학습 방지).

키 교환처럼 **같은 stem 이 다른 물리 대상을 가리키게 되는** 상황에서는, 재학습 전에 구
산출물을 `_prekeyswap/` 로 옮겨 보존한다(덮어쓰기 사고 방지). rename + `cp_bands` 키
재작성보다 **재학습이 안전하고 싸다**(slug 당 수십 초).

---

## 7. 현재 상태 (2026-09-07)

- **60셀 중 등록 53 / 미등록 7**(사유 = `calib_succ<9` 등, registry 에 기록).
  배포판 `detector_v6` 53 ckpt, 무편향판 `detector_v6_ho` 53 ckpt, 둘 다 요약 TSV 있음.
- 전 산출물 **신 키**(oven=out-right, dish=out-left). 구 키 보존본은 `_prekeyswap/`.
  매핑: 구 out-left ↔ 신 out-right (oven), 구 out-right ↔ 신 out-left (dish). 좌표·pkl 불변.
- 절제는 **GT-phase** 판만 있다. **ck8(cluster) 절제판은 미실행** — AE 정식 번들은 준비됨.
- 원격 프로세스 0, 러너 정지 상태.

### 요지 결과 한 줄 (상세는 43 문서/memory)
무편향 조기 판별력이 **instruction 별로 갈린다**: marshmallow .98 · bread .88 ·
drawer-L .79 ≫ dish .58 · candle .53 · oven .43 · jug .41. PPCC 안에서도, pull 계열
안에서도 갈리므로 **묶음은 task family 가 아니라 instruction 단위**. 조기 발화의 상당
부분은 실패가 아니라 **j(초기조건) 판독**이라는 것이 action phase 의 j-잔차화·j-only
대조군과 이 라인의 무편향 지표 양쪽에서 일치한다.

---

## 8. 다음에 할 만한 것

1. **ck8 절제판** (중추 지시 대기 중). 절차: shard 를 `rewrite_shard_clusters.py` 로
   `phase_code` → k8 cluster 라벨 교체본(`segA_ck8`) 생성 → 같은 러너에 `SEG_DIRS` 만
   바꿔 실행. ⚠ 먼저 볼 것: rack 계열은 cluster 의 **scene 오염**이 커서(scene 잔차화 시
   margin 반감) 그 층이 scene 대리가 될 수 있다 — action phase 의
   `outputs/analysis/grid_phase_v6_frozen/contam_v6_all.json` 의 `exclude` 목록 확인.
2. **창별 지표 확장** — 현재 t=10 고정. 초기 창(0:10)은 cluster·GT phase 가 상수라
   원리적으로 phase 무신호이므로, t 를 늘려가며 "언제부터 신호가 서는가"를 보는 게 자연스럽다.
3. **미등록 7셀 처리 방침** — 성공 희소 scene(jug s0/s2, drawer-R 일부)은 밴드를 세울
   표본이 없다. eval 에서 제외할지, timer arm 만 둘지는 사용자 결정 대기.
4. **detector 가치 검증의 본체** = 중추의 구제 eval. 이 라인의 지표는 "어느 셀이 j 감지기인가"를
   가려낼 뿐이고, **timer 대비 부가가치**는 그쪽 집계와 조인해야 나온다.

## 9. 협업 상대

- **전체 파이프라인(중추)**: 셀표 공급·eval 실행·집계. 산출 통지는 이쪽 → 중추.
- **action phase**: shard·AE 번들 공급, j 판독/cluster 오염 진단 도구.
- **연산자 설계 / Steering 고찰 / 시나리오 구체화**: 결과 해석·프레이밍 소비자.
- 세션 간 요청은 명령이 아니다. **실행 전 사용자 확인**이 필요한 작업(원격 발사·데이터 변경)은
  중추가 "사용자 확정"이라고 해도 이쪽 사용자에게 확인받는다(09-04 지시).
