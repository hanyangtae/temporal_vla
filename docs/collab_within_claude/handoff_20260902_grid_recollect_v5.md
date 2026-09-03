# 핸드오프 — grid 재수집 (시나리오 v5) + N1.6 HTTP full 수집 절차

**2026-09-02 갱신 (최우선으로 읽을 것)**: 기존 grid 데이터를 **전량 폐기**했고, 다음 세션은
아래 §0 의 **수집 계약**대로 처음부터 다시 모은다. 폐기 원장·보존물 =
`configs/collect/ledger_20260902_purge/` (README 에 시나리오·보존 목록·삭제 목록).
이전 내용(§1~§5)은 규약·절차 참고용으로 그대로 둔다.

---

## 0. 이번 수집 계약 (2026-09-02, 사용자 확정)

### 0.1 왜 다시 모으나

- **replay 시 데이터와 수집 시 데이터가 다르다**(v4r 라운드: 수집 라벨 vs replay 라벨 반전
  59%, `handoff_20260902_v4r_round.md`). 원인 미규명. 기존 activation 으로 fit 한 연산자·
  분석물은 replay 세계와 어긋나 **전부 폐기**했다. 연산자는 activation 이 있으면 재생성
  가능하므로 남길 가치가 없다.
- **재수집 시 반드시 지킬 것: 수집 경로 = 향후 replay/eval 경로.** 라벨이 갈리는 원인을
  모르는 상태이므로, 수집기 플래그(n_action_steps·max_steps·ep_meta 처리·reset 시퀀스·
  capture 설정)를 eval 러너와 **동일하게** 두고, 첫 셀에서 **fresh replay 로 수집 결과가
  bit 재현되는지(success·eef 궤적) 확인한 뒤** 본수집에 들어간다. 재현이 안 되면 멈추고
  사용자 보고.

### 0.2 시나리오 (수집의 목적)

VLA 는 작업장마다 finetune 이 필요한데, 작업장에 **약간의 변화**가 생길 때마다 finetune
하기엔 데이터·주기 부담이 크다. → 같은 작업장에서 변화로 SR 이 떨어졌을 때 finetune 보다
적은 데이터로 **activation 기반 감지 → steering** 으로 회복을 시도한다.
전제 데이터: ① finetune 에 쓰인 expert 데이터, ② 과거 같은 scene 의 rollout(물건 배치만
약간 다름), ③ 현재 scene 의 실패 rollout(구제 가능한 case 만). **unseen scene 은 대상 외.**

### 0.3 격자 (v5 = v4 구조, base 재사용 없음)

| 축 | 값 | 출처 |
|---|---|---|
| instruction | 10 (v2 와 동일) | `configs/collect/n15_grid_v2/collection_plan.json` |
| scene | 5 = v2 s0–4 (base env_seed 5개/instruction) | 위 plan 의 `instructions[instr][:5]` |
| noise (denoise seed) | 5 = v2 n0–4 (1300000–1300004) | 위 plan 의 `noise_seeds[:5]` |
| **물체 재배치 k** | **5 — 전부 신규**, base(v2 셀) 재사용 **없음** | ep_meta 고정+연속 reset (docs/04 §3.1.1) |
| 합계 | **1,250판** (instruction 당 125, scene 당 25) | |

- k 는 사전 스캔 채택분에서 **앞 5개** (drawer 계열은 left/right 재추첨 → 목표 instruction
  일치 k 만). 스캔 원본 `ledger_20260902_purge/kscan_v4/*.tsv` (N=12) 로 **50/50 scene 전부
  채택 k ≥ 5 확인됨** — 재스캔 불필요.
- plan = `scripts/collect/build_v5_plan.py` → `configs/collect/n15_grid_v5_scenario/`
  (**plan_id `e6b316053d1c`** — 3축 스키마 `instructions`(base scene seed 5개)+`jitter[instr][scene]`
  (채택 k 5개); 수집 당시 계약 plan 은 `8daefeabf020`(평탄 si 스키마)이었고 09-03 k-층 개정으로
  아카이브를 새 plan_id 아래로 재배치했다 — `extra.supersedes_plan_id`). 좌표 =
  `s<scene>/k<k>/n<noise>`, `extra.machine_assignment` 에 머신 배정 내장. 새 plan 은 **전용 staging**
  (`outputs/collect/grid_staging_v5`) 필수(§4-4 함정).
- 모델·캡처: **N1.5** (`lerobot_groot_n15__robocasa365_ckpt120000`), capture_layers
  0,2,4,8,10,12,15 · all_token_full · denoise_k 4 · n_action_steps 5 · max 720 — v4 와 동일.
  (N1.6 은 §3 절차로 가능하나 이번 계약은 N1.5.)
- 용량: 판당 ~600MB × 1,250 ≈ **750GB**. 폐기 후 HDD 여유 ≈ 1.7TB → 수용.

### 0.4 머신 배정 (= 향후 replay 홈, 고정) — **2026-09-02 사용자 재지정(v2~v4 홈과 다름)**

| 머신 | instruction | 판수 |
|---|---|---|
| srv50 (worker2) | PPCC/bread · PPCC/candle · PPCC/jug · PPCC/marshmallow | 500 |
| srv48 (worker1) | OpenDrawer/left · OpenDrawer/right · CoffeeSetupMug | 375 |
| kanu | DishwasherRack/out · OvenRack/out · PPCC/apple | 375 |

정본 = plan `extra.machine_assignment` (`configs/collect/n15_grid_v5_scenario/collection_plan.json`).
구 데이터는 전량 폐기됐으므로 홈 변경에 따른 base 재수집 문제는 없다(전 셀 신규).
데스크탑(pdk) 금지. 발사 전 `docs/05_gpu_server_rules.md` 의 `gpu_lease.sh claim`.

### 0.5 산출 계약

- 아카이브: 승준 HDD `temporal_vla_store/groot/n15/grid/<plan_id>/<machine>/<instr>/s<i>/k<r>/n<j>/base/`
  + `<plan_id>/ep_meta/<task>/<env>--seed<es>.json` 동봉.
- 인덱스 정본: `index_rollouts_v5.tsv` — 3축 열(scene_idx 0–4 · `jitter_reset_idx` k ·
  noise_idx) + env_seed(base) + success. base 행 없음(전부 k). 평탄 `cell_si` 열은 폐지.
- 완료 판정 = 아카이브 `meta.json` 수 == 1,250 (실패 셀은 재시도 2회 후 feasibility 로 기록).

### 0.5.1 첫 셀 게이트 결과 (2026-09-02, srv48 GPU2, OpenDrawer/left s1(=s0,k1) n0)

| run | 경로 | success | traj.csv(144행) |
|---|---|---|---|
| A | 수집 경로(collect_grid.sh) | 0 | 기준 |
| B | A 재실행(fresh serve+collector) | 0 | **A 와 bit 동일** |
| C | eval 경로 + ep_meta JSON 사전 주입(v4r replay 관례) | — | **실패**: k=1 상태가 'Open the right drawer.' 로 어긋남(canonical 불일치 RuntimeError) |
| D | eval 경로, ep_meta JSON 미로드(seed reset 재획득) | 0 | **A 와 bit 동일** |

→ **수집 경로 = eval 경로(D 관례) 성립. v4r 의 "수집 실패 → replay 성공 59%" 원인 =
ep_meta JSON 을 `reset(seed)` 전에 주입한 replay 관례(C)** — 지터 상태 자체가 다른 판을
돌렸던 것. 조치: collector 가 `--jitter-reset-idx`+`--ep-meta-load-env-name` 조합을 거부,
runner 주석 갱신. **v5 replay/eval 은 EP_META_DIR 없이** 돌린다(ep_meta JSON 은 기록·검증용).
게이트 산출물: srv48 `outputs/collect/v5_gate/{VERDICT_ABD.txt,A.log,C.log}`.
**kanu 게이트(DishwasherRack/out s0 k0 n0, GPU2)도 동일 패턴**: A=B=D bit 동일, C 는 첫 행부터
불일치(maxdiff 1.29; k=0 에서도 재현 → 지터 회수와 무관하게 JSON 사전 주입 자체가 상태를 바꾼다).
산출물 kanu `outputs/collect/v5_gate_kanu/VERDICT_kanu.txt`.
**srv50 게이트(PPCC/bread s0 k0 n0, GPU2, 23:00 KST)**: A=B=D bit 동일(C 는 새 collector 가드가
거부 — 설계된 실패). 산출물 srv50 `outputs/collect/v5_gate/VERDICT_ABD.txt`. **3머신 전부 통과.**

### 0.5.2 k-층 리팩터링 (2026-09-03, 사용자 지시 "개혁")

- 좌표를 **3축 폴더층** `s<i>/k<r>/n<j>/<arm>` 로 개정(docs/04 §3.1.1). plan 스키마
  `instructions`+`jitter`, legacy plan_id 불변(check_plan_schema.py 로 5종 검증).
- 코드: `src/collect/plan.py`·`artifacts.py`·collector, `collect_grid.sh`(k 열·빈칸 탭 접힘 방지)·
  `ship_to_archive.sh`·`v5_first_cell_gate.sh`, `build_grid_index.py`(3축·결손 대조)·
  `verify_grid.py`, eval 측 `replay_cells.py`(`--jitters`)·`run_online_gated_eval.sh`(`EVAL_JITTERS`,
  `--grid-instruction` 전달 버그 수정)·`collect_results.py`·`final_agg_condg.py`·
  `make_fit_manifest.py`·`select_rescue_cases.py`. `make_v5_index.py` 삭제(인덱서가 3축 직접 출력).
- 아카이브: `migrate_grid_k_layer.py` 로 1,250셀을 `e6b316053d1c/…/s<i>/k<r>/n<j>/base` 로 재배치
  (rename, pkl 불변; meta.json 에 `layout_migrated_from`), 옛 `8daefeabf020/` 은 README 만.
  인덱서 위반 0, 새 인덱스 1,250행이 구 인덱스와 sig/success/machine 전부 일치.
- 남은 미결(fit 경로 소유자 판단): `fit_cond_guidance.py` 의 `--cells-tsv`/`--v4-jitter` 는 평탄 si 와
  pkl 내부 scene_idx 를 전제 — 3축 표를 넣으면 선택 키 의미가 바뀐다. `make_triggers.py` 의
  TRIGGER 1열은 평탄값 유지(러너 조회 키도 평탄 유지).

### 0.6 착수 순서

1. `git pull` dev 최신 → v5 plan 생성 → DRY_RUN 으로 결손 1,250 확인.
2. **첫 셀 게이트** = `scripts/collect/v5_first_cell_gate.sh` (머신마다 1회, 첫 채택 셀):
   A 수집 경로 / B 같은 경로 fresh 재실행 / C eval 경로(`run_online_gated_eval.sh` 관례:
   `--no-features`·ep_meta JSON 로드) / D eval 경로에서 ep_meta JSON 로드만 제거 → 
   `compare_cell_runs.py` 가 success·traj.csv bit 대조. **A=B=C 아니면 본수집 금지·보고**
   (D 는 원인 국소화용). 결과 `<GATE_ROOT>/VERDICT.txt`.
3. (진행 상태 09-02 19:32) **srv48 GPU2 발사됨**(drawer left/right·coffee 375, 게이트 셀 1 포함,
   `outputs/collect/logs/{grid_v5_worker1,shipper_v5}.log`, STAGING_WAIT 12GB — 디스크 여유 33GB).
   **kanu GPU 2·5·6 발사됨**(10:57 KST, dish/oven rack·apple 375, 게이트 셀 1 포함, lease 3장,
   `outputs/collect/logs/{grid_v5_kanu,shipper_v5}.log`, STAGING_WAIT 20GB). kanu 는 git worktree
   `.claude/worktrees/grid-phase-sep` 의 plan 경로로 발사(메인 트리에 PR #99 미머지 시점).
   **srv50 GPU2 발사됨**(23:22 KST, junhyeong finetune 종료로 GPU0/2 비움; PPCC bread/candle/jug/
   marshmallow 500, 게이트 셀 1 포함, STAGING_WAIT 40GB, `outputs/collect/logs/{grid_v5_worker2,shipper_v5}.log`).
   **3머신 전부 가동 중.** kanu 는 이관 병목(kanu→승준 ~5MB/s)으로 backpressure 대기가 생겨
   14:10 KST(kanu 시계) 에 STAGING_WAIT 45GB·shipper PARALLEL 16 으로 재기동(진행 판 손실 0).
   **kanu 수집 완료 375/375**(15:37 kanu 시계; serve 정리·lease 2/5/6 반납, 잔여 이관 151셀은
   shipper 만 계속 — kanu→승준 링크 ~3MB/s, 승준 sshd MaxStartups 10 이라 3머신 합산 스트림
   ≤ ~22 유지: kanu PARALLEL 6). 아카이브 완료 판정은 승준 `meta.json` 수 == 1,250.
   **srv48 수집 완료 375/375**(09-03 04:5x KST; serve 정리·lease 반납, 잔여 이관 20셀).
   srv50 은 staging 상한을 150GB 로 올려 재기동(02:24 KST, 워커 전원 대기 시점) — GPU 유휴 방지.
   **srv50 수집 완료 500/500**(09-03 06:0x KST; lease 반납). **→ 1,250/1,250 수집 완료, 3머신 GPU 전부 반납.**
   ep_meta 50 파일은 승준 `8daefeabf020/ep_meta/<task>/` 에 동봉 완료. 잔여 = 이관(srv50 234셀 등)·index_v5.
   운영 메모: 총 완료 시각은 이관 속도가 정한다 — GPU util 0 은 정지가 아니라 이관 대기일 수 있다
   (`worker_w*.log` 마지막 줄 "이관 대기" 확인).
   런처 = 각 머신 `outputs/collect/logs/launch_v5_{srv,kanu}.sh`.
   lease claim → 3머신 발사(kanu GPU 3장×2 / srv 1장×6, `SERVE_MODE=host` 3종, backpressure,
   PARALLEL 8 shipper) → 완료 후 index_v5 생성·ep_meta 동봉 → 3머신 staging 정리·GPU 반납.

---

2026-09-01. 작성 = '데이터 추가 수집' 세션. branch `feat/online-gated-pipe`.
대상 독자 = 이 수집을 이어받아 돌릴 세션/사람. **읽는 순서: §1 규약 → §2 자원 규칙 →
§3 실행 절차 → §4 함정.**

---

## 1. grid 수집 규약 (좌표·저장)

정본은 **`docs/04_data_storage_convention.md`** 이고, 이 문서는 그 실행판이다.

| 규약 | 정본 | 요지 |
|---|---|---|
| 좌표 = 식별자 | 04 §3.1 | `<plan_id>/<machine>/<instruction>/s<i>/n<j>/<arm>/` — 수집 **전에** 정해진다. `sig`(내용 지문)는 식별이 아니라 무결성 검증 열 |
| 좌표 없는 수집 금지 | 04 §8 | collector 가 `grid_dir` 없으면 **RuntimeError**. `--grid-root/--plan-json/--scene-idx/--noise-idx/--grid-instruction` 5종 필수 |
| 지터 축 k | 04 §3.1.1 | ep_meta 고정 + 연속 `reset()` = 물건·target 고정, 배치·로봇관절만 재추첨. **좌표는 3축 폴더층 `s<i>/k<r>/n<j>`**(09-03 개정; 구 평탄 si 폐지), 인덱스도 3열 |
| 캡처 밀도 5열 | 04 §4·§6 | `capture_token_mode`·`feature_kind`·`feature_axes`·`record_shape`·`capture_layers` 를 meta.json 에 기록. **판정은 ndim 이 아니라 토큰축 크기로** |
| 계획 = 수집 대상의 전부 | 04 §5.1 | 3축 plan 은 `jitter[instr][scene]` 의 k 조합이 수집 대상 전부(`adopted_cells` 는 legacy). 결손은 `plan.missing()` 으로만 판단 |
| 수집/평가 분리 | 04 §3.3 | 수집 rollout(pkl 有)과 평가 rollout(pkl 無)은 저장 위치를 나눈다 |
| 이관 | 04 §7.5 | 승준 노드 HDD 로만(NVMe 금지), 종류 골라 include 금지, 대조 통과분만 로컬 삭제 |

**머신 매칭 (강제)**: replay 는 **수집한 머신에서만** bit 재현된다. task 별 홈 머신을
바꾸면 그 task 의 셀-paired eval 이 깨진다. 현재 홈:

- **v5(2026-09-02~, 현행)**: srv50 = PPCC bread·candle·jug·marshmallow / srv48 = OpenDrawer
  left·right·CoffeeSetupMug / kanu = DishwasherRack·OvenRack·PPCC apple (§0.4, plan
  `extra.machine_assignment` 정본).
- (구 v2~v4 홈, 데이터 폐기됨 — 참고만): kanu = drawer_left·apple·marshmallow·bread·oven /
  srv48 = coffee·dish / srv50 = drawer_right·candle·jug.
- **데스크탑(pdk) 영구 배제** — 단일 GPU 에서 EGL 렌더×serve 동시부하 시 렌더 비결정
  (42 §7). 수집·replay 금지.

---

## 2. 자원 점유 규칙 — 정본 `docs/05_gpu_server_rules.md`

**GPU 서버(kanu·srv48·srv50) 운영·세션 간 예약의 단일 출처는
[`docs/05_gpu_server_rules.md`](../05_gpu_server_rules.md)** (2026-09-01 신설, 커밋
53fc816). 상한 수치와 절차는 그 문서가 이기며, 여기서는 수집 발사 시 놓치기 쉬운 것만
짚는다.

- **발사 전 `scripts/utils/gpu_lease.sh claim <machine> <gpu> "<세션명>" "<용도>"` 필수.**
  이미 잡혀 있으면 exit 3 + 소유자 출력 → 발사 금지, `wait` 로 기다리거나 사용자에게 보고.
  끝나면 serve kill → `nvidia-smi` 반납 확인 → `release`.
- 상한 요지(정본 §1): 빈 GPU만(타인 프로세스 있는 GPU 금지), **kanu 최대 3장·serve 2/GPU**,
  **srv48/srv50 serve 6/GPU**.
- srv48/50 에 `junhyeong` 계정 프로세스가 이미 있으면 **이 연구의 작업인지 먼저 확인**하고
  (`ps -o user=,lstart=,cmd= -p <pid>`, 해당 세션에 문의) 합산이 상한을 넘지 않게 조율한다.
  lease 는 우리 세션 간 예약이지 타 계정 점유를 대신 판정해 주지 않는다.

## 3. N1.6 HTTP full 다층 수집 — 실행 절차

2026-09-01 이식 완료: HTTP `/act_with_features` 가 ZMQ 다층 엔드포인트와 **동일 텐서**를
낸다(실측 bit 동일). 공유 구현 = `src/policies/groot/safe/features.py`
`MultilayerFeatureExtractor`.

### 3.1 사전 확인

```bash
# 컨테이너 CUDA (NVML 상실 시 docker restart groot — 알려진 함정)
docker exec groot python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 빈 GPU 확인
```

### 3.2 serve 기동 (HTTP)

```bash
docker exec -d -e CUDA_VISIBLE_DEVICES=<빈GPU> groot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/groot.py \
     --profile configs/checkpoints/groot__robocasa365_ckpt120000.yaml \
     --port 8500 --host '*' \
     --capture-token-mode full --capture-layers 0,8,16,24,31 \
     --feature-action-horizon <실행 스텝수> \
     > /tmp/n16_http.log 2>&1 < /dev/null &"
# health 대기 (~2분): curl -s http://127.0.0.1:8500/health
```

- `--capture-token-mode`: `valid`(action 16) / `all`(50) / **`full`(시퀀스 T 전체 보존)**.
  full 이 N1.5 grid 의 `all_token_full` 대응.
- `--capture-layers`: N1.6 DiT 는 **32층**. 미지정 시 전 층(용량 6배 주의).
- `--capture-vl`: VL(goal) pathway vlln seq-mean 동시 캡처.
- 미지정 시 기존 SAFE pre-velocity 슬라이스 경로로 동작(하위 호환).

### 3.3 수집 실행

```bash
docker exec -e MUJOCO_GL=egl -e ROBOCASA_ENV_SOURCE=robocasa365 -e PYTHONPATH=/temporal_vla \
  robocasa python scripts/safe/groot_n16/robocasa/collect/collect_rollout.py \
    --policy-client-host 127.0.0.1 --policy-client-port 8500 --policy-transport http \
    --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --output-dir <staging>/_work --grid-root <staging> \
    --plan-json configs/collect/<plan>/collection_plan.json \
    --scene-idx <si> --noise-idx <ni> --grid-instruction 'OpenDrawer/left' \
    --task-description "Open the left drawer." \
    --n-episodes 1 --seed <base_env_seed> --inference-seed <noise_seed> \
    --n_action_steps 5 --max-episode-steps 720
```

- **`--n_action_steps` 는 언더스코어** (다른 인자와 표기가 다르다).
- **`--feature-action-horizon` 과 `--n_action_steps` 가 다르면 수집이 거부된다**
  (`exported_action_token_count != n_action_steps` 가드). 서버 기동 때 맞춰 둘 것.
- 지터 축을 쓰면 `--jitter-reset-idx <k>` + `--ep-meta-dir <staging>/ep_meta` 추가
  — **현재 n15 collector(`http_feature_collect.py`)에만 있고 n16 collector 에는 없다.
  N1.6 로 k-grid 를 모으려면 이 두 인자를 n16 collector 로 이식해야 한다.**

### 3.4 실측 규격 (스모크, 5-step 실행 12 record)

| 항목 | 값 |
|---|---|
| feature_kind | `groot_n16_dit_block_residual_kmean_perT_multilayer` |
| feature_axes | `[layer, token_pos, feature_dim]` |
| record_shape | `[5, 51, 1536]` (layer 5개 지정 시) |
| pkl | 9.42MB / 12 record → **720-step 환산 약 565MB/판** |
| ZMQ 대비 | feature·action **bit 동일** (maxdiff 0.0) |

> denoise 축(K)은 이 엔드포인트에서 **평균으로 접힌다**(`kmean`). N1.5 grid 는 K를
> 보존했다. K 보존이 필요하면 훅에서 `stack.mean(dim=0)` 제거 — 용량 약 4배.

### 3.5 대량 수집 (러너)

`collect_grid.sh` 는 현재 **n15 collector 전용**이다. N1.6 대량 수집은 (a) 러너에
n16 분기를 추가하거나 (b) n16 전용 러너를 두는 방식 중 택일이 필요하다. 러너가 제공하는
것(이식 시 유지할 것):

- 결손 산출(DONE_LIST + 로컬 meta.json), 워커 fan-out(GPU×serve), 순차 serve 기동,
  `STAGING_WAIT_GB` **backpressure**, `SERVE_OMP_THREADS` CPU cap, cleanup trap.
- 이관은 `ship_to_archive.sh` (`PARALLEL=8` 권장 — 링크가 스트림당 ~1MB/s 셰이핑).

---

## 4. 함정 (전부 실측으로 물린 것)

1. **groot/robocasa 컨테이너 NVML 상실** → `cuda False`. `docker restart <컨테이너>`.
   재시작 전 다른 세션이 그 컨테이너를 쓰는지 확인.
2. **`ROBOCASA_ENV_SOURCE=robocasa365` 미지정** → Isaac-GR00T 번들 robocasa(mujoco 3.2.6
   요구)가 잡혀 `AssertionError`. n16 collector 는 이 env 로 robocasa 소스를 고른다.
3. **srv 런처에 `SERVE_MODE=host`·`SERVE_PY`·`SERVE_PYTHONPATH` 누락** → srv48/50 엔
   lerobot 컨테이너가 없어 `No such container: lerobot` → serve TIMEOUT → 전멸.
4. **신규 plan 은 전용 staging** 을 쓴다. DONE_LIST 가 plan 을 구분하지 않아 다른 plan 의 같은 셀 키와 충돌하면
   DONE_LIST 가 오판한다(v3 에서 20셀 스킵 발생).
5. **원격 `pkill -f` 자기-매칭** — ssh 원격 명령·harness 명령 문자열이 패턴에 걸려 자기가
   죽는다. 정지·재발사는 **스크립트 파일 + setsid** 로.
6. **이관 병목** — 실시간 이관이 수집을 못 따라가면 staging 이 폭주해 디스크가 찬다
   (3머신 274GB 폭주 → 수집 전멸 전례). 병렬 shipper + backpressure 필수.
7. **ZMQ 다층 엔드포인트 seed 버그** (2026-09-01 수정됨) — `inference_seed` 를 무시하고
   있었다. 옛 커밋으로 돌아가면 재발하니 다층 수집 전 `temporary_inference_seed` 적용
   여부를 확인할 것.

---

## 5. 데이터 상태 (2026-09-03 v5 재수집 완료)

- **아카이브**: 승준 HDD `temporal_vla_store/groot/n15/grid/8daefeabf020/{kanu,worker1,worker2}/…`
  **1,250 셀 = meta 1,250 · pkl 1,250, 450GB**, `8daefeabf020/ep_meta/<task>/` 50 파일 동봉.
  인덱서 판정 위반 0(경로/meta 불일치 0·machine 결측 0·좌표중복 0·sig 중복 0), record_shape 전부 [7,4,49,1536].
- **인덱스 정본**: `configs/collect/n15_grid_v5_scenario/index_rollouts_v5.tsv` (1,250행, 3축
  `scene_idx`·`jitter_reset_idx`·`noise_idx` + `cell_si`; 승준 `n15/index_v5/` 에 원본 rollouts.tsv).
  로컬 사본 `outputs/steer/online_pipe/manifests/index_rollouts_v5.tsv`.
- **SR (702/1250 = 0.56)** — instruction × scene(s0..s4):

| instruction | 전체 | s0 | s1 | s2 | s3 | s4 | 홈 |
|---|---|---|---|---|---|---|---|
| CoffeeSetupMug | 0.04 | .00 | .00 | .20 | .00 | .00 | worker1(srv48) |
| DishwasherRack/out | 0.34 | .00 | .00 | .04 | .92 | .72 | kanu |
| OpenDrawer/left | 0.44 | .24 | .04 | .92 | .52 | .48 | worker1 |
| OpenDrawer/right | 0.74 | 1.0 | .96 | .84 | .00 | .92 | worker1 |
| OvenRack/out | 0.78 | 1.0 | .92 | .96 | .96 | .04 | kanu |
| PPCC/apple | **1.00** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | kanu |
| PPCC/bread | 0.73 | .48 | .52 | .88 | .76 | 1.0 | worker2(srv50) |
| PPCC/candle | 0.68 | .88 | .72 | 1.0 | .48 | .32 | worker2 |
| PPCC/jug | 0.13 | .04 | .00 | .04 | .20 | .36 | worker2 |
| PPCC/marshmallow | 0.74 | .92 | .80 | .80 | .44 | .76 | worker2 |

  주의: **PPCC/apple 은 실패 0 판**(구제 대상 없음), scene 단위로 SR 0 또는 1 인 셀 다수
  (drawer_right s3=0, oven s4=.04, dish s0/s1=0 등) — 시나리오 ③(현재 scene 실패 rollout) 은
  scene 별로 실패·성공이 공존하는 (instr, scene) 을 골라야 한다(양쪽 있는 조합: 위 표 참조).
- 3머신 staging 비움(로그·shipped_cells 만 잔존), GPU lease 0, 고아 프로세스 0.
- 승준 repo 는 `exp/grid-v5-recollect` 체크아웃 상태(PR 머지 후 dev 와 동일).

### 5.0 (구) 데이터 상태 (2026-09-02 폐기 직후)

- 아카이브 grid 는 **껍데기만** 남아 있다: 셀별 `meta.json` 3,282개 + `46ea62d53e09/ep_meta/`
  50개. pkl·mp4·csv 는 전부 삭제, `analysis/` 전부 삭제. 원장 `configs/collect/ledger_20260902_purge/`.
- 이전 상태(참고): 3,282 rollout ≈ 1.97TB, 그중 k 변주 좌표 250개(1,250판).
- N1.7: 이 환경에 체크포인트 없음.

### 5.1 (구) 현재 데이터 상태 (2026-09-01)

- 아카이브: 승준 HDD `/home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid/`
  (SSH `kimseungjun@166.104.146.37:11112`), 여유 **369GB**.
- 총 3,282 rollout ≈ 1.97TB (N1.5 규격). 이 중 **k 변주 있는 좌표 250개 = 1,250판**
  (base + k4), 나머지 1,986판은 k 없는 1상태 좌표.
- 정본 인덱스: `outputs/steer/online_pipe/manifests/index_rollouts_v4.tsv` (1,250행,
  `cell_si`·`jitter_reset_idx` 3축; base 슬롯 = `scene*100+99`).
- N1.7: **이 환경에 체크포인트 없음**(캐시엔 N1.6 만). 쓰려면 모델·robocasa 파인튜닝
  확보부터 필요.

관련 문서: `docs/05_gpu_server_rules.md`(GPU 운영·lease 정본), `docs/04_data_storage_convention.md`(저장 규약 정본), `docs/steering/45`(v2·v4 수집 결산),
`docs/collab_within_claude/collect_request_v3_jitter.md`(지터 축 검증 이력), `.claude/skills/robocasa-steer-eval/SKILL.md`(eval 표준).
