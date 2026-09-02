# 핸드오프 — GR00T N1.6 HTTP full 다층 수집 (grid 규약 포함)

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
| 지터 축 k | 04 §3.1.1 | ep_meta 고정 + 연속 `reset()` = 물건·target 고정, 배치·로봇관절만 재추첨. pkl 좌표는 **평탄 `si = scene*100 + k`**, 인덱스 정본은 명시 3축 |
| 캡처 밀도 5열 | 04 §4·§6 | `capture_token_mode`·`feature_kind`·`feature_axes`·`record_shape`·`capture_layers` 를 meta.json 에 기록. **판정은 ndim 이 아니라 토큰축 크기로** |
| 계획 = 수집 대상의 전부 | 04 §5.1 | `plan.extra["adopted_cells"]` 밖 셀은 존재하지 않는 셀. 결손은 `plan.missing()` 으로만 판단 |
| 수집/평가 분리 | 04 §3.3 | 수집 rollout(pkl 有)과 평가 rollout(pkl 無)은 저장 위치를 나눈다 |
| 이관 | 04 §7.5 | 승준 노드 HDD 로만(NVMe 금지), 종류 골라 include 금지, 대조 통과분만 로컬 삭제 |

**머신 매칭 (강제)**: replay 는 **수집한 머신에서만** bit 재현된다. task 별 홈 머신을
바꾸면 그 task 의 셀-paired eval 이 깨진다. 현재 홈:

- kanu: OpenDrawer/left, PPCC/apple, PPCC/marshmallow, PPCC/bread(v2 이후), OvenRack/out
- srv48(worker1): CoffeeSetupMug, DishwasherRack/out
- srv50(worker2): OpenDrawer/right, PPCC/candle, PPCC/jug
- **데스크탑(pdk) 영구 배제** — 단일 GPU 에서 EGL 렌더×serve 동시부하 시 렌더 비결정
  (42 §7). 수집·replay 금지.

> 알려진 예외: **bread 만 base(worker1) ↔ 지터(kanu) 머신이 갈려 있다** — 그 조합의
> paired replay 는 불가. 쓰려면 base 25판을 kanu 에서 재수집할 것.

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
4. **신규 plan 은 전용 staging** 을 쓴다. 평탄 si 가 기존 plan 셀 키와 문자열 충돌해
   DONE_LIST 가 오판한다(v3 에서 20셀 스킵 발생).
5. **원격 `pkill -f` 자기-매칭** — ssh 원격 명령·harness 명령 문자열이 패턴에 걸려 자기가
   죽는다. 정지·재발사는 **스크립트 파일 + setsid** 로.
6. **이관 병목** — 실시간 이관이 수집을 못 따라가면 staging 이 폭주해 디스크가 찬다
   (3머신 274GB 폭주 → 수집 전멸 전례). 병렬 shipper + backpressure 필수.
7. **ZMQ 다층 엔드포인트 seed 버그** (2026-09-01 수정됨) — `inference_seed` 를 무시하고
   있었다. 옛 커밋으로 돌아가면 재발하니 다층 수집 전 `temporary_inference_seed` 적용
   여부를 확인할 것.

---

## 5. 현재 데이터 상태 (2026-09-01)

- 아카이브: 승준 HDD `/home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid/`
  (SSH `kimseungjun@166.104.146.37:11112`), 여유 **369GB**.
- 총 3,282 rollout ≈ 1.97TB (N1.5 규격). 이 중 **k 변주 있는 좌표 250개 = 1,250판**
  (base + k4), 나머지 1,986판은 k 없는 1상태 좌표.
- 정본 인덱스: `outputs/steer/online_pipe/manifests/index_rollouts_v4.tsv` (1,250행,
  `cell_si`·`jitter_reset_idx` 3축; base 슬롯 = `scene*100+99`).
- N1.7: **이 환경에 체크포인트 없음**(캐시엔 N1.6 만). 쓰려면 모델·robocasa 파인튜닝
  확보부터 필요.

관련 문서: `docs/05_gpu_server_rules.md`(GPU 운영·lease 정본), `docs/04_data_storage_convention.md`(저장 규약 정본), `docs/steering/45`(v2·v4 수집 결산),
`docs/collab/collect_request_v3_jitter.md`(지터 축 검증 이력), `.claude/skills/robocasa-steer-eval/SKILL.md`(eval 표준).
