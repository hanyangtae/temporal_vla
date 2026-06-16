# GR00T N1.6 + TTT × RoboCasa Atomic Pretrain — End-to-End 파이프라인

> ⚠️ **보류된 연구 라인 (무기한 연기)**. 이 문서는 과거 TTT/VITA progress-predictor 방향의
> 재현 기록이며 active 작업이 아니다. 현재 메인 라인은 latent steering 이다 →
> [`../steering/`](../steering/README.md). 보존 맥락은 [ttt/README](README.md) 참고.

> **목적**: 실패-루프 탈출 문제를 TTT-based progress predictor 로 해결하는 연구. 이 문서는 Phase 0 (Eagle pre-LLM 추출) → Phase 1 (ProgressPredictor 메타학습) → Phase 2 (GR00T finetune + LHT injection) → eval 까지의 end-to-end 셋업을 재현 가능하게 정리. **이전 시도에서 부딪힌 모든 함정도 같이 기록** → 다음 실험(예: task 확장, ckpt 교체) 시 동일한 에러로 시간 낭비 X.

브랜치: `feat/groot-ttt-phase1-integration`
주요 커밋: `1cee9be`, `8b411f6`, `efac29f`, 그 이후 본 셋업 작업.

---

## 0. 아키텍처 요약

```
RoboCasa atomic 10 task           ─ Phase 0 ─►   Eagle pre-LLM cache
(per-task LeRobot v2.1)            extract        per-task embeddings.pt
                                                   {abs_idx → tensor[2048]}
                                       │
                                       ▼
                                  Phase 1 학습 (lerobot 컨테이너)
                                       │
                                       ▼
                                  ProgressPredictor ckpt
                                  (TTT + ProgressHead, meta-learned θ_0)
                                       │
                                       ▼
                                  Phase 2 finetune (groot 컨테이너)
                                  GR00T N1.6 + TTT in-place attach
                                  10-task mixture w/ episode-prefix replay
                                       │
                                       ▼
                                  RoboCasa eval (10 task)
```

**핵심 차원**: `input_dim = proj_dim = 2048` (Eagle Qwen3-1.7B hidden = DiT backbone_embedding_dim). 옛 1024 ckpt 는 사용 불가.

**TTT 디자인 spec (Phase 2)**:
- TTT meta params (P_K, V, Q, θ_0, f_adapt) **frozen** (Phase 1 에서 학습 완료).
- TTT inner-loop SSL **per-episode 으로 작동** (training/inference 양쪽).
- TTT output **LHT** (Latent History Token) → `.detach()` 후 DiT KV 에 token 추가.
- action loss 의 gradient 는 TTT 로 안 흐름. DiT + GR00T LLM 상위 4 layer 만 학습.

---

## 1. 파일 인벤토리

### 데이터 준비
| 파일 | 역할 |
|---|---|
| `scripts/utils/download_robocasa_pretrain_human.sh` | atomic-human-pretrain 10 task 다운로드 (UTexas Box). task 인자 선택적 (없으면 10 task 전체). |
| `scripts/extract/prepare_robocasa_dataset.py` | 다운받은 v2.1 데이터 in-place 수정 — `progress = frame/(len-1)` 컬럼 추가, info.json/episodes_stats/stats 갱신. idempotent. |

### Phase 0: Eagle pre-LLM 추출
| 파일 | 역할 |
|---|---|
| `scripts/extract/extract_eagle_pre_llm_robocasa.py` | per-task Eagle 추출. **LLM transformer layer skip** (`_eagle_pre_llm_only`) + frame batch + Thread pool processor parallelism. `--batch_size`, `--cpu_workers`, `--tasks` 인자. |
| `scripts/extract/extract_eagle_parallel.sh` | 10 task 를 10개 process 로 병렬 (per-task), 5-stagger 로 모델 로드 peak 완화. |
| `scripts/extract/verify_eagle_extraction.py` | 10 task embeddings.pt 검증: 파일 존재, frame 수, shape (2048,), abs_idx 연속성, NaN/Inf. exit 0 = 통과. |

### Phase 1: ProgressPredictor 메타학습
| 파일 | 역할 |
|---|---|
| `scripts/train/phase1_groot_robocasa.py` | TTT + ProgressHead 학습. functional `meta_forward` 사용 — batch 간 contamination 없음. **매 epoch 끝에 `epoch_NN.pt` (plain state_dict)** 저장 + 최종 `phase1_final.pt`. |
| `scripts/train/phase1_groot_robocasa.sh` | 위 파이썬 entry 의 default 인자. `n_epochs=10`, batch=32, lr=1e-4 (VITA Appendix D). |
| `src/datasets/phase1_v21_dataset.py` | Episodic dataset. `DEFAULT_MAX_EP_LEN=485` (RoboCasa atomic p99). progress label 은 **full episode 기준** (`t / (full_ep_length-1)`) 으로 보정 — truncate 된 episode 도 올바른 라벨. |
| `src/ttt/predictor.py` | `ProgressPredictor` (TTT + ProgressHead). `meta_forward` (functional) vs `forward(update=True)` (in-place) 구분 중요. |

### Phase 2: GR00T finetune + TTT
| 파일 | 역할 |
|---|---|
| `scripts/train/launch_finetune_ttt.py` | upstream `gr00t.experiment.run` 의 mirror entry. `--ttt_predictor_path`, `--ttt_eagle_cache_root`, `--ttt_update_in_train` 추가. `dataset_path` 가 `:` 구분이면 multi-path mixture. `video_backend="decord"` 명시. |
| `scripts/train/groot_ttt_robocasa_finetune.sh` | 10 atomic task 경로 ":" join. `MAX_STEPS=20000`, batch=64, save 1k 마다 ×20. wandb default ON. |
| `src/ttt/integrations/groot_wrapper.py` | `Gr00tN1d6WithTTT` 클래스 + `attach_ttt_to_groot()`. **in-place attach** (재인스턴스화 X, Eagle3 `_init_weights` 버그 우회). `_ttt_token_from_zseq` 가 핵심 — `meta_forward(create_graph=False)` 로 episode-prefix replay. |
| `src/ttt/integrations/launch_patch.py` | `Gr00tN1d6Pipeline._create_model` monkey-patch 로 base 모델 wrap. |
| `src/ttt/integrations/dataset_patch.py` | `ShardedSingleStepDataset` 을 subclass 로 교체 + `Gr00tN1d6DataCollator` 를 wrap. 각 sample 에 `ttt_z_seq` (Eagle cache 0..t slice) + `ttt_valid_mask` 부착. collator 는 dynamic padding (`max_T = max(t_i+1)`). |

### Eval
TBD — `scripts/eval/groot_robocasa.sh` 가 base 가 될 것. ckpt 만 finetune 결과로 가리키게.

---

## 2. 실행 순서 (재현 절차)

### 사전 준비

```bash
# 1) 컨테이너 빌드 (최초 1회, 20~40 분)
docker compose build lerobot groot

# 2) 컨테이너 띄우기
docker compose up -d lerobot groot

# 3) 환경 확인
docker exec lerobot nvidia-smi -L         # → 컨테이너 내부 GPU 0 으로 보임
docker exec lerobot env | grep WANDB_API  # WANDB_API_KEY 가 .env → override 통해 주입됨
```

`docker-compose.override.yml` 이 `device_ids: ['2']` + `CUDA_VISIBLE_DEVICES=0` + `WANDB_API_KEY=${WANDB_API_KEY}` 주입.

### Step 0: 다운로드

```bash
bash scripts/utils/download_robocasa_pretrain_human.sh           # 10 task 전체
# 또는 선택적: bash ... OpenDrawer CloseDrawer
```
출력: `<cache>/datasets/robocasa/v1.0/pretrain/atomic/<Task>/20250819/lerobot/`

### Step 1: progress 컬럼 in-place 추가

```bash
docker exec lerobot bash -lc \
  'cd /temporal_vla && python scripts/extract/prepare_robocasa_dataset.py'
```

검증: 임의 parquet 의 `progress ∈ [0, 1]`.

### Step 2: Eagle pre-LLM 추출 (10 task 병렬)

```bash
docker exec groot bash /temporal_vla/scripts/extract/extract_eagle_parallel.sh
# 각 task 별도 process, GPU 2 에 ~6.5 GB × 10 ≈ 65 GB peak
```

⚠ **GPU 메모리 OOM 위험**: 10 동시 모델 로드 시 80 GB 빠듯. stage 1 에서 일부 OOM 으로 죽으면 stage 2 으로 자동 재시도하는 watcher 스크립트 필요했음. 만약 1 GPU 가 80 GB 보다 작으면 `NUM_PROCS` 줄이고 2-stage 수동 운용.

검증:
```bash
docker exec groot python /temporal_vla/scripts/extract/verify_eagle_extraction.py
# OK: all 10 tasks verified (253,971 frames total).
```

출력: `<cache>/datasets/robocasa_eagle_pre_llm/<Task>/embeddings.pt`

### Step 3: Phase 1 학습 (TTT + ProgressHead 메타학습)

```bash
docker exec lerobot bash -lc 'bash /temporal_vla/scripts/train/phase1_groot_robocasa.sh'
```

- 300 step / 10 epoch / 17M params / ~10 분 wallclock
- 매 epoch `epoch_NN.pt` 저장 + 최종 `phase1_final.pt`
- 출력: `outputs/train/phase1_groot_robocasa/<YYYYMMDD_HHMM>/`

비교 실험: epoch_05.pt vs epoch_10.pt. **단 plateau early → epoch_08.pt 가 우리 데이터에서 val loss 최저** (0.0572).

### Step 4: 10-task merge — **SKIP**

`merge_robocasa_lerobot_v21.py` 는 단일 dataset 으로 합치는데, TTT 의 episode-prefix replay 가 task-별 Eagle cache lookup 필요 → **per-task mixture** 가 훨씬 깔끔. merge 안 함.

(필요 시: baseline GR00T finetune 만 돌릴 땐 merged dataset 도 가능. 이번 셋업에선 안 씀.)

### Step 5: Phase 2 finetune (GR00T + TTT episode-prefix)

```bash
docker exec groot bash /temporal_vla/scripts/train/groot_ttt_robocasa_finetune.sh
```

기본값:
- `TTT_PREDICTOR_PATH=outputs/train/phase1_groot_robocasa/20260511_1040/epoch_08.pt`
- `TTT_EAGLE_CACHE_ROOT=<cache>/datasets/robocasa_eagle_pre_llm`
- `TTT_UPDATE_IN_TRAIN=True`
- `DATASET_PATH=...10 atomic paths joined by :`
- `MAX_STEPS=20000`, `SAVE_STEPS=5000`, `SAVE_TOTAL_LIMIT=4`
- `USE_WANDB=1`

출력: `outputs/groot_ttt_robocasa_10tasks/checkpoint-{1000,2000,...,20000}`. wallclock ~10 시간.

다른 ckpt 로 재실행:
```bash
TTT_PREDICTOR_PATH=/path/to/other_epoch.pt \
OUTPUT_DIR=/tmp/exp2 \
docker exec -i groot bash /temporal_vla/scripts/train/groot_ttt_robocasa_finetune.sh
```

baseline (TTT 없이) — env 만 비우면 됨:
```bash
TTT_PREDICTOR_PATH= TTT_EAGLE_CACHE_ROOT= \
docker exec -i groot bash /temporal_vla/scripts/train/groot_ttt_robocasa_finetune.sh
```

### Step 6: Eval (TBD)

`scripts/eval/groot_robocasa.sh` 의 ckpt 만 finetune 결과로 바꿔 RoboCasa 10 task 평가.

---

## 3. 부딪힌 함정 & 해법 (재발 방지용)

### 3-A. Eagle 추출 너무 느림 (initial: 128 ms/frame, ~9h)

**병목 progression**:
1. **batch=1** GPU forward → batching 으로 2x (61 ms/frame)
2. **LLM 28 layer 통째 실행** 후 hidden_states[0] 만 사용 → `_eagle_pre_llm_only` 가 vision_encoder + embed 만 + image-token merge. 5-10x 절약.
3. **per-frame processor 호출이 CPU 병목** (GPU util 95% → 0% spike 81%). ThreadPool / Multiprocess 시도.
4. **단일 process 안 multiprocess (Pool/spawn)** — 첫 결과까지 4분 침묵, 디버깅 어려움. **포기하고 per-task multi-process** 로 전환.
5. **Per-task multi-process (10 procs in 10 containers / sub-procs)** → 작동. OOM 위험 있어서 모니터 + auto-retry 필요.

**교훈**: GR00T 같은 무거운 backbone 은 multiprocessing 보다 **shell-level multi-process** (per-task 1 proc) 가 더 간단. Pickle 헤매지 말 것.

### 3-B. Phase 1 의 max_ep_len bug

- `DEFAULT_MAX_EP_LEN = 120` (VITA BridgeData) → RoboCasa atomic p99=485, max=618 이라 **97.9% episode 가 잘림**.
- 더 큰 문제: `targets = arange(ep_length) / (ep_length-1)` 가 **잘린 길이 기준** → 잘린 ep 의 마지막 frame 이 잘못 `progress=1.0` 라벨링됨.
- **수정**: `DEFAULT_MAX_EP_LEN=485` + `targets = arange(ep_length) / (full_ep_length-1)` (`full_ep_length` 별도 보존).

### 3-C. WANDB_API_KEY 가 컨테이너에 안 들어감

`.env` 에 `WANDB_API_KEY=...` 있어도 base docker-compose.yml 에 env 매핑 없으면 컨테이너 안에 없음.

**수정**: `docker-compose.override.yml` 에 `- WANDB_API_KEY=${WANDB_API_KEY}` 명시. 컨테이너 recreate 후 `docker exec lerobot env | grep WANDB` 로 확인.

### 3-D. Eagle3 Siglip2 `_init_weights` 버그

`attach_ttt_to_groot` 가 `Gr00tN1d6WithTTT(config, ...)` 로 새 인스턴스 만들면 `super().__init__()` → `AutoModel.from_config(trust_remote_code=True)` → `Siglip2VisionModel.post_init()` → `_init_weights` → `NameError: name 'Siglip2Model' is not defined`.

**원인**: NVIDIA 의 캐시된 `modeling_siglip2.py` (HF cache 의 trust_remote_code 모듈) 안 `_init_weights` 가 `Siglip2Model` 참조하는데 같은 파일 안에 정의 없음. 추출 스크립트는 `from_pretrained` (init_weights skip) 라 안 터지고, finetune 의 `from_config` 만 터짐.

**수정**: `attach_ttt_to_groot` 를 **in-place attach** 로 재작성 — 새 인스턴스 만들지 말고 기존 base_model (이미 from_pretrained 로 정상 로드) 에 `predictor` 추가 + 메서드 binding (`MethodType`). 우회.

### 3-E. Video backend 부재

기본 `video_backend="torchcodec"` 가 groot 컨테이너 미설치. 자동 fallback 이 `pyav` 인데 `get_frames_by_indices` 가 pyav 미지원. `ffmpeg` 도 binary 부재.

**수정**: `pip install decord` (가벼움, 친구 스크립트도 사용) + `launch_finetune_ttt.py` 에서 `config.data.video_backend = "decord"` 명시.

### 3-F. USE_WANDB default 0

`groot_ttt_robocasa_finetune.sh` 의 `USE_WANDB="${USE_WANDB:-0}"` → wandb 로깅 안됨. **default 1 로 바꿈**.

### 3-G. Phase 1 contamination (in-place ttt_step)

Phase 1 학습 은 `meta_forward` (functional) 라 OK. 하지만 Phase 2 의 **single-frame fallback** path (`predictor(z, update=True)`) 는 `self.ttt.inner_params` 를 in-place 수정 → batch 간 누적되어 다른 episode 의 frame 으로 contamination.

**해법**: episode-prefix dataset 구현 (Step 5 정식 경로). 또는 `ttt_update_in_train=False` (단 그러면 TTT 가 static feature extractor 로 전락).

### 3-H. Checkpoint 디스크 풀

체크포인트 1 개 = ~22GB (3B model fp32 + AdamW optimizer state). 1k 주기 × `SAVE_TOTAL_LIMIT=20` 으로 돌리면 누적 440GB 필요. 호스트 `/dev/sda2` free 가 245G 였어서 학습 도중 `_save_optimizer_and_scheduler` → `PytorchStreamWriter failed writing file data/152: file write failed` 로 학습 사망 (2026-05-12 03:00, step ~9000 부근).

**수정**: `SAVE_STEPS=5000`, `SAVE_TOTAL_LIMIT=4` 로 변경. 4 × 22 = 88GB 만 사용. step_05000 ~ step_20000 4 개 ckpt 면 downstream eval 에도 충분.

---

## 4. 환경 / 의존성

### 컨테이너
- **lerobot**: Phase 0 (extract) / Phase 1 학습 / merge. Eagle 모델 불필요.
- **groot**: Phase 2 finetune / Eagle 추출. Eagle backbone 6.5 GB GPU mem.

### Python 패키지 (groot 컨테이너에 추가로 설치된 것)
- `decord` — video backend. `pip install decord`.

### Host 파일
- `.env` — `USER_NAME, USER_ID, GROUP_ID, HF_TOKEN, VNC_PW, WANDB_API_KEY`.
- `docker-compose.override.yml` — GPU 2 device_ids + CUDA_VISIBLE_DEVICES=0 + WANDB_API_KEY env.

### HF 캐시
- 컨테이너의 `~/.cache/huggingface` 가 host의 claude/user `$HOME/.cache/huggingface` bind mount source를 바라본다. 이 디렉토리 권한이 root 면 컨테이너 안 user 가 write 못함 → HF download 실패. **해결**: `docker exec lerobot sudo chown -R $(id -u):$(id -g) ~/.cache/huggingface` (lerobot 의 NOPASSWD sudo 이용).

### GR00T ckpt
- `<cache>/checkpoints/nvidia/GR00T-N1.6-3B/` — 6.2 GB, HF 에서 `huggingface-cli download nvidia/GR00T-N1.6-3B --local-dir ...`.

---

## 5. 새 task 추가하기

만약 atomic 의 다른 task (예: TurnOnMicrowave, CoffeePressButton) 도 학습에 추가하려면:

1. **다운로드**: `download_robocasa_pretrain_human.sh` 의 `ENTRIES` 배열에 `task|date|box_id` 추가 (box_id 는 atomic 공식 release 페이지에서 확인).
2. **progress 컬럼**: `prepare_robocasa_dataset.py` 가 auto-discover 라 그대로 돌리면 됨.
3. **Eagle 추출**: `extract_eagle_parallel.sh` 의 `TASKS` 배열에 새 task 추가. 또는 `--tasks` 인자로 단일 task 만.
4. **verify**: `verify_eagle_extraction.py` 의 `DEFAULT_TASKS` 에 추가.
5. **Phase 1**: `phase1_groot_robocasa.sh` 의 `--tasks` 에 추가.
6. **Phase 2**: `groot_ttt_robocasa_finetune.sh` 의 `DATASET_PATH` 에 `:` 로 새 path append.

자동 변경되는 것: max_T 같은 dataset-dependent 값은 dynamic. 메모리만 충분하면 OK.

Phase 1 ckpt 호환성: episode 길이 분포가 비슷하면 그대로 사용. 매우 다르면 (예: 1000+ frame ep) 재학습 권장.

---

## 6. 디버깅 cheatsheet

| 증상 | 원인 후보 | 첫 확인 |
|---|---|---|
| GPU OOM 학습 시작 직후 | Eagle backbone 다른 proc 가 점유 | `nvidia-smi`, friend 프로세스 kill 확인 |
| Phase 2 가 dataloader 에서 NotImplementedError | video_backend issue | `config.data.video_backend == "decord"` 확인 |
| Phase 2 launch 가 Siglip2 NameError | `attach_ttt_to_groot` 재인스턴스화 코드로 회귀 | groot_wrapper.py 의 in-place attach 확인 |
| wandb 로그 없음 | USE_WANDB=0 또는 WANDB_API_KEY 부재 | `docker exec groot env \| grep WANDB`, sh default |
| Phase 1 ckpt loadshape mismatch | proj_dim ≠ 2048 | input_dim/proj_dim 둘 다 2048 인지 |
| Phase 2 가 단일 frame 모드로 fallback | dataset 이 ttt_z_seq 안 줌 | `--ttt_eagle_cache_root` 가 sh 에서 비어있나 |
| Eagle 추출 매우 느림 | LLM forward 실행 중 | `eagle_pre_llm_forward_batched` 가 `_eagle_pre_llm_only` 호출하는지 |

---

## 7. 핵심 디자인 결정 기록 (왜 이렇게 했는지)

- **Phase 1 functional vs in-place**: `meta_forward` 는 functional → batch 마다 θ_0 부터 fresh start, 학습 차분 가능. `predictor(update=True)` 는 in-place → state 누적 (inference 시 자연스러움).
- **TTT in Phase 2 frozen + SSL only**: 사용자 spec — action gradient 가 TTT 로 안 흐르고 SSL inner-loop 만 작동. `.detach()` 유지 + `_freeze_predictor_outer()`.
- **Episode-prefix vs single-frame**: single-frame 은 batch 간 contamination 위험. Episode-prefix 가 정답이지만 dataset 측 코드 변경 필요 → `dataset_patch.py` 작성.
- **Per-task mixture vs merge**: TTT z_seq 가 task-별 cache 라 per-task 가 mapping 단순. merge 안 함.
- **in-place attach (Eagle bug 우회)**: `Gr00tN1d6WithTTT(config)` 가 from_config trigger → siglip2 NameError. base_model 에 method bind + predictor 추가만으로 동등.
- **video_backend decord**: torchcodec 미설치, pyav 미지원, ffmpeg 미설치 → decord 가 가장 가벼운 옵션.
- **per-epoch ckpt (Phase 1)**: 5ep vs 10ep 비교용으로 `epoch_NN.pt` 모두 plain state_dict 로 저장. Phase 2 가 직접 load 가능.
- **Phase 1 `max_ep_len=485`**: atomic p99. label 은 full episode 기준 (`/ full_ep_length-1`).

---

## 8. 현재 진행 상황 스냅샷

- **Phase 0**: 10 task Eagle pre-LLM 추출 완료 (253,971 frames).
- **Phase 1**: 10 epoch 학습 완료. epoch_08.pt 가 val loss 최저 (0.0572).
- **Phase 2**: 진행 중 — epoch_08.pt 기반, episode-prefix mode, 20k step. wandb `finetune-gr00t-n1d6-ttt/runs/twzzh7se`. ETA ~10 시간.
- **Eval**: TBD — groot 표준 eval pipeline 으로 10 task 평가 예정.
