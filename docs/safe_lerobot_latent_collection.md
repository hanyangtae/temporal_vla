# SAFE latent 수집 — lerobot 모델 × 멀티 벤치 확장 (plan + status + handoff)

> 목적: 기존 GR00T N1.6 SAFE latent 수집(`scripts/safe/groot_n16/`)을 **lerobot 가 제공하는
> 정책(pi0.5, pi0+FAST, X-VLA, GR00T N1.5)** 으로 확장하여, 여러 벤치마크에서 SAFE latent
> 데이터를 대량 수집한다. (detector 학습/분석은 별도 단계.)
>
> 현재 작업 브랜치: `feat/safe-lerobot-latent-collect` (base `dev`).

---

## 1. 데이터 한 점의 의미 (가장 중요)

SAFE feature(arXiv 2506.09937) = **마지막 레이어에서 action 이 velocity field(flow-matching)
또는 token logit(autoregressive)으로 디코딩되기 "직전"의 hidden state**, 추론 1회당 1개 벡터로 집계.
라벨은 **episode-level success/failure (0/1)** 뿐 (per-step 라벨 없음 — SAFE 원논문도 동일).

- **데이터 한 점 = policy 추론(action chunk) 1회의 SAFE latent.**
  lerobot 정책은 내부 action queue 가 빌 때(= `n_action_steps` env step 마다)만 새 추론을 돌린다.
  그 사이 step 의 `select_action` 은 버퍼된 action 만 pop → latent 없음. 따라서 latent 는
  **추론당 1개**(env step 당이 아님)이며, rollout 충실도는 일반 배포와 동일하게 유지된다.
- flow-matching(pi0.5/X-VLA/GR00T N1.5): `[K_denoise, H_action, D]`
- pi0+FAST(autoregressive): `[1, n_tokens, D]` (denoising 축 없음; downstream 3D `[K,H,D]`
  계약 유지를 위해 singleton K 로 감쌈)

### 모델별 hook 지점 (lerobot v0.5.1 소스에서 검증)

| 모델 | policy_type | hook 대상 module | hook | per-step shape | feature_axes |
|---|---|---|---|---|---|
| pi0.5 | `pi05` | `policy.model.action_out_proj` | forward_pre(input[0]) | `[K,H,D]` | denoising_step, action_step, feature_dim |
| pi0 | `pi0` | `policy.model.action_out_proj` | forward_pre | `[K,H,D]` | 〃 |
| X-VLA | `xvla` | `policy.model.transformer.action_decoder` | forward_pre | `[K,H,D]` | 〃 |
| GR00T N1.5 | `groot` | `policy._groot_model.action_head.model`(DiT), 출력 `[:, -H:]` | forward(output) | `[K,H,D]` | 〃 |
| pi0+FAST | `pi0_fast` | `policy.model.paligemma_with_expert.paligemma.lm_head` | forward_pre | `[1,n_tokens,D]` | token_singleton, action_token, feature_dim |

구현: `scripts/serve/safe_hooks.py` (`run_with_features(policy, batch, policy_type)`).

---

## 2. 아키텍처

프로젝트 핵심 패턴(모델↔벤치 분리 + HTTP 통일 API)을 그대로 사용하되, **모델 서버는 host conda
환경**에서, **벤치마크는 각자의 컨테이너(client)** 에서 돈다.

```
[host conda: lerobot_safe]                         [bench (libero/robocasa/vlabench)]
  scripts/serve/lerobot.py (FastAPI, GPU)            scripts/eval/*.py (VLAClient)
  forward hook → SAFE latent                          env.reset/step 루프
  POST /act_with_features  ◀──────── HTTP ─────────  매 step 호출, 추론 발화 step 만 latent
     {action subkeys, has_feature, hidden_states_b64}  → collect_common 으로 per-episode pkl
```

- 모델 서버 한 곳(`scripts/serve/lerobot.py`)이 `--profile` 로 4개 정책을 교체.
- hidden_states 직렬화: float16 ndarray → `np.save` bytes → base64 (JSON list 대비 경량).
- 수집 pkl 스키마는 groot_n16 과 동일 → 기존 split/aggregation 재사용.

---

## 3. 타깃 매트릭스 (공개 체크포인트 기준)

수집 가능한 칸 = 공개 체크포인트가 있는 칸. (없는 칸은 별도 학습 필요 → 현재 범위 밖.)

| 모델 | LIBERO (repo 통합됨) | VLABench (미통합) | RoboCasa |
|---|---|---|---|
| pi0.5 | `lerobot/pi05_libero_finetuned` ✅ | `VLABench/pi05-primitive-10task` | — |
| pi0+FAST | (pi0fast-libero ckpt 확인 필요) | `VLABench/pi0-fast-primitive` | — |
| X-VLA | `2toINF/X-VLA-Libero` / `lerobot/xvla-libero` | `2toINF/X-VLA-VLABench` | — |
| GR00T N1.5 | `Tacoin/GR00T-N1.5-3B-LIBERO-LONG` | — | `robocasa/robocasa365_checkpoints` (gr00t_n1-5/multitask_learning/checkpoint-120000) |

- **LIBERO**: repo 에 이미 통합됨(`scripts/eval/libero.py`, `src/processor/{obs,action}/libero.py`,
  `make_libero_processors`). 마찰 최소 → 1차 타깃.
- **VLABench**: repo 미통합. env clone + processor + eval 스크립트 + 프로파일 신규 필요.
- **RoboCasa × GR00T N1.5**: robocasa365 체크포인트가 **Isaac-GR00T 포맷**(safetensors shards +
  config.json + experiment_cfg/). lerobot `GrootPolicy.from_pretrained` 로 직접 로드 불가능성이
  높음 → **기존 Isaac ZMQ 경로(`scripts/safe/groot_n16/.../serve/feature_server.py` +
  `src/policies/groot/loader.py`) 재사용(Fallback A)** 가 유력. 별도 트랙.

---

## 4. 현재까지 완료 (committed)

브랜치 `feat/safe-lerobot-latent-collect`, 커밋 3개:
- `chore: lerobot submodule v0.5.1 bump (py3.12/torch2.7/transformers5.3.0)`
- `feat: SAFE latent 추출 — lerobot serve hook + /act_with_features + client`
- `feat: SAFE 수집 공통 writer + pi0.5 LIBERO 체크포인트 프로파일`

| 영역 | 산출물 | 상태 |
|---|---|---|
| lerobot bump | submodule cbc8bfb → **v0.5.1**(1396b9f) | ✅ 5개 정책 import 확인 |
| SAFE hook | `scripts/serve/safe_hooks.py` | ✅ 작성·import OK |
| serve 엔드포인트 | `scripts/serve/lerobot.py`: `/act_with_features` + host/컨테이너 경로 remap | ✅ pi05 serve 로드·`/health` OK |
| client | `scripts/utils/vla_client.py`: `predict_with_features` | ✅ 직렬화 라운드트립 OK |
| 수집 writer | `scripts/safe/lerobot/collect_common.py` | ✅ 작성·compile OK |
| 프로파일 | `configs/checkpoints/lerobot_pi05__libero.yaml` | ✅ 검증 OK (action_dim 7) |

검증 수준: **host 정적/로드 검증까지 완료** (compile, import, profile load, serve `/health`,
직렬화 round-trip). `/act_with_features` 의 **실제 추론 1회 end-to-end 호출(hook 발화·shape
실측)은 아직 미실행** — target GPU 에서 §6 smoke 로 마무리 필요.

---

## 5. 환경 재현 (host conda)

lerobot v0.5.1 은 **Python ≥ 3.12, torch ≥ 2.7, transformers 5.3.0** 요구(기존 Docker 의 py3.10 +
transformers fork dance 는 불필요해짐). groot extra 의 `flash-attn` 은 host 빌드가 비현실적이라
**제외**하고 groot 의 순수 deps 만 설치 → groot 는 런타임에 eager attention 으로 fallback.

```bash
# 0) lerobot submodule 을 커밋된 포인터(v0.5.1)로
git submodule update --init lerobot

# 1) py3.12 env
conda create -n lerobot_safe python=3.12 -y

# 2) lerobot + 정책 extras (flash-attn 유발하는 groot extra 는 제외)
conda run -n lerobot_safe pip install -e './lerobot[pi,xvla]'

# 3) groot 런타임 deps (flash-attn 제외)
conda run -n lerobot_safe pip install 'peft>=0.18,<1' 'dm-tree>=0.1.8,<1' 'timm>=1.0,<1.1' 'decord>=0.6,<1'

# 4) serve deps
conda run -n lerobot_safe pip install fastapi uvicorn colorlog
```

검증된 버전: torch 2.10.0+cu128, transformers 5.3.0, lerobot 0.5.1, py3.12.

### HF gated 모델 토큰
pi0.5/pi0+FAST 는 tokenizer 로 **gated repo `google/paligemma-3b-pt-224`** 를 받는다 → `.env` 의
`HF_TOKEN`(paligemma 라이선스 동의된 계정) 필요. serve 기동 시 환경변수로 전달(§6).

---

## 6. 실행 / 검증 (runbook)

### 모델 서버 기동 (pi0.5 × LIBERO 예시)
```bash
cd /home/dongkyu/pkt_ws/temporal_vla
source <miniconda>/etc/profile.d/conda.sh
set -a; . ./.env; set +a   # HF_TOKEN 로드 (gated paligemma)
PYTHONPATH=scripts/utils HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  conda run --no-capture-output -n lerobot_safe python scripts/serve/lerobot.py \
  --profile configs/checkpoints/lerobot_pi05__libero.yaml --device cuda --port 8411
# /health 가 status:ok, n_action_steps, action_keys 반환하면 정상.
```
- pi0.5 libero 의 visual key 는 3개(image, image2, empty_camera_0) → 통일키 static/wrist/wrist2 매핑.

### `/act_with_features` smoke (hook 발화·shape 실측 — **다음에 할 것**)
합성 obs(랜덤 이미지 static/wrist/wrist2 + state + task)로 `predict_with_features` 호출:
- 첫 호출(queue 빔) → `has_feature=True`, `hidden_states.shape == [K, H, D]`(pi05: K=num_inference_steps,
  H=chunk_size, D=action-expert width), NaN 없음.
- 이후 `n_action_steps-1` step → `has_feature=False`(queue pop).

### 수집 (LIBERO, `--safe-collect` — **미구현, §7-2**)
`scripts/eval/libero.py` 에 `--safe-collect` 플래그를 추가하면, 매 step `predict_with_features` 로
전환하고 추론 발화 step 의 latent 를 `collect_common.SafeEpisodeCollector` 로 누적, episode 종료 시
pkl 작성. 출력 규약: `outputs/eval/{benchmark}/{model}/rollouts_{run_id}/{task}/task{id}--ep{idx}--succ{0|1}.pkl`.

---

## 7. 앞으로 할 일 (TODO)

1. **pi0.5 × LIBERO end-to-end smoke 마무리** → §6 smoke 로 hook 발화·shape 실측.
   verify: `hidden_states` 3D `[K,H,D]`, K==num_inference_steps, H==chunk_size, NaN 없음.
2. **`--safe-collect` collector 통합** (`scripts/eval/libero.py`, 이어서 `robocasa_eval.py`).
   - `_predict` → `predict_with_features` 분기, `collect_common.SafeEpisodeCollector` 로 누적,
     episode 종료 시 pkl/csv 작성. `--safe-output-dir`, `--safe-run-id` 인자 추가.
   - verify: 1-episode 수집 → pkl 1개(`task..--ep..--succ?.pkl`), 각 `hidden_states[i]` 3D,
     `episode_success∈{0,1}`.
3. **나머지 LIBERO 프로파일 작성·검증**: `xvla__libero`, `groot_n15__libero_long`,
   `pi0fast__libero`(ckpt 확인). 각 `python scripts/utils/checkpoint_profile.py <yaml>` 로드 검증 +
   serve `/act_with_features` 1회로 hook 발화 확인 (특히 pi0_fast 는 `[1,n_tokens,D]`).
4. **VLABench 통합 (Phase 5)**: `VLABench/VLABench` clone/install → obs/action 스키마 확인 →
   `src/processor/obs/vlabench.py` + `action/vlabench.py` + `factory.make_vlabench_processors` +
   `scripts/eval/vlabench.py`(libero.py 미러, `--safe-collect`) + 프로파일 3개
   (`pi05__vlabench`, `pi0fast__vlabench`, `xvla__vlabench`).
5. **RoboCasa × GR00T N1.5 (Fallback A)**: robocasa365 Isaac 체크포인트를 lerobot `GrootPolicy`
   로 로드 시도 → 실패 시 기존 Isaac ZMQ feature_server 경로 재사용. RoboCasa-365 task/embodiment
   (PandaOmron)이 repo `GrootRoboCasaEnv` 와 정합하는지 확인.
6. **Phase 6 — 수집물 검증**: 신규 범용 split helper(`scripts/safe/lerobot/split/`,
   groot_n16 splitter 의 task-map 인자화 버전) → `safe_feature_vectors.pooled_hidden_states`
   가 `[T,D]` 반환(ndim!=3 에러 없음), pi0_fast singleton-K pooling no-op 확인.

---

## 8. 알려진 함정 / 주의

- **lerobot v0.5.1 = py3.12 강제.** 기존 Docker(py3.10) 이미지는 v0.5.1 소스를 못 돌린다. 모델 서버는
  host conda(`lerobot_safe`) 로 운영. 벤치 컨테이너는 영향 없음.
- **flash-attn 제외**: groot extra 가 요구하지만 host 빌드 비현실적. groot 의 순수 deps 만 설치하고
  eager attention fallback (lerobot 이 자체 처리). 추론 정확도 영향 없음.
- **gated paligemma**: pi0.5/pi0+FAST 는 `HF_TOKEN` 없으면 401 로 startup 실패. `.env` 토큰 사용.
- **프로파일 경로 remap**: 프로파일의 `checkpoint_source.id` 가 컨테이너 절대경로(`/temporal_vla/...`)
  면 serve 가 repo root 기준으로 remap (host/컨테이너 양쪽 동작). HF repo id 는 그대로 사용.
- **데이터 의미 = 추론당 latent** (env step 당 아님). collector 는 `has_feature=True` step 에만 record.
- 수집 pkl 스키마는 groot_n16 과 동일 유지 — 기존 split/train/vis 재사용을 위해 깨지 말 것.
