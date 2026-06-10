# SAFE latent 수집 — lerobot 모델 × 멀티 벤치 확장 (plan + status + handoff)

> 목적: 기존 GR00T N1.6 SAFE latent 수집(`scripts/safe/groot_n16/`)을 **lerobot 가 제공하는
> 정책(pi0.5, pi0+FAST, X-VLA, GR00T N1.5)** 으로 확장하여, 여러 벤치마크에서 SAFE latent
> 데이터를 대량 수집한다. (detector 학습/분석은 별도 단계.)
>
> 원래 작업 브랜치: `feat/safe-lerobot-latent-collect` (base `dev`).

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
     {action subkeys, has_feature, features.hidden_states, hidden_states_b64}
        → VLAClient decode → collect_common 으로 per-episode pkl
```

- 모델 서버 한 곳(`scripts/serve/lerobot.py`)이 `--profile` 로 4개 정책을 교체.
- hidden_states 직렬화: 표준 응답은 `features.hidden_states` feature blob(`data`+`shape`+`dtype`).
  기존 collector 호환을 위해 legacy `hidden_states_b64`(`base64(np.save(...))`)도 함께 낸다.
- 수집 pkl 스키마는 groot_n16 과 동일 → 기존 split/aggregation 재사용.

---

## 2.5 핵심 운영 제약 (pi05×LIBERO 디버깅으로 확정)

수집 serve 를 처음 세울 때 반드시 알아야 하는 두 가지. 둘 다 실측으로 확정했다.

### (1) 이미지 180° 회전 — serve 가 flip 수행
`pi05_libero_finetuned` 는 **180° 회전된 이미지로 학습**됐다(lerobot `LiberoProcessorStep`
이 `torch.flip(img, dims=[2,3])` 적용). serve 가 raw 이미지를 그대로 보내면 **SR 0%** 가 된다.
→ 프로파일에 `image_preprocess.rotate_180: true` 를 두고, `parse_payload` 가 입력 이미지에
`torch.flip(t, dims=[1,2])` 를 적용한다. 이 수정 후 SR **0% → 100%**(libero_object 5/5,
libero_10 3/3) 로 회복. rotate 여부는 체크포인트가 어떤 orientation 으로 학습됐는지에 달렸으니
**프로파일별로 명시**한다.

### (2) torch.compile × SAFE hook — "첫 compile 시점에 hook 이 있어야 발화"
**`pi05_libero` 체크포인트 config 는 `compile_model=True`**(새 `PI05Config()` 기본값 False 와
무관 — config.json 이 override). serve 는 `from_pretrained` 로 로드 시 compiled 로 뜬다.

핵심 메커니즘(실측):
- `compile_model=True` 면 `sample_actions`(SAFE hook 대상 `action_out_proj` 를 내부 호출)가
  통째로 `torch.compile(max-autotune)` 된다.
- forward hook 은 **`sample_actions` 가 "처음" compile 될 때 등록돼 있어야** compiled
  그래프에 포함되어 발화한다. hook 이 그 시점에 없으면 Dynamo 가 **hook 없는 그래프를 캐시**하고,
  이후 hook 을 걸어도 그 캐시를 재사용 → **hook 영영 무시(features=None)**.
- `/act`(hook 없는 추론)와 `/act_with_features`(hook 있는 추론)가 같은 policy 를 공유하므로,
  **`/act` 가 먼저 돌면** hook 없는 그래프가 캐시돼 이후 수집이 전부 features=None 이 된다.

| 시나리오 | hook 이 첫 compile 에 존재 | hook 발화 |
|---|---|---|
| compiled, `/act` 선행 → `/act_with_features` | ❌ | ❌ (features=None) |
| compiled, `/act_with_features` 만 | ✅ | ✅ |
| eager (`TORCHDYNAMO_DISABLE=1`) | — (compile 안 함) | ✅ |

**해결 = serve `--collect` 모드.** 수집 시 serve 를 `--collect` 로 띄우면 **`/act` 를 409 로
거부**하여, 첫 추론이 반드시 `/act_with_features`(hook 있음) 이도록 강제한다. 그러면 **compile 을
끄지 않고도**(=추론 속도 유지) SAFE hook 이 정상 발화한다. eager 강제는 불필요.
(검증: cold compiled serve + `--collect` → `/act` 409, `/act_with_features` 발화 2/12,
shape `[10,50,1024]`.)

> 정리: **추론/eval serve = 평소대로(compiled, flip).
> SAFE 수집 serve = `--collect` 추가**(compiled 유지, `/act` 차단). 둘 다 flip 필요.

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
  높음. 현재 정리 기준에서는 native ZMQ 평가는
  `scripts/safe/groot_n15/robocasa/eval/native_zmq_eval.py` 가 외부 N1.5
  `inference_service.py` 에 말 거는 client 역할만 맡는다. N1.6
  `feature_server.py`/`src/policies/groot/core/loader.py` 재사용이 아니며,
  RoboCasa N1.5 전용 SAFE feature capture pipeline 은 별도 미지원이다.

---

## 4. 당시 완료 상태 (committed)

원래 브랜치 `feat/safe-lerobot-latent-collect`, 커밋 5개:
- `chore: lerobot submodule v0.5.1 bump (py3.12/torch2.7/transformers5.3.0)`
- `feat: SAFE latent 추출 — lerobot serve hook + /act_with_features + client`
- `feat: SAFE 수집 공통 writer + pi0.5 LIBERO 체크포인트 프로파일`
- `docs: SAFE latent 수집 lerobot 확장 — plan/현황/handoff 문서`
- `feat: pi05 LIBERO SAFE 수집 end-to-end — 180° flip + compile/hook --collect 가드` ← 본 작업

| 영역 | 산출물 | 상태 |
|---|---|---|
| lerobot bump | submodule cbc8bfb → **v0.5.1**(1396b9f) | ✅ 5개 정책 import 확인 |
| SAFE hook | `scripts/serve/safe_hooks.py` | ✅ 작성·import OK |
| serve 엔드포인트 | `scripts/serve/lerobot.py`: `/act_with_features` + host/컨테이너 경로 remap | ✅ pi05 serve 로드·`/health` OK |
| client | `scripts/utils/vla_client.py`: `predict_with_features` | ✅ 직렬화 라운드트립 OK |
| 수집 writer | `scripts/safe/lerobot/collect_common.py` | ✅ 작성·compile OK |
| 프로파일 | `configs/checkpoints/lerobot_pi05__libero.yaml` | ✅ 검증 OK (rotate_180:true, action_dim 7) |
| `--safe-collect` collector | `scripts/eval/libero.py` | ✅ pkl 생성·hidden_states 적재 확인 |
| serve `--collect` 모드 | `scripts/serve/lerobot.py` (/act 409 가드) | ✅ §2.5 메커니즘 검증 |

검증 수준: **pi05 × LIBERO end-to-end 완료.**
- 추론 SR: libero_object 5/5, libero_10 3/3 (= **100%**, lerobot-eval 재현치와 일치).
- `/act_with_features`: hook 발화 step 에서 `hidden_states (10,50,1024)`,
  `feature_kind=pi05_action_expert_pre_velocity`, NaN 없음.
- `--safe-collect`: episode pkl 에 `hidden_states` + `episode_success∈{0,1}` 적재 확인.
- 두 핵심 제약(§2.5: 180° flip, compile/hook 순서)은 위에 별도 정리.

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

### `/act_with_features` smoke (hook 발화·shape 실측 — ✅ 완료)
`predict_with_features` 를 reset 후 12회 호출:
- 추론 발화 step(queue 빔, 매 `n_action_steps`) → `has_feature=True`,
  `hidden_states.shape == [K, H, D]`(pi05: `[10, 50, 1024]`), NaN 없음.
- 그 외 step → `has_feature=False`(queue pop).

### SAFE 수집 (LIBERO) — serve 는 반드시 `--collect`
수집 serve 는 `--collect` 로 띄운다(§2.5: compile 유지하며 `/act` 차단 → SAFE hook 보장):
```bash
# 모델 서버 (수집 전용)
... python scripts/serve/lerobot.py \
  --profile configs/checkpoints/lerobot_pi05__libero.yaml --device cuda --port 8411 --collect
# /health 의 collect_mode:true 확인.
```
collector 는 `scripts/eval/libero.py --safe-collect` — 매 step `predict_with_features` 로
전환하고 추론 발화 step 의 latent 를 `collect_common.SafeEpisodeCollector` 로 누적, episode 종료 시
pkl 작성. 출력 규약: `outputs/eval/{benchmark}/{model}/rollouts_{run_id}/{task}/task{id}--ep{idx}--succ{0|1}.pkl`.
```bash
# 벤치(client, libero_bench env). serve 가 host 경로면 PYTHONPATH 로 LIBERO 패키지 지정.
PYTHONPATH="$REPO/src/benchmarks/LIBERO:$REPO/src:$REPO/scripts/utils:$REPO/src/policies/openvla-oft" \
  conda run -n libero_bench python scripts/eval/libero.py \
  --task-suite libero_object --server-url http://localhost:8411 \
  --num-trials 5 --safe-collect --safe-output-dir outputs/eval/libero/pi05 --safe-run-id run0
```

---

## 7. 앞으로 할 일 (TODO)

1. ✅ **pi0.5 × LIBERO end-to-end 완료** (SR 100%, hook 발화·shape 실측, §2.5 제약 확정).
2. ✅ **`--safe-collect` collector 통합** (`scripts/eval/libero.py`). `robocasa_eval.py` 는
   동일 패턴으로 이어서. (수집 serve 는 `--collect` 필수 — §2.5.)
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
- **이미지 180° 회전(§2.5-1)**: 체크포인트가 회전 이미지로 학습됐으면 프로파일 `rotate_180:true`.
  안 맞추면 SR 0%. 모델 추가 시 학습 orientation 확인 필수.
- **수집 serve 는 `--collect`(§2.5-2)**: compile_model=True 정책에서 `/act` 선행 시 SAFE hook 이
  영구 무시(features=None). `--collect` 가 `/act` 를 막아 첫 compile 에 hook 포함 보장. compile 은 유지.
- **데이터 의미 = 추론당 latent** (env step 당 아님). collector 는 `has_feature=True` step 에만 record.
- 수집 pkl 스키마는 groot_n16 과 동일 유지 — 기존 split/train/vis 재사용을 위해 깨지 말 것.
