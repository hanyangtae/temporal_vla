# GR00T N1.6 HTTP `/act` Changes

이 문서는 GR00T N1.6 RoboCasa HTTP serving 경로를 정리하면서 무엇이 바뀌었는지 기록한다. 목적은 SAFE rollout collection 경로를 유지하면서, 프로젝트 공통 FastAPI `/act` 경로가 GR00T RoboCasa observation/action schema와 맞게 동작하도록 만드는 것이다.

> 통일 HTTP API 의 일반 계약 (`/act`, `/act_with_features`, `/reset`, `/health`, sub-key 네임스페이스, 모델 × 벤치 호환 매트릭스) 은 [`../01_serving_interface.md`](../01_serving_interface.md) 가 단일 출처. 이 문서는 그 위에서 **GR00T N1.6 RoboCasa 한정 변경 일지**다.

## Summary

| Area | Before | After |
|---|---|---|
| HTTP port | 일부 문서와 helper가 다른 VLA 기본값을 따라 `:8000` 또는 `:8200`을 암묵적으로 사용 | GR00T HTTP serve와 smoke 기본 경로를 `:8500`으로 정렬 |
| SAFE feature export | HTTP path는 `/act` only, SAFE feature는 ZMQ 전용 | HTTP serve가 `/act_with_features`도 노출 (`features.*` namespace, base64 hidden states). 같은 모델 인스턴스 공유. ZMQ `get_action_with_features`는 upstream SAFE collector msgpack 호환을 위해 유지 |
| SAFE feature 공유 모듈 | DiT hook이 ZMQ feature_server.py에 inline | `src/policies/groot/safe_features.py` 가 HTTP/ZMQ 양쪽이 공유하는 hook + horizon 해석 + base64 직렬화 helpers |
| GR00T input state | HTTP profile이 absolute/joint 계열 state를 요구 | HTTP profile이 GR00T native relative/base state alias를 요구 |
| RoboCasa cameras | res256 GR00T key 중심 | res256 key와 raw RoboCasa camera key를 둘 다 HTTP alias로 수용 |
| Language key | 일부 경로는 `annotation.human.action.task_description`만 사용 | old/new language key를 모두 채움 |
| Error handling | `/act` model-not-loaded가 HTTP 200 body error로 보일 수 있음 | `/act`/`/act_with_features` model-not-loaded는 503, invalid feature slice는 400 |
| Per-call inference RNG | HTTP payload가 seed를 전달하지 않음 | `/act`와 `/act_with_features`가 optional `inference_seed`를 받아 call-local `numpy`/`torch`/CUDA RNG를 고정하고 원래 RNG state를 복원 |
| RoboCasa replay | HTTP eval은 env seed만 받음 | `--use-groot-env`에서 `--ep-meta-dir` import/export와 `--inference-seed`를 지원 |
| Smoke execution | model별 기본 URL/GR00T dummy state가 부족 | profile base_model별 기본 URL, GR00T required state payload, repo-local log/cache path 보강 |
| Runtime safety | active SAFE collection 중 실수로 두 번째 GR00T model을 띄울 수 있음 | guarded wrapper가 active SAFE/GPU 상태에서 HTTP smoke 시작을 거절하고, smoke container를 cleanup |

## What Changed

### HTTP server

`scripts/serve/groot.py`가 GR00T HTTP `/act` + `/act_with_features` endpoint의 canonical server다.

- default port는 `8500`이다.
- profile의 `model_specific.device`를 기본값으로 사용하고, `--device`는 명시 override로만 동작한다.
- `--model-path-override`를 지원한다.
- `/act`는 model이 load되지 않았으면 `503 {"error": "model not loaded"}`를 반환한다.
- response는 profile `emits_subkeys`에 맞춰 `action.eef_pos`, `action.eef_axisangle`, `action.gripper`, `action.base_motion`, `action.control_mode`를 보존한다.
- policy output에 `base_motion`이나 `control_mode`가 없으면 profile/action fallback dimension에 맞춰 zero를 채운다.
- request payload에 `inference_seed`가 있으면 그 call 동안 `numpy`, `torch`, CUDA RNG를 해당 seed로 고정하고, call 종료 후 원래 RNG state를 복원한다.
- `/health`는 action keys, language keys, `supports_features`, `supports_inference_seed`, 그리고 현재 `feature_kind`/`feature_axes`/`feature_slice`/`feature_dtype`/`feature_action_horizon` 메타를 노출한다.

### `/act_with_features` (GR00T 특정)

일반 계약 (요청 payload, `features.*` namespace, base64 blob 형식) 은 [`../01_serving_interface.md`](../01_serving_interface.md#act_with_features) 참조. 아래는 GR00T N1.6 한정 사실:

- `features.kind` = `groot_n16_dit_valid_action_tokens_pre_velocity` (slice=valid) 또는 `..._all_action_tokens_...` (slice=all).
- `features.axes` = `["denoising_step", "valid_action_step", "feature_dim"]` (slice=valid) / `["denoising_step", "model_action_token", "feature_dim"]` (slice=all).
- RoboCasa PandaOmron 기준 `model_action_horizon = 50`, `valid_action_horizon = 16`, K denoising step 은 N1.6 sampler 설정에 따름.
- CLI 옵션: `--feature-slice {valid,all}` (default `valid`), `--feature-dtype {float16,float32}` (default `float16`), `--feature-action-horizon <int>` (default = slice 의 자연 horizon).
- DiT 캡처 로직은 `src/policies/groot/safe_features.capture_dit_features` 에 단일화돼 ZMQ feature_server 와 동일한 텐서 정의를 보장한다 (HTTP/ZMQ 둘 다 같은 hidden state 정의).

### SAFE feature 공유 모듈

`src/policies/groot/safe_features.py` 가 단일 진입점이다.

- `capture_dit_features(sim_policy, observation, ...)` — `action_head.model` 에 forward hook 을 install/remove 하면서 `sim_policy.get_action` 을 실행하고, 각 denoising step 출력을 `[B, K, H, D]` 로 stack 해 반환.
- `resolve_feature_action_horizon(...)` — `feature_slice` 와 `feature_action_horizon` 결합 검증.
- `feature_metadata(slice)` — `feature_kind` / `feature_axes` 페어 반환.
- `encode_features_to_base64(tensor, dtype)` / `decode_features_from_base64(blob)` — JSON-safe 직렬화/복원.

ZMQ `SafeN16FeaturePolicy.get_action_with_features` 와 HTTP `/act_with_features` 모두 이 함수를 호출한다.

### Schema helpers

`src/policies/groot/schema.py`가 GR00T HTTP schema boundary를 공유한다.

- `video.res256_image_side_0/1/wrist_0`와 `video.robot0_agentview_left/right/eye_in_hand`를 각각 `left/right/wrist` alias로 매핑한다.
- `annotation.human.action.task_description`과 `annotation.human.task_description`을 모두 language source로 다룬다.
- HTTP input state alias는 `eef_pos_rel`, `eef_quat_rel`, `gripper_qpos`, `base_position`, `base_rotation`을 GR00T native state key로 보낸다.

### Loader behavior

`src/policies/groot/loader.py`가 host/container path 차이를 흡수한다.

- profile에 `/temporal_vla/...` path가 들어 있어도 host checkout에 대응 path가 있으면 로컬 경로로 resolve한다.
- HF repo id와 이미 존재하는 path는 그대로 둔다.
- device resolution은 explicit CLI override가 profile default보다 우선한다.

### Eval and client

`scripts/eval/robocasa_eval.py`와 `scripts/utils/vla_client.py`가 HTTP GR00T path를 더 명확히 다룬다.

- `robocasa_eval.py --use-groot-env`는 기본 VLA server를 `http://localhost:8500`으로 둔다.
- GR00T env mode는 raw RoboCasa camera key와 new/old instruction key를 모두 처리한다.
- `--seed`는 GrootRoboCasaEnv construction (`gym.make(..., seed=seed)`)과 rollout reset (`env.reset(seed=seed + rollout_i)`)에 들어간다.
- `--ep-meta-dir`는 `(env_name, scenario_seed)` key의 `robocasa_ep_meta_manifest.v1` JSON을 import/export한다. 이 옵션은 `--use-groot-env`와 `--seed`가 필요하다.
- `--inference-seed`는 HTTP `/act` 요청마다 `inference_seed + rollout_i * num_steps + step_i`로 전달된다.
- `VLAClient.health_check()`는 request/json failure를 `None`으로 정리한다.
- `VLAClient.wait_until_ready()`는 `max_wait`를 넘겨 sleep하지 않는다.
- `/act` HTTP error body의 `error` 또는 `detail`은 `RuntimeError`로 전달한다.
- response에 action key가 없으면 받은 key 목록과 함께 실패한다.
- `VLAClient.predict(..., inference_seed=...)`와 `VLAClient.predict_with_features(..., inference_seed=...)`가 seed를 payload에 실어 보낸다.
- `VLAClient.predict_with_features(images, states, instruction, inference_seed=None)` 가 `/act_with_features` 를 호출해 `(action_dict, features_dict, latency_ms)` 를 반환한다. `features_dict["hidden_states"]` 는 서버 shape 그대로 복원된 numpy array.

### Smoke tooling

`scripts/utils/smoke_test_serve.py`와 `scripts/serve/run_groot_http_smoke.sh`가 GR00T HTTP smoke 경로다.

- smoke tool은 profile `base_model`에 따라 default URL을 고른다. GR00T는 `http://localhost:8500`이다.
- GR00T smoke payload는 required state alias를 항상 포함한다.
- guarded wrapper는 active SAFE ZMQ collection/server process 또는 큰 GPU compute process가 있으면 HTTP smoke 시작을 거절한다.
- host smoke client는 repo logger가 `/temporal_vla`에 쓸 수 없는 환경에서 standard logging으로 fallback한다.
- wrapper는 HTTP server의 HuggingFace dynamic module cache를 `/temporal_vla/data/huggingface`로 고정한다.
- wrapper가 띄운 HTTP smoke container는 smoke 후 cleanup한다.

## What Did Not Change

- SAFE rollout collection 의 기본 경로는 여전히 ZMQ `PolicyServer` endpoint `get_action_with_features` (upstream GR00T collector 의 msgpack 호환을 위해 유지).
- SAFE feature server default port는 `5557`.
- ZMQ SAFE feature payload contract (`action`, `hidden_states`, `feature_kind`, `feature_axes`, horizon metadata) 는 유지. HTTP `/act_with_features` 는 같은 metadata 를 `features.*` namespace 로 JSON-safe 직렬화해 export.
- GR00T N1.6 RoboCasa SR 기준선은 upstream/ZMQ official eval path.
- HTTP endpoint action parity와 SR 상태의 단일 출처는 [09 SAFE Parity](n16_09_safe_parity.md)다.
- Active collection을 HTTP smoke를 위해 kill하거나 재사용하지 않음.
- `ep_meta`의 replay 보장 범위 단일 출처는 [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)이다.

## Files Changed By Role

| Role | Files |
|---|---|
| HTTP serve | `scripts/serve/groot.py` |
| SAFE feature 공유 모듈 | `src/policies/groot/safe_features.py` (new), `scripts/safe/groot_n16/robocasa/serve/feature_server.py` (refactored to share) |
| Smoke/run guard | `scripts/utils/smoke_test_serve.py`, `scripts/serve/run_groot_http_smoke.sh` |
| HTTP client | `scripts/utils/vla_client.py` |
| GR00T schema/load | `src/policies/groot/schema.py`, `src/policies/groot/loader.py` |
| RoboCasa eval | `scripts/eval/robocasa_eval.py` |
| Checkpoint profiles | `configs/checkpoints/groot__robocasa365_ckpt120000.yaml`, `configs/checkpoints/groot__robocasa_panda_omron.yaml` |
| Processor guard/test cleanup | `src/processor/action/calvin.py`, `tests/test_processor.py` |
| Tests (new) | `tests/test_groot_safe_features.py`, `tests/test_serve_groot.py` (/act_with_features), `tests/test_vla_client.py` (predict_with_features) |
| Docs | `docs/groot/n16_03_safe_overview.md`, `docs/groot/n16_04_safe_collection.md`, `docs/groot/n16_09_safe_parity.md`, this document |

## Validation Status

Verified:

```bash
timeout 180 python -m pytest \
  tests/test_processor.py \
  tests/test_groot_schema.py \
  tests/test_groot_loader.py \
  tests/test_groot_safe_features.py \
  tests/test_serve_groot.py \
  tests/test_smoke_test_serve.py \
  tests/test_safe_groot_feature_server.py \
  tests/test_safe_groot_collect.py \
  tests/test_robocasa_eval_groot_http.py \
  tests/test_vla_client.py -q
```

Latest related-suite result (post `/act_with_features` 추가):

```text
117 passed, 5 skipped
```

세 실패 (`TestGrootServeMain::test_device_arg_*`, `TestSmokeTestServeRuntime::test_main_falls_back_when_repo_logger_path_is_unwritable`) 는 host 환경의 `colorlog` 미설치로 인한 pre-existing import 실패이며 이번 변경과 무관 (stash 후에도 동일 실패).

Also verified:

- `git diff --check`
- host `py_compile` for changed Python files
- `groot` Docker service import/py_compile for HTTP serve files
- `scripts/serve/run_groot_http_smoke.sh` syntax
- guarded smoke wrapper refuses to start while active SAFE collection/server is running
- `tests/test_smoke_test_serve.py` after wrapper/client cleanup changes: `7 passed`
- HTTP-focused test subset after logger fallback: `43 passed`
- HTTP seed/replay focused subset after `inference_seed` and `--ep-meta-dir`: `41 passed`

Runtime verified:

```bash
bash scripts/serve/run_groot_http_smoke.sh
```

Result:

```text
[DONE] GR00T HTTP smoke passed
```

Server log evidence:

```text
GET /health HTTP/1.1" 200 OK
POST /reset HTTP/1.1" 200 OK
POST /act HTTP/1.1" 200 OK
```

Post-smoke cleanup was also verified:

- no listener remained on port `8500`
- no `run_groot_http_smoke`, `smoke_test_serve.py`, `scripts/serve/groot.py`, or SAFE feature server process remained
- no temporary `groot-http-smoke-*` container remained
- GPU memory returned to idle baseline

Runtime action parity and replay validation are intentionally not duplicated here. The canonical result table, artifact paths, and remaining replay limitation live in [09 SAFE Parity](n16_09_safe_parity.md#runtime-validation-2026-05-29). The replay semantics live in [05 Scenario Reproduction](n16_05_safe_env_reproduction.md#보장-범위).

HTTP `/act_with_features` SAFE collection smoke was also validated on 2026-05-29:

- collector path: `scripts/safe/groot_n16/robocasa/collect/collect_rollout.py --policy-transport http`
- result summary: `outputs/tmp/groot_http_act_features_safe_collect_20260529/http_feature_collection_validation.json`
- pkl schema and metadata: `feature_kind=groot_n16_dit_valid_action_tokens_pre_velocity`, `feature_axes=["denoising_step", "valid_action_step", "feature_dim"]`
- hidden-state shape: `[4, 16, 1024]`
- SAFE loader check: `load_scope_features(...) -> [1, 1024]`

Still not covered:

- HTTP SR evaluation
- closed-loop action trace equality with full sim-state replay

## How To Run Runtime Validation

```bash
bash scripts/serve/run_groot_http_smoke.sh
```

Expected success path:

```text
/health OK
/reset OK
/act OK
[SMOKE TEST PASS]
[DONE] GR00T HTTP smoke passed
```

If the wrapper refuses to start, keep the refusal as the source of truth: another SAFE/GPU process is still active or the GPU/port state could not be verified safely.
