# SAFE x GR00T N1.6 RoboCasa — Parity

이 문서는 **task 성능 평가가 아니라 SAFE wiring의 action parity 검증**이다.

| 목적 | 문서 |
|---|---|
| Fine-tuned 또는 base checkpoint의 task 성능 측정 (Docker workflow) | [02 Evaluation](n16_02_eval.md) |
| SAFE feature server, HTTP `/act`, HTTP `/act_with_features`의 transport parity | **이 문서** |

ZMQ official eval과 HTTP common serving path는 모두 유지한다. GR00T N1.6 RoboCasa SR 기준선은 ZMQ official eval이다. HTTP 경로는 동일 observation에서의 action parity, per-call inference RNG, 짧은 closed-loop SAFE transport smoke까지 확인했다. 전체 benchmark HTTP SR은 아직 산출하지 않았다.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - [02 Evaluation](n16_02_eval.md)
> - [03 SAFE Overview](n16_03_safe_overview.md)
> - [04 SAFE Collection](n16_04_safe_collection.md)
> - [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)
> - [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md)
> - [07 SAFE Detector](n16_07_safe_detector.md)
> - [08 SAFE Visualization](n16_08_safe_visualization.md)
> - **09 SAFE Parity (이 문서)**
> - [10 SAFE Report](n16_10_safe_report.md)

## ZMQ Official Eval

목적: pretrained GR00T N1.6 PandaOmron baseline이 RoboCasa v0.2에서 정상 동작하는지 확인한다.

서버:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T

UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/home/dongkyu/pdk_ws/temporal_vla/data/huggingface \
HF_MODULES_CACHE=/tmp/hf_modules \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path /home/dongkyu/pdk_ws/temporal_vla/outputs/checkpoints/GR00T-N1.6-3B \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper \
  --port 5555
```

클라이언트 예시:

```bash
cd /home/dongkyu/pdk_ws/temporal_vla/src/policies/Isaac-GR00T

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
  --n_episodes 10 \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 5555 \
  --max_episode_steps 720 \
  --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n_action_steps 8 \
  --n_envs 1
```

확인된 결과:

- `OpenDrawer`, 10 episodes, `n_envs=1`, `n_action_steps=8`
- success list: `[True, True, False, False, True, True, True, True, True, True]`
- SR: `0.8`
- official README의 OpenDrawer `81.1%`와 같은 수준

이 결과를 pretrained PandaOmron baseline 정상 동작 기준선으로 둔다.

## HTTP Path And Parity

목적: 프로젝트 공통 VLA serving/eval API를 유지하고, ZMQ official 기준선과 action parity를 맞춘다.

서버:

```bash
docker compose run --rm groot \
  python /temporal_vla/scripts/serve/groot.py \
  --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml \
  --port 8500
```

이 HTTP 경로는 `groot` Docker service에서 실행한다. Isaac-GR00T eval venv는 official ZMQ/eval workflow용이며 FastAPI serve dependency를 소유하지 않는다.

API:

- `POST /act`
- `POST /act_with_features`
- `POST /reset`
- `GET /health`

HTTP serve는 같은 port에서 `/act`와 `/act_with_features`를 함께 노출한다. `/act`는 action만 반환하고, `/act_with_features`는 `features.*` namespace 아래에 DiT pre-velocity hidden state를 함께 반환한다. `--feature-slice`, `--feature-dtype`, `--feature-action-horizon` CLI flag로 export shape을 제어하며, 기본값은 `valid` slice, `float16`, full embodiment horizon이다.

전용 ZMQ `get_action_with_features` endpoint([04 SAFE Collection](n16_04_safe_collection.md#zmq-safe-feature-collection))는 upstream GR00T collector의 msgpack protocol과 호환되는 canonical SAFE rollout-collection path로 유지한다. HTTP `/act_with_features`는 benchmark-agnostic feature surface이며, `src/policies/groot/safe_features.py`를 통해 ZMQ path와 같은 DiT capture logic을 공유한다.

SAFE rollout collection이 이미 port `5557`을 사용 중이면 그 process는 그대로 둔다. HTTP `/act` parity check는 `8500`처럼 별도 port에서 실행한다.

Health check:

```bash
curl http://127.0.0.1:8500/health
python scripts/utils/smoke_test_serve.py \
  --profile configs/checkpoints/groot__robocasa_panda_omron.yaml \
  --url http://127.0.0.1:8500
```

역할:

- HTTP는 프로젝트 공통 VLA serving 경로다.
- GR00T N1.6 RoboCasa SR 기준선은 ZMQ official eval이다.
- HTTP endpoint action parity는 확인됐다.
- HTTP/ZMQ SAFE transport의 closed-loop smoke도 통과했다.
- 전체 benchmark HTTP SR은 아직 산출하지 않았다.
- 낮은 HTTP SR에서는 wiring/runner/schema mismatch를 우선 점검한다.

## Runtime Validation 2026-05-29

### Endpoint Action Parity

검증 환경:

- profile: `groot__robocasa365_ckpt120000`
- env: `robocasa_panda_omron/CloseFridge_PandaOmron_Env`
- RoboCasa env source: `robocasa365`
- scenario seed: `100000`
- inference seed: `424242`
- saved observation: `outputs/tmp/groot_http_zmq_actual_obs.pkl`
- result root: `outputs/tmp/groot_http_zmq_runtime_verify_20260529/`

동일 RoboCasa observation에서 HTTP `/act`와 ZMQ SAFE feature path action을 비교했다. ZMQ path는 `SafeN16FeaturePolicy.get_action_with_features`와 같은 action + feature extraction code path를 사용했다.

| Check | Result |
|---|---|
| HTTP `/act` repeated with same `inference_seed` | `http_repeat_max_abs_all = 0.0` |
| HTTP `/act` vs ZMQ SAFE action on same observation | `http_zmq_max_abs_all = 0.0` |
| action keys | `action.eef_pos`, `action.eef_axisangle`, `action.gripper`, `action.base_motion`, `action.control_mode` all `max_abs=0.0` |
| SAFE feature shape | `[1, 4, 16, 1024]` |
| feature kind | `groot_n16_dit_valid_action_tokens_pre_velocity` |

Artifact:

```text
outputs/tmp/groot_http_zmq_runtime_verify_20260529/action_parity_seed.json
```

HTTP server health also reports:

```json
{
  "supports_features": true,
  "supports_inference_seed": true,
  "n_action_steps": 16
}
```

따라서 transport/schema/policy RNG 관점의 endpoint action parity는 통과다.

### Closed-Loop SAFE Transport Smoke

같은 robocasa365 env, 같은 `ep_meta`, 같은 per-call `inference_seed` schedule로 ZMQ `get_action_with_features`와 HTTP `/act_with_features` collection을 각각 10 episode 실행했다.

검증 조건:

- profile: `groot__robocasa365_ckpt120000`
- env: `robocasa_panda_omron/CloseFridge_PandaOmron_Env`
- seeds: `100000..100009`
- inference seed base: `424242`
- action horizon: `n_action_steps=16`
- max episode steps: `720`
- ep_meta import root: `outputs/tmp/groot_closefridge_zmq_10x_20260529/ep_meta`
- ZMQ artifact root: `outputs/tmp/groot_closefridge_zmq_seeded_10x_20260529/rollouts`
- HTTP artifact root: `outputs/tmp/groot_closefridge_http_seeded_10x_20260529/rollouts`

결과:

| Transport | Success | Failure | Max-step episodes | Notes |
|---|---:|---:|---:|---|
| ZMQ SAFE | 8/10 | 2/10 | 2 | seeds `100000`, `100004` failed at cap |
| HTTP SAFE | 8/10 | 2/10 | 2 | ZMQ와 같은 success/failure set |

Episode-level 비교:

| Check | Result |
|---|---:|
| success/failure match | `10/10` |
| step-count match | `9/10` |
| first-action match | `9/10` |
| pkl/csv/mp4 schema | `ok` |

한 episode는 성공 여부는 같았지만 closed-loop 길이가 달랐다. Seed `100007`에서 ZMQ는 21 policy step, HTTP는 40 policy step에 성공했고, 첫 action 차이는 `max_abs=0.00390625` (`L2=0.0045017307`)였다. 따라서 이 검증은 두 transport가 같은 성공/실패 결과를 내는지 확인하는 smoke이며, trajectory identity 검증은 아니다.

두 transport 모두 pkl/csv/mp4 artifact schema를 만족했다. Hidden state shape은 `[4, 16, 1024]`, action vector dim은 `12`이고, `ep_meta`와 non-empty video/CSV도 확인됐다.

HTTP closed-loop eval replay path도 smoke 수준으로 확인했다.

| Check | Artifact |
|---|---|
| `--ep-meta-dir` first run exports manifest | `http_export/CloseFridge.json` has `ep_meta_mode=exported` |
| second run imports same manifest | `http_import/CloseFridge.json` has `ep_meta_mode=imported` |
| manifest path | `http_ep_meta/robocasa_panda_omron_CloseFridge_PandaOmron_Env--seed100000.json` |

다만 `seed + ep_meta + inference_seed`가 closed-loop action trace equality를 보장하지는 않았다. Export/import action dump 비교 결과:

```text
outputs/tmp/groot_http_zmq_runtime_verify_20260529/action_dump_compare.json
max_abs_all = 0.24609375
action.eef_pos = 0.24609375
action.eef_axisangle = 0.166015625
action.gripper = 0.0078125
action.base_motion = 0.00390625
action.control_mode = 0.0
```

원인 분리 결과:

- 같은 seed만으로 reset 두 번: checked state keys all `max_abs=0.0`
- 같은 seed + `set_ep_meta(...)` 후 reset: eef relative state differs (`eef_pos_rel max_abs=0.020762`, `eef_quat_rel max_abs=0.030109`)

즉 `ep_meta`는 scenario/layout/style/config replay artifact이고, reset 직후 full sim state/qpos까지 고정하는 artifact가 아니다. Closed-loop action trace까지 완전히 맞추려면 reset 직후 `sim.data.qpos/qvel` 또는 full MuJoCo state를 별도 artifact로 저장/복원해야 한다.

현재 HTTP와 ZMQ 비교에는 transport 이외의 차이가 함께 있다.

- HTTP 경로는 보통 project runner를 타며 `src/benchmarks/robocasa`의 robocasa365 v1.0 환경을 사용한다.
- ZMQ official 경로는 `src/policies/Isaac-GR00T/external_dependencies/robocasa`의 RoboCasa v0.2 (`robocasa_v02`) 환경을 사용한다.
- transport, env version, task class name, observation schema, action application 방식 차이가 함께 섞여 있다.

공통 task로 고른 5개는 v0.2와 v1.0 사이에서 의미적으로 대응되는 task다. `OpenSingleDoor`/`OpenCabinet`처럼 official SR이 높은 task에서 HTTP SR이 크게 낮으면 policy wiring을 우선 의심한다.

현재 해석:

- 동일 observation 기준 HTTP `/act`와 ZMQ SAFE action은 `max_abs_all=0.0`이다.
- 같은 robocasa365 env에서 HTTP/ZMQ SAFE transport의 success/failure 결과도 smoke 범위에서는 일치한다.
- 이후 SR 차이가 나타나면 action value보다 action consumption, reset timing, termination, wrapper, env version 차이를 우선 본다.
- `ep_meta`는 scenario replay artifact다. Closed-loop trace identity에는 reset 직후 full sim state/qpos replay artifact가 필요하다.

낮은 SR 진단 기준:

- image keys: `side_0`, `side_1`, `wrist_0`가 GR00T의 `res256_image_*` 계열과 동일 의미인지 확인한다.
- state keys: HTTP payload의 `eef_pos_rel`, `eef_quat_rel`, `gripper_qpos`, `base_position`, `base_rotation`이 GR00T native relative/base state key로 들어가며 batch/time dimension이 official wrapper와 같은지 확인한다.
- action keys: `end_effector_position`, `end_effector_rotation`, `gripper_close`, `base_motion`, `control_mode`가 HTTP의 `action.eef_pos`, `action.eef_axisangle`, `action.gripper`, `action.base_motion`, `action.control_mode`로 손실 없이 매핑되는지 확인한다.
- action chunk: GR00T N1.6의 chunk horizon과 RoboCasa action repeat/consume 방식이 HTTP runner와 ZMQ runner에서 같은지 확인한다.
- reset: episode 시작 시 `/reset`이 반드시 호출되고, server-side policy state가 official ZMQ path와 같은 시점에 초기화되는지 확인한다.
- env version: robocasa365 v1.0에서 task semantics가 v0.2와 충분히 같은지 task별로 확인한다.
