# SAFE x GR00T N1.6 RoboCasa — Parity

이 문서는 **task 성능 평가가 아니라 SAFE wiring의 action parity 검증**이다.

| 목적 | 문서 |
|---|---|
| Fine-tuned 또는 base checkpoint의 task 성능 측정 (Docker workflow) | [02 Evaluation](n16_02_eval.md) |
| (1) SAFE feature server가 공식 RoboCasa eval과 같은 action을 내는지 (2) HTTP `/act`가 ZMQ와 parity 맞는지 | **이 문서** |

ZMQ official baseline과 HTTP common serving path 모두 유지한다. SR 기준선은 ZMQ로 둔다. HTTP는 같은 observation에서의 action parity와 per-call inference RNG 정렬은 확인됐지만, closed-loop SR 지표는 아직 별도 평가가 필요하다.

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

## HTTP Path And SR Recovery

목적: 프로젝트 공통 VLA serving/eval API를 유지하고, ZMQ official 기준선과 action parity를 맞춘다.

서버:

```bash
docker compose run --rm groot \
  python /temporal_vla/scripts/serve/groot.py \
  --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml \
  --port 8500
```

Use the `groot` Docker service for this HTTP path. The Isaac-GR00T eval venv is
for the official ZMQ/eval workflow and does not own the FastAPI serve
dependencies.

API:

- `POST /act`
- `POST /act_with_features`
- `POST /reset`
- `GET /health`

HTTP serve exposes both `/act` (action only) and `/act_with_features` (action +
DiT pre-velocity hidden states under the `features.*` namespace). Both
endpoints share a single loaded model on the same port. `--feature-slice`,
`--feature-dtype`, `--feature-action-horizon` CLI flags control the export
shape — defaults are `valid` slice, `float16`, full embodiment horizon.

The dedicated ZMQ `get_action_with_features` endpoint
([04 SAFE Collection](n16_04_safe_collection.md#zmq-safe-feature-collection))
is still the canonical SAFE rollout-collection path because the upstream
GR00T collector talks msgpack over ZMQ. HTTP `/act_with_features` is the
benchmark-agnostic feature surface for new consumers (e.g. Calvin SAFE), and
shares the same DiT capture logic via `src/policies/groot/safe_features.py`.

If a SAFE rollout collection is already using port 5557, keep that process
running. Start HTTP `/act` parity checks on a separate port such as 8500.

Health check:

```bash
curl http://127.0.0.1:8500/health
python scripts/utils/smoke_test_serve.py \
  --profile configs/checkpoints/groot__robocasa_panda_omron.yaml \
  --url http://127.0.0.1:8500
```

One-command smoke wrapper:

```bash
bash scripts/serve/run_groot_http_smoke.sh
```

The wrapper refuses to start while the SAFE ZMQ collection/server path is
active, starts only its own HTTP server process, and stops that process after
the smoke test.

역할:

- HTTP는 heterogeneous VLA serving 경로다.
- GR00T N1.6 RoboCasa SR 기준선은 ZMQ official eval이다.
- HTTP endpoint action parity는 확인됐다. HTTP SR은 별도 closed-loop 평가 후 SAFE/GR00T N1.6 성능 지표에 편입한다.
- 낮은 HTTP SR에서는 wiring/runner/schema mismatch를 우선 점검한다.

## Runtime Validation 2026-05-29

검증 환경:

- profile: `groot__robocasa365_ckpt120000`
- env: `robocasa_panda_omron/CloseFridge_PandaOmron_Env`
- RoboCasa env source: `robocasa365`
- scenario seed: `100000`
- inference seed: `424242`
- saved observation: `outputs/tmp/groot_http_zmq_actual_obs.pkl`
- result root: `outputs/tmp/groot_http_zmq_runtime_verify_20260529/`

같은 actual RoboCasa observation에서 HTTP `/act`와 ZMQ SAFE feature path action을 비교했다. ZMQ path는 `SafeN16FeaturePolicy.get_action_with_features`와 같은 action + feature extraction code path를 사용했다.

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

즉 `ep_meta`는 scenario/layout/style/config replay artifact이고, reset 직후 full sim state/qpos까지 고정하는 artifact가 아니다. Closed-loop action trace까지 byte-level로 맞추려면 reset 직후 `sim.data.qpos/qvel` 또는 full MuJoCo state를 별도 artifact로 저장/복원해야 한다.

현재 HTTP와 ZMQ 비교에는 transport 이외의 차이가 함께 있다.

- HTTP 경로는 보통 project runner를 타며 `src/benchmarks/robocasa`의 robocasa365 v1.0 환경을 사용한다.
- ZMQ official 경로는 `src/policies/Isaac-GR00T/external_dependencies/robocasa`의 RoboCasa v0.2 (`robocasa_v02`) 환경을 사용한다.
- transport, env version, task class name, observation schema, action application 방식 차이가 함께 섞여 있다.

공통 task로 고른 5개는 v0.2와 v1.0 사이에서 의미적으로 대응되는 task다. `OpenSingleDoor`/`OpenCabinet`처럼 official SR이 높은 task에서 HTTP SR이 크게 낮으면 policy wiring을 먼저 점검한다.

HTTP SR 회복을 위한 검증 순서:

1. Same-observation HTTP `/act` vs ZMQ SAFE action parity를 기준으로 둔다. 2026-05-29 기준 `max_abs_all=0.0`.
2. 같은 env에서 HTTP action application loop만 교체해서 SR을 본다.
3. action은 같은데 SR만 낮으면 action consumption, reset, termination, wrapper, env version 차이를 본다.
4. closed-loop trace equality가 필요하면 `ep_meta`만 쓰지 말고 reset 직후 full sim state/qpos replay artifact를 추가한다.

우선순위가 높은 체크포인트:

- image keys: `side_0`, `side_1`, `wrist_0`가 GR00T의 `res256_image_*` 계열과 동일 의미인지 확인한다.
- state keys: HTTP payload의 `eef_pos_rel`, `eef_quat_rel`, `gripper_qpos`, `base_position`, `base_rotation`이 GR00T native relative/base state key로 들어가며 batch/time dimension이 official wrapper와 같은지 확인한다.
- action keys: `end_effector_position`, `end_effector_rotation`, `gripper_close`, `base_motion`, `control_mode`가 HTTP의 `action.eef_pos`, `action.eef_axisangle`, `action.gripper`, `action.base_motion`, `action.control_mode`로 손실 없이 매핑되는지 확인한다.
- action chunk: GR00T N1.6의 chunk horizon과 RoboCasa action repeat/consume 방식이 HTTP runner와 ZMQ runner에서 같은지 확인한다.
- reset: episode 시작 시 `/reset`이 반드시 호출되고, server-side policy state가 official ZMQ path와 같은 시점에 초기화되는지 확인한다.
- env version: robocasa365 v1.0에서 task semantics가 v0.2와 충분히 같은지 task별로 확인한다.
