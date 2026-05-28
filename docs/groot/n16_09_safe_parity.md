# SAFE x GR00T N1.6 RoboCasa — Parity

이 문서는 **task 성능 평가가 아니라 SAFE wiring의 action parity 검증**이다.

| 목적 | 문서 |
|---|---|
| Fine-tuned 또는 base checkpoint의 task 성능 측정 (Docker workflow) | [02 Evaluation](n16_02_eval.md) |
| (1) SAFE feature server가 공식 RoboCasa eval과 같은 action을 내는지 (2) HTTP `/act`가 ZMQ와 parity 맞는지 | **이 문서** |

ZMQ official baseline과 HTTP common serving path 모두 유지한다. SR 기준선은 ZMQ로 두고, HTTP는 action parity 검증 후 편입한다.

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
  --profile /temporal_vla/configs/checkpoints/groot__robocasa_panda_omron.yaml
```

API:

- `POST /act`
- `POST /reset`
- `GET /health`

Health check:

```bash
curl http://127.0.0.1:8000/health
```

역할:

- HTTP는 heterogeneous VLA serving 경로다.
- GR00T N1.6 RoboCasa SR 기준선은 ZMQ official eval이다.
- HTTP SR은 observation/action parity 확인 후 SAFE/GR00T N1.6 성능 지표에 편입한다.
- 낮은 HTTP SR에서는 wiring/runner/schema mismatch를 우선 점검한다.

현재 HTTP와 ZMQ 비교에는 transport 이외의 차이가 함께 있다.

- HTTP 경로는 보통 project runner를 타며 `src/benchmarks/robocasa`의 robocasa365 v1.0 환경을 사용한다.
- ZMQ official 경로는 `src/policies/Isaac-GR00T/external_dependencies/robocasa`의 RoboCasa v0.2 (`robocasa_v02`) 환경을 사용한다.
- transport, env version, task class name, observation schema, action application 방식 차이가 함께 섞여 있다.

공통 task로 고른 5개는 v0.2와 v1.0 사이에서 의미적으로 대응되는 task다. `OpenSingleDoor`/`OpenCabinet`처럼 official SR이 높은 task에서 HTTP SR이 크게 낮으면 policy wiring을 먼저 점검한다.

HTTP SR 회복을 위한 검증 순서:

1. 같은 checkpoint, 같은 seed, 같은 initial env state에 최대한 가깝게 맞춘다.
2. ZMQ official client가 만든 raw observation을 저장한다.
3. 같은 observation을 HTTP `/act` 입력 payload로 변환한다.
4. HTTP output action을 GR00T native action key로 되돌려 ZMQ official action과 비교한다.
5. action key별 shape, first action, chunk horizon, scale, gripper sign, rotation convention을 비교한다.
6. action이 같거나 충분히 가까우면, 같은 env에서 HTTP action application loop만 교체해서 SR을 본다.
7. action은 같은데 SR만 낮으면 action consumption, reset, termination, wrapper, env version 차이를 본다.
8. action부터 다르면 observation conversion 또는 output mapping을 먼저 고친다.

우선순위가 높은 체크포인트:

- image keys: `side_0`, `side_1`, `wrist_0`가 GR00T의 `res256_image_*` 계열과 동일 의미인지 확인한다.
- state keys: `eef_pos`, `eef_quat`, `gripper_qpos`, `joint_pos`의 순서와 batch/time dimension이 official wrapper와 같은지 확인한다.
- action keys: `end_effector_position`, `end_effector_rotation`, `gripper_close`, `base_motion`, `control_mode`가 HTTP의 `action.eef_pos`, `action.eef_axisangle`, `action.gripper`로 손실 없이 매핑되는지 확인한다.
- action chunk: GR00T N1.6의 chunk horizon과 RoboCasa action repeat/consume 방식이 HTTP runner와 ZMQ runner에서 같은지 확인한다.
- reset: episode 시작 시 `/reset`이 반드시 호출되고, server-side policy state가 official ZMQ path와 같은 시점에 초기화되는지 확인한다.
- env version: robocasa365 v1.0에서 task semantics가 v0.2와 충분히 같은지 task별로 확인한다.
