# GR00T N1.6 RoboCasa — Evaluation

`n16_01_finetune.md`의 후속 문서다. GR00T N1.6 fine-tuned checkpoint 또는 base checkpoint를 RoboCasa simulator에서 평가하는 방법을 정리한다.

> 관련 문서 (N1.6 reading order)
> - [Doc map](README.md)
> - [01 Fine-Tuning](n16_01_finetune.md)
> - **02 Evaluation (이 문서)**
> - [03 SAFE Overview](n16_03_safe_overview.md)
> - [04 SAFE Collection](n16_04_safe_collection.md)
> - [05 Scenario Reproduction](n16_05_safe_env_reproduction.md)
> - [06 Inference Datapoint Semantics](n16_06_safe_inference_semantics.md)
> - [07 SAFE Detector + Visualization + Report](n16_07_safe_detector_report.md)
> - [09 SAFE Parity](n16_09_safe_parity.md)

N1.5 평가는 `n15_02_eval.md`를 사용한다. N1.5와 N1.6은 container, entrypoint, embodiment/data config가 다르다.

사용 파일:

```text
scripts/eval/groot_robocasa.sh
scripts/eval/groot_robocasa_zmq_eval.py
```

## 평가 구조

권장 방식은 Docker container 두 개를 쓰는 ZMQ 평가다.

```text
groot container
└── GR00T model server
    └── run_gr00t_server.py

robocasa container
└── RoboCasa simulator client
    └── groot_robocasa_zmq_eval.py
```

`groot`는 model inference만 맡고, `robocasa`는 MuJoCo/RoboCasa simulator rollout을 맡는다. Client는 observation을 server에 보내고 action chunk를 받아 simulator에 적용한다.

`scripts/eval/groot_robocasa.sh`는 upstream Isaac-GR00T의 `uv` 기반 local eval 환경을 쓰지 않는다. Docker container의 `python`과 `PYTHONPATH`를 기준으로 실행된다.

upstream local eval 관련 파일은 별도로 존재한다.

```text
src/policies/Isaac-GR00T/gr00t/eval/sim/robocasa/setup_RoboCasa.sh
src/policies/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py
```

이 workflow에서는 `local` / `local-batch` mode를 쓰지 않는다. `groot` container 하나에 GR00T, RoboCasa, robosuite, MuJoCo, assets를 모두 맞추면 dependency 충돌이 생기기 쉽다.

## 모델 경로

Base checkpoint:

```text
/cache/checkpoints/nvidia/GR00T-N1.6-3B
```

Fine-tuned checkpoint:

```text
/temporal_vla/outputs/groot_robocasa_target_15tasks/checkpoint-20000
```

Server의 `MODEL_PATH`만 바꾸면 같은 client command로 base와 fine-tuned model을 비교할 수 있다.
기존 pretrain 10-task baseline checkpoint는 `/temporal_vla/outputs/groot_robocasa_baseline_10tasks/checkpoint-20000`에 둔다.

## 최초 준비

`groot` container:

```bash
docker exec -it groot bash /temporal_vla/scripts/eval/groot_robocasa.sh setup-server
```

`robocasa` container:

```bash
docker exec -it robocasa bash /temporal_vla/scripts/eval/groot_robocasa.sh setup-client
```

`setup-*` mode는 이미 준비된 항목을 skip한다. `setup-server`는 server import 준비만 하며, local simulator dependency 설치를 목표로 하지 않는다.

## Docker 통신

Compose 구성에서는 `groot`와 `robocasa`가 같은 network namespace를 공유하므로 client는 localhost로 server에 붙는다.

```text
POLICY_CLIENT_HOST=127.0.0.1
```

확인:

```bash
docker exec robocasa bash -lc 'timeout 3 bash -lc "</dev/tcp/127.0.0.1/5556" && echo open'
```

이 환경에서는 `getent hosts groot`가 실패할 수 있다. `POLICY_CLIENT_HOST=groot`를 쓰면 ZMQ client가 server 응답을 기다리며 멈춘 것처럼 보일 수 있다.

## Base Checkpoint 평가

Server:

```bash
docker exec -it groot bash -lc '
MODEL_PATH=/cache/checkpoints/nvidia/GR00T-N1.6-3B \
PORT=5556 \
bash /temporal_vla/scripts/eval/groot_robocasa.sh server
'
```

Target atomic-seen 15-task eval, task당 50 episodes:

```bash
docker exec -it robocasa bash -lc '
POLICY_CLIENT_HOST=127.0.0.1 \
PORT=5556 \
EVAL_RUN_ID=base_target15_50ep_env1_$(date +%Y%m%d_%H%M%S) \
bash /temporal_vla/scripts/eval/groot_robocasa.sh client-target15 50 1 8 720
'
```

Smoke eval:

```bash
docker exec -it robocasa bash -lc '
POLICY_CLIENT_HOST=127.0.0.1 \
PORT=5556 \
EVAL_RUN_ID=base_smoke_open_drawer \
bash /temporal_vla/scripts/eval/groot_robocasa.sh \
  client robocasa_panda_omron/OpenDrawer_PandaOmron_Env 5 1 8 720
'
```

## Fine-Tuned Checkpoint 평가

Server:

```bash
docker exec -it groot bash -lc '
MODEL_PATH=/temporal_vla/outputs/groot_robocasa_target_15tasks/checkpoint-20000 \
PORT=5556 \
bash /temporal_vla/scripts/eval/groot_robocasa.sh server
'
```

`scripts/eval/groot_robocasa.sh server`의 기본 `EMBODIMENT_TAG`는
`ROBOCASA_PANDA_OMRON`이다. RoboCasa365 `checkpoint-120000`
(`/temporal_vla/outputs/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000`)
은 checkpoint metadata와 `groot__robocasa365_ckpt120000.yaml`에 맞춰
`EMBODIMENT_TAG=NEW_EMBODIMENT`를 명시해서 띄운다.

```bash
docker exec -it groot bash -lc '
MODEL_PATH=/temporal_vla/outputs/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000 \
EMBODIMENT_TAG=NEW_EMBODIMENT \
PORT=5556 \
bash /temporal_vla/scripts/eval/groot_robocasa.sh server
'
```

Target atomic-seen 15-task eval:

```bash
docker exec -it robocasa bash -lc '
POLICY_CLIENT_HOST=127.0.0.1 \
PORT=5556 \
EVAL_RUN_ID=ft20000_target15_50ep_env1_$(date +%Y%m%d_%H%M%S) \
bash /temporal_vla/scripts/eval/groot_robocasa.sh client-target15 50 1 8 720
'
```

## Remote Server

Server를 다른 머신에서 직접 띄울 때:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"

CUDA_VISIBLE_DEVICES=3 python "${REPO_ROOT}/src/policies/Isaac-GR00T/gr00t/eval/run_gr00t_server.py" \
  --model-path "${REPO_ROOT}/outputs/groot_robocasa_target_15tasks/checkpoint-20000" \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper \
  --port 5556
```

Client는 server IP만 바꾼다.

```bash
docker exec -it robocasa bash -lc '
POLICY_CLIENT_HOST=166.104.35.98 \
PORT=5556 \
EVAL_RUN_ID=remote_target15_50ep_env1_$(date +%Y%m%d_%H%M%S) \
bash /temporal_vla/scripts/eval/groot_robocasa.sh client-target15 50 1 8 720
'
```

## Script Modes

| Mode | Container | 용도 |
| --- | --- | --- |
| `setup-server` | `groot` | GR00T server import/install 확인 |
| `setup-client` | `robocasa` | GR00T client import/install, RoboCasa asset 확인 |
| `server` | `groot` | GR00T ZMQ inference server 실행 |
| `client` | `robocasa` | 단일 task 평가 |
| `client-target15` | `robocasa` | target atomic-seen 15-task 평가 |
| `client-train10` | `robocasa` | legacy pretrain 10-task 평가 |
| `client-batch` | `robocasa` | 별도 task 묶음 평가 |

단일 task:

```text
client [ENV_NAME] [N_EPISODES] [N_ENVS] [N_ACTION_STEPS] [MAX_STEPS]
```

Target 15-task:

```text
client-target15 [N_EPISODES] [N_ENVS] [N_ACTION_STEPS] [MAX_STEPS]
```

권장값:

| 인자 | 의미 | 값 |
| --- | --- | --- |
| `N_EPISODES` | task별 episode 수 | 50 |
| `N_ENVS` | 병렬 simulator 수 | 공식 비교 1, 빠른 sweep 4 |
| `N_ACTION_STEPS` | action chunk 중 실행할 step 수 | 8 |
| `MAX_STEPS` | episode max step | 720 |

`N_ENVS > 1`에서는 vector env boundary 때문에 episode가 50개보다 조금 더 기록될 수 있다. 보고는 first 50 기준으로 맞춘다.

## Target Atomic-Seen 15-Task Scope

`client-target15`는 `scripts/train/groot_robocasa_target_finetune.sh`의 15개 target atomic-seen task family를 평가한다.

| # | Env name |
| --- | --- |
| 1 | `robocasa_panda_omron/CloseFridge_PandaOmron_Env` |
| 2 | `robocasa_panda_omron/CloseToasterOvenDoor_PandaOmron_Env` |
| 3 | `robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env` |
| 4 | `robocasa_panda_omron/OpenCabinet_PandaOmron_Env` |
| 5 | `robocasa_panda_omron/OpenDrawer_PandaOmron_Env` |
| 6 | `robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env` |
| 7 | `robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env` |
| 8 | `robocasa_panda_omron/PickPlaceDrawerToCounter_PandaOmron_Env` |
| 9 | `robocasa_panda_omron/PickPlaceSinkToCounter_PandaOmron_Env` |
| 10 | `robocasa_panda_omron/PickPlaceToasterToCounter_PandaOmron_Env` |
| 11 | `robocasa_panda_omron/SlideDishwasherRack_PandaOmron_Env` |
| 12 | `robocasa_panda_omron/TurnOffStove_PandaOmron_Env` |
| 13 | `robocasa_panda_omron/TurnOnElectricKettle_PandaOmron_Env` |
| 14 | `robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env` |
| 15 | `robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env` |

이 평가는 training LeRobot dataset의 recorded episode replay가 아니다. Simulator reset 후 policy rollout을 수행한다.

### Atomic task only 확인

Target atomic 15-task만 평가하려면 `client-target15`를 사용한다. `scripts/eval/groot_robocasa.sh`의 `client-target15` branch가 위 15개 env name만 순회하므로 `client-batch`나 `robocasa_eval_25`를 쓰지 않아도 된다.

확인 근거:

- `scripts/eval/groot_robocasa.sh`의 `client-target15` `TASKS` 배열은 위 15개 env name으로 고정되어 있다.
- RoboCasa fork는 `robocasa.environments.kitchen.KitchenEnvMeta`로 atomic task class를 `REGISTERED_KITCHEN_ENVS`에 등록한다.
- `robocasa.utils.gym_utils.gymnasium_groot`는 `REGISTERED_ENVS`와 `PandaOmron` robot alias를 조합해 `robocasa_panda_omron/<Task>_PandaOmron_Env` gym id를 만든다.
- `client-target15`의 15개 class는 `src/benchmarks/robocasa/robocasa/__init__.py`에서 import되어 registration 대상에 들어간다.
- 실제 `robocasa` container에서 `import robocasa.utils.gym_utils.gymnasium_groot` 후 아래 registry check가 `registered: 15 / 15`를 출력하는 것을 확인했다.

Docker 안에서 registry만 확인:

```bash
docker exec robocasa bash -lc '
PYTHONPATH=/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite python - <<'"'"'PY'"'"'
import gymnasium as gym
import robocasa  # noqa: F401
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

tasks = [
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenDrawer",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

missing = []
for task in tasks:
    env_id = f"robocasa_panda_omron/{task}_PandaOmron_Env"
    if env_id not in gym.envs.registry:
        missing.append(env_id)

print("registered:", len(tasks) - len(missing), "/", len(tasks))
if missing:
    raise SystemExit("missing: " + ", ".join(missing))
PY
'
```

단일 atomic task만 빠르게 확인할 때는 `client` mode를 직접 쓴다.

```bash
docker exec -it robocasa bash -lc '
POLICY_CLIENT_HOST=127.0.0.1 \
PORT=5556 \
EVAL_RUN_ID=smoke_open_drawer_atomic \
bash /temporal_vla/scripts/eval/groot_robocasa.sh \
  client robocasa_panda_omron/OpenDrawer_PandaOmron_Env 5 1 8 720
'
```

`client-batch`는 legacy sweep 목록이라 target 15-task 공식 비교에는 쓰지 않는다. 특히 일부 PnP env는 예전 `PnP...` alias를 사용하므로 target 15-task 문서의 `PickPlace...` 이름과 섞어 해석하면 안 된다.

## Eval Split

`src/benchmarks/robocasa/robocasa/utils/gym_utils/gymnasium_basic.py`는 Kitchen env에 다음 설정을 쓴다.

```python
layout_and_style_ids = [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]]
obj_instance_split = "target"
randomize_cameras = False
```

따라서 결과는 `pretrain/atomic` train episode 재현이 아니라 target split simulator rollout 성능으로 해석한다.

## 결과 저장

기본 위치:

```text
/temporal_vla/outputs/eval/robocasa/groot/<EVAL_RUN_ID>/
└── videos/
    ├── OpenDrawer/
    ├── CloseDrawer/
    └── ...
```

Video filename suffix:

```text
..._s1.mp4  # success
..._s0.mp4  # failure
```

Task stdout:

```text
Video saved to:  /temporal_vla/outputs/eval/robocasa/groot/<run>/videos/<task>
results:  ('robocasa_panda_omron/<Task>_PandaOmron_Env', [True, False, ...], {})
success rate:  0.62
```

## 로그 저장

긴 eval은 `run.log`를 남긴다.

```bash
docker exec -it robocasa bash -lc '
RUN_ID=base_target15_50ep_env1_$(date +%Y%m%d_%H%M%S)
OUT=/temporal_vla/outputs/eval/robocasa/groot/${RUN_ID}
mkdir -p "${OUT}"

POLICY_CLIENT_HOST=127.0.0.1 \
PORT=5556 \
EVAL_RUN_ID=${RUN_ID} \
bash /temporal_vla/scripts/eval/groot_robocasa.sh client-target15 50 1 8 720 \
  2>&1 | tee "${OUT}/run.log"
'
```

## 진행 확인

Server:

```bash
docker exec groot bash -lc 'ps -ef | grep -E "run_gr00t_server|groot_robocasa" | grep -v grep || true'
```

Client:

```bash
docker exec robocasa bash -lc 'ps -ef | grep -E "groot_robocasa|python" | grep -v grep || true'
```

GPU:

```bash
docker exec groot nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Video count:

```bash
docker exec robocasa bash -lc '
RUN=$(find /temporal_vla/outputs/eval/robocasa/groot -maxdepth 1 -type d -name "base_target15_50ep_env1_*" | sort | tail -1)
echo "${RUN}"
find "${RUN}" -type f -name "*.mp4" | wc -l
'
```

## SR 계산

`run.log`에서 task별 SR을 계산한다.

```bash
python - <<'PY'
import ast
import re
from pathlib import Path

log = Path("outputs/eval/robocasa/groot/<RUN_ID>/run.log")
pat = re.compile(r"results:\s+(\(.*\))")

rows = []
for line in log.read_text(errors="replace").splitlines():
    m = pat.search(line)
    if not m:
        continue
    env, values, _ = ast.literal_eval(m.group(1))
    name = env.split("/")[-1].replace("_PandaOmron_Env", "")
    first50 = values[:50]
    succ = sum(bool(x) for x in first50)
    rows.append((name, succ, len(first50), succ / len(first50)))

print("task,success,n,sr")
for name, succ, n, sr in rows:
    print(f"{name},{succ},{n},{sr:.3f}")

total_succ = sum(r[1] for r in rows)
total_n = sum(r[2] for r in rows)
macro = sum(r[3] for r in rows) / len(rows)
print(f"total,{total_succ},{total_n},{total_succ / total_n:.3f}")
print(f"macro_sr,{macro:.3f}")
PY
```

## Observation Key Alias

Fine-tuned RoboCasa checkpoint는 training dataset key를 기대한다.

```text
video.robot0_agentview_left
video.robot0_agentview_right
video.robot0_eye_in_hand
annotation.human.task_description
```

Simulator wrapper는 다른 key를 낸다.

```text
video.res256_image_side_0
video.res256_image_side_1
video.res256_image_wrist_0
annotation.human.action.task_description
```

server 호출 전에 이 alias 보정을 적용해야 한다. 보정 매핑은 다음과 같다.

| simulator wrapper key | server 기대 key |
|---|---|
| `video.res256_image_side_0` | `video.robot0_agentview_left` |
| `video.res256_image_side_1` | `video.robot0_agentview_right` |
| `video.res256_image_wrist_0` | `video.robot0_eye_in_hand` |
| `annotation.human.action.task_description` | `annotation.human.task_description` |

이 매핑은 더 이상 eval 스크립트가 자체 dict로 들고 있지 않다. 공통 adapter
`src/policies/groot/robocasa/io.py`의 `OBS_ALIASES`(`_bidirectional_aliases(VIDEO_KEY_GROUPS)`로
생성)와 `prepare_groot_robocasa_observation(observation, strict=True)`가 담당하고,
`scripts/eval/groot_robocasa_zmq_eval.py`는 이 함수를 호출(L63)해서 policy payload를 만든다.
덕분에 HTTP eval, SAFE HTTP, SAFE ZMQ, N1.6 ZMQ eval이 같은 native observation alias와
required-key 검증을 공유한다 (상세: [n16_12 RoboCasa Refactor Report](n16_12_robocasa_refactor_report.md)).

이 보정이 없으면 fine-tuned checkpoint에서 observation validation이 실패할 수 있다.

## 해석 기준

1. Base checkpoint로 simulator와 ZMQ 통신을 먼저 확인한다.
2. Fine-tuned checkpoint를 같은 task와 같은 setting으로 비교한다.
3. SR이 낮으면 video를 본다.
4. Opening task는 handle contact와 pull 방향을 본다.
5. Pick-place task는 grasp, gripper command, target approach를 본다.
6. Fine-tuned checkpoint만 크게 나쁘면 processor statistics, modality config, fine-tuning distribution을 확인한다.

## Troubleshooting

### Client가 server에 붙지 못함

Compose 구성에서는 `POLICY_CLIENT_HOST=127.0.0.1`을 쓴다.

```bash
docker exec robocasa bash -lc 'timeout 3 bash -lc "</dev/tcp/127.0.0.1/5556" && echo open'
docker exec groot bash -lc 'ps -ef | grep run_gr00t_server | grep -v grep || true'
```

### Video가 안 보임

Episode가 끝나야 mp4가 flush된다. 초반에는 output directory만 있고 `.mp4`가 0개일 수 있다.

```bash
docker exec robocasa bash -lc '
RUN=/temporal_vla/outputs/eval/robocasa/groot/<RUN_ID>
find "${RUN}" -type f -name "*.mp4" | wc -l
'
```

### `/tmp`에 저장됨

`local` mode는 upstream `rollout_policy.py` 기본 path를 쓰므로 `/tmp/sim_eval_videos_*`가 생길 수 있다. 권장 ZMQ mode인 `client`, `client-target15`, `client-train10`은 `${EVAL_OUTPUT_DIR}/${EVAL_RUN_ID}/videos/<task>`에 저장한다.

### SR이 매우 낮음

- Base도 낮으면 simulator split, task difficulty, camera/action wrapper를 본다.
- Fine-tuned만 낮으면 checkpoint statistics, modality config, fine-tuning data distribution을 본다.
- Close task만 상대적으로 높으면 pushing은 되지만 handle pull이나 object grasp가 약한 패턴일 수 있다.
