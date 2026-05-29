# GR00T N1.5 RoboCasa Evaluation

`n15_01_finetune.md`의 후속 문서다. N1.5 평가는 N1.6 평가와 실행 경로가 다르므로 `n16_02_eval.md`와 분리한다. N1.5는 ZMQ-only protocol을 사용하며, 통일 HTTP API (`/act`, `/act_with_features`) 는 N1.6 전용이다 (참고: [`n16_11_http_act_changes.md`](n16_11_http_act_changes.md), [`../01_serving_interface.md`](../01_serving_interface.md)).

## Scope

| Scope | Embodiment | Data config | Main scripts |
| --- | --- | --- | --- |
| Official GR1 tabletop | `gr1` | `fourier_gr1_arms_waist` | `inference_service.py`, `simulation_service.py` |
| PandaOmron target atomic-seen 15-task | `new_embodiment` | `robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig` | `eval_policy.py`, `inference_service.py`, `groot_n15_robocasa_zmq_eval.py`, `run_groot_n15_target15_seedpairs.sh` |

N1.6 eval은 `groot`/`robocasa` 두 container와 `scripts/eval/groot_robocasa.sh`를 사용한다. N1.5 eval은 `groot_n15` service와 N1.5 submodule의 scripts를 사용한다.

## 준비

```bash
cd "$(git rev-parse --show-toplevel)"
docker compose build groot_n15
docker compose up -d groot_n15
```

Container 안 import 확인:

```bash
docker compose exec groot_n15 bash -lc '
python -c "import torch, flash_attn, gr00t; print(torch.__version__, torch.version.cuda); print(flash_attn.__version__); print(gr00t.__file__)"
'
```

N1.5 base model은 repository-local Hugging Face cache에 있다.

```text
host:
<cache>/datasets/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e

container:
/cache/datasets/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

`groot_n15` service의 `HF_HOME`은 `/cache/datasets/huggingface`로 둔다. 따라서 `nvidia/GR00T-N1.5-3B` repo id도 이 cache에서 해석될 수 있지만, eval 명령에서는 네트워크 접근을 피하기 위해 snapshot path를 직접 넘긴다.

PandaOmron `new_embodiment` baseline은 base snapshot을 그대로 넘기지 않는다. N1.5 base snapshot의 `experiment_cfg/metadata.json`에는 `gr1`, `oxe_droid`, `agibot_genie1` metadata만 있고 `new_embodiment` metadata가 없다. Base weights를 그대로 쓰되 PandaOmron target-15 metadata를 붙인 local wrapper checkpoint를 먼저 만든다.

```bash
docker exec -it groot_n15 bash -lc '
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla \
python /temporal_vla/scripts/utils/prepare_groot_n15_base_new_embodiment.py
'
```

Wrapper checkpoint:

```text
/temporal_vla/outputs/checkpoints/groot_n15_base_pandaomron_new_embodiment
```

## PandaOmron Target 15-Task Eval

PandaOmron은 N1.5에 pretrained embodiment tag가 없으므로 fine-tuning과 evaluation 모두 `new_embodiment`를 사용한다. `nvidia/GR00T-N1.5-3B` base model과 fine-tuned checkpoint 비교는 Panda pretrained 성능 비교가 아니라 같은 PandaOmron schema에서 fine-tuning 전후를 보는 ablation이다.

첫 검증은 simulator rollout보다 offline dataset MSE를 우선 사용한다.

Base model:

```bash
docker compose exec groot_n15 bash -lc '
N15_PANDA_BASE_MODEL=/temporal_vla/outputs/checkpoints/groot_n15_base_pandaomron_new_embodiment
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/eval_policy.py \
  --model-path "$N15_PANDA_BASE_MODEL" \
  --dataset-path /cache/datasets/robocasa/v1.0/target/atomic/OpenCabinet/20250813/lerobot \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --embodiment-tag new_embodiment \
  --modality-keys base_motion control_mode end_effector_position end_effector_rotation gripper_close \
  --steps 150 \
  --trajs 3
'
```

Fine-tuned checkpoint:

```bash
docker compose exec groot_n15 bash -lc '
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/eval_policy.py \
  --model-path /temporal_vla/outputs/train/groot_n1_5/<RUN>/checkpoint-<STEP> \
  --dataset-path /cache/datasets/robocasa/v1.0/target/atomic/OpenCabinet/20250813/lerobot \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --embodiment-tag new_embodiment \
  --modality-keys base_motion control_mode end_effector_position end_effector_rotation gripper_close \
  --steps 150 \
  --trajs 3
'
```

Inference server smoke, base model:

```bash
docker compose exec groot_n15 bash -lc '
N15_PANDA_BASE_MODEL=/temporal_vla/outputs/checkpoints/groot_n15_base_pandaomron_new_embodiment
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model-path "$N15_PANDA_BASE_MODEL" \
  --embodiment-tag new_embodiment \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --port 5555
'
```

Inference server smoke, fine-tuned checkpoint:

```bash
docker compose exec groot_n15 bash -lc '
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model-path /temporal_vla/outputs/train/groot_n1_5/<RUN>/checkpoint-<STEP> \
  --embodiment-tag new_embodiment \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --port 5555
'
```

`groot_n15` container는 N1.5 policy server / offline eval 용도다. RoboCasa simulator rollout은 `robocasa` container에서 실행한다. 기존 `scripts/eval/groot_robocasa.sh client-target15`는 N1.6 server protocol용이므로 N1.5 server에는 붙이지 않는다.

PandaOmron rollout smoke, client:

아래 client는 위 `Inference server smoke` 중 하나가 떠 있는 상태에서 실행한다. Base server를 띄웠으면 base rollout smoke, fine-tuned server를 띄웠으면 fine-tuned rollout smoke가 된다.

```bash
docker exec -it robocasa bash -lc '
export MUJOCO_GL=egl
PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla \
python /temporal_vla/scripts/eval/groot_n15_robocasa_zmq_eval.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --env-name robocasa_panda_omron/OpenCabinet_PandaOmron_Env \
  --n-episodes 1 \
  --n-envs 1 \
  --n-action-steps 8 \
  --max-episode-steps 120 \
  --steps-per-render 2 \
  --video-fps 10 \
  --video-dir /temporal_vla/outputs/eval/robocasa/groot_n15/smoke_open_cabinet
'
```

Full target-15 rollout은 위 smoke가 끝난 뒤 episode 수와 task loop를 늘린다. N1.5 base model은 PandaOmron pretrained head가 아니므로, base rollout result는 fine-tuning 전후 비교용 sanity baseline으로만 본다.

`groot_n15_robocasa_zmq_eval.py`는 RoboCasa env observation 중 N1.5 PandaOmron data config가 요구하는 key만 서버로 보낸다. Extra `state.*`, original `video.res*`, duplicate language key를 같이 보내면 N1.5 transform이 잘못된 modality set으로 처리할 수 있다. Base `new_embodiment` rollout은 wiring 검증용이며, 행동 성능은 fine-tuned checkpoint에서 판단한다.

Video length is controlled by `max_episode_steps / steps_per_render / video_fps`. For example, `max_episode_steps=120`, `steps_per_render=2`, `video_fps=20` produces `60 frames / 20 fps = 3s`. Use `--video-fps 10` with `--steps-per-render 2` to show the 120-step smoke as roughly 6s.

### PandaOmron target-15 rollout runner

위의 `Inference server smoke` / `PandaOmron rollout smoke, client`가 가장 기본적인 manual server-client 통신 확인 절차다. Full target-15 SR rollout은 server를 같은 방식으로 먼저 띄워 둔 뒤, `scripts/eval/run_groot_n15_target15_seedpairs.sh`로 client 실행을 반복한다. 이 wrapper는 server를 띄우지 않고, `robocasa` container에서 client만 실행한다.

`scripts/eval/run_groot_n15_target15_seedpairs.sh`는 내부적으로 N1.5 server protocol용 `groot_n15_robocasa_zmq_eval.py`를 호출한다. `scripts/eval/groot_robocasa.sh client-target15`는 N1.6 server protocol용이므로 N1.5 server에는 사용하지 않는다.

평가 단위는 dataset episode index가 아니라 simulator seed다. 예를 들어 `--seed-start 101 --seed-count 4`는 task별로 seeds `101,102,103,104` 네 scene을 rollout한다. `n_envs=2`이면 seed pair를 `101-102`, `103-104`로 나누어 실행한다.

Single server-client eval은 server 하나를 이미 띄운 상태에서 client만 한 종류로 실행한다:

```bash
# tuned server 하나만 평가. Remote tuned server 예시.
TUNED_HOST=166.104.35.50 \
TUNED_PORT=5556 \
RUN_ID=n15_tuned_seed101_104_$(date +%Y%m%d_%H%M%S) \
bash scripts/eval/run_groot_n15_target15_seedpairs.sh \
  --mode tuned \
  --seed-start 101 \
  --seed-count 4 \
  --n-envs 4
```

```bash
# local base server 하나만 평가. Base server가 127.0.0.1:5557에 떠 있어야 한다.
BASE_HOST=127.0.0.1 \
BASE_PORT=5557 \
RUN_ID=n15_base_seed101_104_$(date +%Y%m%d_%H%M%S) \
bash scripts/eval/run_groot_n15_target15_seedpairs.sh \
  --mode base \
  --seed-start 101 \
  --seed-count 4 \
  --n-envs 4
```

Base/tuned pair comparison은 base server와 tuned server를 둘 다 띄운 상태에서 같은 task/seed client를 병렬 실행한다:

```bash
# base 2 env + tuned 2 env = 총 4 RoboCasa env 동시 실행.
BASE_HOST=127.0.0.1 \
BASE_PORT=5557 \
TUNED_HOST=166.104.35.50 \
TUNED_PORT=5556 \
RUN_ID=n15_target15_seed101_104_base_tuned_$(date +%Y%m%d_%H%M%S) \
bash scripts/eval/run_groot_n15_target15_seedpairs.sh \
  --mode both \
  --seed-start 101 \
  --seed-count 4 \
  --n-envs 2
```

다음 4개 seed set은 같은 방식으로 `--seed-start 105 --seed-count 4`를 사용한다. 두 set은 서로 다른 seeds를 쓰므로 같은 task라도 scene 초기화가 겹치지 않는다.

단일 task만 확인할 때:

```bash
TASKS_CSV=OpenCabinet \
RUN_ID=n15_tuned_open_cabinet_seed101_104_$(date +%Y%m%d_%H%M%S) \
bash scripts/eval/run_groot_n15_target15_seedpairs.sh \
  --mode tuned \
  --seed-start 101 \
  --seed-count 4 \
  --n-envs 4
```

Output layout:

```text
outputs/eval/robocasa/groot_n15/target15_seedpairs_<RUN_ID>/
├── summary.tsv
├── vram.csv                         # VRAM monitor를 별도로 붙인 경우
├── base/<Task>/seed_<N>/run.log
├── base/<Task>/seed_<N>/videos/<Task>_seed<N>.mp4
├── tuned/<Task>/seed_<N>/run.log
└── tuned/<Task>/seed_<N>/videos/<Task>_seed<N>.mp4
```

`base`와 `tuned`의 같은 task/seed 영상은 파일명이 같고 경로만 다르다. 예를 들어 `base/OpenCabinet/seed_101/videos/OpenCabinet_seed101.mp4`와 `tuned/OpenCabinet/seed_101/videos/OpenCabinet_seed101.mp4`를 직접 비교한다. 성공 여부는 파일명 suffix가 아니라 `summary.tsv`와 각 `run.log`의 `results: [True, False, ...]`로 판단한다.

VRAM 주의: `--mode both --n-envs 2`는 local base server와 RoboCasa render/env까지 겹쳐 16GB GPU에서 15GB 후반까지 올라갈 수 있다. OOM이 나면 `--mode tuned --n-envs 4`처럼 단일 server-client로 먼저 돌리거나, `--mode both --n-envs 1`로 낮춘다.

## Official GR1 Tabletop Eval

공식 N1.5 RoboCasa README는 먼저 inference server를 열고, RoboCasa simulation client를 실행하는 방식을 사용한다. 공식 reported SR은 task당 50 rollouts 기준이며, `n_envs > 1`은 success rate를 낮출 수 있다는 경고가 있다.

이 절은 official GR1 tabletop 재현용 참고다. PandaOmron target-15 평가는 위 `PandaOmron Target 15-Task Eval` 절의 `new_embodiment` / `robocasa` container client 절차를 따른다.

Base model server:

```bash
docker compose exec groot_n15 bash -lc '
N15_BASE_MODEL=/cache/datasets/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model-path "$N15_BASE_MODEL" \
  --embodiment-tag gr1 \
  --data-config fourier_gr1_arms_waist \
  --port 5555
'
```

Official fine-tuned checkpoint server:

```bash
docker compose exec groot_n15 bash -lc '
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model-path youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain \
  --embodiment-tag gr1 \
  --data-config fourier_gr1_arms_waist \
  --port 5555
'
```

Local GR1 checkpoint server:

```bash
docker compose exec groot_n15 bash -lc '
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model-path /temporal_vla/outputs/groot_n1_5_robocasa_tabletop/checkpoint-60000 \
  --embodiment-tag gr1 \
  --data-config fourier_gr1_arms_waist \
  --port 5555
'
```

Simulation client:

```bash
docker compose exec groot_n15 bash -lc '
python3 src/policies/Isaac-GR00T-N1.5/scripts/simulation_service.py --client \
  --env_name <TASK_NAME> \
  --video_dir /temporal_vla/outputs/eval/robocasa/groot_n15/smoke \
  --max_episode_steps 720 \
  --n_episodes 50
'
```

## Reported Reference

Official N1.5 RoboCasa checkpoint:

```text
youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain
```

Reported average success rate:

```text
0.48
```

Evaluation condition:

```text
24 RoboCasa GR1 tabletop tasks
50 rollouts per task
```

Do not compare this number directly with N1.6 PandaOmron or N1.5 PandaOmron `new_embodiment` results. The embodiment, task set, action/state schema, and evaluation stack differ.
