# GR00T N1.5 RoboCasa Evaluation

`docs/groot_n1_5_robocasa_finetune_setup.md`의 후속 문서다. N1.5 평가는 N1.6 평가와 실행 경로가 다르므로 `docs/groot_robocasa_eval_setup.md`와 분리한다.

## Scope

| Scope | Embodiment | Data config | Main scripts |
| --- | --- | --- | --- |
| Official GR1 tabletop | `gr1` | `fourier_gr1_arms_waist` | `inference_service.py`, `simulation_service.py` |
| PandaOmron target atomic-seen 15-task | `new_embodiment` | `robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig` | `eval_policy.py`, `inference_service.py` |

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
data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e

container:
/temporal_vla/data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

`groot_n15` service의 `HF_HOME`은 `/temporal_vla/data/huggingface`로 둔다. 따라서 `nvidia/GR00T-N1.5-3B` repo id도 이 cache에서 해석될 수 있지만, eval 명령에서는 네트워크 접근을 피하기 위해 snapshot path를 직접 넘긴다.

## PandaOmron Target 15-Task Eval

PandaOmron은 N1.5에 pretrained embodiment tag가 없으므로 fine-tuning과 evaluation 모두 `new_embodiment`를 사용한다. `nvidia/GR00T-N1.5-3B` base model과 fine-tuned checkpoint 비교는 Panda pretrained 성능 비교가 아니라 같은 PandaOmron schema에서 fine-tuning 전후를 보는 ablation이다.

첫 검증은 simulator rollout보다 offline dataset MSE를 우선 사용한다.

Base model:

```bash
docker compose exec groot_n15 bash -lc '
N15_BASE_MODEL=/temporal_vla/data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/eval_policy.py \
  --model-path "$N15_BASE_MODEL" \
  --dataset-path /temporal_vla/data/robocasa/v1.0/target/atomic/OpenCabinet/20250813/lerobot \
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
  --dataset-path /temporal_vla/data/robocasa/v1.0/target/atomic/OpenCabinet/20250813/lerobot \
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
N15_BASE_MODEL=/temporal_vla/data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model_path "$N15_BASE_MODEL" \
  --embodiment_tag new_embodiment \
  --data_config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --port 5555
'
```

Inference server smoke, fine-tuned checkpoint:

```bash
docker compose exec groot_n15 bash -lc '
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:$PYTHONPATH \
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model_path /temporal_vla/outputs/train/groot_n1_5/<RUN>/checkpoint-<STEP> \
  --embodiment_tag new_embodiment \
  --data_config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --port 5555
'
```

`groot_n15` container는 N1.5 policy server / offline eval 용도다. RoboCasa simulator rollout은 `robocasa` container에서 실행한다. 기존 `scripts/eval/groot_robocasa.sh client-target15`는 N1.6 server protocol용이므로 N1.5 server에는 붙이지 않는다.

PandaOmron rollout smoke, server:

```bash
docker compose up -d groot_n15

docker exec -it groot_n15 bash -lc '
N15_BASE_MODEL=/temporal_vla/data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
PYTHONPATH=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla \
python3 /temporal_vla/src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model_path "$N15_BASE_MODEL" \
  --embodiment_tag new_embodiment \
  --data_config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --port 5555
'
```

PandaOmron rollout smoke, client:

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
  --video-dir /temporal_vla/outputs/eval/robocasa/groot_n15/smoke_open_cabinet
'
```

Full target-15 rollout은 위 smoke가 끝난 뒤 episode 수와 task loop를 늘린다. N1.5 base model은 PandaOmron pretrained head가 아니므로, base rollout result는 fine-tuning 전후 비교용 sanity baseline으로만 본다.

## Official GR1 Tabletop Eval

공식 N1.5 RoboCasa README는 먼저 inference server를 열고, RoboCasa simulation client를 실행하는 방식을 사용한다. 공식 reported SR은 task당 50 rollouts 기준이며, `n_envs > 1`은 success rate를 낮출 수 있다는 경고가 있다.

Base model server:

```bash
docker compose exec groot_n15 bash -lc '
N15_BASE_MODEL=/temporal_vla/data/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model_path "$N15_BASE_MODEL" \
  --embodiment_tag gr1 \
  --data_config fourier_gr1_arms_waist \
  --port 5555
'
```

Official fine-tuned checkpoint server:

```bash
docker compose exec groot_n15 bash -lc '
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model_path youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain \
  --embodiment_tag gr1 \
  --data_config fourier_gr1_arms_waist \
  --port 5555
'
```

Local GR1 checkpoint server:

```bash
docker compose exec groot_n15 bash -lc '
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model_path /temporal_vla/outputs/groot_n1_5_robocasa_tabletop/checkpoint-60000 \
  --embodiment_tag gr1 \
  --data_config fourier_gr1_arms_waist \
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
