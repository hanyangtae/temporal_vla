# GR00T N1.5 RoboCasa — Native ZMQ OpenFridge Smoke

이 문서는 LeRobot N1.5 behavior mismatch를 분리하기 위한 native Isaac-GR00T N1.5
ZMQ 비교 기록이다. 기준 runbook은 [`n15_02_eval.md`](n15_02_eval.md)이며,
N1.6과 같은 `robocasa` simulator client + policy server 구조로 실행한다.

## Runtime 메모

`groot_n15` compose service의 기본 cache mount는 `/cache`다. 현재 host
`/home/dongkyu/.cache/temporal_vla`가 `nobody:nogroup` 소유라 N1.5 server가
`/cache/datasets/huggingface` 아래 dynamic module cache를 만들 수 없다. 서버 실행
시 repo-local HF cache로 override한다.

```bash
docker compose run --rm --no-deps -T \
  -e HF_HOME=/temporal_vla/data/huggingface \
  -e HF_HUB_CACHE=/temporal_vla/data/huggingface/hub \
  -e HUGGINGFACE_HUB_CACHE=/temporal_vla/data/huggingface/hub \
  -e TRANSFORMERS_CACHE=/temporal_vla/data/huggingface/hub \
  -e HF_MODULES_CACHE=/temporal_vla/data/huggingface/modules \
  groot_n15 bash -lc '
python3 src/policies/Isaac-GR00T-N1.5/scripts/inference_service.py --server \
  --model-path /temporal_vla/data/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/14895998fe7c8f8f2441cc8957ec2c510302758b/gr00t_n1-5/multitask_learning/checkpoint-120000 \
  --embodiment-tag new_embodiment \
  --data-config robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig \
  --port 5558'
```

서버 확인:

```bash
ss -ltnp | rg ':5558'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## OpenFridge Smoke: Legacy local env ID

```bash
docker exec temporal_vla-robocasa-run-3705634bbbf6 bash -lc '
export MUJOCO_GL=egl
PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla \
python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/native_zmq_eval.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5558 \
  --env-name robocasa_panda_omron/OpenFridge_PandaOmron_Env \
  --n-episodes 1 \
  --n-envs 1 \
  --n-action-steps 16 \
  --max-episode-steps 400 \
  --steps-per-render 2 \
  --video-fps 10 \
  --seed 100002 \
  --video-dir /temporal_vla/outputs/debug/groot_n15_native_OpenFridge_seed100002/videos'
```

2026-06-09 결과:

```text
results:  ('robocasa_panda_omron/OpenFridge_PandaOmron_Env', [False], {})
success rate:  0.0
```

이 run은 raw `metadata.json` insertion order를 따르던 예전 repo-local data config를
사용했다. RoboCasa benchmark fork의 `PandaOmronDataConfig`는
`eef_pos/eef_rot/gripper/base_pos/base_rot` 정책 순서를 쓰고, state quaternion에는
6D rotation 변환을 적용한다. 따라서 이 예전 결과는 policy quality 결과가 아니라
wiring artifact로 본다.

산출물:

```text
outputs/debug/groot_n15_native_OpenFridge_seed100002/videos/3099486d-c4e7-4c98-8a05-554cb08b7ca2_s0.mp4
outputs/debug/groot_n15_native_OpenFridge_seed100002/contact_sheet.jpg
outputs/debug/groot_n15_native_OpenFridge_seed100002/contact_sheet_large.jpg
```

## OpenFridge Smoke: RoboCasa benchmark env

RoboCasa benchmark fork는 `panda_omron`을 `robocasa/<Task>`와 명시적인
`pretrain`/`target` split으로 평가한다. OpenFridge 기준으로는
`robocasa/OpenFridge`, `split=target`, 그리고 `get_task_horizon("OpenFridge")`에서
가져온 task horizon을 쓴다.

```bash
docker exec \
  -e MUJOCO_GL=egl \
  -e PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla:/temporal_vla/configs/policies \
  temporal_vla-robocasa-run-3705634bbbf6 \
  python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py \
    --host 127.0.0.1 \
    --port 5558 \
    --task OpenFridge \
    --split target \
    --n-episodes 1 \
    --n-action-steps 16 \
    --video-fps 10 \
    --steps-per-render 2 \
    --video-dir /temporal_vla/outputs/debug/groot_n15_officialrun_OpenFridge_target_verified/videos
```

2026-06-09 결과:

```text
Creating OpenFridge with split=target
EP 1 success: True; Cumulative success rate: 1.0
Collecting 1 episodes took 29.11 seconds
results:  ('robocasa/OpenFridge', [True])
success rate:  1.0
```

산출물:

```text
outputs/debug/groot_n15_officialrun_OpenFridge_target_verified/videos/616a8eaf-c1ee-40e0-a693-bf99753745d7_success1.mp4
outputs/debug/groot_n15_officialrun_OpenFridge_target_verified/contact_sheet_large.jpg
```

## OpenFridge Smoke: LeRobot HTTP official env

LeRobot HTTP 경로도 같은 official env id와 split convention을 사용한다. 다만 policy
호출은 이 repo의 통일 `/act` server를 거친다. Server-side GR00T adapter는 두 quaternion
state field를 같은 PyTorch3D/native `wxyz` convention으로 `rotation_6d`로 변환해
native state feature shape(`20D`)와 맞춘다.

```bash
docker exec \
  -e MUJOCO_GL=egl \
  -e NO_PROXY=127.0.0.1,localhost \
  -e no_proxy=127.0.0.1,localhost \
  -e PYTHONPATH=/temporal_vla:/temporal_vla/scripts/utils:/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite \
  temporal_vla-robocasa-run-3705634bbbf6 \
  python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/lerobot_http_eval.py \
    --vla-server http://127.0.0.1:8400 \
    --task OpenFridge \
    --split target \
    --n-episodes 1 \
    --seed 0 \
    --timeout 300 \
    --video-fps 10 \
    --steps-per-render 2 \
    --success-debug-every 80 \
    --video-dir /temporal_vla/outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_seed0_success_probe/videos
```

2026-06-09 seed-0 target 결과:

```text
Creating OpenFridge with split=target
[success-debug] episode=1 step=160 reward=0.0 success=False open=fridge_right_group_fridge_door_joint=0.633 info_keys=['success']
[success-debug] episode=1 step=172 reward=1.0 success=True open=fridge_right_group_fridge_door_joint=0.904 info_keys=['success']
EP 1 success: True; Cumulative success rate: 1.0
success_rate: 1.0
episode_lengths: [172]
first_success_steps: [172]
max_fixture_open: [0.9035059514573512]
```

산출물:

```text
outputs/debug/lerobot_groot_n15_officialrun_OpenFridge_target_seed0_success_probe/videos/0eeb725c-c76b-46cc-9dc9-e12e5f2c9d18_success1.mp4
```

해석: official RoboCasa env가 LeRobot HTTP에 연결됐고, OpenFridge target 성공을 낼 수
있다. RoboCasa Gym wrapper는 `info["success"] = reward > 0`로 설정하며,
OpenFridge의 내부 checker는 `fixture.is_open(th=0.90)`이다. Video frame상으로는 문이
열려 보여도 normalized door joint가 `0.90`을 넘기 전까지는 success가 아니다. 시각적
성공과 RoboCasa success 판정이 어긋나면 `--success-debug-every`를 같이 사용한다.

관련 one-off probe: `OpenCabinet`(예전 naming 기준 `OpenSingleDoor`) target 1ep smoke는
success flag만 실패한 경우로 보이지 않았다. Debug output은 `reward=0.0`,
`success=False`, `max_fixture_open=0.006`에 머물렀고, 해당 run은 RoboCasa open-door
threshold를 물리적으로 넘지 못했다.
산출물: `outputs/debug/lerobot_groot_n15_officialrun_OpenCabinet_target_success_probe/videos/582833ff-b990-4911-ae3c-a787f4b4e064_success0.mp4`.

## 주의점

현재 `src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py` helper는 `eval_seed`,
`filename_prefix`, `one_episode_per_env`를 받지 않는다. N1.5 client는 local helper를
호출하기 전에 지원하지 않는 kwargs를 걸러낸다. 이 checkout에서는 imported helper가
matching parameter를 노출할 때만 `--seed`가 보존된다.

`scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py`의 benchmark-style client는 benchmark
fork와 같은 env id/split convention을 사용한다. 다만 RoboCasa container에 새 dependency를
추가하지 않기 위해 repo-local N1.5 ZMQ msgpack client는 유지한다.
