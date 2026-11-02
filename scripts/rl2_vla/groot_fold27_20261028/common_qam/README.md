# atomic65 common QAM (공식 RoboCasa atomic65 human pretrain, 100_demos 회원, task-balanced)

독립 실험 `atomic65_common_qam_20261029`. 이전 10-task·target13 run과 별개이고 warm start하지 않음.
정의 단일 출처: `experiment.py`. 학습 recipe·architecture·budget은 `groot_target13_20261022/train_scratch.py`와 동일.

## 계약 (기존 코드에서 확인한 값)
- 관측: frozen GR00T N1.5 ckpt-120000(rev c484448a), `action_decoder_input_last16_mean_denoise0` 1024-d,
  signature `ccbfb51ca01724cb`, 결정 간격 5 env-step, seed=`decision_seed(42, episode_id, step)`, flow 4.
- 행동: 데이터셋 expert action **12-D** (PandaOmron: base_motion4 · control_mode1 · eef_pos3 · eef_rot3 · gripper1,
  raw LeRobot 순서) → 모델 순서 [5..11,0..4] → checkpoint min_max/binary 정규화. 32 중 active12, H16, execute5.
  (8-D가 아님.)
- next obs = 같은 episode의 다음 결정 latent. reward = RL2 성공 expert final3(-1, 마지막 3 step은 0).
- 데이터: task마다 `100_demos` 회원(episodes.jsonl 순서를 random.seed(0)로 shuffle해 앞 100개, RoboCasa 원 함수와 동일 —
  테스트로 확인), stager receipt의 회원 목록과 일치해야 함. 회원 밖 episode는 디스크에 그대로 두고 manifest에만 넣지 않음.
  val은 회원의 10%(sha rank), sampling은 task 균등 후 transition 균등.
- 학습: production prefix QAM, seed 42, batch 256, 500k step, 50k마다 checkpoint+승준 archive(해시 검증),
  선택은 최종 step으로 고정.

## 경로
- staging(compute 사본): `~/.cache/temporal_vla/datasets/atomic65_staging_20261029/robocasa/v1.0/...`
  (컨테이너 경로 `/cache/datasets/atomic65_staging_20261029`)
- work: `outputs/rl2_vla/atomic65_common_qam_work_20261029/{metadata,smoke_latents,latents,scratch_smoke,scratch_run}`
- archive: 승준 `/home/kimseungjun/datasets/temporal_vla_store/checkpoints/atomic65_common_qam_20261029`

## 명령 (repo root에서. `P=.venvs/rl2_iid/bin/python`, `Q=scripts/rl2_vla/groot_fold27_20261028/common_qam`)
```bash
P=.venvs/rl2_iid/bin/python; Q=scripts/rl2_vla/groot_fold27_20261028/common_qam
$P $Q/cli.py status
# 1) staging (CPU; stager가 다운로드 중이면 반복 실행 — 완료된 task는 skip, 대기 task는 waiting으로 보고)
$P $Q/cli.py stage --jobs 2
# 2) smoke (staged 2 task)
$P $Q/cli.py manifest --smoke-tasks AdjustToasterOvenTemperature AdjustWaterTemperature
$P $Q/cli.py preflight --step extract-smoke
GPUS=<g> HARNESS_SESSION=<session> bash $Q/run_extract.sh --smoke
$P $Q/cli.py finalize-extract --smoke
$P $Q/cli.py preflight --step train-smoke
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<g> \
  bash scripts/utils/with_gpu_lease.sh kanu <g> <session> atomic65-qam-smoke -- \
  $P -u $Q/train.py --manifest outputs/rl2_vla/atomic65_common_qam_work_20261029/metadata/manifest_smoke.json \
  --extraction outputs/rl2_vla/atomic65_common_qam_work_20261029/smoke_latents \
  --out outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_smoke --smoke
# 3) 본 실행 (65/65 staged 후)
$P $Q/cli.py manifest                       # 65 task 모두 오류 없어야 manifest.json 생성
$P $Q/cli.py preflight --step extract
GPUS=<g0> HARNESS_SESSION=<session> bash $Q/run_extract.sh --shard 0 --num-shards 2   # GPU당 1 shard
GPUS=<g1> HARNESS_SESSION=<session> bash $Q/run_extract.sh --shard 1 --num-shards 2
$P $Q/cli.py finalize-extract
$P $Q/cli.py preflight --step train
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<g> setsid nohup \
  bash scripts/utils/with_gpu_lease.sh kanu <g> <session> atomic65-qam-train -- \
  $P -u $Q/train.py --manifest outputs/rl2_vla/atomic65_common_qam_work_20261029/metadata/manifest.json \
  --extraction outputs/rl2_vla/atomic65_common_qam_work_20261029/latents \
  --out outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run \
  > outputs/rl2_vla/atomic65_common_qam_work_20261029/train_launch.log 2>&1 < /dev/null &
# 4) 정리 판단 (실행하지 않고 rm 명령만 출력; 조건: 학습 COMPLETE+archive 검증, 원본 receipt 65/65, 다른 consumer 없음)
$P $Q/cli.py cleanup-plan --rehash
```
`run_extract.sh`는 `scripts/utils/groot_launch_guard.sh`(harness lease)를 거쳐 `run_extract.py`를 실행하고,
`run_extract.py`가 `paired_eval.MODEL_IMAGE` 컨테이너를 띄운다. 추출은 episode 단위로 재개된다(기존 npz는 sha 확인 후 skip).
같은 `--out`을 다른 manifest나 mode로 쓰면 contract 불일치로 거부된다.

## 테스트
`.venvs/rl2_iid/bin/python -m unittest scripts/rl2_vla/groot_fold27_20261028/tests/test_common_qam.py -v` (CPU, 18개)
