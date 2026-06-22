#!/bin/bash
# COAST-faithful native-stack steered-vs-baseline ΔSR eval (GR00T N1.5 RoboCasa).
# 우리가 COAST에 맞춘 4축: native ZMQ serve(native_steered_serve.py) + 공식 robocasa/<Task>
# env + split=pretrain + replan=5. layer/α/β = COAST 공개값 L10/0.1/0.1 (uniform).
# 각 task: baseline serve(steer 없음)→eval, steered serve(task별 L10 conceptor)→eval, ΔSR.
set -u
REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"; cd "$REPO_HOST"

GPUS=(${GPUS:-0 1 2 3})
NEP=${EVAL_EPISODES:-10}
NAS=${N_ACTION_STEPS:-5}
SPLIT=${SPLIT:-pretrain}
BASE_PORT=${BASE_PORT:-5560}
CKPT=${CKPT:-/cache/datasets/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/14895998fe7c8f8f2441cc8957ec2c510302758b/gr00t_n1-5/multitask_learning/checkpoint-120000}
DC=robocasa_n15_panda_omron_data_config:RobocasaPandaOmron10TaskDataConfig
CONC_ROOT=${CONC_ROOT:-/temporal_vla/outputs/eval/robocasa/groot_n15/coast_faithful_7task_30ep/conceptor_taskfit/dit_layer10/coast}
L=${STEER_LAYER:-10}; A=${STEER_ALPHA:-0.1}; B=${STEER_BETA:-0.1}
RUN=${RUN_ID:-coast_native_steer_replan5}
OUT=outputs/eval/robocasa/groot_n15/$RUN; mkdir -p "$OUT"
RES=$OUT/results.tsv; printf "task\tbase_n\tbase_sr\tsteer_sr\tdelta\n" > "$RES"
TASKS=(${TASKS_OVERRIDE:-CloseFridge CoffeeSetupMug OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove TurnOnElectricKettle})
PYP_SRV=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla
PYP_CLI=/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla:/temporal_vla/configs/policies
NPROC=${#GPUS[@]}

start_serve() {  # $1=gpu $2=port $3=extra(steering args or "") $4=container_name
  docker rm -f "$4" >/dev/null 2>&1
  docker compose run -d --name "$4" -e CUDA_VISIBLE_DEVICES=$1 groot_n15 bash -lc \
    "cd /temporal_vla && PYTHONPATH=$PYP_SRV python3 scripts/safe/groot_n15/robocasa/serve/native_steered_serve.py --model-path $CKPT --embodiment-tag new_embodiment --data-config $DC --port $2 $3 > /tmp/ns_$2.log 2>&1" >/dev/null 2>&1
  for i in $(seq 1 50); do
    ss -ltn 2>/dev/null | grep -q ":$2 " && return 0
    docker logs "$4" 2>&1 | grep -qiE "error|traceback" && { echo "[serve ERR $4]"; docker logs "$4" 2>&1 | tail -8; return 1; }
    sleep 6
  done
  echo "[serve TIMEOUT $4]"; return 1
}
eval_sr() {  # $1=port $2=task -> prints SR float
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYP_CLI" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py \
      --host 127.0.0.1 --port $1 --task $2 --split $SPLIT --n-episodes $NEP --n-action-steps $NAS 2>&1 \
    | grep -oE "success rate: +[0-9.]+" | tail -1 | grep -oE "[0-9.]+"
}

cleanup() { for si in "${!GPUS[@]}"; do docker rm -f "ns_b_$((BASE_PORT+si))" "ns_s_$((BASE_PORT+si))" >/dev/null 2>&1; done; }
trap cleanup EXIT INT TERM

worker() {
  local si=$1 gpu=${GPUS[$si]} port=$((BASE_PORT+si))
  for t in "${!TASKS[@]}"; do
    [ $((t % NPROC)) -ne $si ] && continue
    local task=${TASKS[$t]} conc=$CONC_ROOT/${TASKS[$t]}/conceptors.npz
    echo "[gpu$gpu] $task baseline serve..."
    start_serve $gpu $port "" "ns_b_$port" || continue
    local bsr=$(eval_sr $port $task)
    docker rm -f "ns_b_$port" >/dev/null 2>&1; sleep 2
    echo "[gpu$gpu] $task steered serve..."
    start_serve $gpu $port "--steering-npz $conc --steering-layer $L --steering-alpha $A --steering-beta $B" "ns_s_$port" || continue
    local ssr=$(eval_sr $port $task)
    docker rm -f "ns_s_$port" >/dev/null 2>&1; sleep 2
    local d=$(~/miniconda3/envs/hyundai_aigs/bin/python -c "print(round(${ssr:-0}-${bsr:-0},3))")
    printf "%s\t%s\t%s\t%s\t%s\n" "$task" "$NEP" "${bsr:-NA}" "${ssr:-NA}" "$d" >> "$RES"
    echo "[gpu$gpu] $task base=$bsr steer=$ssr ΔSR=$d"
  done
}
for si in "${!GPUS[@]}"; do worker $si & done
wait
echo "[done] -> $RES"; column -t -s $'\t' "$RES"
