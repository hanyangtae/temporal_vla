#!/bin/bash
# COAST-faithful native-stack steered-vs-baseline ΔSR eval (GR00T **N1.6** RoboCasa).
#
# N1.6 analogue of coast_native_steer_eval.sh (N1.5). Identical faithful stack:
# native ZMQ serve(native_steered_serve_n16.py) + official robocasa/<Task> env +
# split=pretrain + replan=5 (n-action-steps=5). Driven by the SAME N1.5 client
# (native_official_zmq_eval.py) => directly comparable to the N1.5 n=30 run.
#
# Differences from the N1.5 launcher (the whole reason this file exists):
#   * Serve runs in the **persistent `groot` container** via `docker exec -d`
#     (NOT `docker compose run groot_n15` — that's the N1.5 stack).
#   * N1.6 submodule (Isaac-GR00T) on PYTHONPATH; N1.6 checkpoint; embodiment
#     ROBOCASA_PANDA_OMRON.
#   * Steering on DiT output: --steering-layer is OMITTED (type=int; passing
#     literal "None" crashes; omit => action_head.model output, D=1024 == conceptors).
#   * Conceptor dirs carry a `task_<N>_` prefix (mapped below).
#   * Teardown = pkill the serve python by port INSIDE groot (NEVER `docker rm` groot).
#
# Each task: baseline serve(no steer)->eval, steered serve(task conceptor)->eval, ΔSR.
set -u
REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"; cd "$REPO_HOST"

GPUS=(${GPUS:-0 1 2 3 4})
NEP=${EVAL_EPISODES:-30}
NAS=${N_ACTION_STEPS:-5}
SPLIT=${SPLIT:-pretrain}
BASE_PORT=${BASE_PORT:-5570}
CKPT=${CKPT:-/cache/checkpoints/Abhi03/grootn16_robocasa365_multitask_learning/checkpoint-120000}
EMB=${EMBODIMENT_TAG:-ROBOCASA_PANDA_OMRON}
CONC_ROOT=${CONC_ROOT:-/temporal_vla/outputs/eval/robocasa/groot_n16/target_atomic_seen18_ckpt120000_robocasa365_100ep/analysis/conceptor_steering/coast}
A=${STEER_ALPHA:-0.1}; B=${STEER_BETA:-0.1}; PATHWAY=${STEER_PATHWAY:-dit}
RUN=${RUN_ID:-coast_native_steer_n16_n30}
OUT=outputs/eval/robocasa/groot_n15/$RUN; mkdir -p "$OUT"
RES=$OUT/results.tsv; printf "task\tbase_n\tbase_sr\tsteer_sr\tdelta\n" > "$RES"
TASKS=(${TASKS_OVERRIDE:-CloseFridge CoffeeSetupMug OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove TurnOnElectricKettle})

# Task -> conceptor dir (N1.6 dirs have a task_<N>_ prefix).
declare -A CONC_DIR=(
  [CloseFridge]=task_1_CloseFridge
  [CoffeeSetupMug]=task_3_CoffeeSetupMug
  [OpenDrawer]=task_6_OpenDrawer
  [OpenStandMixerHead]=task_7_OpenStandMixerHead
  [PickPlaceCounterToCabinet]=task_8_PickPlaceCounterToCabinet
  [PickPlaceCounterToStove]=task_9_PickPlaceCounterToStove
  [TurnOnElectricKettle]=task_15_TurnOnElectricKettle
)

# N1.6 serve PYTHONPATH (Isaac-GR00T submodule, NOT N1.5).
PYP_SRV=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla:/temporal_vla/scripts/utils
# N1.5 client PYTHONPATH (unchanged from N1.5 stack).
PYP_CLI=/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla
SERVE_PY=/temporal_vla/scripts/safe/groot_n15/robocasa/serve/native_steered_serve_n16.py
NPROC=${#GPUS[@]}

start_serve() {  # $1=gpu $2=port $3=extra(steering args or "")
  docker exec -d \
    -e CUDA_VISIBLE_DEVICES=$1 \
    -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    groot bash -lc \
    "cd /temporal_vla && PYTHONPATH=$PYP_SRV python3 $SERVE_PY --model-path $CKPT --embodiment-tag $EMB --port $2 $3 > /tmp/ns16_$2.log 2>&1"
  # groot container has no ss/netstat; detect readiness from the serve log line
  # ("[serve] N15CompatInferenceServer ready on ...") printed right after bind.
  for i in $(seq 1 80); do
    docker exec groot bash -lc "grep -q 'N15CompatInferenceServer ready' /tmp/ns16_$2.log 2>/dev/null" && return 0
    docker exec groot bash -lc "grep -qiE 'error|traceback' /tmp/ns16_$2.log 2>/dev/null" && {
      echo "[serve ERR port $2]"; docker exec groot bash -lc "tail -12 /tmp/ns16_$2.log"; return 1; }
    sleep 6
  done
  echo "[serve TIMEOUT port $2]"; docker exec groot bash -lc "tail -12 /tmp/ns16_$2.log"; return 1
}
stop_serve() {  # $1=port
  docker exec groot bash -lc "pkill -f 'native_steered_serve_n16.py.*--port $1'" >/dev/null 2>&1 || true
}
eval_sr() {  # $1=port $2=task -> prints SR float
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYP_CLI" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py \
      --host 127.0.0.1 --port $1 --task $2 --split $SPLIT --n-episodes $NEP --n-action-steps $NAS 2>&1 \
    | grep -oE "success rate: +[0-9.]+" | tail -1 | grep -oE "[0-9.]+"
}

cleanup() { for si in "${!GPUS[@]}"; do stop_serve $((BASE_PORT+si)); done; }
trap cleanup EXIT INT TERM

worker() {
  local si=$1 gpu=${GPUS[$si]} port=$((BASE_PORT+si))
  for t in "${!TASKS[@]}"; do
    [ $((t % NPROC)) -ne $si ] && continue
    local task=${TASKS[$t]} conc=$CONC_ROOT/${CONC_DIR[${TASKS[$t]}]}/conceptors.npz
    echo "[gpu$gpu] $task baseline serve..."
    stop_serve $port; sleep 1
    start_serve $gpu $port "" || continue
    local bsr=$(eval_sr $port $task)
    stop_serve $port; sleep 2
    echo "[gpu$gpu] $task steered serve..."
    start_serve $gpu $port "--steering-npz $conc --steering-alpha $A --steering-beta $B --steering-pathway $PATHWAY" || continue
    local ssr=$(eval_sr $port $task)
    stop_serve $port; sleep 2
    local d=$(python3 -c "print(round(${ssr:-0}-${bsr:-0},3))")
    printf "%s\t%s\t%s\t%s\t%s\n" "$task" "$NEP" "${bsr:-NA}" "${ssr:-NA}" "$d" >> "$RES"
    echo "[gpu$gpu] $task base=$bsr steer=$ssr ΔSR=$d"
  done
}
for si in "${!GPUS[@]}"; do worker $si & done
wait
echo "[done] -> $RES"; column -t -s $'\t' "$RES"
