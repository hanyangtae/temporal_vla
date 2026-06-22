#!/bin/bash
# Phase 4: COAST-faithful steered-vs-baseline ΔSR eval (GR00T N1.5 RoboCasa).
#
# Per task, uses COAST's published global (layer, alpha, beta) from
# coast_n15_hyperparams.json (Table 9). For each task on a round-robin GPU:
#   1) baseline serve (no steering) -> collect EVAL_EPISODES held-out rollouts -> SR
#   2) steered serve (--steering-npz <task conceptor at its layer> --steering-layer L
#      --steering-alpha a --steering-beta b) -> same held-out seeds -> SR
# SR is counted from pkl filenames (succ1/total) — the runner status is unreliable.
# Held-out eval seeds (EVAL_SEED_START, default 400000) are disjoint from the
# collection seeds (100000+), so no train/test leakage.
#
# Requires Phase 3 task-level conceptors at:
#   <CONC_ROOT>/dit_layer<L>/coast/<task>/conceptors.npz   (alpha<a>_C_steer)
set -u

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
REPO=/temporal_vla
cd "${REPO_HOST}"

GPUS=(${GPUS:-0 1 2 3})
EVAL_EPISODES=${EVAL_EPISODES:-30}
EVAL_SEED_START=${EVAL_SEED_START:-400000}
BASE_PORT=${BASE_PORT:-8420}
PROFILE=${PROFILE:-${REPO}/configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml}
# HYP is read by the HOST python (hp helper), so it must be a host path, not /temporal_vla.
HYP=${HYP:-${REPO_HOST}/scripts/safe/groot_n15/robocasa/steer/coast_n15_hyperparams.json}
CONC_ROOT=${CONC_ROOT:?set CONC_ROOT to Phase3 task-level conceptor root}
RUN_ID=${RUN_ID:-coast_faithful_steer_eval}
OUT_ROOT=${REPO}/outputs/eval/robocasa/groot_n15/${RUN_ID}
OUT_ROOT_HOST=outputs/eval/robocasa/groot_n15/${RUN_ID}
CAPTURE_LAYERS=${CAPTURE_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
LEROBOT_CONTAINER=${LEROBOT_CONTAINER:-lerobot}
ROBOCASA_CONTAINER=${ROBOCASA_CONTAINER:-robocasa}
N_ACTION_STEPS=${N_ACTION_STEPS:-5}   # COAST replan_steps=5 (predicts 16, executes 5)
LOG=/tmp/coast_steer_eval
PY=${PY:-~/miniconda3/envs/hyundai_aigs/bin/python}
NPROC=${#GPUS[@]}

TASKS=(CloseFridge CoffeeSetupMug OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet
       PickPlaceCounterToStove TurnOnElectricKettle)
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then TASKS=( ${TASKS_OVERRIDE} ); fi

PYPATH="${REPO}/src/policies/Isaac-GR00T:${REPO}/src/benchmarks/robocasa:${REPO}/src/benchmarks/robosuite:${REPO}"
mkdir -p "${OUT_ROOT_HOST}" "${LOG}"
docker exec "${LEROBOT_CONTAINER}" bash -lc "mkdir -p ${LOG}" 2>/dev/null
RESULTS="${OUT_ROOT_HOST}/results.tsv"
printf "task\tlayer\talpha\tbeta\tbase_n\tbase_succ\tbase_sr\tsteer_n\tsteer_succ\tsteer_sr\tdelta_sr\n" > "${RESULTS}"

hp() {  # $1=task $2=field  -> value from JSON
  ${PY} -c "import json,sys; d=json.load(open('${HYP}'))['per_task']['$1']; print(d['$2'])"
}

cleanup() {
  for idx in "${!GPUS[@]}"; do
    local port=$((BASE_PORT + idx))
    docker exec "${LEROBOT_CONTAINER}" bash -lc "pkill -f 'lerobot.py.*--port ${port}'" 2>/dev/null
    docker exec "${ROBOCASA_CONTAINER}" bash -lc "pkill -f 'vla-server http://127.0.0.1:${port}'" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

start_serve() {  # $1=gpu $2=port $3=extra steering args (or empty)
  local gpu=$1 port=$2 extra=$3
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${LEROBOT_CONTAINER}" bash -lc \
    "cd ${REPO} && python scripts/serve/lerobot.py --profile ${PROFILE} --host '*' --port ${port} \
       --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAPTURE_LAYERS} ${extra} \
       > ${LOG}/serve_${port}.log 2>&1"
  for i in $(seq 1 90); do
    curl -fsS --max-time 4 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
    sleep 6
  done
  echo "[serve] port ${port} TIMEOUT"; docker exec "${LEROBOT_CONTAINER}" bash -lc "tail -20 ${LOG}/serve_${port}.log"; return 1
}

stop_serve() {  # $1=port
  docker exec "${LEROBOT_CONTAINER}" bash -lc "pkill -f 'lerobot.py.*--port $1'" 2>/dev/null; sleep 3
}

collect_eval() {  # $1=port $2=task $3=env $4=outdir
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="${PYPATH}" "${ROBOCASA_CONTAINER}" \
    python ${REPO}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:$1" --task "$2" --env-name "$3" \
      --output-dir "$4" --task-id 0 --episode-start-idx 0 --n-episodes ${EVAL_EPISODES} \
      --seed ${EVAL_SEED_START} --n-action-steps ${N_ACTION_STEPS} \
      >> ${LOG}/eval_$2_$1.log 2>&1
}

sr_of() {  # $1=dir -> "n succ"
  local n s
  n=$(ls "$1"/*.pkl 2>/dev/null | wc -l)
  s=$(ls "$1"/*--succ1.pkl 2>/dev/null | wc -l)
  echo "${n} ${s}"
}

worker() {
  local server_idx=$1
  local gpu=${GPUS[$server_idx]}; local port=$((BASE_PORT + server_idx))
  for t in "${!TASKS[@]}"; do
    [ $((t % NPROC)) -ne "${server_idx}" ] && continue
    local task="${TASKS[$t]}"; local env="robocasa_panda_omron/${task}_PandaOmron_Env"
    local L A B conc base_dir steer_dir
    L=$(hp "${task}" layer); A=$(hp "${task}" alpha); B=$(hp "${task}" beta)
    conc="${CONC_ROOT}/dit_layer${L}/coast/${task}/conceptors.npz"
    base_dir="${OUT_ROOT}/baseline/${task}"; steer_dir="${OUT_ROOT}/steered/${task}"
    local base_dir_host="${OUT_ROOT_HOST}/baseline/${task}" steer_dir_host="${OUT_ROOT_HOST}/steered/${task}"
    mkdir -p "${base_dir_host}" "${steer_dir_host}"

    echo "[eval gpu=${gpu}] ${task} L=${L} a=${A} b=${B}"
    # baseline
    start_serve "${gpu}" "${port}" "" || continue
    collect_eval "${port}" "${task}" "${env}" "${base_dir}"
    stop_serve "${port}"
    # steered
    start_serve "${gpu}" "${port}" "--steering-npz ${conc} --steering-pathway dit --steering-layer ${L} --steering-alpha ${A} --steering-beta ${B}" || continue
    collect_eval "${port}" "${task}" "${env}" "${steer_dir}"
    stop_serve "${port}"

    # http_feature_collect's _resolve_task_dir returns base_dir unchanged when its name
    # already == task, so pkls land directly in base_dir_host (no extra /<task> level).
    read bn bs < <(sr_of "${base_dir_host}")
    read sn ss < <(sr_of "${steer_dir_host}")
    local bsr ssr dsr
    bsr=$(${PY} -c "print(round(${bs}/max(${bn},1),4))")
    ssr=$(${PY} -c "print(round(${ss}/max(${sn},1),4))")
    dsr=$(${PY} -c "print(round(${ss}/max(${sn},1) - ${bs}/max(${bn},1),4))")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${task}" "${L}" "${A}" "${B}" "${bn}" "${bs}" "${bsr}" "${sn}" "${ss}" "${ssr}" "${dsr}" >> "${RESULTS}"
    echo "[eval gpu=${gpu}] ${task} baseline=${bsr} steered=${ssr} dSR=${dsr}"
  done
}

for idx in "${!GPUS[@]}"; do worker "${idx}" & done
wait

echo "[done] results -> ${RESULTS}"
column -t -s $'\t' "${RESULTS}"
echo "=== mean ΔSR ==="
${PY} -c "
import csv
rows=list(csv.DictReader(open('${RESULTS}'),delimiter='\t'))
d=[float(r['delta_sr']) for r in rows]
print('per-task ΔSR:', {r['task']:r['delta_sr'] for r in rows})
print('mean ΔSR =', round(sum(d)/len(d),4) if d else 'n/a', ' (COAST +0.16)')
"
