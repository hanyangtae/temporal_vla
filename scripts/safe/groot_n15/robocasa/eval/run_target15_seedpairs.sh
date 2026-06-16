#!/usr/bin/env bash
# Run GR00T N1.5 RoboCasa target atomic 15-task eval with exact seed pairs.
#
# Default plan:
#   - 15 target atomic-seen PandaOmron tasks
#   - seeds 101..110
#   - base and tuned clients in parallel
#   - each model uses n_envs=2, n_episodes=2 per seed pair
#   - counted episodes are the first episode from each env only

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_config.sh"

REPO_ROOT="${REPO_ROOT:-${GROOT_N15_REPO_ROOT}}"
CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-${GROOT_N15_CONTAINER_REPO_ROOT}}"
ROBOCASA_CONTAINER="${ROBOCASA_CONTAINER:-robocasa}"

BASE_HOST="${BASE_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-5557}"
TUNED_HOST="${TUNED_HOST:-166.104.35.50}"
TUNED_PORT="${TUNED_PORT:-5556}"

MODE="${MODE:-both}"
RUN_ID="${RUN_ID:-${GROOT_N15_TARGET15_RUN_ID}}"
SEED_START="${SEED_START:-101}"
SEED_COUNT="${SEED_COUNT:-10}"
N_ENVS="${N_ENVS:-2}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
STEPS_PER_RENDER="${STEPS_PER_RENDER:-2}"
VIDEO_FPS="${VIDEO_FPS:-10}"
OUT_ROOT="${OUT_ROOT:-${GROOT_N15_TARGET15_RUN_ROOT}}"
OUT_ROOT_CONTAINER="${OUT_ROOT_CONTAINER:-${GROOT_N15_TARGET15_RUN_ROOT_CONTAINER}}"
TASKS_CSV="${TASKS_CSV:-}"

TARGET15_TASKS=(
    "CloseFridge"
    "CloseToasterOvenDoor"
    "CoffeeSetupMug"
    "OpenCabinet"
    "OpenDrawer"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "PickPlaceDrawerToCounter"
    "PickPlaceSinkToCounter"
    "PickPlaceToasterToCounter"
    "SlideDishwasherRack"
    "TurnOffStove"
    "TurnOnElectricKettle"
    "TurnOnMicrowave"
    "TurnOnSinkFaucet"
)

usage() {
    cat <<USAGE
Usage:
  RUN_ID=my_eval bash scripts/safe/groot_n15/robocasa/eval/run_target15_seedpairs.sh [options]

Options:
  --mode both|base|tuned          Models to evaluate. Default: ${MODE}
  --run-id ID                     Output run id. Default: timestamp
  --seed-start N                  First seed. Default: ${SEED_START}
  --seed-count N                  Number of seeds. Default: ${SEED_COUNT}
  --n-envs N                      Seeds per wave per model. Default: ${N_ENVS}
  --tasks CSV                     Task names or full env ids, comma-separated.
  --base-host HOST                Base server host. Default: ${BASE_HOST}
  --base-port PORT                Base server port. Default: ${BASE_PORT}
  --tuned-host HOST               Tuned server host. Default: ${TUNED_HOST}
  --tuned-port PORT               Tuned server port. Default: ${TUNED_PORT}
  --out-root PATH                 Host output root.
  -h, --help                      Show this help.

Environment overrides:
  ROBOCASA_CONTAINER, CONTAINER_REPO_ROOT, GROOT_N15_* run_config variables,
  N_ACTION_STEPS, MAX_EPISODE_STEPS, STEPS_PER_RENDER, VIDEO_FPS
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            OUT_ROOT="${GROOT_N15_OUT_ROOT}/target15_seedpairs_${RUN_ID}"
            OUT_ROOT_CONTAINER="${GROOT_N15_OUT_ROOT_CONTAINER}/target15_seedpairs_${RUN_ID}"
            shift 2
            ;;
        --seed-start)
            SEED_START="$2"
            shift 2
            ;;
        --seed-count)
            SEED_COUNT="$2"
            shift 2
            ;;
        --n-envs)
            N_ENVS="$2"
            shift 2
            ;;
        --tasks)
            TASKS_CSV="$2"
            shift 2
            ;;
        --base-host)
            BASE_HOST="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --tuned-host)
            TUNED_HOST="$2"
            shift 2
            ;;
        --tuned-port)
            TUNED_PORT="$2"
            shift 2
            ;;
        --out-root)
            OUT_ROOT="$2"
            OUT_ROOT_CONTAINER="${CONTAINER_REPO_ROOT}${OUT_ROOT#${REPO_ROOT}}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${MODE}" in
    both|base|tuned) ;;
    *)
        echo "ERROR: --mode must be one of: both, base, tuned" >&2
        exit 2
        ;;
esac

if (( N_ENVS < 1 )); then
    echo "ERROR: --n-envs must be >= 1" >&2
    exit 2
fi

env_name_for_task() {
    local task="$1"
    if [[ "${task}" == */* ]]; then
        printf "%s" "${task}"
    else
        printf "robocasa_panda_omron/%s_PandaOmron_Env" "${task}"
    fi
}

task_slug() {
    local env_name="$1"
    local slug="${env_name##*/}"
    slug="${slug%_PandaOmron_Env}"
    printf "%s" "${slug}"
}

if [[ -n "${TASKS_CSV}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
else
    TASKS=("${TARGET15_TASKS[@]}")
fi

mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/summary.tsv"
printf "model\ttask\tenv_name\tseed_start\tseed_end\texit_code\tsuccess_rate\tout_dir\n" > "${SUMMARY}"

echo "Output: ${OUT_ROOT}"
echo "Mode: ${MODE}, seeds: ${SEED_START}..$((SEED_START + SEED_COUNT - 1)), n_envs: ${N_ENVS}"
echo "Base: ${BASE_HOST}:${BASE_PORT}, tuned: ${TUNED_HOST}:${TUNED_PORT}"

run_eval() {
    local model="$1"
    local host="$2"
    local port="$3"
    local env_name="$4"
    local task="$5"
    local seed="$6"
    local envs_this="$7"
    local out_host="${OUT_ROOT}/${model}/${task}/seed_${seed}"
    local out_container="${OUT_ROOT_CONTAINER}/${model}/${task}/seed_${seed}"

    mkdir -p "${out_host}"
    printf "%s\n" \
        "model=${model}" \
        "host=${host}" \
        "port=${port}" \
        "env_name=${env_name}" \
        "seed=${seed}" \
        "n_envs=${envs_this}" \
        "n_episodes=${envs_this}" \
        "n_action_steps=${N_ACTION_STEPS}" \
        "max_episode_steps=${MAX_EPISODE_STEPS}" \
        "steps_per_render=${STEPS_PER_RENDER}" \
        "video_fps=${VIDEO_FPS}" \
        "video_file_prefix=${task}" \
        "no_video_outcome_suffix=true" > "${out_host}/run_args.txt"

    docker exec \
        -e MUJOCO_GL=egl \
        -e PYTHONPATH="${CONTAINER_REPO_ROOT}/src/policies/Isaac-GR00T:${CONTAINER_REPO_ROOT}/src/benchmarks/robocasa:${CONTAINER_REPO_ROOT}/src/benchmarks/robosuite:${CONTAINER_REPO_ROOT}" \
        "${ROBOCASA_CONTAINER}" \
        bash -lc "set -o pipefail; mkdir -p '${out_container}/videos' && python '${CONTAINER_REPO_ROOT}/scripts/safe/groot_n15/robocasa/eval/native_zmq_eval.py' \
            --policy-client-host '${host}' \
            --policy-client-port '${port}' \
            --env-name '${env_name}' \
            --n-episodes '${envs_this}' \
            --n-envs '${envs_this}' \
            --n-action-steps '${N_ACTION_STEPS}' \
            --max-episode-steps '${MAX_EPISODE_STEPS}' \
            --steps-per-render '${STEPS_PER_RENDER}' \
            --video-fps '${VIDEO_FPS}' \
            --seed '${seed}' \
            --video-file-prefix '${task}' \
            --no-video-outcome-suffix \
            --one-episode-per-env \
            --video-dir '${out_container}/videos' 2>&1 | tee '${out_container}/run.log'"
}

record_result() {
    local model="$1"
    local env_name="$2"
    local task="$3"
    local seed="$4"
    local envs_this="$5"
    local exit_code="$6"
    local out_host="${OUT_ROOT}/${model}/${task}/seed_${seed}"
    local seed_end=$((seed + envs_this - 1))
    local success_rate="NA"

    if [[ -f "${out_host}/run.log" ]]; then
        success_rate="$(grep -E 'success rate:' "${out_host}/run.log" | tail -n 1 | awk '{print $3}')"
        if [[ -z "${success_rate}" ]]; then
            success_rate="NA"
        fi
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${model}" "${task}" "${env_name}" "${seed}" "${seed_end}" "${exit_code}" "${success_rate}" "${out_host}" \
        | tee -a "${SUMMARY}"
}

for raw_task in "${TASKS[@]}"; do
    env_name="$(env_name_for_task "${raw_task}")"
    task="$(task_slug "${env_name}")"
    seed_limit=$((SEED_START + SEED_COUNT))

    seed="${SEED_START}"
    while (( seed < seed_limit )); do
        remaining=$((seed_limit - seed))
        envs_this="${N_ENVS}"
        if (( remaining < envs_this )); then
            envs_this="${remaining}"
        fi

        echo
        echo "== ${task} seeds ${seed}..$((seed + envs_this - 1)) =="

        base_pid=""
        tuned_pid=""

        if [[ "${MODE}" == "both" || "${MODE}" == "base" ]]; then
            run_eval "base" "${BASE_HOST}" "${BASE_PORT}" "${env_name}" "${task}" "${seed}" "${envs_this}" &
            base_pid="$!"
        fi
        if [[ "${MODE}" == "both" || "${MODE}" == "tuned" ]]; then
            run_eval "tuned" "${TUNED_HOST}" "${TUNED_PORT}" "${env_name}" "${task}" "${seed}" "${envs_this}" &
            tuned_pid="$!"
        fi

        if [[ -n "${base_pid}" ]]; then
            if wait "${base_pid}"; then
                base_status=0
            else
                base_status=$?
            fi
            record_result "base" "${env_name}" "${task}" "${seed}" "${envs_this}" "${base_status}"
        fi

        if [[ -n "${tuned_pid}" ]]; then
            if wait "${tuned_pid}"; then
                tuned_status=0
            else
                tuned_status=$?
            fi
            record_result "tuned" "${env_name}" "${task}" "${seed}" "${envs_this}" "${tuned_status}"
        fi

        seed=$((seed + envs_this))
    done
done

echo
echo "Done. Summary: ${SUMMARY}"
