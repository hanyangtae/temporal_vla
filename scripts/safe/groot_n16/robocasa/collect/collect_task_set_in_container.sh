#!/usr/bin/env bash
# Collect GR00T N1.6 RoboCasa SAFE rollouts from inside the robocasa container.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_config.sh"
source "${SCRIPT_DIR}/task_sets.sh"

CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-/temporal_vla}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5557}"
EPISODE_START_IDX="${EPISODE_START_IDX:-0}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
TASK_SET="${TASK_SET:-safe_seen6}"
if [[ -z "${RUN_ID:-}" ]]; then
    if [[ "${TASK_SET}" == "safe_seen6" ]]; then
        RUN_ID="n16_task_set_safe_flow_features_smoke"
    else
        RUN_ID="n16_${TASK_SET}_safe_flow_features"
    fi
fi
RUN_ROOT="${RUN_ROOT:-${CONTAINER_REPO_ROOT}/outputs/eval/robocasa/groot_n16/${ROBOCASA_SAFE_RUN_ID}}"

load_robocasa_collection_task_set "${TASK_SET}"
SEED_START="${SEED_START:-${ROBOCASA_SEED_START_FOR_TASK_SET}}"
ROBOCASA_ENV_SOURCE="$(normalize_robocasa_collection_env_source "${ROBOCASA_ENV_SOURCE:-${ROBOCASA_ENV_SOURCE_FOR_TASK_SET}}")"
if [[ -z "${OUT_ROOT:-}" ]]; then
    if [[ "${TASK_SET}" == "safe_seen6" ]]; then
        OUT_ROOT="${RUN_ROOT}/experiments/collection_smoke/rollouts_${RUN_ID}"
    else
        OUT_ROOT="${RUN_ROOT}/raw_rollouts"
    fi
fi

mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/collection_summary.tsv"
if [[ ! -f "${SUMMARY}" ]]; then
    printf "task_id\ttask\tepisode_idx\tseed\texit_code\tpkl\n" > "${SUMMARY}"
fi

echo "Output: ${OUT_ROOT}"
echo "Server: ${HOST}:${PORT}"
echo "Task set: ${TASK_SET} (${#TASKS[@]} tasks)"
echo "RoboCasa env source: ${ROBOCASA_ENV_SOURCE}"
echo "Episodes per task: ${EPISODES_PER_TASK}, episode_idx: ${EPISODE_START_IDX}..$((EPISODE_START_IDX + EPISODES_PER_TASK - 1)), seeds: ${SEED_START}..$((SEED_START + EPISODES_PER_TASK - 1))"

for task_id in "${!TASKS[@]}"; do
    task="${TASKS[${task_id}]}"
    env_name="robocasa_panda_omron/${task}_PandaOmron_Env"
    task_dir="${OUT_ROOT}/${task}"
    mkdir -p "${task_dir}"

    for local_episode_idx in $(seq 0 $((EPISODES_PER_TASK - 1))); do
        episode_idx=$((EPISODE_START_IDX + local_episode_idx))
        seed=$((SEED_START + local_episode_idx))
        existing="$(find "${task_dir}" -maxdepth 1 \( -type f -o -type l \) -name "task${task_id}--ep${episode_idx}--succ*.pkl" | head -n 1 || true)"
        if [[ -n "${existing}" ]]; then
            existing_mp4="${existing%.pkl}.mp4"
            if [[ -s "${existing_mp4}" ]]; then
                printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${task_id}" "${task}" "${episode_idx}" "${seed}" "SKIP" "${existing}" | tee -a "${SUMMARY}"
                continue
            fi
            echo "Re-collecting incomplete output: ${existing}" >&2
        fi

        echo
        echo "== task${task_id} ${task} ep ${episode_idx} seed ${seed} =="

        if ROBOCASA_ENV_SOURCE="${ROBOCASA_ENV_SOURCE}" \
            MUJOCO_GL=egl \
            PYOPENGL_PLATFORM=egl \
            PYTHONPATH="${CONTAINER_REPO_ROOT}/src/policies/Isaac-GR00T:${CONTAINER_REPO_ROOT}/src/benchmarks/robocasa:${CONTAINER_REPO_ROOT}/src/benchmarks/robosuite:${CONTAINER_REPO_ROOT}" \
            python "${CONTAINER_REPO_ROOT}/scripts/safe/groot_n16/robocasa/collect/collect_rollout.py" \
                --policy-client-host "${HOST}" \
                --policy-client-port "${PORT}" \
                --env-name "${env_name}" \
                --robocasa-env-source "${ROBOCASA_ENV_SOURCE}" \
                --output-dir "${task_dir}" \
                --task-id "${task_id}" \
                --episode-start-idx "${episode_idx}" \
                --n-episodes 1 \
                --seed "${seed}" \
                2>&1 | tee "${task_dir}/ep${episode_idx}.log"; then
            exit_code=0
        else
            exit_code=$?
        fi

        pkl="$(find "${task_dir}" -maxdepth 1 \( -type f -o -type l \) -name "task${task_id}--ep${episode_idx}--succ*.pkl" | head -n 1 || true)"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${task_id}" "${task}" "${episode_idx}" "${seed}" "${exit_code}" "${pkl}" | tee -a "${SUMMARY}"

        if [[ "${exit_code}" != "0" ]]; then
            echo "ERROR: collection failed for ${task} episode ${episode_idx}" >&2
            exit "${exit_code}"
        fi
    done
done

echo
echo "Done. Summary: ${SUMMARY}"
