#!/usr/bin/env bash
# Phase-event-aligned N1.5 RoboCasa collection: 4 instruction cells, 1 fixed scenario_seed
# per cell, 15 rollouts/cell with VARYING inference_seed (same scene, different policy
# samples → succ/fail mix). One worker = one (GPU,serve-port). Jobs are round-robin split
# across WORKERS by global job index, so each worker owns a disjoint subset of the 60
# (cell, rollout) jobs and the three workers collect in parallel.
#
# Activation↔phase alignment: n_action_steps=5 (predict-16/execute-5). One get_action =
# one SAFE feature record = one labeler.step() = 5 env-steps. http_feature_collect.py writes
# feature_phases (aligned 1:1 to hidden_states), phase_timeline, event_steps, event_order.
#
# inference_seed = rollout_idx * SEED_STRIDE. The stride must exceed the per-episode
# get_action count (~144 at execute-5/720-step) because each call uses inference_seed +
# len(records); a stride of 1 makes consecutive rollouts near-identical trajectories.
#
# Usage (run one per worker, detached):
#   WORKER=0 N_WORKERS=3 PORT=8400 collect_phase_event_4cell.sh
#   WORKER=1 N_WORKERS=3 PORT=8401 collect_phase_event_4cell.sh
#   WORKER=2 N_WORKERS=3 PORT=8402 collect_phase_event_4cell.sh
set -euo pipefail

WORKER="${WORKER:?set WORKER (0-based)}"
N_WORKERS="${N_WORKERS:-3}"
PORT="${PORT:?set PORT}"
CONTAINER="${CONTAINER:-robocasa}"
N_ROLLOUTS="${N_ROLLOUTS:-15}"
SEED_STRIDE="${SEED_STRIDE:-1000}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
RUN_ID="${RUN_ID:-phase_event_aligned_4cell}"
# This launcher runs on the HOST and docker-execs the collector. Host-side file ops
# (resume-check, sentinel) use HOST paths; --output-dir passed into the container uses
# the container mount (/temporal_vla).
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
HOST_OUT_ROOT="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/${RUN_ID}"
CONTAINER_OUT_DIR="/temporal_vla/outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts"
SERVER="http://127.0.0.1:${PORT}"
mkdir -p "${HOST_OUT_ROOT}/logs"

# cell rows: cell_id|task|env_name|cell_index|scenario_seed|canonical_instruction
CELLS=(
  "ppcc_bread|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|100084|Pick the bread from the counter and place it in the cabinet."
  "ppcc_potato|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|4|200010|Pick the potato from the counter and place it in the cabinet."
  "ppcs_apple|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100050|Pick the apple from the plate and place it in the pan."
  "ppcs_onion|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|0|100000|Pick the onion from the plate and place it in the pan."
)

PYTHONPATH_IN="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

job=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r cell_id task env_name cell_index seed instr <<<"$cell"
  for idx in $(seq 0 $((N_ROLLOUTS - 1))); do
    if [ $((job % N_WORKERS)) -eq "$WORKER" ]; then
      inf=$((idx * SEED_STRIDE))
      # skip if this rollout pkl already exists (resume-safe; host-side check)
      if ls "${HOST_OUT_ROOT}/raw_rollouts/${task}/${cell_id}/task${cell_index}--ep${idx}--succ"*.pkl >/dev/null 2>&1; then
        echo "[w${WORKER}] skip existing ${cell_id} ep${idx}"
      else
        echo "[w${WORKER}] collect ${cell_id} ep${idx} seed=${seed} inf=${inf}"
        docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYTHONPATH_IN" "$CONTAINER" \
          python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
          --vla-server "$SERVER" --task "$task" --env-name "$env_name" \
          --output-dir "$CONTAINER_OUT_DIR" --cell-id "$cell_id" --cell-index "$cell_index" \
          --canonical-instruction "$instr" \
          --episode-start-idx "$idx" --n-episodes 1 \
          --seed "$seed" --inference-seed "$inf" \
          --n-action-steps "$N_ACTION_STEPS" --max-episode-steps "$MAX_EPISODE_STEPS" \
          --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
          | grep -E "^wrote|Error|Traceback" || true
      fi
    fi
    job=$((job + 1))
  done
done
echo "[w${WORKER}] DONE"
touch "${HOST_OUT_ROOT}/logs/worker${WORKER}.done"
