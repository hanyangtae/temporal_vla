#!/usr/bin/env bash
# strict(drop-aware, 5-phase carved) conceptor steer 큐 (GPU 4,5 / ports 8440,8441).
# 사용자 matrix: strict + 단일 layer / strict + multi layer. 실패=재grasp형(reach 92%)이므로
# phase 후보 = pre-grasp(임계 순간)·reach(실패 dwell). global(strict-fit)은 대조군.
# detached: setsid nohup bash scripts/safe/groot_n15/robocasa/steer/strict_steer_queue_gpu45.sh \
#   > outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/strict_queue.log 2>&1 < /dev/null &
set -uo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RUN="${REPO_ROOT}/scripts/safe/groot_n15/robocasa/steer/steer_eval_multilayer_30.sh"
BASE=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_strict/analysis/conceptor_steering_n15/ppcc_bread

QUEUE=(
  "pre-grasp|4,8,12|st_pregrasp_ml"
  "reach-to-object|4,8,12|st_reach_ml"
  "global|4,8,12|st_global_ml"
  "pre-grasp|4|st_pregrasp_L4"
  "reach-to-object|4|st_reach_L4"
)
for item in "${QUEUE[@]}"; do
  IFS='|' read -r group layers tag <<<"$item"
  echo "[st-queue] $(date '+%F %T') === ${tag} (group=${group} layers=${layers}) ==="
  NPZ_BASE="$BASE" GROUP="$group" LAYERS="$layers" TAG="$tag" \
    GPU_A=4 GPU_B=5 PORT_A=8440 PORT_B=8441 bash "$RUN" \
    || echo "[st-queue] ${tag} FAILED (rc=$?) — continuing"
done
echo "[st-queue] $(date '+%F %T') ALL DONE"
touch "${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/logs/STRICT_QUEUE_DONE"
