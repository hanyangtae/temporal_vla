#!/usr/bin/env bash
# Layer ablation 큐 (GPU 0,1 / ports 8420,8421): global conceptor, 조합별 30 rollout 순차.
# global-multi(4,8,12)=+0.20 이 어느 layer 조합에서 오는지 분해.
# detached: setsid nohup bash scripts/safe/groot_n15/robocasa/steer/ablation_queue_gpu01.sh \
#   > outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/ablation_queue.log 2>&1 < /dev/null &
set -uo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RUN="${REPO_ROOT}/scripts/safe/groot_n15/robocasa/steer/steer_eval_multilayer_30.sh"

# combo|tag (순차; resume-safe)
QUEUE=(
  "0,2,4,8,10,12,15|mlg_all7"
  "4,8|mlg_L4_8"
  "8,12|mlg_L8_12"
  "8|mlg_L8only"
  "12|mlg_L12only"
)
for item in "${QUEUE[@]}"; do
  IFS='|' read -r layers tag <<<"$item"
  echo "[queue] $(date '+%F %T') === ${tag} (layers=${layers}) ==="
  GROUP=global LAYERS="$layers" TAG="$tag" GPU_A=0 GPU_B=1 PORT_A=8420 PORT_B=8421 \
    bash "$RUN" || echo "[queue] ${tag} FAILED (rc=$?) — continuing"
done
echo "[queue] $(date '+%F %T') ALL DONE"
touch "${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/logs/ABLATION_QUEUE_DONE"
