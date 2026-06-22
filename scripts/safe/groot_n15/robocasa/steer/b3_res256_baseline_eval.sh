#!/bin/bash
# B3 resolution probe: baseline-only (no steering) N1.5 native eval at 256x256.
# Tests whether feeding 256-px frames (COAST's resolution) raises baseline SR
# toward COAST vs our 224 baseline. 3 biggest-gap tasks, n=20, parallel GPU 0,1,2.
set -u
REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"; cd "$REPO_HOST"

GPUS=(0 1 2)
TASKS=(CloseFridge TurnOnElectricKettle OpenDrawer)
NEP=20
NAS=5
SPLIT=pretrain
BASE_PORT=5580
CKPT=/cache/datasets/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/14895998fe7c8f8f2441cc8957ec2c510302758b/gr00t_n1-5/multitask_learning/checkpoint-120000
DC=robocasa_n15_panda_omron_data_config_256:RobocasaPandaOmron10TaskDataConfig256
PYP_SRV=/temporal_vla/configs/policies:/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla
PYP_CLI=/temporal_vla/src/policies/Isaac-GR00T-N1.5:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla

OUT=outputs/eval/robocasa/groot_n15/b3_res256_baseline; mkdir -p "$OUT"
RES=$OUT/results.tsv; printf "task\tbase_n\tbase_sr_256\n" > "$RES"

cname() { echo "ns_b3_$1"; }  # $1=port
cleanup() { for si in "${!GPUS[@]}"; do docker rm -f "$(cname $((BASE_PORT+si)))" >/dev/null 2>&1; done; }
trap cleanup EXIT INT TERM

start_serve() {  # $1=gpu $2=port $3=container_name
  docker rm -f "$3" >/dev/null 2>&1
  docker compose run -d --name "$3" -e CUDA_VISIBLE_DEVICES=$1 -e B3_VERIFY_RESIZE=1 groot_n15 bash -lc \
    "cd /temporal_vla && PYTHONPATH=$PYP_SRV python3 scripts/safe/groot_n15/robocasa/serve/native_steered_serve.py --model-path $CKPT --embodiment-tag new_embodiment --data-config $DC --port $2 > /tmp/ns_b3_$2.log 2>&1" >/dev/null 2>&1
  for i in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":$2 " && return 0
    docker logs "$3" 2>&1 | grep -qiE "error|traceback" && { echo "[serve ERR $3]"; docker logs "$3" 2>&1 | tail -20; return 1; }
    sleep 6
  done
  echo "[serve TIMEOUT $3]"; docker logs "$3" 2>&1 | tail -20; return 1
}
eval_sr() {  # $1=port $2=task
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYP_CLI" robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py \
      --host 127.0.0.1 --port $1 --task $2 --split $SPLIT --n-episodes $NEP --n-action-steps $NAS 2>&1 \
    | grep -oE "success rate: +[0-9.]+" | tail -1 | grep -oE "[0-9.]+"
}

worker() {
  local si=$1 gpu=${GPUS[$si]} port=$((BASE_PORT+si)) task=${TASKS[$si]} cn
  cn=$(cname $port)
  echo "[gpu$gpu] $task : starting 256 baseline serve (port $port)..."
  start_serve $gpu $port "$cn" || { printf "%s\t%s\t%s\n" "$task" "$NEP" "SERVE_FAIL" >> "$RES"; return; }
  # confirm 256 actually applied in serve log
  docker logs "$cn" 2>&1 | grep -q "B3-VERIFY.*height=256" && echo "[gpu$gpu] $task : 256 confirmed in serve log" || echo "[gpu$gpu] $task : WARN 256 verify line not found"
  echo "[gpu$gpu] $task : running eval n=$NEP..."
  local sr=$(eval_sr $port $task)
  docker rm -f "$cn" >/dev/null 2>&1
  printf "%s\t%s\t%s\n" "$task" "$NEP" "${sr:-NA}" >> "$RES"
  echo "[gpu$gpu] $task : base_sr_256=$sr"
}

echo "[b3] start $(date -Iseconds)"
for si in "${!GPUS[@]}"; do worker $si & done
wait
echo "[b3] done $(date -Iseconds) -> $RES"
column -t -s $'\t' "$RES"
