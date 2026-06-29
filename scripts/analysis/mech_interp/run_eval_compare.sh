#!/usr/bin/env bash
# LIBERO-10, 5 trial/task, baseline/fast/slow, GPU 0/1/2 병렬. coef=6. up 제외.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate openvla-interp
cd /home/dongkyu/pkt_ws/mechanistic-steering-vlas/openvla
SC=/tmp/claude-1004/-home-dongkyu-pkt-ws-temporal-vla/80d5a2ad-9c6d-4d8d-a42d-f046988e25a8/scratchpad
DICT=libero_experiments/configs/interventions/dictionaries_full.yaml
CFG=libero_experiments/configs/runs/example_run.yaml
MAXT=10; TRIALS=5; COEF=6.0
run(){ local gpu=$1 tag=$2; shift 2; local log="$SC/t5_${tag}.log"
  echo "[$tag] start $(date) gpu=$gpu" > "$log"
  MAX_TASKS=$MAXT LIBERO_CONFIG_PATH=../utils/libero_config CUDA_VISIBLE_DEVICES=$gpu \
    python libero_experiments/scripts/run_eval.py --config "$CFG" \
      --override env.num_trials_per_task=$TRIALS "$@" --interventions "$DICT" >> "$log" 2>&1
  echo "[$tag] EXIT=$? $(date)" >> "$log"; }
run 0 baseline --override intervention.enabled=false &
run 1 fast --override intervention.enabled=true --override intervention.dict_name=fast_10 --override intervention.coef=$COEF &
run 2 slow --override intervention.enabled=true --override intervention.dict_name=slow_10 --override intervention.coef=$COEF &
wait
echo "ALL DONE $(date)"
