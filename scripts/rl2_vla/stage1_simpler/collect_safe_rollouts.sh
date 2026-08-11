#!/bin/bash
# SAFE 재학습용 rollout 수집 (우리 플랫폼) — 저자 collect_rollouts_for_safe_training.sh 와
# 동일 구성(rephrase arm + log_safe_training_data), lane=(task×seed) 큐 병렬판.
#
# 사용: bash collect_safe_rollouts.sh "<gpu 목록>"   예) "1 2 3"
set -uo pipefail

GPUS=${1:?"gpu list"}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RL2="$REPO_ROOT/RL2-VLA"
OUT="$RL2/safe_rollouts"          # 저자 restructure 스크립트가 기대하는 위치(레포 상대)
QUEUE="$OUT/queue.txt"
LOCK="$OUT/queue.lock"
mkdir -p "$OUT"

TASKS=(simpler_put_eggplant_in_basket simpler_spoon_on_towel simpler_stack_cube simpler_carrot_on_plate)
SEEDS=(42 0 7)
: > "$QUEUE"
for s in "${SEEDS[@]}"; do for t in "${TASKS[@]}"; do echo "$t $s" >> "$QUEUE"; done; done

pop_job() {
    flock "$LOCK" bash -c '
        q="$1"; [ -s "$q" ] || exit 1
        head -1 "$q"; tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
    ' _ "$QUEUE"
}

worker() {
    local gpu=$1
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate rl2
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa WANDB_MODE=offline PRISMATIC_DATA_ROOT=.
    export PYTHONPATH="$RL2:$RL2/RL2_CoVer_VLA:${PYTHONPATH:-}"
    cd "$RL2/RL2_CoVer_VLA/simpler"
    while true; do
        local job; job=$(pop_job) || { echo "[gpu$gpu] 큐 소진"; return 0; }
        read -r task seed <<< "$job"
        local sentinel="$OUT/DONE_${task}_s${seed}"
        [ -f "$sentinel" ] && { echo "[gpu$gpu] skip $task s$seed"; continue; }
        echo "[gpu$gpu] START $task s$seed $(date +%H:%M)"
        CUDA_VISIBLE_DEVICES=$gpu python run_simpler_eval_with_openpi.py \
            --task_suite_name "$task" \
            --lang_transform_type rephrase \
            --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
            --num_trials_per_task 100 \
            --use_failure_prediction False \
            --lang_rephrase_num_prefail 8 --action_samples_prefail 5 --composed_samples_prefail 0 \
            --use_verifier True --critic cover \
            --seed "$seed" \
            --local_log_dir "$OUT" \
            --wandb_project Rephrase-Safe-Collect \
            --log_safe_training_data True \
            > "$OUT/collect_${task}_s${seed}.log" 2>&1 \
            && touch "$sentinel"
        echo "[gpu$gpu] END   $task s$seed rc=$? $(date +%H:%M)"
    done
}

for g in $GPUS; do worker "$g" & done
wait
touch "$OUT/COLLECT_ALL_DONE"
echo "[collect] 전체 완료 $(date +%F\ %H:%M)"
