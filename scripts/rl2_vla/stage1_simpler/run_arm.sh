#!/bin/bash
# RL2-VLA SIMPLER 축소 재현 러너 (Stage 1b) — arm 하나를 GPU 하나에서 실행.
#
# 사용: bash run_arm.sh <arm> <gpu> [suite] [seed] [trials] [alpha]
#   arm   : vanilla | rephrase | always | adaptive
#   suite : IID | OOD (기본 OOD — 논문 Fig 8 대조)
#   alpha : adaptive 전용 CP significance level. 미지정 시 해당 seed의 top-1 사용.
#
# 플래그는 RL2-VLA/RL2_CoVer_VLA/simpler/bashes/eval_*.sh 원본과 동일하게 유지
# (차이: 절대경로, WANDB offline, INFERENCE_ROOT 경로 정정).
# 완료 시 로그 디렉토리에 DONE sentinel 기록.
set -euo pipefail

ARM=${1:?arm}
GPU=${2:?gpu}
SUITE=${3:-OOD}
SEED=${4:-42}
TRIALS=${5:-50}
ALPHA_ARG=${6:-}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RL2="$REPO_ROOT/RL2-VLA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa WANDB_MODE=offline PRISMATIC_DATA_ROOT=.
export PYTHONPATH="$RL2:$RL2/RL2_CoVer_VLA:${PYTHONPATH:-}"

QAM_CKPT="$RL2/third_party/qam/exp/SAVED/rl2-vla-qam-bridge/rl2_vla_qam_bridge_500k.pkl"
SAFE_DIR_IID="$RL2/third_party/SAFE/scripts/batch_training/logs/SAVED/rl2_pi0_bridge_safe_ckpt_per_task_cp"
SAFE_DIR_OOD="${SAFE_DIR_OVERRIDE:-$RL2/third_party/SAFE/scripts/batch_training/logs/SAVED/rl2_pi0_bridge_safe_ckpt_combined_cp}"
# SAFE_DIR_OVERRIDE: 재학습 SAFE ckpt 디렉토리로 교체할 때 사용 (OOD suite 한정)
CKPT="juexzz/INTACT-pi0-finetune-bridge"
LANE="$ARM"
[ -n "$ALPHA_ARG" ] && LANE="${ARM}_a${ALPHA_ARG}"
[ -n "${LANE_TAG:-}" ] && LANE="${LANE}_${LANE_TAG}"
LOG_DIR="$RL2/experiments/stage1b_${SUITE}_seed${SEED}/${LANE}"
mkdir -p "$LOG_DIR"

if [[ "$SUITE" == "IID" ]]; then
    TASKS=(simpler_put_eggplant_in_basket simpler_spoon_on_towel simpler_stack_cube simpler_carrot_on_plate)
    TASKWISE=True; SAFE_DIR="$SAFE_DIR_IID"; ALPHA_JSON="$RL2/RL2_CoVer_VLA/simpler/bashes/rl2_cp_alphas_per_task.json"
else
    TASKS=(simpler_orange_juice_on_plate simpler_spoon_on_towel_google simpler_tape_measure_in_basket simpler_toy_dinosaur_on_towel)
    TASKWISE=False; SAFE_DIR="$SAFE_DIR_OOD"; ALPHA_JSON="$RL2/RL2_CoVer_VLA/simpler/bashes/rl2_cp_alphas_combined.json"
fi

cd "$RL2/RL2_CoVer_VLA/simpler"

for TASK in "${TASKS[@]}"; do
    COMMON=(--task_suite_name "$TASK" --lang_transform_type rephrase
            --pretrained_checkpoint "$CKPT" --num_trials_per_task "$TRIALS"
            --use_verifier True --critic cover --seed "$SEED"
            --local_log_dir "$LOG_DIR" --wandb_project "RL2-repro-$ARM")
    case "$ARM" in
      vanilla)
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction False \
            --lang_rephrase_num_prefail 1 --action_samples_prefail 1 --composed_samples_prefail 0
        ;;
      rephrase)
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction False \
            --lang_rephrase_num_prefail 8 --action_samples_prefail 5 --composed_samples_prefail 0
        ;;
      always)
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction False --qam_ckpt "$QAM_CKPT" \
            --lang_rephrase_num_prefail 8 --action_samples_prefail 1 --composed_samples_prefail 5
        ;;
      adaptive)
        if [ -n "$ALPHA_ARG" ]; then
            ALPHA="$ALPHA_ARG"
        elif [[ "$SUITE" == "IID" ]]; then
            ALPHA=$(python -c "import json;print(json.load(open('$ALPHA_JSON'))['alpha']['$TASK']['$SEED'][0])")
        else
            ALPHA=$(python -c "import json;print(json.load(open('$ALPHA_JSON'))['alpha']['combined']['$SEED'][0])")
        fi
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction True --use_taskwise_cp_band "$TASKWISE" \
            --failure_checkpoint_dir "$SAFE_DIR" --failure_cp_alpha "$ALPHA" \
            --qam_ckpt "$QAM_CKPT" \
            --lang_rephrase_num_prefail 8 --action_samples_prefail 5 --composed_samples_prefail 0 \
            --lang_rephrase_num 8 --action_samples 1 --composed_samples 5
        ;;
      *) echo "unknown arm: $ARM"; exit 1;;
    esac
done

touch "$LOG_DIR/DONE"
echo "[run_arm] $LANE ($SUITE, seed $SEED) 완료 → $LOG_DIR/DONE"
