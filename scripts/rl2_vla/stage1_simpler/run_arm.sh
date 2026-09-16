#!/bin/bash
# RL2-VLA SIMPLER 축소 재현 러너 (Stage 1b) — arm 하나를 GPU 하나에서 실행.
#
# 사용: bash run_arm.sh <arm> <gpu> [suite] [seed] [trials] [alpha]
#   arm   : vanilla | rephrase | always | adaptive | resample | resample_gated
#           | compose | compose_gated  (rephrase 없이 QAM 합성만 — 상시 / 발화 후만)
#   resample*: 순수 noise 재샘플 + verifier (rephrase 1종·QAM 無). N=RESAMPLE_N(기본 40
#   = RL2 기본 후보수 8×5와 동일 예산). resample_gated는 SAFE 발화 후만 재샘플(prefail=vanilla).
#   suite : IID | OOD (기본 OOD — 논문 Fig 8 대조)
#   alpha : gated arms의 CP significance level (기본 0.2).
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RL2="$REPO_ROOT/RL2-VLA"
GATED_ARM=0
case "$ARM" in
  adaptive|resample_gated|compose_gated|compose1_gated) GATED_ARM=1 ;;
esac
if (( GATED_ARM )); then
    : "${SAFE_DIR_OVERRIDE:?gated arms require SAFE_DIR_OVERRIDE with temporal_vla provenance}"
    SAFE_DIR_OVERRIDE="$(cd "$SAFE_DIR_OVERRIDE" && pwd)"
    SAFE_MANIFEST="${SAFE_PROVENANCE_OVERRIDE:-$SAFE_DIR_OVERRIDE/provenance.json}"
    python3 "$REPO_ROOT/scripts/rl2_vla/stage2_cluster_reward/safe_provenance.py" \
        --safe-dir "$SAFE_DIR_OVERRIDE" --manifest "$SAFE_MANIFEST"
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl2

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa WANDB_MODE=offline PRISMATIC_DATA_ROOT=.
export PYTHONPATH="$RL2:$RL2/RL2_CoVer_VLA:${PYTHONPATH:-}"

QAM_CKPT="${QAM_CKPT_OVERRIDE:-$RL2/third_party/qam/exp/SAVED/rl2-vla-qam-bridge/rl2_vla_qam_bridge_500k.pkl}"
CKPT="juexzz/INTACT-pi0-finetune-bridge"
LANE="$ARM"
[ -n "$ALPHA_ARG" ] && LANE="${ARM}_a${ALPHA_ARG}"
[ -n "${LANE_TAG:-}" ] && LANE="${LANE}_${LANE_TAG}"
LOG_DIR="$RL2/experiments/stage1b_${SUITE}_seed${SEED}/${LANE}"
mkdir -p "$LOG_DIR"

if [[ "$SUITE" == "IID" ]]; then
    TASKS=(simpler_put_eggplant_in_basket simpler_spoon_on_towel simpler_stack_cube simpler_carrot_on_plate)

else
    TASKS=(simpler_orange_juice_on_plate simpler_spoon_on_towel_google simpler_tape_measure_in_basket simpler_toy_dinosaur_on_towel)

fi
TASKWISE="${SAFE_TASKWISE:-False}"
SAFE_DIR="${SAFE_DIR_OVERRIDE:-}"

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
        ALPHA="${ALPHA_ARG:-0.2}"
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction True --use_taskwise_cp_band "$TASKWISE" \
            --failure_checkpoint_dir "$SAFE_DIR" --failure_cp_alpha "$ALPHA" \
            --qam_ckpt "$QAM_CKPT" \
            --lang_rephrase_num_prefail 8 --action_samples_prefail 5 --composed_samples_prefail 0 \
            --lang_rephrase_num 8 --action_samples 1 --composed_samples 5
        ;;
      resample)
        # 상시 재샘플: 원문 지시 1종 × N noise 후보 → CoVer best-of-N (합성·rephrase 없음)
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction False \
            --lang_rephrase_num_prefail 1 --action_samples_prefail "${RESAMPLE_N:-40}" --composed_samples_prefail 0
        ;;
      resample_gated)
        # SAFE 발화 후만 재샘플: prefail=vanilla(1×1), 발화 시 1×N noise 재샘플 + verifier
        ALPHA="${ALPHA_ARG:-0.2}"
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction True --use_taskwise_cp_band "$TASKWISE" \
            --failure_checkpoint_dir "$SAFE_DIR" --failure_cp_alpha "$ALPHA" \
            --lang_rephrase_num_prefail 1 --action_samples_prefail 1 --composed_samples_prefail 0 \
            --lang_rephrase_num 1 --action_samples "${RESAMPLE_N:-40}" --composed_samples 0
        ;;
      compose)
        # rephrase 없이 QAM 합성만: 원문 지시 1종 × 합성 후보 5 → CoVer 선별 (상시)
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction False --qam_ckpt "$QAM_CKPT" \
            --lang_rephrase_num_prefail 1 --action_samples_prefail 1 --composed_samples_prefail 5
        ;;
      compose_gated)
        # rephrase 없이, SAFE 발화 후만 QAM 합성 (prefail=vanilla 1×1)
        ALPHA="${ALPHA_ARG:-0.2}"
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction True --use_taskwise_cp_band "$TASKWISE" \
            --failure_checkpoint_dir "$SAFE_DIR" --failure_cp_alpha "$ALPHA" \
            --qam_ckpt "$QAM_CKPT" \
            --lang_rephrase_num_prefail 1 --action_samples_prefail 1 --composed_samples_prefail 0 \
            --lang_rephrase_num 1 --action_samples 1 --composed_samples 5
        ;;
      compose1_gated)
        # 선별·rephrase 없는 단일 합성: 평시 vanilla(1x1), SAFE 발화 시에만 QAM 합성 후보 1개
        # (composed_samples=1 -> verifier 선택이 항등). w = MERGE_W 고정(기본 0.5),
        # -1 이면 스텝마다 N(0.5,0.25)에서 추첨.
        ALPHA="${ALPHA_ARG:-0.2}"
        CUDA_VISIBLE_DEVICES=$GPU python run_simpler_eval_with_openpi.py "${COMMON[@]}" \
            --use_failure_prediction True --use_taskwise_cp_band "$TASKWISE" \
            --failure_checkpoint_dir "$SAFE_DIR" --failure_cp_alpha "$ALPHA" \
            --qam_ckpt "$QAM_CKPT" --merge_rel_weight "${MERGE_W:-0.5}" \
            --use_rephrased_latents_for_qam False \
            --lang_rephrase_num_prefail 1 --action_samples_prefail 1 --composed_samples_prefail 0 \
            --lang_rephrase_num 1 --action_samples 1 --composed_samples 1
        ;;
      *) echo "unknown arm: $ARM"; exit 1;;
    esac
done

touch "$LOG_DIR/DONE"
echo "[run_arm] $LANE ($SUITE, seed $SEED) 완료 → $LOG_DIR/DONE"
