#!/bin/bash
# Phase 1 학습 + 평가 비교 파이프라인
#
# 데이터 세팅 (VITA 재현):
#   pick-and-place 5000개 풀 → 3273(2986 train + 287 val) + 나머지 OOD 100개
#
# 실험 조건 (inner_model=linear 고정):
#   1. episodic + mean (VITA 원본)
#   2. episodic + concat
#   3. window + mean (phase1_step50k 재현)
#   4. window + concat
#
# 사용법:
#   docker exec -it lerobot bash /temporal_vla/scripts/run_phase1_compare.sh [실험번호]
#   예: bash run_phase1_compare.sh 1     # episodic + mean만 실행
#       bash run_phase1_compare.sh all   # 전체 실행

set -e
cd /temporal_vla

# ── 공통 설정 ──
INNER_MODEL="linear"
TRAIN_EPISODES=2986
VAL_EPISODES=287
BATCH_SIZE=32
LR=1e-4
LAMBDA_SELF=0.5
WINDOW_SIZE=8
MAX_WINDOWS=8
EMBED_CACHE="data/bridge_v2_pnp_5000_clip_embeddings.pt"
TASK_KEYWORDS="put place pick"

# episodic은 epoch 기반, window는 step 기반
EPISODIC_EPOCHS=5
WINDOW_STEPS=50000

run_experiment() {
    local MODE=$1        # episodic or window
    local DISS=$2        # mean or concat
    local LABEL="${MODE}_${DISS}_${INNER_MODEL}"
    local DATE_STR=$(date +%Y%m%d_%H%M)
    local EVAL_DIR="outputs/eval_results/phase1_${LABEL}_${DATE_STR}"

    echo ""
    echo "=========================================="
    echo " Phase 1: ${MODE} + ${DISS} (${INNER_MODEL})"
    echo "=========================================="

    # ── 학습 ──
    TRAIN_ARGS=(
        --dataset_mode ${MODE}
        --dissimilarity_mode ${DISS}
        --inner_model_type ${INNER_MODEL}
        --batch_size ${BATCH_SIZE}
        --lr ${LR}
        --lambda_self ${LAMBDA_SELF}
        --window_size ${WINDOW_SIZE}
        --max_windows_per_episode ${MAX_WINDOWS}
        --train_episodes ${TRAIN_EPISODES}
        --val_episodes ${VAL_EPISODES}
        --embed_cache_path ${EMBED_CACHE}
        --task_keywords ${TASK_KEYWORDS}
        --save_dir checkpoints/phase1
        --save_suffix "${LABEL}"
        --wandb_run_name "phase1_${LABEL}"
    )

    if [ "${MODE}" = "episodic" ]; then
        TRAIN_ARGS+=(--n_epochs ${EPISODIC_EPOCHS})
    else
        TRAIN_ARGS+=(--total_steps ${WINDOW_STEPS} --val_interval 1000 --save_interval 10000)
    fi

    python scripts/train_phase1_predictor.py "${TRAIN_ARGS[@]}"

    # ── 체크포인트 찾기 ──
    LATEST_CKPT_DIR=$(ls -dt checkpoints/phase1/*_${LABEL} 2>/dev/null | head -1)
    FINAL_CKPT="${LATEST_CKPT_DIR}/phase1_final.pt"

    if [ ! -f "${FINAL_CKPT}" ]; then
        echo "ERROR: ${FINAL_CKPT} not found"
        return 1
    fi
    echo "Checkpoint: ${FINAL_CKPT}"

    # ── 평가 (n_samples=100, n_videos=10) ──
    python scripts/eval_phase1_predictor.py \
        --ckpt "${FINAL_CKPT}" \
        --inner_model_type ${INNER_MODEL} \
        --train_episodes ${TRAIN_EPISODES} \
        --val_episodes ${VAL_EPISODES} \
        --task_keywords ${TASK_KEYWORDS} \
        --n_samples 100 \
        --save_dir "${EVAL_DIR}" \
        --save_video \
        --n_videos 10

    echo ""
    echo "── ${LABEL} results ──"
    cat "${EVAL_DIR}/metrics.txt"
    echo ""
}

# ── 실행 ──
EXPERIMENT="${1:-all}"

case "${EXPERIMENT}" in
    1) run_experiment episodic mean ;;
    2) run_experiment episodic concat ;;
    3) run_experiment window mean ;;
    4) run_experiment window concat ;;
    all)
        run_experiment episodic mean
        run_experiment episodic concat
        run_experiment window mean
        run_experiment window concat
        ;;
    *)
        echo "Usage: $0 [1|2|3|4|all]"
        echo "  1: episodic + mean (VITA 원본)"
        echo "  2: episodic + concat"
        echo "  3: window + mean (step50k 재현)"
        echo "  4: window + concat"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo " All experiments done!"
echo "=========================================="
