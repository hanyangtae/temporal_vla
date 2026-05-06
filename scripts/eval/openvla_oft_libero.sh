#!/usr/bin/env bash
# OpenVLA-OFT LIBERO 평가 스크립트
#
# openvla_oft 컨테이너에서 실행:
#   docker compose run --rm openvla_oft bash /temporal_vla/scripts/eval/openvla_oft_libero.sh
#
# 환경 변수로 task suite 변경 가능:
#   TASK_SUITE=libero_object docker compose run --rm openvla_oft bash /temporal_vla/scripts/eval/openvla_oft_libero.sh
#
# 개별 suite 체크포인트 사용 시:
#   CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-spatial TASK_SUITE=libero_spatial \
#     docker compose run --rm openvla_oft bash /temporal_vla/scripts/eval/openvla_oft_libero.sh

set -euo pipefail

# ── 기본값 ──────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"

OFT_ROOT="/temporal_vla/src/policies/openvla-oft"
TIMESTAMP=$(date +%y%m%d%H%M%S)
OUTPUT_DIR="/temporal_vla/outputs/eval/libero/openvla_oft/${TIMESTAMP}"

# ── LIBERO 설치 확인 ────────────────────────────────────────────────
# editable install 이 user-site 의 .pth 에만 등록되어 sys.path 에 잡히지 않는 컨테이너가 있음.
# LIBERO 소스가 mount 된 디렉토리를 직접 PYTHONPATH 에 추가하면 둘 다 커버.
export PYTHONPATH="/temporal_vla/src/benchmarks/LIBERO:${PYTHONPATH:-}"

if ! python -c "import libero" 2>/dev/null; then
    echo "[INFO] LIBERO not installed. Installing..."
    if [ ! -d "/temporal_vla/src/benchmarks/LIBERO" ]; then
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
            /temporal_vla/src/benchmarks/LIBERO
    fi
    pip install -e /temporal_vla/src/benchmarks/LIBERO
    pip install -r "${OFT_ROOT}/experiments/robot/libero/libero_requirements.txt"
fi

# ── LIBERO config (~/.libero/config.yaml) 사전 작성 ─────────────────
# 첫 import 시 stdin 으로 dataset 경로를 묻기 때문에 background 실행에서 EOFError.
LIBERO_CONFIG_DIR="${HOME}/.libero"
mkdir -p "${LIBERO_CONFIG_DIR}"
if [ ! -f "${LIBERO_CONFIG_DIR}/config.yaml" ]; then
    LIBERO_PKG_DIR="/temporal_vla/src/benchmarks/LIBERO/libero/libero"
    cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
benchmark_root: ${LIBERO_PKG_DIR}
bddl_files: ${LIBERO_PKG_DIR}/bddl_files
init_states: ${LIBERO_PKG_DIR}/init_files
datasets: ${LIBERO_PKG_DIR}/../datasets
assets: ${LIBERO_PKG_DIR}/assets
YAML
fi

# ── openvla-oft editable install 확인 ────────────────────────────────
if ! python -c "import prismatic" 2>/dev/null; then
    echo "[INFO] Installing openvla-oft in editable mode..."
    pip install -e "${OFT_ROOT}"
fi

# ── 평가 실행 ────────────────────────────────────────────────────────
echo "=========================================="
echo " OpenVLA-OFT LIBERO Evaluation"
echo "  checkpoint : ${CHECKPOINT}"
echo "  task_suite : ${TASK_SUITE}"
echo "  num_trials : ${NUM_TRIALS}"
echo "  seed       : ${SEED}"
echo "  output     : ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

cd "${OFT_ROOT}"

python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name "${TASK_SUITE}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --center_crop True \
    --use_l1_regression True \
    --use_proprio True \
    --num_images_in_input 2 \
    --num_open_loop_steps 8 \
    --local_log_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --use_wandb False

echo ""
echo "[DONE] Results saved to ${OUTPUT_DIR}"
