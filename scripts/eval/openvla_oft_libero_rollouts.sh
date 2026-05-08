#!/usr/bin/env bash
# OpenVLA-OFT × LIBERO 4-suite 롤아웃 영상 수집.
#
# 통일 API 경로 사용:
#   serve/openvla_oft.py (port 8400) ← 서버
#   eval/libero.py                    ← env loop, VLAClient 로 /act 호출
#
# 동작:
#   - 4 suite (spatial / object / goal / 10) 각각 별도 profile 로 서버 재기동
#   - max_steps = DEFAULT × MAX_STEPS_MULT (default 3)
#   - success_deadline = DEFAULT (원래 max_steps) → deadline 초과 done 은 실패 처리
#   - 실패 영상만 outputs/rollouts/libero/openvla_oft/<TS>/<suite>/ 로 저장
#
# 실행 (openvla_oft 컨테이너 안):
#   docker compose run --rm openvla_oft bash /temporal_vla/scripts/eval/openvla_oft_libero_rollouts.sh
#
# 환경 변수 override:
#   NUM_TRIALS=20 MAX_STEPS_MULT=2 SUITES="libero_spatial libero_object" bash ...

set -euo pipefail

# ── 기본값 ──────────────────────────────────────────────────────────
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_STEPS_MULT="${MAX_STEPS_MULT:-3}"
SEED="${SEED:-7}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

TIMESTAMP=$(date +%y%m%d%H%M%S)
ROLLOUT_ROOT="/temporal_vla/outputs/rollouts/libero/openvla_oft/${TIMESTAMP}"
LOG_DIR="${ROLLOUT_ROOT}/_logs"

OFT_ROOT="/temporal_vla/src/policies/openvla-oft"
SERVE_SCRIPT="/temporal_vla/scripts/serve/openvla_oft.py"
EVAL_SCRIPT="/temporal_vla/scripts/eval/libero.py"
PROFILE_DIR="/temporal_vla/configs/checkpoints"
PORT=8400
SERVER_URL="http://localhost:${PORT}"

# Suite 별 표준 max_steps (eval/libero.py 의 DEFAULT_MAX_STEPS 와 일치해야 함)
declare -A DEFAULT_MAX_STEPS=(
    [libero_spatial]=220
    [libero_object]=280
    [libero_goal]=300
    [libero_10]=520
)

mkdir -p "${LOG_DIR}"

# ── LIBERO 환경 셋업 (기존 openvla_oft_libero.sh 와 동일) ───────────
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

if ! python -c "import prismatic" 2>/dev/null; then
    pip install -e "${OFT_ROOT}"
fi

# ── 서버 헬퍼 ──────────────────────────────────────────────────────
wait_server() {
    local max_wait=300
    local elapsed=0
    while [ ${elapsed} -lt ${max_wait} ]; do
        if curl -sf "${SERVER_URL}/health" 2>/dev/null | grep -q '"status":"ok"'; then
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    echo "[ERR] server not ready after ${max_wait}s" >&2
    return 1
}

stop_server() {
    if [ -n "${SERVE_PID:-}" ]; then
        kill ${SERVE_PID} 2>/dev/null || true
        wait ${SERVE_PID} 2>/dev/null || true
        SERVE_PID=""
    fi
}
trap stop_server EXIT

# ── 메인 루프 ──────────────────────────────────────────────────────
echo "=========================================="
echo " OpenVLA-OFT × LIBERO Rollouts"
echo "  timestamp     : ${TIMESTAMP}"
echo "  num_trials    : ${NUM_TRIALS}"
echo "  max_steps_mult: ${MAX_STEPS_MULT}"
echo "  suites        : ${SUITES}"
echo "  rollout_root  : ${ROLLOUT_ROOT}"
echo "=========================================="

for SUITE in ${SUITES}; do
    PROFILE="${PROFILE_DIR}/openvla_oft__moojink_${SUITE}.yaml"
    if [ ! -f "${PROFILE}" ]; then
        echo "[ERR] profile not found: ${PROFILE}" >&2
        continue
    fi

    DEFAULT="${DEFAULT_MAX_STEPS[${SUITE}]}"
    MAX_STEPS=$((DEFAULT * MAX_STEPS_MULT))

    echo ""
    echo "── suite=${SUITE} (default=${DEFAULT}, max_steps=${MAX_STEPS}) ──"

    # serve 기동
    SERVE_LOG="${LOG_DIR}/serve_${SUITE}.log"
    python "${SERVE_SCRIPT}" --profile "${PROFILE}" --port "${PORT}" \
        > "${SERVE_LOG}" 2>&1 &
    SERVE_PID=$!
    echo "[serve] pid=${SERVE_PID} log=${SERVE_LOG}"

    if ! wait_server; then
        echo "[ERR] serve failed for ${SUITE}; tail ${SERVE_LOG}:" >&2
        tail -n 50 "${SERVE_LOG}" >&2 || true
        stop_server
        continue
    fi

    # eval 실행
    EVAL_LOG="${LOG_DIR}/eval_${SUITE}.log"
    python "${EVAL_SCRIPT}" \
        --task-suite "${SUITE}" \
        --num-trials "${NUM_TRIALS}" \
        --max-steps "${MAX_STEPS}" \
        --success-deadline "${DEFAULT}" \
        --server-url "${SERVER_URL}" \
        --video-dir "${ROLLOUT_ROOT}" \
        --seed "${SEED}" \
        2>&1 | tee "${EVAL_LOG}"

    stop_server
done

echo ""
echo "[DONE] rollouts saved under ${ROLLOUT_ROOT}"
