#!/usr/bin/env bash
# GR00T N1.5 (vanilla 또는 +TTT) × RoboCasa target/atomic 15-task closed-loop 평가.
#
# 분리된 모드:
#   1) server         — groot_serve_n1d5 컨테이너에서 추론 서버 기동 (port 8510)
#   2) health         — /health 응답 확인
#   3) client TASK    — 단일 태스크 N_EPISODES rollout (robocasa 컨테이너)
#   4) client-batch   — target 15-task 전체 순차 rollout
#   5) smoke-test     — 서버 백그라운드 + CloseFridge 1 rollout (단일 머신 테스트)
#   6) help           — 사용법
#
# 환경 변수:
#   PROFILE          체크포인트 프로파일 YAML
#   PORT             서버 포트 (default 8510)
#   VLA_SERVER_URL   client 가 사용할 서버 URL
#   N_EPISODES       태스크당 rollout 수 (default 20)
#   MAX_STEPS        episode 최대 step 수 (default 600)
#   OUTPUT_DIR       결과 저장 경로
#   CUDA_DEVICE      서버에서 사용할 디바이스 (default cuda — 컨테이너 안에선 GPU 1 매핑됨)

set -euo pipefail

MODE=${1:-help}

PORT=${PORT:-8510}
PROFILE=${PROFILE:-"/temporal_vla/configs/checkpoints/groot_n1d5_ttt__robocasa_target15.yaml"}
VLA_SERVER_URL=${VLA_SERVER_URL:-"http://localhost:${PORT}"}
N_EPISODES=${N_EPISODES:-20}
MAX_STEPS=${MAX_STEPS:-600}
CUDA_DEVICE=${CUDA_DEVICE:-"cuda"}
RUN_ID=$(date +%y%m%d%H%M)
OUTPUT_DIR=${OUTPUT_DIR:-"/temporal_vla/outputs/eval/robocasa/groot_n1d5_ttt/${RUN_ID}"}

# target/atomic 15 task (Phase 2 finetune 학습 set 과 동일).
TARGET_TASKS=(
    "CloseFridge"
    "CloseToasterOvenDoor"
    "CoffeeSetupMug"
    "OpenCabinet"
    "OpenDrawer"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "PickPlaceDrawerToCounter"
    "PickPlaceSinkToCounter"
    "PickPlaceToasterToCounter"
    "SlideDishwasherRack"
    "TurnOffStove"
    "TurnOnElectricKettle"
    "TurnOnMicrowave"
    "TurnOnSinkFaucet"
)

case "$MODE" in

  # ── 서버 기동 (groot_serve_n1d5 컨테이너 안에서 실행) ──
  server)
    echo "================================================"
    echo "GR00T N1.5 (+TTT) 서버 기동"
    echo "  Profile : ${PROFILE}"
    echo "  Port    : ${PORT}"
    echo "  Device  : ${CUDA_DEVICE}"
    echo "================================================"
    exec python /temporal_vla/scripts/serve/groot_n1d5_ttt.py \
        --profile "${PROFILE}" \
        --port "${PORT}" \
        --device "${CUDA_DEVICE}"
    ;;

  # ── /health 확인 (어디서든 가능) ──
  health)
    curl -s "${VLA_SERVER_URL}/health" | python3 -m json.tool
    ;;

  # ── 단일 태스크 평가 (robocasa 컨테이너) ──
  client)
    TASK=${2:-"CloseFridge"}
    echo "================================================"
    echo "단일 태스크 평가: ${TASK}"
    echo "  서버:     ${VLA_SERVER_URL}"
    echo "  rollouts: ${N_EPISODES}, max_steps: ${MAX_STEPS}"
    echo "  출력:     ${OUTPUT_DIR}"
    echo "================================================"
    mkdir -p "${OUTPUT_DIR}"
    python /temporal_vla/scripts/eval/robocasa_eval.py \
        --task "${TASK}" \
        --vla-server "${VLA_SERVER_URL}" \
        --num-rollouts "${N_EPISODES}" \
        --num-steps "${MAX_STEPS}" \
        --output-dir "${OUTPUT_DIR}" \
        --use-groot-env \
        --three-cameras
    ;;

  # ── target 15-task 배치 평가 (robocasa 컨테이너) ──
  client-batch)
    N_EP=${2:-${N_EPISODES}}
    echo "================================================"
    echo "GR00T N1.5 target/atomic 15-task 배치 평가"
    echo "  서버:     ${VLA_SERVER_URL}"
    echo "  rollouts: ${N_EP}/task   max_steps: ${MAX_STEPS}"
    echo "  출력:     ${OUTPUT_DIR}"
    echo "================================================"
    mkdir -p "${OUTPUT_DIR}"

    for TASK in "${TARGET_TASKS[@]}"; do
      echo
      echo "============================================"
      echo "Task: ${TASK} (${N_EP} episodes)"
      echo "============================================"
      python /temporal_vla/scripts/eval/robocasa_eval.py \
          --task "${TASK}" \
          --vla-server "${VLA_SERVER_URL}" \
          --num-rollouts "${N_EP}" \
          --num-steps "${MAX_STEPS}" \
          --output-dir "${OUTPUT_DIR}" \
          --use-groot-env \
          --three-cameras \
          --resume
    done

    echo
    echo "================================================"
    echo "배치 완료. 결과 dir: ${OUTPUT_DIR}"
    echo "================================================"
    ;;

  # ── 서버 + 단일 rollout 동시 실행 (단일 머신 smoke test 용) ──
  smoke-test)
    echo "Smoke test: 서버를 백그라운드 기동 후 CloseFridge 1 rollout"
    python /temporal_vla/scripts/serve/groot_n1d5_ttt.py \
        --profile "${PROFILE}" \
        --port "${PORT}" \
        --device "${CUDA_DEVICE}" &
    SERVER_PID=$!
    echo "서버 PID: ${SERVER_PID}"

    # 서버 준비 대기 (최대 5분).
    for i in $(seq 1 60); do
      sleep 5
      STATUS=$(curl -s "${VLA_SERVER_URL}/health" 2>/dev/null \
                | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null \
                || echo "")
      if [ "${STATUS}" = "ok" ]; then
        echo "서버 준비 완료 (${i}회 시도)"
        break
      fi
      echo "대기 중... (${i}/60)"
    done

    python /temporal_vla/scripts/eval/robocasa_eval.py \
        --task "CloseFridge" \
        --vla-server "${VLA_SERVER_URL}" \
        --num-rollouts 1 \
        --num-steps "${MAX_STEPS}" \
        --use-groot-env \
        --three-cameras

    kill "${SERVER_PID}" 2>/dev/null || true
    echo "Smoke test 완료"
    ;;

  *)
    cat <<EOF
사용법:
  $0 server                — 서버 기동 (groot_serve_n1d5 컨테이너 내부에서 실행)
  $0 health                — /health 응답 확인
  $0 client [TASK]         — 단일 태스크 평가 (robocasa 컨테이너)
  $0 client-batch [N_EP]   — target 15-task 전체 배치 평가
  $0 smoke-test            — 서버 백그라운드 + 1-episode rollout (테스트)

환경 변수:
  PROFILE=${PROFILE}
  PORT=${PORT}
  VLA_SERVER_URL=${VLA_SERVER_URL}
  N_EPISODES=${N_EPISODES}
  MAX_STEPS=${MAX_STEPS}
  OUTPUT_DIR=${OUTPUT_DIR}
EOF
    ;;
esac
