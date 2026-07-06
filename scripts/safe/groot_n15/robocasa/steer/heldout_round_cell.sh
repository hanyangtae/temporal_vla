#!/usr/bin/env bash
# 범용 held-out 라운드 (cell 파라미터화): fit=ep0-29 → eval ep30-59 (fit-seed 분리).
# arms: ho_base(무steer) → ho_perm(strict-global multi 4,8,12) → ho_gated(phase-gated).
# env: CELL_ID TASK ENVN CELL_INDEX SEED INSTR GPU_A GPU_B PORT_A PORT_B
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
EP0="${EP0:-30}"; EP1="${EP1:-59}"; STRIDE=1000; NAS=5; MAXEP=720
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${SEED:?}" "${INSTR:?}"
# SUF: steer arm 접미(예: 60 → ho_perm60/ho_gated60, conceptor 세대 구분). base 는 공유라 무접미.
SUF="${SUF:-}"
STEER_LAYERS="${STEER_LAYERS:-4,8,12}"
# STEER_ALPHA: 지정 시 --steering-alpha 전달(미지정=기존과 동일하게 NPZ 첫 키)
STEER_ALPHA_FLAG=""; [ -n "${STEER_ALPHA:-}" ] && STEER_ALPHA_FLAG="--steering-alpha ${STEER_ALPHA}"
STEER_BETA="${STEER_BETA:-0.3}"
# ARMS: 실행할 arm 부분집합 (기본 전부). 예: ARMS="gated"
ARMS="${ARMS:-base perm gated}"
# PROX=1 → 전 arm 수집에 --proximity-phases(6/7-phase 라벨) 부여
PROX_FLAG=""; [ "${PROX:-0}" = "1" ] && PROX_FLAG="--proximity-phases"
NPZ_ROOT="${NPZ_ROOT:-/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_strict/analysis/conceptor_steering_n15}"
BASE=${NPZ_ROOT}/${CELL_ID}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
# GPUS_L/PORTS_L: 공백구분 리스트(길이 같아야; GPU 중복 허용=GPU당 serve 2개). NW=worker 수.
GPUS=(${GPUS_L:-${GPU_A:?} ${GPU_B:?}}); PORTS=(${PORTS_L:-${PORT_A:?} ${PORT_B:?}})
NW=${#PORTS[@]}

start_serves() {
  for i in $(seq 0 $((NW-1))); do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda --collect --capture-vl \
         --groot-dit-capture-layers ${CAP} $* > /tmp/ho_${CELL_ID}_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[${CELL_ID}] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/ho_${CELL_ID}_${port}.log"; then
      echo "[${CELL_ID}] ABORT ${port}"; docker exec lerobot bash -lc "tail -20 /tmp/ho_${CELL_ID}_${port}.log"; exit 11
    fi
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }

run_arm() {  # tag extra...
  local tag="$1"; shift; local extra="$*"
  run_w() {
    local wid=$1 port=$2
    for ep in $(seq $EP0 $EP1); do
      [ $((ep % NW)) -eq "$wid" ] || continue
      if ls "${OUT_HOST}/${tag}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
      docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
        python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
        --output-dir "${OUT_CONT}/${tag}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
        --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
        --seed "$SEED" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
        --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready $PROX_FLAG $extra 2>&1 \
        | grep -E "^wrote|Error|Traceback" || true
    done
  }
  for wid in $(seq 0 $((NW-1))); do
    run_w "$wid" "${PORTS[$wid]}" > "${LOGDIR}/${tag}_w${wid}.log" 2>&1 &
  done
  wait
}

if [[ " $ARMS " == *" base "* ]]; then
echo "[${CELL_ID}] $(date '+%F %T') arm ho_base"
start_serves
run_arm ho_base
kill_serves
fi
if [[ " $ARMS " == *" perm "* ]]; then
echo "[${CELL_ID}] $(date '+%F %T') arm ho_perm"
start_serves --steering-npz-dir ${BASE}/global --steering-layers ${STEER_LAYERS} --steering-beta ${STEER_BETA} ${STEER_ALPHA_FLAG} --steering-key C_steer
run_arm ho_perm${SUF}
kill_serves
fi
if [[ " $ARMS " == *" gated "* ]]; then
echo "[${CELL_ID}] $(date '+%F %T') arm ho_gated"
start_serves --steering-phase-npz-base ${BASE} --steering-layers ${STEER_LAYERS} --steering-beta ${STEER_BETA} ${STEER_ALPHA_FLAG} --steering-key C_steer
run_arm ho_gated${SUF} --gated-steering
kill_serves
fi

RES="${OUT_HOST}/sr_result_heldout.tsv"
{
  echo -e "condition\tsucc\tfail\ttotal\tSR"
  for tag in ho_base ho_perm${SUF} ho_gated${SUF}; do
    d="${OUT_HOST}/${tag}/raw_rollouts/${TASK}/${CELL_ID}"
    s=$(ls "$d"/*succ1.pkl 2>/dev/null|wc -l); f=$(ls "$d"/*succ0.pkl 2>/dev/null|wc -l)
    echo -e "${tag}\t${s}\t${f}\t$((s+f))\t$(awk "BEGIN{printf \"%.3f\", ($s+$f>0)?$s/($s+$f):0}")"
  done
} > "$RES"
cat "$RES"
touch "${LOGDIR}/HELDOUT_${CELL_ID}${SUF}_DONE"
echo "[${CELL_ID}] $(date '+%F %T') DONE -> ${RES}"
