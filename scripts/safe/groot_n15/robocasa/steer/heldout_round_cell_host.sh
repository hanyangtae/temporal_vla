#!/usr/bin/env bash
# heldout_round_cell.sh 의 worker2(A100) 변형: serve 를 lerobot 컨테이너가 아니라
# 호스트 conda(lerobot_050_groot)로 기동. collector 는 동일하게 robocasa 컨테이너.
# 스모크·calibration(2026-07-07)에서 검증된 호스트 serve 커맨드 사용.
# env: CELL_ID TASK ENVN CELL_INDEX SEED INSTR GPUS_L PORTS_L [EP0 EP1 SUF ARMS NPZ_ROOT(호스트 절대경로) STEER_* PROX]
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
EP0="${EP0:-60}"; EP1="${EP1:-119}"; STRIDE=1000; NAS=5; MAXEP=720
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${SEED:?}" "${INSTR:?}"
SUF="${SUF:-}"
STEER_LAYERS="${STEER_LAYERS:-4,8,12}"
STEER_ALPHA_FLAG=""; [ -n "${STEER_ALPHA:-}" ] && STEER_ALPHA_FLAG="--steering-alpha ${STEER_ALPHA}"
STEER_BETA="${STEER_BETA:-0.3}"
ARMS="${ARMS:-base perm gated}"
PROX_FLAG=""; [ "${PROX:-0}" = "1" ] && PROX_FLAG="--proximity-phases"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
HPY="$HOME/miniconda3/envs/lerobot_050_groot/bin/python"
NPZ_ROOT="${NPZ_ROOT:-${REPO_ROOT}/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_ps60}"
BASE=${NPZ_ROOT}/${CELL_ID}
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:-2 2}); PORTS=(${PORTS_L:-8480 8481})
NW=${#PORTS[@]}

start_serves() {
  local i
  for i in $(seq 0 $((NW-1))); do
    ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        PYTHONPATH="$REPO_ROOT/lerobot/src" "$HPY" scripts/serve/lerobot.py --profile ${PROFILE} \
        --host '*' --port "${PORTS[$i]}" --device cuda --collect --capture-vl \
        --groot-dit-capture-layers ${CAP} "$@" > "/tmp/ho_${CELL_ID}_${PORTS[$i]}.log" 2>&1 < /dev/null & )
  done
  local port ok
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }; sleep 5
    done
    echo "[${CELL_ID}] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    if grep -qiE 'Traceback|FAILED|FileNotFound' "/tmp/ho_${CELL_ID}_${port}.log"; then
      echo "[${CELL_ID}] ABORT ${port}"; tail -20 "/tmp/ho_${CELL_ID}_${port}.log"; exit 11
    fi
    [ $ok = 1 ] || exit 11
  done
}
kill_serves() { local port pid; for port in "${PORTS[@]}"; do
    for pid in $(pgrep -f "lerobot.py.*--port ${port}"); do kill "$pid" 2>/dev/null || true; done
  done; sleep 5; }

run_arm() {  # tag extra...
  local tag="$1"; shift; local extra="$*"
  run_w() {
    local wid=$1 port=$2 ep
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
  local wid
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
