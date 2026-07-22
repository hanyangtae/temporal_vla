#!/usr/bin/env bash
# srv50(구 w2/worker2, A100)에서 fit 재료 수집 (P0 후보 scene, ep0-59, steering 없음).
# heldout_round_cell_host.sh 의 검증된 host-conda serve 블록 + 로컬 run_collect 의 수집 루프.
# 출력: srv50 repo 의 phase_event_6p/raw_rollouts/<TASK>/<CELL_ID>/ (직송은 lane runner 가 담당).
# env: CELL_ID TASK ENVN CELL_INDEX SEED INSTR PORTS_L [GPUS_L(기본 2 2 2)]
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
STRIDE=1000; NAS=5; MAXEP=720
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${SEED:?}" "${INSTR:?}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
HPY="$HOME/miniconda3/envs/lerobot_050_groot/bin/python"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(${GPUS_L:-2 2 2}); PORTS=(${PORTS_L:?}); NW=${#PORTS[@]}

start_serves() {
  local i
  for i in $(seq 0 $((NW-1))); do
    ( cd "$REPO_ROOT" && setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        PYTHONPATH="$REPO_ROOT/lerobot/src" "$HPY" scripts/serve/lerobot.py --profile ${PROFILE} \
        --host '*' --port "${PORTS[$i]}" --device cuda --collect --capture-vl \
        --groot-dit-capture-layers ${CAP} > "/tmp/p0c_${CELL_ID}_${PORTS[$i]}.log" 2>&1 < /dev/null & )
  done
  local port ok
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      curl -s -m 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }; sleep 5
    done
    echo "[${CELL_ID}] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    if grep -qiE 'Traceback|FAILED|FileNotFound' "/tmp/p0c_${CELL_ID}_${port}.log"; then
      echo "[${CELL_ID}] ABORT ${port}"; tail -20 "/tmp/p0c_${CELL_ID}_${port}.log"; exit 11
    fi
    [ $ok = 1 ] || exit 11
  done
}
kill_serves() { local port pid; for port in "${PORTS[@]}"; do
    for pid in $(pgrep -f "lerobot.py.*--port ${port}"); do kill "$pid" 2>/dev/null || true; done
  done; sleep 5; }

cw() { local wid=$1 port=$2 ep
  for ep in $(seq 0 59); do
    [ $((ep % NW)) -eq "$wid" ] || continue
    ls "${OUT_HOST}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.pkl >/dev/null 2>&1 && continue
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENVN" \
      --output-dir "${OUT_CONT}" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "$SEED" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --proximity-phases 2>&1 | grep -E "^wrote|Traceback" || true
  done
}

echo "[${CELL_ID}] $(date '+%F %T') P0 collect 시작 (ep0-59, ${NW} serve)"
start_serves
for wid in $(seq 0 $((NW-1))); do cw "$wid" "${PORTS[$wid]}" & done
wait
kill_serves
n=$(ls "${OUT_HOST}/${TASK}/${CELL_ID}"/task*--ep*--succ*.pkl 2>/dev/null | wc -l)
echo "[${CELL_ID}] $(date '+%F %T') collect DONE ${n}/60"
[ "$n" -ge 60 ]
