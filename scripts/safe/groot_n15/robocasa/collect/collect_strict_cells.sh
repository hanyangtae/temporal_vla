#!/usr/bin/env bash
# 새 instruction strict(drop-aware) fit-수집: apple(seed100074)·potato(seed200019) × ep0-29.
# GPU 2,3,4,5 / ports 8450-8453 (cell당 serve 2개). RUN_ID=phase_event_strict 트리에 저장.
# detached: setsid nohup bash scripts/safe/groot_n15/robocasa/collect/collect_strict_cells.sh \
#   > outputs/eval/robocasa/groot_n15/phase_event_strict/logs/cells_orch.log 2>&1 < /dev/null &
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
N=30; STRIDE=1000; NAS=5; MAXEP=720
RUN_ID=phase_event_strict
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/${RUN_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

# cell|task|env|idx|seed|instr|gpuA|portA|gpuB|portB
CELLS=(
  "ppcs_apple|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100074|Pick the apple from the plate and place it in the pan.|2|8450|3|8451"
  "ppcc_potato|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|4|200019|Pick the potato from the counter and place it in the cabinet.|4|8452|5|8453"
)
start_serve() { docker exec -d -e CUDA_VISIBLE_DEVICES="$1" lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
     --host '*' --port ${2} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
     > /tmp/strict_cells_${2}.log 2>&1 < /dev/null &"; }

echo "[cells] $(date '+%F %T') starting 4 serves"
for c in "${CELLS[@]}"; do IFS='|' read -r _ _ _ _ _ _ gA pA gB pB <<<"$c"; start_serve "$gA" "$pA"; start_serve "$gB" "$pB"; done
for c in "${CELLS[@]}"; do IFS='|' read -r _ _ _ _ _ _ _ pA _ pB <<<"$c"
  for port in "$pA" "$pB"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[cells] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || { echo "[cells] ABORT ${port}"; exit 12; }
  done
done

run_worker() {  # cell-row worker_id port
  local row="$1" wid="$2" port="$3"
  IFS='|' read -r cell task env idx seed instr _ _ _ _ <<<"$row"
  for ep in $(seq 0 $((N - 1))); do
    [ $((ep % 2)) -eq "$wid" ] || continue
    if ls "${OUT_HOST}/raw_rollouts/${task}/${cell}/task${idx}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
    echo "[${cell} w${wid}] ep${ep}"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$task" --env-name "$env" \
      --output-dir "$OUT_CONT" --cell-id "$cell" --cell-index "$idx" \
      --canonical-instruction "$instr" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "$seed" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[${cell} w${wid}] DONE"
}
for c in "${CELLS[@]}"; do IFS='|' read -r cell _ _ _ _ _ _ pA _ pB <<<"$c"
  run_worker "$c" 0 "$pA" > "${LOGDIR}/cells_${cell}_w0.log" 2>&1 &
  run_worker "$c" 1 "$pB" > "${LOGDIR}/cells_${cell}_w1.log" 2>&1 &
done
wait
for port in 8450 8451 8452 8453; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done
for c in "${CELLS[@]}"; do IFS='|' read -r cell task _ _ _ _ _ _ _ _ <<<"$c"
  d="${OUT_HOST}/raw_rollouts/${task}/${cell}"
  echo "${cell}: succ=$(ls $d/*succ1.pkl 2>/dev/null|wc -l) fail=$(ls $d/*succ0.pkl 2>/dev/null|wc -l)"
done
touch "${LOGDIR}/CELLS_DONE"
echo "[cells] $(date '+%F %T') DONE"
