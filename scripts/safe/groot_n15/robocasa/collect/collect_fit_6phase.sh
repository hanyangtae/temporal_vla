#!/usr/bin/env bash
# 6/7-phase(proximity+wrong-grasp) fit 데이터 재수집: 3 cell × ep0-59 (결정적 재현).
# RUN_ID=phase_event_6p. 6 serve / 3 GPU (0,4,6) — 사용자 상한. cell당 GPU1×serve2.
set -uo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
N=60; STRIDE=1000; NAS=5; MAXEP=720
RUN_ID=phase_event_6p
OUT_HOST="$(pwd)/outputs/eval/robocasa/groot_n15/${RUN_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

# cell|task|env|idx|seed|instr|gpu|portA|portB
CELLS=(
  "ppcc_bread|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|100084|Pick the bread from the counter and place it in the cabinet.|0|8480|8481"
  "ppcs_apple|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100074|Pick the apple from the plate and place it in the pan.|4|8470|8471"
  "ppcc_potato|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|4|200019|Pick the potato from the counter and place it in the cabinet.|6|8472|8473"
)
for c in "${CELLS[@]}"; do IFS='|' read -r _ _ _ _ _ _ gpu pA pB <<<"$c"
  for port in "$pA" "$pB"; do
    docker exec -d -e CUDA_VISIBLE_DEVICES="$gpu" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${port} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
         > /tmp/fit6p_${port}.log 2>&1 < /dev/null &"
  done
done
for c in "${CELLS[@]}"; do IFS='|' read -r cell _ _ _ _ _ _ pA pB <<<"$c"
  for port in "$pA" "$pB"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[fit6p] $cell serve ${port}: $([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    [ $ok = 1 ] || exit 12
  done
done

run_worker() {  # row wid port
  local row="$1" wid="$2" port="$3"
  IFS='|' read -r cell task env idx seed instr _ _ _ <<<"$row"
  for ep in $(seq 0 $((N - 1))); do
    [ $((ep % 2)) -eq "$wid" ] || continue
    if ls "${OUT_HOST}/raw_rollouts/${task}/${cell}/task${idx}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$task" --env-name "$env" \
      --output-dir "$OUT_CONT" --cell-id "$cell" --cell-index "$idx" \
      --canonical-instruction "$instr" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "$seed" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --proximity-phases 2>&1 | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[${cell} w${wid}] DONE"
}
for c in "${CELLS[@]}"; do IFS='|' read -r cell _ _ _ _ _ _ pA pB <<<"$c"
  run_worker "$c" 0 "$pA" > "${LOGDIR}/${cell}_w0.log" 2>&1 &
  run_worker "$c" 1 "$pB" > "${LOGDIR}/${cell}_w1.log" 2>&1 &
done
wait
for port in 8480 8481 8470 8471 8472 8473; do docker exec lerobot bash -lc "pkill -9 -f 'lerobot.py.*${port}' || true" 2>/dev/null || true; done
for c in "${CELLS[@]}"; do IFS='|' read -r cell task _ _ _ _ _ _ _ <<<"$c"
  d="${OUT_HOST}/raw_rollouts/${task}/${cell}"
  echo "${cell}: succ=$(ls $d/*succ1.pkl 2>/dev/null|wc -l) fail=$(ls $d/*succ0.pkl 2>/dev/null|wc -l)"
done
touch "${LOGDIR}/FIT6P_DONE"
echo "[fit6p] $(date '+%F %T') DONE"
