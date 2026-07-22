#!/usr/bin/env bash
# cam-attention 수집: DiT cross-attn 카메라 뷰별 mass 를 phase GT 와 함께 기록.
# 대비쌍 2 cell × ep0-19 (fixed-scene, inference noise 변주 — fit 수집 규약).
# GPU 2 전용 (2026-07-16 사용자 지시), cell 당 serve 1개 (2 serve ≈ 15GB/16GB).
# serve: --collect --capture-cross-attn / collector: --attn-only-records (pkl 수 MB).
set -uo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
N=20; STRIDE=1000; NAS=5; MAXEP=720; GPU=2
RUN_ID=cam_attn
OUT_HOST="$(pwd)/outputs/eval/robocasa/groot_n15/${RUN_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

# cell|task|env|idx|seed|instr|port  (bread=PnP 대비 drawer=fixture 조작)
CELLS=(
  "ppcc_bread|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|100084|Pick the bread from the counter and place it in the cabinet.|8492"
  "pq3_drawer_right|OpenDrawer|robocasa_panda_omron/OpenDrawer_PandaOmron_Env|7|100000|Open the right drawer.|8493"
)

for c in "${CELLS[@]}"; do IFS='|' read -r _ _ _ _ _ _ port <<<"$c"
  docker exec -d -e CUDA_VISIBLE_DEVICES="$GPU" lerobot bash -lc \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
       --host '*' --port ${port} --device cuda --collect --capture-cross-attn \
       > /tmp/cam_attn_${port}.log 2>&1 < /dev/null &"
done
for c in "${CELLS[@]}"; do IFS='|' read -r cell _ _ _ _ _ port <<<"$c"
  ok=0; for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" || true)
    if echo "$st" | grep -q '"status":"ok"'; then
      # attn preflight: capture_cross_attn 광고 + boot 로그의 attn-preflight 라인 존재
      echo "$st" | grep -q '"capture_cross_attn":true' || { echo "[cam_attn] $cell serve ${port}: capture_cross_attn MISSING"; exit 13; }
      docker exec lerobot bash -lc "grep -q attn-preflight /tmp/cam_attn_${port}.log" || { echo "[cam_attn] $cell serve ${port}: no attn-preflight log"; exit 14; }
      ok=1; break
    fi; sleep 5
  done
  echo "[cam_attn] $cell serve ${port}: $([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  [ $ok = 1 ] || exit 12
done

run_cell() {  # row
  local row="$1"
  IFS='|' read -r cell task env idx seed instr port <<<"$row"
  for ep in $(seq 0 $((N - 1))); do
    if ls "${OUT_HOST}/raw_rollouts/${task}/${cell}/task${idx}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$task" --env-name "$env" \
      --output-dir "$OUT_CONT" --cell-id "$cell" --cell-index "$idx" \
      --canonical-instruction "$instr" --episode-start-idx "$ep" --n-episodes 1 \
      --seed "$seed" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready \
      --proximity-phases --attn-only-records 2>&1 | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[${cell}] DONE"
}
for c in "${CELLS[@]}"; do IFS='|' read -r cell _ _ _ _ _ _ <<<"$c"
  run_cell "$c" > "${LOGDIR}/${cell}.log" 2>&1 &
done
wait
for c in "${CELLS[@]}"; do IFS='|' read -r _ _ _ _ _ _ port <<<"$c"
  docker exec lerobot bash -lc "pkill -9 -f 'lerobot.py.*${port}' || true" 2>/dev/null || true
done
for c in "${CELLS[@]}"; do IFS='|' read -r cell task _ _ _ _ _ <<<"$c"
  d="${OUT_HOST}/raw_rollouts/${task}/${cell}"
  echo "${cell}: succ=$(ls $d/*succ1.pkl 2>/dev/null|wc -l) fail=$(ls $d/*succ0.pkl 2>/dev/null|wc -l)"
done
touch "${LOGDIR}/CAM_ATTN_DONE"
echo "[cam_attn] $(date '+%F %T') DONE"
