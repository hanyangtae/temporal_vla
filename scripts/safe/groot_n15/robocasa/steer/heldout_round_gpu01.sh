#!/usr/bin/env bash
# Held-out(ep60-89) 라운드: strict conceptor(fit=ep0-59, seed 분리) 로
#   arm1 hg_perm  = permanent strict-global multi(4,8,12)
#   arm2 hg_gated = oracle phase-gated (reach/transport/insert-settle 스위칭, /steering_phase)
# baseline ep60-89 = 기존 23/30 재사용. GPU 0,1 / ports 8460,8461. 순차 2 arm.
# detached: setsid nohup bash scripts/safe/groot_n15/robocasa/steer/heldout_round_gpu01.sh \
#   > outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/heldout_round.log 2>&1 < /dev/null &
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
EP0=60; EP1=89; STRIDE=1000; NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENV=robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env
CELL_INDEX=5; CELL_ID=ppcc_bread; SEED=100084
INSTR="Pick the bread from the counter and place it in the cabinet."
BASE=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_strict/analysis/conceptor_steering_n15/ppcc_bread
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla"
GPUS=(0 1); PORTS=(8460 8461)

start_serves() {  # steer_args...
  for i in 0 1; do
    docker exec -d -e CUDA_VISIBLE_DEVICES="${GPUS[$i]}" lerobot bash -lc \
      "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
         --host '*' --port ${PORTS[$i]} --device cuda --collect --capture-vl \
         --groot-dit-capture-layers ${CAP} $* > /tmp/heldout_${PORTS[$i]}.log 2>&1 < /dev/null &"
  done
  for port in "${PORTS[@]}"; do
    ok=0; for _ in $(seq 1 150); do
      st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
      [ -n "$st" ] && { ok=1; break; }; sleep 5
    done
    echo "[ho] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
    if docker exec lerobot bash -lc "grep -qiE 'Traceback|FAILED|FileNotFound' /tmp/heldout_${port}.log"; then
      echo "[ho] ABORT ${port}"; docker exec lerobot bash -lc "tail -25 /tmp/heldout_${port}.log"; exit 11
    fi
  done
}
kill_serves() { for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done; sleep 5; }

run_arm() {  # tag extra_collector_args...
  local tag="$1"; shift
  local extra="$*"
  run_w() {
    local wid=$1 port=$2
    for ep in $(seq $EP0 $EP1); do
      [ $((ep % 2)) -eq "$wid" ] || continue
      if ls "${OUT_HOST}/steered_${tag}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${ep}--succ"*.pkl >/dev/null 2>&1; then continue; fi
      echo "[${tag} w${wid}] ep${ep}"
      docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
        python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENV" \
        --output-dir "${OUT_CONT}/steered_${tag}/raw_rollouts" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
        --canonical-instruction "$INSTR" --episode-start-idx "$ep" --n-episodes 1 \
        --seed "$SEED" --inference-seed "$((ep * STRIDE))" --n-action-steps "$NAS" \
        --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready $extra 2>&1 \
        | grep -E "^wrote|Error|Traceback" || true
    done
    echo "[${tag} w${wid}] DONE"
  }
  run_w 0 "${PORTS[0]}" > "${LOGDIR}/${tag}_w0.log" 2>&1 &
  run_w 1 "${PORTS[1]}" > "${LOGDIR}/${tag}_w1.log" 2>&1 &
  wait
}

echo "[ho] $(date '+%F %T') === arm1 hg_perm (permanent strict-global multi 4,8,12) ==="
start_serves --steering-npz-dir ${BASE}/global --steering-layers 4,8,12 --steering-beta 0.3 --steering-key C_steer
run_arm hg_perm
kill_serves

echo "[ho] $(date '+%F %T') === arm2 hg_gated (oracle phase-gated) ==="
start_serves --steering-phase-npz-base ${BASE} --steering-layers 4,8,12 --steering-beta 0.3 --steering-key C_steer
# gated 등록 확인
for port in "${PORTS[@]}"; do
  docker exec lerobot bash -lc "grep -q 'Phase-gated conceptor steering registered' /tmp/heldout_${port}.log && echo '[ho] ${port} gated OK' || echo '[ho] ${port} gated 로그 미확인(logger handler 없음 가능)'"
done
run_arm hg_gated --gated-steering
kill_serves

RES="${OUT_HOST}/sr_result_heldout.tsv"
{
  echo -e "condition\tsucc\tfail\ttotal\tSR"
  bd="${OUT_HOST}/baseline/raw_rollouts/${TASK}/${CELL_ID}"
  s=0; f=0
  for ep in $(seq $EP0 $EP1); do
    ls $bd/task${CELL_INDEX}--ep${ep}--succ1.pkl >/dev/null 2>&1 && s=$((s+1))
    ls $bd/task${CELL_INDEX}--ep${ep}--succ0.pkl >/dev/null 2>&1 && f=$((f+1))
  done
  echo -e "baseline_ep60-89\t${s}\t${f}\t$((s+f))\t$(awk "BEGIN{printf \"%.3f\", $s/($s+$f)}")"
  for tag in hg_perm hg_gated; do
    d="${OUT_HOST}/steered_${tag}/raw_rollouts/${TASK}/${CELL_ID}"
    s=$(ls "$d"/*succ1.pkl 2>/dev/null|wc -l); f=$(ls "$d"/*succ0.pkl 2>/dev/null|wc -l)
    echo -e "${tag}\t${s}\t${f}\t$((s+f))\t$(awk "BEGIN{printf \"%.3f\", ($s+$f>0)?$s/($s+$f):0}")"
  done
} > "$RES"
cat "$RES"
touch "${LOGDIR}/HELDOUT_ROUND_DONE"
echo "[ho] $(date '+%F %T') DONE -> ${RES}"
