#!/usr/bin/env bash
# 확증 (GPU 2,3 / ports 8430,8431): baseline·global-multi(4,8,12) 각각 ep30..89 추가 → n=90/조건.
# 기존 ep0-29는 resume-skip. serve0(GPU2)=baseline(무steer), serve1(GPU3)=steered ml_global.
# detached: setsid nohup bash scripts/safe/groot_n15/robocasa/steer/confirm90_gpu23.sh \
#   > outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/confirm90.log 2>&1 < /dev/null &
set -uo pipefail
PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
N_START=30; N_END=89; STRIDE=1000; NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENV=robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env
CELL_INDEX=5; CELL_ID=ppcc_bread; SEED=100084
INSTR="Pick the bread from the counter and place it in the cabinet."
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
NPZ_DIR=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/analysis/conceptor_steering_n15/ppcc_bread/global
STEER_ARGS="--steering-npz-dir ${NPZ_DIR} --steering-layers 4,8,12 --steering-beta 0.3 --steering-key C_steer"

echo "[confirm] $(date '+%F %T') starting serves: baseline GPU2:8430, steered GPU3:8431"
docker exec -d -e CUDA_VISIBLE_DEVICES=2 lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
     --host '*' --port 8430 --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
     > /tmp/confirm_8430.log 2>&1 < /dev/null &"
docker exec -d -e CUDA_VISIBLE_DEVICES=3 lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
     --host '*' --port 8431 --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
     ${STEER_ARGS} > /tmp/confirm_8431.log 2>&1 < /dev/null &"
for port in 8430 8431; do
  ok=0
  for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && { ok=1; break; }; sleep 5
  done
  echo "[confirm] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  [ $ok = 1 ] || { echo "[confirm] ABORT ${port}"; exit 12; }
done

run_cond() {  # cond port
  local cond=$1 port=$2
  local outdir="${OUT_CONT}/${cond}/raw_rollouts" hostdir="${OUT_HOST}/${cond}/raw_rollouts"
  for idx in $(seq $N_START $N_END); do
    if ls "${hostdir}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${idx}--succ"*.pkl >/dev/null 2>&1; then
      echo "[${cond}] skip ep${idx}"; continue; fi
    echo "[${cond}] ep${idx}"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENV" \
      --output-dir "$outdir" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$idx" --n-episodes 1 \
      --seed "$SEED" --inference-seed "$((idx * STRIDE))" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[${cond}] DONE"
}
run_cond baseline 8430 > "${LOGDIR}/confirm_baseline.log" 2>&1 &
run_cond steered_ml_global 8431 > "${LOGDIR}/confirm_steered.log" 2>&1 &
wait
for port in 8430 8431; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done
RES="${OUT_HOST}/sr_result_confirm90.tsv"
{
  echo -e "condition\tsucc\tfail\ttotal\tSR"
  for cond in baseline steered_ml_global; do
    d="${OUT_HOST}/${cond}/raw_rollouts/${TASK}/${CELL_ID}"
    s=$(ls "$d"/*succ1.pkl 2>/dev/null|wc -l); f=$(ls "$d"/*succ0.pkl 2>/dev/null|wc -l); t=$((s+f))
    echo -e "${cond}\t${s}\t${f}\t${t}\t$(awk "BEGIN{printf \"%.3f\", ($t>0)?$s/$t:0}")"
  done
} > "$RES"
cat "$RES"
touch "${LOGDIR}/CONFIRM90_DONE"
echo "[confirm] $(date '+%F %T') DONE -> ${RES}"
