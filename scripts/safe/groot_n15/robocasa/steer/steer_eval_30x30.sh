#!/usr/bin/env bash
# Steered ΔSR eval (Rung 3): 같은 scene/instruction 에서 baseline 30 vs steered 30 rollout → SR 비교.
# 첫 타깃: bread(scenario_seed 100084), conceptor global/dit_L4 (COAST cabinet L5→우리 L4),
#          alpha=0.3, beta=0.3, permanent(전 스텝 주입).
# baseline serve(무steer) 2개(GPU6/7) + steered serve 2개(GPU4/5) 병렬. 각 조건 30 rollout 을
# 자기 2 serve 에 round-robin. 완료 시 serve kill + SR 집계.
# detached: cd REPO && setsid nohup bash scripts/safe/groot_n15/robocasa/steer/steer_eval_30x30.sh \
#   > outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/orchestrator.log 2>&1 < /dev/null &
set -uo pipefail

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
N=30; STRIDE=1000; NAS=5; MAXEP=720

# --- cell (bread) ---
TASK=PickPlaceCounterToCabinet
ENV=robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env
CELL_INDEX=5
CELL_ID=ppcc_bread
SEED=100084
INSTR="Pick the bread from the counter and place it in the cabinet."

# --- steering target ---
NPZ=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/analysis/conceptor_steering_n15/ppcc_bread/global/dit_L4/conceptors.npz
STEER_ARGS="--steering-npz ${NPZ} --steering-pathway dit --steering-layer 4 --steering-alpha 0.3 --steering-beta 0.3 --steering-key C_steer"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

declare -A COND_PORTS=( [baseline]="8400 8401" [steered]="8410 8411" )
declare -A COND_GPUS=(  [baseline]="6 7"       [steered]="4 5"       )

start_serve() {  # gpu port [steering args...]
  local gpu=$1 port=$2; shift 2
  docker exec -d -e CUDA_VISIBLE_DEVICES="$gpu" lerobot bash -lc \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
       --host '*' --port ${port} --device cuda --collect --capture-vl \
       --groot-dit-capture-layers ${CAP} $* > /tmp/steer_eval_${port}.log 2>&1 < /dev/null &"
}

echo "[orch] $(date '+%F %T') starting 4 serves (baseline GPU6/7, steered GPU4/5)"
read -r bg0 bg1 <<<"${COND_GPUS[baseline]}"; read -r bp0 bp1 <<<"${COND_PORTS[baseline]}"
read -r sg0 sg1 <<<"${COND_GPUS[steered]}";  read -r sp0 sp1 <<<"${COND_PORTS[steered]}"
start_serve "$bg0" "$bp0"; start_serve "$bg1" "$bp1"
start_serve "$sg0" "$sp0" $STEER_ARGS; start_serve "$sg1" "$sp1" $STEER_ARGS

# wait /health for all 4 ports (cold compile ~2min)
ALL_PORTS="$bp0 $bp1 $sp0 $sp1"
for port in $ALL_PORTS; do
  ok=0
  for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && { ok=1; break; }
    sleep 5
  done
  echo "[orch] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  [ $ok = 1 ] || { echo "[orch] ABORT: serve ${port} not ready"; docker exec lerobot bash -lc "tail -20 /tmp/steer_eval_${port}.log"; }
done
# verify steered serves loaded steering without error
for port in $sp0 $sp1; do
  if docker exec lerobot bash -lc "grep -qiE 'Traceback|load_model FAILED|KeyError|ValueError' /tmp/steer_eval_${port}.log"; then
    echo "[orch] ABORT: steered serve ${port} load error"; docker exec lerobot bash -lc "tail -30 /tmp/steer_eval_${port}.log"; exit 11
  fi
done

run_share() {  # cond port worker_id
  local cond=$1 port=$2 wid=$3
  local outdir="${OUT_CONT}/${cond}/raw_rollouts"
  local hostdir="${OUT_HOST}/${cond}/raw_rollouts"
  for idx in $(seq 0 $((N - 1))); do
    [ $((idx % 2)) -eq "$wid" ] || continue
    if ls "${hostdir}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${idx}--succ"*.pkl >/dev/null 2>&1; then
      echo "[${cond} w${wid}] skip ep${idx}"; continue
    fi
    local inf=$((idx * STRIDE))
    echo "[${cond} w${wid}] ep${idx} seed=${SEED} inf=${inf} port=${port}"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENV" \
      --output-dir "$outdir" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$idx" --n-episodes 1 \
      --seed "$SEED" --inference-seed "$inf" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[${cond} w${wid}] DONE"
}

echo "[orch] $(date '+%F %T') running 30 baseline + 30 steered"
run_share baseline "$bp0" 0 > "${LOGDIR}/baseline_w0.log" 2>&1 &
run_share baseline "$bp1" 1 > "${LOGDIR}/baseline_w1.log" 2>&1 &
run_share steered  "$sp0" 0 > "${LOGDIR}/steered_w0.log" 2>&1 &
run_share steered  "$sp1" 1 > "${LOGDIR}/steered_w1.log" 2>&1 &
wait
echo "[orch] $(date '+%F %T') rollouts done; killing serves"
for port in $ALL_PORTS; do
  docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true
done

# SR 집계
RES="${OUT_HOST}/sr_result.tsv"
{
  echo -e "condition\tsucc\tfail\ttotal\tSR"
  for cond in baseline steered; do
    d="${OUT_HOST}/${cond}/raw_rollouts/${TASK}/${CELL_ID}"
    s=$(ls "$d"/*succ1.pkl 2>/dev/null | wc -l); f=$(ls "$d"/*succ0.pkl 2>/dev/null | wc -l); t=$((s+f))
    sr=$(awk "BEGIN{printf \"%.3f\", ($t>0)?$s/$t:0}")
    echo -e "${cond}\t${s}\t${f}\t${t}\t${sr}"
  done
} > "$RES"
cat "$RES"
touch "${LOGDIR}/STEER_EVAL_DONE"
echo "[orch] $(date '+%F %T') DONE -> ${RES}"
