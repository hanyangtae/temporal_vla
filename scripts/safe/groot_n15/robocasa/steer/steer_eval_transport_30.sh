#!/usr/bin/env bash
# phase-matched steered eval: bread transport conceptor(dit_L4, α0.3, β0.3) permanent 주입,
# 30 steered rollout. baseline(22/30)은 이전 global eval 것 재사용(같은 scene/seed 결정적).
# global(phase 혼합)이 ΔSR −0.10(무효)였으니 phase-matched(transport)가 다른지 검정.
# permanent 주입 = self-gating 가설(transport conceptor를 전 스텝 켜도 transport에만 작용하나).
# detached: cd REPO && setsid nohup bash scripts/safe/groot_n15/robocasa/steer/steer_eval_transport_30.sh \
#   > outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/orchestrator_transport.log 2>&1 < /dev/null &
set -uo pipefail

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
N=30; STRIDE=1000; NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENV=robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env
CELL_INDEX=5; CELL_ID=ppcc_bread; SEED=100084
INSTR="Pick the bread from the counter and place it in the cabinet."

NPZ=/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/analysis/conceptor_steering_n15/ppcc_bread/transport/dit_L4/conceptors.npz
STEER_ARGS="--steering-npz ${NPZ} --steering-pathway dit --steering-layer 4 --steering-alpha 0.3 --steering-beta 0.3 --steering-key C_steer"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
GPUS=(6 7); PORTS=(8412 8413)

start_serve() {  # gpu port
  docker exec -d -e CUDA_VISIBLE_DEVICES="$1" lerobot bash -lc \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
       --host '*' --port ${2} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
       ${STEER_ARGS} > /tmp/steer_tr_${2}.log 2>&1 < /dev/null &"
}
echo "[orch-tr] $(date '+%F %T') starting 2 steered serves (transport, GPU ${GPUS[*]})"
for i in 0 1; do start_serve "${GPUS[$i]}" "${PORTS[$i]}"; done
for port in "${PORTS[@]}"; do
  ok=0
  for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && { ok=1; break; }; sleep 5
  done
  echo "[orch-tr] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  if docker exec lerobot bash -lc "grep -qiE 'Traceback|load_model FAILED|KeyError|ValueError' /tmp/steer_tr_${port}.log"; then
    echo "[orch-tr] ABORT: steered serve ${port} load error"; docker exec lerobot bash -lc "tail -30 /tmp/steer_tr_${port}.log"; exit 11
  fi
done

run_share() {  # port worker_id
  local port=$1 wid=$2
  local outdir="${OUT_CONT}/steered_transport/raw_rollouts"
  local hostdir="${OUT_HOST}/steered_transport/raw_rollouts"
  for idx in $(seq 0 $((N - 1))); do
    [ $((idx % 2)) -eq "$wid" ] || continue
    if ls "${hostdir}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${idx}--succ"*.pkl >/dev/null 2>&1; then
      echo "[tr w${wid}] skip ep${idx}"; continue; fi
    local inf=$((idx * STRIDE))
    echo "[tr w${wid}] ep${idx} inf=${inf} port=${port}"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENV" \
      --output-dir "$outdir" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$idx" --n-episodes 1 \
      --seed "$SEED" --inference-seed "$inf" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[tr w${wid}] DONE"
}
echo "[orch-tr] $(date '+%F %T') running 30 steered(transport)"
run_share "${PORTS[0]}" 0 > "${LOGDIR}/steered_transport_w0.log" 2>&1 &
run_share "${PORTS[1]}" 1 > "${LOGDIR}/steered_transport_w1.log" 2>&1 &
wait
echo "[orch-tr] $(date '+%F %T') done; killing serves"
for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done

RES="${OUT_HOST}/sr_result_transport.tsv"
sd="${OUT_HOST}/steered_transport/raw_rollouts/${TASK}/${CELL_ID}"
bd="${OUT_HOST}/baseline/raw_rollouts/${TASK}/${CELL_ID}"
{
  echo -e "condition\tsucc\tfail\ttotal\tSR"
  for name in baseline steered_transport; do
    d=$([ "$name" = baseline ] && echo "$bd" || echo "$sd")
    s=$(ls "$d"/*succ1.pkl 2>/dev/null|wc -l); f=$(ls "$d"/*succ0.pkl 2>/dev/null|wc -l); t=$((s+f))
    echo -e "${name}\t${s}\t${f}\t${t}\t$(awk "BEGIN{printf \"%.3f\", ($t>0)?$s/$t:0}")"
  done
} > "$RES"
cat "$RES"
touch "${LOGDIR}/STEER_TRANSPORT_DONE"
echo "[orch-tr] $(date '+%F %T') DONE -> ${RES}"
