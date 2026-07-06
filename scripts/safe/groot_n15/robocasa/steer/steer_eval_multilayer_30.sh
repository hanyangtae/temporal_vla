#!/usr/bin/env bash
# Multi-layer phase-matched steered eval: bread transport conceptor를 layers 4,8,12 에 동시 주입
# (신규 배선 --steering-npz-dir/--steering-layers), β0.3 permanent, 30 steered rollout.
# baseline(22/30)은 이전 eval 재사용. 단일 layer(global −0.10, transport-L4 −0.067)가 무효였으니
# 여러 layer 동시 주입이 다른지 검정.
# detached: cd REPO && setsid nohup bash scripts/safe/groot_n15/robocasa/steer/steer_eval_multilayer_30.sh \
#   > outputs/eval/robocasa/groot_n15/steer_eval/ppcc_bread/orchestrator_multilayer.log 2>&1 < /dev/null &
set -uo pipefail

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
N=30; STRIDE=1000; NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENV=robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env
CELL_INDEX=5; CELL_ID=ppcc_bread; SEED=100084
INSTR="Pick the bread from the counter and place it in the cabinet."

# 파라미터화: GROUP(global|transport|reach-to-object) × LAYERS × TAG(출력 분리) — ablation 재사용.
GROUP="${GROUP:-transport}"
LAYERS="${LAYERS:-4,8,12}"
TAG="${TAG:-ml_${GROUP}}"
NPZ_BASE="${NPZ_BASE:-/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/analysis/conceptor_steering_n15/ppcc_bread}"
NPZ_DIR=${NPZ_BASE}/${GROUP}
STEER_ARGS="--steering-npz-dir ${NPZ_DIR} --steering-layers ${LAYERS} --steering-beta 0.3 --steering-key C_steer"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/steer_eval/${CELL_ID}
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
GPUS=("${GPU_A:-6}" "${GPU_B:-7}"); PORTS=("${PORT_A:-8412}" "${PORT_B:-8413}")

start_serve() {
  docker exec -d -e CUDA_VISIBLE_DEVICES="$1" lerobot bash -lc \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
       --host '*' --port ${2} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
       ${STEER_ARGS} > /tmp/steer_ml_${2}.log 2>&1 < /dev/null &"
}
echo "[orch-ml] $(date '+%F %T') starting 2 multi-layer steered serves (GPU ${GPUS[*]})"
for i in 0 1; do start_serve "${GPUS[$i]}" "${PORTS[$i]}"; done
for port in "${PORTS[@]}"; do
  ok=0
  for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && { ok=1; break; }; sleep 5
  done
  echo "[orch-ml] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  if docker exec lerobot bash -lc "grep -qiE 'Traceback|load_model FAILED|KeyError|ValueError|FileNotFound' /tmp/steer_ml_${port}.log"; then
    echo "[orch-ml] ABORT: serve ${port} load error"; docker exec lerobot bash -lc "tail -30 /tmp/steer_ml_${port}.log"; exit 11
  fi
  # confirm multi-layer registration
  docker exec lerobot bash -lc "grep -q 'Multi-layer conceptor steering registered' /tmp/steer_ml_${port}.log && echo '[orch-ml] serve ${port}: multi-layer hooks OK'" 2>/dev/null || true
done

run_share() {
  local port=$1 wid=$2
  local outdir="${OUT_CONT}/steered_${TAG}/raw_rollouts"
  local hostdir="${OUT_HOST}/steered_${TAG}/raw_rollouts"
  for idx in $(seq 0 $((N - 1))); do
    [ $((idx % 2)) -eq "$wid" ] || continue
    if ls "${hostdir}/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${idx}--succ"*.pkl >/dev/null 2>&1; then
      echo "[ml w${wid}] skip ep${idx}"; continue; fi
    local inf=$((idx * STRIDE))
    echo "[ml w${wid}] ep${idx} inf=${inf} port=${port}"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENV" \
      --output-dir "$outdir" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$idx" --n-episodes 1 \
      --seed "$SEED" --inference-seed "$inf" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[ml w${wid}] DONE"
}
echo "[orch-ml] $(date '+%F %T') running 30 steered(multi-layer 4,8,12)"
run_share "${PORTS[0]}" 0 > "${LOGDIR}/steered_${TAG}_w0.log" 2>&1 &
run_share "${PORTS[1]}" 1 > "${LOGDIR}/steered_${TAG}_w1.log" 2>&1 &
wait
echo "[orch-ml] $(date '+%F %T') done; killing serves"
for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done

RES="${OUT_HOST}/sr_result_${TAG}.tsv"
sd="${OUT_HOST}/steered_${TAG}/raw_rollouts/${TASK}/${CELL_ID}"
bd="${OUT_HOST}/baseline/raw_rollouts/${TASK}/${CELL_ID}"
{
  echo -e "condition\tsucc\tfail\ttotal\tSR"
  for name in baseline steered_${TAG}; do
    d=$([ "$name" = baseline ] && echo "$bd" || echo "$sd")
    s=$(ls "$d"/*succ1.pkl 2>/dev/null|wc -l); f=$(ls "$d"/*succ0.pkl 2>/dev/null|wc -l); t=$((s+f))
    echo -e "${name}\t${s}\t${f}\t${t}\t$(awk "BEGIN{printf \"%.3f\", ($t>0)?$s/$t:0}")"
  done
} > "$RES"
cat "$RES"
touch "${LOGDIR}/STEER_${TAG}_DONE"
echo "[orch-ml] $(date '+%F %T') DONE -> ${RES}"
