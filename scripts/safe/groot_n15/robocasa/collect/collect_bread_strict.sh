#!/usr/bin/env bash
# drop-aware(state-based) 라벨러로 bread 재수집 → 엄밀 phase(transport=carrying-only, drop→reach).
# collector가 업그레이드된 make_robocasa_event_labeler를 import하므로 자동 적용 + grasp_timeline/drop_steps 저장.
# 새 RUN_ID(phase_event_strict)로 기존 데이터 보존. 30 rollout (steer eval과 같은 seed 0..29).
# 전제: N1.5 capture serve 2개가 포트 8412/8413(GPU 6,7)에서 ready.
# detached: cd REPO && setsid nohup bash scripts/safe/groot_n15/robocasa/collect/collect_bread_strict.sh \
#   > outputs/eval/robocasa/groot_n15/phase_event_strict/logs/orch.log 2>&1 < /dev/null &
set -uo pipefail

PROFILE=configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml
CAP="0,2,4,8,10,12,15"
N=60; STRIDE=1000; NAS=5; MAXEP=720
TASK=PickPlaceCounterToCabinet
ENV=robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env
CELL_INDEX=5; CELL_ID=ppcc_bread; SEED=100084
INSTR="Pick the bread from the counter and place it in the cabinet."
RUN_ID=phase_event_strict
GPUS=(4 5); PORTS=(8440 8441)

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_HOST="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/${RUN_ID}"
OUT_CONT=/temporal_vla/outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts
LOGDIR="${OUT_HOST}/logs"; mkdir -p "$LOGDIR"
PYPATH="/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"

start_serve() {  # gpu port (capture only, NO steering)
  docker exec -d -e CUDA_VISIBLE_DEVICES="$1" lerobot bash -lc \
    "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py --profile ${PROFILE} \
       --host '*' --port ${2} --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAP} \
       > /tmp/collect_strict_${2}.log 2>&1 < /dev/null &"
}
echo "[strict] $(date '+%F %T') starting 2 capture serves (GPU ${GPUS[*]})"
for i in 0 1; do start_serve "${GPUS[$i]}" "${PORTS[$i]}"; done
for port in "${PORTS[@]}"; do
  ok=0
  for _ in $(seq 1 150); do
    st=$(docker exec lerobot bash -lc "curl -s -m 3 http://127.0.0.1:${port}/health 2>/dev/null" | grep -o '"status":"ok"' || true)
    [ -n "$st" ] && { ok=1; break; }; sleep 5
  done
  echo "[strict] serve ${port} health=$([ $ok = 1 ] && echo ok || echo TIMEOUT)"
  [ $ok = 1 ] || { echo "[strict] ABORT: serve ${port} not ready"; exit 12; }
done

run_share() {  # port worker_id
  local port=$1 wid=$2
  for idx in $(seq 0 $((N - 1))); do
    [ $((idx % 2)) -eq "$wid" ] || continue
    if ls "${OUT_HOST}/raw_rollouts/${TASK}/${CELL_ID}/task${CELL_INDEX}--ep${idx}--succ"*.pkl >/dev/null 2>&1; then
      echo "[w${wid}] skip ep${idx}"; continue; fi
    local inf=$((idx * STRIDE))
    echo "[w${wid}] ep${idx} inf=${inf}"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="$PYPATH" robocasa \
      python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
      --vla-server "http://127.0.0.1:${port}" --task "$TASK" --env-name "$ENV" \
      --output-dir "$OUT_CONT" --cell-id "$CELL_ID" --cell-index "$CELL_INDEX" \
      --canonical-instruction "$INSTR" --episode-start-idx "$idx" --n-episodes 1 \
      --seed "$SEED" --inference-seed "$inf" --n-action-steps "$NAS" \
      --max-episode-steps "$MAXEP" --video-fps 20 --steps-per-render 2 --wait-ready 2>&1 \
      | grep -E "^wrote|Error|Traceback" || true
  done
  echo "[w${wid}] DONE"
}
echo "[strict] $(date '+%F %T') collecting ${N} bread rollouts (drop-aware labeler)"
run_share "${PORTS[0]}" 0 > "${LOGDIR}/w0.log" 2>&1 &
run_share "${PORTS[1]}" 1 > "${LOGDIR}/w1.log" 2>&1 &
wait
echo "[strict] $(date '+%F %T') done; killing serves"
for port in "${PORTS[@]}"; do docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--port ${port}' || true" 2>/dev/null || true; done
d="${OUT_HOST}/raw_rollouts/${TASK}/${CELL_ID}"
echo "bread strict: succ=$(ls $d/*succ1.pkl 2>/dev/null|wc -l) fail=$(ls $d/*succ0.pkl 2>/dev/null|wc -l)"
touch "${LOGDIR}/STRICT_COLLECT_DONE"
echo "[strict] $(date '+%F %T') DONE -> ${d}"
